"""MVA P1 B1/A5. No network downloads; GT auxiliary branches are training-only.

The process-wide torchvision mask loss patch is deliberately common to B1/A5:
only zero-mass sampled mask targets are omitted, never their boxes/classes.
Geometry is done once outside the model; transform skips training resize and
still performs COCO normalization and common stride-32 padding. Eval uses the
standard torchvision resize/postprocess (probability pasted before threshold).
"""
from __future__ import annotations

import contextvars
import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import roi_heads
from torchvision.ops import roi_align
from torchvision.ops.misc import FrozenBatchNorm2d

VERSION = "mva-p1-v1.2"
COCO_V1_SHA256 = "bf2d0c1efbc936eeee2bc95a48e80ebc86b891f61b0106485937fc29f9315fc0"
_MASK_STATS = contextvars.ContextVar("mva_mask_stats", default=None)
_ORIGINAL_MASK_LOSS = roi_heads.maskrcnn_loss


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def standard_mask_targets(masks, boxes, matched, size):
    """Exact torchvision continuous target convention (aligned=False, adaptive).

    Targets need no gradient. Calling the verified native forward avoids the
    deterministic CUDA Python fallback's H*W adaptive-grid allocation on full
    GT masks. Trainable FPN ROIAlign uses the public API; installed CUDA backward
    uses atomic accumulation and is not guaranteed bitwise deterministic.
    This applies equally to B1/A5 and does not change global precision flags.
    """
    with torch.no_grad():
        rois = torch.cat([matched.to(boxes)[:, None], boxes], 1)
        return torch.ops.torchvision.roi_align(masks[:, None].to(rois), rois,
                                              1.0, size, size, -1, False)[:, 0]


def safe_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """Standard continuous ROI targets, with identical B1/A5 zero-mass handling."""
    size = mask_logits.shape[-1]
    labels = torch.cat([lab[idx] for lab, idx in zip(gt_labels, mask_matched_idxs)])
    sampled = torch.cat([
        standard_mask_targets(m, p, idx, size)
        for m, p, idx in zip(gt_masks, proposals, mask_matched_idxs)
    ])
    valid = sampled.flatten(1).sum(1) > 0
    stats = _MASK_STATS.get()
    if stats is not None:
        stats["mask_sampled"] = int(valid.numel())
        stats["mask_invalid"] = int((~valid).sum())
    if not valid.any():
        return mask_logits.sum() * 0
    logits = mask_logits[torch.arange(len(labels), device=labels.device), labels]
    return F.binary_cross_entropy_with_logits(logits[valid], sampled[valid])


def install_common_mask_loss():
    current = roi_heads.maskrcnn_loss
    if current is not safe_maskrcnn_loss and not getattr(current, "_mva_safe", False):
        if current.__module__ != "torchvision.models.detection.roi_heads":
            raise RuntimeError("Unexpected pre-existing torchvision mask loss patch")
        roi_heads.maskrcnn_loss = safe_maskrcnn_loss


safe_maskrcnn_loss._mva_safe = True


def sdf_targets(masks, boxes, size=28):
    """Tight half-open integer crop -> nearest 28 -> zero ring -> signed EDT.

    The original detector boxes survive even if a nearest-resized tiny mask
    disappears. Only the auxiliary target is then invalid. No SDF is RGB input.
    """
    cpu_masks = masks.detach().cpu().numpy().astype(bool)
    cpu_boxes = boxes.detach().cpu().numpy()
    result, valid = [], []
    h, w = cpu_masks.shape[-2:]
    for mask, box in zip(cpu_masks, cpu_boxes):
        x0, y0 = np.floor(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:]).astype(int)
        crop = mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if crop.size:
            small = F.interpolate(torch.from_numpy(crop.copy()).float()[None, None],
                                  (size, size), mode="nearest")[0, 0].numpy() > 0.5
        else:
            small = np.zeros((size, size), dtype=bool)
        good = bool(small.any())
        valid.append(good)
        padded = np.pad(small, 1, constant_values=False)
        sdf = (distance_transform_edt(padded) - distance_transform_edt(~padded))[1:-1, 1:-1]
        result.append(np.clip(sdf / 4.0, -1, 1).astype(np.float32))
    array = np.stack(result) if result else np.empty((0, size, size), dtype=np.float32)
    return (torch.from_numpy(array).to(masks.device),
            torch.tensor(valid, dtype=torch.bool, device=masks.device))


def instance_weights(relative_area, crowding, q25_area, q25_crowding):
    raw = (1 + 0.5 * (relative_area <= q25_area).float()
           + 0.5 * (crowding <= q25_crowding).float())
    return raw / raw.mean() if raw.numel() else raw


def consistency_terms(weak, strong, weights, gamma=0.05):
    """Physical-batch instances, not independent view samples; population std."""
    if len(weak) == 0:
        zero = (weak.sum() + strong.sum()) * 0
        return zero, zero, {"projection_valid": 0, "variance_skipped": 1,
                            "projection_std_median": 0.0, "cosine": 0.0}
    zw, zs = F.normalize(weak.float(), dim=1), F.normalize(strong.float(), dim=1)
    cosine = 1 - (zw * zs).sum(1)
    con = (weights * cosine).mean()
    if len(weak) < 2:
        var = con * 0
        median = 0.0
    else:
        sw = (zw.var(0, unbiased=False) + 1e-6).sqrt()
        ss = (zs.var(0, unbiased=False) + 1e-6).sqrt()
        var = (F.relu(gamma - sw).mean() + F.relu(gamma - ss).mean()) / (2 * gamma)
        median = float(torch.cat([sw, ss]).median().detach())
    return con, var, {"projection_valid": len(weak), "variance_skipped": int(len(weak) < 2),
                      "projection_std_median": median,
                      "cosine": float(cosine.mean().detach())}


def aligned_gt_roi(features, boxes, output_size=14):
    """Standard FPN level heuristic, explicit aligned=True on p2..p5.

    R50-FPN has exact spatial scales 1/4,1/8,1/16,1/32 regardless of
    stride padding. index_copy retains the gradient connection to the FPN.
    """
    maps = [features[str(i)] for i in range(4)]
    rows = [torch.cat([b.new_full((len(b), 1), i), b], 1) for i, b in enumerate(boxes)]
    rois = torch.cat(rows)
    output = maps[0].new_zeros((len(rois), maps[0].shape[1], output_size, output_size))
    if not len(rois):
        return output + sum(f.sum() * 0 for f in maps)
    scale = ((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])).sqrt()
    levels = (torch.floor(4 + torch.log2(scale / 224) + 1e-6).clamp(2, 5).long() - 2)
    # ROIAlign AMP support varies by torchvision build; float32 avoids dtype mismatch.
    for level, feature in enumerate(maps):
        index = torch.where(levels == level)[0]
        if len(index):
            with torch.autocast(device_type=feature.device.type, enabled=False):
                pooled = roi_align(feature.float(), rois[index].float(), output_size,
                                   spatial_scale=2.0 ** -(level + 2),
                                   sampling_ratio=2, aligned=True)
            output = output.index_copy(0, index, pooled.to(output.dtype))
    return output


class AuxiliaryHeads(nn.Module):
    """GT detection/SDF losses do not pass through projection and cannot directly
    prevent its collapse; variance is a measured heuristic, not a guarantee.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.sdf = nn.Conv2d(64, 1, 1)
        self.projection = nn.Linear(256, 128)

    def forward(self, roi):
        shape = self.sdf(F.interpolate(self.conv(roi), size=(28, 28),
                                      mode="bilinear", align_corners=False))[:, 0]
        embedding = self.projection(roi.mean((2, 3)))
        return shape, embedding


def detector_digest(detector):
    h = hashlib.sha256()
    for key, value in detector.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_detector(weights_path=None, seed=43, test_short=640, max_size=1333, smoke=False):
    """Local COCO V1 architecture, including FrozenBN eps=0; never download.

    Construction uses fork_rng, so optional auxiliary initialization cannot
    alter sampler, augmentation, or detector proposal-sampling RNG streams.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=3)
        kwargs = dict(min_size=test_short, max_size=max_size,
                      box_score_thresh=0.001, box_nms_thresh=0.5,
                      box_detections_per_img=100, _skip_resize=True)
        if smoke:
            kwargs.update(rpn_pre_nms_top_n_train=64, rpn_post_nms_top_n_train=32,
                          rpn_pre_nms_top_n_test=32, rpn_post_nms_top_n_test=16,
                          box_batch_size_per_image=32, rpn_batch_size_per_image=64)
        detector = MaskRCNN(backbone, num_classes=91, **kwargs)
        for module in detector.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"Local COCO checkpoint missing: {path}")
            # Lock the verified public COCO file, rejecting renamed task weights.
            if file_sha256(path) != COCO_V1_SHA256:
                raise ValueError("COCO checkpoint SHA256 differs from the verified generic initialization")
            state = torch.load(path, map_location="cpu", weights_only=True)
            detector.load_state_dict(state, strict=True)
        features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
        channels = detector.roi_heads.mask_predictor.conv5_mask.in_channels
        detector.roi_heads.mask_predictor = MaskRCNNPredictor(channels, 256, 2)
    install_common_mask_loss()
    return detector


class MVAModel(nn.Module):
    def __init__(self, method="A5", weights_path=None, seed=43, q25_area=0.0,
                 q25_crowding=0.0, test_short=640, max_size=1333, smoke=False,
                 lambda_m=0.3, lambda_c=0.1, variance_multiplier=0.1):
        super().__init__()
        if method not in ("B1", "A5"):
            raise ValueError("This P1 implementation supports only B1 and A5")
        self.method = method
        self.detector = build_detector(weights_path, seed, test_short, max_size, smoke)
        self.q25_area, self.q25_crowding = q25_area, q25_crowding
        self.lambda_m, self.lambda_c = lambda_m, lambda_c
        self.variance_multiplier = variance_multiplier
        self.auxiliary = None
        if method == "A5":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 3001)
                self.auxiliary = AuxiliaryHeads()
        self.last_stats = {}

    def inference_model(self):
        """Return core detector only: identical inference graph to B1."""
        return self.detector

    def forward(self, images, targets=None):
        if not self.training:
            return self.detector(images)
        if targets is None or len(images) % 2:
            raise ValueError("Training requires adjacent weak/strong pairs and targets")
        for i in range(0, len(targets), 2):
            if images[i].shape != images[i + 1].shape:
                raise ValueError("Pair shapes differ")
            for key in ("boxes", "masks", "leaf_ids"):
                if not torch.equal(targets[i][key], targets[i + 1][key]):
                    raise ValueError(f"Pair geometry/identity mismatch: {key}")
        # Two standard forwards give EXACT .5*(Lseg(weak)+Lseg(strong)),
        # including mask/class reductions when positive proposal counts differ.
        # Both use the same already-resized geometry and common padding sizes.
        view_losses, view_targets, view_rois, view_stats = [], [], [], []
        for view in (0, 1):
            captured = {}
            def transformed_hook(module, args, output):
                captured["image_list"], captured["targets"] = output
            def backbone_hook(module, args, output):
                captured["features"] = output
            hooks = [self.detector.transform.register_forward_hook(transformed_hook),
                     self.detector.backbone.register_forward_hook(backbone_hook)]
            branch_stats = {"mask_sampled": 0, "mask_invalid": 0}
            token = _MASK_STATS.set(branch_stats)
            try:
                view_losses.append(self.detector(images[view::2], targets[view::2]))
            finally:
                _MASK_STATS.reset(token)
                for hook in hooks:
                    hook.remove()
            transformed = captured["targets"]
            for transformed_t, original_t in zip(transformed, targets[view::2]):
                if not torch.equal(transformed_t["boxes"], original_t["boxes"]):
                    raise RuntimeError("Unexpected training transform resize")
            view_targets.append(transformed)
            view_stats.append(branch_stats)
            if self.auxiliary is not None:
                features = captured["features"]
                if isinstance(features, torch.Tensor):
                    features = OrderedDict([("0", features)])
                view_rois.append(aligned_gt_roi(features, [t["boxes"] for t in transformed]))
        losses = {k: 0.5 * (view_losses[0][k] + view_losses[1][k]) for k in view_losses[0]}
        stats = {k: sum(s[k] for s in view_stats) for k in ("mask_sampled", "mask_invalid")}
        stats.update(aux_total=0, aux_invalid=0, projection_valid=0, variance_skipped=0)
        if self.auxiliary is not None:
            predicted_w, projected_w = self.auxiliary(view_rois[0])
            predicted_s, projected_s = self.auxiliary(view_rois[1])
            truth, good, areas, crowds = [], [], [], []
            for weak_t, strong_t in zip(view_targets[0], view_targets[1]):
                if not torch.equal(weak_t["boxes"], strong_t["boxes"]):
                    raise RuntimeError("Model transform independently resized a view")
                sdf, valid = sdf_targets(weak_t["masks"], weak_t["boxes"])
                truth.append(sdf)
                good.append(valid)
                areas.append(weak_t["relative_area"])
                crowds.append(weak_t["crowding"])
            truth, good = torch.cat(truth), torch.cat(good)
            weights = instance_weights(torch.cat(areas)[good], torch.cat(crowds)[good],
                                       self.q25_area, self.q25_crowding)
            if good.any():
                error_w = F.smooth_l1_loss(predicted_w[good].float(), truth[good].float(), beta=0.1,
                                          reduction="none").mean((1, 2))
                error_s = F.smooth_l1_loss(predicted_s[good].float(), truth[good].float(), beta=0.1,
                                          reduction="none").mean((1, 2))
                morph = (weights * (error_w + error_s) * 0.5).mean()
            else:
                morph = (predicted_w.sum() + predicted_s.sum()) * 0
            con, var, con_stats = consistency_terms(projected_w[good], projected_s[good], weights)
            losses["loss_morph"] = self.lambda_m * morph
            losses["loss_consistency"] = self.lambda_c * (con + self.variance_multiplier * var)
            stats.update(con_stats)
            stats.update(aux_total=len(good), aux_invalid=int((~good).sum()),
                         morph_unscaled=float(morph.detach()),
                         consistency_unscaled=float(con.detach()),
                         variance_unscaled=float(var.detach()),
                         weight_mean=float(weights.mean()) if len(weights) else 0.0)
        stats["mask_invalid_ratio"] = stats["mask_invalid"] / max(1, stats["mask_sampled"])
        stats["aux_invalid_ratio"] = stats["aux_invalid"] / max(1, stats["aux_total"])
        self.last_stats = stats
        return losses

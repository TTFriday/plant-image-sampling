"""Direct CLI for the predeclared MVA B1/A5 3000-update P1 pilot.

Only TRAIN is sampled/read for training/backgrounds; evaluation is restricted
to V_SOURCE/V_SHIFT. This module has no dependency on the P0 audit program.
Synthetic unit tests do not constitute experiment results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mva_model import VERSION, MVAModel, detector_digest, file_sha256

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.optimize import linear_sum_assignment
import scipy
import torch
from torch.nn import functional as F
import torchvision
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

ROLES = ("TRAIN", "V_SOURCE", "V_SHIFT")
SCHEMA = {"image_id", "biological_plant_id", "genotype", "replicate_id", "split_role",
          "timestamp", "width", "height", "rgb_path", "mask_path", "label_path", "mask_status"}
DEFAULT_COCO = os.environ.get(
    "PHENOLEAF_COCO_WEIGHTS",
    "checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return "Infinity" if obj > 0 else ("-Infinity" if obj < 0 else "NaN")
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(jsonable(obj), ensure_ascii=False, allow_nan=False) + "\n")


def heartbeat(out, phase, step=0, **kwargs):
    write_json(Path(out) / "status.json", dict(version=VERSION, phase=phase,
               optimizer_step=step, time=time.time(), pid=os.getpid(), **kwargs))


def read_inventory(path, valid_status="valid", data_root=None):
    """Read the actual named schema; do not open labels/images of locked roles."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"P0 inventory does not yet exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = SCHEMA - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory schema missing {sorted(missing)}")
        rows = []
        seen = set()
        for record in reader:
            if record["split_role"] not in ROLES:
                continue
            if record["mask_status"].strip().lower() != valid_status.lower():
                continue
            if record["image_id"] in seen:
                raise ValueError("Duplicate inventory image_id")
            seen.add(record["image_id"])
            for key in ("rgb_path", "label_path"):
                p = Path(record[key])
                if not p.is_absolute():
                    if data_root is None:
                        raise ValueError("Relative data paths require explicit --data-root")
                    p = Path(data_root) / p
                record[key] = str(p.resolve())
            record["width"], record["height"] = int(record["width"]), int(record["height"])
            rows.append(record)
    if not any(r["split_role"] == "TRAIN" for r in rows):
        raise ValueError(f"No valid TRAIN rows (mask_status={valid_status!r}); inspect audit schema")
    return sorted(rows, key=lambda r: (r["split_role"], r["genotype"],
                                       r["biological_plant_id"], r["timestamp"], r["image_id"]))


def target_from_label(label):
    label = np.asarray(label)
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError("label_path must be single-channel integer original paletteId")
    ids = np.unique(label)
    if np.any((ids < 0) | (ids > 31)):
        raise ValueError("Unknown paletteId: exclude/repair through P0, never silently ignore")
    ids = ids[ids != 0]
    h, w = label.shape
    masks = np.stack([label == i for i in ids]) if len(ids) else np.empty((0, h, w), bool)
    boxes, centers, areas = [], [], []
    for mask in masks:
        y, x = np.nonzero(mask)
        boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        centers.append([y.mean(), x.mean()])
        areas.append(len(x))
    area = np.asarray(areas, np.float32)
    crowd = np.full(len(ids), np.inf, dtype=np.float32)
    if len(ids) > 1:
        center = np.asarray(centers)
        distances = np.linalg.norm(center[:, None] - center[None, :], axis=2)
        distances /= np.sqrt(area)[:, None] + np.sqrt(area)[None, :]
        np.fill_diagonal(distances, np.inf)
        crowd = distances.min(1).astype(np.float32)
    return {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(ids), dtype=torch.int64),
            "masks": torch.from_numpy(masks.astype(np.uint8)),
            "leaf_ids": torch.tensor(ids.astype(np.int64)),
            "area": torch.from_numpy(area),
            "relative_area": torch.from_numpy(area / (h * w)),
            "crowding": torch.from_numpy(crowd),
            "iscrowd": torch.zeros(len(ids), dtype=torch.int64)}


def read_instances(path, train_ids):
    """Optional P0 metadata, indexed only for TRAIN. Verify it against labels."""
    result = {}
    if not Path(path).is_file():
        return result
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            obj = json.loads(line)
            key = str(obj["image_id"])
            if key not in train_ids:
                continue
            if key in result:
                raise ValueError("Duplicate TRAIN image in instances.jsonl")
            leaves = obj["instances"]
            required = {"leaf_id", "area", "box_xyxy", "center_yx", "relative_area", "crowding"}
            if any(required - set(leaf) for leaf in leaves):
                raise ValueError("instances.jsonl leaf schema incomplete")
            result[key] = leaves
    return result


class LeafData:
    def __init__(self, rows, instances=None):
        self.rows = {r["image_id"]: r for r in rows}
        self.instances = instances or {}

    def label(self, row):
        if row["split_role"] not in ROLES:
            raise PermissionError("Locked data role")
        if Path(row["label_path"]).suffix.lower() == ".npy":
            label = np.load(row["label_path"], allow_pickle=False)
        else:
            with Image.open(row["label_path"]) as image:
                label = np.asarray(image).copy()
        if label.shape != (row["height"], row["width"]):
            raise ValueError(f"Label size mismatch: {row['image_id']}")
        target = target_from_label(label)
        metadata = self.instances.get(row["image_id"])
        if metadata is not None:
            by_id = {int(x["leaf_id"]): x for x in metadata}
            if set(by_id) != set(target["leaf_ids"].tolist()):
                raise ValueError("Audit/label leaf IDs differ")
            for i, leaf_id in enumerate(target["leaf_ids"].tolist()):
                leaf = by_id[leaf_id]
                if int(leaf["area"]) != int(target["area"][i]):
                    raise ValueError("Audit/label instance area differs")
                if not np.allclose(leaf["box_xyxy"], target["boxes"][i].numpy()):
                    raise ValueError("Audit/label half-open box differs")
                for key in ("relative_area", "crowding"):
                    value = leaf[key]
                    value = float("inf") if value is None else float(value)
                    if not np.isclose(value, float(target[key][i]), rtol=1e-5, atol=1e-7):
                        raise ValueError(f"Audit/label {key} differs")
                    target[key][i] = value
        return target

    def load(self, row):
        target = self.label(row)
        with Image.open(row["rgb_path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        if rgb.shape[:2] != (row["height"], row["width"]):
            raise ValueError("RGB size mismatch")
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255, target


class BalancedSampler:
    """genotype -> biological plant -> frame; no flatten/random ratio splitting."""
    def __init__(self, rows, seed=43):
        self.groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["split_role"] != "TRAIN":
                continue
            self.groups[row["genotype"]][row["biological_plant_id"]].append(row)
        self.genotypes = sorted(self.groups)
        if not self.genotypes:
            raise ValueError("No TRAIN sampling population")
        for plants in self.groups.values():
            for frames in plants.values():
                frames.sort(key=lambda r: (r["timestamp"], r["image_id"]))
        self.rng = random.Random(seed)
        self.draws = 0

    def draw(self):
        genotype = self.rng.choice(self.genotypes)
        plants = self.groups[genotype]
        plant = self.rng.choice(sorted(plants))
        self.draws += 1
        return self.rng.choice(plants[plant])

    def donor(self, current_plant, rng):
        options = [(g, p) for g in self.genotypes for p in sorted(self.groups[g])
                   if p != current_plant]
        if not options:
            return None
        g, p = rng.choice(options)
        return rng.choice(self.groups[g][p])

    def state_dict(self):
        return {"rng": self.rng.getstate(), "draws": self.draws}

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        self.draws = state["draws"]


def clone_target(target):
    return {k: v.clone() for k, v in target.items()}


def shared_geometry(image, target, params, max_size=1333):
    """Flip/rot90 on boxes analytically, one joint nearest-mask resize.

    Never recompute boxes from resized masks: a vanishing tiny sampled mask
    must retain its detection box/class. Area/crowding weights stay original.
    """
    image, target = image.clone(), clone_target(target)
    masks, boxes = target["masks"], target["boxes"]
    h, w = image.shape[-2:]
    if params["horizontal"]:
        image, masks = image.flip(-1), masks.flip(-1)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if params["vertical"]:
        image, masks = image.flip(-2), masks.flip(-2)
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    for _ in range(params["rot90"]):
        image, masks = image.rot90(1, (-2, -1)), masks.rot90(1, (-2, -1))
        old = boxes.clone()
        boxes[:, 0], boxes[:, 2] = old[:, 1], old[:, 3]
        boxes[:, 1], boxes[:, 3] = w - old[:, 2], w - old[:, 0]
        h, w = w, h
    factor = min(params["short"] / min(h, w), max_size / max(h, w))
    new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
    image = F.interpolate(image[None], (new_h, new_w), mode="bilinear",
                          align_corners=False)[0]
    if len(masks):
        masks = F.interpolate(masks[:, None].float(), (new_h, new_w),
                              mode="nearest")[:, 0].to(torch.uint8)
    else:
        masks = torch.empty((0, new_h, new_w), dtype=torch.uint8)
    boxes *= boxes.new_tensor([new_w / w, new_h / h, new_w / w, new_h / h])
    target["boxes"], target["masks"] = boxes, masks
    return image, target


def donor_background(image, masks):
    union = masks.bool().any(0).numpy()
    removal = union.copy()
    if union.any():
        # Remove the whole donor plant envelope, not only its annotated pixels.
        # Annotation fringes/petioles otherwise leave unlabeled green silhouettes
        # and nearest-fill Voronoi patches that resemble additional leaves.
        ys, xs = np.nonzero(union)
        margin = max(12, int(round(0.05 * min(union.shape))))
        removal[max(0, ys.min() - margin):min(union.shape[0], ys.max() + margin + 1),
                max(0, xs.min() - margin):min(union.shape[1], xs.max() + margin + 1)] = True
    if removal.all() and not union.all():
        # Widely separated visible fragments can span the entire image. In that
        # case use a generous union dilation rather than reusing raw leaf fringes.
        removal = distance_transform_edt(~union) <= margin
    if removal.all():
        raise ValueError("TRAIN donor envelope has no valid background; no invented fill allowed")
    rgb = image.permute(1, 2, 0).numpy()
    if removal.any():
        nearest = distance_transform_edt(removal, return_distances=False, return_indices=True)
        rgb = rgb[nearest[0], nearest[1]]
    rgb = gaussian_filter(rgb, sigma=(2, 2, 0), mode="reflect")
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


def replace_background(image, union, background):
    union = union.bool()
    background = F.interpolate(background[None], image.shape[-2:], mode="bilinear",
                               align_corners=False)[0]
    outside_distance = distance_transform_edt(~union.numpy())
    alpha = torch.from_numpy(np.clip((2.0 - outside_distance) / 2.0, 0, 1)).float()
    alpha[union] = 1
    mixed = image * alpha[None] + background * (1 - alpha[None])
    # Preserve every leaf RGB pixel bit-for-bit, feather only OUTSIDE the leaf.
    mixed[:, union] = image[:, union]
    if not torch.equal(mixed[:, union], image[:, union]):
        raise AssertionError("Background intervention changed leaf pixels")
    return mixed


def apply_color(image, params):
    result = TF.adjust_brightness(image, params["brightness"])
    result = TF.adjust_contrast(result, params["contrast"])
    if "saturation" in params:
        result = TF.adjust_saturation(result, params["saturation"])
        result = TF.adjust_hue(result, params["hue"])
        result = TF.adjust_gamma(result, params["gamma"])
        if params["blur"]:
            result = TF.gaussian_blur(result, [3, 3], [0.5, 0.5])
        if params["noise"]:
            generator = torch.Generator().manual_seed(params["noise_seed"])
            result = result + torch.randn(result.shape, generator=generator) * 0.01
    return result.clamp(0, 1)


class AugmentStreams:
    def __init__(self, seed=43, shorts=(512, 576, 640), max_size=1333):
        self.geometry_rng = random.Random(seed + 101)
        self.color_rng = random.Random(seed + 202)
        self.intervention_rng = random.Random(seed + 303)
        self.donor_rng = random.Random(seed + 404)
        self.shorts, self.max_size = tuple(shorts), max_size

    def parameters(self):
        g, c, a = self.geometry_rng, self.color_rng, self.intervention_rng
        geometry = {"horizontal": g.random() < 0.5, "vertical": g.random() < 0.5,
                    "rot90": g.randrange(4), "short": g.choice(self.shorts)}
        weak = {"brightness": c.uniform(0.9, 1.1), "contrast": c.uniform(0.9, 1.1)}
        strong = {"brightness": c.uniform(0.7, 1.3), "contrast": c.uniform(0.7, 1.3),
                  "saturation": c.uniform(0.7, 1.3), "hue": c.uniform(-0.03, 0.03),
                  "gamma": c.uniform(0.8, 1.2), "blur": c.random() < 0.2,
                  "noise": c.random() < 0.2, "noise_seed": c.randrange(2**63)}
        interventions = {"replace": a.random() < 0.5, "plant_color": a.random() < 0.5,
                         "brightness": a.uniform(0.9, 1.1), "contrast": a.uniform(0.9, 1.1),
                         "saturation": a.uniform(0.9, 1.1)}
        return geometry, weak, strong, interventions

    def pair(self, image, target, row, method, data, sampler):
        geom, weak_p, strong_p, extra = self.parameters()
        donor = sampler.donor(row["biological_plant_id"], self.donor_rng)
        base, transformed = shared_geometry(image, target, geom, self.max_size)
        weak, strong = apply_color(base, weak_p), apply_color(base, strong_p)
        union = transformed["masks"].bool().any(0)
        if method == "A5" and extra["replace"]:
            if donor is None:
                raise ValueError("A5 background requires a different TRAIN biological plant")
            donor_rgb, donor_target = data.load(donor)
            strong = replace_background(strong, union,
                                        donor_background(donor_rgb, donor_target["masks"]))
        if method == "A5" and extra["plant_color"]:
            mild = TF.adjust_brightness(strong, extra["brightness"])
            mild = TF.adjust_contrast(mild, extra["contrast"])
            mild = TF.adjust_saturation(mild, extra["saturation"]).clamp(0, 1)
            strong = torch.where(union[None], mild, strong)
        # Targets are cloned after the one geometry operation, not from RGB.
        first, second = clone_target(transformed), clone_target(transformed)
        if not torch.equal(first["masks"], second["masks"]) or not torch.equal(first["boxes"], second["boxes"]):
            raise AssertionError("Appearance changed geometry")
        record = {"image_id": row["image_id"], "genotype": row["genotype"],
                  "biological_plant_id": row["biological_plant_id"], "geometry": geom,
                  "weak": weak_p, "strong": strong_p, "intervention_draw": extra,
                  "donor_image_id": donor["image_id"] if donor else None,
                  "donor_plant": donor["biological_plant_id"] if donor else None,
                  "background_applied": method == "A5" and extra["replace"],
                  "plant_color_applied": method == "A5" and extra["plant_color"],
                  "leaf_ids": transformed["leaf_ids"].tolist(),
                  "leaf_pixels_geometry": transformed["masks"].sum((1, 2)).tolist(),
                  "leaf_pixels_weak": first["masks"].sum((1, 2)).tolist(),
                  "leaf_pixels_strong": second["masks"].sum((1, 2)).tolist(),
                  "boxes_geometry": transformed["boxes"].tolist(),
                  "boxes_weak": first["boxes"].tolist(), "boxes_strong": second["boxes"].tolist(),
                  "geometry_equal": True}
        return [weak, strong], [first, second], record, base

    def state_dict(self):
        return {name: getattr(self, name).getstate() for name in
                ("geometry_rng", "color_rng", "intervention_rng", "donor_rng")}

    def load_state_dict(self, state):
        for name, rng_state in state.items():
            getattr(self, name).setstate(rng_state)


def quantile25(values):
    values = np.sort(np.asarray(values, np.float64))
    if not len(values):
        raise ValueError("No TRAIN instances available for q25")
    position = (len(values) - 1) * 0.25
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi or np.isinf(values[hi]):
        return float(values[hi])
    return float(values[lo] + (position - lo) * (values[hi] - values[lo]))


def dotted_value(obj, key):
    for component in key.split("."):
        obj = obj[component]
    return float(obj)


def freeze_quantiles(args, rows, data):
    if (args.q25_area is None) != (args.q25_crowding is None):
        raise ValueError("Provide both --q25-area and --q25-crowding or neither")
    if args.q25_area is not None:
        return args.q25_area, args.q25_crowding, "explicit_CLI_TRAIN_thresholds"
    audit = Path(args.audit_summary)
    if audit.is_file():
        obj = json.loads(audit.read_text(encoding="utf-8"))
        try:
            area = dotted_value(obj, args.audit_area_key)
            crowd = dotted_value(obj, args.audit_crowding_key)
            return area, crowd, f"audit:{args.audit_area_key},{args.audit_crowding_key}"
        except (KeyError, TypeError, ValueError):
            print("Audit q25 keys unavailable; computing thresholds from valid TRAIN labels only.")
    areas, crowds = [], []
    for row in rows:
        if row["split_role"] == "TRAIN":
            target = data.label(row)
            areas.extend(target["relative_area"].tolist())
            crowds.extend(target["crowding"].tolist())
    return quantile25(areas), quantile25(crowds), "computed_valid_TRAIN_all_instances"


def augmentation_debug(out, rows, data, method, seed, shorts, max_size):
    """Separate streams: debugging never consumes training exposure/RNG."""
    directory = Path(out) / "augmentation_debug"
    directory.mkdir(exist_ok=True)
    sampler = BalancedSampler(rows, seed)
    streams = AugmentStreams(seed, shorts, max_size)
    # Thirty distinct valid TRAIN originals whenever at least thirty exist.
    selected = [r for r in rows if r["split_role"] == "TRAIN"]
    random.Random(seed + 505).shuffle(selected)
    selected = selected[:30]
    audit_records = []
    for index, row in enumerate(selected):
        heartbeat(out, "augmentation_debug", debug_index=index, debug_total=len(selected))
        original, target = data.load(row)
        images, targets, record, geometry = streams.pair(original, target, row, method, data, sampler)
        panels = []
        for title, image in zip(("original RGB", "shared geometry", "weak view", "strong view"),
                                (original, geometry, *images)):
            array = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            panel = Image.new("RGB", (320, 350), "white")
            thumb = Image.fromarray(array)
            thumb.thumbnail((320, 320))
            panel.paste(thumb, ((320 - thumb.width) // 2, 25))
            ImageDraw.Draw(panel).text((5, 5), title, fill="black")
            panels.append(panel)
        mosaic = Image.new("RGB", (1280, 350), "white")
        for j, panel in enumerate(panels):
            mosaic.paste(panel, (320 * j, 0))
        mosaic.save(directory / f"{index:02d}.png")
        record.update(original_leaf_pixels=target["masks"].sum((1, 2)).tolist(),
                      mosaic=f"augmentation_debug/{index:02d}.png")
        audit_records.append(record)
    write_json(directory / "audit.json", {"samples": audit_records, "distinct_originals": len(selected),
                "needs_manual_review": True, "minimum_30_met": len(selected) == 30,
                "area_invariance_scope": "after_shared_geometry_before_after_appearance"})
    return len(selected)


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def dev_rows(rows, max_frames=0):
    groups = defaultdict(list)
    for row in rows:
        if row["split_role"] in ("V_SOURCE", "V_SHIFT"):
            groups[(row["split_role"], row["genotype"], row["biological_plant_id"])].append(row)
    selected = []
    for key in sorted(groups):
        frames = sorted(groups[key], key=lambda r: (r["timestamp"], r["image_id"]))
        if max_frames and len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in indices]
        selected.extend(frames)
    return selected


def coco_ground_truth(rows, data):
    images, annotations = [], []
    for image_id, row in enumerate(rows, 1):
        images.append({"id": image_id, "height": row["height"], "width": row["width"],
                       "source_image_id": row["image_id"], "plant": row["biological_plant_id"]})
        target = data.label(row)
        for index, mask in enumerate(target["masks"].numpy()):
            x0, y0, x1, y1 = target["boxes"][index].tolist()
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                "category_id": 1, "segmentation": encode_mask(mask),
                "area": int(mask.sum()), "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
    return {"info": {"description": "valid dev single-class leaf; all palette instances"},
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "leaf"}]}


def coco_mask_ap(gt_dataset, predictions):
    """One COCO set per genotype; empty prediction set gives AP=0, not omission."""
    if not predictions or not gt_dataset["annotations"]:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dataset
        gt.createIndex()
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, "segm")
        evaluator.params.imgIds = [x["id"] for x in gt_dataset["images"]]
        evaluator.params.catIds = [1]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {name: float(max(0, evaluator.stats[i]) * 100)
            for i, name in enumerate(("AP", "AP50", "AP75"))}


def discrete_metrics(target, predicted_masks, q25_area):
    gt = target["masks"].numpy().astype(bool)
    pred = np.asarray(predicted_masks, dtype=bool)
    ng, npred = len(gt), len(pred)
    if ng and npred:
        grle, prle = [encode_mask(m) for m in gt], [encode_mask(m) for m in pred]
        # RLE pairwise IoU avoids multi-GB float64 flattened full-resolution masks.
        iou = mask_utils.iou(grle, prle, [0] * npred)
        ga = np.asarray(mask_utils.area(grle), dtype=np.float64)[:, None]
        pa = np.asarray(mask_utils.area(prle), dtype=np.float64)[None, :]
        intersection = iou * (ga + pa) / (1 + iou)
        dice = 2 * intersection / np.maximum(ga + pa, 1)
        valid = iou >= 0.5
        gi, pi = linear_sum_assignment(valid * (min(ng, npred) + 1 + iou), maximize=True)
        matched = gi[valid[gi, pi]]
        sbd = min(dice.max(1).mean(), dice.max(0).mean())
    else:
        matched = np.empty(0, dtype=int)
        sbd = 1.0 if ng == npred == 0 else 0.0
    small = target["relative_area"].numpy() <= q25_area
    union_gt = gt.any(0) if ng else np.zeros(gt.shape[-2:], bool)
    union_pred = pred.any(0) if npred else np.zeros(gt.shape[-2:], bool)
    return {"count_abs_error": abs(npred - ng), "count_bias": npred - ng,
            "gt_count": ng, "pred_count": npred, "SBD": float(sbd) * 100,
            "area_error_pp": float(union_pred.mean() - union_gt.mean()) * 100,
            "small_gt": int(small.sum()), "small_matched": int(small[matched].sum()),
            "matched_gt": len(matched)}


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def global_rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_global_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@torch.no_grad()
def evaluate_dev(model, rows, data, out, step, device, q25_area, inventory_sha,
                 max_frames=0, amp=False):
    selected = dev_rows(rows, max_frames)
    groups = defaultdict(list)
    for row in selected:
        groups[(row["split_role"], row["genotype"])].append(row)
    if not groups:
        raise ValueError("No valid V_SOURCE/V_SHIFT frames available; cannot select checkpoint")
    state, training = global_rng_state(device), model.training
    model.eval()
    sync_device(device)
    started = time.perf_counter()
    results = {}
    cache = Path(out) / "dev_gt_cache"
    cache.mkdir(exist_ok=True)
    try:
        for (role, genotype), frames in sorted(groups.items()):
            key = hashlib.sha256(json.dumps([inventory_sha, [r["image_id"] for r in frames]]).encode()).hexdigest()
            cache_path = cache / f"{key}.json"
            heartbeat(out, "dev_gt", step, role=role, genotype=genotype)
            if cache_path.is_file():
                gt_dataset = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                gt_dataset = coco_ground_truth(frames, data)
                write_json(cache_path, gt_dataset)
            predictions, per_plant = [], defaultdict(list)
            for image_id, row in enumerate(frames, 1):
                heartbeat(out, "dev_inference", step, role=role, genotype=genotype,
                          frame=image_id, frames=len(frames))
                image, target = data.load(row)
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = model([image.to(device)])[0]
                scores = output["scores"].float().cpu()
                probs = output["masks"].float().cpu()[:, 0]
                # Standard detector postprocess already pastes soft masks at original
                # resolution. If an interface ever changes, resize PROBABILITY first.
                if probs.shape[-2:] != (row["height"], row["width"]) and len(probs):
                    probs = F.interpolate(probs[:, None], (row["height"], row["width"]),
                                          mode="bilinear", align_corners=False)[:, 0]
                masks = (probs >= 0.5).numpy()
                for index in torch.where(scores >= 0.001)[0].tolist():
                    predictions.append({"image_id": image_id, "category_id": 1,
                                        "score": float(scores[index]),
                                        "segmentation": encode_mask(masks[index])})
                discrete = masks[(scores >= 0.5).numpy()]
                per_plant[row["biological_plant_id"]].append(discrete_metrics(target, discrete, q25_area))
            heartbeat(out, "dev_COCO_AP", step, role=role, genotype=genotype)
            metrics = coco_mask_ap(gt_dataset, predictions)
            metrics.update(frames=len(frames), plants=len(per_plant),
                           gt_instances=len(gt_dataset["annotations"]),
                           ap_score_threshold=0.001, count_score_threshold=0.5,
                           mask_threshold=0.5, max_detections=100)
            for field in ("count_abs_error", "count_bias", "SBD", "area_error_pp"):
                metrics[field + "_plant_macro"] = float(np.mean([
                    np.mean([f[field] for f in plant]) for plant in per_plant.values()]))
            small_gt = sum(f["small_gt"] for plant in per_plant.values() for f in plant)
            small_hit = sum(f["small_matched"] for plant in per_plant.values() for f in plant)
            metrics.update(small_gt=small_gt, small_matched=small_hit,
                           small_recall_at_iou50_score50=100 * small_hit / small_gt if small_gt else None)
            results.setdefault(role, {})[genotype] = metrics
            # Only a single genotype's COCO set is in memory at a time.
            del gt_dataset, predictions
        role_ap = {role: float(np.mean([g["AP"] for g in genotypes.values()]))
                   for role, genotypes in results.items()}
        if set(role_ap) != {"V_SOURCE", "V_SHIFT"}:
            raise ValueError("Both dev roles must be available for equally weighted selection")
        sync_device(device)
        report = {"optimizer_step": step, "per_genotype": results,
                  "role_macro_AP": role_ap, "dev_score": float(np.mean(list(role_ap.values()))),
                  "dev_max_frames_per_plant": max_frames,
                  "diagnostic_subset": bool(max_frames), "all_instances_kept": True,
                  "elapsed_seconds": time.perf_counter() - started,
                  "eval_stream_sha256": hashlib.sha256(json.dumps(
                      [r["image_id"] for r in selected]).encode()).hexdigest()}
        write_json(Path(out) / f"dev_{step:06d}.json", report)
        return report
    finally:
        model.train(training)
        restore_global_rng(state, device)


def lr_for_update(update, budget=3000, warmup=500, base=0.005):
    """One-based optimizer update. P1 cosine horizon is 3000, not 12000."""
    if update <= warmup:
        return base * (0.01 + 0.99 * update / warmup)
    progress = min(1.0, (update - warmup) / max(1, budget - warmup))
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path, model, optimizer, scaler, sampler, streams, device, step, config,
                    best, collapse_streak, logging_state=None, evaluation_complete=True):
    # Optimizer-boundary only: no partial accumulation/pending gradients.
    checkpoint = {"version": VERSION, "optimizer_step": step, "next_optimizer_step": step + 1,
                  "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict(), "sampler": sampler.state_dict(),
                  "augmentation": streams.state_dict(), "global_rng": global_rng_state(device),
                  "config": config, "best": best, "collapse_streak": collapse_streak,
                  "logging_state": logging_state or {"window": [], "timings": [], "projection_window": []},
                  "evaluation_complete": evaluation_complete}
    temporary = Path(str(path) + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def evaluate_and_checkpoint(args, model, optimizer, scaler, sampler, streams, device,
                            rows, data, step, config, qarea, best, collapse_streak,
                            window, timings, projection_window):
    """Persist optimizer boundary BEFORE dev work; finish pending eval on resume."""
    out = Path(args.out)
    path = out / f"checkpoint_{step:06d}.pt"
    heartbeat(out, "checkpoint_before_dev", step)
    logging = {"window": window, "timings": timings, "projection_window": projection_window}
    save_checkpoint(path, model, optimizer, scaler, sampler, streams, device, step, config,
                    best, collapse_streak, logging_state=logging, evaluation_complete=False)
    write_json(out / "latest.json", {"checkpoint": str(path), "next_optimizer_step": step + 1,
                                    "evaluation_complete": False})
    report = evaluate_dev(model, rows, data, out, step, device, qarea,
                          config["inventory_sha256"], args.dev_max_frames_per_plant, args.amp)
    if args.method == "A5":
        usable = [s for s in projection_window if s["projection_valid"] >= 2]
        median = float(np.median([s["projection_std_median"] for s in usable])) if usable else None
        cosine = float(np.mean([s["cosine"] for s in usable])) if usable else None
        degenerate = median is not None and median < 0.01 and cosine < 0.001
        collapse_streak = collapse_streak + 1 if degenerate else 0
        report.update(projection_std_median=median, projection_cosine=cosine,
                      collapse_streak=collapse_streak,
                      collapse_criterion="std median<.01 and cosine<.001 at 3 consecutive checks")
        write_json(out / f"dev_{step:06d}.json", report)
    projection_window.clear()
    is_best = report["dev_score"] > best["score"]
    if is_best:
        best = {"score": report["dev_score"], "step": step,
                "diagnostic_subset": report["diagnostic_subset"]}
    heartbeat(out, "checkpoint_after_dev", step, dev_score=report["dev_score"], best=best)
    save_checkpoint(path, model, optimizer, scaler, sampler, streams, device, step, config,
                    best, collapse_streak, logging_state=logging, evaluation_complete=True)
    write_json(out / "latest.json", {"checkpoint": str(path), "next_optimizer_step": step + 1,
                                    "evaluation_complete": True})
    if is_best:
        write_json(out / "best.json", {**best, "checkpoint": str(path)})
    if collapse_streak >= 3:
        heartbeat(out, "paused_projection_collapse", step, best=best)
    return best, collapse_streak


def trim_jsonl_for_resume(path, step):
    path = Path(path)
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # interrupted final append
        if obj.get("optimizer_step", 0) <= step:
            kept.append(line)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(temporary, path)


def make_config(args, qarea, qcrowd, quantile_source, initial_sha):
    sources = {str(p): {"sha256": file_sha256(p), "source": p.read_text(encoding="utf-8")}
               for p in (Path(__file__), Path(__file__).with_name("mva_model.py"))}
    return {"version": VERSION, "method": args.method, "seed": args.seed,
            "precision": "AMP" if args.amp else "FP32", "device": args.device,
            "steps_requested": args.steps, "schedule_steps": args.schedule_steps, "warmup": 500,
            "physical_base_batch": args.physical_batch, "accumulation": args.accumulation,
            "effective_base_images": args.physical_batch * args.accumulation,
            "views_per_base": 2, "optimizer": {"type": "SGD", "lr": 0.005,
                "momentum": 0.9, "weight_decay": 1e-4, "gradient_clip": 5},
            "lambda_m": 0.3, "lambda_c": 0.1, "variance_multiplier": 0.1,
            "variance_gamma": 0.05, "variance_std": "population_sqrt(var+1e-6)",
            "q25_area": qarea, "q25_crowding": qcrowd, "quantile_source": quantile_source,
            "inventory": str(Path(args.inventory).resolve()),
            "inventory_sha256": file_sha256(args.inventory),
            "instances_sha256": file_sha256(args.instances) if Path(args.instances).is_file() else None,
            "audit_sha256": file_sha256(args.audit_summary) if Path(args.audit_summary).is_file() else None,
            "coco_weights": str(Path(args.weights).resolve()), "coco_weight_sha256": file_sha256(args.weights),
            "coco_source": "torchvision MaskRCNN_ResNet50_FPN_Weights.COCO_V1 (public generic COCO)",
            "coco_license_reference": "https://github.com/pytorch/vision/blob/v0.21.0/LICENSE",
            "initial_detector_sha256": initial_sha, "sources": sources,
            "geometry": {"shorts": [256] if args.smoke else [512, 576, 640],
                         "max_size": args.max_size, "eval_short": 256 if args.smoke else 640,
                         "train_transform_skip_resize": True, "padding": 32},
            "mask_loss_process_patch": "common B1/A5 zero continuous target mass skip only mask",
            "determinism": {"torch_algorithms": True, "warn_only": True,
                            "numerical_bitwise_replay_guaranteed": False,
                            "cuda_roi_align_backward": "native atomic kernel; seeded/RNG-state replay, not bitwise numerical replay",
                            "cudnn_benchmark": False,
                            "cudnn_deterministic": True,
                            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                            "GT_mask_target_sampling": "native_no_grad_standard_alignedFalse_adaptive",
                            "FPN_ROIAlign": "public_API_alignedTrue_samplingRatio2_warn_only_not_bitwise"},
            "dev_max_frames_per_plant": args.dev_max_frames_per_plant,
            "eval_interval": args.eval_interval, "log_every": args.log_every,
            "smoke": args.smoke, "diagnostic_not_experiment": args.smoke,
            "evaluation_roles": ["V_SOURCE", "V_SHIFT"], "sampler_roles": ["TRAIN"],
            "sequential_methods_only": True, "valid_mask_status": args.valid_mask_status,
            "versions": {"python": sys.version, "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "numpy": np.__version__,
                         "scipy": scipy.__version__, "PIL": Image.__version__,
                         "pycocotools": importlib.metadata.version("pycocotools"),
                         "cuda_runtime": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(torch.device(args.device))
                             if args.device.startswith("cuda") else None,
                         "platform": platform.platform()}}


def run(args):
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    args.out = str(out.resolve())
    out.mkdir(parents=True, exist_ok=True)
    heartbeat(out, "initializing")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("Pilot AMP is CUDA-only; both methods must use identical precision")
    if args.physical_batch * args.accumulation != 8 and not args.smoke:
        raise ValueError("P1 requires 8 base images / 16 views per optimizer update")
    if args.steps < 1 or args.steps > args.schedule_steps:
        raise ValueError("--steps must be between 1 and --schedule-steps")
    if not args.resume and (out / "run_config.json").exists():
        raise ValueError("Run already exists; use --resume or a new --out")
    if args.resume and not (out / "run_config.json").is_file():
        raise ValueError("Exact resume requires the original --out with its run_config.json")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # CUDA ROIAlign backward in the installed build uses atomic accumulation.
    # Keep seeds and exact RNG/checkpoint state, but log unsupported deterministic
    # operators instead of falsely promising bitwise CUDA numerical replay.
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rows = read_inventory(args.inventory, args.valid_mask_status, args.data_root)
    if {r["split_role"] for r in rows if r["split_role"] != "TRAIN"} != {"V_SOURCE", "V_SHIFT"}:
        raise ValueError("Both valid dev roles are required before starting any pilot update")
    if args.method == "A5" and len({r["biological_plant_id"] for r in rows
                                     if r["split_role"] == "TRAIN"}) < 2:
        raise ValueError("A5 needs at least two TRAIN biological plants for donors")
    train_ids = {r["image_id"] for r in rows if r["split_role"] == "TRAIN"}
    data = LeafData(rows, read_instances(args.instances, train_ids))
    heartbeat(out, "TRAIN_quantiles")
    qarea, qcrowd, quantile_source = freeze_quantiles(args, rows, data)
    if not 0 <= qarea <= 1 or qcrowd < 0 or math.isnan(qcrowd):
        raise ValueError("Invalid TRAIN q25 thresholds")
    sampler = BalancedSampler(rows, args.seed)
    shorts = (256,) if args.smoke else (512, 576, 640)
    streams = AugmentStreams(args.seed, shorts, args.max_size)
    heartbeat(out, "building_local_COCO_model")
    if Path(args.weights).name != Path(DEFAULT_COCO).name:
        raise ValueError("Only the named public COCO checkpoint is accepted; no PhenoLeaf weights")
    model = MVAModel(args.method, args.weights, args.seed, qarea, qcrowd,
                     test_short=256 if args.smoke else 640, max_size=args.max_size,
                     smoke=args.smoke)
    initial_sha = detector_digest(model.detector)
    config = make_config(args, qarea, qcrowd, quantile_source, initial_sha)
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    step, best, collapse_streak = 0, {"score": -1.0, "step": None}, 0
    window, timings, projection_window = [], [], []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        previous = checkpoint["config"]
        recorded = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        recorded["steps_requested"] = previous["steps_requested"]
        if recorded != jsonable(previous):
            raise ValueError("Checkpoint belongs to a different run_config; mixed run roots refused")
        compare = dict(config)
        compare["steps_requested"] = previous["steps_requested"]
        if jsonable(compare) != jsonable(previous):
            raise ValueError("Resume config/source/inventory/weight/precision differs; exact resume refused")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, tensor in state.items():
                if isinstance(tensor, torch.Tensor):
                    state[key] = tensor.to(device)
        scaler.load_state_dict(checkpoint["scaler"])
        sampler.load_state_dict(checkpoint["sampler"])
        streams.load_state_dict(checkpoint["augmentation"])
        restore_global_rng(checkpoint["global_rng"], device)
        step, best = checkpoint["optimizer_step"], checkpoint["best"]
        collapse_streak = checkpoint["collapse_streak"]
        if collapse_streak >= 3:
            raise RuntimeError("P1 projection collapse pause: same-version continuation is prohibited")
        window = checkpoint["logging_state"]["window"]
        timings = checkpoint["logging_state"]["timings"]
        projection_window = checkpoint["logging_state"]["projection_window"]
        if checkpoint["next_optimizer_step"] != step + 1:
            raise ValueError("Not an optimizer-boundary checkpoint")
        for name in ("train.jsonl", "exposure.jsonl", "timing.jsonl"):
            trim_jsonl_for_resume(out / name, step)
        write_json(out / "latest.json", {"checkpoint": str(Path(args.resume).resolve()),
                   "next_optimizer_step": step + 1,
                   "evaluation_complete": checkpoint["evaluation_complete"]})
        write_json(out / "best.json", {**best, "checkpoint":
                   str(out / f"checkpoint_{best['step']:06d}.pt") if best["step"] is not None else None})
        if not checkpoint["evaluation_complete"]:
            best, collapse_streak = evaluate_and_checkpoint(
                args, model, optimizer, scaler, sampler, streams, device, rows, data,
                step, config, qarea, best, collapse_streak, window, timings, projection_window)
            if collapse_streak >= 3:
                return
    else:
        write_json(out / "run_config.json", config)
        count = augmentation_debug(out, rows, data, args.method, args.seed, shorts, args.max_size)
        if count < 30 and not args.smoke:
            raise ValueError("Fewer than 30 distinct valid TRAIN originals for required augmentation audit")
    while step < args.steps:
        update = step + 1
        heartbeat(out, "training_update", step, next_optimizer_step=update)
        lr = lr_for_update(update, args.schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sync_device(device)
        started = time.perf_counter()
        logs, exposure, stats_list = [], [], []
        for micro in range(args.accumulation):
            heartbeat(out, "training_augmentation", step, next_optimizer_step=update, microbatch=micro)
            images, targets = [], []
            for _ in range(args.physical_batch):
                row = sampler.draw()
                image, target = data.load(row)
                views, truths, record, _ = streams.pair(image, target, row, args.method, data, sampler)
                record["sample_ordinal"] = sampler.draws
                exposure.append(record)
                images.extend(v.to(device) for v in views)
                targets.extend({k: v.to(device) for k, v in truth.items()} for truth in truths)
            heartbeat(out, "training_forward_backward", step, next_optimizer_step=update, microbatch=micro)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                losses = model(images, targets)
                total = sum(losses.values()) / args.accumulation
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss; checkpoint is not advanced")
            scaler.scale(total).backward()
            logs.append({k: float(v.detach()) for k, v in losses.items()})
            stats_list.append(dict(model.last_stats))
            del images, targets, losses, total
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        step = update
        sync_device(device)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        mean_losses = {key: float(np.mean([log[key] for log in logs])) for key in logs[0]}
        mean_stats = {key: float(np.mean([stat[key] for stat in stats_list])) for key in stats_list[0]}
        window.append(dict(mean_losses, loss_total=sum(mean_losses.values()), **mean_stats))
        if args.method == "A5":
            projection_window.extend(stats_list)
        append_jsonl(out / "exposure.jsonl", {"optimizer_step": step, "base_draws": exposure,
                                              "base_images": len(exposure), "views": 2 * len(exposure)})
        if step % args.log_every == 0 or step == args.steps:
            aggregate = {key: float(np.mean([log[key] for log in window])) for key in window[0]}
            append_jsonl(out / "train.jsonl", dict(optimizer_step=step, lr=lr,
                         gradient_norm_before_clip=float(grad_norm), window_updates=len(window),
                         sampled_base_total=sampler.draws, view_exposures_total=2 * sampler.draws,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                         **aggregate))
            print(json.dumps(jsonable({"step": step, "lr": lr, "loss": aggregate["loss_total"]})))
            window = []
        if step % 100 == 0 or step == args.steps:
            append_jsonl(out / "timing.jsonl", {"optimizer_step": step, "method": args.method,
                "mean_update_seconds": float(np.mean(timings)), "updates": len(timings)})
            timings = []
        if step % args.eval_interval == 0 or step == args.steps:
            best, collapse_streak = evaluate_and_checkpoint(
                args, model, optimizer, scaler, sampler, streams, device, rows, data,
                step, config, qarea, best, collapse_streak, window, timings, projection_window)
            if collapse_streak >= 3:
                return
        heartbeat(out, "update_complete", step, next_optimizer_step=step + 1, best=best)
    heartbeat(out, "complete", step, best=best, diagnostic=args.smoke or bool(args.dev_max_frames_per_plant))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["B1", "A5"], required=True)
    p.add_argument("--steps", type=int, default=3000, help="absolute stop update, including resume")
    p.add_argument("--schedule-steps", type=int, default=3000, help="P1 warmup+cosine horizon")
    p.add_argument("--out", required=True, help="absolute or relative to experiment root, e.g. mva/B1_pilot_v1")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--physical-batch", type=int, choices=[1, 2], default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--smoke", action="store_true", help="diagnostic 256 input and reduced proposals; not experiment scores")
    p.add_argument("--resume", help="trusted optimizer-boundary checkpoint created by this program")
    p.add_argument("--inventory", default=str(ROOT / "shared/data_inventory.csv"))
    p.add_argument("--instances", default=str(ROOT / "shared/instances.jsonl"))
    p.add_argument("--audit-summary", default=str(ROOT / "shared/audit_summary.json"))
    p.add_argument("--valid-mask-status", default="valid")
    p.add_argument("--data-root", help="explicit root if the actual inventory uses relative paths")
    p.add_argument("--audit-area-key", default="train.relative_area.q25")
    p.add_argument("--audit-crowding-key", default="train.crowding.q25")
    p.add_argument("--q25-area", type=float)
    p.add_argument("--q25-crowding", type=float)
    p.add_argument("--weights", default=DEFAULT_COCO)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--dev-max-frames-per-plant", type=int, default=0, help="0=all; 8 for diagnostic smoke only")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.accumulation < 1 or args.eval_interval < 1 or not 10 <= args.log_every <= 25:
        raise SystemExit("accumulation/eval interval positive; log interval must be 10..25")
    try:
        run(args)
    except Exception as error:
        if Path(args.out).is_absolute() and Path(args.out).exists():
            heartbeat(args.out, "failed", error=str(error), traceback=traceback.format_exc())
        raise

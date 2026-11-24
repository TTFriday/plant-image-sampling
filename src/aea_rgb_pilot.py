"""Frozen exploratory cross-route RGB reference; never a formal AEA model.

extract must be explicitly scheduled AFTER the entire existing MVA queue finishes.
analyze reads only inventory metadata, the original GT window manifest and cache.
Neither action trains, calibrates thresholds, selects a best phase or evaluates tests.
Synthetic tests do not run either action on real inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import aea_sampling as gt

VERSION = "aea_rgb_exploratory_v1"
REFERENCE_KIND = "exploratory_cross_route_reference"
ROLES = ("V_SOURCE", "V_SHIFT")
EXPECTED_FRAMES = 2947
MODEL_VERSION = "mva-p1-v1.2"
SOURCE_PINS = {
    "aea_sampling.py": "4ab19618d3a9771c41f64314ef31263ffc0eb25b08136c894e246a56062cd2e2",
    "mva_model.py": "736715b9ad8e1c2ed8a75072378b3e9b9d8578f01ae0ac7cd62856e1ce2eb55e",
    "mva_pilot.py": "3f4cf375722ddc1126b5c6ada1f50cffb81bac820a15628d3c24c54a0d9fb89e",
}
FIELDS = (
    "image_id", "genotype", "biological_plant_id", "replicate_id", "timestamp",
    "split_role", "width", "height", "gt_area_fraction", "n_gt_leaves",
    "model_area_fraction", "n_pred_leaves", "union_pixels", "latency_s",
    "rgb_sha256", "mask_sha256", "input_sha256", "checkpoint_sha256",
    "code_sha256", "reference_config_sha256", "measurement_status", "row_sha256",
)
METHODS = ("U_-1/3", "U_0", "U_+1/3", "C", "J", "R", "online")
FUNCTIONALS = ("area_mean", "leaf_mean", "area_integral")
TERMS = ("e_M", "e_S", "e_X", "e_total")
METRICS = tuple(f"{f}_{term}{suffix}" for f in FUNCTIONALS for term in TERMS
                for suffix in ("", "_abs")) + (
    "E_A_pp", "E_N_leaves", "joint_pass", "leaf_pass", "relative_area_integral_pct")
NOTE = ("Exploratory cross-route B1 development-best reference, not the independent "
        "40-epoch AEA measurement network. Formal measurement, calibration and external "
        "validation remain pending. Real GT is a discrete visible-projection reference, "
        "not biomass; fixed-pixel area remains provisional with uncertain scale.")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    return gt.sha256_file(Path(path))


def canonical(value):
    return json.dumps(gt.json_safe(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(obj))
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        result = list(reader)
        require(all(None not in r and all(v is not None for v in r.values()) for r in result),
                "malformed_csv")
        return result


def boolean(value):
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise ValueError("invalid_boolean")


def integer(value):
    x = float(value)
    require(math.isfinite(x) and x >= 0 and x == int(x), "invalid_integer")
    return int(x)


def check_sources():
    for name, expected in SOURCE_PINS.items():
        require(sha(ROOT / "src" / name) == expected, f"protected_source_changed:{name}")
    return {**SOURCE_PINS, "aea_rgb_pilot.py": sha(Path(__file__))}


def input_metadata(row):
    """Called only after role filtering; never open any GT or RGB file."""
    require(row["split_role"] in ROLES, "forbidden_measurement_role")
    require(row["mask_status"] == "valid", "invalid_GT_metadata_status")
    require(integer(row["unknown_pixels"]) == 0, "unknown_GT_pixels")
    w, h = integer(row["width"]), integer(row["height"])
    pixels, leaves = integer(row["union_pixels"]), integer(row["n_instances"])
    area = float(row["area_fraction"])
    require(w > 0 and h > 0 and pixels <= w * h and math.isfinite(area)
            and 0 <= area <= 1 and abs(area - pixels / (w * h)) <= 5e-7,
            "inconsistent_GT_area_metadata")
    require((pixels == 0) == (leaves == 0), "inconsistent_GT_instance_union")
    require(all(gt._hash_valid(row[k]) for k in ("rgb_sha256", "mask_sha256")),
            "invalid_input_hash")
    require(all(row[k].strip() for k in ("image_id", "biological_plant_id", "genotype",
                                        "replicate_id", "rgb_path", "mask_path")),
            "missing_input_identity")
    gt.parse_timestamp(row["timestamp"])
    # Bind all source metadata, including paths and identity/quarantine annotations.
    return {"width": w, "height": h, "gt_area_fraction": pixels / (w * h),
            "n_gt_leaves": leaves, "input_sha256": digest(row)}


def eligible_inventory(rows):
    """Locked/TRAIN rows: only role and identity/hash metadata are consulted."""
    identities = defaultdict(set)
    counts = Counter()
    for row in rows:
        role = row["split_role"]
        counts[role] += 1
        for kind, value in (("plant", row.get("biological_plant_id", "")),
                            ("unit", f'{row.get("genotype", "")}::{row.get("replicate_id", "")}'),
                            ("RGB", row.get("rgb_sha256", "")),
                            ("image_id", row.get("image_id", ""))):
            if value:
                identities[(kind, value)].add(role)
    result, seen = [], set()
    for row in rows:
        if row["split_role"] not in ROLES:
            continue
        if row["mask_status"] != "valid":
            continue
        for kind, value in (("plant", row["biological_plant_id"]),
                            ("unit", f'{row["genotype"]}::{row["replicate_id"]}'),
                            ("RGB", row["rgb_sha256"]), ("image_id", row["image_id"])):
            require(len(identities[(kind, value)]) == 1, f"cross_role_identity:{kind}")
        require(row["image_id"] not in seen, "duplicate_development_image_id")
        seen.add(row["image_id"])
        input_metadata(row)
        result.append(row)
    return sorted(result, key=lambda r: (r["split_role"], r["genotype"],
                  r["biological_plant_id"], r["timestamp"], r["image_id"])), dict(counts)


def contained_file(run, text):
    path = Path(text)
    path = (path if path.is_absolute() else run / path).resolve()
    require(path.parent == run.resolve() and path.is_file(), "artifact_outside_local_run_or_missing")
    return path


def normalized_config(obj):
    # Checkpoints from exact resume may retain the original requested stop count.
    obj = dict(obj)
    obj.pop("steps_requested", None)
    return gt.json_safe(obj)


def validate_run_config(config, inventory_hash):
    require(config["version"] == MODEL_VERSION and config["method"] == "B1", "not_B1_v1.2")
    require(config["schedule_steps"] == 3000 and config["steps_requested"] == 3000,
            "not_final_3000_run_config")
    require(config["seed"] == 43 and config["precision"] == "FP32", "unexpected_seed_or_precision")
    require(config["smoke"] is False and config["diagnostic_not_experiment"] is False
            and config["dev_max_frames_per_plant"] == 0, "diagnostic_run_refused")
    require(config["inventory_sha256"] == inventory_hash, "model_inventory_hash_mismatch")
    require(config["evaluation_roles"] == list(ROLES) and config["sampler_roles"] == ["TRAIN"],
            "training_or_selection_role_mismatch")
    require(config["geometry"] == {"shorts": [512, 576, 640], "max_size": 1333,
            "eval_short": 640, "train_transform_skip_resize": True, "padding": 32},
            "geometry_changed")
    require(config["effective_base_images"] == 8 and config["views_per_base"] == 2
            and config["eval_interval"] == 500, "training_protocol_changed")
    require(config["optimizer"] == {"type": "SGD", "lr": .005, "momentum": .9,
            "weight_decay": 1e-4, "gradient_clip": 5}, "optimizer_protocol_changed")
    for name in ("mva_model.py", "mva_pilot.py"):
        snapshots = [v for k, v in config["sources"].items() if Path(k).name == name]
        require(len(snapshots) == 1 and snapshots[0]["sha256"] == SOURCE_PINS[name],
                f"checkpoint_source_hash_changed:{name}")
        require(snapshots[0]["source"] == (ROOT / "src" / name).read_text(encoding="utf-8"),
                f"checkpoint_source_snapshot_changed:{name}")


def validate_report(report, step, expected_frames=EXPECTED_FRAMES):
    require(report["optimizer_step"] == step and report["diagnostic_subset"] is False
            and report["dev_max_frames_per_plant"] == 0 and report["all_instances_kept"] is True,
            "incomplete_or_subset_dev_report")
    require(set(report["role_macro_AP"]) == set(ROLES)
            and set(report["per_genotype"]) == set(ROLES), "dev_report_roles_changed")
    require(sum(g["frames"] for role in report["per_genotype"].values() for g in role.values())
            == expected_frames, "dev_report_frame_count_mismatch")
    score = float(report["dev_score"])
    require(math.isfinite(score) and abs(score - np.mean(list(report["role_macro_AP"].values())))
            <= 1e-10, "dev_score_not_equal_role_macro")
    for role in ROLES:
        require(all(g.get("mask_threshold") == .5 and g.get("count_score_threshold") == .5
                    and g.get("ap_score_threshold") == .001 and g.get("max_detections") == 100
                    for g in report["per_genotype"][role].values()), "dev_measurement_parameters_changed")
        require(report["per_genotype"][role], "empty_dev_genotype_group")
        require(abs(float(report["role_macro_AP"][role]) - np.mean([
            float(g["AP"]) for g in report["per_genotype"][role].values()])) <= 1e-10,
            "dev_score_not_genotype_macro")
    return score


def validate_checkpoint(checkpoint, config, step, best):
    require(checkpoint["version"] == MODEL_VERSION and checkpoint["optimizer_step"] == step
            and checkpoint["next_optimizer_step"] == step + 1
            and checkpoint["evaluation_complete"] is True, "partial_or_wrong_checkpoint")
    require(normalized_config(checkpoint["config"]) == normalized_config(config),
            "checkpoint_run_config_mismatch")
    require(checkpoint["best"]["step"] == best["step"]
            and abs(float(checkpoint["best"]["score"]) - float(best["score"])) <= 1e-10
            and checkpoint["best"]["diagnostic_subset"] is False, "checkpoint_best_mismatch")
    require(isinstance(checkpoint["model"], dict) and checkpoint["model"], "missing_model_state")
    require(checkpoint["sampler"]["draws"] == step * 8, "checkpoint_training_exposure_incomplete")


def queue_finished(path):
    state = read_json(path)
    require(state["status"] == "completed" and state["current_job"] is None,
            "original_MVA_queue_not_fully_finished")
    jobs = state["completed_jobs"]
    require(len(jobs) == 2 and all(j["returncode"] == 0 for j in jobs)
            and {j["name"] for j in jobs} == {"B1_seed43_3000_updates", "A5_seed43_3000_updates"},
            "original_MVA_queue_incomplete")
    queue = path.with_name("pilot_queue_v1.json")
    require(state["queue_sha256"] == sha(queue), "original_queue_hash_mismatch")
    return {"path": str(path), "sha256": sha(path), "queue_sha256": sha(queue)}


def frozen_checkpoint(run, inventory_hash, expected_frames=EXPECTED_FRAMES, expected_stream_hash=None):
    """No CUDA calls; validate provenance BEFORE trusted-local pickle loading."""
    sources = check_sources()
    config = read_json(run / "run_config.json")
    validate_run_config(config, inventory_hash)
    current_versions = {k: importlib.metadata.version(package) for k, package in
                        (("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
                         ("scipy", "scipy"), ("PIL", "Pillow"))}
    require(all(str(config["versions"][k]) == value for k, value in current_versions.items()),
            "inference_library_version_changed_since_training")
    from src.mva_model import COCO_V1_SHA256
    require(config["coco_weight_sha256"] == COCO_V1_SHA256
            and config["coco_source"] == "torchvision MaskRCNN_ResNet50_FPN_Weights.COCO_V1 (public generic COCO)",
            "unverified_generic_initialization")
    coco = Path(config["coco_weights"])
    require(coco.is_absolute() and coco.is_file() and sha(coco) == COCO_V1_SHA256,
            "local_COCO_initialization_provenance_missing")
    best, latest, status = [read_json(run / n) for n in ("best.json", "latest.json", "status.json")]
    require(status["phase"] == "complete" and status["optimizer_step"] == 3000
            and status["diagnostic"] is False, "run_not_final_complete")
    require(latest["next_optimizer_step"] == 3001 and latest["evaluation_complete"] is True,
            "latest_not_final_complete")
    reports = {step: read_json(run / f"dev_{step:06d}.json") for step in range(500, 3001, 500)}
    scores = {s: validate_report(r, s, expected_frames) for s, r in reports.items()}
    require(len({r["eval_stream_sha256"] for r in reports.values()}) == 1, "dev_stream_changed")
    if expected_stream_hash is not None:
        require(all(r["eval_stream_sha256"] == expected_stream_hash for r in reports.values()),
                "dev_stream_not_inventory_complete")
    winner = max(scores, key=lambda s: (scores[s], -s))
    require(best["step"] == winner and best["diagnostic_subset"] is False
            and abs(float(best["score"]) - scores[winner]) <= 1e-10, "not_development_selected_best")
    require(status["best"] == {k: best[k] for k in ("score", "step", "diagnostic_subset")},
            "final_status_best_mismatch")
    best_path, latest_path = contained_file(run, best["checkpoint"]), contained_file(run, latest["checkpoint"])
    require(best_path.name == f"checkpoint_{winner:06d}.pt"
            and latest_path.name == "checkpoint_003000.pt", "checkpoint_filename_step_mismatch")
    provenance = {"run": str(run), "best_checkpoint": str(best_path), "best_step": winner,
        "completed_training_updates": 3000, "selection": "equal-role genotype-macro development AP; earliest tie",
        "checkpoint_sha256": sha(best_path), "latest_checkpoint_sha256": sha(latest_path),
        "run_config_sha256": sha(run / "run_config.json"), "source_hashes": sources,
        "artifacts": {n: sha(run / n) for n in ("best.json", "latest.json", "status.json")},
        "dev_report_hashes": {str(s): sha(run / f"dev_{s:06d}.json") for s in reports},
        "initial_COCO_sha256": COCO_V1_SHA256,
        "trusted_local_checkpoint_weights_only": False,
        "trust_boundary": "only checkpoints inside validated local run; never downloaded/untrusted pickle"}
    import torch
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, config, winner, best)
    if latest_path == best_path:
        final = checkpoint
    else:
        final = torch.load(latest_path, map_location="cpu", weights_only=False)
    validate_checkpoint(final, config, 3000, best)
    require(sha(best_path) == provenance["checkpoint_sha256"]
            and sha(latest_path) == provenance["latest_checkpoint_sha256"], "checkpoint_changed_during_read")
    return checkpoint, config, provenance


def restore_model(checkpoint, config):
    """Rebuild architecture WITHOUT treating trained weights as COCO initialization."""
    import torch
    from src.mva_model import MVAModel
    model = MVAModel("B1", weights_path=None, seed=config["seed"], test_short=640,
                     max_size=1333, smoke=False)
    state = checkpoint["model"]
    require(all(isinstance(x, torch.Tensor) and (not x.is_floating_point() or torch.isfinite(x).all())
                for x in state.values()), "nonfinite_or_non_tensor_trained_parameter")
    model.load_state_dict(state, strict=True)
    restored = model.state_dict()
    require(all(restored[k].dtype == v.dtype and restored[k].shape == v.shape
                and torch.equal(restored[k], v) for k, v in state.items()), "trained_parameter_roundtrip_changed")
    model.eval().float().requires_grad_(False)
    detector = model.inference_model()
    require(tuple(detector.transform.min_size) == (640,) and detector.transform.max_size == 1333,
            "actual_eval_transform_changed")
    require(detector.roi_heads.nms_thresh == .5 and detector.roi_heads.detections_per_img == 100,
            "actual_NMS_or_max_detections_changed")
    # Preserve the factory's candidate floor. Torchvision uses strict '>' there;
    # final measurement filtering uses >= .5, exactly as the MVA discrete metric.
    # Low-score candidates cannot remove a higher-score result under greedy NMS.
    require(detector.roi_heads.score_thresh == .001, "factory_candidate_score_floor_changed")
    return model


def measure_output(output, width, height):
    """Torchvision already pastes probabilities; fallback resizes SOFT values first."""
    import torch
    from torch.nn import functional as F
    scores = output["scores"].detach().float().cpu()
    probabilities = output["masks"].detach().float().cpu()
    require(probabilities.ndim == 4 and probabilities.shape[1] == 1
            and len(probabilities) == len(scores) and scores.ndim == 1, "invalid_output_shape")
    require(torch.isfinite(scores).all() and torch.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(), "nonfinite_output_probability")
    require(len(scores) <= 100, "max_detections_exceeded")
    keep = scores >= .5
    probabilities = probabilities[keep, 0]
    if len(probabilities) and probabilities.shape[-2:] != (height, width):
        probabilities = F.interpolate(probabilities[:, None], (height, width),
                                      mode="bilinear", align_corners=False)[:, 0]
    binary = probabilities >= .5
    union = binary.any(0) if len(binary) else torch.zeros((height, width), dtype=torch.bool)
    pixels = int(union.sum())
    # Count score-qualified instances, as in MVA discrete metrics, even zero-support masks.
    return {"model_area_fraction": pixels / (width * height), "n_pred_leaves": len(binary),
            "union_pixels": pixels, "measurement_status": "valid"}


def make_measurement(row, output, latency, config_hash, provenance):
    result = {k: row[k] for k in ("image_id", "genotype", "biological_plant_id", "replicate_id",
                                   "timestamp", "split_role", "rgb_sha256", "mask_sha256")}
    result.update(input_metadata(row))
    result.update(measure_output(output, result["width"], result["height"]))
    result.update(latency_s=float(latency), checkpoint_sha256=provenance["checkpoint_sha256"],
                  code_sha256=provenance["source_hashes"]["aea_rgb_pilot.py"],
                  reference_config_sha256=config_hash)
    result["row_sha256"] = digest(result)
    return result


def normalized_measurement(row):
    require(set(row) == set(FIELDS), "cache_schema_changed")
    result = dict(row)
    for k in ("width", "height", "n_gt_leaves", "n_pred_leaves", "union_pixels"):
        result[k] = integer(result[k])
    for k in ("gt_area_fraction", "model_area_fraction", "latency_s"):
        result[k] = float(result[k])
        require(math.isfinite(result[k]), "nonfinite_cache_measurement")
    require(result["latency_s"] >= 0 and 0 <= result["model_area_fraction"] <= 1
            and result["measurement_status"] == "valid" and result["n_pred_leaves"] <= 100,
            "invalid_cache_measurement")
    require(result["union_pixels"] <= result["width"] * result["height"]
            and abs(result["model_area_fraction"] - result["union_pixels"] /
                    (result["width"] * result["height"])) <= 1e-12, "inconsistent_cache_union")
    expected = digest({k: v for k, v in result.items() if k != "row_sha256"})
    require(result["row_sha256"] == expected, "cache_row_poisoned")
    return result


def verify_cache_records(records, inventory, config_hash, provenance):
    inputs = {r["image_id"]: r for r in inventory}
    cache = {}
    for raw in records:
        r = normalized_measurement(raw)
        key = r["image_id"]
        require(key in inputs and key not in cache, "unknown_or_duplicate_cache_image")
        source = inputs[key]
        for k in ("genotype", "biological_plant_id", "replicate_id", "timestamp", "split_role",
                  "rgb_sha256", "mask_sha256"):
            require(r[k] == source[k], f"cache_input_identity_changed:{k}")
        require(all(r[k] == v for k, v in input_metadata(source).items()), "cache_input_metadata_changed")
        require(r["reference_config_sha256"] == config_hash
                and r["checkpoint_sha256"] == provenance["checkpoint_sha256"]
                and r["code_sha256"] == provenance["source_hashes"]["aea_rgb_pilot.py"],
                "cache_reference_poisoned")
        cache[key] = r
    return cache


def cache_chain(records):
    chain = "0" * 64
    for row in records:
        chain = hashlib.sha256((chain + row["row_sha256"]).encode("ascii")).hexdigest()
    return chain


def load_cache(directory, inventory, inventory_hash, complete=True):
    config_path = directory / "model_reference_config.json"
    config_hash = sha(config_path)
    require((directory / "model_reference_config.sha256").read_text().strip() == config_hash,
            "reference_config_hash_changed")
    config = read_json(config_path)
    require(config["version"] == VERSION and config["reference_kind"] == REFERENCE_KIND
            and config["inventory_sha256"] == inventory_hash
            and config["allowed_roles"] == list(ROLES), "cache_configuration_changed")
    require(config["provenance"]["source_hashes"] == check_sources(), "cache_code_changed")
    require(config["inference"] == inference_config(), "cache_inference_configuration_changed")
    status = read_json(directory / "status.json")
    records = read_csv(directory / "perframe_measurements.csv")
    cache = verify_cache_records(records, inventory, config_hash, config["provenance"])
    require(status["reference_config_sha256"] == config_hash and status["verified_frames"] == len(records)
            and status["cache_chain_sha256"] == cache_chain(records), "cache_status_chain_mismatch")
    traces = [json.loads(line) for line in (directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    require([(t["image_id"], t["row_sha256"]) for t in traces] ==
            [(r["image_id"], r["row_sha256"]) for r in records], "cache_trace_mismatch")
    if complete:
        require(status["phase"] == "complete" and len(cache) == len(inventory)
                and config["expected_frames"] == len(inventory), "cache_not_complete")
        require(status["cache_file_sha256"] == sha(directory / "perframe_measurements.csv"),
                "cache_file_poisoned")
    else:
        require(status["phase"] in ("extracting", "failed", "complete"), "unsafe_resume_phase")
    return cache, config, config_hash


def inference_config():
    return {"precision": "FP32", "eval_resize": "torchvision short side 640, max long side 1333",
        "eval_short": 640, "max_size": 1333, "normalization": "unchanged COCO mean/std; stride32 padding",
        "not_AEA_long_side640": True, "mask_threshold": .5, "score_threshold": .5,
        "score_comparison": ">=", "unchanged_detector_candidate_score_floor": .001,
        "NMS_threshold": .5, "max_detections": 100, "batch": 1,
        "probability_postprocess": "paste/restore original image dimensions BEFORE threshold",
        "empty_output": "retain every valid image; zero union and zero leaves for no detections",
        "leaf_count": "score-qualified instances, including any zero-support thresholded masks",
        "timing": "synchronized per-frame wall time including hash/read/decode/input transfer/inference/measurement; not a formal JRTIP timing claim"}


def extract(inventory_path, run, out, device="cuda", data_root=None, resume=False):
    queue_provenance = queue_finished(ROOT / "mva" / "pilot_queue_v1.status.json")
    check_sources()
    inventory_hash = sha(inventory_path)
    rows, role_counts = eligible_inventory(gt.read_inventory(inventory_path))
    require(len(rows) == EXPECTED_FRAMES, "unexpected_development_valid_frame_count")
    require({r["split_role"] for r in rows} == set(ROLES), "missing_development_role")
    stream_hash = hashlib.sha256(json.dumps([r["image_id"] for r in rows]).encode()).hexdigest()
    checkpoint, training_config, provenance = frozen_checkpoint(run, inventory_hash, expected_stream_hash=stream_hash)
    data_root = (data_root or inventory_path.parent.parent).resolve()
    config = {"version": VERSION, "reference_kind": REFERENCE_KIND, "notice": NOTE,
        "formal_AEA_40epoch_measurement_completed": False, "external_validation_completed": False,
        "allowed_roles": list(ROLES), "inventory": str(inventory_path), "inventory_sha256": inventory_hash,
        "expected_frames": len(rows), "role_metadata_counts": role_counts, "data_root": str(data_root),
        "provenance": provenance, "queue_completion": queue_provenance, "device": device,
        "inference": inference_config(), "no_threshold_scan": True, "no_C_POLICY_or_test_effects": True,
        "measurement_cache_only": True, "GT_metadata_definition": "audited union_pixels / original width*height and n_instances",
        "software": {k: importlib.metadata.version(k) for k in ("numpy", "torch", "torchvision", "Pillow")}}
    cached = {}
    if resume:
        cached, previous, config_hash = load_cache(out, rows, inventory_hash, complete=False)
        require(config == previous, "resume_configuration_changed")
    else:
        require(not out.exists() or not any(out.iterdir()), "output_not_empty_use_explicit_verified_resume")
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "model_reference_config.json", config)
        config_hash = sha(out / "model_reference_config.json")
        (out / "model_reference_config.sha256").write_text(config_hash + "\n", encoding="ascii")
        gt.write_csv(out / "perframe_measurements.csv", [], FIELDS)
        (out / "trace.jsonl").write_text("", encoding="utf-8")
    # Verify source input hashes on resume too; skip only content-hash-verified frames.
    import torch
    from PIL import Image
    target_device = torch.device(device)
    require(target_device.type in ("cpu", "cuda"), "unsupported_device")
    model = restore_model(checkpoint, training_config).to(target_device)
    del checkpoint
    records = list(cached.values())
    chain = cache_chain(records)
    started = gt.utc_now()
    def status(phase, **extra):
        obj = {"version": VERSION, "phase": phase, "UTC": gt.utc_now(), "UTC_start": started,
               "reference_config_sha256": config_hash, "verified_frames": len(records),
               "expected_frames": len(rows), "cache_chain_sha256": chain, **extra}
        atomic_json(out / "status.json", obj)
    def sync():
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
    status("extracting")
    try:
        with (out / "perframe_measurements.csv").open("a", encoding="utf-8", newline="") as stream, \
                (out / "trace.jsonl").open("a", encoding="utf-8") as trace:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            for row in rows:
                path = Path(row["rgb_path"])
                path = path if path.is_absolute() else data_root / path
                if row["image_id"] in cached:
                    require(sha(path) == row["rgb_sha256"], "resume_RGB_content_hash_changed")
                    continue
                sync()
                tick = time.perf_counter()
                require(sha(path) == row["rgb_sha256"], "RGB_content_hash_changed")
                with Image.open(path) as image:
                    require(image.size == (integer(row["width"]), integer(row["height"])), "RGB_dimensions_changed")
                    rgb = np.asarray(image.convert("RGB")).copy()
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).to(target_device)
                with torch.inference_mode(), torch.autocast(device_type=target_device.type, enabled=False):
                    output = model([tensor])[0]
                measurement = make_measurement(row, output, 0, config_hash, provenance)
                sync()
                measurement["latency_s"] = time.perf_counter() - tick
                measurement["row_sha256"] = digest({k: v for k, v in measurement.items() if k != "row_sha256"})
                writer.writerow(measurement)
                stream.flush()
                trace.write(canonical({"image_id": row["image_id"], "row_sha256": measurement["row_sha256"],
                                       "latency_s": measurement["latency_s"]}).decode() + "\n")
                trace.flush()
                records.append(measurement)
                chain = hashlib.sha256((chain + measurement["row_sha256"]).encode("ascii")).hexdigest()
                # Commit every frame for safe prefix verification; print heartbeat every100.
                status("extracting")
                if len(records) % 100 == 0 or len(records) == len(rows):
                    print(json.dumps({"phase": "extracting", "frames": len(records), "total": len(rows)}), flush=True)
                del output, tensor
        require(len(records) == len(rows), "extraction_incomplete")
        status("complete", cache_file_sha256=sha(out / "perframe_measurements.csv"))
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}:{exc}")
        raise
    return {"phase": "complete", "frames": len(records), "reference_kind": REFERENCE_KIND}


def validate_window(w):
    require(w.role in ROLES, "forbidden_RGB_replay_role")
    gt._check_times(w.times)
    require(156 <= w.length <= 168 and len(w.times) >= 15
            and float(np.median(np.diff(w.times))) <= 12 and float(max(np.diff(w.times))) <= 18,
            "incomplete_original_window")
    require(w.times[0] == 0 and len(w.image_ids) == len(w.times) and len(set(w.image_ids)) == len(w.times),
            "invalid_original_window_grid")
    require(w.area.shape == w.times.shape and w.leaves.shape == w.times.shape
            and np.all(np.isfinite(w.area)) and np.all(np.isfinite(w.leaves)), "invalid_window_GT_signals")


def windows_from_manifest(audits, inventory):
    """Reconstruct exactly the previously audited common grid; never call audit_windows."""
    lookup = {r["image_id"]: r for r in inventory}
    windows, failed, seen = [], [], set()
    for audit in audits:
        if audit["split_role"] not in ROLES:
            continue
        plant = audit["plant_id"]
        require(plant not in seen, "duplicate_original_window")
        seen.add(plant)
        if not boolean(audit["window_eligible"]):
            failed.append({"plant_id": plant, "biological_plant_id": audit["biological_plant_id"],
                           "split_role": audit["split_role"], "status": "original_window_ineligible",
                           "failure_reason": audit["exclusion_reasons"]})
            continue
        require(not json.loads(audit["exclusion_reasons"]), "eligible_GT_window_has_exclusions")
        a, _ = gt.parse_timestamp(audit["a"])
        b, _ = gt.parse_timestamp(audit["b"])
        entries = json.loads(audit["row_times_and_errors"])
        selected = [r for r in entries if boolean(r["rgb_gt_valid"]) and not boolean(r["duplicate"])
                    and r["elapsed_h"] is not None and 0 <= float(r["elapsed_h"]) <= (b-a).total_seconds()/3600]
        selected.sort(key=lambda r: float(r["elapsed_h"]))
        require(len(selected) == integer(audit["N"]) and len(selected) >= 15,
                "original_window_manifest_incomplete")
        values, times, ids, dimensions = [], [], [], set()
        for entry in selected:
            key = entry["image_id"]
            require(key in lookup, "original_window_image_missing_from_pilot_inventory")
            r = lookup[key]
            require(r["split_role"] == audit["split_role"] and r["biological_plant_id"] == audit["biological_plant_id"]
                    and r["genotype"] == audit["genotype"] and r["replicate_id"] == audit["replicate_id"],
                    "GT_window_inventory_identity_changed")
            stamp, _ = gt.parse_timestamp(r["timestamp"])
            require(stamp.isoformat() == entry["datetime"] and
                    abs((stamp-a).total_seconds()/3600 - float(entry["elapsed_h"])) <= 1e-10,
                    "GT_window_timestamp_changed")
            m = input_metadata(r)
            values.append((m["gt_area_fraction"], m["n_gt_leaves"]))
            times.append(float(entry["elapsed_h"]))
            ids.append(key)
            dimensions.add((m["width"], m["height"]))
        area_eligible = boolean(audit["area_eligible"])
        all_dimensions = json.loads(audit["dimensions_all_GT_frames"])
        require(not area_eligible or (len(dimensions) == 1 and len(all_dimensions) == 1),
                "unstable_dimensions_marked_area_eligible")
        signal = np.asarray(values, dtype=float)
        first = lookup[ids[0]]
        w = gt.Window(plant, audit["biological_plant_id"], audit["genotype"], audit["replicate_id"],
            audit["split_role"], a, np.asarray(times), signal[:, 0], signal[:, 1], ids,
            area_eligible, integer(first["width"]), integer(first["height"]))
        validate_window(w)
        require(abs(w.length - float(audit["L_h"])) <= 1e-10
                and a + timedelta(hours=w.length) == b, "GT_window_boundary_changed")
        windows.append(w)
    return windows, failed


def functional_values(w, area, leaves, indices):
    s = np.asarray(indices, dtype=int)
    require(len(s) >= 2 and s[0] == 0 and s[-1] == len(w.times)-1
            and np.all(np.diff(s) > 0) and np.all(s >= 0) and np.all(s < len(w.times)),
            "functional_requires_unique_indices_and_endpoints")
    integral = gt.trapezoid(w.times[s], np.asarray(area)[s])
    leaf_integral = gt.trapezoid(w.times[s], np.asarray(leaves)[s])
    return {"area_mean": integral / w.length, "leaf_mean": leaf_integral / w.length,
            "area_integral": integral}


def joint_errors(w, rgb_area, rgb_leaves, selected):
    """All four F on exactly the same window/grid; full reference always real GT."""
    full = np.arange(len(w.times))
    g_full = functional_values(w, w.area, w.leaves, full)
    r_full = functional_values(w, rgb_area, rgb_leaves, full)
    g_sub = functional_values(w, w.area, w.leaves, selected)
    r_sub = functional_values(w, rgb_area, rgb_leaves, selected)
    result = {"full_GT_mean_area_fraction": g_full["area_mean"] if w.area_eligible else math.nan,
              "full_GT_mean_leaves": g_full["leaf_mean"],
              "relative_integral_status": "area_ineligible" if not w.area_eligible else
                  "defined" if g_full["area_mean"] >= .001 else "undefined_small_GT_denominator"}
    for f in FUNCTIONALS:
        e_m = r_full[f] - g_full[f]
        e_s = g_sub[f] - g_full[f]
        e_x = (r_sub[f] - g_sub[f]) - e_m
        e_total = r_sub[f] - g_full[f]
        require(math.isclose(e_total, e_m + e_s + e_x, abs_tol=1e-12, rel_tol=1e-10),
                "signed_decomposition_identity_failed")
        result[f + "_identity_residual"] = e_total - (e_m + e_s + e_x)
        allowed = f == "leaf_mean" or w.area_eligible
        for term, value in zip(TERMS, (e_m, e_s, e_x, e_total)):
            result[f"{f}_{term}"] = value if allowed else math.nan
            result[f"{f}_{term}_abs"] = abs(value) if allowed else math.nan
        for name, value in (("full_GT", g_full[f]), ("full_RGB", r_full[f]),
                            ("sub_GT", g_sub[f]), ("sub_RGB", r_sub[f])):
            result[f"{f}_{name}"] = value if allowed else math.nan
    ea = 100 * result["area_mean_e_total_abs"]
    en = result["leaf_mean_e_total_abs"]
    result.update(E_A_pp=ea, E_N_leaves=en, leaf_pass=float(en <= .5 + 1e-12),
                  joint_pass=float(ea <= 1 + 1e-12 and en <= .5 + 1e-12) if w.area_eligible else math.nan,
                  relative_area_integral_pct=(100 * result["area_integral_e_total_abs"] / g_full["area_integral"]
                      if w.area_eligible and g_full["area_mean"] >= .001 else math.nan))
    return result


def select_schedule(w, k, method, rgb_read):
    if method in ("U_-1/3", "U_0", "U_+1/3"):
        return gt.uniform_schedule(w, k, {"U_-1/3": -1/3, "U_0": 0., "U_+1/3": 1/3}[method])
    if method == "C":
        return gt.clock_schedule(w, k)
    if method == "J":
        return gt.index_schedule(w.times, k)
    if method == "online":
        return gt.online_schedule(w.times, k, rgb_read, w.tolerance)
    raise ValueError("unknown_schedule")


def gt_oracle(w, k, candidates):
    """GT-only hindsight candidate/greedy diagnostic; cannot enter deployable gates."""
    def loss(s):
        v = functional_values(w, w.area, w.leaves, s)
        f = functional_values(w, w.area, w.leaves, np.arange(len(w.times)))
        leaf = abs(v["leaf_mean"] - f["leaf_mean"]) / .5
        return max(100 * abs(v["area_mean"] - f["area_mean"]), leaf) if w.area_eligible else leaf
    selected, evaluations = [0, len(w.times)-1], 0
    while len(selected) < k:
        options = []
        for i in range(1, len(w.times)-1):
            if i not in selected:
                options.append((loss(sorted(selected + [i])), i))
                evaluations += 1
        selected = sorted(selected + [min(options)[1]])
    options = candidates + [np.asarray(selected)]
    return min(options, key=lambda s: (loss(s), tuple(s))), evaluations + len(options)


def replay_joint(windows, cache):
    records = []
    for w in windows:
        require(w.role in ROLES, "forbidden_RGB_replay_role")
        failure = ""
        try:
            validate_window(w)
        except (ValueError, gt.ScheduleFailure) as exc:
            failure = str(exc)
        missing = [key for key in w.image_ids if key not in cache]
        if missing:
            failure = "window_failed_cache_incomplete"
        if not failure:
            a = np.asarray([cache[key]["model_area_fraction"] for key in w.image_ids])
            n = np.asarray([cache[key]["n_pred_leaves"] for key in w.image_ids], dtype=float)
            require(np.all(np.isfinite(a)) and np.all(np.isfinite(n)), "cache_nonfinite_signal")
        for k in (*gt.BUDGETS, "full_N"):
            candidates = []
            specs = [(m, 0) for m in METHODS if m != "R"] + [("R", r) for r in range(20)] + [("oracle", 0)]
            if k == "full_N":
                specs = [("fullRGB", 0)]
            for method, repeat in specs:
                row = {"plant_id": w.plant_id, "biological_plant_id": w.biological_plant_id,
                    "genotype": w.genotype, "split_role": w.role, "K": k, "N": len(w.times),
                    "L_h": w.length, "a": w.start.isoformat(), "b": (w.start + timedelta(hours=w.length)).isoformat(),
                    "method": method, "repeat": repeat, "area_eligible": w.area_eligible,
                    "area_interpretation": gt.AREA_NOTE, "status": "failed", "failure_reason": failure,
                    "missing_cache_image_ids": missing, "selected_indices": [], "actual_K": 0,
                    "oracle_diagnostic_only": method == "oracle", "oracle_candidate_evaluations": 0,
                    "deployability": "NOT_DEPLOYABLE_GT_HINDSIGHT" if method == "oracle" else
                        "RGB_OBSERVED_ONLY_PREKNOWN_TIME_OPPORTUNITIES" if method == "online" else
                        "METADATA_PROJECTED_REPLAY_NOT_CAMERA_VALIDATION"}
                start = time.perf_counter()
                try:
                    if failure:
                        raise gt.ScheduleFailure(failure)
                    if k != "full_N" and k > len(w.times):
                        raise gt.ScheduleFailure("K_exceeds_N")
                    observed = []
                    def read_rgb(i):
                        observed.append(int(i))
                        return a[i], n[i]
                    if method == "fullRGB":
                        s = np.arange(len(w.times))
                    elif method == "R":
                        seed = gt.plant_seed(w.plant_id, k, repeat)
                        row["seed"] = seed
                        s = gt.random_schedule(w.times, k, seed)
                    elif method == "oracle":
                        s, count = gt_oracle(w, k, candidates)
                        row["oracle_candidate_evaluations"] = count
                    else:
                        s = select_schedule(w, k, method, read_rgb)
                    require(len(s) == (len(w.times) if k == "full_N" else k), "not_exact_charged_K")
                    row.update(scheduling_wall_s=time.perf_counter()-start)
                    row.update(joint_errors(w, a, n, s))
                    row.update(status="success", failure_reason="", selected_indices=s.tolist(),
                        selected_image_ids=[w.image_ids[i] for i in s], selected_elapsed_h=w.times[s].tolist(),
                        observed_RGB_indices=observed, actual_K=len(s), warmup_K=3 if method == "online" else 0)
                    if method not in ("oracle", "fullRGB"):
                        candidates.append(s)
                except gt.ScheduleFailure as exc:
                    row["failure_reason"] = str(exc)
                    row["selected_indices"] = list(exc.selected)
                    row["actual_K"] = len(exc.selected)
                row.setdefault("scheduling_wall_s", time.perf_counter()-start)
                row["replay_wall_s"] = time.perf_counter()-start
                records.append(row)
    return records


def collapse(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["plant_id"], r["K"], r["method"])].append(r)
    result = []
    for batch in groups.values():
        row = dict(batch[0])
        row["status"] = "success" if all(r["status"] == "success" for r in batch) else "failed"
        row["repeats"] = len(batch)
        for metric in METRICS:
            v = [r.get(metric, math.nan) for r in batch]
            row[metric] = float(np.mean(v)) if np.all(np.isfinite(v)) else math.nan
        row["random_plant_q90_E_A_pp"] = gt.empirical_q90([r.get("E_A_pp", math.nan) for r in batch]) if row["method"] == "R" else math.nan
        result.append(row)
    return result


def summarize(records):
    collapsed = collapse(records)
    lookup = {(r["plant_id"], r["K"], r["method"]): r for r in collapsed}
    common = {k: set() for k in gt.BUDGETS}
    for k in gt.BUDGETS:
        plants = {r["plant_id"] for r in collapsed if r["K"] == k}
        for p in plants:
            if all((p, k, m) in lookup and lookup[(p, k, m)]["status"] == "success" for m in METHODS):
                common[k].add(p)
    intersection = set.intersection(*common.values())
    summary, bootstrap = [], []
    for scope in ("per_K_common", "across_K_common"):
        for k in gt.BUDGETS:
            for method in (*METHODS, "oracle"):
                attempted = [r for r in collapsed if r["K"] == k and r["method"] == method]
                if not attempted:
                    continue
                pool = common[k] if scope == "per_K_common" else intersection
                # Oracle reported separately, never used to construct common sets/gates.
                sub = [r for r in attempted if r["plant_id"] in pool and r["status"] == "success"]
                area = [r for r in sub if r["area_eligible"]]
                all_area = [r for r in attempted if r["area_eligible"]]
                row = {"scope": scope, "K": k, "method": method, "eligible_N": len(attempted),
                    "area_eligible_N": len(all_area), "common_leaf_N": len(sub), "common_area_N": len(area),
                    "failed_N": sum(r["status"] != "success" for r in attempted),
                    "failure_rate": sum(r["status"] != "success" for r in attempted) / len(attempted),
                    "oracle_diagnostic_only": method == "oracle", "provisional": True,
                    "joint_pass_rate_all_area_eligible": sum(r.get("joint_pass", 0) if r["status"] == "success" else 0
                        for r in all_area)/len(all_area) if all_area else math.nan,
                    "random_mean_plant_q90_E_A_pp": float(np.mean([r["random_plant_q90_E_A_pp"] for r in area])) if area and method == "R" else math.nan}
                for metric in METRICS:
                    batch = sub if metric.startswith("leaf_") or metric in ("E_N_leaves", "leaf_pass") else area
                    row["mean_" + metric] = float(np.mean([r[metric] for r in batch])) if batch else math.nan
                summary.append(row)
                for baseline in ("U_0", "fullRGB"):
                    for metric, batch in (("E_A_pp", area), ("E_N_leaves", sub)):
                        differences = [r[metric] - lookup[(r["plant_id"], "full_N" if baseline == "fullRGB" else k, baseline)][metric]
                                       for r in batch]
                        estimate, lo, hi = gt.bootstrap_mean(differences,
                            gt.plant_seed(f"RGB|{scope}|{method}|{baseline}|{metric}", k), 2000)
                        bootstrap.append({"type": "paired_method_minus_baseline", "scope": scope,
                            "K": k, "method": method, "baseline": baseline, "metric": metric,
                            "whole_plant_N": len(differences), "mean_difference": estimate,
                            "CI95_low": lo, "CI95_high": hi, "bootstrap_repeats": 2000,
                            "equal_budget": baseline == "U_0", "oracle_diagnostic_only": method == "oracle"})
    full = [r for r in collapsed if r["method"] == "fullRGB"]
    if full:
        success = [r for r in full if r["status"] == "success"]
        area = [r for r in success if r["area_eligible"]]
        row = {"scope": "full_reference", "K": "full_N", "method": "fullRGB", "eligible_N": len(full),
               "common_leaf_N": len(success), "common_area_N": len(area),
               "failed_N": len(full)-len(success), "failure_rate": 1-len(success)/len(full), "provisional": True}
        for metric in METRICS:
            batch = success if metric.startswith("leaf_") or metric in ("E_N_leaves", "leaf_pass") else area
            row["mean_"+metric] = float(np.mean([r[metric] for r in batch])) if batch else math.nan
        summary.append(row)
    for k in gt.BUDGETS:
        plants = sorted(p for p in common[k] if lookup[(p, k, "U_0")]["area_eligible"])
        amplitudes = [max(lookup[(p, k, m)]["E_A_pp"] for m in METHODS[:3]) -
                      min(lookup[(p, k, m)]["E_A_pp"] for m in METHODS[:3]) for p in plants]
        estimate, lo, hi = gt.bootstrap_mean(amplitudes, gt.plant_seed("RGB_phase_amplitude", k), 2000)
        bootstrap.append({"type": "fixed_phase_max_minus_min", "K": k, "whole_plant_N": len(plants),
                          "mean_difference": estimate, "CI95_low": lo, "CI95_high": hi, "bootstrap_repeats": 2000})
    for low, high in ((5, 9), (9, 15), (5, 15)):
        plants = sorted(p for p in common[low] & common[high] if lookup[(p, low, "U_0")]["area_eligible"])
        for m in (METHODS[0], METHODS[2]):
            values = [(lookup[(p, low, m)]["E_A_pp"] - lookup[(p, low, "U_0")]["E_A_pp"])
                      - (lookup[(p, high, m)]["E_A_pp"] - lookup[(p, high, "U_0")]["E_A_pp"]) for p in plants]
            estimate, lo, hi = gt.bootstrap_mean(values, gt.plant_seed(f"RGB_DID|{m}|{high}", low), 2000)
            bootstrap.append({"type": "fixed_phase_difference_of_differences", "K_low": low, "K_high": high,
                "method": m, "whole_plant_N": len(plants), "mean_difference": estimate,
                "CI95_low": lo, "CI95_high": hi, "bootstrap_repeats": 2000})
    info = {"common_by_K": {str(k): sorted(v) for k, v in common.items()},
            "across_K_common": sorted(intersection), "common_sets_condition_on_error": False,
            "random_aggregation": "20 draws averaged within plant BEFORE whole-plant bootstrap; abs errors averaged separately from signed errors",
            "primary_exploratory_baselines": ["fullRGB", "U_0"], "no_best_phase_selection": True,
            "formal_gate": "NOT_RUN_EXPLORATORY_ONLY", "oracle_never_in_gate": True}
    return summary, bootstrap, info


def make_plot(out, summary):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    u0 = sorted([r for r in summary if r["scope"] == "across_K_common" and r["method"] == "U_0"], key=lambda r: r["K"])
    for ax, f, scale, tolerance, title in ((axes[0, 0], "area_mean", 100, 1, "Area time-mean absolute error (pp)"),
                                         (axes[0, 1], "leaf_mean", 1, .5, "Visible-leaf time-mean absolute error")):
        for term, label in (("e_M", "fullRGB measurement vs fullGT"), ("e_S", "pure GT sampling, same S"),
                            ("e_total", "joint RGB sampling vs fullGT (U0)")):
            ax.plot([r["K"] for r in u0], [scale*r[f"mean_{f}_{term}_abs"] for r in u0], "o-", label=label)
        ax.axhline(tolerance, linestyle="--", color="gray", label="design tolerance (not calibrated)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    for method in METHODS[:3]:
        sub = sorted([r for r in summary if r["scope"] == "across_K_common" and r["method"] == method], key=lambda r: r["K"])
        axes[1, 0].plot([r["K"] for r in sub], [r["mean_E_A_pp"] for r in sub], "o-", label=method)
    axes[1, 0].set_title("Fixed phases, equal budget; no best-phase envelope")
    axes[1, 0].legend(fontsize=8)
    for term in TERMS:
        axes[1, 1].plot([r["K"] for r in u0], [100*r[f"mean_area_mean_{term}"] for r in u0], "o-", label=term)
    axes[1, 1].axhline(0, color="gray", linewidth=.5)
    axes[1, 1].set_title("Signed decomposition (pp), U0; absolute errors NOT additive")
    axes[1, 1].legend(fontsize=8)
    n_area = u0[0]["common_area_N"] if u0 else 0
    n_leaf = u0[0]["common_leaf_N"] if u0 else 0
    for ax in axes.flat:
        ax.set_xlabel("Charged K; same-plant across-K intersection")
        ax.set_xticks(gt.BUDGETS)
    fig.suptitle(f"Exploratory frozen B1 reference; provisional area n={n_area}, leaves n={n_leaf}\nFull GT discrete reference, not biomass; increasing K need not fix model bias")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"measurement_sampling_interaction.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze(inventory_path, cache_dir, gt_pilot, out):
    check_sources()
    require(not out.exists() or not any(out.iterdir()), "analysis_output_not_empty")
    inventory_hash = sha(inventory_path)
    rows, counts = eligible_inventory(gt.read_inventory(inventory_path))
    config_gt = read_json(gt_pilot / "config.json")
    require(config_gt["inventory_sha256"] == inventory_hash
            and config_gt["source_sha256"] == SOURCE_PINS["aea_sampling.py"], "GT_pilot_input_or_source_changed")
    require(sha(gt_pilot / "source_code_snapshot.py") == SOURCE_PINS["aea_sampling.py"], "GT_source_snapshot_changed")
    gt_status = read_json(gt_pilot / "status.json")
    require(gt_status["status"] in ("completed_provisional_GT_development_pilot", "leaf_only_area_blocked"),
            "GT_pilot_not_completed")
    cache, reference_config, reference_hash = load_cache(cache_dir, rows, inventory_hash)
    audits = read_csv(gt_pilot / "window_audit.csv")
    windows, excluded = windows_from_manifest(audits, rows)
    # Original full-GT quantities must roundtrip to the already saved GT pilot.
    old_full = {r["plant_id"]: r for r in read_csv(gt_pilot / "replay_per_plant.csv")
                if r["split_role"] in ROLES and r["method"] == "full"}
    for w in windows:
        require(w.plant_id in old_full, "original_GT_full_reference_missing")
        full = functional_values(w, w.area, w.leaves, np.arange(len(w.times)))
        require(abs(full["leaf_mean"] - float(old_full[w.plant_id]["reference_mean_leaves"])) <= 1e-10,
                "original_GT_leaf_reference_changed")
        if w.area_eligible:
            require(abs(full["area_mean"] - float(old_full[w.plant_id]["reference_mean_area_fraction"])) <= 1e-10,
                    "original_GT_area_reference_changed")
    records = replay_joint(windows, cache)
    summary, bootstrap, info = summarize(records)
    out.mkdir(parents=True, exist_ok=True)
    gt.write_csv(out / "per_plant_joint_errors.csv", records, ("plant_id", "split_role", "K", "method", "repeat", "status", "failure_reason"))
    gt.write_csv(out / "summary.csv", summary, ("scope", "K", "method", "eligible_N", "failure_rate"))
    gt.write_csv(out / "paired_bootstrap.csv", bootstrap, ("type", "scope", "K", "method", "whole_plant_N", "bootstrap_repeats"))
    gt.write_csv(out / "window_failures.csv", excluded, ("plant_id", "split_role", "status", "failure_reason"))
    make_plot(out, summary)
    full = [r for r in records if r["method"] == "fullRGB" and r["status"] == "success"]
    full_area = [r for r in full if r["area_eligible"]]
    model_bias_fail = sum(r["E_A_pp"] > 1 or r["E_N_leaves"] > .5 for r in full_area)
    phase = {"version": VERSION, "phase": "complete_exploratory_CPU_analysis", "notice": NOTE,
        "inventory_sha256": inventory_hash, "reference_config_sha256": reference_hash,
        "GT_pilot": str(gt_pilot), "GT_artifact_hashes": {n: sha(gt_pilot / n) for n in
            ("config.json", "status.json", "window_audit.csv", "replay_per_plant.csv", "source_code_snapshot.py")},
        "reference_provenance": reference_config["provenance"], "role_metadata_counts": counts,
        "window_N": len(windows), "area_eligible_window_N": sum(w.area_eligible for w in windows),
        "original_window_failed_N": len(excluded), "replay_units": len(records),
        "failure_counts": dict(Counter(r["failure_reason"] for r in records if r["status"] != "success")),
        "fullRGB_joint_tolerance_failed_N": model_bias_fail, "fullRGB_area_eligible_N": len(full_area),
        "statistics": info, "no_risk_model_or_calibrator": True, "no_C_POLICY_or_locked_test_evaluation": True,
        "formal_measurement_and_external_validation": "pending", "area_scale": gt.AREA_NOTE}
    atomic_json(out / "status.json", phase)
    atomic_json(out / "trace.json", {"window_ids": [w.plant_id for w in windows],
        "time_grids_sha256": digest({w.plant_id: {"image_ids": w.image_ids, "times": w.times} for w in windows}),
        "analysis_source_sha256": sha(Path(__file__)), "sampling_source_sha256": SOURCE_PINS["aea_sampling.py"],
        "selection_values": "online callback: observed RGB only; GT oracle separate hindsight diagnostic"})
    lines = [NOTE, "Only V_SOURCE/V_SHIFT; original audited windows and fullGT definitions preserved.",
        f"Windows: {len(windows)}; area eligible (provisional): {len(full_area)}; original excluded: {len(excluded)}.",
        f"fullRGB exceeds joint design tolerance in {model_bias_fail}/{len(full_area)} area-eligible plants.",
        "Increasing K removes sampling loss at full N but does not change e_M; it need not repair model bias.",
        "e_total=e_M+e_S+e_X verified per functional/record. Absolute errors never added; cancellation is not accuracy.",
        "U0/fullRGB are exploratory baselines. Fixed phases compared at equal K, never selected by observed error.",
        "Random20 aggregated within plant, paired whole-plant bootstrap2000; failures/eligible N retained.",
        "No C_POLICY calibration, confirmatory gate, risk fit, biological ranking, locked-test or external effects evaluated."]
    (out / "phase_result.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"phase": phase["phase"], "windows": len(windows), "reference_kind": REFERENCE_KIND}


def project_path(path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def main(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("extract", "analyze"), required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-run", type=Path, default=Path("mva/B1_pilot_v1"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--resume", action="store_true", help="skip only verified cache/input hashes; interrupted noncommitted tail is refused")
    p.add_argument("--measurement-cache", type=Path)
    p.add_argument("--gt-pilot", type=Path, default=Path("aea/pilot_v2"))
    # Accept the original compact --devicecuda spelling without changing defaults.
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = ["--device=cuda" if x == "--devicecuda" else x for x in argv]
    args = p.parse_args(argv)
    try:
        if args.action == "extract":
            result = extract(project_path(args.inventory), project_path(args.model_run), project_path(args.out),
                args.device, project_path(args.data_root) if args.data_root else None, args.resume)
        else:
            require(args.measurement_cache is not None and not args.resume, "analyze_requires_cache_and_fresh_output")
            result = analyze(project_path(args.inventory), project_path(args.measurement_cache),
                             project_path(args.gt_pilot), project_path(args.out))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"AEA RGB blocked: {type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    print(json.dumps(gt.json_safe(result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

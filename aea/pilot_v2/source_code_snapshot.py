"""AEA development-only, GT sampling pilot. No RGB model or locked-role evaluation.

Run from the experiment directory:
    python src/aea_sampling.py --inventory shared/data_inventory.csv --out aea/pilot_v1
Relative image paths are resolved against --data-root (default: inventory.parent.parent).
Inventory mask_status=valid attests to audited GT instance counts; RGB is decode-checked.
Naive timestamps receive a computational zero-offset anchor for elapsed-time arithmetic;
the acquisition timezone remains unknown and is not claimed to be UTC.
Area is conditional fixed-pixel-field fraction, never physically calibrated area.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from importlib import metadata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROLES = ("TRAIN", "V_SOURCE", "V_SHIFT")
BUDGETS = (5, 9, 15)
PHASES = (-1 / 3, 0.0, 1 / 3)
SEED = 43
VERSION = "aea_gt_pilot_v1"
SCHEMA = (
    "image_id", "biological_plant_id", "genotype", "replicate_id", "timestamp",
    "rgb_path", "mask_path", "width", "height", "rgb_sha256", "mask_sha256",
    "mask_status", "unknown_pixels", "n_instances", "union_pixels", "area_fraction",
    "label_path", "split_role",
)
AREA_NOTE = "provisional: fixed pixel field area fraction conditional on unverified physical ROI scale"
METHOD_ORDER = ("U_-1/3", "U_0", "U_+1/3", "C", "J", "R", "online", "oracle", "full")


class ScheduleFailure(ValueError):
    """Keep same-K failures and any already consumed online measurements."""

    def __init__(self, reason: str, selected: Sequence[int] = ()):
        super().__init__(reason)
        self.selected = tuple(selected)


class L2QuantileModel:
    """CPU-only weighted quantile regression; no pandas/sklearn import required.

    Check loss has a declared 1e-6 log-unit differentiable approximation. The
    frequency/phase coefficients have L2 penalty .1; the intercept is unpenalized.
    """

    def __init__(self, quantile: float = .9, penalty: float = .1):
        self.quantile, self.penalty = quantile, penalty
        self.smoothing = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        from scipy.optimize import minimize
        design = np.column_stack((np.ones(len(x)), x))
        weights = sample_weight / np.sum(sample_weight)
        def objective(beta):
            residual = y - design @ beta
            norm = np.sqrt(residual * residual + self.smoothing ** 2)
            loss = (self.quantile - .5) * residual + .5 * norm
            value = float(weights @ loss + .5 * self.penalty * (beta[1:] @ beta[1:]))
            gradient = -design.T @ (weights * (self.quantile - .5 + .5 * residual / norm))
            gradient[1:] += self.penalty * beta[1:]
            return value, gradient
        initial = np.r_[empirical_q90(y), np.zeros(x.shape[1])]
        solution = minimize(objective, initial, jac=True, method="L-BFGS-B",
                            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 60})
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise RuntimeError(f"quantile_solver_failed:{solution.message}")
        self.intercept_, self.coef_ = float(solution.x[0]), solution.x[1:]
        self.solver_info = {"method": "L-BFGS-B", "success": True, "iterations": int(solution.nit),
                            "objective": float(solution.fun), "logunit_smoothing": self.smoothing}
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept_ + x @ self.coef_


@dataclass
class Window:
    plant_id: str
    biological_plant_id: str
    genotype: str
    replicate_id: str
    role: str
    start: datetime
    times: np.ndarray
    area: np.ndarray
    leaves: np.ndarray
    image_ids: list[str]
    area_eligible: bool
    width: int
    height: int
    encoded_rgb_bytes: int = 0

    @property
    def length(self) -> float:
        return float(self.times[-1] - self.times[0])

    @property
    def tolerance(self) -> float:
        return min(float(np.median(np.diff(self.times))), 12.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_versions() -> dict:
    versions = {"python": sys.version, "numpy": np.__version__}
    for package in ("scipy", "Pillow", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plant_seed(plant_id: str, budget: int, repeat: int = 0) -> int:
    payload = f"{VERSION}|{SEED}|{plant_id}|{budget}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def parse_timestamp(text: str) -> tuple[datetime, bool]:
    text = text.strip()
    if not text:
        raise ValueError("missing_timestamp")
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        value = None
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d_%H%M%S", "%Y%m%d_%H%M",
                    "%Y-%m-%d_%H-%M-%S", "%Y_%m_%d_%H_%M_%S",
                    "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                value = datetime.strptime(text, fmt)
                break
            except ValueError:
                pass
        if value is None:
            raise ValueError("unparseable_calendar_timestamp")
    naive = value.tzinfo is None
    return (value.replace(tzinfo=timezone.utc) if naive else value), naive


def _integer(row: dict, name: str) -> int:
    value = float(row[name])
    if not math.isfinite(value) or value != int(value) or value < 0:
        raise ValueError(f"invalid_{name}")
    return int(value)


def _hash_valid(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def check_measurement(row: dict, data_root: Path) -> tuple[dict | None, list[str], bool]:
    """GT metadata comes from the audit, not new segmentation or inferred labels."""
    errors: list[str] = []
    gt = None
    try:
        if row["mask_status"] != "valid":
            raise ValueError("mask_not_valid")
        if _integer(row, "unknown_pixels"):
            raise ValueError("unknown_mask_pixels")
        w, h = _integer(row, "width"), _integer(row, "height")
        if w == 0 or h == 0:
            raise ValueError("zero_image_dimension")
        leaves = _integer(row, "n_instances")
        pixels = _integer(row, "union_pixels")
        fraction = float(row["area_fraction"])
        if (not math.isfinite(fraction) or not 0 <= fraction <= 1 or pixels > w * h
                or abs(fraction - pixels / (w * h)) > 5e-7):
            raise ValueError("inconsistent_area_fraction")
        if (pixels == 0) != (leaves == 0):
            raise ValueError("inconsistent_instance_union")
        if not row["mask_path"] or not _hash_valid(row["mask_sha256"]):
            raise ValueError("missing_mask_path_or_hash")
        mask_path = Path(row["mask_path"])
        mask_path = mask_path if mask_path.is_absolute() else data_root / mask_path
        if not mask_path.is_file():
            raise ValueError("mask_file_missing")
        gt = {"width": w, "height": h, "area": pixels / (w * h), "leaves": leaves}
    except (ValueError, KeyError, OSError) as exc:
        errors.append(str(exc))
    rgb_ok = False
    if gt is not None:
        try:
            if not row["rgb_path"] or not _hash_valid(row["rgb_sha256"]):
                raise ValueError("missing_rgb_path_or_hash")
            rgb_path = Path(row["rgb_path"])
            rgb_path = rgb_path if rgb_path.is_absolute() else data_root / rgb_path
            if not rgb_path.is_file():
                raise ValueError("rgb_file_missing")
            from PIL import Image
            with Image.open(rgb_path) as image:
                if image.size != (gt["width"], gt["height"]):
                    raise ValueError("rgb_dimension_mismatch")
                image.load()
            gt["rgb_bytes"] = rgb_path.stat().st_size
            rgb_ok = True
        except (ValueError, OSError) as exc:
            errors.append(f"rgb_invalid:{exc}")
    return gt, errors, rgb_ok


def audit_windows(rows: list[dict], data_root: Path) -> tuple[list[dict], list[Window], dict]:
    """Inspect every development plant, never phenotype fields of excluded roles."""
    groups: dict[str, list[dict]] = defaultdict(list)
    identity_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    role_rows = Counter()
    role_plants: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        role = row["split_role"].strip()
        key = f'{row["genotype"].strip()}::{row["replicate_id"].strip()}'
        role_rows[role] += 1
        role_plants[role].add(key)
        # Identity/hash metadata only: locked-role GT never enters replay or selection.
        for kind, identity in (("plant", key), ("biological_id", row["biological_plant_id"]),
                               ("rgb_hash", row["rgb_sha256"].lower())):
            if identity:
                identity_roles[(kind, identity)].add(role)
        if role in ROLES:
            groups[key].append(row)
    audits, windows = [], []
    naive_count = duplicate_count = conflict_count = 0
    for key in sorted(groups):
        plant_rows = groups[key]
        roles = sorted({r["split_role"].strip() for r in plant_rows})
        biological_ids = sorted({r["biological_plant_id"].strip() for r in plant_rows})
        first = plant_rows[0]
        reasons: list[str] = []
        if not first["genotype"].strip() or not first["replicate_id"].strip():
            reasons.append("missing_genotype_or_replicate")
        if len(biological_ids) != 1 or not biological_ids[0]:
            reasons.append("inconsistent_or_missing_biological_id")
        if len(roles) != 1:
            reasons.append("cross_role_plant")
        for r in plant_rows:
            for kind, identity in (("plant", key), ("biological_id", r["biological_plant_id"]),
                                   ("rgb_hash", r["rgb_sha256"].lower())):
                if identity and len(identity_roles[(kind, identity)]) > 1:
                    reasons.append(f"cross_role_{kind}")
        observations, row_audit = [], []
        for r in plant_rows:
            errors = []
            stamp = None
            naive = False
            try:
                stamp, naive = parse_timestamp(r["timestamp"])
                naive_count += int(naive)
            except ValueError as exc:
                errors.append(str(exc))
            gt, measurement_errors, rgb_ok = check_measurement(r, data_root)
            errors += measurement_errors
            item = {"image_id": r["image_id"], "timestamp_raw": r["timestamp"],
                    "datetime": stamp.isoformat() if stamp else None,
                    "naive_assumed_UTC": naive, "gt_valid": gt is not None,
                    "rgb_gt_valid": gt is not None and rgb_ok,
                    "errors": errors, "duplicate": False, "elapsed_h": None}
            row_audit.append(item)
            if stamp is not None:
                observations.append({"stamp": stamp, "row": r, "gt": gt, "rgb_ok": rgb_ok,
                                     "audit": item})
        same_time: dict[datetime, list[dict]] = defaultdict(list)
        for item in observations:
            same_time[item["stamp"]].append(item)
        unique, duplicates, conflicts = [], 0, 0
        for stamp in sorted(same_time):
            batch = same_time[stamp]
            duplicates += len(batch) - 1
            valid = [x for x in batch if x["gt"] is not None and x["rgb_ok"]]
            gt_valid = [x for x in batch if x["gt"] is not None]
            signatures = {(x["row"]["rgb_sha256"], x["row"]["mask_sha256"],
                           x["gt"]["width"], x["gt"]["height"],
                           x["gt"]["area"], x["gt"]["leaves"]) for x in valid}
            gt_signatures = {(x["row"]["mask_sha256"], x["gt"]["width"],
                              x["gt"]["height"], x["gt"]["area"], x["gt"]["leaves"])
                             for x in gt_valid}
            if len(signatures) > 1 or len(gt_signatures) > 1:
                reasons.append("conflicting_same_timestamp")
                conflicts += 1
            chosen = min(batch, key=lambda x: (not (x["gt"] is not None and x["rgb_ok"]),
                         x["gt"] is None, x["row"]["rgb_sha256"],
                         x["row"]["mask_sha256"], x["row"]["image_id"]))
            unique.append(chosen)
            for item in batch:
                item["audit"]["duplicate"] = item is not chosen
                if len(signatures) > 1 or len(gt_signatures) > 1:
                    item["audit"]["errors"].append("timestamp_conflict")
        duplicate_count += duplicates
        conflict_count += conflicts
        joint = [x for x in unique if x["gt"] is not None and x["rgb_ok"]]
        gt_only = [x for x in unique if x["gt"] is not None]
        audit = {"plant_id": key, "biological_plant_id": "|".join(biological_ids),
                 "genotype": first["genotype"], "replicate_id": first["replicate_id"],
                 "split_role": "|".join(roles), "input_rows": len(plant_rows),
                 "duplicate_timestamps": duplicates, "timestamp_conflicts": conflicts,
                 "gt_valid_total": len(gt_only), "rgb_gt_valid_total": len(joint),
                 "a": "", "b": "", "L_h": math.nan, "N": 0,
                 "gt_only_N_window": 0, "gt_only_L_h": math.nan,
                 "gt_standalone_a": "", "gt_standalone_b": "", "gt_standalone_N": 0,
                 "gt_standalone_L_h": math.nan,
                 "dimensions_all_joint_frames": sorted({(x["gt"]["width"], x["gt"]["height"]) for x in joint}),
                 "dimensions_all_GT_frames": sorted({(x["gt"]["width"], x["gt"]["height"]) for x in gt_only}),
                 "median_dt_h": math.nan, "max_dt_h": math.nan,
                 "window_eligible": False, "area_eligible": False,
                 "roi_status": AREA_NOTE, "planned_completeness": "unknown_no_schedule",
                 "actual_capture_90pct": "not_claimed", "row_times_and_errors": row_audit}
        if gt_only:
            ga = gt_only[0]["stamp"]
            gt_standalone = [x for x in gt_only if (x["stamp"] - ga).total_seconds() <= 168 * 3600]
            gb = gt_standalone[-1]["stamp"]
            audit.update(gt_standalone_a=ga.isoformat(), gt_standalone_b=gb.isoformat(),
                         gt_standalone_N=len(gt_standalone), gt_standalone_L_h=(gb - ga).total_seconds() / 3600)
        if not joint:
            reasons.append("no_valid_rgb_gt_frames")
        else:
            a = joint[0]["stamp"]
            for item in observations:
                item["audit"]["elapsed_h"] = (item["stamp"] - a).total_seconds() / 3600
            chosen_window = [x for x in joint if (x["stamp"] - a).total_seconds() <= 168 * 3600]
            times = np.array([(x["stamp"] - a).total_seconds() / 3600 for x in chosen_window])
            gt_window = [x for x in gt_only if 0 <= (x["stamp"] - a).total_seconds() <= 168 * 3600]
            dt = np.diff(times)
            length = float(times[-1])
            median = float(np.median(dt)) if len(dt) else math.nan
            maximum = float(max(dt)) if len(dt) else math.nan
            audit.update(a=a.isoformat(), b=chosen_window[-1]["stamp"].isoformat(),
                         L_h=length, N=len(times), median_dt_h=median, max_dt_h=maximum,
                         gt_only_N_window=len(gt_window),
                         gt_only_L_h=((gt_window[-1]["stamp"] - gt_window[0]["stamp"]).total_seconds()
                                      / 3600 if gt_window else math.nan))
            if not 156 <= length <= 168:
                reasons.append("L_outside_156_168h")
            if len(times) < 15:
                reasons.append("N_below_15")
            if not math.isfinite(median) or median > 12:
                reasons.append("median_dt_above_12h")
            if not math.isfinite(maximum) or maximum > 18:
                reasons.append("max_dt_above_18h")
            stable = len({(x["gt"]["width"], x["gt"]["height"]) for x in gt_only}) == 1
            audit["window_eligible"] = not reasons
            audit["area_eligible"] = not reasons and stable
            if not stable:
                audit["area_exclusion_reason"] = "inconsistent_plant_dimensions"
            else:
                audit["area_exclusion_reason"] = "" if not reasons else "window_or_identity_failure"
            if not reasons:
                data = [x["gt"] for x in chosen_window]
                windows.append(Window(key, biological_ids[0], first["genotype"], first["replicate_id"],
                               roles[0], a, times, np.array([x["area"] for x in data]),
                               np.array([x["leaves"] for x in data], dtype=float),
                               [x["row"]["image_id"] for x in chosen_window], stable,
                               data[0]["width"], data[0]["height"],
                               sum(x["rgb_bytes"] for x in data)))
        audit["exclusion_reasons"] = sorted(set(reasons))
        audits.append(audit)
    counts = {"inventory_rows": len(rows), "role_rows": dict(role_rows),
              "role_plants": {r: len(p) for r, p in role_plants.items()},
              "development_plants": len(groups), "window_eligible_plants": len(windows),
              "area_eligible_plants": sum(w.area_eligible for w in windows),
              "window_roles": dict(Counter(w.role for w in windows)),
              "area_window_roles": dict(Counter(w.role for w in windows if w.area_eligible)),
              "duplicate_timestamps": duplicate_count, "timestamp_conflicts": conflict_count,
              "naive_timestamps_assumed_UTC": naive_count,
              "exclusion_counts": dict(Counter(reason for a in audits for reason in a["exclusion_reasons"])),
              "area_dimension_exclusions": sum(a.get("area_exclusion_reason") ==
                                                "inconsistent_plant_dimensions" for a in audits)}
    return audits, windows, counts


def _check_times(times: np.ndarray) -> None:
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ScheduleFailure("timestamps_must_be_finite_strictly_increasing_unique")


def project_targets(times: Sequence[float], targets: Sequence[float], tolerance: float) -> np.ndarray:
    """Exact monotone assignment DP with lexicographically earliest tie and traceback."""
    t, q = np.asarray(times, dtype=float), np.asarray(targets, dtype=float)
    _check_times(t)
    m, n = len(q), len(t) - 2
    if m > n:
        raise ScheduleFailure("K_exceeds_N")
    if not np.all(np.isfinite(q)) or np.any(np.diff(q) < 0):
        raise ScheduleFailure("invalid_target_order")
    if tolerance < 0 or math.isnan(tolerance):
        raise ScheduleFailure("invalid_projection_tolerance")
    cost = np.full((m + 1, n + 1), np.inf)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    paths: list[list[tuple | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    cost[0, :] = 0
    paths[0] = [()] * (n + 1)
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            cost[j, i], paths[j][i] = cost[j, i - 1], paths[j][i - 1]
            distance = abs(t[i] - q[j - 1])
            if distance <= tolerance + 1e-10 and paths[j - 1][i - 1] is not None:
                candidate_cost = cost[j - 1, i - 1] + distance
                candidate_path = paths[j - 1][i - 1] + (i,)
                if (candidate_cost < cost[j, i] - 1e-10 or
                        (abs(candidate_cost - cost[j, i]) <= 1e-10 and
                         (paths[j][i] is None or candidate_path < paths[j][i]))):
                    cost[j, i], paths[j][i], take[j, i] = candidate_cost, candidate_path, True
    if paths[m][n] is None:
        raise ScheduleFailure("no_unique_monotone_projection_within_tolerance")
    selected = []
    j, i = m, n
    while j:
        if take[j, i]:
            selected.append(i)
            j -= 1
        i -= 1
    result = np.array([0] + selected[::-1] + [len(t) - 1], dtype=int)
    assert len(result) == m + 2 and np.all(np.diff(result) > 0)
    return result


def uniform_schedule(w: Window, k: int, phase: float) -> np.ndarray:
    targets = w.times[0] + (np.arange(1, k - 1) + phase) * w.length / (k - 1)
    return project_targets(w.times, targets, w.tolerance)


def clock_schedule(w: Window, k: int) -> np.ndarray:
    delta = w.length / (k - 2)
    midnight = w.start.replace(hour=0, minute=0, second=0, microsecond=0)
    d = ((midnight - w.start).total_seconds() / 3600) % delta
    if abs(d) < 1e-10 or abs(d - delta) < 1e-10:
        # A deterministic one-second right shift (declared timestamp resolution).
        d = 1 / 3600
    targets = w.times[0] + d + np.arange(k - 2) * delta
    return project_targets(w.times, targets, w.tolerance)


def index_schedule(times: Sequence[float], k: int) -> np.ndarray:
    n = len(times)
    _check_times(np.asarray(times, dtype=float))
    # Unique integer assignment, not rounded indices and not uniform elapsed time.
    return project_targets(np.arange(n, dtype=float), np.linspace(0, n - 1, k)[1:-1], math.inf)


def random_schedule(times: Sequence[float], k: int, seed: int) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    _check_times(t)
    if k > len(t) or k < 2:
        raise ScheduleFailure("K_exceeds_N_or_below_2")
    inside = np.sort(np.random.default_rng(seed).choice(np.arange(1, len(t) - 1), k - 2, replace=False))
    return np.concatenate(([0], inside, [len(t) - 1])).astype(int)


def online_schedule(times: Sequence[float], k: int, read_value: Callable[[int], Sequence[float]],
                    tolerance: float) -> np.ndarray:
    """Only callback-access selected GT; the preknown opportunity table is time-only.

    Callback returns (area_fraction, visible_leaves). All three warmup frames and
    the reserved endpoint count towards K. No full-curve features are computed.
    """
    t = np.asarray(times, dtype=float)
    _check_times(t)
    if k < 5 or k > len(t):
        raise ScheduleFailure("online_requires_5_le_K_le_N")
    warmup = project_targets(t, [t[0] + 6, t[0] + 12], tolerance)[:-1]
    if len(t) - 2 - warmup[-1] < k - 4:
        raise ScheduleFailure("warmup_leaves_insufficient_opportunities")
    selected = list(map(int, warmup))
    values = [np.array(read_value(i), dtype=float) * np.array([100.0, 2.0]) for i in selected]
    if not np.all(np.isfinite(values)):
        raise ScheduleFailure("nonfinite_observed_online_measurement", selected)
    remaining = k - 4
    while remaining:
        observed_times = t[selected]
        slopes = np.diff(np.array(values), axis=0) / np.diff(observed_times)[:, None]
        v = float(np.max(np.abs(np.median(slopes[-3:], axis=0))))
        mids = (observed_times[1:] + observed_times[:-1]) / 2
        c = (float(np.max(np.abs((slopes[-1] - slopes[-2]) / (mids[-1] - mids[-2]))))
             if len(slopes) >= 2 else 0.0)
        v = 0.0 if v < .02 else v
        c = 0.0 if c < .002 else c
        h0 = (t[-1] - t[selected[-1]]) / (remaining + 1)
        h = float(np.clip(h0 / (1 + v * h0 / .5 + c * h0 * h0 / 1.0), .5 * h0, 1.5 * h0))
        target = t[selected[-1]] + h
        latest = len(t) - remaining - 1
        index = max(selected[-1] + 1, min(int(np.searchsorted(t, target, side="left")), latest))
        # Budget preservation may force an earlier opportunity than the requested
        # time. Unlike offline projection, the plan does not reject this choice
        # on distance-to-target; it remains a real, charged observation.
        if index > len(t) - 2:
            raise ScheduleFailure("online_no_remaining_internal_opportunity", selected)
        selected.append(index)
        values.append(np.array(read_value(index), dtype=float) * np.array([100.0, 2.0]))
        if not np.all(np.isfinite(values[-1])):
            raise ScheduleFailure("nonfinite_observed_online_measurement", selected)
        remaining -= 1
    result = np.array(selected + [len(t) - 1])
    assert len(result) == k and np.all(np.diff(result) > 0)
    return result


def trapezoid(times: Sequence[float], values: Sequence[float]) -> float:
    t, y = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    _check_times(t)
    if y.shape != t.shape or not np.all(np.isfinite(y)):
        raise ValueError("invalid_integrand")
    return float(np.sum(np.diff(t) * (y[:-1] + y[1:]) / 2))


def evaluate(w: Window, indices: Sequence[int]) -> dict:
    s = np.asarray(indices, dtype=int)
    if len(s) < 2 or s[0] != 0 or s[-1] != len(w.times) - 1 or np.any(np.diff(s) <= 0):
        raise ValueError("evaluation_requires_unique_indices_and_both_endpoints")
    ref_a = trapezoid(w.times, w.area)
    ref_n = trapezoid(w.times, w.leaves)
    integral = trapezoid(w.times[s], w.area[s])
    leaf_integral = trapezoid(w.times[s], w.leaves[s])
    signed_area = (integral - ref_a) / w.length
    leaf_error = abs(leaf_integral - ref_n) / w.length
    ea = 100 * abs(signed_area) if w.area_eligible else math.nan
    z = max(ea, leaf_error / .5) if w.area_eligible else math.nan
    return {"E_A_pp": ea, "E_N_leaves": leaf_error,
            "signed_sampling_area_pp": 100 * signed_area if w.area_eligible else math.nan,
            "signed_sampling_leaf": (leaf_integral - ref_n) / w.length,
            "abs_area_integral_fraction_h": abs(integral - ref_a) if w.area_eligible else math.nan,
            "relative_area_integral_pct": (100 * abs(integral - ref_a) / ref_a
                   if w.area_eligible and ref_a / w.length >= .001 else math.nan),
            "reference_mean_area_fraction": ref_a / w.length if w.area_eligible else math.nan,
            "reference_mean_leaves": ref_n / w.length, "joint_z": z,
            "joint_pass": float(z <= 1 + 1e-12) if w.area_eligible else math.nan,
            "leaf_pass": float(leaf_error <= .5 + 1e-12)}


def oracle_schedule(w: Window, k: int, candidates: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Diagnostic candidate-best + forward greedy; NOT global optimum/deployable."""
    if k > len(w.times) or k < 2:
        raise ScheduleFailure("oracle_K_exceeds_N_or_below_2")
    def loss(s: Sequence[int]) -> float:
        e = evaluate(w, s)
        return e["joint_z"] if w.area_eligible else e["E_N_leaves"] / .5
    selected, evaluations = [0, len(w.times) - 1], 0
    while len(selected) < k:
        choices = []
        for i in range(1, len(w.times) - 1):
            if i not in selected:
                trial = sorted(selected + [i])
                choices.append((loss(trial), i))
                evaluations += 1
        selected = sorted(selected + [min(choices)[1]])
    options = candidates + [np.array(selected)]
    valid = [s for s in options if len(s) == k and len(set(s)) == k]
    result = min(valid, key=lambda s: (loss(s), tuple(s)))
    return np.asarray(result, dtype=int), evaluations + len(valid)


def replay(windows: list[Window]) -> tuple[list[dict], dict]:
    records, seeds = [], {}
    for w in windows:
        if w.role not in ROLES:
            raise ValueError(f"forbidden_replay_role:{w.role}")
        seeds[w.plant_id] = {}
        for k in BUDGETS:
            candidates = []
            for method, phase, repeat in (
                [("U_-1/3", PHASES[0], 0), ("U_0", 0.0, 0), ("U_+1/3", PHASES[2], 0),
                 ("C", None, 0), ("J", None, 0)] + [("R", None, r) for r in range(20)] +
                [("online", None, 0), ("oracle", None, 0)]
            ):
                seed = plant_seed(w.plant_id, k, repeat) if method == "R" else SEED
                if method == "R":
                    seeds[w.plant_id][f"K{k}_R{repeat}"] = seed
                start = time.perf_counter()
                record = {"plant_id": w.plant_id, "biological_plant_id": w.biological_plant_id,
                          "genotype": w.genotype, "replicate_id": w.replicate_id, "split_role": w.role,
                          "layer": "GT", "K": k, "N": len(w.times), "L_h": w.length,
                          "rate_per_h": (k - 1) / w.length, "method": method, "phase": phase,
                          "repeat": repeat, "seed": seed, "area_eligible": w.area_eligible,
                          "roi_status": AREA_NOTE, "status": "failed", "failure_reason": "",
                          "selected_indices": [], "selected_elapsed_h": [], "selected_image_ids": [],
                          "actual_K": 0, "oracle_candidate_evaluations": 0,
                          "deployability": ("NOT_DEPLOYABLE_NOT_GLOBAL_OPTIMAL" if method == "oracle"
                              else "IDEAL_GT_SENSOR_PREKNOWN_TIMESTAMPS" if method == "online"
                              else "GT_REPLAY_PREKNOWN_TIMESTAMPS_NOT_RGB_VALIDATION")}
                try:
                    if k > len(w.times):
                        raise ScheduleFailure("K_exceeds_N")
                    if method.startswith("U_"):
                        selected = uniform_schedule(w, k, phase)
                    elif method == "C":
                        selected = clock_schedule(w, k)
                    elif method == "J":
                        selected = index_schedule(w.times, k)
                    elif method == "R":
                        selected = random_schedule(w.times, k, seed)
                    elif method == "online":
                        selected = online_schedule(w.times, k, lambda i: (w.area[i], w.leaves[i]), w.tolerance)
                    else:
                        selected, evaluations = oracle_schedule(w, k, candidates)
                        record["oracle_candidate_evaluations"] = evaluations
                    if len(selected) != k or len(set(selected)) != k:
                        raise ScheduleFailure("non_exact_K")
                    scheduling_time = time.perf_counter() - start
                    record.update(evaluate(w, selected))
                    record.update(status="success", selected_indices=selected.tolist(),
                                  selected_elapsed_h=w.times[selected].tolist(), actual_K=len(selected),
                                  selected_image_ids=[w.image_ids[i] for i in selected])
                    if method != "oracle":
                        candidates.append(selected)
                except ScheduleFailure as exc:
                    scheduling_time = time.perf_counter() - start
                    record["failure_reason"] = str(exc)
                    if exc.selected:
                        partial = list(exc.selected)
                        record.update(selected_indices=partial, selected_elapsed_h=w.times[partial].tolist(),
                                      selected_image_ids=[w.image_ids[i] for i in partial], actual_K=len(partial))
                record["scheduling_wall_s"] = scheduling_time
                record["replay_wall_s"] = time.perf_counter() - start
                records.append(record)
        full_start = time.perf_counter()
        record = {"plant_id": w.plant_id, "biological_plant_id": w.biological_plant_id,
                  "genotype": w.genotype, "replicate_id": w.replicate_id, "split_role": w.role,
                  "layer": "GT", "K": len(w.times), "N": len(w.times), "L_h": w.length,
                  "method": "full", "phase": None, "repeat": 0, "seed": SEED,
                  "rate_per_h": (len(w.times) - 1) / w.length, "area_eligible": w.area_eligible,
                  "roi_status": AREA_NOTE, "status": "success", "failure_reason": "",
                  "selected_indices": list(range(len(w.times))), "selected_elapsed_h": w.times.tolist(),
                  "selected_image_ids": w.image_ids, "actual_K": len(w.times),
                  "scheduling_wall_s": 0.0, "oracle_candidate_evaluations": 0,
                  "deployability": "FULL_GT_DISCRETE_REFERENCE_NOT_RGB_VALIDATION"}
        record.update(evaluate(w, np.arange(len(w.times))))
        record["replay_wall_s"] = time.perf_counter() - full_start
        records.append(record)
    return records, seeds


def empirical_q90(values: Sequence[float]) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    return float(x[math.ceil(.9 * len(x)) - 1]) if len(x) else math.nan


def bootstrap_mean(values: Sequence[float], seed: int, repeats: int = 2000) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    if len(x) == 1:
        return float(x[0]), math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(x, len(x), replace=True)) for _ in range(repeats)])
    lo, hi = np.quantile(means, [.025, .975])
    return float(np.mean(x)), float(lo), float(hi)


def collapse_repeats(records: list[dict]) -> list[dict]:
    """Random draws contribute ONE plant, not 20 independent experimental units."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["plant_id"], r["K"], r["method"])].append(r)
    result = []
    metrics = ("E_A_pp", "E_N_leaves", "joint_z", "joint_pass", "leaf_pass")
    for key, batch in groups.items():
        r = dict(batch[0])
        r["status"] = "success" if all(x["status"] == "success" for x in batch) else "failed"
        r["random_repeats"] = len(batch) if key[2] == "R" else 0
        for metric in metrics:
            values = [x.get(metric, math.nan) for x in batch]
            r[metric] = float(np.mean(values)) if np.all(np.isfinite(values)) else math.nan
        r["random_E_A_q90_pp"] = empirical_q90([x.get("E_A_pp", math.nan) for x in batch]) if key[2] == "R" else math.nan
        result.append(r)
    return result


def common_sets(records: list[dict], area_only: bool) -> tuple[dict[int, set[str]], set[str]]:
    groups = defaultdict(list)
    for r in records:
        if r["method"] not in ("oracle", "full") and (r["area_eligible"] or not area_only):
            groups[(r["K"], r["plant_id"])].append(r)
    common = {k: set() for k in BUDGETS}
    for (k, plant), batch in groups.items():
        if len(batch) == 26 and all(r["status"] == "success" for r in batch):
            common[k].add(plant)
    return common, set.intersection(*(common[k] for k in BUDGETS))


def summarize(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    collapsed = collapse_repeats(records)
    area_common, area_intersection = common_sets(records, True)
    leaf_common, leaf_intersection = common_sets(records, False)
    lookup = {(r["plant_id"], r["K"], r["method"]): r for r in collapsed}
    summaries, genotype_rows = [], []
    genotypes = sorted({r["genotype"] for r in records})
    for scope in ("per_K_common", "across_K_common"):
        for k in BUDGETS:
            for method in METHOD_ORDER[:-1]:
                for genotype in [None] + genotypes:
                    attempted = [r for r in collapsed if r["K"] == k and r["method"] == method and
                                 (genotype is None or r["genotype"] == genotype)]
                    if not attempted:
                        continue
                    aset = area_common[k] if scope == "per_K_common" else area_intersection
                    lset = leaf_common[k] if scope == "per_K_common" else leaf_intersection
                    area = [r for r in attempted if r["plant_id"] in aset and r["status"] == "success"]
                    leaf = [r for r in attempted if r["plant_id"] in lset and r["status"] == "success"]
                    area_all = [r for r in attempted if r["area_eligible"]]
                    differences = [r["E_A_pp"] - lookup[(r["plant_id"], k, "U_0")]["E_A_pp"] for r in area]
                    difference, lo, hi = bootstrap_mean(differences, plant_seed(f"{scope}|{method}|{genotype}", k))
                    def mean(batch: list[dict], metric: str) -> float:
                        return float(np.mean([r[metric] for r in batch])) if batch else math.nan
                    row = {"scope": scope, "K": k, "method": method, "genotype": genotype or "ALL",
                           "n_attempted_plants": len(attempted), "n_area_eligible_plants": len(area_all),
                           "n_area_common": len(area), "n_leaf_common": len(leaf),
                           "schedule_failed_plants": sum(r["status"] != "success" for r in attempted),
                           "schedule_failure_rate": sum(r["status"] != "success" for r in attempted) / len(attempted),
                           "mean_E_A_pp": mean(area, "E_A_pp"), "mean_E_N_leaves": mean(leaf, "E_N_leaves"),
                           "joint_pass_rate_common": mean(area, "joint_pass"),
                           "leaf_pass_rate_common": mean(leaf, "leaf_pass"),
                           "joint_pass_rate_all_area_eligible": (sum(r.get("joint_pass", 0) if r["status"] == "success" else 0
                               for r in area_all) / len(area_all) if area_all else math.nan),
                           "empirical_q90_joint_z_common": empirical_q90([r["joint_z"] for r in area]),
                           "paired_mean_method_minus_U0_pp": difference,
                           "paired_bootstrap95_lo_pp": lo, "paired_bootstrap95_hi_pp": hi,
                           "random_mean_plant_q90_E_A_pp": mean(area, "random_E_A_q90_pp") if method == "R" else math.nan,
                           "oracle_diagnostic_only": method == "oracle", "bootstrap_repeats": 2000}
                    row["empirical_90pct_pass_common"] = bool(area and row["joint_pass_rate_common"] >= .9)
                    (summaries if genotype is None else genotype_rows).append(row)
    for method in ("full",):
        batch = [r for r in collapsed if r["method"] == method]
        for genotype in [None] + genotypes:
            sub = [r for r in batch if genotype is None or r["genotype"] == genotype]
            if sub:
                area = [r for r in sub if r["area_eligible"]]
                row = {"scope": "full_reference", "K": "full_N", "method": method,
                       "genotype": genotype or "ALL", "n_attempted_plants": len(sub),
                       "n_area_common": len(area), "n_leaf_common": len(sub),
                       "mean_E_A_pp": 0.0 if area else math.nan, "mean_E_N_leaves": 0.0,
                       "joint_pass_rate_common": 1.0 if area else math.nan, "schedule_failure_rate": 0.0}
                (summaries if genotype is None else genotype_rows).append(row)
    interactions = []
    for k in BUDGETS:
        values = []
        for plant in sorted(area_common[k]):
            errors = [lookup[(plant, k, m)]["E_A_pp"] for m in METHOD_ORDER[:3]]
            values.append(max(errors) - min(errors))
        estimate, lo, hi = bootstrap_mean(values, plant_seed("phase_amplitude", k))
        interactions.append({"type": "phase_max_minus_min", "K": k, "n_plants": len(values),
                             "mean_pp": estimate, "bootstrap95_lo_pp": lo, "bootstrap95_hi_pp": hi})
    for low, high in ((5, 9), (9, 15), (5, 15)):
        plants = area_common[low] & area_common[high]
        for method in ("U_-1/3", "U_+1/3"):
            values = [(lookup[(p, low, method)]["E_A_pp"] - lookup[(p, low, "U_0")]["E_A_pp"])
                      - (lookup[(p, high, method)]["E_A_pp"] - lookup[(p, high, "U_0")]["E_A_pp"])
                      for p in sorted(plants)]
            estimate, lo, hi = bootstrap_mean(values, plant_seed(f"DID|{method}|{high}", low))
            interactions.append({"type": "difference_of_differences", "K_low": low, "K_high": high,
                                 "phase_method": method, "n_plants": len(values), "mean_pp": estimate,
                                 "bootstrap95_lo_pp": lo, "bootstrap95_hi_pp": hi})
    info = {"common_feasibility_definition": "all 26 non-oracle plans including all R draws and GT-online succeed; never conditioned on error pass",
            "area_common_by_K": {str(k): sorted(v) for k, v in area_common.items()},
            "leaf_common_by_K": {str(k): sorted(v) for k, v in leaf_common.items()},
            "area_across_K_intersection": sorted(area_intersection),
            "leaf_across_K_intersection": sorted(leaf_intersection), "phase_interactions": interactions,
            "ranking": "not_computed: comparable physical ROI, age/batch and identical L not established",
            "random_aggregation": "mean within plant first; draw pass fraction and plant q90 retained"}
    return summaries, genotype_rows, info


def model_features(k: float, length: float, phase: float) -> np.ndarray:
    rate = math.log((k - 1) / length)
    sine, cosine = math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)
    return np.array([rate, rate * rate, sine, cosine, rate * sine, rate * cosine])


def fit_quantile_models(records: list[dict], windows: list[Window]) -> dict:
    """Two CPU GT models, TRAIN only, with whole-plant total weight one."""
    eligible = [r for r in records if r["status"] == "success" and r["area_eligible"] and r["method"].startswith("U_")]
    train = [r for r in eligible if r["split_role"] == "TRAIN"]
    train_plants = {r["plant_id"] for r in train}
    train_counts = Counter(r["plant_id"] for r in train)
    result = {"status": "insufficient_training_plants", "min_training_plants": 10,
              "min_validation_plants_per_role": 5, "train_plants": len(train_plants),
              "train_replays": len(train), "quantile": .9, "epsilon_area_pp": .01,
              "epsilon_leaf": .01, "penalty": .1,
              "penalty_definition": "0.5 * 0.1 * sum(coef_squared); unpenalized intercept",
              "solver": "scipy L-BFGS-B; check-loss smoothing=1e-6 log units",
              "features": ["log_rate", "log_rate_squared", "sin_phase", "cos_phase", "rate_sin", "rate_cos"],
              "weight_definition": "each TRAIN plant's total successful U replay weight is 1",
              "models": {}, "phase_table_development_only": [], "fit_wall_s": 0.0}
    if len(train_plants) < 10:
        return result
    started = time.perf_counter()
    x = np.array([model_features(r["K"], r["L_h"], r["phase"]) for r in train])
    weights = np.array([1 / train_counts[r["plant_id"]] for r in train])
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1
    models = {}
    try:
        for endpoint in ("E_A_pp", "E_N_leaves"):
            y = np.log(np.array([r[endpoint] for r in train]) + .01)
            model = L2QuantileModel(quantile=.9, penalty=.1)
            model.fit((x - center) / scale, y, sample_weight=weights)
            models[endpoint] = model
            model_info = {"coef": model.coef_.tolist(), "intercept": float(model.intercept_),
                          "center": center.tolist(), "scale": scale.tolist(), "solver": model.solver_info,
                          "validation": {}}
            for role in ("V_SOURCE", "V_SHIFT"):
                val = [r for r in eligible if r["split_role"] == role]
                plants = {r["plant_id"] for r in val}
                coverage = []
                for plant in sorted(plants):
                    sub = [r for r in val if r["plant_id"] == plant]
                    vx = np.array([model_features(r["K"], r["L_h"], r["phase"]) for r in sub])
                    pred = model.predict((vx - center) / scale)
                    observed = np.log(np.array([r[endpoint] for r in sub]) + .01)
                    coverage.append(float(np.mean(observed <= pred + 1e-12)))
                model_info["validation"][role] = {"n_plants": len(plants), "n_replays": len(val),
                    "plant_weighted_logerror_quantile_coverage": float(np.mean(coverage)) if coverage else math.nan,
                    "status": "evaluated" if len(plants) >= 5 else "insufficient_validation_plants"}
            result["models"][endpoint] = model_info
        for w in windows:
            if not w.area_eligible or w.role not in ROLES:
                continue
            for k in BUDGETS:
                options = []
                for phase in PHASES:
                    f = (model_features(k, w.length, phase) - center) / scale
                    estimates = {e: max(0.0, float(np.exp(np.clip(m.predict(f[None])[0], -700, 700))) - .01)
                                 for e, m in models.items()}
                    options.append((max(estimates["E_A_pp"], estimates["E_N_leaves"] / .5), abs(phase), phase, estimates))
                z, _, phase, estimates = min(options, key=lambda o: (round(o[0], 12), o[1], o[2]))
                result["phase_table_development_only"].append({"plant_id": w.plant_id, "split_role": w.role,
                    "K": k, "L_h": w.length, "phase": phase, "predicted_joint_quantile_loss": z,
                    "predicted_E_A_pp": estimates["E_A_pp"], "predicted_E_N_leaves": estimates["E_N_leaves"],
                    "not_C_POLICY_calibration": True})
        validation_ok = all(result["models"]["E_A_pp"]["validation"][r]["n_plants"] >= 5 for r in ("V_SOURCE", "V_SHIFT"))
        result["status"] = "fit_GT_only_validation_evaluated" if validation_ok else "fit_GT_only_insufficient_validation"
    except (ValueError, RuntimeError) as exc:
        result["status"] = "fit_failed"
        result["failure_reason"] = str(exc)
        result["phase_table_development_only"] = []
    result["fit_wall_s"] = time.perf_counter() - started
    return result


def p1_gates(summary: list[dict], info: dict) -> dict:
    # No oracle, no full-N, no locked roles, and failures count against the pass gate.
    candidates = [r for r in summary if r["scope"] == "per_K_common" and
                  r["method"] not in ("oracle", "full", "R")]
    adequate = [r for r in candidates if r["n_area_common"] >= 10]
    passing = [r for r in adequate if r["joint_pass_rate_common"] >= .9 and
               r["joint_pass_rate_all_area_eligible"] >= .9 and r["empirical_q90_joint_z_common"] <= 1 + 1e-12]
    usable = [r for r in adequate if r["method"] in ("U_-1/3", "U_+1/3", "online") and
              r["paired_mean_method_minus_U0_pp"] <= -.1]
    amplitude = [r for r in info["phase_interactions"] if r["type"] == "phase_max_minus_min" and
                 r["n_plants"] >= 10 and r["mean_pp"] >= .1]
    decision = ("insufficient_development_common_plants" if not adequate else
                "continue_phase_online_investigation" if passing and (usable or amplitude) else
                "prefer_simple_fixed_rule" if passing else "retain_failure_stop_complexification")
    return {"decision": decision, "development_only": True, "GT_only_not_deployment_validation": True,
            "provisional_ROI": True, "empirical_not_90pct_guarantee": True,
            "minimum_plants_for_pilot_gate": 10, "joint_tolerances": {"area_pp": 1, "leaves": .5},
            "worthwhile_difference_pp": .1, "nonfull_90pct_gate": bool(passing),
            "phase_online_signal_gate": bool(usable or amplitude),
            "qualifying_protocols": [{"K": r["K"], "method": r["method"],
                "n_common": r["n_area_common"], "all_eligible_joint_pass_rate": r["joint_pass_rate_all_area_eligible"]} for r in passing],
            "oracle_excluded": True, "random_not_used_for_deterministic_protocol_gate": True,
            "C_POLICY_not_run": True, "signals_are_exploratory_not_confirmatory": True}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], required: Sequence[str]) -> None:
    fields = list(dict.fromkeys(list(required) + [k for row in rows for k in row]))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {}
            for name, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    output[name] = json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)
                elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
                    output[name] = "NaN"
                else:
                    output[name] = value
            writer.writerow(output)


def make_plots(out: Path, audits: list[dict], summary: list[dict], records: list[dict], info: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    files = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            file = out / f"{name}.{extension}"
            fig.savefig(file, dpi=160, bbox_inches="tight")
            files.append(file.name)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    n = len(audits)
    labels = ("Dev plants", "Window eligible", "Area eligible")
    counts = (n, sum(a["window_eligible"] for a in audits), sum(a["area_eligible"] for a in audits))
    axes[0].bar(labels, counts, color=("#657d8a", "#449c89", "#ca9e55"))
    axes[0].set_ylabel("Biological plants")
    axes[0].set_title(f"Audited development plants: n={n}")
    coverage = [a["L_h"] for a in audits if math.isfinite(a["L_h"])]
    if coverage:
        axes[1].hist(coverage, bins=min(15, max(1, len(coverage))), color="#657d8a")
        axes[1].axvline(156, color="#b75050", linestyle="--")
        axes[1].axvline(168, color="#b75050", linestyle="--")
    else:
        axes[1].text(.5, .5, "No valid coverage data", ha="center", transform=axes[1].transAxes)
    axes[1].set_xlabel("Actual first-window coverage (h)")
    axes[1].set_ylabel("Plants")
    save(fig, "window_coverage")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, metric, tolerance, ylabel, nkey in (
        (axes[0], "mean_E_A_pp", 1, "GT area mean absolute error (pp)", "n_area_common"),
        (axes[1], "mean_E_N_leaves", .5, "GT visible-leaf mean absolute error", "n_leaf_common")
    ):
        drawn = False
        for method in ("U_-1/3", "U_0", "U_+1/3", "C", "J", "R", "online"):
            sub = [r for r in summary if r["scope"] == "across_K_common" and r["method"] == method
                   and math.isfinite(r[metric])]
            if sub:
                axis.plot([r["K"] for r in sub], [r[metric] for r in sub], marker="o",
                          label=f"{method} (n={sub[0][nkey]})")
                drawn = True
        if drawn:
            axis.axhline(tolerance, color="#b75050", linestyle="--", label="Tolerance")
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "No common feasible plants\nNo error curve estimated", ha="center", transform=axis.transAxes)
        axis.set_xticks(BUDGETS)
        axis.set_xlabel("Actual budget K (same-plant across-K intersection)")
        axis.set_ylabel(ylabel)
    fig.suptitle("GT only; uncalibrated fixed pixel field; not RGB deployment evidence")
    save(fig, "sampling_errors")
    if records:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        matrix = np.full((3, 3), np.nan)
        for i, method in enumerate(METHOD_ORDER[:3]):
            for j, k in enumerate(BUDGETS):
                sub = [r for r in summary if r["scope"] == "per_K_common" and r["method"] == method and r["K"] == k]
                if sub:
                    matrix[i, j] = sub[0]["mean_E_A_pp"]
        if np.any(np.isfinite(matrix)):
            image = axes[0].imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis")
            fig.colorbar(image, ax=axes[0], label="Mean area error (pp)")
        else:
            axes[0].text(.5, .5, "No common area data", ha="center", transform=axes[0].transAxes)
        axes[0].set_xticks(range(3), BUDGETS)
        axes[0].set_yticks(range(3), ["-1/3", "0", "+1/3"])
        axes[0].set_xlabel("K (per-K common plants)")
        axes[0].set_ylabel("Relative phase")
        axes[0].set_title("Phase comparisons, not best-phase envelope")
        plants = info["leaf_across_K_intersection"]
        plant = plants[0] if plants else None
        sub = [r for r in records if r["plant_id"] == plant and r["method"] in ("U_0", "online") and r["K"] == 9 and r["status"] == "success"]
        if sub:
            for i, r in enumerate(sub):
                axes[1].scatter(r["selected_elapsed_h"], [i] * len(r["selected_elapsed_h"]), label=r["method"], s=25)
                if r["method"] == "online":
                    axes[1].scatter(r["selected_elapsed_h"][:3], [i] * 3, facecolors="none", edgecolors="black", s=70, label="Paid warmup")
            axes[1].legend(fontsize=7)
            axes[1].set_title(f"First eligible plant (not best case): {plant}")
        else:
            axes[1].text(.5, .5, "No paired online trajectory", ha="center", transform=axes[1].transAxes)
        axes[1].set_xlabel("Elapsed hours from actual first valid frame")
        axes[1].set_yticks([])
        save(fig, "phase_and_schedule")
    return files


def read_inventory(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(SCHEMA) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("inventory_missing_columns:" + ",".join(sorted(missing)))
        rows = []
        for row in reader:
            if None in row or any(row.get(name) is None for name in SCHEMA):
                raise ValueError(f"malformed_inventory_row:{reader.line_num}")
            rows.append(row)
        return rows


def run(inventory: Path, out: Path, data_root: Path | None = None) -> dict:
    inventory, out = inventory.resolve(), out.resolve()
    data_root = (data_root or inventory.parent.parent).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output_directory_not_empty: use a fresh versioned output, no overwrite")
    out.mkdir(parents=True, exist_ok=True)
    started, clock = utc_now(), time.perf_counter()
    config = {"version": VERSION, "inventory": str(inventory), "out": str(out), "data_root": str(data_root),
              "UTC_start": started, "UTC_end": None, "runtime_wall_s": None, "seed": SEED,
              "allowed_effect_and_selection_roles": list(ROLES), "budgets": list(BUDGETS), "phases": list(PHASES),
              "random_repeats": 20, "planned_units_per_window": 82, "source_path": str(Path(__file__).resolve()),
              "inventory_sha256": None, "source_sha256": sha256_file(Path(__file__)),
              "environment": environment_versions(),
              "mask_status_required": "valid", "timestamp_semantics": "calendar datetime; naive values use computational zero-offset only, acquisition timezone unknown; aware elapsed time; clock uses supplied calendar midnight",
              "image_path_semantics": "absolute or relative to data_root; never infer from image filename",
              "GT_semantics": "audited union_pixels and n_instances; area_fraction cross-checked, no new masks inferred",
              "RGB_validity": "inventory hash present plus file exists and PIL pixel decode and dimensions match; content hashes not recomputed",
              "physical_ROI": AREA_NOTE, "clock_zero_offset_resolution_s": 1,
              "window_gates": {"L_min_h": 156, "L_max_h": 168, "N_min": 15, "median_dt_max_h": 12, "max_dt_max_h": 18},
              "online_parameters": {"warmup_h": [0, 6, 12], "eta": .5, "v_deadzone": .02, "c_deadzone": .002,
                                    "wait_clip_h0": [.5, 1.5], "known_opportunity_table": True},
              "limitations": ["GT discrete reference only", "No RGB segmentation/measurement decomposition",
                  "No C_POLICY or locked-test calibration/evaluation", "No measured energy/GPU saving",
                  "No physical ROI calibration or cross-plant biological ranking", "No relaxed-window rescue"]}
    audits, windows, records, summary, per_genotype = [], [], [], [], []
    info = {"phase_interactions": [], "leaf_across_K_intersection": [], "area_across_K_intersection": []}
    status = {"status": "blocked", "blocking_reasons": [], "real_analysis": True, "GT_only": True}
    try:
        config["inventory_sha256"] = sha256_file(inventory)
        rows = read_inventory(inventory)
        audit_start = time.perf_counter()
        audits, windows, counts = audit_windows(rows, data_root)
        config["counts"] = counts
        config["audit_wall_s"] = time.perf_counter() - audit_start
        records, config["derived_random_seeds"] = replay(windows)
        summary, per_genotype, info = summarize(records)
        config["statistics"] = info
        config["quantile_models"] = fit_quantile_models(records, windows)
        config["planned_P1_gates"] = p1_gates(summary, info)
        config["actual_replays"] = {"attempted": len(records), "successful": sum(r["status"] == "success" for r in records),
              "failed": sum(r["status"] != "success" for r in records),
              "failure_counts": dict(Counter(r["failure_reason"] for r in records if r["status"] != "success")),
              "oracle_candidate_evaluations": sum(r["oracle_candidate_evaluations"] for r in records),
              "schedule_wall_s_by_method": {m: sum(r["scheduling_wall_s"] for r in records if r["method"] == m) for m in METHOD_ORDER}}
        config["costs"] = {"reference_encoded_rgb_bytes": sum(w.encoded_rgb_bytes for w in windows),
              "reference_rgb_frames": sum(len(w.times) for w in windows), "GPU_training_runs": 0,
              "deployment_byte_savings": "not_measured", "hardware_energy_savings": "not_measured"}
        if not windows:
            status["blocking_reasons"] = ["no_window_passes_strict_primary_gates"]
        elif not any(w.area_eligible for w in windows):
            status["status"] = "leaf_only_area_blocked"
            status["blocking_reasons"] = ["no_stable_dimension_area_window"]
        else:
            status["status"] = "completed_provisional_GT_development_pilot"
        status["counts"] = counts
        status["P1_decision"] = config["planned_P1_gates"]["decision"]
        status["quantile_model_status"] = config["quantile_models"]["status"]
    except (OSError, ValueError, RuntimeError) as exc:
        status["status"] = "blocked_input_or_execution_error"
        status["blocking_reasons"] = [f"{type(exc).__name__}:{exc}"]
    write_csv(out / "window_audit.csv", audits, ("plant_id", "split_role", "a", "b", "L_h", "N", "window_eligible", "area_eligible", "exclusion_reasons", "row_times_and_errors"))
    write_csv(out / "replay_per_plant.csv", records, ("plant_id", "split_role", "K", "method", "repeat", "status", "failure_reason", "E_A_pp", "E_N_leaves", "joint_pass"))
    write_csv(out / "summary.csv", summary, ("scope", "K", "method", "n_area_common", "n_leaf_common", "mean_E_A_pp", "mean_E_N_leaves"))
    write_csv(out / "per_genotype.csv", per_genotype, ("genotype", "scope", "K", "method", "n_area_common", "mean_E_A_pp", "mean_E_N_leaves"))
    try:
        config["figures"] = make_plots(out, audits, summary, records, info)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        config["plot_failure"] = str(exc)
        status["plot_failure"] = str(exc)
    config["UTC_end"], config["runtime_wall_s"] = utc_now(), time.perf_counter() - clock
    status.update(UTC_start=started, UTC_end=config["UTC_end"], runtime_wall_s=config["runtime_wall_s"],
                  area_interpretation=AREA_NOTE)
    lines = [f"AEA开发角色GT最小验证：{status['status']}",
             "仅TRAIN、V_SOURCE、V_SHIFT；未分析或选择C_POLICY/T_ID/T_GENO等锁定角色。",
             "面积分析为固定像素视场fraction条件结果，物理ROI尺度未经证明，全部provisional；不报告cm²。",
             "首个RGB和GT共同有效帧起唯一前168h窗；严格156–168h/N≥15/medianΔt≤12h/maxΔt≤18h，不放宽、不补帧。",
             f"实际株与窗口计数：{json.dumps(config.get('counts', {}), ensure_ascii=False)}",
             f"实际回放计数：{len(records)}；理论每合格窗82单元，失败保留、oracle不可部署且非全局最优。",
             f"P1开发门槛：{status.get('P1_decision', '未执行')}；模型：{status.get('quantile_model_status', '未执行')}。",
             f"跨K面积共同株数：{len(info['area_across_K_intersection'])}。随机20次先株内聚合；GT达标不是RGB部署达标。",
             "无计划采集表不声称90%采集完整性；90%误差通过是经验率而非保证。",
             f"阻碍：{'；'.join(status['blocking_reasons']) or '物理尺度、RGB测量层和后续冻结校准尚待验证'}",
             f"UTC起止：{started} / {config['UTC_end']}；墙钟秒：{config['runtime_wall_s']:.3f}"]
    (out / "phase_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config["derived_outputs_bytes_before_json"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    write_json(out / "config.json", config)
    write_json(out / "status.json", status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AEA GT development-only sampling replay; no locked roles or real RGB model")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="base for relative image paths; default inventory.parent.parent")
    args = parser.parse_args(argv)
    try:
        status = run(args.inventory, args.out, args.data_root)
    except (OSError, ValueError) as exc:
        print(f"AEA blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(json_safe(status), ensure_ascii=False))
    # Strict-window scientific failure is a completed audit, not a software crash.
    return 2 if status["status"] == "blocked_input_or_execution_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Derive the numbers quoted in the manuscript from the two replay runs.

The script reads the development-role replay outputs (a ground-truth layer over
the 107 window-eligible plants and a joint layer over the 32 windows that also
have RGB measurements) and recomputes, rather than copies, every quantity that
appears in the paper: the window funnel, the per-budget schedule comparison with
its paired intervals, the phase amplitude and the budget-by-schedule
difference-of-differences that the design gate turns on, and the three-term
error decomposition of the joint layer.

Run from the paper directory:

    python scripts/derive_numbers.py --gt GT_RUN_DIR --rgb RGB_RUN_DIR \
        --out data/numbers.json

Both directories are expected to hold ``summary.csv``, ``config.json`` and
``replay_per_plant.csv`` / ``per_plant_joint_errors.csv`` respectively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

SIDES = ("left", "top", "right", "bottom")


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def num(value):
    if value in ("", None, "NaN", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def load_gt(gt_dir):
    cfg = json.load(open(os.path.join(gt_dir, "config.json"), encoding="utf-8-sig"))
    summary = read_csv(os.path.join(gt_dir, "summary.csv"))
    per_plant = read_csv(os.path.join(gt_dir, "replay_per_plant.csv"))
    audit = read_csv(os.path.join(gt_dir, "window_audit.csv"))
    return cfg, summary, per_plant, audit


def window_density(audit):
    """Descriptive statistics of the windows the gates kept.

    These say what the eligible population actually looks like: how many jointly
    valid frames it holds, how long it spans and how evenly the frames are spread.
    """
    def stats(values):
        values = sorted(values)
        mid = len(values) // 2
        median = (values[mid] if len(values) % 2
                  else 0.5 * (values[mid - 1] + values[mid]))
        return {"min": values[0], "median": median, "max": values[-1]}

    live = [rec for rec in audit if truthy(rec.get("window_eligible"))]
    area = [rec for rec in audit if truthy(rec.get("area_eligible"))]
    out = {}
    for label, group in (("window_eligible", live), ("area_eligible", area)):
        frames = [float(rec["N"]) for rec in group]
        spans = [float(rec["L_h"]) for rec in group]
        gaps = [float(rec["median_dt_h"]) for rec in group]
        out[label] = {
            "plants": len(group),
            "N": stats(frames),
            "L_h": stats(spans),
            "median_dt_h": stats(gaps),
            "mean_interval_h": stats([s / (n - 1.0) for s, n in zip(spans, frames)]),
        }
    return out


def funnel(cfg):
    counts = cfg["counts"]
    total = counts["development_plants"]
    out = {
        "inventory_rows": counts["inventory_rows"],
        "development_plants": total,
        "window_eligible": counts["window_eligible_plants"],
        "area_eligible": counts["area_eligible_plants"],
        "window_roles": counts["window_roles"],
        "area_window_roles": counts["area_window_roles"],
        "exclusions": counts["exclusion_counts"],
        "exclusion_share": {k: v / total for k, v in counts["exclusion_counts"].items()},
        "area_dimension_exclusions": counts["area_dimension_exclusions"],
        "duplicate_timestamps": counts["duplicate_timestamps"],
        "timestamp_conflicts": counts["timestamp_conflicts"],
        "naive_timestamps": counts["naive_timestamps_assumed_UTC"],
        "window_share_of_development": counts["window_eligible_plants"] / total,
        "area_share_of_development": counts["area_eligible_plants"] / total,
    }
    return out


def schedule_table(summary):
    rows = {}
    for rec in summary:
        if rec["scope"] != "per_K_common":
            continue
        key = (int(float(rec["K"])), rec["method"])
        rows[key] = {
            "common_plants": int(float(rec["n_area_common"])),
            "mean_E_A_pp": num(rec["mean_E_A_pp"]),
            "mean_E_N_leaves": num(rec["mean_E_N_leaves"]),
            "joint_pass_rate": num(rec["joint_pass_rate_all_area_eligible"]),
            "q90_joint_z": num(rec["empirical_q90_joint_z_common"]),
            "diff_vs_U0_pp": num(rec["paired_mean_method_minus_U0_pp"]),
            "ci_lo_pp": num(rec["paired_bootstrap95_lo_pp"]),
            "ci_hi_pp": num(rec["paired_bootstrap95_hi_pp"]),
            "plan_failures": int(float(rec["schedule_failed_plants"])),
            "oracle": rec["oracle_diagnostic_only"] == "True",
        }
    return rows


def budget_spread(rows, ks, oracle="oracle"):
    """Widest gap between two families at the same budget, with and without the oracle.

    The paired differences against U_0 cannot answer "what is the largest difference
    anywhere in the design space", because two families can sit on opposite sides of
    the reference. This is the max-minus-min of the mean area error at each K.
    """
    out = {}
    for k in ks:
        present = {m: value["mean_E_A_pp"] for kk, m in rows if kk == k
                   for value in [rows[(kk, m)]]}
        deployable = {m: v for m, v in present.items() if m != oracle}
        for label, group in (("all", present), ("deployable", deployable)):
            best = min(group, key=group.get)
            worst = max(group, key=group.get)
            out["K%d/%s" % (k, label)] = {
                "spread_pp": group[worst] - group[best],
                "worst": worst, "best": best,
                "worst_pp": group[worst], "best_pp": group[best],
            }
    return out


def phase_amplitude(rows, ks, phases):
    amp = {}
    for k in ks:
        vals = [rows[(k, "U_%s" % p)]["mean_E_A_pp"] for p in phases]
        amp[k] = {
            "min_pp": min(vals),
            "max_pp": max(vals),
            "amplitude_pp": max(vals) - min(vals),
        }
    return amp


def difference_of_differences(rows, phases, k_low, k_high):
    """[E(k_low, phi) - E(k_low, 0)] - [E(k_high, phi) - E(k_high, 0)]."""
    out = {}
    for phi in list(phases) + ["C", "J", "R", "online"]:
        name = "U_%s" % phi if phi in phases else phi
        lo = rows[(k_low, name)]["diff_vs_U0_pp"]
        hi = rows[(k_high, name)]["diff_vs_U0_pp"]
        out[name] = {"at_K%d_pp" % k_low: lo, "at_K%d_pp" % k_high: hi,
                     "difference_pp": lo - hi}
    return out


def p1_gates(cfg):
    """The gate block written before the replay ran, minus its long protocol list."""
    gates = dict(cfg["planned_P1_gates"])
    gates["qualifying_protocols_n"] = len(gates.pop("qualifying_protocols"))
    return gates


def quantile_model(cfg):
    """Fitted quantile-regression error model and its validation-role coverage."""
    q = cfg["quantile_models"]
    simple = ("status", "quantile", "train_plants", "train_replays", "features",
              "penalty", "penalty_definition", "epsilon_area_pp", "epsilon_leaf",
              "fit_wall_s", "phase_table_development_only", "min_training_plants",
              "min_validation_plants_per_role")
    out = {key: q[key] for key in simple if key in q}
    out["validation_coverage"] = {
        target: {role: rec["plant_weighted_logerror_quantile_coverage"]
                 for role, rec in model["validation"].items()}
        for target, model in q["models"].items()
    }
    out["validation_sizes"] = {
        target: {role: [rec["n_plants"], rec["n_replays"]]
                 for role, rec in model["validation"].items()}
        for target, model in q["models"].items()
    }
    return out


def superseded(path):
    """Accounting of an earlier run of the same protocol, for disclosure."""
    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        old = json.load(handle)
    counts = old.get("counts", {})
    return {
        "name": os.path.basename(os.path.normpath(path)),
        "source_sha256": old["source_sha256"],
        "UTC_start": old["UTC_start"],
        "actual_replays": old["actual_replays"],
        "runtime_wall_s": old["runtime_wall_s"],
        "window_eligible_plants": counts.get("window_eligible_plants"),
        "area_eligible_plants": counts.get("area_eligible_plants"),
    }


def paired_budget_flips(per_plant, method, k_low, k_high):
    """Plants whose E_A rises or falls when the budget goes from k_low to k_high."""
    by = {}
    for rec in per_plant:
        if rec.get("layer") != "GT" or rec.get("method") != method:
            continue
        if not truthy(rec.get("area_eligible")) or rec.get("status") != "success":
            continue
        value = num(rec["E_A_pp"])
        if value is None:
            continue
        by.setdefault(rec["plant_id"], {})[rec["K"]] = value
    pairs = [v for v in by.values() if k_low in v and k_high in v]
    return {
        "method": method,
        "k_low": k_low,
        "k_high": k_high,
        "plants": len(pairs),
        "worse_at_high": sum(1 for v in pairs if v[k_high] > v[k_low]),
        "better_at_high": sum(1 for v in pairs if v[k_high] < v[k_low]),
        "identical": sum(1 for v in pairs if v[k_high] == v[k_low]),
    }


def decomposition(rgb_dir):
    rows = read_csv(os.path.join(rgb_dir, "per_plant_joint_errors.csv"))
    live = [r for r in rows if r["area_eligible"] == "True"]
    residual = 0.0
    for r in rows:
        for col in ("area_mean_identity_residual", "leaf_mean_identity_residual",
                    "area_integral_identity_residual"):
            value = num(r[col])
            if value is not None:
                residual = max(residual, abs(value))
    per_budget = {}
    for k in ("5", "9", "15", "full_N"):
        method = "fullRGB" if k == "full_N" else "U_0"
        group = [r for r in live if r["K"] == k and r["method"] == method]
        if not group:
            continue

        def mean_abs(col):
            return sum(abs(num(r[col])) for r in group) / len(group) * 100.0

        per_budget[k] = {
            "plants": len(group),
            "mean_abs_e_M_pp": mean_abs("area_mean_e_M_abs"),
            "mean_abs_e_S_pp": mean_abs("area_mean_e_S_abs"),
            "mean_abs_e_X_pp": mean_abs("area_mean_e_X_abs"),
            "mean_abs_e_total_pp": mean_abs("area_mean_e_total_abs"),
            "share_measurement_dominant": sum(
                1 for r in group
                if abs(num(r["area_mean_e_M_abs"])) > abs(num(r["area_mean_e_S"]))
            ) / len(group),
            "mean_E_A_pp": sum(num(r["E_A_pp"]) for r in group) / len(group),
            "mean_E_N_leaves": sum(num(r["E_N_leaves"]) for r in group) / len(group),
            "joint_pass": sum(1 for r in group if r["joint_pass"] == "1.0"),
        }
    reference = [r for r in live if r["K"] == "full_N" and r["method"] == "fullRGB"]
    fails = []
    for r in reference:
        if r["joint_pass"] != "1.0":
            fails.append({
                "plant": r["plant_id"],
                "genotype": r["genotype"],
                "role": r["split_role"],
                "E_A_pp": num(r["E_A_pp"]),
                "E_N_leaves": num(r["E_N_leaves"]),
                "abs_e_M_pp": num(r["area_mean_e_M_abs"]) * 100.0,
                "N": int(float(r["N"])),
            })
    return {
        "records": len(rows),
        "area_eligible_records": len(live),
        "plants": len({r["plant_id"] for r in live}),
        "plants_by_role": {
            role: len({r["plant_id"] for r in live if r["split_role"] == role})
            for role in sorted({r["split_role"] for r in live})
        },
        "_area_eligible_plant_ids": sorted({r["plant_id"] for r in live}),
        "max_identity_residual": residual,
        "per_budget_U0_and_fullRGB": per_budget,
        "full_density_joint_failures": fails,
        "deployability_labels": sorted({r["deployability"] for r in rows}),
        "area_pass_at_full": sum(1 for r in reference if num(r["E_A_pp"]) <= 1.0),
        "leaf_pass_at_full": sum(1 for r in reference if num(r["E_N_leaves"]) <= 0.5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1", default=None,
                    help="directory of the superseded pilot, if one is being disclosed")
    args = ap.parse_args()

    cfg, summary, per_plant, audit = load_gt(args.gt)
    rows = schedule_table(summary)
    ks = sorted({k for k, _ in rows})
    phases = ("-1/3", "0", "+1/3")
    payload = {
        "protocol": {
            "window_gates": cfg["window_gates"],
            "budgets": cfg["budgets"],
            "phases": cfg["phases"],
            "random_repeats": cfg["random_repeats"],
            "planned_units_per_window": cfg["planned_units_per_window"],
            "online_parameters": cfg["online_parameters"],
            "seed": cfg["seed"],
            "selection_roles": cfg["allowed_effect_and_selection_roles"],
            "physical_roi": cfg["physical_ROI"],
            "source_sha256": cfg["source_sha256"],
            "inventory_sha256": cfg["inventory_sha256"],
            "joint_tolerances": cfg["planned_P1_gates"]["joint_tolerances"],
            "worthwhile_difference_pp":
                cfg["planned_P1_gates"]["worthwhile_difference_pp"],
        },
        "gt_p1_gates": p1_gates(cfg),
        "gt_quantile_model": quantile_model(cfg),
        "gt_limitations": cfg["limitations"],
        "superseded_run": superseded(args.v1) if args.v1 else None,
        "gt_funnel": funnel(cfg),
        "gt_window_density": window_density(audit),
        "gt_replays": cfg["actual_replays"],
        "gt_wall_clock_s": cfg["runtime_wall_s"],
        "gt_schedule_table": {"K%d/%s" % key: value for key, value in sorted(rows.items())},
        "gt_budget_spread": budget_spread(rows, ks),
        "gt_phase_amplitude": phase_amplitude(rows, ks, phases),
        "gt_interaction_5_9": difference_of_differences(rows, phases, 5, 9),
        "gt_interaction_9_15": difference_of_differences(rows, phases, 9, 15),
        "gt_interaction_5_15": difference_of_differences(rows, phases, 5, 15),
        "gt_paired_flips": {
            "%s_to_%s" % (low, high): paired_budget_flips(per_plant, "U_0", low, high)
            for low, high in (("5", "9"), ("9", "15"), ("5", "15"))
        },
        "joint_layer": decomposition(args.rgb),
    }
    gt_area_plants = {rec["plant_id"] for rec in per_plant
                      if rec.get("layer") == "GT" and truthy(rec.get("area_eligible"))}
    joint_ids = payload["joint_layer"].pop("_area_eligible_plant_ids")
    payload["joint_layer"]["plants_subset_of_gt_area_eligible"] = (
        set(joint_ids) <= gt_area_plants)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print("wrote %s" % args.out)
    print(json.dumps({
        "phase amplitude": payload["gt_phase_amplitude"],
        "interaction 9->15": payload["gt_interaction_9_15"],
        "joint per budget": payload["joint_layer"]["per_budget_U0_and_fullRGB"],
        "max identity residual": payload["joint_layer"]["max_identity_residual"],
        "full-density passes": (payload["joint_layer"]["area_pass_at_full"],
                                payload["joint_layer"]["leaf_pass_at_full"]),
    }, indent=1))


if __name__ == "__main__":
    main()

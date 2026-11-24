"""CPU synthetic-only tests: metadata/cache/filesystem outputs stay in memory.

No real inventory, image, GT, checkpoint or queue is opened. Source-code hashes
are the sole project files read. No GPU model, network, pip or background job.
"""
import sys
sys.dont_write_bytecode = True

from contextlib import ExitStack
import copy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import socket
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import aea_rgb_pilot as rgb

gt = rgb.gt
FAKE_ROOT = ROOT / "synthetic_in_memory_only"
PROVENANCE = {"checkpoint_sha256": "c"*64,
              "source_hashes": {**rgb.SOURCE_PINS, "aea_rgb_pilot.py": "d"*64}}
CONFIG_HASH = "e"*64
INVENTORY_HASH = "f"*64


def synthetic_window(plant="g::r1", role="V_SOURCE", times=None, area_eligible=True):
    t = np.arange(0, 169, 6, dtype=float) if times is None else np.asarray(times, dtype=float)
    return gt.Window(plant, "bio_"+plant, plant.split("::")[0], plant.split("::")[-1], role,
        datetime(2026, 1, 1, tzinfo=timezone.utc), t, np.full(len(t), .08), np.full(len(t), 4.),
        [f"synthetic_{plant}_{i}" for i in range(len(t))], area_eligible, 20, 20)


def synthetic_rows(w):
    result = []
    for i, hour in enumerate(w.times):
        row = {k: "" for k in gt.SCHEMA}
        row.update(image_id=w.image_ids[i], biological_plant_id=w.biological_plant_id,
            genotype=w.genotype, replicate_id=w.replicate_id, timestamp=(w.start+timedelta(hours=float(hour))).isoformat(),
            split_role=w.role, rgb_path="SYNTHETIC_NEVER_OPEN_RGB.png", mask_path="SYNTHETIC_NEVER_OPEN_GT.png",
            label_path="SYNTHETIC_NEVER_OPEN_LABEL.npy", width="20", height="20", mask_status="valid",
            unknown_pixels="0", union_pixels="32", n_instances="4", area_fraction="0.08",
            rgb_sha256=hashlib.sha256((w.plant_id+str(i)+"rgb").encode()).hexdigest(),
            mask_sha256=hashlib.sha256((w.plant_id+str(i)+"gt").encode()).hexdigest())
        result.append(row)
    return result


def mask_output(pixels=32, leaves=4):
    masks = torch.zeros((leaves, 1, 20, 20), dtype=torch.float32)
    if leaves:
        for i in range(pixels):
            masks[i % leaves, 0, i // 20, i % 20] = .9
    return {"scores": torch.ones(leaves), "masks": masks}


def memory_mask_cache(w, pixels=32, leaves=4, config_hash=CONFIG_HASH):
    # Dense soft instance masks generated only in CPU RAM; no real images.
    output = mask_output(pixels, leaves)
    return {row["image_id"]: rgb.make_measurement(row, output, .01, config_hash, PROVENANCE)
            for row in synthetic_rows(w)}


def audit_manifest(w):
    entries = [{"image_id": key, "datetime": (w.start+timedelta(hours=float(t))).isoformat(),
                "rgb_gt_valid": True, "gt_valid": True, "duplicate": False,
                "elapsed_h": float(t), "errors": []} for key, t in zip(w.image_ids, w.times)]
    return {"plant_id": w.plant_id, "biological_plant_id": w.biological_plant_id,
        "genotype": w.genotype, "replicate_id": w.replicate_id, "split_role": w.role,
        "a": w.start.isoformat(), "b": (w.start+timedelta(hours=w.length)).isoformat(),
        "N": str(len(w.times)), "L_h": str(w.length), "window_eligible": "True",
        "area_eligible": str(w.area_eligible), "exclusion_reasons": "[]",
        "dimensions_all_GT_frames": "[[20,20]]", "row_times_and_errors": json.dumps(entries)}


def csv_text(rows, fields=None):
    stream = io.StringIO(newline="")
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class CPUOnlyTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("NO_GPU")))
        self.stack.enter_context(patch.object(torch.cuda, "synchronize", side_effect=AssertionError("NO_GPU")))
        self.stack.enter_context(patch.object(socket, "socket", side_effect=AssertionError("NO_NETWORK")))


class DecompositionTests(CPUOnlyTest):
    def test_identical_constant_signals_total_zero_all_functionals(self):
        w = synthetic_window()
        with patch.object(gt, "evaluate", side_effect=AssertionError("DO_NOT_EVALUATE_RGB_AS_GT")):
            records = rgb.replay_joint([w], memory_mask_cache(w))
        self.assertEqual(len(records), 82)
        for r in records:
            if r["status"] == "success":
                for f in rgb.FUNCTIONALS:
                    for term in rgb.TERMS:
                        self.assertAlmostEqual(r[f"{f}_{term}"], 0, places=12)
                self.assertEqual(r["actual_K"], len(w.times) if r["K"] == "full_N" else r["K"])
        self.assertTrue(any(r["method"] == "oracle" and r["oracle_diagnostic_only"] for r in records))

    def test_constant_RGB_bias_eM_nonzero_eX_zero(self):
        w = synthetic_window()
        w.area += .002*np.sin(w.times/13)
        w.leaves += .2*np.cos(w.times/11)
        e = rgb.joint_errors(w, w.area+.02, w.leaves+2, [0, 5, 13, 22, 28])
        self.assertAlmostEqual(e["area_mean_e_M"], .02)
        self.assertAlmostEqual(e["leaf_mean_e_M"], 2)
        self.assertAlmostEqual(e["area_integral_e_M"], .02*w.length)
        for f in rgb.FUNCTIONALS:
            self.assertAlmostEqual(e[f+"_e_X"], 0, places=12)

    def test_time_varying_bias_nonzero_interaction(self):
        w = synthetic_window()
        bias = np.zeros(len(w.times))
        bias[1] = .2  # Not selected, but has weight in dense reference.
        e = rgb.joint_errors(w, w.area+bias, w.leaves+20*bias, [0, 7, 14, 21, 28])
        self.assertNotEqual(e["area_mean_e_X"], 0)
        self.assertNotEqual(e["leaf_mean_e_X"], 0)
        self.assertNotEqual(e["area_integral_e_X"], 0)
        self.assertAlmostEqual(e["area_mean_e_total"], 0)
        self.assertAlmostEqual(e["area_mean_e_M"], -e["area_mean_e_X"])
        self.assertGreater(e["area_mean_e_M_abs"]+e["area_mean_e_X_abs"], e["area_mean_e_total_abs"])

    def test_randomized_identity_unequal_time_weights(self):
        rng = np.random.default_rng(43)
        w = synthetic_window(times=[0, 6, 12, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 168])
        for _ in range(120):
            w.area = rng.uniform(0, .5, len(w.times))
            w.leaves = rng.uniform(0, 15, len(w.times))
            a, n = rng.uniform(0, 1, len(w.times)), rng.uniform(0, 20, len(w.times))
            selected = gt.random_schedule(w.times, 5, int(rng.integers(0, 10000)))
            e = rgb.joint_errors(w, a, n, selected)
            for f in rgb.FUNCTIONALS:
                self.assertAlmostEqual(e[f+"_e_total"], sum(e[f+"_"+t] for t in rgb.TERMS[:3]), places=11)
                self.assertAlmostEqual(e[f+"_e_total_abs"], abs(e[f+"_e_total"]))

    def test_identical_nonlinear_RGB_still_has_GT_sampling_error(self):
        w = synthetic_window()
        w.area[1] += .2
        e = rgb.joint_errors(w, w.area, w.leaves, [0, 7, 14, 21, 28])
        self.assertEqual(e["area_mean_e_M"], 0)
        self.assertEqual(e["area_mean_e_X"], 0)
        self.assertNotEqual(e["area_mean_e_total"], 0)
        self.assertEqual(e["area_mean_e_total"], e["area_mean_e_S"])

    def test_fullRGB_not_truth_and_K_cannot_fix_constant_bias(self):
        w = synthetic_window()
        records = rgb.replay_joint([w], memory_mask_cache(w, pixels=40))
        full = next(r for r in records if r["method"] == "fullRGB")
        self.assertAlmostEqual(full["E_A_pp"], 2)
        self.assertEqual(full["area_mean_e_S"], 0)
        self.assertEqual(full["area_mean_e_X"], 0)
        self.assertFalse(full["joint_pass"])
        for r in records:
            if r["method"] == "U_0" and r["status"] == "success":
                self.assertAlmostEqual(r["E_A_pp"], 2)

    def test_small_denominator_no_clamp_and_area_leaf_only(self):
        w = synthetic_window()
        for area in (0, .0009):
            w.area[:] = area
            e = rgb.joint_errors(w, w.area+.01, w.leaves, [0, 7, 14, 21, 28])
            self.assertTrue(math.isnan(e["relative_area_integral_pct"]))
            self.assertAlmostEqual(e["E_A_pp"], 1)
            self.assertIn("small", e["relative_integral_status"])
        w.area[:] = .001
        self.assertTrue(math.isfinite(rgb.joint_errors(w, w.area+.01, w.leaves, [0, 28])["relative_area_integral_pct"]))
        w.area_eligible = False
        e = rgb.joint_errors(w, w.area+.01, w.leaves+1, [0, 28])
        self.assertTrue(math.isnan(e["E_A_pp"]))
        self.assertTrue(math.isnan(e["area_mean_full_GT"]))
        self.assertEqual(e["E_N_leaves"], 1)
        self.assertEqual(e["relative_integral_status"], "area_ineligible")

    def test_endpoint_unique_index_contract(self):
        w = synthetic_window()
        for indices in ([1, 28], [0, 27], [0, 7, 7, 28], [0, 28, 29]):
            with self.assertRaises(ValueError):
                rgb.joint_errors(w, w.area, w.leaves, indices)


class CausalityAndRolesTests(CPUOnlyTest):
    def test_future_RGB_and_GT_cannot_affect_prior_online_selection(self):
        w = synthetic_window()
        original = np.column_stack((w.area, w.leaves))
        future = original.copy()
        future[w.times >= 84] = (.9, 60)
        reads = []
        s1 = rgb.select_schedule(w, 9, "online", lambda i: (reads.append(i) or original[i]))
        changed = copy.deepcopy(w)
        changed.area[w.times >= 6] = .99
        changed.leaves[w.times >= 6] = 99999  # GT never read by online.
        s_gt = rgb.select_schedule(changed, 9, "online", lambda i: original[i])
        np.testing.assert_array_equal(s1, s_gt)
        s2 = rgb.select_schedule(changed, 9, "online", lambda i: future[i])
        first_change = next(j for j, i in enumerate(s1) if w.times[i] >= 84)
        np.testing.assert_array_equal(s1[:first_change+1], s2[:first_change+1])
        self.assertEqual(reads, s1[:-1].tolist())
        self.assertEqual(s1[:3].tolist(), [0, 1, 2])
        self.assertEqual(s1[-1], len(w.times)-1)
        self.assertEqual(len(s1), 9)

    def test_online_decomposition_returns_GT_on_the_same_RGB_selected_S(self):
        w = synthetic_window()
        w.area = .08+.01*np.sin(w.times/11)
        cache = memory_mask_cache(synthetic_window())
        records = rgb.replay_joint([w], cache)
        online = next(r for r in records if r["method"] == "online" and r["K"] == 9)
        expected = rgb.joint_errors(w, np.full(29, .08), np.full(29, 4), online["selected_indices"])
        self.assertEqual(online["area_mean_e_S"], expected["area_mean_e_S"])
        self.assertEqual(online["area_mean_e_total"], expected["area_mean_e_total"])
        self.assertEqual(online["observed_RGB_indices"], online["selected_indices"][:-1])
        self.assertEqual(online["actual_K"], 9)
        self.assertEqual(online["warmup_K"], 3)

    def test_locked_and_TRAIN_phenotype_fields_not_parsed(self):
        dev = synthetic_rows(synthetic_window())
        denied = []
        for i, role in enumerate(("TRAIN", "C_POLICY", "T_ID", "T_GENO", "T_EXT")):
            rows = synthetic_rows(synthetic_window(plant=f"locked{i}::r1", role=role))
            for r in rows:
                for k in ("width", "height", "area_fraction", "n_instances", "unknown_pixels",
                          "union_pixels", "timestamp", "mask_status", "rgb_path", "mask_path"):
                    r[k] = "POISON_DO_NOT_PARSE"
            denied.extend(rows)
        selected, counts = rgb.eligible_inventory(dev+denied)
        self.assertEqual(selected, sorted(dev, key=lambda r: (r["split_role"], r["genotype"], r["biological_plant_id"], r["timestamp"], r["image_id"])))
        self.assertEqual(counts["T_GENO"], 29)
        for role in ("TRAIN", "C_POLICY", "T_ID", "T_GENO", "T_EXT"):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                rgb.replay_joint([synthetic_window(role=role)], {})

    def test_cross_role_identity_and_duplicate_rejected(self):
        rows = synthetic_rows(synthetic_window())
        with self.assertRaisesRegex(ValueError, "cross_role"):
            rgb.eligible_inventory(rows+[dict(rows[0], split_role="T_ID")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rgb.eligible_inventory(rows+[dict(rows[0])])

    def test_incomplete_window_never_shrunk_or_rescued(self):
        for times in (np.arange(0, 145, 6), np.arange(0, 169, 24), [0, 6, 12]+list(range(42, 169, 6))):
            w = synthetic_window(times=times)
            records = rgb.replay_joint([w], memory_mask_cache(w))
            self.assertEqual(len(records), 82)
            self.assertTrue(all(r["status"] == "failed" for r in records))
            self.assertTrue(all(r["N"] == len(times) for r in records))

    def test_known_times_only_fixed_schedules_invariant_to_all_phenotypes(self):
        w, other = synthetic_window(), synthetic_window()
        other.area[:] = np.linspace(0, 1, 29)
        other.leaves[:] = np.arange(29)*100
        for k in gt.BUDGETS:
            for method in ("U_-1/3", "U_0", "U_+1/3", "C", "J"):
                np.testing.assert_array_equal(rgb.select_schedule(w, k, method, None),
                                              rgb.select_schedule(other, k, method, None))


class MeasurementCacheTests(CPUOnlyTest):
    def test_empty_measurements_saved_zero_not_dropped(self):
        w = synthetic_window()
        cache = memory_mask_cache(w, pixels=0, leaves=0)
        self.assertEqual(len(cache), len(w.times))
        text = csv_text(list(cache.values()), rgb.FIELDS)
        parsed = list(csv.DictReader(io.StringIO(text)))
        verified = rgb.verify_cache_records(parsed, synthetic_rows(w), CONFIG_HASH, PROVENANCE)
        self.assertEqual(len(verified), 29)
        self.assertTrue(all(r["model_area_fraction"] == 0 and r["n_pred_leaves"] == 0 and r["union_pixels"] == 0
                            for r in verified.values()))

    def test_soft_restore_then_threshold_not_binary_resize(self):
        output = {"scores": torch.tensor([.5]), "masks": torch.tensor([[[[.49, .51]]]])}
        result = rgb.measure_output(output, 3, 1)
        self.assertEqual(result["union_pixels"], 2)  # Center .5 is retained after soft interpolation.
        low = rgb.measure_output({"scores": torch.tensor([.499]), "masks": output["masks"]}, 3, 1)
        self.assertEqual(low["n_pred_leaves"], 0)
        self.assertEqual(low["union_pixels"], 0)

    def test_output_nonfinite_shape_and_max100_refused(self):
        for out in ({"scores": torch.tensor([1.]), "masks": torch.full((1, 1, 2, 2), math.nan)},
                    {"scores": torch.ones(101), "masks": torch.zeros(101, 1, 2, 2)},
                    {"scores": torch.ones(1), "masks": torch.zeros(1, 2, 2)}):
            with self.assertRaises(ValueError):
                rgb.measure_output(out, 2, 2)

    def test_cache_poison_identity_and_reference_changes_rejected(self):
        w = synthetic_window()
        record = next(iter(memory_mask_cache(w).values()))
        for field, value in (("model_area_fraction", .1), ("n_pred_leaves", 7),
                             ("split_role", "TRAIN"), ("checkpoint_sha256", "0"*64),
                             ("input_sha256", "0"*64), ("timestamp", "2026-02-01T00:00:00+00:00")):
            bad = dict(record, **{field: value})
            with self.assertRaises(ValueError):
                rgb.verify_cache_records([bad], synthetic_rows(w), CONFIG_HASH, PROVENANCE)
            # Recomputed row hash cannot defeat inventory/provenance/union checks.
            # A changed model count is instead caught by the committed cache-chain
            # boundary (tested below), not by inventing a GT count constraint.
            bad["row_sha256"] = rgb.digest({k: v for k, v in bad.items() if k != "row_sha256"})
            if field != "n_pred_leaves":
                with self.assertRaises(ValueError):
                    rgb.verify_cache_records([bad], synthetic_rows(w), CONFIG_HASH, PROVENANCE)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rgb.verify_cache_records([record, record], synthetic_rows(w), CONFIG_HASH, PROVENANCE)

    def test_missing_cache_whole_window_failed_no_effect_filter(self):
        w = synthetic_window()
        cache = memory_mask_cache(w)
        del cache[w.image_ids[10]]
        records = rgb.replay_joint([w], cache)
        self.assertEqual(len(records), 82)
        self.assertTrue(all(r["status"] == "failed" and r["failure_reason"] == "window_failed_cache_incomplete"
                            for r in records))
        self.assertTrue(all(r["N"] == 29 and r["actual_K"] == 0 for r in records))

    def memory_store(self, partial=False):
        w = synthetic_window()
        config = {"version": rgb.VERSION, "reference_kind": rgb.REFERENCE_KIND,
                  "inventory_sha256": INVENTORY_HASH, "allowed_roles": list(rgb.ROLES),
                  "provenance": PROVENANCE, "expected_frames": 29, "inference": rgb.inference_config()}
        text = rgb.canonical(config).decode()
        config_hash = hashlib.sha256(text.encode()).hexdigest()
        records = list(memory_mask_cache(w, config_hash=config_hash).values())
        if partial:
            records = records[:10]
        csv_data = csv_text(records, rgb.FIELDS)
        status = {"phase": "extracting" if partial else "complete", "verified_frames": len(records),
                  "reference_config_sha256": config_hash, "cache_chain_sha256": rgb.cache_chain(records),
                  "cache_file_sha256": hashlib.sha256(csv_data.encode()).hexdigest()}
        store = {"model_reference_config.json": text, "model_reference_config.sha256": config_hash+"\n",
                 "perframe_measurements.csv": csv_data, "status.json": rgb.canonical(status).decode(),
                 "trace.jsonl": "\n".join(json.dumps({"image_id": r["image_id"], "row_sha256": r["row_sha256"]}) for r in records)}
        return w, store

    def with_store(self, store):
        self.stack.enter_context(patch.object(rgb, "check_sources", return_value=PROVENANCE["source_hashes"]))
        self.stack.enter_context(patch.object(rgb, "sha", side_effect=lambda p: hashlib.sha256(store[Path(p).name].encode()).hexdigest()))
        self.stack.enter_context(patch.object(rgb, "read_json", side_effect=lambda p: json.loads(store[Path(p).name])))
        self.stack.enter_context(patch.object(rgb, "read_csv", side_effect=lambda p: list(csv.DictReader(io.StringIO(store[Path(p).name])))))
        self.stack.enter_context(patch.object(Path, "read_text", lambda p, *a, **kw: store[p.name]))

    def test_hash_verified_prefix_resume_and_complete_missing_rejected(self):
        w, store = self.memory_store(partial=True)
        self.with_store(store)
        cache, _, _ = rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH, complete=False)
        self.assertEqual(len(cache), 10)
        with self.assertRaisesRegex(ValueError, "not_complete"):
            rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH)

    def test_complete_cache_verified_and_tail_trace_status_poison_rejected(self):
        w, store = self.memory_store()
        self.with_store(store)
        self.assertEqual(len(rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH)[0]), 29)
        original = store["status.json"]
        bad = json.loads(original)
        bad["cache_chain_sha256"] = "0"*64
        store["status.json"] = json.dumps(bad)
        with self.assertRaisesRegex(ValueError, "chain"):
            rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH, complete=False)
        store["status.json"] = original
        store["trace.jsonl"] += '\n{"image_id":"extra","row_sha256":"bad"}'
        with self.assertRaisesRegex(ValueError, "trace"):
            rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH)

    def test_changed_count_with_recomputed_row_hash_fails_committed_chain(self):
        w, store = self.memory_store()
        rows = list(csv.DictReader(io.StringIO(store["perframe_measurements.csv"])))
        normalized = [rgb.normalized_measurement(r) for r in rows]
        normalized[0]["n_pred_leaves"] = 7
        normalized[0]["row_sha256"] = rgb.digest({k: v for k, v in normalized[0].items() if k != "row_sha256"})
        store["perframe_measurements.csv"] = csv_text(normalized, rgb.FIELDS)
        self.with_store(store)
        with self.assertRaisesRegex(ValueError, "chain"):
            rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH, complete=False)

    def test_config_poison_rejected_before_measurements(self):
        w, store = self.memory_store()
        store["model_reference_config.json"] += " "
        self.with_store(store)
        with self.assertRaisesRegex(ValueError, "config_hash"):
            rgb.load_cache(FAKE_ROOT, synthetic_rows(w), INVENTORY_HASH)


class ManifestStatisticsTests(CPUOnlyTest):
    def test_original_grid_a_b_and_GT_values_roundtrip(self):
        w = synthetic_window()
        manifest = audit_manifest(w)
        windows, failed = rgb.windows_from_manifest([manifest], synthetic_rows(w))
        self.assertFalse(failed)
        actual = windows[0]
        np.testing.assert_array_equal(w.times, actual.times)
        np.testing.assert_array_equal(w.area, actual.area)
        np.testing.assert_array_equal(w.leaves, actual.leaves)
        self.assertEqual(w.image_ids, actual.image_ids)
        bad = copy.deepcopy(manifest)
        entries = json.loads(bad["row_times_and_errors"])
        del entries[9]
        bad["row_times_and_errors"] = json.dumps(entries)
        with self.assertRaisesRegex(ValueError, "manifest_incomplete"):
            rgb.windows_from_manifest([bad], synthetic_rows(w))
        rows = synthetic_rows(w)
        rows[5]["timestamp"] = rows[6]["timestamp"]
        with self.assertRaisesRegex(ValueError, "timestamp_changed"):
            rgb.windows_from_manifest([manifest], rows)

    def test_excluded_and_locked_original_manifests_not_rewindowed(self):
        w = synthetic_window()
        failed = audit_manifest(w)
        failed["window_eligible"] = "False"
        failed["exclusion_reasons"] = '["identity_failure"]'
        locked = {"split_role": "T_GENO", "row_times_and_errors": "POISON"}
        windows, exclusions = rgb.windows_from_manifest([failed, locked], synthetic_rows(w))
        self.assertFalse(windows)
        self.assertEqual(len(exclusions), 1)
        self.assertIn("identity", exclusions[0]["failure_reason"])

    def test_unstable_GT_dimensions_only_leaf_metrics(self):
        w = synthetic_window(area_eligible=False)
        manifest = audit_manifest(w)
        manifest["dimensions_all_GT_frames"] = "[[20,20],[40,20]]"
        windows, _ = rgb.windows_from_manifest([manifest], synthetic_rows(w))
        self.assertFalse(windows[0].area_eligible)
        bad = dict(manifest, area_eligible="True")
        with self.assertRaisesRegex(ValueError, "unstable"):
            rgb.windows_from_manifest([bad], synthetic_rows(w))

    def test_random20_collapsed_and_paired_bootstrap_whole_plants(self):
        a, b = synthetic_window(), synthetic_window("h::r2", "V_SHIFT")
        records = rgb.replay_joint([a, b], {**memory_mask_cache(a), **memory_mask_cache(b, pixels=40)})
        collapsed = rgb.collapse(records)
        random = [r for r in collapsed if r["method"] == "R"]
        self.assertEqual(len(random), 6)
        self.assertTrue(all(r["repeats"] == 20 for r in random))
        summary, paired, info = rgb.summarize(records)
        random_summary = [r for r in summary if r["scope"] == "per_K_common" and r["method"] == "R"]
        self.assertTrue(all(r["eligible_N"] == 2 for r in random_summary))
        self.assertTrue(all(r["whole_plant_N"] == 2 for r in paired if r["type"] == "paired_method_minus_baseline"))
        self.assertTrue(all(r["bootstrap_repeats"] == 2000 for r in paired))
        self.assertFalse(info["common_sets_condition_on_error"])
        self.assertTrue(info["oracle_never_in_gate"])
        self.assertEqual(info["formal_gate"], "NOT_RUN_EXPLORATORY_ONLY")
        full = next(r for r in summary if r["method"] == "fullRGB")
        self.assertAlmostEqual(full["mean_E_A_pp"], 1)
        self.assertEqual(info["across_K_common"], [a.plant_id, b.plant_id])

    def test_failures_denominator_retained_and_oracle_not_gate(self):
        w = synthetic_window()
        cache = memory_mask_cache(w)
        del cache[w.image_ids[4]]
        summary, paired, info = rgb.summarize(rgb.replay_joint([w], cache))
        for row in summary:
            self.assertEqual(row["eligible_N"], 1)
            self.assertEqual(row["failure_rate"], 1)
        self.assertEqual(info["across_K_common"], [])
        self.assertTrue(all(r["whole_plant_N"] == 0 for r in paired))

    def test_abs_metrics_averaged_before_signed_random_aggregation(self):
        rows = [{"plant_id": "p", "K": 5, "method": "R", "status": "success",
                 "area_mean_e_total": sign, "area_mean_e_total_abs": abs(sign), "E_A_pp": 100*abs(sign)}
                for sign in (-.01, .01)]
        r = rgb.collapse(rows)[0]
        self.assertEqual(r["area_mean_e_total"], 0)
        self.assertEqual(r["area_mean_e_total_abs"], .01)
        self.assertEqual(r["E_A_pp"], 1)

    def test_plot_png_pdf_only_in_memory(self):
        import matplotlib.figure
        w = synthetic_window()
        summary, _, _ = rgb.summarize(rgb.replay_joint([w], memory_mask_cache(w, pixels=40)))
        outputs = {}
        save = matplotlib.figure.Figure.savefig
        def in_memory(fig, path, **kwargs):
            stream = io.BytesIO()
            save(fig, stream, format=Path(path).suffix[1:], **kwargs)
            outputs[Path(path).suffix] = stream.getvalue()
        with patch.object(matplotlib.figure.Figure, "savefig", in_memory):
            rgb.make_plot(FAKE_ROOT, summary)
        self.assertTrue(outputs[".png"].startswith(b"\x89PNG"))
        self.assertTrue(outputs[".pdf"].startswith(b"%PDF"))


class FrozenModelAndCLITests(CPUOnlyTest):
    def config(self):
        sources = {str(ROOT/"src"/name): {"sha256": rgb.SOURCE_PINS[name],
                   "source": (ROOT/"src"/name).read_text(encoding="utf-8")}
                   for name in ("mva_model.py", "mva_pilot.py")}
        return {"version": rgb.MODEL_VERSION, "method": "B1", "schedule_steps": 3000,
            "steps_requested": 3000, "seed": 43, "precision": "FP32", "smoke": False,
            "diagnostic_not_experiment": False, "dev_max_frames_per_plant": 0,
            "inventory_sha256": INVENTORY_HASH, "evaluation_roles": list(rgb.ROLES), "sampler_roles": ["TRAIN"],
            "geometry": {"shorts": [512, 576, 640], "max_size": 1333, "eval_short": 640,
                         "train_transform_skip_resize": True, "padding": 32},
            "effective_base_images": 8, "views_per_base": 2, "eval_interval": 500,
            "optimizer": {"type": "SGD", "lr": .005, "momentum": .9, "weight_decay": 1e-4, "gradient_clip": 5},
            "versions": {k: rgb.importlib.metadata.version(package) for k, package in
                         (("torch", "torch"), ("torchvision", "torchvision"), ("numpy", "numpy"),
                          ("scipy", "scipy"), ("PIL", "Pillow"))},
            "sources": sources, "coco_weights": str(FAKE_ROOT/"coco.pt"), "coco_weight_sha256": "a"*64,
            "coco_source": "torchvision MaskRCNN_ResNet50_FPN_Weights.COCO_V1 (public generic COCO)"}

    def report(self, step, score):
        g = {"frames": 1, "AP": score, "mask_threshold": .5, "count_score_threshold": .5,
             "ap_score_threshold": .001, "max_detections": 100}
        return {"optimizer_step": step, "diagnostic_subset": False, "dev_max_frames_per_plant": 0,
                "all_instances_kept": True, "per_genotype": {role: {"g": g} for role in rgb.ROLES},
                "role_macro_AP": {role: score for role in rgb.ROLES}, "dev_score": score,
                "eval_stream_sha256": "stream"}

    def fixture(self):
        config = self.config()
        run = FAKE_ROOT/"B1"
        best = {"step": 1000, "score": 5., "diagnostic_subset": False,
                "checkpoint": str(run/"checkpoint_001000.pt")}
        best_fields = {k: best[k] for k in ("step", "score", "diagnostic_subset")}
        def cp(step):
            return {"version": rgb.MODEL_VERSION, "optimizer_step": step, "next_optimizer_step": step+1,
                    "evaluation_complete": True, "model": {"weight": torch.tensor([.2])},
                    "config": config, "best": best_fields, "sampler": {"draws": step*8}}
        checkpoints = {1000: cp(1000), 3000: cp(3000)}
        objects = {"run_config.json": config, "best.json": best,
            "latest.json": {"checkpoint": str(run/"checkpoint_003000.pt"), "next_optimizer_step": 3001, "evaluation_complete": True},
            "status.json": {"phase": "complete", "optimizer_step": 3000, "diagnostic": False, "best": best_fields}}
        for step, score in zip(range(500, 3001, 500), (1, 5, 2, 3, 4, 2)):
            objects[f"dev_{step:06d}.json"] = self.report(step, score)
        return run, objects, checkpoints

    def with_fixture(self, objects, checkpoints):
        self.stack.enter_context(patch.object(rgb, "check_sources", return_value=PROVENANCE["source_hashes"]))
        self.stack.enter_context(patch.object(rgb, "read_json", side_effect=lambda p: objects[Path(p).name]))
        self.stack.enter_context(patch.object(rgb, "sha", return_value="a"*64))
        self.stack.enter_context(patch.object(Path, "is_file", return_value=True))
        fake = types.SimpleNamespace(COCO_V1_SHA256="a"*64)
        self.stack.enter_context(patch.dict(sys.modules, {"src.mva_model": fake}))
        loads = self.stack.enter_context(patch.object(torch, "load", side_effect=lambda p, **kw:
                                                     checkpoints[int(Path(p).stem.split("_")[-1])]))
        return loads

    def test_complete3000_best1000_not_latest_frozen_trusted_local_only(self):
        run, objects, checkpoints = self.fixture()
        loads = self.with_fixture(objects, checkpoints)
        cp, _, provenance = rgb.frozen_checkpoint(run, INVENTORY_HASH, expected_frames=2, expected_stream_hash="stream")
        self.assertIs(cp, checkpoints[1000])
        self.assertEqual(provenance["completed_training_updates"], 3000)
        self.assertEqual(provenance["best_step"], 1000)
        self.assertEqual(loads.call_count, 2)
        self.assertTrue(all(c.kwargs == {"map_location": "cpu", "weights_only": False} for c in loads.call_args_list))

    def test_partial_running_final_eval_false_rejected(self):
        run, objects, checkpoints = self.fixture()
        self.with_fixture(objects, checkpoints)
        for field, value in (("phase", "training_update"), ("optimizer_step", 1000)):
            old = objects["status.json"][field]
            objects["status.json"][field] = value
            with self.assertRaisesRegex(ValueError, "not_final"):
                rgb.frozen_checkpoint(run, INVENTORY_HASH, 2)
            objects["status.json"][field] = old
        objects["latest.json"]["evaluation_complete"] = False
        with self.assertRaisesRegex(ValueError, "latest"):
            rgb.frozen_checkpoint(run, INVENTORY_HASH, 2)
        objects["latest.json"]["evaluation_complete"] = True
        checkpoints[3000]["evaluation_complete"] = False
        with self.assertRaisesRegex(ValueError, "partial"):
            rgb.frozen_checkpoint(run, INVENTORY_HASH, 2)

    def test_wrong_source_method_budget_geometry_best_or_stream_rejected(self):
        config = self.config()
        rgb.validate_run_config(config, INVENTORY_HASH)
        for field, value in (("method", "A5"), ("steps_requested", 1000), ("precision", "AMP"),
                             ("dev_max_frames_per_plant", 8), ("inventory_sha256", "0"*64)):
            bad = copy.deepcopy(config)
            bad[field] = value
            with self.assertRaises(ValueError):
                rgb.validate_run_config(bad, INVENTORY_HASH)
        bad = copy.deepcopy(config)
        next(iter(bad["sources"].values()))["source"] += "\n# changed"
        with self.assertRaisesRegex(ValueError, "source_snapshot"):
            rgb.validate_run_config(bad, INVENTORY_HASH)
        run, objects, checkpoints = self.fixture()
        self.with_fixture(objects, checkpoints)
        objects["best.json"]["step"] = 3000
        with self.assertRaisesRegex(ValueError, "selected_best"):
            rgb.frozen_checkpoint(run, INVENTORY_HASH, 2)
        objects["best.json"]["step"] = 1000
        with self.assertRaisesRegex(ValueError, "stream"):
            rgb.frozen_checkpoint(run, INVENTORY_HASH, 2, "wrong_stream")

    def test_checkpoint_config_and_exposure_mismatch_rejected(self):
        _, objects, cps = self.fixture()
        best = objects["best.json"]
        cp = cps[1000]
        rgb.validate_checkpoint(cp, objects["run_config.json"], 1000, best)
        cp["sampler"]["draws"] = 1
        with self.assertRaisesRegex(ValueError, "exposure"):
            rgb.validate_checkpoint(cp, objects["run_config.json"], 1000, best)
        cp["sampler"]["draws"] = 8000
        cp["config"] = dict(cp["config"], method="A5")
        with self.assertRaisesRegex(ValueError, "config_mismatch"):
            rgb.validate_checkpoint(cp, objects["run_config.json"], 1000, best)

    def test_restore_factory_does_not_COCO_hash_trained_weights(self):
        class TinyB1(torch.nn.Module):
            def __init__(self, *args, **kw):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.]))
                self.transform = types.SimpleNamespace(min_size=(640,), max_size=1333)
                self.roi_heads = types.SimpleNamespace(nms_thresh=.5, detections_per_img=100, score_thresh=.001)
            def inference_model(self):
                return self
        factory_calls = []
        def factory(*args, **kw):
            factory_calls.append((args, kw))
            return TinyB1()
        with patch.dict(sys.modules, {"src.mva_model": types.SimpleNamespace(MVAModel=factory)}):
            model = rgb.restore_model({"model": {"weight": torch.tensor([.75])}}, {"seed": 43})
            self.assertEqual(model.weight.item(), .75)
            self.assertFalse(model.training)
            self.assertFalse(model.weight.requires_grad)
            self.assertEqual(model.roi_heads.score_thresh, .001)  # Factory preserved; final >= .5 is external.
            self.assertIsNone(factory_calls[0][1]["weights_path"])
            self.assertEqual(factory_calls[0][0], ("B1",))
            with self.assertRaises(RuntimeError):
                rgb.restore_model({"model": {"wrong_key": torch.tensor([.75])}}, {"seed": 43})
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                rgb.restore_model({"model": {"weight": torch.tensor([math.nan])}}, {"seed": 43})

    def test_original_queue_not_done_blocks_before_model_or_image(self):
        with patch.object(rgb, "read_json", return_value={"status": "running", "current_job": {"name": "A5"}}), \
                patch.object(rgb, "frozen_checkpoint", side_effect=AssertionError("NO_MODEL_READ")), \
                patch.object(gt, "read_inventory", side_effect=AssertionError("NO_INVENTORY_READ")):
            with self.assertRaisesRegex(ValueError, "queue_not_fully_finished"):
                rgb.extract(FAKE_ROOT/"inventory.csv", FAKE_ROOT/"B1", FAKE_ROOT/"out")

    def test_CLI_dispatch_only_and_compact_device_alias(self):
        with patch.object(rgb, "extract", return_value={"phase": "SYNTHETIC_DISPATCH_ONLY"}) as extract, \
                patch.object(rgb, "analyze", return_value={"phase": "SYNTHETIC_DISPATCH_ONLY"}) as analyze, \
                patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(rgb.main(["--action", "extract", "--model-run", "mva/B1_pilot_v1", "--out", "aea/rgb_reference_v1",
                                       "--inventory", "shared/pilot_inventory.csv", "--devicecuda"]), 0)
            self.assertEqual(extract.call_args.args[3], "cuda")
            self.assertEqual(rgb.main(["--action", "analyze", "--out", "aea/rgb_sampling_pilot_v1", "--inventory", "shared/pilot_inventory.csv",
                                       "--measurement-cache", "aea/rgb_reference_v1", "--gt-pilot", "aea/pilot_v2"]), 0)
            self.assertEqual(analyze.call_count, 1)
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(rgb.main(["--action", "analyze", "--out", "x", "--inventory", "x"]), 2)

    def test_extract_empty_outputs_and_resume_entirely_in_CPU_memory(self):
        from PIL import Image
        windows = [synthetic_window(), synthetic_window("h::r2", "V_SHIFT")]
        rows = [row for w in windows for row in synthetic_rows(w)]
        for row in rows:
            row["rgb_path"] = str(FAKE_ROOT / (row["image_id"] + ".png"))
        store = {}
        rgb_hashes = {r["rgb_path"]: r["rgb_sha256"] for r in rows}
        class MemoryStream(io.StringIO):
            def __init__(self, key, mode):
                self.key = key
                super().__init__(store.get(key, ""))
                if "a" in mode:
                    self.seek(0, 2)
            def flush(self):
                store[self.key] = self.getvalue()
            def close(self):
                self.flush()
                super().close()
        def open_memory(path, mode="r", *args, **kwargs):
            return MemoryStream(path.name, mode)
        def write_text(path, text, *args, **kwargs):
            store[path.name] = text
            return len(text)
        def hash_memory(path):
            p = Path(path)
            if str(p) in rgb_hashes:
                return rgb_hashes[str(p)]
            if p.name == "inventory.csv":
                return INVENTORY_HASH
            return hashlib.sha256(store[p.name].encode()).hexdigest()
        class MemoryModel:
            def __init__(self):
                self.calls = 0
            def to(self, device):
                if device.type != "cpu":
                    raise AssertionError("SYNTHETIC_CPU_ONLY")
                return self
            def __call__(self, images):
                self.calls += 1
                return [mask_output(0, 0)]
        model = MemoryModel()
        with ExitStack() as s:
            s.enter_context(patch.object(rgb, "EXPECTED_FRAMES", len(rows)))
            s.enter_context(patch.object(rgb, "queue_finished", return_value={"verified": "synthetic_completed_queue"}))
            s.enter_context(patch.object(rgb, "check_sources", return_value=PROVENANCE["source_hashes"]))
            s.enter_context(patch.object(gt, "read_inventory", return_value=rows))
            s.enter_context(patch.object(rgb, "frozen_checkpoint", return_value=({}, {"seed": 43}, PROVENANCE)))
            s.enter_context(patch.object(rgb, "restore_model", return_value=model))
            s.enter_context(patch.object(Path, "exists", return_value=False))
            s.enter_context(patch.object(Path, "mkdir"))
            s.enter_context(patch.object(Path, "open", open_memory))
            s.enter_context(patch.object(Path, "read_text", lambda p, *a, **k: store[p.name]))
            s.enter_context(patch.object(Path, "write_text", write_text))
            s.enter_context(patch.object(rgb, "sha", side_effect=hash_memory))
            s.enter_context(patch.object(rgb, "atomic_json", side_effect=lambda p, obj: store.update({Path(p).name: rgb.canonical(obj).decode()})))
            s.enter_context(patch.object(gt, "write_csv", side_effect=lambda p, data, fields: store.update({Path(p).name: csv_text(data, fields)})))
            s.enter_context(patch.object(Image, "open", side_effect=lambda p: Image.new("RGB", (20, 20))))
            s.enter_context(patch("sys.stdout", new=io.StringIO()))
            result = rgb.extract(FAKE_ROOT/"inventory.csv", FAKE_ROOT/"B1", FAKE_ROOT/"out", device="cpu")
            self.assertEqual(result["frames"], 58)
            self.assertEqual(model.calls, 58)
            cache, config, _ = rgb.load_cache(FAKE_ROOT/"out", rows, INVENTORY_HASH)
            self.assertTrue(all(r["model_area_fraction"] == 0 and r["n_pred_leaves"] == 0 for r in cache.values()))
            self.assertEqual(config["reference_kind"], "exploratory_cross_route_reference")
            self.assertFalse(config["formal_AEA_40epoch_measurement_completed"])
            self.assertEqual(config["inference"]["eval_short"], 640)
            again = rgb.extract(FAKE_ROOT/"inventory.csv", FAKE_ROOT/"B1", FAKE_ROOT/"out", device="cpu", resume=True)
            self.assertEqual(again["frames"], 58)
            self.assertEqual(model.calls, 58)  # No measurements rerun or effects filtered.

    def test_analyze_original_windows_and_outputs_entirely_in_memory(self):
        a, b = synthetic_window(), synthetic_window("h::r2", "V_SHIFT")
        rows = synthetic_rows(a)+synthetic_rows(b)
        cache = {**memory_mask_cache(a), **memory_mask_cache(b, pixels=40)}
        manifests = [audit_manifest(a), audit_manifest(b)]
        old_full = [{"plant_id": w.plant_id, "split_role": w.role, "method": "full",
                     "reference_mean_area_fraction": ".08", "reference_mean_leaves": "4"} for w in (a, b)]
        store = {}
        csv_outputs = {}
        def write_text(p, text, *args, **kwargs):
            store[p.name] = text
            return len(text)
        def read_json(p):
            if p.name == "config.json":
                return {"inventory_sha256": INVENTORY_HASH, "source_sha256": rgb.SOURCE_PINS["aea_sampling.py"]}
            return {"status": "completed_provisional_GT_development_pilot"}
        with ExitStack() as s:
            s.enter_context(patch.object(rgb, "check_sources", return_value=PROVENANCE["source_hashes"]))
            s.enter_context(patch.object(Path, "exists", return_value=False))
            s.enter_context(patch.object(Path, "mkdir"))
            s.enter_context(patch.object(Path, "write_text", write_text))
            s.enter_context(patch.object(rgb, "sha", side_effect=lambda p: rgb.SOURCE_PINS["aea_sampling.py"] if Path(p).name == "source_code_snapshot.py" else INVENTORY_HASH))
            s.enter_context(patch.object(rgb, "read_json", side_effect=read_json))
            s.enter_context(patch.object(gt, "read_inventory", return_value=rows))
            s.enter_context(patch.object(rgb, "load_cache", return_value=(cache, {"provenance": PROVENANCE}, CONFIG_HASH)))
            s.enter_context(patch.object(rgb, "read_csv", side_effect=lambda p: manifests if p.name == "window_audit.csv" else old_full))
            s.enter_context(patch.object(gt, "evaluate", side_effect=AssertionError("WRONG_REFERENCE_EVALUATOR")))
            s.enter_context(patch.object(gt, "write_csv", side_effect=lambda p, data, fields: csv_outputs.update({p.name: data})))
            s.enter_context(patch.object(rgb, "atomic_json", side_effect=lambda p, obj: store.update({p.name: obj})))
            plotted = s.enter_context(patch.object(rgb, "make_plot"))
            result = rgb.analyze(FAKE_ROOT/"inventory.csv", FAKE_ROOT/"cache", FAKE_ROOT/"GT_pilot", FAKE_ROOT/"out")
        self.assertEqual(result["windows"], 2)
        self.assertEqual(len(csv_outputs["per_plant_joint_errors.csv"]), 164)
        self.assertIn("summary.csv", csv_outputs)
        self.assertIn("paired_bootstrap.csv", csv_outputs)
        self.assertIn("phase_result.txt", store)
        self.assertIn("trace.json", store)
        self.assertEqual(store["status.json"]["formal_measurement_and_external_validation"], "pending")
        self.assertEqual(plotted.call_count, 1)
        self.assertTrue(all(r["bootstrap_repeats"] == 2000 for r in csv_outputs["paired_bootstrap.csv"]))

    def test_protected_sources_unchanged_actual_code_hashes(self):
        self.assertEqual({k: rgb.sha(ROOT/"src"/k) for k in rgb.SOURCE_PINS}, rgb.SOURCE_PINS)
        self.assertIn("aea_rgb_pilot.py", rgb.check_sources())


if __name__ == "__main__":
    unittest.main(verbosity=2)

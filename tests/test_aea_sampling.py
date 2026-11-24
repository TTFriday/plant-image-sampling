"""Synthetic-only tests. No inventory, real images, or output files are accessed/written."""
import sys
sys.dont_write_bytecode = True

import importlib.util
import io
import itertools
import json
import builtins
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "src" / "aea_sampling.py"
spec = importlib.util.spec_from_file_location("aea_sampling", SOURCE)
aea = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = aea
spec.loader.exec_module(aea)


def synthetic_window(times=None, plant="synthetic::r1", role="TRAIN", area_eligible=True):
    t = np.arange(0, 169, 6, dtype=float) if times is None else np.asarray(times, dtype=float)
    area = .08 + .0001 * t + .004 * np.sin(t / 17)
    leaves = 4 + .007 * t + .1 * np.cos(t / 19)
    return aea.Window(plant, plant, plant.split("::")[0], plant.split("::")[-1], role,
                      datetime(2026, 1, 1, tzinfo=timezone.utc), t, area, leaves,
                      [f"synthetic_{plant}_{i}" for i in range(len(t))], area_eligible, 100, 100, 0)


def synthetic_rows(plant="g::1", times=None, role="TRAIN", width=100):
    times = list(range(0, 169, 6)) if times is None else list(times)
    genotype, rep = plant.split("::")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index, hour in enumerate(times):
        rgb_hash = aea.hashlib.sha256(f"synthetic_RGB|{plant}|{index}".encode()).hexdigest()
        mask_hash = aea.hashlib.sha256(f"synthetic_mask|{plant}|{index}".encode()).hexdigest()
        rows.append(dict(zip(aea.SCHEMA, [
            f"synthetic_{plant}_{index}", f"bio_{plant}", genotype, rep,
            (start + timedelta(hours=hour)).isoformat(), "synthetic_rgb.png", "synthetic_mask.png",
            str(width), "100", rgb_hash, mask_hash, "valid", "0", "4", "800", str(800 / (width * 100)),
            "synthetic_label.txt", role,
        ])))
    return rows


def audited_metadata(row, root):
    # Only synthetic audit tests use this replacement; production never bypasses validation.
    if row["mask_status"] != "valid":
        return None, ["mask_not_valid"], False
    gt = {"width": int(row["width"]), "height": int(row["height"]),
          "area": float(row["area_fraction"]), "leaves": int(row["n_instances"]), "rgb_bytes": 0}
    return gt, ([] if row["rgb_path"] else ["rgb_missing"]), bool(row["rgb_path"])


class ProjectionTests(unittest.TestCase):
    def test_dp_brute_equivalence(self):
        rng = np.random.default_rng(334)
        for _ in range(180):
            n = int(rng.integers(4, 10))
            t = np.r_[0, np.cumsum(rng.integers(1, 12, size=n - 1))].astype(float)
            m = int(rng.integers(1, n - 1))
            q = np.sort(rng.uniform(t[0], t[-1], size=m))
            tolerance = float(rng.integers(1, 15))
            candidates = []
            for selected in itertools.combinations(range(1, n - 1), m):
                distances = np.abs(t[list(selected)] - q)
                if np.all(distances <= tolerance + 1e-10):
                    candidates.append((float(np.sum(distances)), selected))
            if not candidates:
                with self.assertRaises(aea.ScheduleFailure):
                    aea.project_targets(t, q, tolerance)
            else:
                expected = min(candidates, key=lambda pair: (pair[0], pair[1]))[1]
                actual = aea.project_targets(t, q, tolerance)
                self.assertEqual(tuple(actual[1:-1]), expected)
                self.assertEqual(len(actual), m + 2)
                self.assertTrue(np.all(np.diff(actual) > 0))

    def test_earliest_tie_and_duplicate_targets_exact_K(self):
        self.assertEqual(aea.project_targets([0, 2, 4, 6], [3], 1).tolist(), [0, 1, 3])
        self.assertEqual(aea.project_targets([0, 2, 4, 6, 8], [3, 3], 3).tolist(), [0, 1, 2, 4])
        for k in (5, 9, 15):
            s = aea.index_schedule(np.arange(15), k)
            self.assertEqual(len(s), k)
            self.assertEqual(len(set(s)), k)

    def test_duplicate_observations_and_failures(self):
        for times, targets, tolerance in (([0, 6, 6, 12], [6], 6),
                                           ([0, 30, 60], [6], 12),
                                           ([0, 6, 12], [2, 4], 12),
                                           ([0, 6, 12, 18], [12, 6], 12)):
            with self.assertRaises(aea.ScheduleFailure):
                aea.project_targets(times, targets, tolerance)
        with self.assertRaises(aea.ScheduleFailure):
            aea.random_schedule([0, 6, 12], 5, 43)

    def test_all_projected_schedules_within_tolerance(self):
        w = synthetic_window()
        for k in aea.BUDGETS:
            for phi in aea.PHASES:
                s = aea.uniform_schedule(w, k, phi)
                targets = (np.arange(1, k - 1) + phi) * w.length / (k - 1)
                self.assertLessEqual(np.max(np.abs(w.times[s[1:-1]] - targets)), w.tolerance + 1e-10)
                self.assertEqual(len(set(s)), k)
            clock = aea.clock_schedule(w, k)
            self.assertEqual(len(set(clock)), k)
            self.assertGreater(clock[1], 0)
            self.assertLess(clock[-2], len(w.times) - 1)


class ErrorAndCausalityTests(unittest.TestCase):
    def test_irregular_trapezoid_not_frame_average(self):
        t, y = [0, 1, 4], [0, 2, 2]
        self.assertEqual(aea.trapezoid(t, y), 7)
        self.assertEqual(aea.trapezoid(t, y) / 4, 1.75)
        self.assertNotEqual(aea.trapezoid(t, y) / 4, np.mean(y))

    def test_full_GT_error_zero_and_low_denominator(self):
        w = synthetic_window()
        result = aea.evaluate(w, np.arange(len(w.times)))
        self.assertEqual(result["E_A_pp"], 0)
        self.assertEqual(result["E_N_leaves"], 0)
        self.assertEqual(result["joint_z"], 0)
        w.area[:] = .0009
        self.assertTrue(math.isnan(aea.evaluate(w, [0, 7, len(w.times) - 1])["relative_area_integral_pct"]))
        self.assertEqual(aea.evaluate(w, np.arange(len(w.times)))["E_A_pp"], 0)
        w.area[:] = .001
        self.assertFalse(math.isnan(aea.evaluate(w, np.arange(len(w.times)))["relative_area_integral_pct"]))
        w.area[:] = 0
        self.assertTrue(math.isnan(aea.evaluate(w, np.arange(len(w.times)))["relative_area_integral_pct"]))

    def test_oracle_all_N_zero_and_candidate_upper_bound(self):
        w = synthetic_window(np.arange(0, 169, 12))
        s, count = aea.oracle_schedule(w, len(w.times), [])
        self.assertEqual(s.tolist(), list(range(len(w.times))))
        self.assertEqual(aea.evaluate(w, s)["joint_z"], 0)
        self.assertGreater(count, 1)
        candidates = [aea.uniform_schedule(w, 5, phase) for phase in aea.PHASES]
        s, _ = aea.oracle_schedule(w, 5, candidates)
        self.assertLessEqual(aea.evaluate(w, s)["joint_z"], min(aea.evaluate(w, x)["joint_z"] for x in candidates))

    def test_online_does_not_read_future_phenotypes(self):
        times = np.arange(0, 169, 6, dtype=float)
        original = np.column_stack((np.full(len(times), .08), np.full(len(times), 4.0)))
        modified = original.copy()
        modified[times >= 84] = (.9, 60)
        calls = []
        def observe(i):
            calls.append(i)
            return original[i]
        first = aea.online_schedule(times, 9, observe, 6)
        second = aea.online_schedule(times, 9, lambda i: modified[i], 6)
        # The first newly changed measurement is selected before it is read too.
        cutoff = next(i for i, selected in enumerate(first) if times[selected] >= 84)
        np.testing.assert_array_equal(first[:cutoff + 1], second[:cutoff + 1])
        self.assertEqual(calls, first[:-1].tolist())
        self.assertEqual(first[:3].tolist(), [0, 1, 2])
        self.assertEqual(len(set(first)), 9)
        self.assertEqual(first[-1], len(times) - 1)

    def test_online_reserves_budget_on_legal_nonuniform_full_schedule(self):
        # Regression: forced earlier opportunities preserve remaining budget;
        # they are not subject to the offline nearest-time projection tolerance.
        times = np.r_[np.arange(0, 49, 6), np.arange(66, 157, 18)]
        selected = aea.online_schedule(times, len(times), lambda i: (.02, 6), 6)
        np.testing.assert_array_equal(selected, np.arange(len(times)))

    def test_online_failure_not_silently_rebudgeted(self):
        with self.assertRaises(aea.ScheduleFailure):
            aea.online_schedule([0, 18, 36, 54, 168], 5, lambda i: (.1, 4), 6)
        with self.assertRaises(aea.ScheduleFailure):
            aea.online_schedule([0, 6, 12, 168], 5, lambda i: (.1, 4), 6)


class AuditTests(unittest.TestCase):
    def audit(self, rows):
        with patch.object(aea, "check_measurement", side_effect=audited_metadata):
            return aea.audit_windows(rows, Path("./synthetic_not_on_disk"))

    def test_actual_calendar_elapsed_and_first_window_only(self):
        rows = synthetic_rows(times=range(0, 241, 12))
        audit, windows, counts = self.audit(rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].times[-1], 168)
        self.assertEqual(audit[0]["N"], 15)
        self.assertEqual(counts["window_eligible_plants"], 1)
        self.assertEqual(audit[0]["row_times_and_errors"][-1]["elapsed_h"], 240)
        a, naive = aea.parse_timestamp("2026-01-01T10:00:00+02:00")
        b, _ = aea.parse_timestamp("2026-01-01T09:00:00+00:00")
        self.assertEqual((b - a).total_seconds() / 3600, 1)
        self.assertFalse(naive)
        self.assertTrue(aea.parse_timestamp("20260101_100000")[1])
        with self.assertRaises(ValueError):
            aea.parse_timestamp("42")

    def test_strict_failure_no_later_window_search(self):
        audit, windows, _ = self.audit(synthetic_rows(times=[0] + list(range(30, 241, 6))))
        self.assertEqual(windows, [])
        self.assertIn("max_dt_above_18h", audit[0]["exclusion_reasons"])
        audit, windows, _ = self.audit(synthetic_rows(times=range(0, 145, 12)))
        self.assertEqual(windows, [])
        self.assertIn("L_outside_156_168h", audit[0]["exclusion_reasons"])

    def test_duplicate_exact_time_dedup_and_conflict(self):
        rows = synthetic_rows()
        rows.append(dict(rows[4]))
        audit, windows, counts = self.audit(rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0].times), 29)
        self.assertEqual(counts["duplicate_timestamps"], 1)
        self.assertEqual(sum(r["duplicate"] for r in audit[0]["row_times_and_errors"]), 1)
        rows[-1]["rgb_sha256"] = "f" * 64
        audit, windows, _ = self.audit(rows)
        self.assertEqual(windows, [])
        self.assertIn("conflicting_same_timestamp", audit[0]["exclusion_reasons"])

    def test_unstable_dimensions_leaf_only_and_provisional(self):
        rows = synthetic_rows()
        rows[5]["width"] = "200"
        rows[5]["area_fraction"] = ".04"
        audit, windows, _ = self.audit(rows)
        self.assertEqual(len(windows), 1)
        self.assertFalse(windows[0].area_eligible)
        self.assertFalse(audit[0]["area_eligible"])
        self.assertIn("inconsistent", audit[0]["area_exclusion_reason"])
        errors = aea.evaluate(windows[0], [0, 7, 14, 21, 28])
        self.assertTrue(math.isnan(errors["E_A_pp"]))
        self.assertFalse(math.isnan(errors["E_N_leaves"]))
        self.assertIn("provisional", audit[0]["roi_status"])

    def test_locked_roles_excluded_and_crossrole_identity_blocked(self):
        rows = synthetic_rows("dev::1") + synthetic_rows("locked::1", role="T_GENO")
        # Poison all locked phenotype fields: must never be parsed/validated.
        for row in rows[29:]:
            row["area_fraction"] = "DO_NOT_READ_LOCKED_PHENOTYPE"
        audit, windows, counts = self.audit(rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(audit), 1)
        self.assertEqual(counts["role_rows"]["T_GENO"], 29)
        duplicate = dict(rows[0], split_role="C_POLICY")
        _, windows, _ = self.audit(rows + [duplicate])
        self.assertEqual(windows, [])

    def test_joint_grid_GT_only_coverage_and_row_errors(self):
        rows = synthetic_rows()
        rows[1]["rgb_path"] = ""
        rows[3]["timestamp"] = "invalid"
        audit, windows, _ = self.audit(rows)
        self.assertEqual(len(windows), 1)
        self.assertEqual(audit[0]["gt_only_N_window"], 28)
        self.assertEqual(audit[0]["N"], 27)
        self.assertIn("unparseable_calendar_timestamp", audit[0]["row_times_and_errors"][3]["errors"])
        self.assertEqual(audit[0]["actual_capture_90pct"], "not_claimed")

    def test_measurement_schema_checks_without_images(self):
        row = synthetic_rows()[0]
        with patch.object(Path, "is_file", return_value=True):
            invalid = dict(row, unknown_pixels="1")
            gt, errors, rgb_ok = aea.check_measurement(invalid, Path("./synthetic"))
            self.assertIsNone(gt)
            self.assertIn("unknown_mask_pixels", errors)
            invalid = dict(row, area_fraction=".8")
            self.assertIsNone(aea.check_measurement(invalid, Path("./synthetic"))[0])
            invalid = dict(row, n_instances="4.2")
            self.assertIsNone(aea.check_measurement(invalid, Path("./synthetic"))[0])


class ReplayStatisticsModelTests(unittest.TestCase):
    def test_82_units_exact_K_full_zero_and_seeds(self):
        records, seeds = aea.replay([synthetic_window()])
        self.assertEqual(len(records), 82)
        for k in aea.BUDGETS:
            batch = [r for r in records if r["K"] == k and r["method"] != "full"]
            self.assertEqual(len(batch), 27)
            self.assertEqual(sum(r["method"] == "R" for r in batch), 20)
            for r in batch:
                if r["status"] == "success":
                    self.assertEqual(r["actual_K"], k)
                    self.assertEqual(len(set(r["selected_indices"])), k)
        self.assertEqual(records[-1]["E_A_pp"], 0)
        self.assertEqual(records[-1]["E_N_leaves"], 0)
        self.assertEqual(len(seeds[records[0]["plant_id"]]), 60)
        self.assertEqual(aea.plant_seed("a", 5, 0), aea.plant_seed("a", 5, 0))
        self.assertNotEqual(aea.plant_seed("a", 5, 0), aea.plant_seed("b", 5, 0))
        oracle = next(r for r in records if r["method"] == "oracle")
        self.assertIn("NOT_DEPLOYABLE_NOT_GLOBAL_OPTIMAL", oracle["deployability"])

    def test_online_irregular_gap_charged_not_rejected_by_offline_tolerance(self):
        t = np.r_[np.arange(0, 85, 6), np.arange(102, 169, 6)]
        w = synthetic_window(t)
        w.area[:] = .08
        w.leaves[:] = 4
        records, _ = aea.replay([w])
        result = next(r for r in records if r["method"] == "online" and r["K"] == 5)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actual_K"], 5)
        self.assertEqual(result["selected_elapsed_h"], [0, 6, 12, 102, 168])
        self.assertEqual(len(records), 82)
        with self.assertRaisesRegex(ValueError, "forbidden_replay_role"):
            aea.replay([synthetic_window(role="T_ID")])

    def test_common_feasibility_failure_retained_random_is_one_plant(self):
        records, _ = aea.replay([synthetic_window(plant="p::1"), synthetic_window(plant="p::2")])
        target = next(r for r in records if r["plant_id"] == "p::2" and r["K"] == 9 and r["method"] == "C")
        target["status"] = "failed"
        target["failure_reason"] = "synthetic_forced_projection_failure"
        summary, groups, info = aea.summarize(records)
        self.assertEqual(len(info["area_common_by_K"]["9"]), 1)
        self.assertEqual(len(info["area_across_K_intersection"]), 1)
        r = next(r for r in summary if r["K"] == 9 and r["method"] == "R" and r["scope"] == "per_K_common")
        self.assertEqual(r["n_attempted_plants"], 2)
        self.assertEqual(r["n_area_common"], 1)
        c = next(r for r in summary if r["K"] == 9 and r["method"] == "C" and r["scope"] == "per_K_common")
        self.assertEqual(c["schedule_failure_rate"], .5)
        self.assertLessEqual(c["joint_pass_rate_all_area_eligible"], .5)
        self.assertEqual(len([r for r in aea.collapse_repeats(records) if r["method"] == "R"]), 6)
        self.assertEqual(len(groups), len(summary))
        self.assertEqual(aea.p1_gates(summary, info)["decision"], "insufficient_development_common_plants")

    def test_oracle_cannot_rescue_P1_gate(self):
        fake = [{"scope": "per_K_common", "K": 5, "method": "oracle", "n_area_common": 100,
                 "joint_pass_rate_common": 1, "joint_pass_rate_all_area_eligible": 1,
                 "empirical_q90_joint_z_common": 0, "paired_mean_method_minus_U0_pp": -10}]
        gates = aea.p1_gates(fake, {"phase_interactions": []})
        self.assertFalse(gates["nonfull_90pct_gate"])
        self.assertTrue(gates["oracle_excluded"])

    def test_two_real_CPU_quantile_fits_no_locked_leakage(self):
        windows, records = [], []
        for role, n in (("TRAIN", 10), ("V_SOURCE", 5), ("V_SHIFT", 5)):
            for i in range(n):
                w = synthetic_window(plant=f"{role}::s{i}", role=role)
                windows.append(w)
                for k in aea.BUDGETS:
                    for phase, method in zip(aea.PHASES, aea.METHOD_ORDER[:3]):
                        records.append({"plant_id": w.plant_id, "split_role": role, "K": k, "L_h": 168,
                                        "phase": phase, "method": method, "status": "success", "area_eligible": True,
                                        "E_A_pp": .3 + (15 - k) * .01 + abs(phase) * .05 + i * .001,
                                        "E_N_leaves": .1 + (15 - k) * .002 + abs(phase) * .01 + i * .001})
        # A locked role could appear in an external caller's input: never training/validation.
        records.append(dict(records[0], split_role="T_GENO", plant_id="locked", E_A_pp=1e100))
        windows.append(synthetic_window(plant="locked::1", role="T_GENO"))
        result = aea.fit_quantile_models(records, windows)
        self.assertEqual(result["status"], "fit_GT_only_validation_evaluated")
        self.assertEqual(set(result["models"]), {"E_A_pp", "E_N_leaves"})
        self.assertEqual(result["train_plants"], 10)
        self.assertEqual(len(result["phase_table_development_only"]), 60)
        for endpoint in result["models"].values():
            self.assertTrue(endpoint["solver"]["success"])
            self.assertEqual(endpoint["solver"]["logunit_smoothing"], 1e-6)
            self.assertEqual(endpoint["validation"]["V_SOURCE"]["n_plants"], 5)
            self.assertEqual(endpoint["validation"]["V_SHIFT"]["n_plants"], 5)
        insufficient = aea.fit_quantile_models(records[:9], windows[:1])
        self.assertEqual(insufficient["status"], "insufficient_training_plants")
        self.assertEqual(insufficient["models"], {})


class OutputTests(unittest.TestCase):
    def test_csv_JSON_nonfinite_and_empty_headers_in_memory(self):
        class NonClosingIO(io.StringIO):
            def close(self):
                pass
        stream = NonClosingIO()
        with patch.object(Path, "open", return_value=stream):
            aea.write_csv(Path("./not_written.csv"), [{"a": math.nan, "indices": [0, 2], "nested": {"b": math.nan}}], ["a"])
        self.assertIn("NaN", stream.getvalue())
        self.assertIn("null", stream.getvalue())
        with patch.object(Path, "write_text") as writer:
            aea.write_json(Path("./not_written.json"), {"a": math.nan})
            self.assertEqual(json.loads(writer.call_args.args[0]), {"a": None})
        stream = NonClosingIO()
        with patch.object(Path, "open", return_value=stream):
            aea.write_csv(Path("./not_written.csv"), [], ["plant_id", "status"])
        self.assertEqual(stream.getvalue().strip(), "plant_id,status")

    def test_matplotlib_Agg_PNG_PDF_rendered_only_in_memory(self):
        original_open, original_io_open = builtins.open, io.open
        def memory_writes(original):
            def open_file(file, mode="r", *args, **kwargs):
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    return io.BytesIO() if "b" in mode else io.StringIO()
                return original(file, mode, *args, **kwargs)
            return open_file
        # Redirect even incidental matplotlib font-cache writes; create no files.
        with patch("builtins.open", side_effect=memory_writes(original_open)), \
             patch("io.open", side_effect=memory_writes(original_io_open)), \
             patch.object(Path, "mkdir"), patch.object(Path, "unlink"):
            import matplotlib
            matplotlib.use("Agg", force=True)
            from matplotlib.figure import Figure
            original_save = Figure.savefig
            rendered = {}
            def save_to_memory(fig, filename, **kwargs):
                buffer = io.BytesIO()
                fmt = Path(filename).suffix[1:]
                original_save(fig, buffer, format=fmt, **kwargs)
                rendered[Path(filename).name] = buffer.getvalue()
            rows = synthetic_rows()
            with patch.object(aea, "check_measurement", side_effect=audited_metadata):
                audits, windows, _ = aea.audit_windows(rows, Path("./synthetic"))
            records, _ = aea.replay(windows)
            summary, _, info = aea.summarize(records)
            with patch.object(Figure, "savefig", autospec=True, side_effect=save_to_memory):
                files = aea.make_plots(Path("./never_written"), audits, summary, records, info)
                self.assertEqual(len(files), 6)
                for name in files:
                    self.assertGreater(len(rendered[name]), 1000)
                    signature = b"%PDF" if name.endswith(".pdf") else b"\x89PNG"
                    self.assertTrue(rendered[name].startswith(signature))
                empty_files = aea.make_plots(Path("./never_written"), [], [], [],
                                            {"leaf_across_K_intersection": []})
                self.assertEqual(len(empty_files), 4)

    def run_memory(self, rows=None, input_error=False):
        csv_outputs, json_outputs, text_outputs = {}, {}, {}
        def save_csv(path, rows, required):
            csv_outputs[path.name] = rows
        def save_json(path, data):
            json_outputs[path.name] = aea.json_safe(data)
        def save_text(path, text, **kwargs):
            text_outputs[path.name] = text
            return len(text)
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "mkdir"), \
             patch.object(Path, "iterdir", side_effect=lambda: iter(())), \
             patch.object(Path, "write_text", autospec=True, side_effect=save_text), \
             patch.object(aea, "sha256_file", return_value="a" * 64), \
             patch.object(aea, "read_inventory", side_effect=ValueError("synthetic_schema_error") if input_error else None, return_value=rows or []), \
             patch.object(aea, "check_measurement", side_effect=audited_metadata), \
             patch.object(aea, "write_csv", side_effect=save_csv), \
             patch.object(aea, "write_json", side_effect=save_json), \
             patch.object(aea, "make_plots", return_value=[]):
            status = aea.run(Path("synthetic/shared/data_inventory.csv"), Path("synthetic/aea/pilot_v1"))
        return status, csv_outputs, json_outputs, text_outputs

    def test_complete_synthetic_pipeline_in_memory_no_files(self):
        status, csvs, jsons, texts = self.run_memory(synthetic_rows())
        self.assertEqual(status["status"], "completed_provisional_GT_development_pilot")
        self.assertEqual(set(csvs), {"window_audit.csv", "replay_per_plant.csv", "summary.csv", "per_genotype.csv"})
        self.assertEqual(len(csvs["replay_per_plant.csv"]), 82)
        self.assertEqual(set(jsons), {"config.json", "status.json"})
        config = jsons["config.json"]
        self.assertTrue(config["UTC_start"])
        self.assertTrue(config["UTC_end"])
        self.assertGreaterEqual(config["runtime_wall_s"], 0)
        self.assertEqual(config["inventory_sha256"], "a" * 64)
        self.assertEqual(config["source_sha256"], "a" * 64)
        self.assertEqual(config["quantile_models"]["status"], "insufficient_training_plants")
        self.assertIn("provisional", texts["phase_result.txt"])
        self.assertTrue(config["planned_P1_gates"]["C_POLICY_not_run"])

    def test_no_qualified_windows_and_input_failure_still_report(self):
        status, csvs, jsons, texts = self.run_memory(synthetic_rows(times=[0, 12, 24]))
        self.assertEqual(status["status"], "blocked")
        self.assertEqual(csvs["replay_per_plant.csv"], [])
        self.assertEqual(jsons["config.json"]["counts"]["window_eligible_plants"], 0)
        self.assertIn("no_window_passes", texts["phase_result.txt"])
        status, csvs, jsons, texts = self.run_memory(input_error=True)
        self.assertEqual(status["status"], "blocked_input_or_execution_error")
        self.assertIn("synthetic_schema_error", status["blocking_reasons"][0])
        self.assertEqual(csvs["window_audit.csv"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

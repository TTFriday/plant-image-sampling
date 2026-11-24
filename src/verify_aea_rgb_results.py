"""Independently recompute the signed RGB sampling decomposition from input tables."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functional(times, area, count, indices):
    t = np.asarray(times)[indices]
    a, n = np.asarray(area)[indices], np.asarray(count)[indices]
    dt = np.diff(t)
    ia = float(np.sum(dt * (a[1:] + a[:-1]) / 2))
    inn = float(np.sum(dt * (n[1:] + n[:-1]) / 2))
    return {'area_mean': ia / (t[-1] - t[0]), 'leaf_mean': inn / (t[-1] - t[0]), 'area_integral': ia}


def main():
    inventory_path = ROOT / 'shared/pilot_inventory.csv'
    cache_path = ROOT / 'aea/rgb_reference_v1/perframe_measurements.csv'
    out = ROOT / 'aea/rgb_sampling_pilot_v1'
    inventory = {r['image_id']: r for r in table(inventory_path)}
    cache_rows = table(cache_path)
    cache = {r['image_id']: r for r in cache_rows}
    cache_status = json.loads((cache_path.parent / 'status.json').read_text(encoding='utf-8'))
    assert len(cache_rows) == len(cache) == cache_status['verified_frames'] == 2947
    assert sha(cache_path) == cache_status['cache_file_sha256']
    expected_ids = {k for k, r in inventory.items() if r['split_role'] in {'V_SOURCE', 'V_SHIFT'} and r['mask_status'] == 'valid'}
    assert set(cache) == expected_ids
    for key, r in cache.items():
        original = inventory[key]
        assert r['split_role'] == original['split_role']
        for name in ['rgb_sha256', 'mask_sha256', 'timestamp', 'biological_plant_id', 'width', 'height']:
            assert r[name] == original[name], (key, name)
        assert abs(float(r['model_area_fraction']) - int(r['union_pixels']) / (int(r['width']) * int(r['height']))) < 1e-12
        assert float(r['n_pred_leaves']) == int(r['n_pred_leaves']) and 0 <= int(r['n_pred_leaves']) <= 100
    rows = table(out / 'per_plant_joint_errors.csv')
    full = {r['plant_id']: json.loads(r['selected_image_ids']) for r in rows if r['method'] == 'fullRGB'}
    assert len(full) == 32
    references = {}
    for plant, ids in full.items():
        stamps = [datetime.fromisoformat(inventory[str(i)]['timestamp']) for i in ids]
        times = [(t - stamps[0]).total_seconds() / 3600 for t in stamps]
        garea = [int(inventory[str(i)]['union_pixels']) / (int(inventory[str(i)]['width']) * int(inventory[str(i)]['height'])) for i in ids]
        gcount = [int(inventory[str(i)]['n_instances']) for i in ids]
        rarea = [float(cache[str(i)]['model_area_fraction']) for i in ids]
        rcount = [int(cache[str(i)]['n_pred_leaves']) for i in ids]
        references[plant] = (times, garea, gcount, rarea, rcount, {str(i): j for j, i in enumerate(ids)})
    checked = 0
    full_fail = full_area_n = 0
    selected_summary = defaultdict(list)
    for row in rows:
        assert row['split_role'] in {'V_SOURCE', 'V_SHIFT'}
        assert row['status'] == 'success'
        times, ga, gn, ra, rn, lookup = references[row['plant_id']]
        ids = [str(i) for i in json.loads(row['selected_image_ids'])]
        indices = np.asarray([lookup[i] for i in ids])
        assert len(indices) == int(row['actual_K']) and len(set(ids)) == len(ids)
        assert indices[0] == 0 and indices[-1] == len(times) - 1 and np.all(np.diff(indices) > 0)
        all_indices = np.arange(len(times))
        gf, rf = functional(times, ga, gn, all_indices), functional(times, ra, rn, all_indices)
        gs, rs = functional(times, ga, gn, indices), functional(times, ra, rn, indices)
        area_ok = row['area_eligible'].lower() == 'true'
        for name in gf:
            if name != 'leaf_mean' and not area_ok:
                continue
            values = {'e_M': rf[name] - gf[name], 'e_S': gs[name] - gf[name],
                      'e_X': (rs[name] - gs[name]) - (rf[name] - gf[name]), 'e_total': rs[name] - gf[name]}
            assert abs(values['e_total'] - values['e_M'] - values['e_S'] - values['e_X']) < 1e-10
            for term, value in values.items():
                assert abs(float(row[f'{name}_{term}']) - value) < 1e-9
                assert abs(float(row[f'{name}_{term}_abs']) - abs(value)) < 1e-9
        leaf_error = abs(rs['leaf_mean'] - gf['leaf_mean'])
        assert abs(float(row['E_N_leaves']) - leaf_error) < 1e-9
        if area_ok:
            area_error = 100 * abs(rs['area_mean'] - gf['area_mean'])
            assert abs(float(row['E_A_pp']) - area_error) < 1e-9
            passed = area_error <= 1 + 1e-12 and leaf_error <= .5 + 1e-12
            assert float(row['joint_pass']) == float(passed)
            if row['method'] == 'fullRGB':
                full_area_n += 1
                full_fail += int(not passed)
            if row['method'] == 'U_0':
                selected_summary[int(row['K'])].append({'area_error_pp': area_error, 'leaf_error': leaf_error, 'passed': passed})
        checked += 1
    assert checked == 2624 and full_area_n == 28 and full_fail == 4
    result = {'status': 'PASS', 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'cache_frames_verified': len(cache), 'windows': len(full), 'records_recomputed': checked,
              'signed_and_absolute_decomposition_verified': True, 'fullRGB_joint_failures': full_fail,
              'fullRGB_area_eligible_n': full_area_n, 'cache_sha256': sha(cache_path),
              'replay_sha256': sha(out / 'per_plant_joint_errors.csv'), 'verifier_sha256': sha(Path(__file__)),
              'U0_area_eligible_joint_summary': {str(k): {'n': len(v),
                   'mean_area_error_pp': float(np.mean([r['area_error_pp'] for r in v])),
                   'mean_leaf_error': float(np.mean([r['leaf_error'] for r in v])),
                   'joint_pass_n': int(sum(r['passed'] for r in v))} for k, v in sorted(selected_summary.items())},
              'note': 'Development-only exploratory frozen B1 measurement; not formal AEA model, calibration or external validation.'}
    (out / 'independent_verification_recomputed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

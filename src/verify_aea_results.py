"""Independently recompute GT replay errors from the frozen inventory."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integral_mean(times, values):
    times, values = np.asarray(times), np.asarray(values)
    return float(np.sum((times[1:] - times[:-1]) * (values[1:] + values[:-1]) * .5) / (times[-1] - times[0]))


def main():
    folder = ROOT / 'aea/pilot_v2'
    inventory_path = ROOT / 'shared/pilot_inventory.csv'
    inventory = {r['image_id']: r for r in csv.DictReader(inventory_path.open(encoding='utf-8-sig'))}
    records = list(csv.DictReader((folder / 'replay_per_plant.csv').open(encoding='utf-8-sig')))
    full = {r['plant_id']: json.loads(r['selected_image_ids']) for r in records if r['method'] == 'full'}
    reference = {}
    for plant, ids in full.items():
        stamps = [datetime.fromisoformat(inventory[str(i)]['timestamp']) for i in ids]
        times = [(t - stamps[0]).total_seconds() / 3600 for t in stamps]
        area = [int(inventory[str(i)]['union_pixels']) / (int(inventory[str(i)]['width']) * int(inventory[str(i)]['height'])) for i in ids]
        count = [int(inventory[str(i)]['n_instances']) for i in ids]
        reference[plant] = (times, area, count, {str(i): j for j, i in enumerate(ids)})
    succeeded, failed = 0, 0
    baseline = defaultdict(list)
    area_plants = set()
    for record in records:
        assert record['split_role'] in {'TRAIN', 'V_SOURCE', 'V_SHIFT'}
        if record['status'] != 'success':
            failed += 1
            continue
        succeeded += 1
        ids = [str(i) for i in json.loads(record['selected_image_ids'])]
        assert len(ids) == int(record['actual_K']) and len(set(ids)) == len(ids)
        assert all(inventory[i]['mask_status'] == 'valid' for i in ids)
        t, area, count, lookup = reference[record['plant_id']]
        indices = [lookup[i] for i in ids]
        assert indices[0] == 0 and indices[-1] == len(t) - 1
        assert all(a < b for a, b in zip(indices, indices[1:]))
        selected_times = [t[i] for i in indices]
        ea = 100 * abs(integral_mean(selected_times, [area[i] for i in indices]) - integral_mean(t, area))
        en = abs(integral_mean(selected_times, [count[i] for i in indices]) - integral_mean(t, count))
        assert abs(en - float(record['E_N_leaves'])) < 1e-8
        area_ok = record['area_eligible'].lower() == 'true'
        if area_ok:
            area_plants.add(record['plant_id'])
            assert abs(ea - float(record['E_A_pp'])) < 1e-8
            expected_pass = float(max(ea, en / .5) <= 1 + 1e-12)
            assert expected_pass == float(record['joint_pass'])
            if record['method'] == 'U_0':
                baseline[int(record['K'])].append({'area_error_pp': ea, 'leaf_error': en, 'pass': expected_pass})
        if record['method'] == 'full':
            assert ea == 0 and en == 0
    result = {'status': 'PASS', 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'verifier_sha256': sha256(__file__), 'inventory_sha256': sha256(inventory_path),
              'replay_sha256': sha256(folder / 'replay_per_plant.csv'),
              'replays_checked': len(records), 'successful': succeeded, 'failed': failed,
              'unique_windows': len(full), 'area_eligible_windows': len(area_plants),
              'independently_recomputed_U0': {str(k): {'n_plants': len(v),
                  'mean_area_error_pp': float(np.mean([r['area_error_pp'] for r in v])),
                  'mean_leaf_error': float(np.mean([r['leaf_error'] for r in v])),
                  'joint_pass_count': int(sum(r['pass'] for r in v)),
                  'joint_pass_rate': float(np.mean([r['pass'] for r in v]))} for k, v in sorted(baseline.items())},
              'note': 'GT development-only discrete reference; not RGB measurement or locked-test validation.'}
    (folder / 'independent_verification_recomputed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

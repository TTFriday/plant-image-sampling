# Plant image sampling

Code and archived results for error-constrained image-acquisition
scheduling.  The repository contains a GT sampling-only replay, a frozen
RGB measurement layer, per-plant error tables, paired bootstrap results
and independent verification programs.

## Contents

- `src/`: sampling implementation, frozen RGB reference, shared detector
  utilities and independent result verification.
- `aea/pilot_v2/`: GT replay outputs and window audit.
- `aea/rgb_reference_v1/`: per-frame frozen RGB measurements.
- `aea/rgb_sampling_pilot_v1/`: joint RGB sampling analysis.
- `shared/pilot_inventory.csv`: portable audited inventory.
- `scripts/derive_numbers.py`: recompute the result ledger from archived tables.
- `tests/`: synthetic checks, not scientific results.
- `MANIFEST.json`, `MANIFEST.gitblob.sha1`, `SOURCE_MANIFEST.json`: hashes and provenance.

## Reproduce the table derivation

```bash
python scripts/derive_numbers.py --gt aea/pilot_v2 --rgb aea/rgb_sampling_pilot_v1 --v1 aea/pilot_v1 --out data/numbers_recomputed.json
python -m unittest discover -s tests -p "test_aea*.py"
```

To independently verify the archived tables without raw media:

```bash
python src/verify_aea_results.py
python src/verify_aea_rgb_results.py
```

The verifier programs use the packaged `shared/` and `aea/` layout and write
a fresh `independent_verification_recomputed.json` into the corresponding run
folder without overwriting the archived verification report.

## Scientific scope

The GT layer uses future true area/count values to isolate sampling error.
The RGB layer uses frozen model predictions.  Both know the times of future
sampling opportunities but not future readings; the oracle is a separate
hindsight diagnostic.

The 98.0% figure is a GT sampling-only joint pass rate (100/102), not an RGB
system pass rate.  RGB joint pass rates were 82.1%, 89.3% and 85.7% at the
three budgets, all below 90%.  The RGB reference was selected on the same two
development partitions and is not independent validation.  Area is a
fixed-pixel-field fraction, not physically calibrated biomass.

Public copies remove machine snapshots and local absolute paths.  Scientific
table values are unchanged.  Strict cached-extraction hashes refer to the
original bit-identical inputs; see `DATA_SOURCES.md` and
`SOURCE_MANIFEST.json` before attempting a full re-run.

No code licence is granted by this repository.  Upstream data and model
artefacts retain their own licences.

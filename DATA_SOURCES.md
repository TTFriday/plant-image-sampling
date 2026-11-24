# Data and model sources

No source RGB images, masks, label maps or model checkpoints are
redistributed.  The package contains the audited inventory, GT replay
outputs, frozen RGB measurements and derived sampling analyses.

## PhenoLeaf-TS

Obtain the media from the original PhenoLeaf-TS source and follow its
non-commercial research/teaching licence and citation requirements.  The
upstream licence does not permit public redistribution of dataset files
without written permission.  `shared/pilot_inventory.csv` therefore keeps
only metadata and content hashes; its three path columns are relative to a
user-supplied data root.

The original inventory hash and published hash are recorded in
`shared/inventory_provenance.json`.  Archived configuration files retain
their original hash fields as provenance.  Because the portable inventory
and cleaned configuration files have different byte hashes, the strict
extraction/analysis chain in `aea_rgb_pilot.py` is intended for fresh runs
with matching inputs.  Use `src/verify_aea_rgb_results.py` to independently
recompute the archived RGB decomposition from the packaged tables.

## Frozen RGB reference

RGB measurements were produced by a frozen development-reference model.
They are exploratory measurements, not an independently trained AEA model
or external validation.  The checkpoint and raw media required to repeat
extraction are not included.  COCO initialization follows the torchvision
licence; set `PHENOLEAF_COCO_WEIGHTS` or pass `--weights` when needed.

Locked roles were not used for calibration, model selection or effect
analysis.  TRAIN was used only by the upstream reference-model run.

# Data Layout

This repository includes the data needed to train the 0413 Qwen3 LoRA runs on a
server without regenerating Gemini outputs.

- `raw/`: source disorder records.
- `generated/`: Gemini CoT extraction outputs.
- `llamafactory/`: converted ShareGPT datasets and `dataset_info.json`.
- `samples/`: small evaluation fixtures.
- `schema_regression/`: known structural failure cases derived from the original
  training source; not an independent test set.
- `schema_v2/`: source manifest, deterministic migration, manual-review queue,
  and hierarchy-grouped train/validation/held-out splits.

Committed deployment files:

- `raw/mental_disorders_20251125_165535.json`
- `generated/0413_cot_specific_614_12000_8000.json`
- `generated/0413_cot_standard_635_12000_8000.json`
- `llamafactory/kg_cot_specific_614.json`
- `llamafactory/kg_cot_standard_635.json`

The `llamafactory/` datasets are the files used directly by
`scripts/03_train_lora.sh specific` and `scripts/03_train_lora.sh standard`.

Schema v2 artifacts remain a draft until the medical conflicts in
`schemas/v2.0.0/conflicts.json` are resolved. Do not train on
`data/schema_v2/splits/train_v2.json` as-is while validation errors or unresolved
manual-review items remain.

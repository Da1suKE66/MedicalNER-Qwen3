# Data Layout

This repository includes the data needed to train the 0413 Qwen3 LoRA runs on a
server without regenerating Gemini outputs.

- `raw/`: source disorder records.
- `generated/`: Gemini CoT extraction outputs.
- `llamafactory/`: converted ShareGPT datasets and `dataset_info.json`.
- `samples/`: small evaluation fixtures.

Committed deployment files:

- `raw/mental_disorders_20251125_165535.json`
- `generated/0413_cot_specific_614_12000_8000.json`
- `generated/0413_cot_standard_635_12000_8000.json`
- `llamafactory/kg_cot_specific_614.json`
- `llamafactory/kg_cot_standard_635.json`

The `llamafactory/` datasets are the files used directly by
`scripts/03_train_lora.sh specific` and `scripts/03_train_lora.sh standard`.

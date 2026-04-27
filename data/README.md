# Data Layout

This repository keeps only small fixtures and metadata in Git.

- `raw/`: source disorder records, for example `mental_disorders.json`.
- `generated/`: raw and filtered Gemini CoT extraction outputs.
- `llamafactory/`: converted ShareGPT datasets and `dataset_info.json`.
- `samples/`: small evaluation fixtures.

Full generated datasets are ignored by default. Share them as release assets or
external data files after checking license and privacy constraints.

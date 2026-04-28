# Gemini Split/COT Data Generation

This directory contains the Gemini generation utilities used for the April 2026
split-field data pass.

## Scripts

- `split_gemini_generation.py`: split ICD JSON into structural and long-text inputs, and merge two generated outputs.
- `run_split_gemini_generation.py`: run Gemini Flash on split inputs to produce merge-ready KG records.
- `run_cot_gemini_generation.py`: run Gemini 3 Pro Preview with thought parts enabled to produce COT samples compatible with `src/kg_lora/convert_to_llamafactory.py`.
- `combine_cot_batches.py`: combine multiple COT batch files and summarize sidecar metadata.

## Generated Files

- `data/generated/gemini_split/pro_cot_001_005_schema0413.json`
- `data/generated/gemini_split/pro_cot_006_110_schema0413.json`
- `data/generated/gemini_split/flash_structure_011_110_schema0413_partial.json`
- `data/llamafactory/pro_cot_001_005_llamafactory.json`
- `data/llamafactory/pro_cot_006_110_llamafactory.json`

`flash_structure_011_110_schema0413_partial.json` is intentionally marked
partial: it contains 90 successful records from the interrupted 11-110 Flash
structure run.

## Pro COT Command

```bash
python3 scripts/gemini/split_gemini_generation.py split \
  --input data/raw/mental_disorders_20251125_165535.json \
  --output_dir data/generated/gemini_split \
  --prefix mental_disorders_20251125_165535

python3 scripts/gemini/run_cot_gemini_generation.py \
  --input data/generated/gemini_split/mental_disorders_20251125_165535.long_text.json \
  --output data/generated/gemini_split/pro_cot_next_batch.json \
  --model_name gemini-3-pro-preview \
  --include_thoughts \
  --checkpoint_every 1
```

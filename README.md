# KG LoRA

Clean workflow for generating mental-health knowledge graph extraction data and
fine-tuning Qwen with LoRA.

The project is organized as a runnable pipeline:

```text
data/raw/             Input disorder records, not committed
data/generated/       Gemini CoT generation outputs, not committed
data/llamafactory/    ShareGPT datasets plus dataset_info.json
configs/llamafactory/ LLaMA-Factory LoRA configs
src/kg_lora/          Python modules for generation, conversion, and evaluation
scripts/              Small command entry points
models/adapters/      Local LoRA adapters, not committed
outputs/              Evaluation outputs, not committed
```

## 1. Setup

```bash
cd github/kg-lora
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY` in `.env`. Install LLaMA-Factory separately if
`llamafactory-cli` is not already available in your environment.

## 2. Prepare Input Data

Put the source disorder records at:

```text
data/raw/mental_disorders.json
```

You can also pass another JSON path to the scripts. The generator reads this
path through `KG_DATA_PATH` or the first positional argument.

## 3. Generate CoT Extraction Data

Run a small smoke test first:

```bash
MAX_SAMPLES=3 WORKERS=1 bash scripts/01_generate_cot_data.sh
```

Run the normal specific-CoT generation:

```bash
COT_STYLE=specific WORKERS=4 bash scripts/01_generate_cot_data.sh
```

Outputs are written to `data/generated/`, including raw and filtered JSON files.

## 4. Convert to LLaMA-Factory Format

Convert a filtered generation file into ShareGPT format:

```bash
bash scripts/02_convert_to_llamafactory.sh \
  data/generated/cot_filtered_merged.json \
  kg_cot_specific_614 \
  specific
```

This writes:

```text
data/llamafactory/kg_cot_specific_614.json
data/llamafactory/dataset_info.json
```

For standard template CoT, use:

```bash
bash scripts/02_convert_to_llamafactory.sh \
  data/generated/cot_filtered_merged.json \
  kg_cot_standard_635 \
  template
```

## 5. Fine-Tune LoRA

Train the specific-CoT adapter:

```bash
bash scripts/03_train_lora.sh specific
```

Train the standard-CoT adapter:

```bash
bash scripts/03_train_lora.sh standard
```

The adapter outputs go to `models/adapters/`.

## 6. One-Command Specific-CoT Pipeline

After setup and input data are ready, this runs generation, conversion, and
specific-CoT LoRA training:

```bash
bash scripts/run_specific_pipeline.sh data/raw/mental_disorders.json
```

## 7. Compare Base and LoRA Outputs

Set adapter paths in `.env` if they differ from the defaults, then run:

```bash
bash scripts/04_compare_outputs.sh
```

The script writes model outputs under `outputs/` and prints analysis summaries.
The latest existing report is kept in `reports/qwen_compare_analysis_report.md`.

## Notes

- This repository intentionally does not commit full generated datasets or model
  weights.
- The default training configs use Qwen3-8B with LoRA rank 8 and Qwen3 chat
  template.
- For quick checks, prefer `MAX_SAMPLES=3 WORKERS=1` before launching a full
  generation run.

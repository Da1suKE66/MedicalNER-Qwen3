# KG LoRA

Clean workflow for generating mental-health knowledge graph extraction data and
fine-tuning Qwen with LoRA.

The project is organized as a runnable pipeline:

```text
data/raw/             Input disorder records
data/generated/       Gemini CoT generation outputs
data/llamafactory/    ShareGPT datasets plus dataset_info.json
configs/llamafactory/ LLaMA-Factory LoRA configs
src/kg_lora/          Python modules for generation, conversion, and evaluation
scripts/              Small command entry points
models/adapters/      Local LoRA adapters, not committed
outputs/              Evaluation outputs, not committed
```

## 1. Setup

For ModelArts jobs, use the setup helper. It creates or repairs the conda env,
keeps Python from loading packages installed under `~/.local`, installs a CUDA
12 compatible PyTorch build, installs LLaMA-Factory, and checks CUDA/bf16:

```bash
cd ~/workspace/llc/MedicalNER-Qwen3
ENV_PREFIX=/cache/llc/KG bash scripts/setup_modelarts_env.sh
source /home/ma-user/miniconda3/bin/activate /cache/llc/KG
export PYTHONNOUSERSITE=1
cp .env.example .env
```

For a normal local Python environment:

```bash
cd MedicalNER-Qwen3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `GEMINI_API_KEY` in `.env` before running Gemini data generation. LoRA
training from the committed LLaMA-Factory datasets does not require Gemini.

The training requirements pin `torch==2.6.0` so CUDA 12.x ModelArts drivers do
not accidentally load newer CUDA 13 wheels from user site packages. If you are
using a different driver stack, adjust the PyTorch version before setup.

## 2. Prepare Input Data

The deployment data is committed in the repository:

```text
data/raw/mental_disorders_20251125_165535.json
data/generated/0413_cot_specific_614_12000_8000.json
data/generated/0413_cot_standard_635_12000_8000.json
data/llamafactory/kg_cot_specific_614.json
data/llamafactory/kg_cot_standard_635.json
```

The two `data/llamafactory/*.json` files are already converted to ShareGPT
format for LLaMA-Factory and use Qwen3 `<think>...</think>` tags.

You can still pass another raw JSON path to the generation scripts. The
generator reads this path through `KG_DATA_PATH` or the first positional
argument.

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

### Select NVIDIA CUDA or Huawei Ascend NPU

The unified launcher keeps the CUDA and NPU implementations separate and
selects one with `--backend`. NVIDIA CUDA is the default:

```bash
# NVIDIA CUDA (default)
bash scripts/run_medicalner_qwen3.sh --backend cuda --task smoke
bash scripts/run_medicalner_qwen3.sh --backend cuda --task train
bash scripts/run_medicalner_qwen3.sh --backend cuda --task predict

# Huawei Ascend NPU
bash scripts/run_medicalner_qwen3.sh --backend npu --task smoke
bash scripts/run_medicalner_qwen3.sh --backend npu --task train
bash scripts/run_medicalner_qwen3.sh --backend npu --task predict
```

Use `--device 0` (or a comma-separated device list) to select visible devices.
Use `--config path/to/config.yaml` when the model, adapter, or output paths differ
from the committed defaults. The CUDA configs default to the portable
`Qwen/Qwen3-8B` model ID; the NPU configs retain the Ascend environment paths and
NPU-specific FlashAttention setup. CUDA and NPU training outputs use distinct
directories so that one backend does not overwrite the other.

The underlying scripts remain available for direct use:

```text
scripts/run_medicalner_qwen3_pro858.sh              # CUDA training
scripts/run_medicalner_qwen3_pro858_predict.sh      # CUDA prediction
scripts/run_medicalner_qwen3_pro858_npu.sh          # Ascend NPU training
scripts/run_medicalner_qwen3_pro858_predict_npu.sh  # Ascend NPU prediction
```

On a server with LLaMA-Factory installed, train directly from the committed
datasets:

```bash
PYTHONNOUSERSITE=1 bash scripts/03_train_lora.sh specific
```

Train the standard-CoT adapter:

```bash
PYTHONNOUSERSITE=1 bash scripts/03_train_lora.sh standard
```

The adapter outputs go to `models/adapters/`. The training script sets cache
paths under `.cache/` by default. Override `KG_CACHE_ROOT`, `HF_HOME`, or
related Hugging Face cache variables if your server requires a different disk.

The script validates that the selected training dataset exists and does not
contain Gemini-style `<thinking>` tags before launching LLaMA-Factory.

The committed CoT training configs use 4-bit bitsandbytes QLoRA
(`quantization_bit: 4`) with bf16 compute. To try int8 instead, change
`quantization_bit: 4` to `quantization_bit: 8` in the selected YAML.
Evaluation and checkpointing run every 10 update steps, which gives roughly
20 eval/checkpoint points for the 0413 datasets instead of only 2.

The configs use `Qwen/Qwen3-8B`. The `Qwen/Qwen3-8B-Instruct` identifier is not
used because it is not available on the Hugging Face mirror used by ModelArts.

## 6. One-Command Specific-CoT Pipeline

After setup and input data are ready, this runs generation, conversion, and
specific-CoT LoRA training:

```bash
bash scripts/run_specific_pipeline.sh data/raw/mental_disorders_20251125_165535.json
```

## 7. Compare Base and LoRA Outputs

Set adapter paths in `.env` if they differ from the defaults, then run:

```bash
bash scripts/04_compare_outputs.sh
```

The script writes model outputs under `outputs/` and prints analysis summaries.
The latest existing report is kept in `reports/qwen_compare_analysis_report.md`.

## Notes

- This repository commits the raw source JSON, the two 0413 CoT generated JSON
  files, and the two converted LLaMA-Factory training JSON files needed for
  server training. It still does not commit model weights.
- The default training configs use Qwen3-8B with LoRA rank 8 and Qwen3 chat
  template.
- For quick checks, prefer `MAX_SAMPLES=3 WORKERS=1` before launching a full
  generation run.

## Schema v2 data quality workflow

The current contract is `2.0.0-draft.2` under `schemas/v2.0.0/`. The final
train-ready corpus is produced deterministically from all 858 source records; API
results and every dropped graph item remain auditable in `reports/`. See
`docs/schema_v2_implementation.md` for the design and evidence policy.

### Verified result

| Check | Result |
| --- | ---: |
| Source records | 858 |
| Historical damaged tokens | 22 types / 305 occurrences |
| Legitimate browser-URL JSON nulls | 858 |
| WHO-verified canonical main diseases | 858 / 858 |
| Non-main WHO associations not attached | 72 HTTP 404 / 406 exact-title / 256 title-mismatch |
| Relations removed before training | 728 without verified spans / 182 requiring review |
| Train-ready graph | 8,590 entities / 6,702 relations / 858 repaired records |
| Train / validation / held-out / structural regression | 670 / 84 / 84 / 20 |
| Direct or shared-parent split leakage | 0 |
| Strict training-data audit | all gates passed |
| LLaMAFactory train / validation / held-out | 670 / 84 / 84; no legacy CoT |

The 858 nulls are legitimate source metadata at
`excel_metadata.browser_link` (called `browserUrl` in the review notes), not
damaged words or missing graph values. The historical word corruption is repaired
separately and strict JSON serialization rejects `NaN`.

### DeepSeek and WHO credentials

Copy the template and fill the local file without printing it:

```bash
cp .env.example .env
```

For DeepSeek targeted repair, fill `DEEPSEEK_API_KEY`; the template already selects
`DEEPSEEK_BASE_URL=https://api.deepseek.com`,
`DEEPSEEK_MODEL=deepseek-v4-flash`, thinking disabled for strict JSON,
`DEEPSEEK_REASONING_EFFORT`, `DEEPSEEK_MAX_TOKENS`, and
`DEEPSEEK_TIMEOUT_SECONDS`. `GEMINI_API_KEY` is optional legacy compatibility.

For WHO ICD-11 validation, register an API client at the
[WHO ICD API registration page](https://icd.who.int/icdapi/Account/Register) and
follow the [WHO authentication documentation](https://icd.who.int/docs/icd-api/API-Authentication/).
Fill `WHO_ICD_CLIENT_ID` and `WHO_ICD_CLIENT_SECRET`; the template also defines
`WHO_ICD_BASE_URL`, `WHO_ICD_TOKEN_URL`, `WHO_ICD_API_VERSION`,
`WHO_ICD_RELEASE`, and `WHO_ICD_LANGUAGE`.

The remaining template fields are `KG_DATA_PATH`, `KG_OUTPUT_ROOT`,
`KG_CACHE_ROOT`, `ICD_API_CACHE_DIR`, `DEEPSEEK_REPAIR_CACHE_DIR`, `BASE_MODEL`,
`SPECIFIC_ADAPTER`, and `STANDARD_ADAPTER`. On the configured remote training hosts,
set `KG_CACHE_ROOT` to a directory under `/cache/liluchen`. Never print, commit, or
copy `.env` into reports; only `.env.example` is safe to commit.

### Reproduce the final pipeline

Run these commands from the repository root after filling the WHO credentials:

```bash
# 1. Deterministic Schema v2 migration and structural manifestation collapse.
python3 scripts/migrate_schema_v2.py \
  --apply-high-confidence-collapses \
  --force-renormalize

# 2. Repair exact evidence offsets while retaining unresolved items for audit.
python3 scripts/repair_evidence_spans.py

# 3. Collect the full WHO report. Do not use --strict here: non-main title
#    mismatches must be recorded so the application step can discard them safely.
python3 scripts/validate_icd_codes.py \
  --input data/schema_v2/cleaned/pro_cot_schema_v2_evidence.json \
  --input-format schema-v2 \
  --output reports/icd_api_validation_full_report.json

# 4. Require all canonical main diseases to be WHO verified.
python3 scripts/apply_icd_validation.py --strict

# 5. Remove unverified-span and review-only graph items with an audit trail.
python3 scripts/sanitize_schema_v2_for_training.py --strict

# 6. Freeze and materialize hierarchy-grouped splits from the train-ready corpus.
python3 scripts/build_v2_splits.py \
  --records data/schema_v2/cleaned/pro_cot_schema_v2_train_ready.json

# 7. Gate the complete corpus and split manifest.
python3 scripts/audit_schema_v2_training_data.py --strict

# 8a. Convert the training split (the defaults point to train_v2.json).
python3 scripts/convert_schema_v2_to_llamafactory.py

# 8b. Convert validation and held-out splits explicitly.
python3 scripts/convert_schema_v2_to_llamafactory.py \
  --input data/schema_v2/splits/validation_v2.json \
  --output data/llamafactory/schema_v2_validation_llamafactory.json \
  --manifest data/llamafactory/schema_v2_validation_manifest.json \
  --expected-split validation
python3 scripts/convert_schema_v2_to_llamafactory.py \
  --input data/schema_v2/splits/test_v2_heldout.json \
  --output data/llamafactory/schema_v2_heldout_llamafactory.json \
  --manifest data/llamafactory/schema_v2_heldout_manifest.json \
  --expected-split test_v2_heldout
```

The WHO scan intentionally audits non-main code associations but does not attach
them to graph entities: 72 returned 404, 406 had exact titles, and 256 local
associations had title mismatches. Only the 858 canonical main diseases are required
and attached. `somatic_cause_of` is an active Disease-to-Disease relation in the
schema, but migration does not invent this edge without exact source evidence.

The structural collapse is similarly conservative. In the inattention regression
case, `S1` remains the Symptom and its three descriptive phrases become
`properties.manifestations`; the old `S2`-`S4` description nodes and redundant
relations are removed with source offsets recorded. This rule is not a general
semantic merge.

`schema_regression_20` is a structural regression set drawn from the same 858-record
source, so it is never reported as generalization. Train only on `train_v2.json`, use
`validation_v2.json` for model selection, and reserve `test_v2_heldout.json` for the
new adapter's final evaluation. The Schema v2 converter emits canonical graph JSON
only and never copies historical reasoning traces.

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}}"

cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
KG_CACHE_ROOT="${KG_CACHE_ROOT:-${PROJECT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${KG_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

YAML="${CONFIG_YAML:-configs/llamafactory/predict_qwen3_8b_cot_pro858_format.yaml}"
OUTDIR="${OUTDIR:-models/predict/qwen3-8b-cot-pro858-format-check}"
ADAPTER="${ADAPTER:-models/adapters/qwen3-8b-cot-pro858}"

echo "===== Runtime Info ====="
hostname
date
which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu {i}:", torch.cuda.get_device_name(i))
PY

which llamafactory-cli
nvidia-smi || true

echo "===== Adapter files ====="
ls -lh "$ADAPTER" | head -n 50

echo "===== Trainer state / best checkpoint ====="
ADAPTER="${ADAPTER}" python - <<'PY'
import json, os
from pathlib import Path

p = Path(os.environ["ADAPTER"]) / "trainer_state.json"
if p.exists():
    s = json.loads(p.read_text())
    print("best_model_checkpoint:", s.get("best_model_checkpoint"))
    print("best_metric:", s.get("best_metric"))
    print("global_step:", s.get("global_step"))
else:
    print("trainer_state.json not found")
PY

echo "===== YAML ====="
sed -n '1,240p' "$YAML"

echo "===== Start prediction ====="
llamafactory-cli train "$YAML"

echo "===== Prediction finished ====="
date

echo "===== Output files ====="
ls -lh "$OUTDIR" || true

echo "===== Preview generated predictions ====="
OUTDIR="${OUTDIR}" python - <<'PY'
import json, os
from pathlib import Path

p = Path(os.environ["OUTDIR"]) / "generated_predictions.jsonl"

print("prediction file:", p)
print("exists:", p.exists())

if not p.exists():
    raise SystemExit(0)

with p.open(encoding="utf-8") as f:
    for idx, line in zip(range(3), f):
        obj = json.loads(line)
        print("\n" + "=" * 100)
        print("CASE", idx + 1)
        print("-" * 40)
        print("PROMPT preview:")
        print(obj.get("prompt", "")[:1000])
        print("-" * 40)
        print("PREDICT preview:")
        print(obj.get("predict", "")[:2000])
        print("-" * 40)
        print("LABEL preview:")
        print(obj.get("label", "")[:1000])
PY

echo "===== JSON format quick check ====="
OUTDIR="${OUTDIR}" python - <<'PY'
import json
import os
import re
from pathlib import Path

p = Path(os.environ["OUTDIR"]) / "generated_predictions.jsonl"

def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return text

if not p.exists():
    print("generated_predictions.jsonl not found")
    raise SystemExit(0)

total = 0
valid = 0
bad_examples = []

with p.open(encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        total += 1
        obj = json.loads(line)
        pred = obj.get("predict", "")
        cleaned = clean_json_text(pred)

        try:
            json.loads(cleaned)
            valid += 1
        except Exception as e:
            if len(bad_examples) < 5:
                bad_examples.append((i, str(e), pred[:1000]))

print(f"Total: {total}")
print(f"Valid JSON: {valid}")
print(f"Invalid JSON: {total - valid}")
if total:
    print(f"Valid ratio: {valid / total:.2%}")

for line_no, err, preview in bad_examples:
    print("\n" + "=" * 80)
    print("Line:", line_no)
    print("Error:", err)
    print("Predict preview:")
    print(preview)
PY

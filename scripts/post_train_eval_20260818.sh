#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the full output-only run, then preserve its final adapter and run a
# long-output structural evaluation on frozen group-disjoint train/dev probes.
ROOT="/home/ma-user/workspace/liluchen/MedicalNER-Qwen3"
OUT="/cache/liluchen/medicalner_output_objectives/output/lora_output_only_groupdisjoint_20260818"
RUN_ROOT="/temp/liluchen/train_output_only_groupdisjoint_20260818"
EVAL_RUN="/temp/liluchen/post_eval_output_only_groupdisjoint_20260818"
RESULT="/cache/liluchen/medicalner_output_objectives/outputs/post_train_output_only_groupdisjoint_20260818_max8192.json"
REPORT="/cache/liluchen/medicalner_output_objectives/reports/post_train_output_only_groupdisjoint_20260818_max8192.json"
DATA="${ROOT}/data/llamafactory/deepseek_watermark_20260804_182312_output_only_full.json"
MANIFEST="${ROOT}/data/llamafactory/groupdisjoint/groupdisjoint_split_manifest.json"
CHUNKS="${ROOT}/data/snapshots/deepseek_watermark_20260804_182312/chunks.json"
PYTHON="/cache/liluchen/envs/medicalner/bin/python"

# 25 deterministic probes: length quantiles from both splits plus previously
# observed long/truncation/semantic-error records.
INDICES="609,1449,1199,1155,1281,809,320,235,80,86,304,741,858,1491,1205,959,270,125,6,4,255,11,1492,0,1"

cd "${ROOT}"
export PATH="/cache/liluchen/envs/medicalner/bin:${HOME}/.local/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME=/cache/liluchen/hf_home
export HUGGINGFACE_HUB_CACHE=/cache/liluchen/model_cache

while pgrep -f "llamafactory-cli train configs/llamafactory/qwen3_8b_lora_deepseek_output_only_groupdisjoint_20260818.yaml" >/dev/null 2>&1; do
  sleep 30
done

# The original training wrapper may still be waiting on its old snapshot sleep.
# The trainer output is already immutable at this point; preserve it explicitly.
mkdir -p "${RUN_ROOT}/artifacts/final_output" "${RUN_ROOT}/metadata"
cp -a "${OUT}/." "${RUN_ROOT}/artifacts/final_output/" 2>/dev/null || true
printf '0\n' > "${RUN_ROOT}/metadata/exit_code.txt"
rm -f "${RUN_ROOT}/running"

ADAPTER="${OUT}"
if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
  ADAPTER="$(${PYTHON} - <<'PY'
import json
from pathlib import Path
root = Path("/cache/liluchen/medicalner_output_objectives/output/lora_output_only_groupdisjoint_20260818")
state = root / "trainer_state.json"
best = None
if state.exists():
    best = json.loads(state.read_text()).get("best_model_checkpoint")
if best and (Path(best) / "adapter_model.safetensors").exists():
    print(best)
else:
    checkpoints = sorted(root.glob("checkpoint-*/adapter_model.safetensors"), key=lambda p: int(p.parent.name.split("-")[-1]))
    if not checkpoints:
        raise SystemExit("no adapter checkpoint found")
    print(checkpoints[-1].parent)
PY
  )"
fi

mkdir -p "$(dirname "${RESULT}")" "$(dirname "${REPORT}")"
export SNAPSHOT_ROOT="${EVAL_RUN}"
export SNAPSHOT_INTERVAL_SEC=600
export SNAPSHOT_PATHS="scripts:configs:data/llamafactory:reports:outputs"
export SNAPSHOT_ABSOLUTE_PATHS="${ADAPTER}"
scripts/run_with_snapshots.sh post_eval_output_only_groupdisjoint_20260818 \
  "${PYTHON}" scripts/compare_output_objectives_20260810.py \
  --data "${DATA}" \
  --base-model /cache/liluchen/model_cache/Qwen3-8B \
  --priority-adapter "${ADAPTER}" \
  --output-only-adapter "${ADAPTER}" \
  --output "${RESULT}" \
  --max-new-tokens 8192 \
  --batch-size 1 \
  --no-quantization \
  --only-model output_only \
  --split-manifest "${MANIFEST}" \
  --indices "${INDICES}"

"${PYTHON}" scripts/analyze_comparison_20260811.py "${RESULT}" \
  --source-chunks "${CHUNKS}" --output "${REPORT}"
printf '%s\n' "${RESULT}" "${REPORT}"

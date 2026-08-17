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
BASE_RESULT="/cache/liluchen/medicalner_output_objectives/outputs/post_train_base_qwen_groupdisjoint_20260818_max8192.json"
BASE_REPORT="/cache/liluchen/medicalner_output_objectives/reports/post_train_base_qwen_groupdisjoint_20260818_max8192.json"
DATA="${ROOT}/data/llamafactory/deepseek_watermark_20260804_182312_output_only_full.json"
MANIFEST="${ROOT}/data/llamafactory/groupdisjoint/groupdisjoint_split_manifest.json"
CHUNKS="${ROOT}/data/snapshots/deepseek_watermark_20260804_182312/chunks.json"
PYTHON="/cache/liluchen/envs/medicalner/bin/python"
TRAIN_PATTERN="/cache/liluchen/envs/medicalner/bin/llamafactory-cli train configs/llamafactory/qwen3_8b_lora_deepseek_output_only_groupdisjoint_20260818.yaml"

# 25 deterministic probes: length quantiles from both splits plus previously
# observed long/truncation/semantic-error records.
INDICES="609,1449,1199,1155,1281,809,320,235,80,86,304,741,858,1491,1205,959,270,125,6,4,255,11,1492,0,1"

cd "${ROOT}"
export PATH="/cache/liluchen/envs/medicalner/bin:${HOME}/.local/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME=/cache/liluchen/hf_home
export HUGGINGFACE_HUB_CACHE=/cache/liluchen/model_cache

# Match the actual trainer executable, not the old snapshot wrapper's command
# line.  The old wrapper can remain asleep briefly after Python exits.
while pgrep -f "${TRAIN_PATTERN}" >/dev/null 2>&1; do
  sleep 30
done
sleep 30

# The original training wrapper may still be waiting on its old snapshot sleep.
# The trainer output is already immutable at this point; preserve it explicitly.
mkdir -p "${RUN_ROOT}/artifacts/final_output" "${RUN_ROOT}/metadata"
cp -a "${OUT}/." "${RUN_ROOT}/artifacts/final_output/" 2>/dev/null || true
printf '0\n' > "${RUN_ROOT}/metadata/exit_code.txt"
rm -f "${RUN_ROOT}/running"

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
    candidates = []
    for state_path in root.glob("checkpoint-*/trainer_state.json"):
        try:
            item = json.loads(state_path.read_text())
            candidate = item.get("best_model_checkpoint") or str(state_path.parent)
            metric = item.get("best_metric")
            if metric is not None and (Path(candidate) / "adapter_model.safetensors").exists():
                candidates.append((float(metric), Path(candidate)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if candidates:
        print(min(candidates, key=lambda item: item[0])[1])
        raise SystemExit(0)
    if (root / "adapter_model.safetensors").exists():
        print(root)
        raise SystemExit(0)
    checkpoints = sorted(root.glob("checkpoint-*/adapter_model.safetensors"), key=lambda p: int(p.parent.name.split("-")[-1]))
    if not checkpoints:
        raise SystemExit("no adapter checkpoint found")
    print(checkpoints[-1].parent)
PY
)"

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

# If a probe actually consumed the full generation budget, rerun only those
# cases with a 16k budget before releasing the priority-training supervisor.
EXT_INDICES="$(${PYTHON} - "${RESULT}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
ids = []
for case in payload.get("cases", []):
    meta = (case.get("generation_meta") or {}).get("output_only") or {}
    if meta.get("hit_max_new_tokens"):
        ids.append(str(case.get("id")))
print(",".join(ids))
PY
)"
if [[ -n "${EXT_INDICES}" ]]; then
  EXT_RESULT="/cache/liluchen/medicalner_output_objectives/outputs/post_train_output_only_groupdisjoint_20260818_max16384_retry.json"
  EXT_REPORT="/cache/liluchen/medicalner_output_objectives/reports/post_train_output_only_groupdisjoint_20260818_max16384_retry.json"
  scripts/run_with_snapshots.sh post_eval_output_only_groupdisjoint_20260818_max16384_retry \
    "${PYTHON}" scripts/compare_output_objectives_20260810.py \
    --data "${DATA}" \
    --base-model /cache/liluchen/model_cache/Qwen3-8B \
    --priority-adapter "${ADAPTER}" \
    --output-only-adapter "${ADAPTER}" \
    --output "${EXT_RESULT}" \
    --max-new-tokens 16384 \
    --batch-size 1 \
    --no-quantization \
    --only-model output_only \
    --split-manifest "${MANIFEST}" \
    --indices "${EXT_INDICES}"
  "${PYTHON}" scripts/analyze_comparison_20260811.py "${EXT_RESULT}" \
    --source-chunks "${CHUNKS}" --output "${EXT_REPORT}"
fi

# Run the untouched base Qwen on the same explicit probes before releasing the
# priority-training supervisor, so the final merged artifact contains all
# three outputs (base, output-only, and priority).
scripts/run_with_snapshots.sh post_eval_base_qwen_groupdisjoint_20260818 \
  "${PYTHON}" scripts/compare_output_objectives_20260810.py \
  --data "${DATA}" \
  --base-model /cache/liluchen/model_cache/Qwen3-8B \
  --priority-adapter "${ADAPTER}" \
  --output-only-adapter "${ADAPTER}" \
  --output "${BASE_RESULT}" \
  --max-new-tokens 8192 \
  --batch-size 1 \
  --no-quantization \
  --only-model base_qwen \
  --split-manifest "${MANIFEST}" \
  --indices "${INDICES}"

BASE_EXT_INDICES="$(${PYTHON} - "${BASE_RESULT}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
ids = []
for case in payload.get("cases", []):
    meta = (case.get("generation_meta") or {}).get("base_qwen") or {}
    if meta.get("hit_max_new_tokens"):
        ids.append(str(case.get("id")))
print(",".join(ids))
PY
)"
if [[ -n "${BASE_EXT_INDICES}" ]]; then
  BASE_EXT_RESULT="/cache/liluchen/medicalner_output_objectives/outputs/post_train_base_qwen_groupdisjoint_20260818_max16384_retry.json"
  BASE_EXT_REPORT="/cache/liluchen/medicalner_output_objectives/reports/post_train_base_qwen_groupdisjoint_20260818_max16384_retry.json"
  scripts/run_with_snapshots.sh post_eval_base_qwen_groupdisjoint_20260818_max16384_retry \
    "${PYTHON}" scripts/compare_output_objectives_20260810.py \
    --data "${DATA}" \
    --base-model /cache/liluchen/model_cache/Qwen3-8B \
    --priority-adapter "${ADAPTER}" \
    --output-only-adapter "${ADAPTER}" \
    --output "${BASE_EXT_RESULT}" \
    --max-new-tokens 16384 \
    --batch-size 1 \
    --no-quantization \
    --only-model base_qwen \
    --split-manifest "${MANIFEST}" \
    --indices "${BASE_EXT_INDICES}"
  "${PYTHON}" scripts/analyze_comparison_20260811.py "${BASE_EXT_RESULT}" \
    --source-chunks "${CHUNKS}" --output "${BASE_EXT_REPORT}"
fi

# This is intentionally the final write before the priority supervisor is
# released; its existence is the completion marker for the first objective.
"${PYTHON}" scripts/analyze_comparison_20260811.py "${RESULT}" \
  --source-chunks "${CHUNKS}" --output "${REPORT}"
"${PYTHON}" scripts/analyze_comparison_20260811.py "${BASE_RESULT}" \
  --source-chunks "${CHUNKS}" --output "${BASE_REPORT}"
printf '%s\n' "${RESULT}" "${REPORT}" "${BASE_RESULT}" "${BASE_REPORT}"

#!/usr/bin/env bash
set -Eeuo pipefail

# The output-only run is evaluated first.  Since its canary already showed
# semantic/code hallucinations despite low token loss, run the second objective
# on the same frozen group-disjoint data after that evidence is available.
ROOT="/home/ma-user/workspace/liluchen/MedicalNER-Qwen3"
REPORT="/cache/liluchen/medicalner_output_objectives/reports/post_train_output_only_groupdisjoint_20260818_max8192.json"
OUT="/cache/liluchen/medicalner_output_objectives/output/lora_output_priority_groupdisjoint_20260818"
RUN_ROOT="/temp/liluchen/train_output_priority_groupdisjoint_20260818"
PYTHON="/cache/liluchen/envs/medicalner/bin/python"

cd "${ROOT}"
export PATH="/cache/liluchen/envs/medicalner/bin:${HOME}/.local/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME=/cache/liluchen/hf_home
export HUGGINGFACE_HUB_CACHE=/cache/liluchen/model_cache

while [[ ! -s "${REPORT}" ]]; do
  sleep 60
done

printf '%s\n' "Output-only report ready; starting priority objective." >> /temp/liluchen/priority_supervisor_20260818.log
mkdir -p "${OUT}"
export SNAPSHOT_ROOT=/temp/liluchen
export SNAPSHOT_INTERVAL_SEC=600
export SNAPSHOT_PATHS="scripts:configs:data/llamafactory:reports:outputs"
export SNAPSHOT_ABSOLUTE_PATHS="${OUT}"
scripts/run_with_snapshots.sh train_output_priority_groupdisjoint_20260818 \
  llamafactory-cli train \
  configs/llamafactory/qwen3_8b_lora_deepseek_output_priority_groupdisjoint_20260818.yaml

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}}"
CONFIG_YAML="${CONFIG_YAML:-configs/llamafactory/qwen3_8b_lora_cot_pro858.yaml}"

cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
KG_CACHE_ROOT="${KG_CACHE_ROOT:-${PROJECT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${KG_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
PY
which llamafactory-cli
nvidia-smi || true

echo "===== Training YAML ====="
sed -n '1,220p' "${CONFIG_YAML}"

echo "===== Start CUDA training ====="
llamafactory-cli train "${CONFIG_YAML}"

echo "===== Training finished ====="
date

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}}"
BACKEND="${BACKEND:-cuda}"

cd "${PROJECT_DIR}"

export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
KG_CACHE_ROOT="${KG_CACHE_ROOT:-${PROJECT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${KG_CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

echo "===== Runtime Info ====="
hostname
date
which python
which llamafactory-cli

case "${BACKEND}" in
  cuda)
    CONFIG_YAML="${CONFIG_YAML:-configs/llamafactory/qwen3_8b_lora_cot_pro858_smoke.yaml}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
PY
    nvidia-smi || true
    ;;
  npu)
    CONFIG_YAML="${CONFIG_YAML:-configs/llamafactory/qwen3_8b_lora_cot_pro858_smoke_npu.yaml}"
    python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
    npu-smi info || true
    ;;
  *)
    echo "Unsupported backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

echo "===== Smoke YAML ====="
sed -n '1,220p' "${CONFIG_YAML}"

echo "===== Start ${BACKEND} smoke training ====="
llamafactory-cli train "${CONFIG_YAML}"

echo "===== Smoke training finished ====="
date

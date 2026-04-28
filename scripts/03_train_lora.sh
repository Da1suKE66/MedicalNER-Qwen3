#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

KG_CACHE_ROOT="${KG_CACHE_ROOT:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${KG_CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${KG_CACHE_ROOT}}"
export TMPDIR="${TMPDIR:-${KG_CACHE_ROOT}/tmp}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${TMPDIR}" models/adapters

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/03_train_lora.sh <specific|standard|sft>"
  exit 1
fi

case "$1" in
  specific)
    CONFIG="configs/llamafactory/qwen3_8b_lora_cot_specific.yaml"
    DATASET_JSON="data/llamafactory/kg_cot_specific_614.json"
    ;;
  standard)
    CONFIG="configs/llamafactory/qwen3_8b_lora_cot_standard.yaml"
    DATASET_JSON="data/llamafactory/kg_cot_standard_635.json"
    ;;
  sft)
    CONFIG="configs/llamafactory/qwen3_8b_lora_sft.yaml"
    DATASET_JSON="data/llamafactory/kg_ner_sft.json"
    ;;
  *)
    echo "Unknown training target: $1"
    echo "Expected one of: specific, standard, sft"
    exit 1
    ;;
esac

if [[ ! -f "${DATASET_JSON}" ]]; then
  echo "Missing training dataset: ${DATASET_JSON}" >&2
  echo "Run scripts/02_convert_to_llamafactory.sh first, or use the committed specific/standard datasets." >&2
  exit 1
fi

if grep -q "<thinking>" "${DATASET_JSON}" || grep -q "</thinking>" "${DATASET_JSON}"; then
  echo "ERROR: ${DATASET_JSON} contains <thinking> tags. Qwen3 training requires <think> tags." >&2
  exit 1
fi

llamafactory-cli train "${CONFIG}"

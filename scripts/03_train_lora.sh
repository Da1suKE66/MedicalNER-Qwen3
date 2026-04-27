#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/03_train_lora.sh <specific|standard|sft>"
  exit 1
fi

case "$1" in
  specific)
    CONFIG="configs/llamafactory/qwen3_8b_lora_cot_specific.yaml"
    ;;
  standard)
    CONFIG="configs/llamafactory/qwen3_8b_lora_cot_standard.yaml"
    ;;
  sft)
    CONFIG="configs/llamafactory/qwen3_8b_lora_sft.yaml"
    ;;
  *)
    echo "Unknown training target: $1"
    echo "Expected one of: specific, standard, sft"
    exit 1
    ;;
esac

llamafactory-cli train "${CONFIG}"

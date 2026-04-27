#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/02_convert_to_llamafactory.sh <generated_filtered_json> <dataset_name> [cot_style]"
  echo "Example: bash scripts/02_convert_to_llamafactory.sh data/generated/cot_filtered_merged.json kg_cot_specific specific"
  exit 1
fi

INPUT_JSON="$1"
DATASET_NAME="$2"
COT_STYLE="${3:-specific}"
OUTPUT_JSON="data/llamafactory/${DATASET_NAME}.json"
DATASET_INFO="data/llamafactory/dataset_info.json"

PYTHONPATH=src python3 -m kg_lora.convert_to_llamafactory \
  --input "${INPUT_JSON}" \
  --output "${OUTPUT_JSON}" \
  --dataset-info "${DATASET_INFO}" \
  --dataset-name "${DATASET_NAME}" \
  --fallback-cot-style "${COT_STYLE}"

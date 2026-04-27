#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

INPUT_JSON="${1:-${KG_DATA_PATH:-data/raw/mental_disorders.json}}"

COT_STYLE=specific bash scripts/01_generate_cot_data.sh "${INPUT_JSON}" data/generated

FILTERED_JSON="$(find data/generated -maxdepth 1 -type f -name 'cot_filtered_*.json' -print | sort | tail -1)"
if [[ -z "${FILTERED_JSON}" ]]; then
  echo "No filtered generated JSON found under data/generated."
  exit 1
fi

bash scripts/02_convert_to_llamafactory.sh "${FILTERED_JSON}" kg_cot_specific_614 specific
bash scripts/03_train_lora.sh specific

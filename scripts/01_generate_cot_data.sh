#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

INPUT_JSON="${1:-${KG_DATA_PATH:-data/raw/mental_disorders_20251125_165535.json}}"
OUTPUT_DIR="${2:-data/generated}"
COT_STYLE="${COT_STYLE:-specific}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
WORKERS="${WORKERS:-4}"
MODEL_NAME="${MODEL_NAME:-gemini-3-flash-preview}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4000}"

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "Input data not found: ${INPUT_JSON}"
  echo "Put a source JSON file under data/raw/ or pass it as the first argument."
  exit 1
fi

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --workers "${WORKERS}"
  --model_name "${MODEL_NAME}"
  --max_output_tokens "${MAX_OUTPUT_TOKENS}"
  --enable_cot
  --cot_style "${COT_STYLE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

PYTHONPATH=src KG_DATA_PATH="${INPUT_JSON}" python3 -m kg_lora.generate_cot_data "${ARGS[@]}"

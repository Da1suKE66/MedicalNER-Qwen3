#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

DATA_PATH="${1:-${KG_DATA_PATH:-data/raw/mental_disorders.json}}"
SAMPLES_PATH="${2:-data/samples/sample_eval_cases.json}"
OUTPUT_PREFIX="${3:-outputs/qwen_compare_$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
SPECIFIC_ADAPTER="${SPECIFIC_ADAPTER:-models/adapters/qwen3-8b-cot-specific}"
STANDARD_ADAPTER="${STANDARD_ADAPTER:-models/adapters/qwen3-8b-cot-standard}"

PYTHONPATH=src python3 -m kg_lora.compare_qwen_outputs \
  --data "${DATA_PATH}" \
  --samples "${SAMPLES_PATH}" \
  --base-model "${BASE_MODEL}" \
  --specific-adapter "${SPECIFIC_ADAPTER}" \
  --standard-adapter "${STANDARD_ADAPTER}" \
  --output "${OUTPUT_PREFIX}"

for name in base_qwen specific_adapter standard_adapter; do
  PYTHONPATH=src python3 -m kg_lora.analyze_compare_outputs --input "${OUTPUT_PREFIX}_${name}.json"
done

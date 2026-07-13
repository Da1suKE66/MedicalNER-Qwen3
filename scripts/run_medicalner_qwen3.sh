#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="cuda"
TASK="train"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_DIR}}"
CONFIG_YAML="${CONFIG_YAML:-}"
DEVICE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_medicalner_qwen3.sh [options]

Options:
  --backend <cuda|npu>     Hardware backend (default: cuda)
  --task <train|predict|smoke>
                            Operation to run (default: train)
  --project-dir <path>     Repository root (default: detected automatically)
  --config <path>          Override the selected LLaMA-Factory YAML
  --device <id[,id...]>    CUDA or Ascend visible device list
  --dry-run                Print the selected command without executing it
  -h, --help               Show this help

Examples:
  bash scripts/run_medicalner_qwen3.sh
  bash scripts/run_medicalner_qwen3.sh --backend cuda --task predict --device 0
  bash scripts/run_medicalner_qwen3.sh --backend npu --task train --device 0
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend|--accelerator)
      require_value "$@"
      BACKEND="$2"
      shift 2
      ;;
    --task)
      require_value "$@"
      TASK="$2"
      shift 2
      ;;
    --project-dir)
      require_value "$@"
      PROJECT_DIR="$2"
      shift 2
      ;;
    --config)
      require_value "$@"
      CONFIG_YAML="$2"
      shift 2
      ;;
    --device)
      require_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${BACKEND}" in
  cuda|nvidia|gpu)
    BACKEND="cuda"
    ;;
  npu|ascend|huawei)
    BACKEND="npu"
    ;;
  *)
    echo "Unsupported backend: ${BACKEND}. Expected cuda or npu." >&2
    exit 2
    ;;
esac

case "${TASK}" in
  train)
    if [[ "${BACKEND}" == "cuda" ]]; then
      TARGET_SCRIPT="scripts/run_medicalner_qwen3_pro858.sh"
    else
      TARGET_SCRIPT="scripts/run_medicalner_qwen3_pro858_npu.sh"
    fi
    ;;
  predict)
    if [[ "${BACKEND}" == "cuda" ]]; then
      TARGET_SCRIPT="scripts/run_medicalner_qwen3_pro858_predict.sh"
    else
      TARGET_SCRIPT="scripts/run_medicalner_qwen3_pro858_predict_npu.sh"
    fi
    ;;
  smoke)
    TARGET_SCRIPT="scripts/run_medicalner_qwen3_smoke.sh"
    ;;
  *)
    echo "Unsupported task: ${TASK}. Expected train, predict, or smoke." >&2
    exit 2
    ;;
esac

export PROJECT_DIR
export BACKEND
[[ -z "${CONFIG_YAML}" ]] || export CONFIG_YAML

if [[ -n "${DEVICE}" ]]; then
  if [[ "${BACKEND}" == "cuda" ]]; then
    export CUDA_VISIBLE_DEVICES="${DEVICE}"
  else
    export ASCEND_VISIBLE_DEVICES="${DEVICE}"
    export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
  fi
fi

echo "backend    : ${BACKEND}"
echo "task       : ${TASK}"
echo "project dir: ${PROJECT_DIR}"
echo "script     : ${TARGET_SCRIPT}"
[[ -z "${CONFIG_YAML}" ]] || echo "config     : ${CONFIG_YAML}"
[[ -z "${DEVICE}" ]] || echo "device     : ${DEVICE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

cd "${PROJECT_DIR}"
exec bash "${TARGET_SCRIPT}"

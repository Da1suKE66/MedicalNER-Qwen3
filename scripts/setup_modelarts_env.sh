#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/cache/llc/KG}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PIP_INDEX_URL="${PIP_INDEX_URL:-http://repo.myhuaweicloud.com/repository/pypi/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-repo.myhuaweicloud.com}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

echo "==> Preparing conda env: ${ENV_PREFIX}"
if [[ ! -d "${ENV_PREFIX}/conda-meta" ]]; then
  conda create -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip -y
elif [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda install -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip -y
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ENV_PREFIX}/.cache/matplotlib}"
hash -r
mkdir -p "${MPLCONFIGDIR}"

echo "==> Using python: $(command -v python)"
python -V
python -m pip -V

echo "==> Installing project and training dependencies"
python -m pip install --no-user --upgrade pip setuptools wheel \
  -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

python -m pip install --no-user -r "${ROOT_DIR}/requirements.txt" \
  -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

echo "==> Checking dependency consistency"
python -m pip check

echo "==> Checking torch and CUDA"
python - <<'PY'
import torch

print("torch file:", torch.__file__)
print("torch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to torch. Check nvidia-smi and job GPU settings.")

if not torch.cuda.is_bf16_supported():
    raise SystemExit("bf16 is not supported by this GPU/runtime.")
PY

echo "==> Checking LLaMA-Factory"
llamafactory-cli version

echo
echo "Environment is ready."
echo "Run training with:"
echo "  cd ${ROOT_DIR}"
echo "  PYTHONNOUSERSITE=1 bash scripts/03_train_lora.sh specific"

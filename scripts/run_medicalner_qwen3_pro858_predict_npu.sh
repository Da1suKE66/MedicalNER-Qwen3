#!/usr/bin/env bash
# 在 NPU 容器内部运行的 predict 入口（被 scripts/submit_medicalner_qwen3_pro858_predict_npu.sh
# 提交的 vc 任务调用，也可在调试机的容器里直接 bash 这一行手动跑）。
#
# 加载训练好的 LoRA adapter，在 treatment predict 集上跑生成
# （Cochrane 6 条、Embase 8 条、PubMed 8 条），
# 落 generated_predictions.jsonl
# 并做一个粗略的 JSON 合法率统计，方便肉眼检查输出格式。
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/aistor/sjtu/hpc_stor01/home/wangxiran/projects/MedicalNER-Qwen3}"
CONFIG_YAML="${CONFIG_YAML:-configs/llamafactory/predict_qwen3_8b_cot_pro858_treatment_22_npu.yaml}"
ADAPTER="${ADAPTER:-models/adapters/qwen3-8b-cot-pro858}"
OUTDIR="${OUTDIR:-models/predict/qwen3-8b-cot-pro858-treatment-22-npu}"

cd "${PROJECT_DIR}"

# 缓存路径放共享存储，别落到容器临时目录
KG_CACHE_ROOT="${KG_CACHE_ROOT:-${PROJECT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${KG_CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${KG_CACHE_ROOT}}"
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${KG_CACHE_ROOT}/matplotlib}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" \
         "${MPLCONFIGDIR}" "${OUTDIR}"
mkdir -p "${TMPDIR}" 2>/dev/null || true

export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Predict 单卡足够，关掉 torchrun
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-0}"

echo "===== Runtime Info ====="
hostname
date
echo "PROJECT_DIR : ${PROJECT_DIR}"
echo "CONFIG_YAML : ${CONFIG_YAML}"
echo "ADAPTER     : ${ADAPTER}"
echo "OUTDIR      : ${OUTDIR}"
echo "ASCEND_VISIBLE_DEVICES   = ${ASCEND_VISIBLE_DEVICES:-<unset>}"
echo "ASCEND_RT_VISIBLE_DEVICES= ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
which python3
which llamafactory-cli
npu-smi info || true

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
try:
    import torch_npu
    print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
    print("torch_npu.npu.device_count():", torch_npu.npu.device_count())
except Exception as e:
    print("torch_npu import failed:", e)
if hasattr(torch, "npu"):
    print("torch.npu.is_available():", torch.npu.is_available())
    print("torch.npu.device_count():", torch.npu.device_count())
PY

# === Runtime NPU FA patch via import-time shadowing ===
# 容器内不是 root，不能写 site-packages。改用 sitecustomize + meta path finder：
# Python 解释器启动时（site.py）会 import sitecustomize；我们的 finder 把
# llamafactory.model.model_utils.npu_flash_attn 重定向到 repo 里 docker/npu_flash_attn.py。
# 镜像 / site-packages 一字不动，每次任务起新容器都自动生效。
echo "===== Apply runtime NPU FA patch (import shadow) ====="
FA_PATCH_SRC="${PROJECT_DIR}/docker/npu_flash_attn.py"
if [[ ! -f "${FA_PATCH_SRC}" ]]; then
    echo "[predict] ERROR: ${FA_PATCH_SRC} not found in repo." >&2
    exit 1
fi
if ! grep -q "attn_mask is not None" "${FA_PATCH_SRC}"; then
    echo "[predict] ERROR: ${FA_PATCH_SRC} does not contain the fixed guard. Did the repo file get reverted?" >&2
    exit 1
fi

SHADOW_DIR="${KG_CACHE_ROOT}/py_shadow"
mkdir -p "${SHADOW_DIR}"
cat >"${SHADOW_DIR}/sitecustomize.py" <<EOF
import importlib.util, sys
_TARGET = "llamafactory.model.model_utils.npu_flash_attn"
_FIXED = r"${FA_PATCH_SRC}"

class _NpuFAFinder:
    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if name == _TARGET:
            return importlib.util.spec_from_file_location(name, _FIXED)
        return None

sys.meta_path.insert(0, _NpuFAFinder)
print("[sitecustomize] NPU FA shadow active ->", _FIXED, flush=True)
EOF

export PYTHONPATH="${SHADOW_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
echo "[predict] PYTHONPATH=${PYTHONPATH}"

# 自检：新解释器进程能否解析到修复版文件
python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("llamafactory.model.model_utils.npu_flash_attn")
origin = spec.origin if spec else None
print("[predict] resolved npu_flash_attn ->", origin)
assert origin and "/projects/MedicalNER-Qwen3/docker/npu_flash_attn.py" in origin, \
    f"import shadow not active, still resolving to {origin}"
print("[predict] shadow OK")
PY

echo "===== Adapter files ====="
ls -lh "${ADAPTER}" 2>/dev/null | head -n 50 || {
    echo "ERROR: adapter dir not found: ${ADAPTER}" >&2
    exit 1
}

echo "===== Trainer state / best checkpoint ====="
ADAPTER="${ADAPTER}" python3 - <<'PY'
import json, os
from pathlib import Path

p = Path(os.environ["ADAPTER"]) / "trainer_state.json"
if p.exists():
    s = json.loads(p.read_text())
    print("best_model_checkpoint:", s.get("best_model_checkpoint"))
    print("best_metric:", s.get("best_metric"))
    print("global_step:", s.get("global_step"))
else:
    print("trainer_state.json not found at", p)
PY

echo "===== Predict YAML ====="
sed -n '1,240p' "${CONFIG_YAML}"

echo "===== Start llamafactory-cli train (do_predict) ====="
llamafactory-cli train "${CONFIG_YAML}"

echo "===== Prediction finished ====="
date

echo "===== Output files ====="
ls -lh "${OUTDIR}" || true

echo "===== Preview generated predictions ====="
OUTDIR="${OUTDIR}" python3 - <<'PY'
import json, os
from pathlib import Path

p = Path(os.environ["OUTDIR"]) / "generated_predictions.jsonl"
print("prediction file:", p)
print("exists:", p.exists())
if not p.exists():
    raise SystemExit(0)

with p.open(encoding="utf-8") as f:
    for idx, line in zip(range(3), f):
        obj = json.loads(line)
        print("\n" + "=" * 100)
        print("CASE", idx + 1)
        print("-" * 40)
        print("PROMPT preview:")
        print(obj.get("prompt", "")[:1000])
        print("-" * 40)
        print("PREDICT preview:")
        print(obj.get("predict", "")[:2000])
        print("-" * 40)
        print("LABEL preview:")
        print(obj.get("label", "")[:1000])
PY

echo "===== JSON format quick check ====="
OUTDIR="${OUTDIR}" python3 - <<'PY'
import json, os, re
from pathlib import Path

p = Path(os.environ["OUTDIR"]) / "generated_predictions.jsonl"

def clean_json_text(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

if not p.exists():
    print("generated_predictions.jsonl not found")
    raise SystemExit(0)

total = 0
valid = 0
bad_examples = []
with p.open(encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        total += 1
        obj = json.loads(line)
        pred = obj.get("predict", "")
        cleaned = clean_json_text(pred)
        try:
            json.loads(cleaned)
            valid += 1
        except Exception as e:
            if len(bad_examples) < 5:
                bad_examples.append((i, str(e), pred[:1000]))

print(f"Total: {total}")
print(f"Valid JSON: {valid}")
print(f"Invalid JSON: {total - valid}")
if total:
    print(f"Valid ratio: {valid / total:.2%}")

for line_no, err, preview in bad_examples:
    print("\n" + "=" * 80)
    print("Line:", line_no)
    print("Error:", err)
    print("Predict preview:")
    print(preview)
PY

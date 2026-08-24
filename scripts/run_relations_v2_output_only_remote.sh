#!/usr/bin/env bash
set -Eeuo pipefail

# Self-contained launcher for the pinned remote node.  It records the exact
# command/environment, audits the 16K cutoff before training, starts the
# periodic snapshot loop, and keeps LLaMA-Factory's native checkpoints/plots.

WORKSPACE="${WORKSPACE:-/home/ma-user/workspace/liluchen/MedicalNER-Qwen3}"
ENV_PREFIX="${ENV_PREFIX:-/cache/liluchen/envs/medicalner}"
RUN_ROOT="${RUN_ROOT:-/cache/liluchen/medicalner_relations_v2}"
CONFIG="${CONFIG:-$WORKSPACE/configs/llamafactory/qwen3_8b_lora_deepseek_relations_v2_output_only_20260825.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/output/lora_output_only_20260825}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"
SNAP_ROOT="${SNAP_ROOT:-/temp/liluchen/medicalner_relations_v2/snapshots}"
RESUME="${RESUME:-0}"

export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/cache/liluchen/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/cache/liluchen/model_cache}"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

mkdir -p "$RUN_ROOT" "$LOG_DIR" "$REPORT_DIR" "$OUTPUT_DIR" "$SNAP_ROOT"
cd "$WORKSPACE"

MODEL_DIR="$(sed -n 's/^model_name_or_path:[[:space:]]*//p' "$CONFIG" | head -1)"
DATA_DIR="$(sed -n 's/^dataset_dir:[[:space:]]*//p' "$CONFIG" | head -1)"
TRAIN_FILE="$DATA_DIR/deepseek_watermark_20260804_182312_output_only_full_groupdisjoint_train_relations_v2.json"
DEV_FILE="$DATA_DIR/deepseek_watermark_20260804_182312_output_only_full_groupdisjoint_dev_relations_v2.json"

for required in "$MODEL_DIR/config.json" "$MODEL_DIR/tokenizer.json" "$TRAIN_FILE" "$DEV_FILE" "$CONFIG"; do
    [[ -e "$required" ]] || { echo "MISSING $required" >&2; exit 2; }
done

date -Is > "$RUN_ROOT/run_started_at.txt"
{
    echo "host=$(hostname)"
    echo "pwd=$PWD"
    echo "config=$CONFIG"
    echo "model_dir=$MODEL_DIR"
    echo "train_file=$TRAIN_FILE"
    echo "dev_file=$DEV_FILE"
    echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
    echo "train_sha256=$(sha256sum "$TRAIN_FILE" | awk '{print $1}')"
    echo "dev_sha256=$(sha256sum "$DEV_FILE" | awk '{print $1}')"
    echo "python=$($ENV_PREFIX/bin/python --version 2>&1)"
    echo "llamafactory=$($ENV_PREFIX/bin/llamafactory-cli version 2>&1 | tail -1)"
    echo "torch=$($ENV_PREFIX/bin/python -c 'import torch; print(torch.__version__)')"
    git rev-parse HEAD 2>/dev/null || true
} | tee "$RUN_ROOT/run_metadata.txt"
nvidia-smi > "$RUN_ROOT/gpu_before.txt" || true

"$ENV_PREFIX/bin/python" "$WORKSPACE/scripts/audit_training_cutoff.py" \
    --data "$TRAIN_FILE" --data "$DEV_FILE" \
    --tokenizer "$MODEL_DIR" --cutoff-len 16384 \
    --output "$REPORT_DIR/cutoff_audit.json" \
    > "$LOG_DIR/cutoff_audit.log" 2>&1
cp -a "$REPORT_DIR/cutoff_audit.json" "$SNAP_ROOT/"

# The snapshot loop uses a separate process so an SSH disconnect cannot lose
# the latest trainer state or GPU evidence.
if [[ -x "$WORKSPACE/scripts/remote_snapshot_relations_v2.sh" ]]; then
    INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}" RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
        SNAP_ROOT="$SNAP_ROOT" WORKSPACE="$WORKSPACE" LOG_DIR="$LOG_DIR" \
        nohup bash "$WORKSPACE/scripts/remote_snapshot_relations_v2.sh" \
        > "$LOG_DIR/snapshot_loop.log" 2>&1 &
    echo $! > "$RUN_ROOT/snapshot_loop.pid"
fi

args=("$CONFIG")
if [[ "$RESUME" == "1" ]]; then
    latest="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [[ -n "$latest" ]]; then
        args+=("--resume_from_checkpoint" "$latest")
        echo "resume_from_checkpoint=$latest" | tee "$RUN_ROOT/resume.txt"
    fi
fi

set +e
"$ENV_PREFIX/bin/llamafactory-cli" train "${args[@]}" 2>&1 | tee "$LOG_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "$rc" > "$RUN_ROOT/train_exit_code.txt"
date -Is > "$RUN_ROOT/run_finished_at.txt"
nvidia-smi > "$RUN_ROOT/gpu_after.txt" || true

if [[ -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$SNAP_ROOT/final"
    cp -a "$OUTPUT_DIR/." "$SNAP_ROOT/final/"
fi
exit "$rc"

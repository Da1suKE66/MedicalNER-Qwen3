#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${WORKSPACE:-/home/ma-user/workspace/liluchen/MedicalNER-Qwen3}"
ENV_PREFIX="${ENV_PREFIX:-/cache/liluchen/envs/medicalner}"
RUN_ROOT="${RUN_ROOT:-/cache/liluchen/medicalner_relations_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/output/lora_output_only_20260825}"
REPORT_DIR="${REPORT_DIR:-$RUN_ROOT/reports}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
MODEL_DIR="${MODEL_DIR:-/cache/liluchen/model_cache/models/Qwen--Qwen3-8B/snapshots/master}"
DEV_DATA="$RUN_ROOT/data/deepseek_watermark_20260804_182312_output_only_full_groupdisjoint_dev_relations_v2.json"
TRAIN_DATA="$RUN_ROOT/data/deepseek_watermark_20260804_182312_output_only_full_groupdisjoint_train_relations_v2.json"

export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$REPORT_DIR" "$LOG_DIR"
cd "$WORKSPACE"

# Wait for the launcher to finish.  A nonzero run is still diagnosable: use
# the newest checkpoint if one exists instead of silently dropping evaluation.
while [[ ! -f "$RUN_ROOT/train_exit_code.txt" ]]; do
    sleep 60
done
exit_code="$(cat "$RUN_ROOT/train_exit_code.txt")"
adapter="$OUTPUT_DIR"
if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
    adapter="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)"
fi
[[ -n "$adapter" && -f "$adapter/adapter_model.safetensors" ]] || {
    echo "No adapter available after train exit code $exit_code" | tee "$LOG_DIR/post_eval.log"
    exit 3
}

echo "train_exit_code=$exit_code" | tee "$REPORT_DIR/post_eval_metadata.txt"
echo "adapter=$adapter" | tee -a "$REPORT_DIR/post_eval_metadata.txt"
sha256sum "$adapter/adapter_model.safetensors" | tee -a "$REPORT_DIR/post_eval_metadata.txt"

# These probes include known long diagnostic/exclusion cases and the historical
# relation cases that were previously used for scorer audits when present in
# the group-disjoint dev/train positional manifests.
dev_indices="4,15,40,99,143,147,0,74"
train_indices="251,265,327,589,1284,1344"

run_compare() {
    local split_name="$1" data="$2" indices="$3" output="$4"
    "$ENV_PREFIX/bin/python" "$WORKSPACE/scripts/compare_output_objectives_20260810.py" \
        --data "$data" \
        --base-model "$MODEL_DIR" \
        --priority-adapter "$adapter" \
        --output-only-adapter "$adapter" \
        --output "$output" \
        --max-new-tokens 16384 \
        --stop-on-structured-complete \
        --no-quantization \
        --only-model output_only \
        --batch-size 1 \
        --indices "$indices" \
        2>&1 | tee "$LOG_DIR/post_eval_${split_name}.log"
    "$ENV_PREFIX/bin/python" "$WORKSPACE/scripts/audit_generation_closure.py" \
        "$output" --output "$REPORT_DIR/closure_${split_name}.json" \
        | tee -a "$LOG_DIR/post_eval_${split_name}.log"
    "$ENV_PREFIX/bin/python" "$WORKSPACE/scripts/audit_relation_scorer_20260819.py" \
        "$output" --model-field output_only --all-cases \
        --output "$REPORT_DIR/relation_${split_name}.json" \
        --raw-output "$REPORT_DIR/raw_${split_name}.json" \
        2>&1 | tee -a "$LOG_DIR/post_eval_${split_name}.log"
}

run_compare dev "$DEV_DATA" "$dev_indices" "$REPORT_DIR/comparison_dev_probe.json"
run_compare train "$TRAIN_DATA" "$train_indices" "$REPORT_DIR/comparison_train_probe.json"

cp -a "$OUTPUT_DIR/." "$RUN_ROOT/final/" 2>/dev/null || true
date -Is > "$REPORT_DIR/post_eval_finished_at.txt"

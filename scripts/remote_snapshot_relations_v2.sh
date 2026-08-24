#!/usr/bin/env bash
set -Eeuo pipefail

# Periodically copy the active run's recoverable state to /temp.  LLaMA-
# Factory checkpoints are already durable under /cache; this loop adds a
# second location for logs, plots, trainer state, config and GPU evidence.

INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
RUN_ROOT="${RUN_ROOT:-/cache/liluchen/medicalner_relations_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/output/lora_output_only_20260825}"
SNAP_ROOT="${SNAP_ROOT:-/temp/liluchen/medicalner_relations_v2/snapshots}"
WORKSPACE="${WORKSPACE:-/home/ma-user/workspace/liluchen/MedicalNER-Qwen3}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"

mkdir -p "$SNAP_ROOT/latest" "$SNAP_ROOT/periodic" "$LOG_DIR"

copy_tree() {
    local source="$1" destination="$2"
    mkdir -p "$destination"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$source/" "$destination/"
    else
        cp -a "$source/." "$destination/"
    fi
}

snapshot() {
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SNAP_ROOT/periodic/$stamp"
    if [[ -d "$OUTPUT_DIR" ]]; then
        copy_tree "$OUTPUT_DIR" "$SNAP_ROOT/latest/output"
        copy_tree "$OUTPUT_DIR" "$SNAP_ROOT/periodic/$stamp/output"
    fi
    if [[ -d "$LOG_DIR" ]]; then
        copy_tree "$LOG_DIR" "$SNAP_ROOT/latest/logs"
        copy_tree "$LOG_DIR" "$SNAP_ROOT/periodic/$stamp/logs"
    fi
    for file in \
        "$WORKSPACE/configs/llamafactory/qwen3_8b_lora_deepseek_relations_v2_output_only_20260825.yaml" \
        "$WORKSPACE/data/llamafactory/relations_v2_output_only/relations_v2_build_report.json" \
        "$WORKSPACE/data/llamafactory/relations_v2_output_only/relations_v2_active_policy.json" \
        "$RUN_ROOT/reports/cutoff_audit.json"; do
        if [[ -f "$file" ]]; then
            cp -a "$file" "$SNAP_ROOT/latest/"
            cp -a "$file" "$SNAP_ROOT/periodic/$stamp/"
        fi
    done
    {
        date -Is
        printf 'host='; hostname
        printf 'run_root=%s\noutput_dir=%s\n' "$RUN_ROOT" "$OUTPUT_DIR"
        nvidia-smi || true
    } > "$SNAP_ROOT/latest/gpu_snapshot.txt"
    cp -a "$SNAP_ROOT/latest/gpu_snapshot.txt" "$SNAP_ROOT/periodic/$stamp/gpu_snapshot.txt"
    find "$SNAP_ROOT/periodic" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | tail -20 > "$SNAP_ROOT/latest/periodic_snapshots.txt"
}

snapshot
while true; do
    sleep "$INTERVAL_SECONDS"
    snapshot
done

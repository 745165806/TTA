#!/bin/bash
# Confirmatory multi-domain launcher with a dynamic (dataset, method) job queue.
#
# Jobs (dataset x method) are claimed atomically by whichever GPU is free, so
# large datasets (e.g. DF) do not serialize one GPU while others idle.  This
# launcher only DISPATCHES jobs; it does not re-tune any parameter (workers read
# baselines/locked_port_configs.json only).
#
# Usage: run_confirmatory.sh <RUN_DIR> [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_DIR="${1:?usage: run_confirmatory.sh RUN_DIR [--dry-run]}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -ne 4 ]; then
    echo "ERROR: confirmatory requires exactly 4 GPUs" >&2
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

DATASETS=(itw_target90 asv2021_la asv2021_df control_test)
METHODS=(norm_only_audio tent_audio_ep sar_audio_ep memo_audio_ep_full)

# Build the job queue file (one "dataset|method" per line).
QUEUE="$(mktemp)"
for ds in "${DATASETS[@]}"; do
    for m in "${METHODS[@]}"; do
        echo "$ds|$m" >> "$QUEUE"
    done
done
TOTAL_JOBS=$(wc -l < "$QUEUE")
COUNTER="$QUEUE.counter"
LOCK="$QUEUE.lock"
echo 0 > "$COUNTER"

echo "confirmatory jobs: $TOTAL_JOBS (4 datasets x 4 methods)"
if [ "$DRY_RUN" = "1" ]; then
    cat "$QUEUE"
    echo "DRY-RUN complete: no job dispatched."
    exit 0
fi

mkdir -p "$RUN_DIR/confirmatory/logs"

# Atomically claim the next job index under flock.
next_index() {
    exec 9>"$LOCK"
    flock 9
    local n
    n=$(<"$COUNTER")
    echo $((n + 1)) > "$COUNTER"
    flock -u 9
    exec 9>&-
    echo "$n"
}

worker() {
    local gpu=$1
    while true; do
        local idx
        idx=$(next_index)
        [ "$idx" -ge "$TOTAL_JOBS" ] && break
        local job ds m
        job=$(sed -n "$((idx + 1))p" "$QUEUE")
        ds=${job%%|*}
        m=${job##*|}
        echo "[GPU$gpu] job $idx/$TOTAL_JOBS: $ds / $m"
        if ! CUDA_VISIBLE_DEVICES="$gpu" python baselines/run_port.py \
                --method "$m" --split "$ds" \
                --output "$RUN_DIR/confirmatory/$ds/$m" \
                > "$RUN_DIR/confirmatory/logs/${ds}_${m}.log" 2>&1; then
            echo "[GPU$gpu] job $idx FAILED: $ds / $m" >&2
        fi
    done
}

export -f next_index worker
export QUEUE COUNTER LOCK TOTAL_JOBS RUN_DIR
for gpu in "${GPUS[@]}"; do
    worker "$gpu" &
done
wait
echo "confirmatory queue finished"

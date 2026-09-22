#!/bin/bash
# P1 pilot executor: four GPU variants -> p1_<variant>.jsonl, then aggregate.
set -euo pipefail

# Usage: run_p1_pilot.sh <GPU_LIST> <RUN_DIR> <LOG_DIR>
GPU_LIST="${1:?usage: run_p1_pilot.sh GPU_LIST RUN_DIR LOG_DIR}"
RUN_DIR="${2:?usage: run_p1_pilot.sh GPU_LIST RUN_DIR LOG_DIR}"
LOG_DIR="${3:?usage: run_p1_pilot.sh GPU_LIST RUN_DIR LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
VARIANTS=(taskaware_full taskaware_no_gate taskaware_no_source_keep taskaware_control)
if [ "${#GPUS[@]}" -lt 4 ]; then
    echo "ERROR: P1 pilot requires at least 4 GPUs, got ${#GPUS[@]}" >&2
    exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p "$RUN_DIR/p1" "$LOG_DIR"

PIDS=()
for i in 0 1 2 3; do
    gpu="${GPUS[$i]}"
    variant="${VARIANTS[$i]}"
    echo "P1: launch $variant -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python p1/pilot_worker.py \
        --variant "$variant" --output-dir "$RUN_DIR/p1" \
        > "$LOG_DIR/p1_gpu$gpu.log" 2>&1 &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: P1 pilot worker (pid $pid) failed" >&2
        exit 1
    fi
done
echo "P1: all pilot variants finished"

python p1/pilot_aggregate.py --run-dir "$RUN_DIR"

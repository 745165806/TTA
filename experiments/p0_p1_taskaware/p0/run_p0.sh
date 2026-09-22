#!/bin/bash
# P0 executor: launch the four label-free score workers (round-robin over the
# 25 frozen candidates) and aggregate only after every worker succeeds.
#
# NOTE: cache-based P0 adaptation runs on CPU; worker IDs are parallel process
# slots, not CUDA devices.
set -euo pipefail

# Usage: run_p0.sh <WORKER_LIST> <RUN_DIR> <LOG_DIR>
WORKER_LIST="${1:?usage: run_p0.sh WORKER_LIST RUN_DIR LOG_DIR}"
RUN_DIR="${2:?usage: run_p0.sh WORKER_LIST RUN_DIR LOG_DIR}"
LOG_DIR="${3:?usage: run_p0.sh WORKER_LIST RUN_DIR LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

IFS=',' read -r -a WORKERS <<< "$WORKER_LIST"
NUM_GROUPS="${#WORKERS[@]}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p "$RUN_DIR/p0" "$LOG_DIR"

PIDS=()
for i in "${!WORKERS[@]}"; do
    worker_id="${WORKERS[$i]}"
    echo "P0: launch group $i -> worker slot $worker_id"
    python p0/score_worker.py \
        --group "$i" --num-groups "$NUM_GROUPS" --output-dir "$RUN_DIR/p0" \
        > "$LOG_DIR/p0_worker$worker_id.log" 2>&1 &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: P0 worker (pid $pid) failed" >&2
        exit 1
    fi
done
echo "P0: all groups finished"

python p0/aggregate.py --run-dir "$RUN_DIR"
echo "P0: aggregate finished"

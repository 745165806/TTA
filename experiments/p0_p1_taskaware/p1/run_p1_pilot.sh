#!/bin/bash
# P1 pilot executor: four parallel variants -> p1_<variant>.jsonl, then aggregate.
#
# NOTE: cache-based P1 adaptation runs on CPU; worker IDs are parallel process
# slots, not CUDA devices.
set -euo pipefail

# Usage: run_p1_pilot.sh <WORKER_LIST> <RUN_DIR> <LOG_DIR>
WORKER_LIST="${1:?usage: run_p1_pilot.sh WORKER_LIST RUN_DIR LOG_DIR}"
RUN_DIR="${2:?usage: run_p1_pilot.sh WORKER_LIST RUN_DIR LOG_DIR}"
LOG_DIR="${3:?usage: run_p1_pilot.sh WORKER_LIST RUN_DIR LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$EXP_DIR"

IFS=',' read -r -a WORKERS <<< "$WORKER_LIST"
VARIANTS=(taskaware_full taskaware_no_gate taskaware_no_source_keep taskaware_control)
if [ "${#WORKERS[@]}" -ne 4 ]; then
    echo "ERROR: p0_p1_taskaware protocol requires exactly 4 parallel workers, got ${#WORKERS[@]}" >&2
    exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p "$RUN_DIR/p1" "$LOG_DIR"

PIDS=()
for i in 0 1 2 3; do
    worker_id="${WORKERS[$i]}"
    variant="${VARIANTS[$i]}"
    echo "P1: launch $variant -> worker slot $worker_id"
    python p1/pilot_worker.py \
        --variant "$variant" --output-dir "$RUN_DIR/p1" \
        > "$LOG_DIR/p1_worker$worker_id.log" 2>&1 &
    PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: P1 pilot worker (pid $pid) failed" >&2
        exit 1
    fi
done
echo "P1: all pilot variants finished"

set +e
python p1/pilot_aggregate.py --run-dir "$RUN_DIR"
PILOT_RC=$?
set -e

case "$PILOT_RC" in
    0)
        echo "P1_MECHANISM_PASS"
        exit 0
        ;;
    3)
        echo "P1_MECHANISM_FAIL"
        exit 3
        ;;
    *)
        echo "ERROR: P1 pilot aggregation failed technically, rc=$PILOT_RC" >&2
        exit "$PILOT_RC"
        ;;
esac

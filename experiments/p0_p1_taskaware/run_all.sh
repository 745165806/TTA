#!/bin/bash
# Unified launcher for the p0_p1_taskaware experiment.
#
# NOTE: cache-based P0/P1 adaptation runs on CPU.  Worker IDs are parallel
# process slots, NOT CUDA devices.  CUDA availability is only recorded for
# environment audit, never required.
#
# Phases:
#   0  preflight
#   1  P0 4-way parallel score
#   2  P0 aggregate
#   3  P1 unit/integration test
#   4  P1 4-way parallel pilot
#   5  P1 pilot aggregate
#   6  mechanism gate
#   7  confirmatory 4-way parallel (only if the pilot gate passes)
#   8  final aggregate
#
#   WORKER_LIST : comma-separated parallel process slot ids (default "0,1,2,3").
#                 GPU_LIST is accepted as a compatibility alias.
#   --dry-run   : run Phase 0 only (all validations, no adaptation work).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GIT_ROOT="$(git rev-parse --show-toplevel)"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
elif [ -n "${1:-}" ]; then
    echo "unknown argument: $1" >&2
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# WORKER_LIST parsing + validation (exactly 4 process slots).
# GPU_LIST is kept only as a backwards-compatible alias.
# ---------------------------------------------------------------------------
WORKER_LIST="${WORKER_LIST:-${GPU_LIST:-0,1,2,3}}"
WORKERS=()
IFS=',' read -r -a WORKERS <<< "$WORKER_LIST" || true
for i in "${!WORKERS[@]}"; do
    WORKERS[$i]="${WORKERS[$i]//[[:space:]]/}"
done
for w in "${WORKERS[@]}"; do
    if ! [[ "$w" =~ ^[0-9]+$ ]]; then
        echo "ERROR: WORKER_LIST entries must be integers, got '$w'" >&2
        exit 1
    fi
done
for ((i = 0; i < ${#WORKERS[@]}; i++)); do
    for ((j = i + 1; j < ${#WORKERS[@]}; j++)); do
        if [ "${WORKERS[$i]}" = "${WORKERS[$j]}" ]; then
            echo "ERROR: duplicate worker id '${WORKERS[$i]}' in WORKER_LIST" >&2
            exit 1
        fi
    done
done
NUM_WORKERS="${#WORKERS[@]}"

# The P0 aggregator reads exactly p0_group_{0..3}.jsonl; the P1 pilot maps the
# four ablation variants onto four fixed slots.  Keep the protocol fixed at 4.
if [ "$NUM_WORKERS" -ne 4 ]; then
    echo "ERROR: p0_p1_taskaware protocol requires exactly 4 parallel workers, got $NUM_WORKERS" >&2
    exit 1
fi

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

mkdir -p logs results
LOG_DIR="$SCRIPT_DIR/logs"
RUN_DIR="$SCRIPT_DIR/results/run_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "p0_p1_taskaware experiment"
[ "$DRY_RUN" = "1" ] && echo "mode : DRY-RUN (preflight only)"
echo "start: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "workers: ${WORKERS[*]} (num_workers=$NUM_WORKERS)"
echo "NOTE : cache-based P0/P1 adaptation runs on CPU;"
echo "       worker IDs are parallel process slots, not CUDA devices."
echo "run  : $RUN_DIR"
echo "commit: $(git rev-parse HEAD)"
echo "=============================================="

# ---------------------------------------------------------------------------
# Phase 0: preflight
# ---------------------------------------------------------------------------
echo
echo "===== Phase 0: preflight ====="
python scripts/preflight.py --run-dir "$RUN_DIR" --worker-count "$NUM_WORKERS" \
    --report "$LOG_DIR/preflight.json"

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "DRY-RUN complete: preflight passed, no adaptation work started."
    exit 0
fi

mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------------------
# Phase 1: P0 4-way parallel score
# ---------------------------------------------------------------------------
echo
echo "===== Phase 1: P0 4-way parallel score ====="
PIDS=()
for i in "${!WORKERS[@]}"; do
    worker_id="${WORKERS[$i]}"
    echo "P0: group $i -> worker slot $worker_id"
    python p0/score_worker.py \
        --group "$i" --num-groups "$NUM_WORKERS" --output-dir "$RUN_DIR/p0" \
        > "$LOG_DIR/p0_worker$worker_id.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: P0 worker (pid $pid) failed" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Phase 2: P0 aggregate
# ---------------------------------------------------------------------------
echo
echo "===== Phase 2: P0 aggregate ====="
python p0/aggregate.py --run-dir "$RUN_DIR"

# ---------------------------------------------------------------------------
# Phase 3: P1 unit/integration test
# ---------------------------------------------------------------------------
echo
echo "===== Phase 3: P1 unit/integration test ====="
( cd "$GIT_ROOT" && python -m pytest -q \
    tests/unit/test_ep.py \
    tests/unit/test_mechanisms.py \
    tests/unit/test_taskaware_ep.py \
    tests/unit/test_p0_p1_semantics.py \
    tests/unit/test_target10_selection_audit.py \
    tests/integration/test_cached_suite.py )

# ---------------------------------------------------------------------------
# Phase 4: P1 4-way parallel pilot
# ---------------------------------------------------------------------------
echo
echo "===== Phase 4: P1 4-way parallel pilot ====="
VARIANTS=(taskaware_full taskaware_no_gate taskaware_no_source_keep taskaware_control)
PIDS=()
for i in 0 1 2 3; do
    worker_id="${WORKERS[$i]}"
    variant="${VARIANTS[$i]}"
    echo "P1: $variant -> worker slot $worker_id"
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

# ---------------------------------------------------------------------------
# Phase 5: P1 pilot aggregate
# ---------------------------------------------------------------------------
echo
echo "===== Phase 5: P1 pilot aggregate ====="
set +e
python p1/pilot_aggregate.py --run-dir "$RUN_DIR"
PILOT_RC=$?
set -e

# ---------------------------------------------------------------------------
# Phase 6: mechanism gate
# ---------------------------------------------------------------------------
echo
echo "===== Phase 6: mechanism gate ====="
case "$PILOT_RC" in
    0)
        PASS=1
        echo "P1_MECHANISM_PASS"
        ;;
    3)
        PASS=0
        echo "P1_MECHANISM_FAIL"
        ;;
    *)
        echo "ERROR: P1 pilot aggregation failed technically, rc=$PILOT_RC" >&2
        exit "$PILOT_RC"
        ;;
esac

# ---------------------------------------------------------------------------
# Phase 7: confirmatory 4-way parallel (only if the gate passed)
# ---------------------------------------------------------------------------
if [ "$PASS" = "1" ]; then
    echo
    echo "===== Phase 7: confirmatory 4-way parallel ====="
    python scripts/write_confirmatory_configs.py
    DATASETS=(itw_target90 asv2021_la asv2021_df control_test)
    PIDS=()
    for i in 0 1 2 3; do
        worker_id="${WORKERS[$i]}"
        ds="${DATASETS[$i]}"
        echo "confirm: $ds -> worker slot $worker_id"
        python scripts/confirmatory_worker.py \
            --dataset "$ds" --output-dir "$RUN_DIR/confirmatory" \
            > "$LOG_DIR/confirm_worker$worker_id.log" 2>&1 &
        PIDS+=("$!")
    done
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: confirmatory worker (pid $pid) failed" >&2
            exit 1
        fi
    done
    python scripts/confirmatory_aggregate.py --run-dir "$RUN_DIR"
else
    echo "confirmatory skipped (P1_MECHANISM_FAIL); stopping before confirmatory."
fi

# ---------------------------------------------------------------------------
# Phase 8: final aggregate
# ---------------------------------------------------------------------------
echo
echo "===== Phase 8: final aggregate ====="
python scripts/final_report.py --run-dir "$RUN_DIR"

echo
echo "=============================================="
echo "done"
echo "run dir: $RUN_DIR"
echo "end    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=============================================="

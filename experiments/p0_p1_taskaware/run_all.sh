#!/bin/bash
# Unified 4-GPU launcher for the p0_p1_taskaware experiment.
#
# Phases:
#   0  preflight
#   1  P0 4-GPU score
#   2  P0 aggregate
#   3  P1 unit/integration test
#   4  P1 4-GPU pilot
#   5  P1 pilot aggregate
#   6  mechanism gate
#   7  confirmatory 4-GPU (only if the pilot gate passes)
#   8  final aggregate
#
#   GPU_LIST : comma-separated GPU ids (default "0,1,2,3").
#   --dry-run: run Phase 0 only (all validations, no GPU work).
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
# GPU_LIST parsing + validation
# ---------------------------------------------------------------------------
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPUS=()
IFS=',' read -r -a GPUS <<< "$GPU_LIST" || true
for i in "${!GPUS[@]}"; do
    GPUS[$i]="${GPUS[$i]//[[:space:]]/}"
done
if [ "${#GPUS[@]}" -lt 1 ]; then
    echo "ERROR: GPU_LIST must contain at least one GPU id" >&2
    exit 1
fi
for g in "${GPUS[@]}"; do
    if ! [[ "$g" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU_LIST entries must be integers, got '$g'" >&2
        exit 1
    fi
done
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    for ((j = i + 1; j < ${#GPUS[@]}; j++)); do
        if [ "${GPUS[$i]}" = "${GPUS[$j]}" ]; then
            echo "ERROR: duplicate GPU id '${GPUS[$i]}' in GPU_LIST" >&2
            exit 1
        fi
    done
done
NUM_GROUPS="${#GPUS[@]}"

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
echo "gpus : ${GPUS[*]} (num_groups=$NUM_GROUPS)"
echo "run  : $RUN_DIR"
echo "commit: $(git rev-parse HEAD)"
echo "=============================================="

# ---------------------------------------------------------------------------
# Phase 0: preflight
# ---------------------------------------------------------------------------
echo
echo "===== Phase 0: preflight ====="
python scripts/preflight.py --run-dir "$RUN_DIR" --gpu-count "$NUM_GROUPS" \
    --report "$LOG_DIR/preflight.json"

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "DRY-RUN complete: preflight passed, no GPU work started."
    exit 0
fi

mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------------------
# Phase 1: P0 four-GPU score
# ---------------------------------------------------------------------------
echo
echo "===== Phase 1: P0 four-GPU score ====="
PIDS=()
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    echo "P0: group $i -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python p0/score_worker.py \
        --group "$i" --num-groups "$NUM_GROUPS" --output-dir "$RUN_DIR/p0" \
        > "$LOG_DIR/p0_gpu$gpu.log" 2>&1 &
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
    tests/unit/test_target10_selection_audit.py \
    tests/integration/test_cached_suite.py )

# ---------------------------------------------------------------------------
# Phase 4: P1 four-GPU pilot
# ---------------------------------------------------------------------------
echo
echo "===== Phase 4: P1 four-GPU pilot ====="
VARIANTS=(taskaware_full taskaware_no_gate taskaware_no_source_keep taskaware_control)
if [ "$NUM_GROUPS" -lt 4 ]; then
    echo "ERROR: P1 pilot requires at least 4 GPUs" >&2
    exit 1
fi
PIDS=()
for i in 0 1 2 3; do
    gpu="${GPUS[$i]}"
    variant="${VARIANTS[$i]}"
    echo "P1: $variant -> GPU $gpu"
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
PASS=0
if [ "$PILOT_RC" -eq 0 ]; then
    PASS=1
    echo "P1_MECHANISM_PASS"
else
    echo "P1_MECHANISM_FAIL (aggregate exit $PILOT_RC)"
fi

# ---------------------------------------------------------------------------
# Phase 7: confirmatory four-GPU (only if the gate passed)
# ---------------------------------------------------------------------------
if [ "$PASS" = "1" ]; then
    echo
    echo "===== Phase 7: confirmatory four-GPU ====="
    python scripts/write_confirmatory_configs.py
    DATASETS=(itw_target90 asv2021_la asv2021_df control_test)
    PIDS=()
    for i in 0 1 2 3; do
        gpu="${GPUS[$i]}"
        ds="${DATASETS[$i]}"
        echo "confirm: $ds -> GPU $gpu"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/confirmatory_worker.py \
            --dataset "$ds" --output-dir "$RUN_DIR/confirmatory" \
            > "$LOG_DIR/confirm_gpu$gpu.log" 2>&1 &
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

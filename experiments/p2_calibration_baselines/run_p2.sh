#!/bin/bash
# P2 calibration/adaptability + published TTA baselines launcher.
#
# Phases:
#   0  preflight
#   1  calibration decomposition (CPU)
#   2  oracle-teacher counterfactual (CPU)
#   3  published-port audits + tests
#   4  frozen waveform parity
#   5  target10 TENT/SAR/Frozen 4-GPU pilot
#   6  pilot aggregate + paired bootstrap
#   8  final research-route decision
#
# NOTE: calibration + oracle run on CPU over the frozen feature cache; the
# published waveform ports (TENT/SAR) run on CUDA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GIT_ROOT="$(git rev-parse --show-toplevel)"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NUM_WORKERS="${#GPUS[@]}"
if [ "$NUM_WORKERS" -ne 4 ]; then
    echo "ERROR: P2 protocol requires exactly 4 CUDA GPUs" >&2
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

mkdir -p logs results
LOG_DIR="$SCRIPT_DIR/logs"
RUN_DIR="$SCRIPT_DIR/results/run_$(date +%Y%m%d_%H%M%S)"

echo "===== Phase 0: preflight ====="
python scripts/preflight.py --run-dir "$RUN_DIR" --worker-count "$NUM_WORKERS"
if [ "$DRY_RUN" = "1" ]; then echo "DRY-RUN complete"; exit 0; fi
mkdir -p "$RUN_DIR"

echo "===== Phase 1: calibration decomposition (CPU) ====="
for s in distribution_diagnosis threshold_oracle affine_oracle; do
    python calibration/$s.py --run-dir "$RUN_DIR"
done
python calibration/aggregate.py --run-dir "$RUN_DIR"

echo "===== Phase 2: oracle-teacher counterfactual (CPU) ====="
python calibration/oracle_teacher.py --run-dir "$RUN_DIR"

echo "===== Phase 3: port audits + tests ====="
( cd "$GIT_ROOT" && python -m pytest -q \
    tests/unit/test_p2_waveform_ports.py \
    tests/unit/test_p0_p1_semantics.py \
    tests/unit/test_taskaware_ep.py )

echo "===== Phase 4: frozen waveform parity ====="
CUDA_VISIBLE_DEVICES="${GPUS[3]}" python baselines/score_frozen.py \
    --split target10 --output "$RUN_DIR/parity" --n 128

echo "===== Phase 5: target10 TENT/SAR pilot (4 GPU) ====="
PIDS=()
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python baselines/run_port.py \
    --method tent_audio_ep --split target10 --output "$RUN_DIR/pilot/tent" \
    > "$LOG_DIR/p2_tent_gpu${GPUS[0]}.log" 2>&1 &
PIDS+=("$!")
CUDA_VISIBLE_DEVICES="${GPUS[1]}" python baselines/run_port.py \
    --method sar_audio_ep --split target10 --output "$RUN_DIR/pilot/sar" \
    > "$LOG_DIR/p2_sar_gpu${GPUS[1]}.log" 2>&1 &
PIDS+=("$!")
# MEMO is PORT_BLOCKED_AUDIT (official optimizer/lr unverified) -> not run.
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then echo "ERROR: pilot worker failed" >&2; exit 1; fi
done

echo "===== Phase 6: pilot aggregate + bootstrap ====="
python baselines/aggregate.py --pilot-dir "$RUN_DIR/pilot" --run-dir "$RUN_DIR"

echo "===== Phase 8: final research-route decision ====="
python scripts/final_report.py --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"

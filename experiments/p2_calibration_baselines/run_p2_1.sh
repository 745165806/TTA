#!/bin/bash
# P2.1 baseline validation + multi-domain confirmatory launcher.
#
# Phases:
#   0  preflight
#   1  official audit validation
#   2  unit/integration tests
#   3  direct waveform/logit/cache parity
#   4  target10 4-GPU: NormOnly / TENT / SAR / MEMO
#   5  target10 aggregate + bootstrap
#   6  port validity gate
#   8  confirmatory 4-GPU (if valid ports)
#  10  confirmatory aggregate + bootstrap
#  11  final report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
GIT_ROOT="$(git rev-parse --show-toplevel)"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; fi

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -ne 4 ]; then echo "ERROR: P2.1 requires exactly 4 GPUs" >&2; exit 1; fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

mkdir -p logs results
LOG_DIR="$SCRIPT_DIR/logs"
RUN_DIR="$SCRIPT_DIR/results/p2_1_$(date +%Y%m%d_%H%M%S)"

echo "===== Phase 0: preflight ====="
python scripts/preflight.py --run-dir "$RUN_DIR" --worker-count 4
if [ "$DRY_RUN" = "1" ]; then echo "DRY-RUN complete"; exit 0; fi
mkdir -p "$RUN_DIR"

echo "===== Phase 2: tests ====="
( cd "$GIT_ROOT" && python -m pytest -q \
    tests/unit/test_p2_waveform_ports.py \
    tests/unit/test_p2_1_baseline_semantics.py \
    tests/unit/test_taskaware_ep.py \
    tests/unit/test_p0_p1_semantics.py \
    tests/integration/test_p2_waveform_frozen_parity.py )

echo "===== Phase 3: direct parity ====="
CUDA_VISIBLE_DEVICES="${GPUS[3]}" python baselines/score_frozen.py --split target10 \
    --output "$RUN_DIR/parity" --n 128

echo "===== Phase 4: target10 4-GPU pilot ====="
declare -A M=([0]=norm_only_audio [1]=tent_audio_ep [2]=sar_audio_ep [3]=memo_audio_ep_full)
PIDS=()
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python baselines/run_port.py \
        --method "${M[$i]}" --split target10 --output "$RUN_DIR/pilot/${M[$i]//_audio*/}" \
        > "$LOG_DIR/p2_1_gpu${GPUS[$i]}.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do wait "$pid" || { echo "pilot worker failed" >&2; exit 1; }; done

echo "===== Phase 5: target10 aggregate ====="
python baselines/aggregate_p2_1.py --pilot-dir "$RUN_DIR/pilot" --run-dir "$RUN_DIR"

echo "===== Phase 11: final report ====="
python scripts/final_report_p2_1.py --run-dir "$RUN_DIR"

echo "done: $RUN_DIR"

#!/bin/bash
set -euo pipefail

# =============================================================================
# target10_selection experiment — unified entry script (label-free protocol).
#
#   0  environment + git BEFORE
#   1  preflight: check existing 10/90 split, derive/verify the label-free
#      inwild_target10_select.json, label-leakage hard check, disjoint check,
#      checkpoint/GPU/param-combination checks
#   2  parallel label-free parameter search (one group per GPU in GPU_LIST)
#   3  aggregate + freeze best_param.json
#   4  target90 final evaluation (fixed params, first GPU)
#   5  source-select EP baseline on the same target90 (first GPU)
#   6  comparison.csv
#   7  summary + git AFTER + protected-path check
#
#   --dry-run : run steps 0-1 only (all validations + GPU mapping, no GPU work).
#   GPU_LIST  : comma-separated GPU ids (default "0,1,2,3"); NUM_GROUPS follows.
# Any failing step stops immediately (set -euo pipefail).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
elif [ -n "${1:-}" ]; then
    echo "unknown argument: $1" >&2
    echo "usage: $0 [--dry-run]" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# GPU_LIST parsing + validation (runs before any work; errors exit immediately).
# -----------------------------------------------------------------------------
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPUS=()
IFS=',' read -r -a GPUS <<< "$GPU_LIST" || true
# strip any surrounding whitespace from each token
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
FIRST_GPU="${GPUS[0]}"

# Bound per-process BLAS/OpenMP threads so the parallel search groups do not
# oversubscribe the CPU (same convention as scripts/run_r7_param_search.py).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p logs results
LOG="logs/run_target10_$(date +%Y%m%d_%H%M%S).log"
# Tee everything to the timestamped log while keeping console output.
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "target10_selection experiment"
[ "$DRY_RUN" = "1" ] && echo "mode  : DRY-RUN (validations only, no GPU experiment)"
echo "start : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "log   : $LOG"
echo "dir   : $SCRIPT_DIR"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 0: environment check + git BEFORE
# -----------------------------------------------------------------------------
echo
echo "===== Step 0: environment + git BEFORE ====="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

echo "python : $(command -v python)"
python --version

echo "commit : $(git rev-parse HEAD)"
git status --short | tee logs/git_status_before.txt
git status --short -- src configs > logs/git_protected_before.txt

echo "--- GPU info ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 || echo "nvidia-smi failed (no CUDA driver / GPU available)"
else
    echo "nvidia-smi not found (no GPU)"
fi
python - <<'PY'
import torch
print("torch.cuda.is_available() =", torch.cuda.is_available())
print("torch.cuda.device_count() =", torch.cuda.device_count())
PY

# -----------------------------------------------------------------------------
# GPU config mapping (printed in both normal and dry-run mode).
# -----------------------------------------------------------------------------
echo
echo "===== GPU config ====="
echo "Search GPUs: ${GPUS[*]}"
echo "num_groups: $NUM_GROUPS"
for i in "${!GPUS[@]}"; do
    echo "group $i -> GPU ${GPUS[$i]}"
done

# -----------------------------------------------------------------------------
# Step 1: preflight (split check, select manifest, label-leakage, disjoint,
#         checkpoint/GPU/param checks).  Any failure exits non-zero here.
# -----------------------------------------------------------------------------
echo
echo "===== Step 1: preflight ====="
python scripts/preflight.py

if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "DRY-RUN complete: all validations passed, no GPU experiment started."
    echo "report : results/preflight.json"
    echo "select : manifests/inwild_target10_select.json"
    exit 0
fi

# -----------------------------------------------------------------------------
# Step 2: parallel label-free parameter search (one group per GPU in GPU_LIST).
#         Group i runs on GPU_LIST[i] with --group i --num-groups NUM_GROUPS.
# -----------------------------------------------------------------------------
echo
echo "===== Step 2: label-free parameter search ($NUM_GROUPS groups) ====="
# Never aggregate partial results from the previous 13-candidate protocol.
rm -f results/search_group_*.json
PIDS=()
for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    echo "launch group $i -> GPU $gpu (--num-groups $NUM_GROUPS)"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/select_target10_params.py \
        --group "$i" --num-groups "$NUM_GROUPS" &
    PIDS+=("$!")
done

# Wait each worker individually; any failure aborts the whole script non-zero
# (so select_best_param is never reached after a worker failure).
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "ERROR: search worker (pid $pid) failed" >&2
        exit 1
    fi
done
echo "Step 2: all groups finished"

# -----------------------------------------------------------------------------
# Step 3: aggregate search results and freeze best_param.json
# -----------------------------------------------------------------------------
echo
echo "===== Step 3: aggregate + freeze best_param.json ====="
python scripts/select_best_param.py

# -----------------------------------------------------------------------------
# Step 4: target90 final evaluation with the fixed selected parameters (first GPU)
# -----------------------------------------------------------------------------
echo
echo "===== Step 4: target90 final evaluation (GPU $FIRST_GPU) ====="
CUDA_VISIBLE_DEVICES="$FIRST_GPU" python scripts/eval_target90.py

# -----------------------------------------------------------------------------
# Step 5: source-select EP baseline on the same target90 (first GPU)
# -----------------------------------------------------------------------------
echo
echo "===== Step 5: source-select EP baseline (same target90, GPU $FIRST_GPU) ====="
CUDA_VISIBLE_DEVICES="$FIRST_GPU" python scripts/eval_source_select_baseline.py

# -----------------------------------------------------------------------------
# Step 6: final comparison table
# -----------------------------------------------------------------------------
echo
echo "===== Step 6: comparison.csv ====="
python scripts/compare_results.py

# -----------------------------------------------------------------------------
# Step 7: summary + git AFTER + protected-path check
# -----------------------------------------------------------------------------
echo
echo "===== Step 7: summary ====="
echo "--- best_param.json (selected) ---"
python - <<'PY'
import json
d = json.load(open("results/best_param.json", encoding="utf-8"))
print(json.dumps(d["selected"], ensure_ascii=False, indent=2))
PY
echo "--- comparison.csv ---"
cat results/comparison.csv

echo
echo "--- git AFTER ---"
echo "commit : $(git rev-parse HEAD)"
git status --short | tee logs/git_status_after.txt
git status --short -- src configs > logs/git_protected_after.txt

if ! diff -q logs/git_protected_before.txt logs/git_protected_after.txt >/dev/null; then
    echo "ERROR: protected paths (src/ configs/) changed during the run" >&2
    diff logs/git_protected_before.txt logs/git_protected_after.txt >&2 || true
    exit 1
fi
echo "protected paths (src/ configs/) unchanged: OK"

echo
echo "=============================================="
echo "实验完成"
echo "结果路径：results/comparison.csv"
echo "end   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "log   : $LOG"
echo "=============================================="

#!/bin/bash
# Fail-closed, resumable P2 multi-domain confirmatory launcher.
# Usage: run_confirmatory.sh RUN_DIR --validation-run VALIDATION_RUN [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_DIR="${1:?usage: run_confirmatory.sh RUN_DIR --validation-run DIR [--dry-run]}"
shift
DRY_RUN=0
VALIDATION_RUN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --validation-run) VALIDATION_RUN="${2:?--validation-run requires a directory}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$VALIDATION_RUN" ]; then
    echo "ERROR: --validation-run is required" >&2
    exit 2
fi
LOCKED_CONFIG="$SCRIPT_DIR/baselines/locked_port_configs.json"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -ne 4 ]; then
    echo "ERROR: confirmatory requires exactly 4 GPUs" >&2
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tta

echo "===== confirmatory preflight ====="
CUDA_VISIBLE_DEVICES="$GPU_LIST" python scripts/confirmatory_preflight.py \
    --run-dir "$RUN_DIR" --validation-run "$VALIDATION_RUN" \
    --locked-config "$LOCKED_CONFIG"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY-RUN complete: preflight passed; no job dispatched."
    exit 0
fi

python scripts/confirmatory_queue.py --run-dir "$RUN_DIR" \
    --validation-run "$VALIDATION_RUN" --locked-config "$LOCKED_CONFIG" \
    --gpus "$GPU_LIST"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

if [ "${CONDA_DEFAULT_ENV:-}" != "tta" ]; then
    echo "ERROR: activate conda environment 'tta' before running P3" >&2
    exit 2
fi

CONFIG="$SCRIPT_DIR/configs/main.json"
MANIFESTS="$SCRIPT_DIR/manifests"
SOURCE_MANIFEST="$ROOT/experiments/target10_selection/manifests/inwild_target10_select.json"
SALT="p3-main-2026"

python "$SCRIPT_DIR/scripts/split_target10.py" \
    --input "$SOURCE_MANIFEST" --output-dir "$MANIFESTS" --salt "$SALT" --fraction 0.10

CAL_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["count"])' "$MANIFESTS/target10_calU.json")"
EVAL_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["count"])' "$MANIFESTS/target10_evalU.json")"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/results/run_$RUN_STAMP"

if [ "${1:-}" = "--dry-run" ]; then
    echo "calU/evalU counts: $CAL_COUNT/$EVAL_COUNT"
    echo "split rule: sha256_order_of_salt_nul_sample_id salt=$SALT fraction=0.10"
    echo "GMM config: init=q25/q75 weights=0.5/0.5 variance=global max_iter=100 tol=1e-6 variance_floor=1e-6"
    echo "variants: Frozen, CalOnly, SourceTauTeacher, CalibratedTeacher-NoSelect, CalibratedSelective, OracleTeacher"
    echo "CPU worker assignment: worker0=Frozen/CalOnly worker1=SourceTauTeacher worker2=CalibratedTeacher-NoSelect worker3=CalibratedSelective; Oracle=post-hoc"
    echo "result directory: $RUN_DIR"
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "usage: bash experiments/p3_calibrated_teacher/run_p3.sh [--dry-run]" >&2
    exit 2
fi
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: refusing to overwrite result directory: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR/logs"
cp "$CONFIG" "$RUN_DIR/config.json"

python "$SCRIPT_DIR/calibration/fit_calibrator.py" --output "$RUN_DIR/calibration.json" \
    > "$RUN_DIR/logs/calibration.log" 2>&1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

VARIANTS=("Frozen-CalOnly" "SourceTauTeacher" "CalibratedTeacher-NoSelect" "CalibratedSelective")
OUTPUTS=("frozen_calonly.jsonl" "source_tau_teacher.jsonl" "calibrated_no_select.jsonl" "calibrated_selective.jsonl")
PIDS=()
for index in 0 1 2 3; do
    python "$SCRIPT_DIR/workers/run_variant.py" \
        --variant "${VARIANTS[$index]}" \
        --calibration "$RUN_DIR/calibration.json" \
        --output "$RUN_DIR/${OUTPUTS[$index]}" \
        > "$RUN_DIR/logs/worker$index.log" 2>&1 &
    PIDS+=("$!")
done
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

python "$SCRIPT_DIR/workers/oracle_teacher.py" \
    --calibration "$RUN_DIR/calibration.json" \
    --output "$RUN_DIR/oracle_teacher.jsonl" \
    > "$RUN_DIR/logs/oracle.log" 2>&1
python "$SCRIPT_DIR/aggregate.py" --run-dir "$RUN_DIR" | tee "$RUN_DIR/logs/aggregate.log"
echo "P3 result directory: $RUN_DIR"

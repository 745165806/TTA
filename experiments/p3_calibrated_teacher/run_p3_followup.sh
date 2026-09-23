#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

if [ "${CONDA_DEFAULT_ENV:-}" != "tta" ]; then
    echo "ERROR: activate conda environment 'tta' before running P3 follow-up" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

MAIN_RUN="$SCRIPT_DIR/results/run_20260922_201755"
SOURCE_MANIFEST="$ROOT/experiments/target10_selection/manifests/inwild_target10_select.json"
SALTS=("p3-sensitivity-2027" "p3-sensitivity-2028")
STAMP="$(date +%Y%m%d_%H%M%S)"
FOLLOWUP_DIR="$SCRIPT_DIR/results/followup_$STAMP"

for salt in "${SALTS[@]}"; do
    manifest_dir="$SCRIPT_DIR/manifests/$salt"
    python "$SCRIPT_DIR/scripts/split_target10.py" \
        --input "$SOURCE_MANIFEST" --output-dir "$manifest_dir" \
        --salt "$salt" --fraction 0.10
done

if [ "${1:-}" = "--dry-run" ]; then
    echo "main statistical reanalysis: $MAIN_RUN/statistical_reanalysis.{json,md} (no adaptation rerun)"
    echo "sensitivity salts: p3-sensitivity-2027, p3-sensitivity-2028"
    echo "variants: Frozen, CalOnly, SourceTauTeacher, CalibratedTeacher-NoSelect, CalibratedSelective, OracleTeacher"
    echo "asymmetry variants: Oracle-All/BonaOnly/SpoofOnly, Calibrated-All/BonaOnly/SpoofOnly"
    echo "CPU worker assignment: sensitivity worker0=Frozen/CalOnly worker1=SourceTauTeacher worker2=CalibratedTeacher-NoSelect worker3=CalibratedSelective; Oracle post-hoc"
    echo "CPU worker assignment: asymmetry worker0=Oracle-BonaOnly worker1=Oracle-SpoofOnly worker2=Calibrated-BonaOnly worker3=Calibrated-SpoofOnly"
    echo "CUDA disabled: CUDA_VISIBLE_DEVICES is empty"
    echo "output dirs: $FOLLOWUP_DIR/{sensitivity_2027,sensitivity_2028,asymmetry}"
    exit 0
fi
if [ "$#" -ne 0 ]; then
    echo "usage: bash experiments/p3_calibrated_teacher/run_p3_followup.sh [--dry-run]" >&2
    exit 2
fi
if [ -e "$FOLLOWUP_DIR" ]; then
    echo "ERROR: refusing to overwrite: $FOLLOWUP_DIR" >&2
    exit 2
fi
mkdir -p "$FOLLOWUP_DIR/logs"

python "$SCRIPT_DIR/scripts/reanalyse_main.py" --run-dir "$MAIN_RUN" \
    > "$FOLLOWUP_DIR/logs/main_reanalysis.log" 2>&1

for year in 2027 2028; do
    salt="p3-sensitivity-$year"
    manifest_dir="$SCRIPT_DIR/manifests/$salt"
    run_dir="$FOLLOWUP_DIR/sensitivity_$year"
    mkdir -p "$run_dir/logs"
    python "$SCRIPT_DIR/calibration/fit_calibrator.py" \
        --manifest-dir "$manifest_dir" --output "$run_dir/calibration.json" \
        > "$run_dir/logs/calibration.log" 2>&1
    variants=("Frozen-CalOnly" "SourceTauTeacher" "CalibratedTeacher-NoSelect" "CalibratedSelective")
    outputs=("frozen_calonly.jsonl" "source_tau_teacher.jsonl" "calibrated_no_select.jsonl" "calibrated_selective.jsonl")
    pids=()
    for index in 0 1 2 3; do
        python "$SCRIPT_DIR/workers/run_variant.py" \
            --variant "${variants[$index]}" \
            --manifest-dir "$manifest_dir" \
            --calibration "$run_dir/calibration.json" \
            --output "$run_dir/${outputs[$index]}" \
            > "$run_dir/logs/worker$index.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python "$SCRIPT_DIR/workers/oracle_teacher.py" \
        --manifest-dir "$manifest_dir" \
        --calibration "$run_dir/calibration.json" \
        --output "$run_dir/oracle_teacher.jsonl" \
        > "$run_dir/logs/oracle.log" 2>&1
done

ASYM_DIR="$FOLLOWUP_DIR/asymmetry"
mkdir -p "$ASYM_DIR/logs"
asym_variants=("Oracle-BonaOnly" "Oracle-SpoofOnly" "Calibrated-BonaOnly" "Calibrated-SpoofOnly")
asym_outputs=("oracle_bona_only.jsonl" "oracle_spoof_only.jsonl" "calibrated_bona_only.jsonl" "calibrated_spoof_only.jsonl")
pids=()
for index in 0 1 2 3; do
    python "$SCRIPT_DIR/workers/asymmetry_worker.py" \
        --variant "${asym_variants[$index]}" \
        --calibration "$MAIN_RUN/calibration.json" \
        --output "$ASYM_DIR/${asym_outputs[$index]}" \
        > "$ASYM_DIR/logs/worker$index.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

python "$SCRIPT_DIR/scripts/followup_aggregate.py" \
    --followup-dir "$FOLLOWUP_DIR" --main-run "$MAIN_RUN" \
    | tee "$FOLLOWUP_DIR/logs/aggregate.log"
echo "P3 follow-up result directory: $FOLLOWUP_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/audio_native_tta"
ASSET_ROOT="${TTA_ASSET_ROOT:-$(dirname "$ROOT")/TTA}"
GPU_CSV="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
FIRST_BATCH=(tent_audio_native_v1 sar_audio_native_v1 memo_audio_full_safeaug_v1 memo_audio_native_v1)
ALL_METHODS=("${FIRST_BATCH[@]}" tent_audio_native_scope_b_v1)
SCOPES=(backend_norm_affine_v1 backend_norm_affine_v1 full_ssl_aasist backend_norm_affine_v1 backend_norm_plus_graph_modulation_v1)
VIEWS=(original original original+fir_side_gain0.05 original+fir_side_gain0.05 original)
NORMALIZATION="model_eval_with_source_bn_running_statistics"
PRIMARY_BASELINE="Frozen-Waveform(score_before_update)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXP/results/run_$STAMP"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "comparison_track=audio_native_standard_tta"
  echo "normalization_policy=$NORMALIZATION"
  echo "primary_frozen_baseline=$PRIMARY_BASELINE"
  echo "asset_root_read_only=$ASSET_ROOT"
  for i in "${!ALL_METHODS[@]}"; do
    if [[ $i -lt 4 ]]; then gpu="${GPUS[$((i % ${#GPUS[@]}))]}"; else gpu="${GPUS[0]} (after batch 1)"; fi
    echo "method=${ALL_METHODS[$i]} GPU=$gpu parameter_scope=${SCOPES[$i]} normalization_policy=$NORMALIZATION augmentation_views=${VIEWS[$i]} primary_frozen_baseline=$PRIMARY_BASELINE output=$RUN_DIR/${ALL_METHODS[$i]}"
  done
  echo "smoke=N32_before_path_parity_then_full"
  echo "aggregate_output=$RUN_DIR"
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -lt 4 ]]; then
  echo "Task2.1 formal schedule requires at least four GPU_LIST entries" >&2
  exit 2
fi

python "$EXP/preflight.py" --gpu-list "$GPU_CSV" --asset-root "$ASSET_ROOT" \
  --output-root "$RUN_DIR"
mkdir -p "$RUN_DIR/smoke"

run_first_batch() {
  local output_root="$1"
  local limit_arg="$2"
  local pids=()
  for i in "${!FIRST_BATCH[@]}"; do
    local method="${FIRST_BATCH[$i]}"
    local gpu="${GPUS[$i]}"
    CUDA_VISIBLE_DEVICES="$gpu" python "$EXP/target_worker.py" \
      --method "$method" --split target10 --asset-root "$ASSET_ROOT" \
      --output "$output_root/$method" $limit_arg >"$output_root/$method.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ $failed -ne 0 ]]; then
    echo "Task2.1 first batch failed; inspect $output_root/*.log" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" python "$EXP/target_worker.py" \
    --method tent_audio_native_scope_b_v1 --split target10 --asset-root "$ASSET_ROOT" \
    --output "$output_root/tent_audio_native_scope_b_v1" $limit_arg \
    >"$output_root/tent_audio_native_scope_b_v1.log" 2>&1
}

run_first_batch "$RUN_DIR/smoke" "--limit 32"
python "$EXP/preflight.py" --gpu-list "$GPU_CSV" --asset-root "$ASSET_ROOT" \
  --check-smoke "$RUN_DIR/smoke" | tee "$RUN_DIR/smoke_preflight.json"

run_first_batch "$RUN_DIR" ""
python "$EXP/aggregate.py" --run-dir "$RUN_DIR" --asset-root "$ASSET_ROOT" \
  >"$RUN_DIR/aggregate.log" 2>&1
echo "completed=$RUN_DIR"

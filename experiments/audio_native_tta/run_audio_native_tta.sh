#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/audio_native_tta"
ASSET_ROOT="${TTA_ASSET_ROOT:-$(dirname "$ROOT")/TTA}"
GPU_CSV="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
METHODS=(tent_audio_native_v1 sar_audio_native_v1 memo_audio_native_v1 tent_audio_native_scope_b_v1)
SCOPES=(backend_norm_affine_v1 backend_norm_affine_v1 backend_norm_affine_v1 backend_norm_plus_graph_modulation_v1)
NORMALIZATION="model_eval_with_source_bn_running_statistics"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXP/results/run_$STAMP"

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "comparison_track=audio_native_standard_tta"
  echo "normalization_semantics=$NORMALIZATION"
  echo "asset_root_read_only=$ASSET_ROOT"
  for i in "${!METHODS[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    echo "method=${METHODS[$i]} GPU=$gpu parameter_scope=${SCOPES[$i]} output=$RUN_DIR/${METHODS[$i]}"
  done
  echo "aggregate_output=$RUN_DIR"
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
python -c 'import sys, torch
requested = [int(value) for value in sys.argv[1].split(",")]
available = torch.cuda.device_count()
if not torch.cuda.is_available() or any(index < 0 or index >= available for index in requested):
    raise SystemExit("requested GPUs %s unavailable (torch sees %d CUDA devices)" % (requested, available))' "$GPU_CSV"
if [[ ! -f "$EXP/augmentation_audit.json" ]]; then
  echo "missing source-side augmentation audit; run audit_augmentations.py first" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

pids=()
for i in "${!METHODS[@]}"; do
  method="${METHODS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" python "$EXP/target_worker.py" \
    --method "$method" --split target10 --asset-root "$ASSET_ROOT" \
    --output "$RUN_DIR/$method" >"$RUN_DIR/$method.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then
  echo "one or more audio-native workers failed; inspect $RUN_DIR/*.log" >&2
  exit 1
fi
python "$EXP/aggregate.py" --run-dir "$RUN_DIR" --asset-root "$ASSET_ROOT" \
  >"$RUN_DIR/aggregate.log" 2>&1
echo "completed=$RUN_DIR"

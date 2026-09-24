#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="$ROOT/experiments/protocol_tta"
ASSET_ROOT="${TTA_ASSET_ROOT:-$ROOT}"
GPU_CSV="${GPU_LIST:-0,1,2,3}"
PYTHON="${TTA_PYTHON:-python}"
PROTOCOLS=(episodic continual reset32 reset128)
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi
if [[ ! "$GPU_CSV" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "GPU_LIST requires exactly four distinct nonnegative device IDs" >&2
  exit 2
fi
for i in "${!GPUS[@]}"; do
  for j in "${!GPUS[@]}"; do
    if [[ $i -ne $j && $((10#${GPUS[$i]})) -eq $((10#${GPUS[$j]})) ]]; then
      echo "duplicate GPU ID" >&2
      exit 2
    fi
  done
done
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "comparison_track=protocol_tta split=target10 sample_order_policy=manifest_order shuffle=false"
  echo "manifest=$ROOT/experiments/target10_selection/manifests/inwild_target10_select.json"
  echo "asset_root=$ASSET_ROOT checkpoint_bundle=$ASSET_ROOT/outputs_v2/ssl_aasist/frozen/bundle.json"
  echo "output=$EXP/results/run_<timestamp>_<unique_suffix>"
  for i in "${!PROTOCOLS[@]}"; do
    echo "GPU=${GPUS[$i]} protocol=${PROTOCOLS[$i]} method=tent_audio_native_v1 optimizer=Adam lr=0.001 steps=1 weight_decay=0 parameter_scope=backend_norm_affine_v1 normalization=model_eval_with_source_bn_running_statistics"
  done
  echo "optimizer_state_policy=retain_until_protocol_reset; source_reference=unadapted_waveform_pass"
  echo "target10_contract=role_select_count_3178_unique_label_free_records"
  echo "episodic_smoke=32_real_waveform_replay_via_legacy_tent_adapt GPU=${GPUS[0]} parity_atol=1e-5"
  echo "schedule=preflight -> four_GPU_smoke_32_with_legacy_TENT_parity -> validate_smoke -> four_GPU_target10 -> aggregate"
  exit 0
fi

# Run preflight before creating an experiment directory; preserve its actual log.
PREFLIGHT_LOG="$(mktemp)"
trap 'rm -f -- "$PREFLIGHT_LOG"' EXIT
"$PYTHON" "$EXP/preflight.py" --gpu-list "$GPU_CSV" --asset-root "$ASSET_ROOT" 2>&1 | tee "$PREFLIGHT_LOG"
mkdir -p "$EXP/results"
RUN_DIR="$(mktemp -d "$EXP/results/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
cp -- "$PREFLIGHT_LOG" "$RUN_DIR/preflight.log"
mkdir "$RUN_DIR/smoke"
PIDS=()
finish() {
  local code=$?
  trap - EXIT
  if [[ $code -ne 0 ]]; then
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
    printf 'FAIL exit_code=%s\n' "$code" > "$RUN_DIR/status.txt"
    echo "FAILED result_directory=$RUN_DIR" >&2
  fi
  rm -f -- "$PREFLIGHT_LOG"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'RUNNING\n' > "$RUN_DIR/status.txt"
git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
cp "$EXP/config.json" "$RUN_DIR/config.json"

run_batch() {
  local output_root="$1"
  shift
  PIDS=()
  for i in "${!PROTOCOLS[@]}"; do
    local protocol="${PROTOCOLS[$i]}"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON" -u "$EXP/target_worker.py" \
      --protocol "$protocol" --split target10 --asset-root "$ASSET_ROOT" \
      --output "$output_root/$protocol" "$@" > "$output_root/$protocol.log" 2>&1 &
    PIDS+=("$!")
  done
  local failed=0
  local code=0
  for i in "${!PIDS[@]}"; do
    local pid="${PIDS[$i]}"
    if wait "$pid"; then code=0; else
      code=$?
      failed=1
      for other in "${PIDS[@]}"; do kill "$other" 2>/dev/null || true; done
    fi
    printf '%s exit_code=%s log=%s.log\n' "${PROTOCOLS[$i]}" "$code" "${PROTOCOLS[$i]}" >> "$output_root/worker_exit_codes.txt"
  done
  PIDS=()
  if [[ $failed -ne 0 ]]; then
    echo "worker failure; inspect $output_root/*.log" >&2
    return 1
  fi
}

run_batch "$RUN_DIR/smoke" --limit 32
"$PYTHON" "$EXP/preflight.py" --check-smoke "$RUN_DIR/smoke" 2>&1 | tee "$RUN_DIR/smoke_check.log"
echo "SMOKE PASS (including legacy TENT parity); starting full target10 sequences from source"
run_batch "$RUN_DIR"
"$PYTHON" "$EXP/aggregate.py" --run-dir "$RUN_DIR" > "$RUN_DIR/aggregate.log" 2>&1
printf 'PASS\n' > "$RUN_DIR/status.txt"
echo "completed result_directory=$RUN_DIR"

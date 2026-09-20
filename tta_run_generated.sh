#!/usr/bin/env bash
# Run from the TTA repository root after activating the existing tta environment.
# Usage: bash tta_run_generated.sh CONFIG_DIR RUN_ROOT LABELS_JSONL
# This is a one-shot, non-overwriting native-CLI batch; it does not resume partial runs.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash $0 CONFIG_DIR RUN_ROOT LABELS_JSONL" >&2
    exit 2
fi
CONFIG_DIR=$1
RUN_ROOT=$2
LABELS=$3
HELPER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/tta_experiments.py"
[[ -f "$HELPER" ]] || { echo "Missing $HELPER" >&2; exit 2; }
[[ -f "$LABELS" ]] || { echo "Missing evaluation label file: $LABELS" >&2; exit 2; }
shopt -s nullglob
CONFIGS=("$CONFIG_DIR"/*.yaml)
((${#CONFIGS[@]} > 0)) || { echo "No YAML configs in $CONFIG_DIR" >&2; exit 2; }
# Verify only paths/configuration here; evaluation labels are not opened.
python - "$CONFIG_DIR" "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path
import yaml
config_dir, expected_root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
for path in config_dir.glob('*.yaml'):
    cfg = yaml.safe_load(path.read_text())
    if (cfg.get('command') != 'run-tta' or cfg.get('run_name') != path.stem or
            Path(cfg['output_root']).resolve() != expected_root):
        raise SystemExit(f'Unexpected generated run configuration: {path}')
PY
# Score all methods before reading any evaluation labels.
# Frozen goes first to make feature-cache creation easy to observe.
if [[ -f "$CONFIG_DIR/frozen.yaml" ]]; then
    python -u -m eptta.cli run-tta --config "$CONFIG_DIR/frozen.yaml"
fi
for cfg in "${CONFIGS[@]}"; do
    [[ "$(basename -- "$cfg")" == frozen.yaml ]] && continue
    python -u -m eptta.cli run-tta --config "$cfg"
done
python "$HELPER" check --runs "$RUN_ROOT"
# Only this stage passes the labels to the evaluator.
for cfg in "${CONFIGS[@]}"; do
    name=$(basename -- "$cfg" .yaml)
    python -m eptta.cli evaluate --run "$RUN_ROOT/$name" --labels "$LABELS"
done
python -m eptta.cli report --runs "$RUN_ROOT" --out "$RUN_ROOT/summary.csv"
echo "Summary: $RUN_ROOT/summary.csv"

# EP-TTA experiment config helpers

Inspected repository: 745165806/TTA, main, 2026-09-20.

Place `tta_experiments.py` and `tta_run_generated.sh` in the repository root. Activate the existing `tta` environment. These helpers do not replace or patch the repository. The Python helper generates ordinary JSON-formatted YAML configurations and checks existing run metadata. The shell helper explicitly runs the repository's six-command CLI, scores all methods first, checks validity, then evaluates and reports.

## Requirements and limits

- Complete a real full source training run and its `prepare-source` stage first.
- Current local_v2 source configurations use explicit paths. The Python helper also accepts `--paths` for the project's normal path expansion.
- New config and run directories must be unused. Neither helper resumes incomplete runs or deletes/overwrites outputs.
- Use separate source resources and caches for each model/checkpoint. Share a target cache only when manifest, preprocessing, views, seed and numerical mode all match.
- `check` rejects a selected/scored run with incomplete coverage, excess fallback, or `valid_for_comparison=false`. It does not repair source selection. The current repository selector should be updated to exclude invalid candidates before EER ranking if such candidates arise; never resolve that issue with target-test labels.
- `paper-selection` retains three original source-selected references, adds matched-parameter ablations, evaluates all three random-U controls separately, searches the original EP step/lr grid for four alternative-objective baselines, and searches 17 source-only static shrinkage values. With the inspected 9-point EP grid this is 70 candidates and 22 selected families; 3 candidates reuse prior runs.
- Only the cache-adaptation portion is covered by the native runtime log. Do not describe it as end-to-end audio latency.
- Target manifests must already be prepared from verified metadata. The helpers never guess labels, groups, release/subset or audio paths from filenames. `runs` reads inference manifests and source-selection evidence, not target labels.

## Initial control test (after source preparation)

```bash
MODEL=aasist
python tta_experiments.py check --selection outputs_v2/${MODEL}/selection.json
python tta_experiments.py runs \
  --source-config configs/local_v2/source_${MODEL}.yaml \
  --selection outputs_v2/${MODEL}/selection.json \
  --data-dir data/manifests_v2/asv2019_la \
  --role control_test --tag control-main --cache-name control_test \
  --out-dir configs/local_v2/${MODEL}_control_main
CUDA_VISIBLE_DEVICES=0 bash tta_run_generated.sh \
  configs/local_v2/${MODEL}_control_main \
  outputs_v2/${MODEL}/experiments/control-main \
  data/manifests_v2/asv2019_la/labels/control_test.jsonl
```

Set MODEL=ssl_aasist for the second backbone after its independent source preparation.

## Extended source-only selection and control experiment

Choose this matrix before inspecting final-test results. Do not tune it on control_test or target_test.

```bash
python tta_experiments.py paper-selection \
  --source-config configs/local_v2/source_${MODEL}.yaml \
  --out-dir configs/local_v2/${MODEL}_paper
python -m eptta.cli prepare-source \
  --config configs/local_v2/${MODEL}_paper/prepare-selection.yaml
python tta_experiments.py check --selection outputs_v2/${MODEL}/selection-paper.json
python tta_experiments.py runs \
  --source-config configs/local_v2/source_${MODEL}.yaml \
  --selection outputs_v2/${MODEL}/selection-paper.json \
  --data-dir data/manifests_v2/asv2019_la \
  --role control_test --tag control-paper --cache-name control_test \
  --out-dir configs/local_v2/${MODEL}_control_paper
CUDA_VISIBLE_DEVICES=0 bash tta_run_generated.sh \
  configs/local_v2/${MODEL}_control_paper \
  outputs_v2/${MODEL}/experiments/control-paper \
  data/manifests_v2/asv2019_la/labels/control_test.jsonl
```

The two control rounds share one cache per model. All three random-U controls remain visible; they are not three independently trained source models.

## Testing performed here

Python syntax/CLI checks, shell syntax checks, and fixture-only configuration tests: candidate counts, family counts, random-U indices, K=0 config, shared cache references, overwrite protection and invalid-selection rejection. No real model, GPU, checkpoint, local audio, training or target-set scoring was executed here.

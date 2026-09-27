# Single development-informed revision

Working directory: `/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-overnight-tta`; Python: `/home/dell/anaconda3/envs/tta/bin/python`; environment: `PYTHONPATH=src:. OMP_NUM_THREADS=2`.

```bash
python experiments/overnight_tta/adapt_eval.py --config experiments/overnight_tta/config_revision_lr001.json --run-id overnight_20260927_seed13_revision_lr001 --seed 13
python experiments/overnight_tta/capture_states.py --config experiments/overnight_tta/config_revision_lr001.json --run-id overnight_20260927_seed13_revision_lr001 --train-run-id overnight_20260927_seed13 --seed 13
```

The selected source checkpoint is read from the original run's absolute path. Evaluation exited 0. This rate was chosen after seeing development results; it is not source-only model selection.

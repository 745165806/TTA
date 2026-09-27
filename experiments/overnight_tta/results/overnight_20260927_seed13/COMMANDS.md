# Commands

Working directory: `/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-overnight-tta`; Python: `/home/dell/anaconda3/envs/tta/bin/python`; environment: `PYTHONPATH=src:. OMP_NUM_THREADS=2`.

```bash
python experiments/overnight_tta/train.py --config experiments/overnight_tta/config.json --run-id overnight_20260927_seed13 --seed 13
python experiments/overnight_tta/adapt_eval.py --config experiments/overnight_tta/config.json --run-id overnight_20260927_seed13 --seed 13
python experiments/overnight_tta/capture_states.py --config experiments/overnight_tta/config.json --run-id overnight_20260927_seed13 --train-run-id overnight_20260927_seed13 --seed 13
```

All commands were actually run with the absolute `tta` Python and `PYTHONPATH=src:. OMP_NUM_THREADS=2`; train/evaluate exited 0. The source-training config snapshot retains the originally inspected broad ITW cache path, while `evaluation_config.json` records the exact target10-only cache used for both target scoring and state replay.

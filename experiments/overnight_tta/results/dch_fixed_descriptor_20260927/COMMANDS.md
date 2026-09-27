# Historical DCH fixed-source-descriptor ablation

Read-only checkpoint reuse. Working directory: `/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-overnight-tta`; environment: `PYTHONPATH=src:. OMP_NUM_THREADS=2`; Python: `/home/dell/anaconda3/envs/tta/bin/python`.

```bash
python experiments/overnight_tta/dch_fixed_descriptor.py --config experiments/overnight_tta/config_dch_ablation.json --output experiments/overnight_tta/results/dch_fixed_descriptor_20260927
```

Exit 0, PASS. The four historical DCH checkpoints remain in the separate `exp-distribution-conditioned-head` worktree; no training or mutation of that worktree occurred. Scores are saved before development labels are read.

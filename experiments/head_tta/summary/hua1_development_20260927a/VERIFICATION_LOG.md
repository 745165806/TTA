# H-UA1 execution and verification log

Date: 2026-09-27. Branch: `exp-head-tta`. The score worker ran from commit
`6c87d49`, after the mathematical contract and tests had been committed. All
commands below used the `tta` conda environment. The exact command arguments
and package versions are retained in `run_config.json`; source/cache references
and selected-only read boundaries are retained in `provenance.json`.

| Step | Command or source | Recorded result |
|---|---|---|
| Label-free smoke | `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_tta/run_scores.py --run-id hua1_smoke_20260927a --smoke` | Exit 0. ITW: 32 scores, 1 buffer, Frozen parity 0. WaveFake: 32 scores, 1 buffer, Frozen parity 0. No task metrics computed. |
| Full score generation | `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_tta/run_scores.py --run-id hua1_development_20260927a` | Exit 0. ITW: 3,178 scores, 13 buffers, Frozen parity 0. WaveFake: 4,096 scores, 16 buffers, Frozen parity 0. All selected IDs covered once; numeric failures 0. |
| Post-score label audit | `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python experiments/head_tta/analyze.py --run-id hua1_development_20260927a` | Exit 0. Both complete score files passed schema, finite-value and exact-ID checks before selected development labels were opened. `HEAD_ADAPTATION_NOT_YET_ACTIONABLE`; no passing alpha. |
| Unit tests | `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python -m pytest tests/unit/test_head_tta.py tests/unit/test_head_geometry.py -q` | Exit 0, `7 passed in 1.07s` in the final verification. |
| Compile and whitespace check | `PYTHONPATH=src:. OMP_NUM_THREADS=1 conda run -n tta python -m compileall -q experiments/head_tta experiments/head_capacity_geometry tests/unit/test_head_tta.py`; `git diff --check` | Exit 0 for both checks at the time of the experiment. |

`score_completion.json` and `smoke_completion.json` retain machine-readable
coverage/parity counts. The full per-sample score and buffer diagnostic files
remain in ignored, unique `experiments/head_tta/results/<run_id>/` directories
on the workstation; this Git summary intentionally excludes them. The
supervised H3 linear probe used target development labels only in five-fold
held-out diagnosis and is **not** an unsupervised TTA result.

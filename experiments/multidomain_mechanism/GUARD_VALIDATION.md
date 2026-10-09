# Guard capacity engineering validation — 2026-09-27

`conda run -n tta python -m pytest tests/unit/test_guard_capacity.py tests/unit/test_mechanisms.py tests/unit/test_oracle_diagnosis.py -q` exited 0: **43 passed in 1.57 s**. This covers exact hard production dispatch, a guard-severity-only relaxed path, existing unguarded path, fixed K/lr/rho tuples, Frozen identity, per-sample reset and projection, strict select manifest validation, and existing Oracle engineering tests.

`PYTHONPATH=src:. conda run -n tta python experiments/multidomain_mechanism/guard_worker.py --smoke` exited 0. Result directory: `experiments/multidomain_mechanism/results/smoke_20260927T040129Z/`. In-the-Wild, ASVspoof2021 LA, and ASVspoof2021 DF each produced 32 unique sample IDs × (Frozen + 3 settings × 3 arms) = 320 finite score rows. `analysis/summary.json` records `SCORES_COMPLETE_LABELS_NOT_READ`. The worker did not open any audit manifest. No EER/AUC or scientific improvement claim is made from this smoke.

Codecfake and WaveFake smoke remain **NOT_RUN** because their production-compatible feature cache is absent; WaveFake also lacks a Parquet reader in `tta`. GPU availability was `False` under PyTorch 2.1.0+cu121. The production EP feature path used CPU, as required by its frozen resource contract. No alternate numerical route was substituted.

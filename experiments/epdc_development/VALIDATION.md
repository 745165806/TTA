# EPDC v0-A engineering smoke — 2026-09-27

`PYTHONPATH=src:. conda run -n tta python -m pytest tests/unit/test_epdc_v0a.py tests/unit/test_guard_capacity.py -q` exited 0: 6 passed in 1.10 s. `lambda_preserve=0` exactly matches production `ep_no_keep`; the new objective uses source-only anchor labels and resets R per sample.

`PYTHONPATH=src:. conda run -n tta python experiments/epdc_development/v0a_worker.py --smoke` exited 0. `experiments/epdc_development/results/smoke_20260927T041513Z/` contains 32 real cached samples per available domain × Frozen/Base/Preserve = 96 finite rows per domain, with no audit labels opened. Codecfake and WaveFake remain NOT_RUN for production-compatible features. This smoke is an engineering check, not a method-success result.

# 2026-09-18 dispatch / approval / planning regression report

Status: **ENGINEERING PASS / REAL GPU AND NEW TARGETS NOT_RUN**.

This patch keeps the EP equation, Nd denominator, fixed three views, per-sample reset,
original-view final score, and canonical label meaning unchanged.  It changes dispatch,
diagnostics, evidence validation, source-export generality, small-data sampling, and
review-plan generation.

## Implemented

- `run_suite → run_method → run_cache_method` is the production cache scoring path.
  Method/resource configuration is validated before the sample loop.  Only typed
  floating-point failures become isolated frozen-score fallback events.
- Final diagnostics are recomputed from final R.  Margin state, actual regularizer,
  regularizer gradient activity, raw pre-projection history, margin change, final R,
  feature/score delta, steps and objective evaluations are distinct fields.
- Suite v0.2 binds manifest path/hash, role and scope.  New role views require the
  snapshot-published inference index; the historical cache is accepted only through
  its unchanged LOCKED extraction plan.  Cache keys and arrays are revalidated.
- Score sealing requires `run.json` plus a preregistered exact ID set.  Complete score
  coverage and `valid_for_comparison` are separate; no-improvement remains a valid run.
- R4 has a generic AASIST path for selected epoch, snapshot counts, audited parity
  budget/attack policy and ordinary from-scratch completion.  The historical ASV2019
  bundle retains its old epoch/count/six-attack checks.  SSL-AASIST is not promoted.
- Small training sets repeat from the complete shuffled order, drop-tail works for one
  or many ranks, empty/no-batch cases fail, and logs disclose repeats/drops.
- `select-methods`, `freeze`, and `report` now have direct implementations.  The only
  new planner is `scripts/make_plans.py`; `scripts/log.sh` supplies reusable evidence
  logging functions.
- Offline 16 kHz derivation uses decode → mono mean → `scipy.signal.resample_poly` →
  new float WAV and records source/derived hashes and recipe.  It never overwrites or
  runs inside dataset loading.

## Executed evidence

- Full CPU suite: `293 passed, 1 skipped`, exit 0.  The skip is the resampling test in
  the py310 environment because `soundfile` is absent.
- The py38 environment has soundfile/scipy but cannot import this Python>=3.10 package
  (`dataclass(slots=True)`), so the direct resampling smoke is **NOT_RUN_ENVIRONMENT**,
  not PASS.
- `make_plans.py source`, `select`, and `lock` smoke checks ran against the existing
  ASV2019 snapshot/R4 bundle/R5 cache and verified unique candidates, four explicit
  static amounts, and rebound internal hashes.  They did not train or score.

Final full-suite evidence: `docs/test_logs/20260918-review-regressions/pytest-full-final5.log`;
environment and expected NOT_RUN/BLOCKED logs are in the same directory.

## Resource/contract blocks

- No GPU command, training, SSH, download, or new scientific evaluation ran.
- ASV2021 LA/DF, In-the-Wild, WaveFake and Codecfake target snapshots/role manifests
  are **BLOCKED_RESOURCE** or **BLOCKED_CONTRACT** in the Git-ignored private
  `configs/experiment.private.json`.
- Published ports remain `NOT_RUN/BLOCKED_AUDIT` and are retained in selection/report
  records.
- Existing historical negative results and experiment directories were not changed.

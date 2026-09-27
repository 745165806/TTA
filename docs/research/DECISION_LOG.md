# Decision log

Append-only causal decisions. Later corrections must be new dated entries.

## 2026-09-27 — Guard-capacity contrast

Why: Protocol diagnosis links accumulation to parameter drift and ranking loss. Oracle diagnosis shows only 0.02243 percentage-point EER gain within guarded EP and zero selector regret. Its guard frequently activates, so guard severity is a plausible, untested cause of weak correction.

What the existing results rule out: On target10, changing only reset policy or choosing a different existing guarded hyperparameter is unlikely to explain a large missing gain. The results do not distinguish an overrestrictive guard from a weak objective or representation.

Next design: Hold the existing view-variance objective, margin regularizer, projection, features, optimizer, and episodic reset fixed. Compare production hard guard, a single relaxed guard threshold, and no guard at three representative K/lr/rho settings. Use per-domain ranking plus per-sample helpful/harmful and source-anchor damage to choose whether EPDC needs softer preservation, a reliability controller, or a new objective.

## 2026-09-27 — Soft source decision-order evidence

Why: Fixed guard contrast `guard_20260927T040841Z` shows that removing the hard guard expands updates and source-anchor damage. On 512 In-the-Wild mechanism samples, unguarded A/B slightly increase AUC but worsen EER; unguarded C reduces AUC from 0.957841 to 0.947653 while mean normalized evidence damage rises to 0.172484. ASV LA/DF show larger evidence damage under C, but their saved groups are single-class, so they cannot support EER/AUC claims.

What this excludes: A simple 90%→80% hard-margin threshold relaxation alone does not produce useful In-the-Wild EER correction at A/B/C. The current data do not prove a cross-domain ranking improvement or calibrate a reliable harmful-update predictor; harmful flips are rare. They also do not justify changing the saved ASV mechanism/final-holdout groups after seeing class composition.

Next design: Use the existing unguarded view-variance update as the base arm, then add one soft loss on **source-only cross-class decision ordering**, allowing individual feature changes. Test Frozen → Base Adapt → Base + Preserve on 32 cached samples per available domain before a sandbox analysis. Do not add a reliability gate or continual accumulation until preservation shows task-useful correction and lower damage.

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

## 2026-09-27 — Retire paired-order loss; test threshold-relevant margins

Why: `epdc_development/v0a_20260927T041709Z` found that the paired-order penalty is nearly zero while source-anchor margins can be damaged substantially. Base+Preserve leaves In-the-Wild EER/AUC and harmful flips unchanged from Base. The cross-class ordering remains intact even when many anchors move toward or across the source threshold.

What the result excludes: This exact paired-order objective at `lambda=1`, retention 0.9 and B optimizer setting is not a useful preservation component on the measured sandbox. It does not exclude all ordering losses or all soft preservation. ASV single-class selection still prevents cross-domain ranking claims.

Next design: Retire the paired-order module from the active method path. Replace it with a normalized source-anchor margin deficit relative to fixed `tau0`, so the loss is active precisely where the guard diagnosis measured evidence damage. Keep the same Base Adapt optimizer and projection, first 32 real samples then the fixed sandbox. Only proceed to reliability gating if task metrics and damage both support it.

## 2026-09-27 — Preservation alone does not solve target ranking

Why: `epdc_development/v0b_20260927T042152Z` reduced source-anchor damage from 0.247588 to about 0.00000116, but In-the-Wild AUC fell below Frozen and EER did not recover. It increased mean absolute score movement. The earlier P1 task-aware target10 pilot (`experiments/p0_p1_taskaware/results/run_20260922_133805/final_report.md`) already found a frozen pseudo-label teacher unreliable at the source threshold; its full method EER was 0.099217 versus Frozen 0.098571. This is existing evidence, not a new run.

What these results exclude: Simply strengthening source preservation around the current feature-variance objective does not yield task-useful correction at the fixed v0-B setting. Naive frozen pseudo-label BCE has also already failed on target10. Neither result proves that all task-aware objectives fail. Single-class ASV mechanism groups cannot validate ranking across domains.

Next design: Retire v0-B rather than stack a reliability gate on an unhelpful candidate. Develop a target objective that has a credible label-free relation to spoof discrimination, test it against Frozen and Base first, and obtain two-class group-disjoint mechanism evidence from permitted official development partitions. Keep the existing ASV mechanism/final-holdout assignment fixed. Only after an objective and preservation show positive task metrics should a candidate gate and safe accumulation be implemented.

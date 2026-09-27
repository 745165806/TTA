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

## 2026-09-27 — Shift to task-aligned unlabeled objectives

Why: Oracle, guard, and two retired preservation prototypes show that tuning, removing the guard, or suppressing source evidence damage do not create a useful target ranking correction within the current generic view-variance base. The existing frozen pseudo-label pilot was also unconvincing. A base objective is the next causal variable to change.

What these results exclude: They exclude promoting the existing EPDC preservation prototypes or adding a gate, rollback, accumulator, drift controller, or method lock now. They do not exclude source-anchor soft affinity or decision-space consistency, which were not tested in this form.

Next design: Pre-register only O1 source-anchor soft affinity, O2 decision-sensitive multi-view consistency, and O3 continuous reliability-weighted O1 plus O2. Fix K=5, lr=0.03, rho=0.1 and all other adaptation geometry. Require an independent second two-class development domain before promoting any objective.

## 2026-09-27 — Codecfake fixed selection is incompatible with production preprocessing

Why: The 32-waveform Codecfake production smoke passed, but the full fixed 512 preflight failed on a 24-kHz file. Label-free metadata audit shows only 222/512 are 16 kHz; production `load_audio` requires exactly 16 kHz and has no resampling branch.

What this excludes: The first-32 smoke does not validate a full Codecfake ranking experiment under the current Frozen feature path. Quietly omitting incompatible samples or introducing an unregistered resampler would change the fixed assignment or numerical pipeline.

Next design: Preserve the failed 512 run, keep Codecfake full scores and labels closed, and add a clearly separate official development domain whose whole selected group passes the same production preprocessing. Do not change the existing Codecfake or ASV2021 mechanism assignments.

## 2026-09-27 — One-domain objective movement is not enough

Why: The committed-code In-the-Wild 512 replay shows O1/O2/O3 AUC gains of only about 0.001 relative to Frozen; all paired ΔAUC intervals include zero and each EER worsens from 0.099010 to 0.103960. O1 improves 23 fixed-threshold predictions but does not provide clear ranking correction.

What this excludes: A claim that O1/O2/O3 already produce repeatable spoof-discriminative correction, or that O1's balanced-accuracy gain alone justifies preservation or gating. These observations cannot establish a two-domain effect.

Next design: Use one complete speaker group from independently named ASVspoof2019 PA official dev as an auxiliary second development domain after a production 32-waveform smoke. Keep ASVspoof2021 LA/DF marked `TWO_CLASS_DEV_UNAVAILABLE`. The PA protocol's first three non-selected rows were displayed during schema inspection; record that access and keep selected PA labels unread until full score coverage.

## 2026-09-27 — PA dev remains single-class; keep objective promotion closed

Why: The fixed PA dev speaker group passed production feature and objective score generation, but selected-label audit after all scores found 270 bonafide and zero spoof. In-the-Wild O1 shows 23 helpful fixed-threshold flips and a positive mean class-score-gap shift, yet EER worsens and its small AUC gain has a paired interval crossing zero. O2 causes 23 harmful PA threshold flips. This distinguishes score movement and source-damage reduction from verified ranking correction.

What this excludes: The PA result cannot serve as a second independent two-class EER/AUC domain. O1's lower PA source damage cannot by itself justify preservation, and fixed-threshold flips cannot justify a gate. The already observed PA group labels also preclude redrawing the same dev assignment to force two-class coverage.

Next design: Preserve O1/O2/O3 as unpromoted objective prototypes and keep preservation, gate, continual, and method lock inactive. Resolve development coverage only through a genuinely new, explicitly permitted two-class resource and a predeclared production-compatible preprocessing contract; do not alter the saved Codecfake/ASV/PA assignments or access target90/final labels. If no such resource is available locally without a new dependency or numerical path, report the blocker rather than tune on one domain.

## 2026-09-27 — Fixed Codecfake compatibility permits a local-context hypothesis test

Why: The already selected Codecfake 512 include 290 files above 16 kHz, while the production waveform loader only accepts 16 kHz. A deterministic, experiment-only resampling contract was needed before a second development domain could be scored without changing IDs or production semantics.

What these results exclude: All 222 native-16-kHz features and scores match the original production path exactly; the final 512 cache has exact selected-ID coverage and finite production-shape features. Thus sample omission and native-path numerical mismatch are no longer blockers. Cache parity alone says nothing about two-class coverage or adaptation benefit.

Next design: Keep the selected Codecfake labels closed until all local-study scores are complete. Compare Frozen and per-sample Base/O1 against B=16/32 shared-R local Base/O1 using fixed K=5, lr=0.03, rho=0.1 and fixed manifest order. Reset adapter and optimizer state at every buffer boundary.

## 2026-09-27 — Bounded shared context does not yet justify a method

Why: Single-sample episodic updates have little context, while historical unbounded continual adaptation drifted. A local shared 8×8 R within fixed-order B=16/32 buffers isolates the context variable under the existing Base/O1 losses and resets before the next buffer.

What these results exclude: It excludes claiming that shared local context alone yields stable two-domain spoof-ranking correction at this budget. Codecfake B32 AUC improves clearly over Frozen and per-sample, but ITW B32 AUC intervals versus Frozen cross zero and local point estimates are below their per-sample counterparts; local EER stays Frozen in both domains. Larger parameter movement or threshold changes alone cannot substitute for cross-domain ranking evidence. The Codecfake effect prevents a blanket claim that local sharing never helps.

Next design: Retain the fixed run as a domain-specific mechanism clue and promote no candidate. Do not expand the buffer sweep or add local-prior preservation, gate, continual memory, drift controller or method lock. Any further objective work needs a new predeclared mechanism that addresses the ITW failure while preserving the bounded-reset safety property; it must be evaluated on the same locked development assignments before any final data access.

## 2026-09-27 — Confirm effect scale before changing a method

Why: The previous Codecfake 512 local B32 AUC gain is roughly +0.01, but ITW 512 shows only a near-zero change. A larger Codecfake sample, full ITW target10 and an independently fixed ASVspoof2019 LA dev sample are necessary to distinguish a stable correction from a small-subset/order effect. Six label-free orders quantify local buffer membership sensitivity.

What these results exclude: The new 5000 Codecfake and 5000 ASVspoof2019 LA feature caches are exact, finite and production-compatible; the old Codecfake 512 features match exactly inside the larger cache. Thus native-path mismatch, changed old512 view seeds and missing selected features cannot explain subsequent score differences. Cache success alone does not establish any AUC or EER effect. WaveFake has an existing pyarrow reader but no materialized study cache.

Next design: Keep the four-arm budget and B32 fixed under the preregistered `CONFIRMATION_CONTRACT.md`; complete all six score orders before opening any new selected label protocol. Apply the prespecified effect-size, order-variability, bootstrap and EER rules without changing thresholds after inspection.

## 2026-09-27 — Official LA dev protocol omits 28 selected audio IDs

Why: The fixed LA assignment was drawn from all available official-dev FLAC files without opening labels. After all three-domain scores passed the label-free gate, the official dev trial protocol failed to cover 28/5000 selected IDs; the local dev label listing omits exactly the same IDs. The first analysis stopped before writing any metric.

What these results exclude: Assuming audio-file availability implies official label availability is invalid for this resource. No adaptation result or class ratio was used to make the correction.

Next design: Preserve all 5000 scores and the original assignment; use the exact 4972 official-protocol-covered selected IDs for LA post-hoc metrics, report the 28 omitted IDs, and make no replacement draw. Keep the four methods and preregistered promotion thresholds fixed.

## 2026-09-27 — Local context fails large-sample, multi-order confirmation

Why: The Codecfake 512 fixed-order local B32 gain could be a sample/order effect. The preregistered confirmation expanded ITW to all target10, nested Codecfake to 5000, added fixed ASVspoof2019 LA dev, and repeated all four methods in six label-free orders. Post-score composition was checked so a source-format association would not be mistaken for spoof evidence.

What these results exclude: A practically meaningful, order-stable two-domain positive local effect at fixed K=5/lr=0.03/rho=0.1/B32. Codecfake's large-set mean ΔAUC is `+0.000706/+0.001033`, smaller than its order std `0.002108/0.002770`; only three of six orders are positive. ITW is approximately zero; LA is tiny negative and near Frozen ceiling. Codecfake's sample rate perfectly determines class in this locked selection, so its AUC movement cannot isolate spoof-discriminative correction from recording/compatibility cues. This does not prove that all forms of local TTA fail.

Next design: Record `SMALL_DEVELOPMENT_ARTIFACT` and stop the Local-TTA method line. Do not extend B, tune the budget, add a new objective/gate/preservation/controller, or access target90/final metrics from this evidence. Any new scientific direction must first identify a development resource where spoof labels are not confounded with format and preregister a separate confirmation; it is outside this phase.

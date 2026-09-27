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

## 2026-09-27 — Meta-Rank full-head TTT completes without a two-domain gain

Why: Four actual source-only meta-training seeds used differentiable five-step unlabeled updates of the complete linear head. ITW target10 Meta-Rank mean AUC/EER was 0.963294/0.098402 versus Frozen 0.963309/0.098571 and ERM 0.963463/0.098526. WaveFake paired development Meta-Rank was 0.912422/0.161133 versus Frozen 0.915003/0.157715 and ERM 0.917206/0.155762. Meta-Rank did not consistently outperform Meta-BCE. Disabled adaptation exactly recovered the source head, so the differences are due to test-time updates; they do not amount to a useful positive effect.

Decision: Retain the source-meta training code, checkpoints, curves, scores, and negative report. Treat this as offline batch-transductive development evidence only. Do not promote this prototype to target90 or final holdout, and do not claim episodic or online TTA gains. The result motivates a later change in the source simulation or auxiliary objective, but that is outside this completed prototype round.

## 2026-09-27 — Distribution-conditioned shared head: WaveFake gain, no dual-domain promotion

Why: Four fixed seeds trained a 320→128→161 network on source attack-family episodes and generated one label-free shared head per target domain. WaveFake development DCH beat Frozen and matched ERM in each seed (mean AUC 0.924103 versus 0.915003/0.921766; EER 0.149414 versus 0.157715/0.152344). ITW target10 DCH AUC 0.963325 and EER 0.098838 did not beat matched ERM 0.963539/0.098402, and the Frozen AUC gain was only +0.000016. Generated heads moved by roughly 13–14° on average, so the flat ITW effect is not a zero-update artifact. WaveFake mean gain versus Frozen (+0.009100 AUC, +0.008301 EER) remains below the fixed +0.01 actionable cutoff.

Decision: `DISTRIBUTION_CONDITIONING_NOT_ACTIONABLE`. Retain the WaveFake development signal and both-domain negative evidence. No parameter sweep, target90 run or final-holdout evaluation follows from this first prototype. The evidence describes offline batch-transductive behavior only.

## 2026-09-27 — Overnight residual TTA gives no target-adaptation gain

Why: A real 10-epoch source fit trained a residual 160D adapter and equal-step static linear ERM, then evaluated Frozen, ERM, residual source-only, residual target-adapted, and T3A-batch port on ITW target10 and paired WaveFake development. Source-select chose a target LR of 1e-4 after all three candidate rates tied at perfect pseudo-domain validation. ITW residual target adaptation worsened EER from 0.100049 to 0.102514; WaveFake EER stayed 0.145020 and ΔAUC from adaptation was only +0.000032. Static ERM achieved the best WaveFake AUC/EER, 0.933126/0.140625. A single declared development-informed 1e-3 revision worsened ITW to AUC/EER 0.939755/0.126663 and remained below ERM on WaveFake.

Historical DCH four-seed WaveFake gains over Frozen were also checked against a fixed balanced source-fit descriptor using the already trained checkpoints. Target conditioning averaged ΔAUC −0.000696 and EER gain −0.061 pp relative to that static head. This distinguishes source head training from actual target distribution use.

Decision: **NO qualifying overnight TTA candidate; best method NONE.** Do not run residual seeds29/47/71, further target hyperparameter search, target90 or final holdout for these routes. GPU unavailable; genuine earlier-layer waveform route remains `BLOCKED_RESOURCE`, so no conclusion about it. The only recommended next action from this study is to stop this ineffective feature-stage line while preserving its negative result and checkpoints.

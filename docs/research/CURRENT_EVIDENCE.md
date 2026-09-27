# Current evidence (2026-09-27)

Scope: development diagnosis on the existing In-the-Wild target10 assignment. Scores increase toward spoof (`0=bonafide, 1=spoof`). EER/AUC are proportions. No held-out conclusion follows from these studies.

## Protocol diagnosis

Source: `experiments/protocol_tta/results/run_20260927_104009_W78MLZ/summary.json` (`status=COMPLETE`, 3,178 samples, one fixed manifest order, source threshold `tau0=-4.770049095153809`).

| Protocol | EER | AUC | Balanced accuracy | Mean parameter drift | Final drift |
|---|---:|---:|---:|---:|---:|
| Frozen waveform | 0.0985707245 | 0.9633087850 | 0.5588960079 | 0 | 0 |
| Episodic | 0.0983463882 | 0.9634366095 | 0.6804998111 | 0.0374171253 | 0.0372495480 |
| Reset32 | 0.1074420897 | 0.9576549090 | 0.7847226958 | 0.2492244683 | 0.1591920669 |
| Reset128 | 0.1070496084 | 0.9491876065 | 0.8170886806 | 0.5538409872 | 0.7520176848 |
| Continual | 0.1679721497 | 0.8998983838 | 0.8381610683 | 3.0627597718 | 4.4619826430 |

Continual mean entropy decreased from 0.02518115596 to 0.02159298317 while ranking deteriorated. Episodic entropy decreased from 0.13078000748 to 0.11103117570 with only 0.0002243363 absolute EER improvement. This supports testing drift and evidence damage as mechanisms; it does not prove entropy minimization is universally harmful.

## Oracle diagnosis

Source: `experiments/oracle_diagnosis/results/oracle_20260927_110841_839245/analysis/summary.json` (`development_diagnosis_only`, 113 candidates including Frozen, 3,178 target10 samples).

| Quantity | Value |
|---|---:|
| Frozen EER / AUC | 0.0985707245 / 0.9633087850 |
| Full target10 oracle EER / AUC | 0.0983463882 / 0.9633195086 |
| Five-fold oracle EER / AUC | 0.0983463882 / 0.9631191929 |
| Oracle EER gain | 0.0002243363 absolute = 0.02243363 percentage points |
| Selection regret (existing guarded-v2) | 0 |
| Oracle best | K=5, lr=0.01, rho=0.05 |
| Existing unsupervised selected | K=10, lr=0.3, rho=0.2 |

The oracle best has mean absolute score delta 0.0017483247 and mean adapter norm 0.0004787336. Its guard activated in 35.12% of steps. The existing unsupervised selection activates the guard in 64.22% of steps and reverts in 22.77% of steps. The small oracle gain bounds correction capacity **within this guarded EP candidate space on target10**. Zero selection regret weakens the selector-bottleneck explanation here. It does not identify the objective or guard as the cause without a controlled comparison.

## Boundary and next test

Existing target10/target90 assignment remains fixed. No target90 sample labels or metrics were opened for this new branch. During initial repository orientation, the historical `split_meta.json` printed aggregate target90 class counts; those counts are excluded from all decisions. The next controlled test holds objective, features, optimizer, and parameter budget fixed while varying only source-margin guard severity across available domains. The 5-domain claim and EPDC method claim remain **NOT_RUN / unverified**.

## New mechanism-development evidence (append 2026-09-27)

The fixed 512-sample In-the-Wild mechanism subset gave Frozen EER/AUC `0.099010/0.957841`. In the guard contrast, unguarded C gave `0.103960/0.947653` and mean normalized source evidence damage `0.172484`; hard C gave `0.099010/0.955589`. Unguarded A/B had small AUC gains but worse EER. ASV LA/DF mechanism groups were all bonafide (282/370), so they have no valid EER/AUC; the result is **partial three-domain mechanism evidence, not a five-domain ranking study**. Codecfake/WaveFake were not run because production-compatible features are absent.

The paired-order v0-A preservation term was nearly inactive. The replacement normalized-margin v0-B nearly eliminated source evidence damage but lowered In-the-Wild AUC below Frozen and did not recover EER. Both components were retired from the active method path and retained in experiment code for negative-result reproducibility. No EPDC candidate gate, continual accumulator, method lock, or held-out evaluation has been justified. See the append-only ledger and decision log for exact runs and decisions.

Audit boundary correction: the first post-score ASV audit scanned the full official eval label files with `rg` while emitting only mechanism-selected rows. This counts as accessing final-holdout label-file bytes under the strict boundary, despite no holdout row being returned or used. Earlier `final_holdout_labels_accessed=false` fields use the narrower "parsed or used" meaning and must be read with `experiments/multidomain_mechanism/AUDIT_BOUNDARY_CORRECTION.md`. Later analysis uses selected-only local audit artifacts.

## Task-aligned unlabeled objective discovery (append 2026-09-27)

The fixed-budget O1/O2/O3 objective study generated complete In-the-Wild mechanism-dev scores before opening selected-only audit labels. Frozen EER/AUC was `0.099010/0.957841`. O1, O2, and O3 each had EER `0.103960`, with AUC `0.958799`, `0.958895`, and `0.958879`; their paired bootstrap ΔAUC intervals all included zero. O1 produced 23 helpful and 0 harmful fixed-threshold flips, but ranking correction is weak and EER worsened. This is one-domain development evidence and cannot promote any objective.

Codecfake 32 real WAV production-cache smoke and two-domain objective smoke passed. Full fixed Codecfake 512 feature materialization failed because 290 selected WAVs are not 16 kHz and production preprocessing requires 16 kHz; IDs were not reselected or silently resampled. WaveFake lacks a compatible Parquet reader in `tta`; ASV2021 LA/DF have no suitable two-class development source locally. New Base Objective promotion, preservation, reliability gate, continual drift control, method lock, target90 metrics, and final held-out claims remain **NO / NOT_RUN**. Historical ASV eval label-file access correction above remains in force.

## Auxiliary PA development continuation (append 2026-09-27)

An independently named ASVspoof2019 PA official-dev group of 270 complete 16-kHz waveforms was fixed before selected labels were read; its 32-sample and 270-sample production cache paths passed. `objective_two_domain_20260927a` generated all 3,910 score rows across In-the-Wild and PA before selected audits. PA's selected group then proved **270 bonafide / 0 spoof**, so PA EER/AUC are undefined. It remains single-class mechanism evidence and will not be reselected. In-the-Wild O1 AUC gain over Frozen is about 0.000958 with development bootstrap interval spanning zero; EER worsens by 0.004950. O1 reduces PA mean source damage versus Base but does not meet the two-class task-correction criterion. No objective is promoted, and method lock remains NO.

## Codecfake compatibility continuation (append 2026-09-27)

The fixed 512 Codecfake official-dev IDs have a complete production-compatible feature cache under `codecfake_compat/compat_20260927a`. The experiment-only path bypassed resampling for all 222 native 16-kHz samples and used SciPy 1.13.0 `resample_poly` for the other 290. Both 32-sample and complete 222-sample native parity audits had zero waveform, feature, and score difference against the unchanged production path (required tolerance 1e-5). Final feature coverage is exactly 512 unique IDs with finite float32 3×160 views. Labels were not read during cache creation. This removes a data-path blocker but supplies no ranking evidence yet; O1/O2/O3 remain unpromoted.

## Local-distribution TTA development continuation (append 2026-09-27)

`local_dev_20260927a` completed all seven arms for fixed ITW and Codecfake 512 selections at K=5, lr=0.03, rho=0.1 with B=16/32 fixed-order local buffers. Selected labels were read only after 7,168 score rows and 192 buffer records passed exact coverage/numeric/reset validation. Codecfake has 81 bonafide and 431 spoof; ITW has 310 and 202. Codecfake B32 local Base/O1 AUC rose from Frozen `0.813354` to `0.823294/0.825585`, with positive paired development ΔAUC intervals. ITW B32 local Base/O1 AUC was `0.958128/0.958224` versus Frozen `0.957841`, but both intervals included zero and both were below the corresponding per-sample AUC `0.958783/0.958799`. All local EERs equaled Frozen within each domain. Mean local R norms stayed below 0.015 and no buffer inherited an update, while source evidence damage remained nonzero. The local-context hypothesis has a Codecfake-specific positive result but no stable two-domain correction; candidate promotion remains **NONE**, method lock **NO**, target90 metrics **NO**, final held-out metrics **NO**.

## Large-scale confirmation preparation (append 2026-09-27)

The confirmation contract and label-free assignments are fixed before new large-set metrics. ITW reuses all 3178 existing target10 IDs. Codecfake official dev has a fixed 5000-ID nested sample (prior512 plus 4488 new IDs); ASVspoof2019 LA official dev has 5000 fixed available FLAC IDs. The two new Frozen production caches each pass 5000/5000 exact coverage with finite 3×160 float32 views. Codecfake has 2140 native 16-kHz and 2860 deterministic-resampled inputs, and old fixed512 embeddings match the previous cache exactly (maximum difference 0). LA inputs are all native 16 kHz. No new selected labels have been opened; all ranking/effect claims remain pending. WaveFake pyarrow 21.0.0 is available, but its feature pipeline is not materialized and does not block this study.

## Confirmation score and protocol coverage correction (append 2026-09-27)

All 18 domain×order score files are complete: 316,272 four-arm rows and 4,968 local buffer records, with exact ID/order coverage, independently rerun Frozen/per-sample controls invariant by ID, zero numeric failures and zero buffer reset violations. Only after this gate passed was the selected development protocol opened. ASVspoof2019 LA official dev has protocol labels for 4972 of the fixed 5000 scored audio IDs; 28 IDs are absent from both official dev trial and local dev label listing. The first analysis stopped before producing metrics, and its failure is retained. LA ranking will use only the fixed 4972 official-protocol-covered IDs, with no redraw or method change; ITW and Codecfake remain fully covered. No ranking confirmation result has yet been recorded.

## Large-scale confirmation result (append 2026-09-27)

The preregistered six-order four-arm study finds no replicated practical local-context correction. On ITW target10 3178, Local-Base/O1 B32 mean ΔAUC is `+0.000008/+0.000031`, with order std `0.000036/0.000021`; Frozen EER/AUC is `0.098571/0.963309`. On Codecfake 5000, the corresponding mean ΔAUC is `+0.000706/+0.001033` with order std `0.002108/0.002770`, only three of six positive order point estimates and two of six positive paired bootstrap lower bounds per arm; Frozen EER/AUC is `0.260933/0.823094`. Thus the earlier fixed-order 512 `~+0.01` gain did not replicate at larger scale. On official-protocol-covered ASVspoof2019 LA 4972, local ΔAUC is `-0.000009/-0.000010` against a near-perfect Frozen AUC `0.999962`; 28 extra fixed audio IDs remain scored but unlabelled.

The post-score composition audit found perfect rate/class confounding in Codecfake: 16/24 kHz are entirely spoof (4314), whereas 44.1/48 kHz are entirely bonafide (686). No within-rate two-class AUC can separate spoof correction from format cues. The preregistered decision is **SMALL_DEVELOPMENT_ARTIFACT**; no mechanism is promoted, no new method is allowed from this result, method lock remains NO, target90 metrics NO, final held-out metrics NO. Full order, bootstrap, nested-subset, composition and failure details are retained in the experiment result and tracked small summary.

## Capacity-audit transition (append 2026-09-27)

Formal research status: **SMALL_DEVELOPMENT_ARTIFACT**. Local-TTA mainline = **CLOSED**; candidate = **NONE**. The fixed Codecfake 5000 population is auxiliary and `RATE_CLASS_CONFOUNDED`; ASVspoof2019 LA is auxiliary and `NEAR_CEILING`. The next study will use development labels solely for supervised held-out upper-bound diagnosis of the existing frozen features, fixed classifier and 8×8 R. It is **not a proposed TTA method, not unsupervised, and not final performance**. No target90 or final held-out metric was accessed for this transition.

## WaveFake resource audit (append 2026-09-27)

Local WaveFake contains 131 Parquet files with 104800 embedded WAV rows: 13100 `audio_id` values each have one `R` and seven `WF1`–`WF7` files. Every WAV header is mono 22050 Hz uncompressed 16-bit PCM, so the Codecfake rate/class confound is absent here. A small systematic real/generated duration offset remains; explicit speaker/language/transcript fields are absent, and exact semantic content matching cannot be independently checked. A fixed seed-2026 related-content development assignment contains 2048 IDs, one real plus one cyclically chosen WF generator each (4096 selected waveforms, balanced classes). This is resource evidence only; Frozen cache and supervised capacity metrics are **NOT_RUN** at this point. No target90 or final holdout data was accessed.

## ITW supervised development capacity result (append 2026-09-27)

Five-fold held-out ITW target10 (3178) capacity diagnosis is complete. C0 Frozen AUC/EER `0.963309/0.098571`; C1 supervised radius-0.1 8×8 R `0.963385/0.099064`; C2 linear readout of the same original-view frozen 160D features `0.973159/0.085292`; C3 160→32→1 probe `0.974185/0.082681`. C1 is practically SMALL; C2 improves EER by 0.013279 and AUC by 0.009850, with improvements in each held-out fold. C3 adds a smaller increment beyond C2. This supports readable target information in ITW frozen features but indicates the current bounded R plus fixed classifier has little useful correction capacity. It is **SUPERVISED DEVELOPMENT DIAGNOSIS, NOT A TTA METHOD OR FINAL PERFORMANCE**. WaveFake capacity and cross-domain replication remain pending; target90 and final holdout remain unopened.

## WaveFake fixed development feature cache (append 2026-09-27)

The fixed 2048-related-content-pair WaveFake assignment passed both a 32-waveform production smoke and a complete 4096-waveform cache validation. All inputs were explicitly resampled from 22050 to 16000 Hz with SciPy 1.13.0 `resample_poly`; the unchanged production Frozen worker yielded exact 4096 unique selected IDs, finite float32 3×160 views, and no selected label sidecar read. The real/generated duration-offset caveat remains. **No WaveFake capacity metric has been computed at this point.** Target90 and final holdout remain unopened.

## Two-domain supervised development capacity audit (append 2026-09-27)

ITW/WaveFake five-fold held-out Frozen AUC/EER is `0.963309/0.098571` and `0.915003/0.157715`. Supervised radius-0.1 8×8 R is `0.963385/0.099064` and `0.912394/0.161621`: no actionable benefit. Linear probes on the same frozen 160D original-view features reach `0.973159/0.085292` and `0.951711/0.114258`, improving both AUC and EER in all five held-out folds of each domain. The ITW linear ΔAUC is +0.009850 (MODERATE) and EER improvement +0.013279 (ACTIONABLE); WaveFake is +0.036708/+0.043457 (ACTIONABLE). Small nonlinear probes add only about +0.001 AUC beyond the linear probes. A separate preregistered convex solver check of the exact bounded-R score family gives ITW ΔAUC +0.000027 and WaveFake −0.006956, so the original C1 result is not explained by Adam early stopping under the same BCE/radius.

Bidirectional supervised cross-domain linear probes both perform worse than the destination Frozen model (ITW→WaveFake AUC 0.903675 versus 0.915003; WaveFake→ITW 0.960763 versus 0.963309), suggesting domain-specific readout rather than a directly shared head. A post-hoc WaveFake stratum of 1722 pairs both longer than the production crop retains a linear AUC improvement 0.923785→0.961800; other source artifacts remain possible. Codecfake remains `RATE_CLASS_CONFOUNDED` and ASVspoof2019 LA `NEAR_CEILING`.

**Decision: `R_PARAMETERIZATION_BOTTLENECK` for the present frozen classifier plus radius-0.1 R; actionable supervised development representation-readout gap exists.** This is **NOT A PROPOSED TTA METHOD, NOT UNSUPERVISED, NOT FINAL PERFORMANCE**. The radius and R subspace have not been separately isolated; no universal cross-domain spoof direction is established. Local-TTA mainline stays `CLOSED`, candidate `NONE`; no gradient-direction study, backend fine-tune, method lock, target90 or final held-out metric was performed.

## Strict target90 feature-I/O correction (append 2026-09-27)

The ITW capacity runs used the inherited `load_context()` reader on the full 31779-row frozen target_test feature cache. It loaded all chunks before retaining only the fixed 3178 target10 IDs. Therefore **target90 feature-cache bytes were accessed (YES)** under the established strict file-byte definition, even though target90 rows were discarded and never entered supervised CV. **Target90 labels, scores, metrics, selection and adaptation remain NO.** Earlier capacity `target90_accessed=false` fields meant the latter narrower condition and must be interpreted with `experiments/capacity_audit/TARGET90_FEATURE_IO_CORRECTION.md`. This correction does not change any capacity result or Local-TTA conclusion. Future strict no-access runs require selected-only I/O.

The requested WaveFake resource audit also read class codes and audio headers for all 104800 local rows. Only the fixed 4096-row development subset entered supervised CV, and no WaveFake final-holdout assignment or metric exists. The unselected audited rows **cannot later be called an untouched final holdout**; see `experiments/capacity_audit/WAVEFAKE_FUTURE_HOLDOUT_BOUNDARY.md`.

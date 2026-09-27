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

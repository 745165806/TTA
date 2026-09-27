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

## Distribution-conditioned shared head (append 2026-09-27)

The later capacity-audit WaveFake selected-only 4,096-row, 2,048-content-pair frozen feature cache supersedes the earlier statement here that WaveFake had no compatible local feature cache; the earlier statement remains historically accurate for that previous study. Four source-trained DCH seeds (13/29/47/71) used six official ASVspoof2019 LA spoof attack families paired with bonafide, with no target label in head generation and no target-time gradient. ITW target10 Frozen/ERM/DCH mean AUC was 0.963309/0.963539/0.963325 and EER 0.098571/0.098402/0.098838. WaveFake development mean AUC was 0.915003/0.921766/0.924103 and EER 0.157715/0.152344/0.149414. WaveFake gains occurred in all four seeds, but ITW lacked a meaningful gain and DCH was worse than matched ERM there. WaveFake content-pair bootstrap and per-seed head-movement records are in `experiments/distribution_conditioned_head/`. The predefined decision is `DISTRIBUTION_CONDITIONING_NOT_ACTIONABLE`; target90 and final holdout were not accessed in this branch.

## Overnight source-trained residual TTA and fixed-descriptor check (append 2026-09-27)

With GPU unavailable (`nvidia-smi` exit 9; CUDA device count 0), a full cached-feature seed13 source fit trained the 160D residual adapter and equal-step static ERM for 10 epochs/1,790 optimizer steps each. Both arms saw all 25,380 unique source fit IDs. Source-select curves chose residual epoch4 and ERM epoch10 from the equal total budget. The offline batch-transductive target protocol saved scores before development labels and restarted from the source checkpoint per domain. Frozen/ERM/residual-off/residual-on/T3A-batch ITW AUC was `0.963309/0.964594/0.961182/0.961181/0.963041`, EER `0.098571/0.098571/0.100049/0.102514/0.100087`. WaveFake AUC was `0.915003/0.933126/0.931893/0.931925/0.920668`, EER `0.157715/0.140625/0.145020/0.145020/0.151367`. Residual target adaptation had no useful increment over its own source-only checkpoint and was below matched ERM in both domains. A single explicit development-informed LR=1e-3 revision worsened ITW markedly and remained below ERM on WaveFake; no seeds29/47/71 were run because the prespecified promotion condition failed.

The historical four-seed DCH WaveFake signal was further separated from static source training by using an equal-class, equal-attack-family source-fit descriptor to generate a fixed head. The target descriptor averaged ΔAUC `−0.000696` and EER gain `−0.061 pp` against that head on WaveFake; ITW increments were near zero. Head movement and a positive target-vs-Frozen comparison therefore did not establish benefit from target-distribution conditioning. This does not retroactively change the historical DCH experiment or its recorded result; the new read-only ablation is in `experiments/overnight_tta/results/dch_fixed_descriptor_20260927/`.

Current decision: **NO** qualifying feature-stage TTA candidate from this overnight run. WaveFake uncertainty uses content-pair grouping. The genuinely earlier-layer waveform route was resource-blocked and remains untested. Full methods, per-sample scores, checkpoints, timings, numeric checks and limitations are in `MORNING_REPORT.md`. No target90 labels/metrics or final holdout were accessed in this new branch.

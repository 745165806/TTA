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

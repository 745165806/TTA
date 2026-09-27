# O2 raw-versus-normalized head update — fixed development diagnostic

Status: fixed **before** any O2 head-update development score/metric. This is
one mechanism test, not a proposed TTA method. The prerequisite compliant
gradient replay `alignment_compliant_20260927a` has ITW/WaveFake full O2 cosine
`+0.319200/+0.527607` and positive chunk fraction `1.0/1.0`, meeting the
user's `>0` and `>=0.9` confirmation thresholds.

## Data, state, and objective

Use only the fixed 3,178-ID ITW target10-only feature cache and fixed 4,096-ID
WaveFake paired development cache. Follow each manifest's existing order in
contiguous B=128 chunks; keep the last shorter ITW chunk. The frozen encoder,
source-labelled anchors, frozen 3×160 features, classifier bias `b_s`, source
score direction, source threshold and source `σ_s` remain unchanged. Every
chunk begins with the same source `w_s`; no optimizer state, head, gradient or
sample is carried across chunks. There is one gradient evaluation and one
update, with no loss/search/buffer change.

For a chunk of N samples with three views `z_iv`, define exactly the existing
O2 decision-sensitive consistency objective at the source head:

`L_O2(w) = (1/N) Σ_i (1/3) Σ_v [(z_iv·w+b_s−mean_j(z_ij·w+b_s))/σ_s]²`,

where `σ_s` is the frozen source-anchor score standard deviation from the
existing O2 source geometry. Let `g=∇_w L_O2(w_s)`. Bias is held fixed and
does not enter an update. Original-view scores of the same chunk use
`z_i0·w_arm+b_s`. No target label, true class ratio, pseudo label, gate or
post-hoc correctness enters this worker.

## Fixed arms and step scales

| Arm | One-step weight |
|---|---|
| Frozen | `w_s` |
| O2-raw | `w_s − η g`, where `η=0.01||w_s||₂` |
| O2-normalized-small | `w_s − 0.01||w_s||₂ g/(||g||₂+1e−12)` |
| O2-normalized-medium | `w_s − 0.03||w_s||₂ g/(||g||₂+1e−12)` |
| O2-normalized-large | `w_s − 0.10||w_s||₂ g/(||g||₂+1e−12)` |

There was no existing **unlabeled** head-update default. The single raw η is
therefore fixed from the source-head norm at one percent, before task metrics;
it is not selected from target results. The three normalized deltas are exactly
those requested. No w renormalization, bias adaptation, multiple steps,
cross-chunk accumulation or extra penalty is used.

## Analysis and decision

Complete and verify all unlabeled ITW and WaveFake scores before opening
selected-only development labels in a separate analyzer. Compute per-domain
AUC/EER and paired stratified 1,000-draw bootstrap intervals with seed 2026;
also report score movement, head angle and step norm per arm. A normalized arm
must meet ITW `ΔAUC>=0.005 OR EER improvement>=0.005` **and** WaveFake
`ΔAUC>=0.01 OR EER improvement>=0.01` at the **same delta** for a cross-domain
effect. `O2_DIRECTION_USEFUL_BUT_RAW_MAGNITUDE_TOO_WEAK` additionally requires
O2-raw to miss the same domain thresholds in both domains. If no normalized
arm reaches either domain threshold, use
`O2_ALIGNMENT_INSUFFICIENT_FOR_TASK_CORRECTION`. If a normalized arm reaches
only one domain threshold, use `DOMAIN_DEPENDENT_O2_EFFECT`. Other patterns are
`INCONCLUSIVE`. These categories are descriptive development decisions, not
final performance claims. Scores, decisions and failed runs retain unique
paths and are never overwritten.

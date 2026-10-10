# Matched supervised one-step head oracle — fixed development diagnostic

Status: fixed before any supervised-oracle update score. This experiment uses
development labels **inside** the B128 update and evaluates the same batches.
It is an intentionally optimistic, label-using capacity diagnosis, not a TTA
method, not held-out performance, and not a final result.

## Exact comparison

Reuse only the fixed ITW target10-only 3178-row and WaveFake paired 4096-row
3×160 float32 feature caches, their fixed manifest orders, the same frozen
source 160D head `(w_s,b_s)`, and the prior O2 strength run
`o2_strength_dev_20260927a`. The O2 normalized scores are read unchanged after
exact ID/order/finite validation. Each contiguous B128 chunk starts from
`w_s`, takes one gradient, scores its current original-view rows, then discards
the head. The last short ITW chunk is retained. Bias and encoder are frozen.

For chunk `C`, let `z_i0` be the original-view embedding and `y_i` its true
selected development label. The supervised reference is

`L_sup(w_s) = mean_{i in C} BCEWithLogits(z_i0 · w_s + b_s, y_i)`;
`g_sup = ∇_w L_sup(w_s)`.

For each **fixed** `δ ∈ {0.01,0.03,0.10}`:

`w_SUP(δ) = w_s − δ ||w_s||₂ g_sup / (||g_sup||₂ + 1e−12)`.

The O2 comparison uses the previously committed identical normalized formula
and same δ, but its label-free O2 gradient. No bias step, multiple steps,
cross-chunk state, other loss, new batch size or scale search is permitted.

## Analysis

Compare seven arms: Frozen, three SUP-normalized, and three previous
O2-normalized. Report AUC/EER and paired ΔAUC/ΔEER versus Frozen; report
SUP-versus-O2 paired deltas at the same δ. Report B128-weighted mean head
angle, effective step norm and absolute score movement. Paired, class-
stratified bootstrap: 1000 resamples with independent per-domain RNG seed
2026, reusing each drawn index vector across all seven arms.

The **same δ** must pass both development-domain practical floors for Case A:
ITW `ΔAUC≥0.005 OR EER improvement≥0.005`, WaveFake
`ΔAUC≥0.01 OR EER improvement≥0.01`, while matched O2 misses both floors.
If no SUP δ meets either domain's floor, classify
`SINGLE_STEP_HEAD_PROTOCOL_INSUFFICIENT`. If SUP meets the floor in only one
domain at any δ, classify `DOMAIN_DEPENDENT_HEAD_CORRECTION_TIMESCALE`.
All other patterns are `INCONCLUSIVE`. Bootstrap intervals are development
uncertainty diagnostics and cannot turn an in-batch supervised oracle into an
independent generalization claim. No post-result threshold or arm change.

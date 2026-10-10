# H-UA1: Source-Anchored Soft-Mixture Head Adaptation — development contract

Status: fixed before any H-UA1 development task metric. H-UA1 is an **experimental label-free target-head prototype**, not a final method. The frozen SSL-AASIST encoder, 3×160 feature cache, source anchors and source head are unchanged. Target labels, target true class ratio, target tau threshold, prior target metrics and final data never enter the score worker.

## Data and reset

- ITW: the existing complete target10 3,178-ID label-free selection and selected-row feature reader; no target90 label/metric access.
- WaveFake: the existing fixed 4,096-row paired label-free selection and validated feature cache. Local full-resource audit has already seen unselected headers/codes, so those rows are not untouched final data.
- Process each domain independently in its **fixed manifest order** as contiguous B=256 buffers. The final smaller buffer is retained with actual size. No shuffle, size sweep, EMA, memory, R update, encoder update, or cross-buffer state. Each buffer starts with the same source anchors/head/covariance, estimates one head from its own unlabeled features, scores that buffer, then discards the estimate.
- Run 32 real selected rows/domain as a label-free smoke before full development scores. Do not calculate EER/AUC in smoke.

## Source-only geometry

Use frozen source-labelled anchor embeddings `a_j∈R^160`, canonical source `y_j∈{0,1}`, and frozen source head `(w_s,b_s)`. Source class centroids are `μ^s_c=mean_{j:y_j=c} a_j`. Pooled source within-class covariance is

`Σ_s = [Σ_c Σ_{j:y_j=c}(a_j−μ^s_c)(a_j−μ^s_c)ᵀ] / (N_s−2)`.

Set the only covariance ridge from source: `λ=0.1·tr(Σ_s)/160`; require λ>0. Source affinity precision is `(Σ_s+λI)⁻¹`. Define source squared distances `d^s_c(z)=(z−μ^s_c)ᵀ(Σ_s+λI)⁻¹(z−μ^s_c)`. Set the source-only soft-assignment temperature `T=median_j |d^s_0(a_j)−d^s_1(a_j)|`, require T>0. No target labels calibrate these quantities.

## Unlabeled target mixture in each buffer

For target i and each of its three fixed safe views v, compute

`q_{iv,1}=sigmoid[(d^s_0(z_iv)−d^s_1(z_iv))/T]`, `q_{iv,0}=1−q_{iv,1}`.

These are continuous source-affinity assignments, never hard pseudo-labels. Define `q_{i,1}=mean_v q_{iv,1}` and reliability `r_i=1−(max_v q_{iv,1}−min_v q_{iv,1})`, so `0≤r_i≤1`. Original-view `z_i=z_i0` is used for target centroids, covariance and final scoring. Estimate the unlabeled target prior `π_1=[Σ_i r_i q_{i,1}]/[Σ_i r_i]`, `π_0=1−π_1`. Clip only for numerical log to `[1e−4,1−1e−4]`; do not assume 50/50 or read the actual class ratio.

Let `n_c=Σ_i r_i q_{i,c}` and `μ^t_c=Σ_i r_i q_{i,c} z_i/n_c`. Require positive finite n_c; a degenerate buffer fails and is recorded rather than silently falling back. Estimate soft within-class target covariance

`Σ_t = [Σ_i Σ_c r_i q_{i,c}(z_i−μ^t_c)(z_i−μ^t_c)ᵀ]/Σ_i r_i`.

Use one fixed regularized covariance `Σ=0.5Σ_s+0.5Σ_t+λI`. Solve by Cholesky for `w_raw=Σ⁻¹(μ^t_1−μ^t_0)`. Raw LDA bias is `b_raw=−0.5(μ^t_0+μ^t_1)·w_raw+log(π_1/π_0)`. Require nonzero finite `w_raw`. Match source head norm with positive scale `c=||w_s||/||w_raw||`: `(w_t,b_t)=c(w_raw,b_raw)`.

For each **predeclared** alpha in `{0.25,0.5,0.75}` independently, use

`(w_α,b_α)=(1−α)(w_s,b_s)+α(w_t,b_t)`, `score_i(α)=z_i·w_α+b_α`.

These are the only H-UA1 arms besides Frozen. The previously saved supervised H3 linear probe is shown solely as a development upper bound. Source and target head norms, angle, prior, effective class counts, source/target covariance condition, q/reliability summaries, score deltas and runtime are saved per buffer. Every sample saves ID, domain, buffer index, Frozen score, three H-UA1 scores, q, reliability and finite status. No post-hoc label can alter a buffer.

## Score gate and task gate

Finish **all** ITW and WaveFake scores, verify exact selected-ID/arm coverage, finite values, source Frozen parity, and confirm the worker imported no target-label sidecar before selected development labels are opened by a separate analyzer. Preserve failures in a unique run `failure.json`; never overwrite/reselect.

For each *same alpha* independently, promotion requires all of:

1. ITW: `ΔAUC≥+0.005` **or** `EER improvement≥+0.005` versus Frozen.
2. WaveFake: `ΔAUC≥+0.01` **or** `EER improvement≥+0.01`.
3. Common positive direction: either both domains have `ΔAUC>0` or both have `EER improvement>0`.
4. No other-metric collapse: in each domain `ΔAUC≥−0.005` and `EER improvement≥−0.005`.

If multiple alphas pass, promote the **smallest passing alpha** to minimize source departure. Otherwise decision = `HEAD_ADAPTATION_NOT_YET_ACTIONABLE`, candidate `NONE`, and stop H-UA1. No buffer/order/temperature/covariance/alpha extension is allowed from this result. For each alpha with positive AUC gain, report `recovery_ratio_AUC=(AUC_H-UA1−AUC_Frozen)/(AUC_supervised_H3−AUC_Frozen)` separately on ITW and WaveFake; this is a development upper-bound fraction, not final performance.

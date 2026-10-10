# Representation sufficiency and adaptation capacity audit — preregistration

Status: fixed before supervised capacity metrics. This is **SUPERVISED DEVELOPMENT DIAGNOSIS**, not unsupervised TTA, a proposed method, or final performance. No target90 or final holdout labels or metrics are inputs.

## Input roles and immutable comparisons

- Primary ITW: all 3,178 existing target10 members, exact source-trained frozen SSL-AASIST cache, original view (`views[0]`), 160 float32 coordinates. Never resplit target10/target90 membership.
- Second primary: WaveFake only if local audit verifies real and fake matched by `audio_id` with no unmanageable format confound. An assignment and pair group are fixed once before model training. All eight generator versions of one `audio_id` must remain in one fold if included.
- Codecfake fixed 5,000: auxiliary **RATE_CLASS_CONFOUNDED**, never evidence of spoof-specific generalization. ASVspoof2019 LA: auxiliary **NEAR_CEILING**.
- `0=bonafide, 1=spoof`, higher score means spoof. All target labels are confined to this supervised diagnostic. Existing label-free TTA workers are not modified.

## Capacity ladder and held-out rule

For each eligible two-class domain, form five fixed stratified folds with integer seed 2026. For content-paired WaveFake, hold all rows of each `audio_id` together and balance class counts across folds. Save the exact assignment. Every prediction comes from a model trained exclusively on the other four folds. The held-out fold never chooses hyperparameters or stopping epoch. Pool the five held-out predictions for headline AUC/EER; report individual fold AUC/EER and class counts. Frozen C0 uses the same held-out IDs and source score as the existing detector.

- **C0 Frozen:** `s_i=z_i·w+b` using the unchanged source-trained head and original-view 160D frozen embedding.
- **C1 supervised 8×8 R:** `z_i(R)=z_i+((z_i U)Rᵀ)Uᵀ`, `s_i(R)=z_i(R)·w+b`; train only 64 entries of R by binary cross entropy on fold-training labels. Freeze U, w, b, encoder and features. Project `||R||_F≤0.1`, the established adapter radius. Initialize R=0 on each fold. The result is a supervised capacity bound for the *current bounded adapter space*, not a TTA arm.
- **C2 supervised linear probe:** train `s_i=z_i·v+c` from fold-training labels on original-view frozen 160D embeddings; no source head or R training. Standardize using fold-training mean and scale only, then apply that transform to its held-out fold. L2 weight penalty and fixed optimizer schedule are shared across domains.
- **C3 small nonlinear probe:** run only if C2 leaves a relevant unresolved gap: either C2 versus Frozen has absolute AUC difference at least 0.005, or both C0 and C2 have at least 0.01 AUC headroom with C2 within 0.005 of C0. Use exactly one 160→32→1 ReLU MLP, fold-training standardization, fixed training schedule and L2; no model search.

For C1/C2/C3, use Adam with fixed learning rate 0.01, at most 100 epochs, mini-batch size 256, seed 2026+fold index, fixed L2 coefficient 1e-4 on trainable weights, and inner training-only 10% stratified validation for early stopping (patience 10, minimum improvement 1e-4 in validation BCE). Restore best inner-validation epoch; fold-held-out data is used once. Identical training schedule in all domains. If no eligible inner split, mark that fold failed rather than inspect held-out labels for tuning.

## Effect interpretation

Report `ΔAUC=Ck−C0` and `EER improvement=C0−Ck`. A capacity gap is LARGE / ACTIONABLE only if `ΔAUC≥0.01` or absolute EER improvement `≥0.01` on pooled held-out predictions, without a contrary major collapse in the other ranking metric, and fold effects are directionally stable. Effect `0.005–<0.01` is MODERATE; `<0.005` is SMALL even if an interval excludes zero. Negative movements are reported with sign. No threshold is changed after results.

- C1 large: adapter has supervised capacity; label-free direction/objective is suspect.
- C1 small, C2 large: fixed source head plus bounded R is restrictive; frozen representation retains readable evidence.
- C1/C2 small, C3 large: nonlinear readout may recover evidence.
- All small: current frozen feature path may lack recoverable evidence in this diagnostic; avoid a stronger universal claim.

Cross-domain probes, if both clean primary domains are available, train on all labelled development features of one domain and evaluate the other in both directions. These are cross-domain **development** diagnostics, not final held-out claims. Gradient cosine with historical Base/O1/O2/O3 is conditional on a large C1 capacity gap and never used to design an update in this stage.

All runs create a fresh `results/<run_id>/` and retain failure records. Large predictions/cache stay untracked; small fold tables, summary and report may be committed. Historical results are read-only.

# Guard capacity contrast v0

The contrast uses the same frozen source checkpoint, source anchors, response subspace `U`, cached three-view float32 features, original-view score, episodic zero initialization, exact projection `||R||_F <= rho`, and plain SGD. For every arm and parameter set:

\[
L_{adapt}(R)=V(Z_x(R))+1.0\,\operatorname{mean}_j
  [\max(0,(1-0.1)m^0_j-m_j(R))]^2.
\]

`V` is production `view_loss` over all feature dimensions. The anchor margin is `m_j(R)=(2y_j-1)(s_j(R)-tau0)` using **source-only** anchor labels and the fixed source `tau0`. No target label, attack, or path meaning enters the update. Three settings only: A `(K=5, lr=0.01, rho=0.05)` from the target10 oracle, B `(K=5, lr=0.03, rho=0.1)` as a predeclared middle strength in the old grid, C `(K=10, lr=0.3, rho=0.2)` from old guarded-v2 unsupervised selection. `gamma=0.1`, `lambda_keep=1.0` always.

| Arm | Production method path | Change after every SGD step and projection |
|---|---|---|
| hard_guard | `run_cache_method("ep_tta_guarded", ...)`, the same core called by production `run_method` | Existing `_enforce_margin_guard_`, gamma=0.1: require every source anchor to retain at least 90% of its source margin, up to numerical tolerance; halve the candidate step up to 16 times, then revert. |
| relaxed_guard | `run_cache_method("ep_tta_guard_relaxed", ...)` | **Only** guard threshold gamma becomes 0.2: require every anchor to retain at least 80%; same backtracking, tolerance, and revert code. The differentiable loss above continues to use gamma=0.1. |
| unguarded | `run_cache_method("ep_tta", ...)`, the existing production unguarded EP core | Skip `_enforce_margin_guard_`; keep the same loss, SGD, and Frobenius projection. |

The guard threshold is a per-anchor feasibility rule, not a target prediction rule. `relaxed_guard` may damage source margins relative to the 90% criterion; that damage is measured post hoc. Hard and relaxed trace counts mean the same thing. Unguarded guard counts are exactly zero by definition. Numeric fallback is a failure to be reported, never silently counted as a successful update.

The production implementation is preserved for `ep_tta_guarded`; the new branch changes only the conditional routing to share its guard loop with the relaxed arm. Tests must compare hard output with production dispatch exactly and confirm the loss equality across arms at a fixed `R`. The arms may differ after the first update because the guard changes the parameter trajectory. The cached EP path remains CPU, regardless of GPU availability.

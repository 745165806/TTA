# Effective-head and R-expressibility audit — fixed before results

Read actual `src/eptta/adaptation/math.py::apply_adapter` and the source-only U construction in `src/eptta/offline/subspace.py`; use the frozen exported U and classifier `w_s,b_s`. For row feature `z`, production defines `z_R=z+((zU)Rᵀ)Uᵀ`, hence

`score_R=z·[w_s+U Rᵀ(Uᵀw_s)]+b_s`.

The effective head displacement is `Δw_R=Uq`, `q=Rᵀ(Uᵀw_s)`. Since `Uᵀw_s≠0`, every direction in `span(U)` is algebraically reachable with unrestricted R, but the production radius `||R||_F≤0.1` limits `||q||≤0.1||Uᵀw_s||`. This is an exact score-family identity, not a learned TTA proposal.

Numerical verification: use 20 seed-2026 random R matrices projected into the radius, 32 fixed ITW original-view embeddings, float32 production `apply_adapter`, and compare direct scores with derived effective-head scores; maximum absolute difference must be ≤1e-5. Check `||UᵀU−I||_max≤1e-4` before using orthogonal projection.

For each five-fold H3 supervised target head, remove arbitrary positive score scale: `target_same_norm=||w_s||·w*/||w*||`, `Δw=target_same_norm−w_s`. Compute `projection=UUᵀΔw`, residual `Δw−projection`, `explained_fraction=||projection||²/||Δw||²`, and `cosine(Δw,projection)`. Also clamp the coefficient vector to the production q-radius and report `bounded_explained_fraction=1−||Δw−Uq_clamped||²/||Δw||²`. Save all fold values and the exact U/w/condition diagnostics in a distinct run. Do not use target labels to change U, R, folds or projection.

Record `R_PARAMETERIZATION_MISMATCH` only if at least four of five folds **in each primary domain** have subspace explained fraction <0.5. Report the bounded fractions separately. If this rule fails, report the measured fractions without a mismatch claim. No H-UA1 experiment begins before this audit is committed.

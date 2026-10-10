# C1 bounded-adapter optimization check — preregistration

The original C1 Adam results remain unchanged. This single follow-up is justified by C1 reaching `||R||_F=0.1` in both primary domains: an optimizer/early-stop artifact would invalidate a parameter-space bottleneck conclusion. This is a **supervised development optimization check**, not another TTA method or hyperparameter sweep.

With frozen classifier and U, `s_i(R)=s_i(0)+(z_i U)·q`, where `q=Rᵀ(Uᵀw)`. There is a feasible R with `||R||_F≤ρ` exactly when `||q||_2≤ρ||Uᵀw||_2`. Thus the same 8×8 bounded R score family can be optimized as an 8-dimensional convex constrained logistic problem. No feature, classifier, radius, label population, outer fold, loss family or regularization coefficient changes.

For each of the saved five outer folds, solve BCE + `1e-4 * ||q||²/||Uᵀw||²` on **all four training folds only**, under `||q||≤0.1||Uᵀw||`. Use SciPy 1.13.0 SLSQP with analytic objective gradient and analytic radius constraint, initial q=0, fixed `maxiter=500`, `ftol=1e-10`. Do not inspect or use held-out labels to select a solution. Require numerical success, finite scores and radius feasibility; otherwise save failure and make no optimized-C1 claim. Pool never-trained-on held-out scores and report AUC/EER and five fold metrics alongside unchanged C0/C1/C2 results.

Interpretation fixed in advance: a material optimized-C1 gap (same ≥0.01 AUC or ≥0.01 EER-improvement threshold) would show the original Adam C1 underestimated R capacity; an optimized result still practically SMALL would strengthen the bounded-R capacity limit. The verification never replaces the preregistered original C1 or changes the chosen development assignment.

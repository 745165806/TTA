# R expressibility from actual production mathematics

`apply_adapter` implements `z_R=z+((zU)Rᵀ)Uᵀ`. Thus `w_eff=w_s+U Rᵀ(Uᵀw_s)` and all unconstrained direction changes lie in span(U). With ||R||_F≤0.1, the coefficient radius is 0.1||Uᵀw_s||.

Float32 parity max |direct−derived score|: 1.90734863e-06. U orthogonality max error: 2.53310306e-08.

| Domain | Mean subspace explained | Mean radius-limited explained | Folds <0.5 |
|---|---:|---:|---:|
| itw | 0.172633 | 0.033830 | 5/5 |
| wavefake | 0.171386 | 0.033058 | 5/5 |

Predeclared R_PARAMETERIZATION_MISMATCH = True.

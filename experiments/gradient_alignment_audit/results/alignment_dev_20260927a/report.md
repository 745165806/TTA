# Gradient Alignment Audit

Supervised gradients use selected development labels solely as a diagnostic reference; no head update or new TTA method was run. All five unlabeled gradients were computed before target labels were opened.

| Domain | Objective | Full cosine | Mean cosine | Median cosine | p25 | p75 | Positive chunk fraction | Full grad norm ratio | Bias sign-match fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| itw | ENT | 0.1476 | 0.1268 | 0.1380 | -0.0031 | 0.2531 | 0.6800 | 0.2828 | 1.0000 |
| itw | PL | -0.6541 | -0.6479 | -0.6549 | -0.6803 | -0.6468 | 0.0000 | 1.4169 | 0.0000 |
| itw | O1 | 0.6786 | 0.6662 | 0.6442 | 0.5585 | 0.8662 | 1.0000 | 0.0251 | 0.9600 |
| itw | O2 | 0.3192 | 0.3216 | 0.3256 | 0.2711 | 0.3459 | 1.0000 | 0.1102 | N/A |
| itw | O3 | 0.3864 | 0.3921 | 0.4028 | 0.3445 | 0.4388 | 1.0000 | 0.1023 | 0.9600 |
| itw | Base_view_variance | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | N/A |
| wavefake | ENT | -0.9002 | -0.8875 | -0.9096 | -0.9369 | -0.8700 | 0.0000 | 0.1280 | 0.0000 |
| wavefake | PL | 0.9933 | 0.9927 | 0.9928 | 0.9919 | 0.9952 | 1.0000 | 0.9096 | 1.0000 |
| wavefake | O1 | -0.9532 | -0.9396 | -0.9563 | -0.9813 | -0.9172 | 0.0000 | 0.0138 | 0.0000 |
| wavefake | O2 | 0.5276 | 0.5247 | 0.5302 | 0.5058 | 0.5409 | 1.0000 | 0.0187 | N/A |
| wavefake | O3 | -0.0871 | -0.1275 | -0.1143 | -0.2326 | 0.0005 | 0.2500 | 0.0155 | 0.0000 |
| wavefake | Base_view_variance | N/A | N/A | N/A | N/A | N/A | N/A | 0.0000 | N/A |

Decision: **DIRECTION_ALIGNED_BUT_EXECUTION_INEFFECTIVE**. ITW/WaveFake selected counts: 3178/4096; chunk counts: 25/32. `Base_view_variance` has no head gradient by definition. Full-domain gradients are exact sample-weighted aggregates of 128-row chunk gradients. For O1/O2/O3, this is a parameter-scope lift of the existing R loss at R=0, not a replay of its R-update gradient. No target90 feature values, labels or metrics were read.

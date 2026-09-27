# O2 direction/strength development diagnostic

Fixed B128, source-reset, one-step O2 head direction; frozen bias. Selected labels were opened only after both score files passed exact coverage.

| Domain | Arm | AUC | EER | ΔAUC | ΔEER | Mean absolute score move | Mean angle (deg) | Mean step norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| itw | Frozen | 0.963309 | 0.098571 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 |
| itw | O2-raw | 0.963312 | 0.098571 | +0.000003 | +0.000000 | 0.016775 | 0.102678 | 0.001833 |
| itw | O2-normalized-small | 0.963308 | 0.098346 | -0.000000 | -0.000224 | 0.077709 | 0.489137 | 0.008476 |
| itw | O2-normalized-medium | 0.963294 | 0.098346 | -0.000015 | -0.000224 | 0.233128 | 1.483629 | 0.025428 |
| itw | O2-normalized-large | 0.963239 | 0.098571 | -0.000070 | +0.000000 | 0.777092 | 5.125888 | 0.084758 |
| wavefake | Frozen | 0.915003 | 0.157715 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wavefake | O2-raw | 0.914999 | 0.157227 | -0.000004 | -0.000488 | 0.009018 | 0.044263 | 0.000868 |
| wavefake | O2-normalized-small | 0.914854 | 0.157227 | -0.000149 | -0.000488 | 0.088089 | 0.490222 | 0.008476 |
| wavefake | O2-normalized-medium | 0.914368 | 0.158203 | -0.000635 | +0.000488 | 0.264267 | 1.486985 | 0.025428 |
| wavefake | O2-normalized-large | 0.910994 | 0.160156 | -0.004009 | +0.002441 | 0.880891 | 5.136130 | 0.084758 |

Decision under the fixed contract: **O2_ALIGNMENT_INSUFFICIENT_FOR_TASK_CORRECTION**. The paired stratified 1,000-draw development intervals are in `bootstrap.csv`. This is a development mechanism test, not final TTA performance.

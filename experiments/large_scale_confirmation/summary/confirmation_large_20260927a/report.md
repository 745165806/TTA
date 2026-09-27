# Large-scale development confirmation

All label-free scores and buffer resets passed exact coverage before selected audit labels opened.

| Domain | Arm | Mean AUC ± std | Mean EER ± std | Mean ΔAUC | ΔAUC order std |
|---|---|---:|---:|---:|---:|
| in_the_wild | Frozen | 0.963309 ± 0.000000 | 0.098571 ± 0.000000 | +0.000000 | 0.000000 |
| in_the_wild | Per-sample Base | 0.963866 ± 0.000000 | 0.099556 ± 0.000000 | +0.000557 | 0.000000 |
| in_the_wild | Local-Base B32 | 0.963317 ± 0.000036 | 0.098503 ± 0.000289 | +0.000008 | 0.000036 |
| in_the_wild | Local-O1 B32 | 0.963340 ± 0.000021 | 0.098466 ± 0.000293 | +0.000031 | 0.000021 |
| codecfake | Frozen | 0.823094 ± 0.000000 | 0.260933 ± 0.000000 | +0.000000 | 0.000000 |
| codecfake | Per-sample Base | 0.822833 ± 0.000000 | 0.260933 ± 0.000000 | -0.000262 | 0.000000 |
| codecfake | Local-Base B32 | 0.823800 ± 0.002108 | 0.260933 ± 0.000000 | +0.000706 | 0.002108 |
| codecfake | Local-O1 B32 | 0.824127 ± 0.002770 | 0.260933 ± 0.000000 | +0.001033 | 0.002770 |
| asv2019_la_dev | Frozen | 0.999962 ± 0.000000 | 0.000445 ± 0.000000 | +0.000000 | 0.000000 |
| asv2019_la_dev | Per-sample Base | 0.999963 ± 0.000000 | 0.000445 ± 0.000000 | +0.000000 | 0.000000 |
| asv2019_la_dev | Local-Base B32 | 0.999953 ± 0.000003 | 0.000445 ± 0.000000 | -0.000009 | 0.000003 |
| asv2019_la_dev | Local-O1 B32 | 0.999952 ± 0.000004 | 0.000445 ± 0.000000 | -0.000010 | 0.000004 |

Preregistered decision: **SMALL_DEVELOPMENT_ARTIFACT**. Bootstrap intervals are development-only diagnostics. No method is implemented in this stage.

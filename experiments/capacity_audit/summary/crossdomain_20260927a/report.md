# Cross-domain supervised development probe

C2 linear probe trained on one development resource and tested on the other. This is not final evaluation.

| Train → Test | Frozen AUC/EER | Probe AUC/EER | ΔAUC | EER improvement |
|---|---:|---:|---:|---:|
| itw → wavefake | 0.915003/0.157715 | 0.903675/0.172363 | -0.011328 | -0.014648 |
| wavefake → itw | 0.963309/0.098571 | 0.960763/0.100049 | -0.002545 | -0.001479 |

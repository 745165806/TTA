# EPDC v0-A development result

Three cached domains scored; only In-the-Wild has both classes. Codecfake/WaveFake NOT_RUN.

| Domain | Arm | EER | AUC | Mean evidence damage | Mean order damage | Helpful | Harmful |
|---|---|---:|---:|---:|---:|---:|---:|
| in_the_wild | Frozen | 0.099010 | 0.957841 | NA | NA | 0 | 0 |
| in_the_wild | Base Adapt | 0.103960 | 0.958783 | 0.170890 | 0.000000 | 0 | 1 |
| in_the_wild | Base + Preserve | 0.103960 | 0.958783 | 0.170887 | 0.000000 | 0 | 1 |
| asv2021_la | Frozen | NA | NA | NA | NA | 0 | 0 |
| asv2021_la | Base Adapt | NA | NA | 0.122739 | 0.000000 | 0 | 1 |
| asv2021_la | Base + Preserve | NA | NA | 0.122738 | 0.000000 | 0 | 1 |
| asv2021_df | Frozen | NA | NA | NA | NA | 0 | 0 |
| asv2021_df | Base Adapt | NA | NA | 0.449134 | 0.000000 | 0 | 4 |
| asv2021_df | Base + Preserve | NA | NA | 0.449121 | 0.000000 | 0 | 4 |

EER/AUC macro values have one-domain coverage; no cross-domain ranking claim.

# EPDC v0-B normalized source-margin evidence development result

Three cached domains scored; only In-the-Wild has both classes. Codecfake/WaveFake NOT_RUN.

| Domain | Arm | EER | AUC | Mean evidence damage | Mean preservation loss | Helpful | Harmful |
|---|---|---:|---:|---:|---:|---:|---:|
| in_the_wild | Frozen | 0.099010 | 0.957841 | NA | NA | 0 | 0 |
| in_the_wild | Base Adapt | 0.103960 | 0.958783 | 0.170890 | 1.521463 | 0 | 1 |
| in_the_wild | Base + Preserve | 0.103960 | 0.956404 | 0.000001 | 0.000000 | 9 | 1 |
| asv2021_la | Frozen | NA | NA | NA | NA | 0 | 0 |
| asv2021_la | Base Adapt | NA | NA | 0.122739 | 1.050840 | 0 | 1 |
| asv2021_la | Base + Preserve | NA | NA | 0.000002 | 0.000000 | 10 | 0 |
| asv2021_df | Frozen | NA | NA | NA | NA | 0 | 0 |
| asv2021_df | Base Adapt | NA | NA | 0.449134 | 4.569358 | 0 | 4 |
| asv2021_df | Base + Preserve | NA | NA | 0.000000 | 0.000000 | 9 | 0 |

EER/AUC macro values have one-domain coverage; no cross-domain ranking claim.

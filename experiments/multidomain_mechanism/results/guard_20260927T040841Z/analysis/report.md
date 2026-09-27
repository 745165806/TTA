# Guard capacity: partial mechanism-dev analysis

Three cached domains only. Codecfake and WaveFake: NOT_RUN. No target90 or final-holdout labels used.

| Domain | Setting | Arm | EER | AUC | Helpful | Harmful | Evidence damage |
|---|---|---|---:|---:|---:|---:|---:|
| in_the_wild | Frozen | Frozen | 0.099010 | 0.957841 | 0 | 0 | NA |
| in_the_wild | A | hard_guard | 0.099010 | 0.957857 | 0 | 0 | 0.000012 |
| in_the_wild | A | relaxed_guard | 0.099010 | 0.957889 | 0 | 0 | 0.000140 |
| in_the_wild | A | unguarded | 0.103226 | 0.958623 | 0 | 0 | 0.019033 |
| in_the_wild | B | hard_guard | 0.099010 | 0.957793 | 0 | 0 | 0.000015 |
| in_the_wild | B | relaxed_guard | 0.099010 | 0.957809 | 0 | 0 | 0.000178 |
| in_the_wild | B | unguarded | 0.103960 | 0.958879 | 0 | 0 | 0.029171 |
| in_the_wild | C | hard_guard | 0.099010 | 0.955589 | 0 | 0 | 0.000022 |
| in_the_wild | C | relaxed_guard | 0.099010 | 0.955509 | 0 | 0 | 0.000284 |
| in_the_wild | C | unguarded | 0.103960 | 0.947653 | 3 | 0 | 0.172484 |
| asv2021_la | Frozen | Frozen | NA | NA | 0 | 0 | NA |
| asv2021_la | A | hard_guard | NA | NA | 0 | 0 | 0.000011 |
| asv2021_la | A | relaxed_guard | NA | NA | 0 | 0 | 0.000131 |
| asv2021_la | A | unguarded | NA | NA | 0 | 0 | 0.014022 |
| asv2021_la | B | hard_guard | NA | NA | 0 | 0 | 0.000016 |
| asv2021_la | B | relaxed_guard | NA | NA | 0 | 0 | 0.000188 |
| asv2021_la | B | unguarded | NA | NA | 0 | 0 | 0.021465 |
| asv2021_la | C | hard_guard | NA | NA | 0 | 0 | 0.000028 |
| asv2021_la | C | relaxed_guard | NA | NA | 0 | 0 | 0.000339 |
| asv2021_la | C | unguarded | NA | NA | 3 | 1 | 0.130316 |
| asv2021_df | Frozen | Frozen | NA | NA | 0 | 0 | NA |
| asv2021_df | A | hard_guard | NA | NA | 0 | 0 | 0.000015 |
| asv2021_df | A | relaxed_guard | NA | NA | 0 | 0 | 0.000188 |
| asv2021_df | A | unguarded | NA | NA | 0 | 1 | 0.049689 |
| asv2021_df | B | hard_guard | NA | NA | 0 | 0 | 0.000017 |
| asv2021_df | B | relaxed_guard | NA | NA | 0 | 0 | 0.000213 |
| asv2021_df | B | unguarded | NA | NA | 0 | 1 | 0.072628 |
| asv2021_df | C | hard_guard | NA | NA | 1 | 0 | 0.000027 |
| asv2021_df | C | relaxed_guard | NA | NA | 1 | 1 | 0.000327 |
| asv2021_df | C | unguarded | NA | NA | 5 | 2 | 0.586509 |

Single-class domains have undefined EER/AUC and are excluded from those macro means.
Figures show descriptive associations. Labels appear only in post-score analysis; no gate uses them.

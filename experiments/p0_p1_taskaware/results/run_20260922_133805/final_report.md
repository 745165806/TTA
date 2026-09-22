# P0/P1 final report

**POST_HOC_DEVELOPMENT_ONLY**

Frozen is always the first baseline.

| dataset | method | EER | AUC | dEER_vs_frozen | dAUC_vs_frozen | coverage | helpful | harmful | mean_signed_delta | mean_R_norm | safety_reject_rate | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| in_the_wild_target10_development | frozen | 0.098571 | 0.963309 | +0.000000 | +0.000000 | 0.0000 | 0 | 0 | +0.000000 | 0.000000 | 0.0000 | 0 |
| in_the_wild_target10_development | taskaware_control | 0.098346 | 0.963304 | -0.000224 | -0.000005 | 0.9978 | 0 | 0 | -0.002084 | 0.000742 | 0.0000 | 0 |
| in_the_wild_target10_development | taskaware_full | 0.099217 | 0.963368 | +0.000646 | +0.000059 | 0.3260 | 0 | 0 | -0.020624 | 0.004533 | 0.1850 | 0 |
| in_the_wild_target10_development | taskaware_no_gate | 0.099217 | 0.963368 | +0.000646 | +0.000059 | 0.4012 | 0 | 0 | +0.011991 | 0.008293 | 0.5988 | 0 |
| in_the_wild_target10_development | taskaware_no_source_keep | 0.098346 | 0.963027 | -0.000224 | -0.000281 | 0.2574 | 0 | 0 | -0.013097 | 0.001966 | 0.2536 | 0 |

mechanism gate (taskaware_full): {"numeric_fallback_count": 0, "accepted_source_anchor_flip_max": 0, "adaptation_coverage": 0.32599118942731276, "mean_signed_task_delta": -0.020624405451826963, "helpful_flips": 0, "harmful_flips": 0, "eer_less_than_frozen": false, "auc_greater_than_frozen": true, "numeric_ok": true, "source_safety_ok": true, "coverage_ok": true, "signed_delta_ok": false, "flip_ok": false, "task_metric_ok": true, "PASS": false}
confirmatory_run: False
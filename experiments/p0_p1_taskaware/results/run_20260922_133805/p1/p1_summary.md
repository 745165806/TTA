# P1 pilot summary

**POST_HOC_DEVELOPMENT_ONLY**

Frozen EER=0.098571 AUC=0.963309

| method | EER | AUC | mean_signed_delta | helpful | harmful | coverage | safety_reject_rate |
|---|---|---|---|---|---|---|---|
| frozen | 0.098571 | 0.963309 | 0.000000 | 0 | 0 | 0.0000 | 0.0000 |
| taskaware_full | 0.099217 | 0.963368 | -0.020624 | 0 | 0 | 0.3260 | 0.1850 |
| taskaware_no_gate | 0.099217 | 0.963368 | 0.011991 | 0 | 0 | 0.4012 | 0.5988 |
| taskaware_no_source_keep | 0.098346 | 0.963027 | -0.013097 | 0 | 0 | 0.2574 | 0.2536 |
| taskaware_control | 0.098346 | 0.963304 | -0.002084 | 0 | 0 | 0.9978 | 0.0000 |

## Mechanism gate (taskaware_full)
- numeric_ok: True
- source_safety_ok: True
- coverage_ok: True
- signed_delta_ok: False
- flip_ok: False
- task_metric_ok: True
- PASS: False

P1 mechanism: FAIL

## Scientific conclusion
- UPDATE_DIRECTION_NOT_TASK_ALIGNED: True
- OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN: True
- FROZEN_TEACHER_UNRELIABLE (frozen accuracy at tau0 < 0.5): True
- The frozen pseudo-label teacher has target10 accuracy 0.4368 at tau0, so task-space pseudo-BCE is misled for the majority of samples; this explains the negative mean signed task delta for taskaware_full.
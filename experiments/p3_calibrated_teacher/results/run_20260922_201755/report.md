# P3 Calibration-Aware Selective EP-TTA

**POST_HOC_DEVELOPMENT_ONLY**

- protocol: `unlabeled_domain_calibration_then_episodic_tta`
- split: `p3-main-2026`
- calU/evalU: 318/2860
- tau0/tau_hat/tau_oracle: -4.770049 / -3.129160 / 2.651596

| variant | EER | AUC | signed delta | coverage | abstention | safety reject |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 0.099804 | 0.962913 | +0.000000 | 0.0000 | 0.0000 | 0.0000 |
| CalOnly | 0.099804 | 0.962913 | +0.000000 | 0.0000 | 0.0000 | 0.0000 |
| SourceTauTeacher | 0.099804 | 0.963060 | -0.020883 | 0.3206 | 0.4944 | 0.1850 |
| CalibratedTeacher-NoSelect | 0.099804 | 0.963117 | +0.148626 | 0.7402 | 0.0000 | 0.2598 |
| CalibratedSelective | 0.099804 | 0.963097 | +0.138402 | 0.7073 | 0.1801 | 0.1126 |
| OracleTeacher | 0.096869 | 0.967869 | +0.166819 | 0.7678 | 0.1801 | 0.0521 |

- teacher accuracy source/calibrated: 0.434266 / 0.782168
- teacher delta 95% CI: [0.3304195804195804, 0.3660839160839161]
- CalOnly EER/AUC invariant: True
- P3_MECHANISM_PASS: True
- P3_MECHANISM_PARTIAL: False
- UNSUPERVISED_CALIBRATION_TEACHER_FAIL: False

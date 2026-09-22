# P2-A Calibration decomposition

**POST_HOC_DEVELOPMENT_ONLY**

## Frozen operating point vs ranking
- target EER = 0.098571, AUC = 0.963309
- source cal0 reference EER = 0.000293, AUC = 0.999990
- frozen tau0 accuracy = 0.4368, balanced accuracy = 0.5589, FPR = 0.8822

## 1. source tau0 vs target oracle threshold
- tau0 = -4.770049
- tau_target_eer = 2.705913 (delta 7.476)
- tau_target_balacc = 2.857471 (delta 7.628)

## 2/3. bias-only recovery
- balanced accuracy 0.5589 -> 0.9020
- FPR 0.8822 -> 0.0941

## 4. EER/AUC invariance (bias/affine)
- affine invariant check: {"a_positive": true, "EER_unchanged": true, "AUC_unchanged": true}

## 5. class separation
- source separation = 10.7626, target separation = 7.3164 (ratio 0.680)
- midpoint shift = 0.5898

## 6. verdict
- CALIBRATION_SHIFT_PRESENT: True
- RANKING_DEGRADATION_PRESENT: True
- CALIBRATION_ONLY_INSUFFICIENT: True
- CALIBRATION_SHIFT_CONFIRMED: False

The AUC~0.963 vs tau0-accuracy~0.437 gap is dominated by a source->target operating-point (calibration) shift of ~7.63 score units; a bias-only shift restores balanced accuracy 0.5589->0.9020 and FPR 0.8822->0.0941 WITHOUT changing EER/AUC. However target class separation also shrank (10.7626->7.3164), so ranking degradation coexists with the calibration shift.
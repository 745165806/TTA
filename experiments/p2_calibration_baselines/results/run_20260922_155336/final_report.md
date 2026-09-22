# P2 Calibration / Adaptability + Published TTA baselines

**POST_HOC_DEVELOPMENT_ONLY**

## Q1/Q2/Q3: AUC~0.963 vs tau0 accuracy~0.437
- tau0 = -4.770049, tau_target_eer = 2.705913, tau_target_balacc = 2.857471
- bias-only calibration: balanced accuracy 0.5589 -> 0.9020, FPR 0.8822 -> 0.0941
- EER/AUC strictly unchanged under monotone bias/affine (invariant check: {"a_positive": true, "EER_unchanged": true, "AUC_unchanged": true})

## Q4: ranking degradation
- source cal0 EER=0.000293 AUC=0.999990; target EER=0.098571 AUC=0.963309
- class separation source=10.7626 target=7.3164; midpoint shift=0.5898

## Q5/Q6: oracle-teacher counterfactual
- pseudo signed delta = -0.020624; oracle signed delta = 0.026228
- TEACHER_MISMATCH_SUPPORTED = True; INSUFFICIENT = False

## Research route
- CALIBRATION_SHIFT_PRESENT = True, RANKING_DEGRADATION_PRESENT = True
- ALLOW_NEXT_ADAPTATION_RESEARCH = True, STOP_NEW_EP_OBJECTIVES = False

## Verdict
The fixed-tau0 failure is dominated by a source->target operating-point shift of ~7.63 score units, but target class separation ALSO shrank (10.7626->7.3164), so ranking degradation coexists with calibration shift. P1 failed primarily because the frozen tau0 pseudo-teacher is wrong for most target samples: an oracle teacher flips the signed task delta from -0.0206 to +0.0262 and improves EER 0.0992->0.0949. Next: calibration-aware/selective adaptation with a corrected teacher, NOT further entropy/view-consistency loss grids.
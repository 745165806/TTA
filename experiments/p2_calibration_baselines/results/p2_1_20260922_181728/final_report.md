# P2.1 Published baseline validation + confirmatory

**POST_HOC_DEVELOPMENT_ONLY**

## Calibration (historical P2-A, read-only)
- CALIBRATION_SHIFT_CONFIRMED = True, CALIBRATION_ONLY = False, RANKING_DEGRADATION_PRESENT = True
- tau0=-4.7700 tau_target_balacc=2.8575; bias restores FPR 0.8822->0.0941 (EER/AUC unchanged)

## Oracle teacher (historical P2-B, read-only)
- TEACHER_MISMATCH_SUPPORTED = True
- pseudo signed delta=-0.0206, oracle signed delta=+0.0262

## P2.1 target10 pilot (NormOnly / TENT / SAR / MEMO)
| method | EER | AUC | dEER | dAUC | sdt_total | sdt_norm | sdt_update |
|---|---|---|---|---|---|---|---|
| Frozen | 0.0986 | 0.9633 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| NormOnly | 0.1619 | 0.9220 | +0.0616 | -0.0414 | -4.4375 | -4.4375 | +0.0000 |
| TENT | 0.1626 | 0.9212 | +0.0628 | -0.0422 | -4.4689 | -4.4375 | -0.0314 |
| SAR | 0.1619 | 0.9220 | +0.0616 | -0.0414 | -4.4375 | -4.4375 | -0.0000 |
| MEMO | 0.1366 | 0.9296 | +0.0371 | -0.0337 | +0.4569 | -0.0000 | +0.4569 |

## Research route
- ALLOW_NEXT_ADAPTATION_RESEARCH = True, STOP_NEW_EP_OBJECTIVES = False
- NEXT: UNSUPERVISED_TARGET_CALIBRATION -> RELIABLE_PSEUDO_TEACHER -> SELECTIVE_EVIDENCE_PRESERVING_ADAPTATION

NOTE: published ports are audited episodic audio ports under the project's no-target-history protocol; they are not claimed as exact reproductions of the original online evaluation protocol.
# Fixed-order local-distribution hypothesis test — 2026-09-27

The run used two unchanged 512-ID mechanism selections and seven preregistered arms. ITW retained 310 bonafide / 202 spoof; Codecfake official dev had 81 bonafide / 431 spoof after the complete score gate. All 7,168 score rows and 192 local buffer records were finite with exact coverage. Labels were opened only after the complete score gate passed. All local buffers started with zero `R`; no buffer inherited optimizer state.

| Domain | Frozen AUC / EER | Per-sample Base AUC | Per-sample O1 AUC | Local-Base B16 / B32 AUC | Local-O1 B16 / B32 AUC |
|---|---:|---:|---:|---:|---:|
| ITW | 0.957841 / 0.099010 | 0.958783 | 0.958799 | 0.958240 / 0.958128 | 0.958368 / 0.958224 |
| Codecfake | 0.813354 / 0.283951 | 0.811607 | 0.811206 | 0.816906 / 0.823294 | 0.819169 / 0.825585 |

Codecfake B32 local adaptation improves AUC over Frozen by `+0.009940` (Base, paired 95% development bootstrap interval `[+0.006130,+0.014495]`) and `+0.012231` (O1, `[+0.008192,+0.017073]`). Against the corresponding per-sample arm, the gains are `+0.011687` and `+0.014379`; their paired intervals remain positive. ITW local B32 gains over Frozen are only `+0.000287` (Base, interval `[-0.000224,+0.000847]`) and `+0.000383` (O1, `[-0.000160,+0.001086]`), while both point estimates are **below** their matching per-sample arms. ITW Local-Base B32 also lowers balanced accuracy from `0.569355` Frozen to `0.541935`. EER is unchanged from Frozen for all local arms in both domains.

Buffer-level mean `R` norms range from `0.004495–0.014041`; the largest observed buffer norm is `0.050450`, below the fixed `rho=0.1`. There are zero numeric failures and zero first-step reset violations. Source evidence damage remains measurable (`0.034479–0.200143`), so score correction is not automatically safe. ITW Local-Base moves more B16 buffers wholly to one threshold side (`5/32` Frozen → `12/32` adapted) and B32 buffers (`0/16` → `5/16`); Codecfake B32 does not show this increase (`6/16` → `6/16`). This is threshold-region concentration evidence, not proof of a universal class collapse.

**Decision: no promoted candidate; cross-domain ranking benefit is inconclusive.** Local context has a clear Codecfake development effect, but the ITW effect is small, uncertain, and weaker than per-sample adaptation. The prespecified criterion requires stable task-useful correction across both two-class domains. Stop buffer-size expansion and do not add local-prior preservation, gate, continual state, or method lock from this result. The Codecfake effect remains a mechanism clue for a future explicitly justified objective hypothesis, not a final method claim. All intervals are development diagnostics only.

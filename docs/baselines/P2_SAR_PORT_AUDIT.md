# P2 SAR audio port audit

- method_id: `sar_audio_ep`
- paper: Towards Stable Test-Time Adaptation in Dynamic Wild World (Niu et al., ICLR 2023 Oral), <https://arxiv.org/abs/2302.12400>
- official repository: <https://github.com/mr-eggplant/SAR>
- license: MIT (repo LICENSE); commit not pinned locally (`UNPINNED_AUDIT_ONLY`)

## Core algorithm (must preserve)
Reliable entropy minimization + Sharpness-Aware Minimization (SAM).
- Reliability filter: keep a sample only if its softmax entropy `E(x) < margin`, where `margin = 0.4 * log(C)`.
- SAM: two-phase update — first-step ascent on the perturbation direction, second-step descent.

## Class-count-scaled audio mapping
Binary anti-spoofing `C = 2`, so `margin = 0.4 * log(2) = 0.2773` (NOT the ImageNet `0.4 * log(1000)` constant).

## Parameter scope
Same controlled normalization scope as TENT: backend BatchNorm1d/BatchNorm2d affine only.

## Optimizer / SAM
- base optimizer: SGD, lr=0.001, momentum=0.9, weight_decay=0.
- SAM rho = 0.05 (official code); true two-step `first_step`/`second_step` (no gradient-clipping substitute).

## Protocol deviations
1. per_sample reset (no online recovery/EMA history carried across samples).
2. empty reliable set => `adaptation_applied=false`, `score_after=frozen`, `abstain_reason=unreliable_entropy` (NOT a numeric failure).

## Status
`AUDITED_UNVERIFIED`.

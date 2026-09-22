# P2.1 SAR audio port audit

- method_id: `sar_audio_ep`
- port_scope: `episodic_audio_port`
- paper: SAR (Niu et al., ICLR 2023 Oral), <https://arxiv.org/abs/2302.12400>
- official repository: <https://github.com/mr-eggplant/SAR>
- pinned commit: `20f6e24b17525f34503510afccedc0629b67b7c4`

## Learning rate (corrected)
Official `main.py` ResNet50-BN / bs<32 branch uses `lr = (0.00025 / 64) * batch_size * 2`; the `exp_type == bs1` SAR branch doubles again.
Audio mapping (single-sample backend BN):
- batch_size = 1
- base = (0.00025/64)*1*2 = 7.8125e-6
- SAR bs1 doubled = **1.5625e-5**

Base optimizer: SGD, momentum=0.9. SAM rho = 0.05.

## Reliable filter + SAM
Two-phase: `entropy1 -> filter1 -> first_step -> entropy2 -> filter2 -> second_step`.
If first pass reliable but the SAM-perturbed second pass `entropy2 >= margin`, skip the second-step update safely (no NaN).

## Status
`VERIFIED` (episodic audio port audit complete).

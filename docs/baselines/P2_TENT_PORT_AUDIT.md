# P2 TENT audio port audit

- method_id: `tent_audio_ep`
- paper: TENT: Fully Test-Time Adaptation by Entropy Minimization (Wang et al., ICLR 2021), <https://arxiv.org/abs/2006.10726>
- official repository: <https://github.com/DequanWang/tent>
- license: MIT (repo LICENSE); commit not pinned locally (`UNPINNED_AUDIT_ONLY`)

## Core algorithm (must preserve)
Prediction entropy minimization over the normalization-layer affine parameters.
`L = -sum_c p_c log p_c` (softmax entropy); update only BatchNorm `weight`/`bias`.

## Parameter scope
- official: BatchNorm affine of all normalization layers.
- audio port: BatchNorm1d / BatchNorm2d affine of the **task-trained anti-spoofing backend only**; the generic XLS-R frontend is excluded (it has no BatchNorm, and is not a task-trained normalization shift signal).

## Optimizer
- official (ImageNet-C): SGD, lr=0.001, momentum=0.9, weight_decay=0, 1 update per batch.
- audio port: same.

## Model mode / protocol deviations
1. **per_sample reset** (project episodic contract) instead of official online continual updates.
2. whole model `eval()`; selected BatchNorm modules `train()` + `affine.requires_grad=True` + use batch statistics + disable running-stat accumulation. This avoids stochastic dropout noise being misattributed to TTA gain.
3. episodic scoring: frozen forward -> entropy loss -> update -> second forward on the SAME sample -> post-update score -> reset.

## Exact updated parameter names
Captured at runtime and written to `updated_parameter_names.txt`; every other parameter must have `requires_grad=False` and `grad is None`.

## Status
`AUDITED_UNVERIFIED` — the algorithm audit is complete; parity + unit tests must pass before the method is marked `IMPLEMENTED_UNVERIFIED`.

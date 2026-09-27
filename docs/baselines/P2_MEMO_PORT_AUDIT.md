# P2 MEMO audio port audit

- method_id: `memo_audio_ep_full`
- paper: MEMO: Test Time Robustness via Adaptation and Augmentation (Zhang et al., NeurIPS 2021), <https://arxiv.org/abs/2110.09506>
- official repository: <https://github.com/zhangmarvin/memo>

## Core algorithm (must preserve)
Single-input test-time adaptation: generate multiple augmentations of one sample, minimize the **marginal entropy of the averaged prediction** (entropy of mean probability), NOT the mean of per-view entropies.

## Blocking issue
The official **optimizer and learning rate** could not be confirmed from the paper/official source within this environment. Per the no-guessing rule, the full port is blocked rather than run with an invented optimizer.

## Status
`PORT_BLOCKED_AUDIT`. If the optimizer/lr is later confirmed from the official repository, the audit can be completed. A backend-only diagnostic (if ever needed) must use a separate method id (`memo_audio_backend_diagnostic`) and may not enter the published-baseline main table.

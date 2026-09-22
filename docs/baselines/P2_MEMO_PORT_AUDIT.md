# P2.1 MEMO audio port audit

- method_id: `memo_audio_ep_full`
- port_scope: `episodic_audio_port`
- paper: MEMO (Zhang et al., NeurIPS 2021), <https://arxiv.org/abs/2110.09506>
- official repository: <https://github.com/zhangmarvin/memo>
- pinned commit: `228b2908d271c954ef8bf19cf143ede3b2fa8e3e`

## Optimizer (corrected — no longer blocked)
Official `imagenet-exps/test_calls/test_adapt.py` default is **SGD, lr=2.5e-4, weight_decay=0, niter=1** (some launchers use AdamW lr=1e-5 wd=0.01).
Audio main port: SGD, lr=2.5e-4, weight_decay=0, niter=1 (official `test_adapt.py` default mapping). AdamW mapping is intentionally not compared this round.

## Objective
Marginal entropy `entropy(mean_i p(y|aug_i))`, NOT `mean_i entropy(p_i)`.

## Augmentation
Official image augmentation ensemble -> project fixed audio probe3 policy (view0 original, view1 noise, view2 FIR). Noise view is known weaker (P0) but kept for fairness.

## Parameter scope
`memo_audio_ep_full` = full SSL-AASIST (XLS-R frontend + backend). If CUDA OOM -> `NOT_RUN_RESOURCE_LIMIT` (never silently reduced to backend-only).

## Status
`VERIFIED` (episodic audio port audit complete).

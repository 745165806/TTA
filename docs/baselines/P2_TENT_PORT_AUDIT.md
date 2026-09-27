# P2.1 TENT audio port audit

- method_id: `tent_audio_ep`
- port_scope: `episodic_audio_port`
- paper: TENT (Wang et al., ICLR 2021), <https://arxiv.org/abs/2006.10726>
- official repository: <https://github.com/DequanWang/tent>
- pinned commit: `e9e926a668d85244c66a6d5c006efbd2b82e83e8`

## Optimizer (corrected)
Official `cfgs/tent.yaml` uses **Adam, lr=1e-3, steps=1, weight_decay=0** (NOT SGD).
Audio mapping: Adam, lr=1e-3, steps=1, weight_decay=0.

## Parameter scope
Backend BatchNorm1d/BatchNorm2d affine only; generic XLS-R frontend excluded.

## Model mode / protocol deviations
- online continual TTA -> per-sample episodic adaptation (no target history).
- `post_update_same_sample_scoring = true`: post-update score is a second forward on the SAME sample (not the original online semantics where the update affects later samples).
- `AUDIO_EVAL_EXCEPT_NORM_MODE`: whole model eval(); selected backend BatchNorm train() with batch statistics; avoids dropout stochasticity.

## Status
`VERIFIED` (episodic audio port audit complete; parity + unit tests gate execution).

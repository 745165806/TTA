# AASIST source detector audit

- Author repository: `https://github.com/clovaai/aasist.git`
- Bound commit: `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`
- Entrypoint: `models/AASIST.py`; SHA-256 `9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50`
- License: MIT, repository `LICENSE` SHA-256 `da2e79b8592d166ef505224300968b80ebe1e4c217c43b94a5ec627d81cd4142`
- Input/output: `[B,64600]` waveform; author forward returns the 160-dimensional `last_hidden` and two native logits.
- Native classes: `spoof=0, bonafide=1`. Project canonical labels are converted by name; project score is always `logit_spoof-logit_bonafide`.
- Initialization: author-native trainable initialization only. Author task checkpoint files are never accepted by the source job.
- Author reference: batch 24, Adam 1e-4, weight decay 1e-4, per-step cosine schedule, semantic weights `spoof=.1, bonafide=.9`, frequency augmentation false when absent from config.
- Project differences: checkpoint selection is complete `source_val` EER rather than ASV t-DCF/eval access; checkpoint state additionally records recipe/data/RNG/sampler lineage. DDP uses an explicit global tail policy and exact weighted denominator.

Local 2026-09-14 structure smoke built this exact source and returned embedding `[1,160]`, logits `[1,2]`, finite output. This is not task training or accuracy evidence.

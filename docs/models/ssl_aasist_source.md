# SSL-AASIST source detector audit

- Author repository: `https://github.com/TakHemlata/SSL_Anti-spoofing.git`
- Bound commit: `4acaa61dcef5f7610f43aa4d0b29c4559b970cd2`
- Entrypoint: `model.py`; SHA-256 `08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6`
- License file SHA-256: `e5ff6b23454767530b5ce87125140fe04b2cb834f29bb735b37275eaf6519fb9`.
- Input/output: `[B,64600]` waveform; XLS-R frame output is projected into the AASIST graph backend. The final linear head input is captured by a pre-hook and is 160-dimensional; native output has two logits.
- Native classes: `spoof=0, bonafide=1`; exported score is `logit_spoof-logit_bonafide`.
- Initialization: an explicitly bound, hash-verified generic XLS-R front-end artifact is mandatory. The author source hard-codes `xlsr2_300m.pt`; the worker applies exactly one runtime source replacement to that path and records the patch hash. No author anti-spoofing task checkpoint is accepted.
- Author reference: batch 14, Adam 1e-6, weight decay 1e-4, semantic weights `spoof=.1, bonafide=.9`; the author CLI defaults to RawBoost algorithm 5. The current project template deliberately sets no augmentation and names that deviation for review rather than silently emulating an unaudited recipe.
- Frozen inference moves the whole model before `eval()` and parity checks module modes and running buffers around forward, covering the author SSL helper's conditional `train()` behavior.

The local tree has no bound generic XLS-R artifact and the py38 environment has no installed `fairseq` package (the author repository contains a vendored revision). SSL construction, gradients, memory and training remain `NOT_RUN/DEFERRED_REMOTE` until the explicit initialization and environment preflight exist.

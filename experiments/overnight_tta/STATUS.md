# Overnight TTA status

- Start: 2026-09-27 14:56:10 UTC; deadline: 2026-09-27 22:56:10 UTC.
- Isolated branch: `exp-overnight-tta`, based on `fa37e9a`.
- GPU: `nvidia-smi` exit 9; `tta` torch CUDA available false, device count 0. Earlier-layer waveform route B `BLOCKED_RESOURCE`.
- T3A: exact original native two-class weight/bias available in project-trained frozen `detector_state.pt`; spoof-minus-bonafide parity with `linear_head.pt` max error 0. Author template behavior inspected.
- Route A seed13 source train: PASS, 10 full epochs; 1,790 steps and 458,240 presentations per arm, all 25,380 fit IDs seen. ERM selected epoch10; residual selected epoch4 by source select. Training curve and every epoch checkpoint saved.
- Route A primary source-selected LR 1e-4: ITW/WaveFake development PASS. No adapted-vs-own-source-only benefit; no promotion to seeds29/47/71. Mature T3A-batch port also completed.
- One development-informed revision at LR 1e-3: PASS, primary retained; ITW degraded substantially, WaveFake below static ERM. No further revision/search.
- Historical four-seed DCH fixed-source-descriptor ablation: PASS, source-only correction explains WaveFake gain; target conditioning has negative mean gain there.
- Final target adapter states captured and exact score replay verified for primary and revision. `MORNING_REPORT.md` outcome NO / method NONE.
- ITW target10/WaveFake development complete. Target90/final holdout: closed.

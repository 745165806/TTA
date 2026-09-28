# Representation TTA status — 2026-09-28

- Root worktree had uncommitted edits; isolated `exp-representation-tta` worktree created from `f58f3c2`, old worktree unchanged.
- Default sandbox: no `/dev/nvidia*`, `tta` torch CUDA unavailable. Escalated workstation context: four RTX A6000 visible, `tta` CUDA forward/backward PASS.
- Actual author model inspected. Probe on one real source waveform: SSL `[1,201,1024]`, LL `[1,201,128]`, complete waveform score vs exact pre-backend replay max absolute logit error `0`. See `probe_model.json`.
- Source/waveform LL extractor, strict sharded reader, 16,576-parameter residual adapter, equal-budget static/candidate source trainer, unlabeled target updater, and two-domain evaluator implemented; `compileall` PASS.
- Bounded 16-audio extraction `ll_smoke_20260928b`: PASS on GPU0, pre-backend features `[16,3,201,128]`, peak 2.12 GiB. First smoke `ll_smoke_20260928a` retained; it revealed TF32 mismatch with production extraction. Corrected numerical mode disabled TF32.
- Adapter smoke: PASS; disabled adapter exactly recovers the replay backend, maximum cross-batch difference from legacy Frozen feature cache 0.000486, finite nonzero gradient norm 4.266, 16,576 trainable parameters. See `smoke_adapter.json`.
- Unlabeled updater smoke with synthetic engineering prototypes: PASS, four updates, finite adapter delta norm 0.03256 and max score change 0.0593. This is **not** source training or target evaluation. See `smoke_adapt.json`.
- Full LL extraction of fixed source fit/select and ITW target10/WaveFake development: **BLOCKED_BY_AUTO_REVIEW**, two explicit rejections. Automatic reviewer cites `AGENTS.md` prohibition of full caching and target evaluation, despite the current user's explicit authorization. No full extraction, source training, target evaluation, source-selected checkpoint, four-seed result, or promotion decision exists yet. No target90/final holdout accessed.
- An explicit approval question for exactly 44,377 selected audio samples and about 12.76 GiB new LL cache is pending. No indirect full run attempted.

# Stage-1/2 continuation environment correction (2026-10-08)

The initial waiting controller was started at 06:38:42 UTC in the default
filesystem sandbox. Stage-1 CE and joint training were already running in a
GPU-visible environment and were not affected. At 07:52:54 UTC, a new Python
process in the default sandbox reported `torch.cuda.is_available() == False`
and zero devices; `nvidia-smi` could not contact NVML there. At 07:54 UTC,
the same `tta` Python command in the approved GPU-visible environment
reported `True, 4`. The first controller was stopped at 07:55:16 UTC with
exit code 143, before either stage-1 gate or stage-2 meta training began.
Its original wait/stop events remain in
`runs/stage2_20261008/launch.jsonl`; no files were overwritten.

The corrected controller started at 07:55:40 UTC in the GPU-visible `tta`
environment, using new exclusive root `runs/stage2_20261008_gpu`. Its command:

```bash
conda run --no-capture-output -n tta python experiments/meta_audio_tta/continue_stage12.py --stage1-root experiments/meta_audio_tta/runs/stage1_20261008 --stage2-root experiments/meta_audio_tta/runs/stage2_20261008_gpu --wait-hours 10 > experiments/meta_audio_tta/runs/stage1_20261008/stage12_controller_gpu.log 2>&1
```

Current status at 07:55:40 UTC: **WAIT_STAGE1**. It will execute the same
real-model gate and fixed Cross/Same config after both eight-epoch source
selections exist. Its actual commands, exit codes and logs are written under
`runs/stage2_20261008_gpu`; phase-2 scientific outcome remains **NOT_RUN**.
The 10-hour controller wait limit is an operational timeout only. If it
expires, phase 2 remains `NOT_RUN`; inspect source-training status and the
controller log before deciding whether to resume. The timeout is not a
scientific result or evidence against either meta variant.
The stage-3 pre-score freeze must point `--stage2-root` to this corrected
root if phase 2 completes. This correction changes only the execution
environment and output path, not data roles, model configuration or budget.

# v2 source mechanism work: execution record

Date: 2026-10-09. Branch: `codex/meta-audio-tta-v2-mechanism`; base: `98df7f255e956f98a33da9dbf575edfbeb7e9303`. Environment: `conda run -n tta`; no target score/evaluator process was run in this worktree.

## Resource/access check

| Command | Exit | Observation |
|---|---:|---|
| Initial `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits` | nonzero | Initial NVML/driver query failed; retained as the first observed resource check. |
| Recheck `nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits` | 0 | Host reported GPU 0: 48,666 MiB free / 0%; GPU 1: 48,666 MiB / 0%; GPU 2: 47,910 MiB / 28%; GPU 3: 48,400 MiB / 0%. |
| `conda run -n tta python -c 'import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())'` | 0 | `torch 2.1.0+cu121`, CUDA unavailable, device count 0. |
| `ls -l /dev/nvidia*` | 2 | No CUDA device nodes were exposed to the `tta` process. |
| `free -h` | 0 | 78 GiB available RAM at inspection. No GPU work launched. |
| `df -h .` | 0 | About 601 GiB free. |

The host resource query could observe the four cards, but CUDA was not passed through to the Python execution environment (`torch.cuda.is_available()==False`, no `/dev/nvidia*`). All newly executed model diagnostics therefore ran on CPU, did not compete for GPU resources, and do not provide new GPU timing/peak-memory measurements. GPU 2 was not selected because it showed existing 28% use. Four checkpoint files and source role manifests were found and read from the v1 worktree/source paths; no checkpoint was copied or modified. The fixed source diagnostic selected source_val IDs already recorded in the v1 diagnostic outputs, plus the v1 fit/FIR IDs. No assignment was regenerated.

## Real SSL-AASIST gradient / oracle diagnostic

Successful frozen configuration/command (64 per-sample rows, one CPU process, 8 threads):

```bash
conda run --no-capture-output -n tta python experiments/meta_audio_tta/mechanism_gradient_diagnostic.py \
  --source-config experiments/meta_audio_tta/config_stage1_20261008.json \
  --stage1-root /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/stage1_20261008 \
  --stage2-root /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/stage2_20261008_gpu \
  --prior-run /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/source_gpu_diagnostics_20261008/run_fixed_repair1 \
  --prior-ce /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce \
  --output experiments/meta_audio_tta/runs/mechanism_cpu_20261009_normmatched_retry1 \
  --threads 8
```

Observed exit **0**, terminal JSON `{"status":"PASS","records":64,"device":"cpu"}`. The complete `sys.argv`, sample IDs/labels/conditions/difficulty tags, gradient dot/cosine/norms, update norm, CE loss and margin deltas are stored in `runs/mechanism_cpu_20261009_normmatched_retry1/run_config.json` and `per_sample.jsonl`. The analysis script below reports its statistical summaries. Exact host stdout was returned by the Codex command runner; it was not redirected to an additional `.log` during execution.

The earlier correct-path diagnostic `--output experiments/meta_audio_tta/runs/mechanism_cpu_20261009_retry1` also exited **0**, 64 records, before norm-matching the oracle. It is preserved as a non-primary initial run. The norm-matched repeat is the primary result. A first attempt using a nonexistent `.../source_gpu_diagnostics.../run_fixed_repair1/ce` path exited **1 before model loading**. A later invocation with a malformed prior-run path exited **1 before model loading** and left only its new empty run directory; a subsequent attempt to reuse that directory correctly refused to overwrite it (exit **1**). These engineering failures are not scientific experiment failures. No result directory was overwritten.

## Matched epoch inference probe

```bash
conda run --no-capture-output -n tta python experiments/meta_audio_tta/matched_epoch_probe.py \
  --source-config experiments/meta_audio_tta/config_stage1_20261008.json \
  --checkpoint-root /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/stage1_20261008 \
  --selected-scores /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce/source_val.scores.jsonl \
  --output experiments/meta_audio_tta/runs/matched_epoch_cpu_20261009 \
  --threads 8
```

It exited **0**, emitted 80 per-item rows, and evaluated CE/Joint checkpoints at epochs 1, 2, 4, 6, and 8 on the same 8 source_val IDs. See `runs/matched_epoch_cpu_20261009/summary.json` and `scores.jsonl`. Exact eight IDs are recorded there. All five matched epochs in both models produce 8/8 correct, EER 0 and AUC 1. This confirms the chosen 8-item margin probe remains saturated; it does not distinguish training methods. Full source_val history EER/loss is read from v1 `history.jsonl` and shown in the main report.

See `runs/matched_epoch_cpu_20261009/summary.json` and `scores.jsonl`. Exact eight IDs are recorded there. All five matched epochs in both models produce 8/8 correct, EER 0 and AUC 1. This confirms the chosen 8-item margin probe remains saturated; it does not distinguish training methods. Full source_val history EER/loss is read from v1 `history.jsonl` and shown in the main report.

## Source result aggregation and figure

```bash
conda run -n tta python experiments/meta_audio_tta/analyze_mechanism_results.py
```

Observed successful exit **0**, JSON `status=PASS`. Complete class/difficulty/acoustic-stratified results: `runs/mechanism_cpu_20261009_normmatched_retry1/analysis_complete.json`; vector figure: `runs/mechanism_cpu_20261009_normmatched_retry1/figures_complete/gradient_alignment_and_loss.svg`. The earlier `analysis_normmatched.json`/`figures_normmatched` remain as the unstratified intermediate record.

Earlier invocations are retained in the output tree. First plotting attempt exited **1** because `matplotlib` could not import missing `pyparsing`; no dependency was installed. A standard-library SVG renderer replaced it. Two intermediate summary attempts exited **1** on a class-label vector shape mismatch and a CE-only null BYOL field; each error was repaired in code. Both the first non-norm-matched summary (`runs/mechanism_cpu_20261009_retry1/analysis_retry1.json`, SVG in `figures_retry1/`) and the intermediate norm-matched summary (`runs/mechanism_cpu_20261009_normmatched_retry1/analysis.json`, SVG in `figures/`) completed; the final complete norm-matched report uses `analysis_normmatched.json` and `figures_normmatched/` so all run records remain separate.

## Source protocol and negative-result retention

The main report reuses, without rewriting, v1 run records:

- `runs/stage1_20261008/{ce,joint}/history.jsonl` and `selection.json`;
- `runs/stage2_20261008_gpu/{cross,same}/history.jsonl` and `selection.json`;
- `runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce/`;
- `runs/source_gpu_diagnostics_20261008/run_fixed_repair1/{joint,cross,same}/`;
- `runs/stage3_20261009_freeze/STAGE3_FINAL_REPORT.md` and the preserved access incident record.

Those v1 files remain read-only references. This work did not change them or calculate any new target metrics. No target90/final holdout file, manifest, or evaluator was opened.

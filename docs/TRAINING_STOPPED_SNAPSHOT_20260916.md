# AASIST / SSL-AASIST stopped-training snapshot

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-16T18:16:39+08:00
- Verification Status: VERIFIED_STOPPED_ARTIFACTS
- Version Label: source_training_stopped_snapshot_v1

Both full-training processes are stopped. These runs are incomplete and must not be
reported as `TRAINED`, `FINALIZED`, or frozen models. Existing epoch-boundary
checkpoints and logs are retained in place. No task checkpoint was replaced by an
author-provided anti-spoofing weight.

## AASIST

- Run directory: `/media/dell/data/fakedata/eptta_work/training_runs/aasist_source/full-gpu2-b48-001`
- Completed epochs: 14 (`0..13`); last global step: 7406
- Last source-val EER/loss: `0.004428341384863144` / `0.04379110849910407`
- Best epoch/source-val EER: `12` / `0.0029154518950437317`
- `source_train_job.json`: `bb0cd2be40195a09271d805a3c5ef2c32d55cf23f75534de3ac0c44a41156b98`
- `train_log.jsonl`: `7bb43c7e7a00b244631d39e64fc0590d87f40c602688bfe150d03079323d2a77`
- `metrics.jsonl`: `1e11c767cbcc0fbea35cd62ddc9023104f41e75b962db74c0bf15f7ea7d4c881`
- `checkpoints/last.pt`: `d712df099c19e3d0d191065eb7cdbfb84ddae45cd17e84725e8334c7e39c3b19`
- `checkpoints/last.pt.json`: `f17869f7aa87b75e22089c89727ffe732ca4ee97e0a4c54a0dd7afb1ff144583`
- `checkpoints/best.pt`: `89cf9c65b3274c62e8e2d9b0bff0ab898cbaeabc2578dc80d6eb6e84273da295`
- `checkpoints/best.pt.json`: `605a623d1f3f43eb815d94d56a5038a1b145ae9a930aec979b0c37cd58cd67e6`

## SSL-AASIST

- Run directory: `/media/dell/data/fakedata/eptta_work/training_runs/ssl_aasist_source/full-gpu1-b48-001`
- Completed epochs: 91 (`0..90`); last global step: 48139
- Last source-val EER/loss: `0.0` / `2.9905445716983296e-05`
- Best epoch/source-val EER: `5` / `0.0`
- `source_train_job.json`: `df0b0968bb621f16f281b58eaca1e9ce49e1b5965de7266747c36cb5cec1a305`
- `train_log.jsonl`: `960619e09499f95fb8af0a450086fb05a61fcdec709bcaf2930af917442fd2ab`
- `metrics.jsonl`: `48e97d622b43517192b619fe0ca4405d1bab254c6dfde7d7d768c5ffac3cf836`
- `checkpoints/last.pt`: `efb3c259f05a24fd7992c771328aa27ef12c4f5a8f7cc084ec52947dd9f7cdde`
- `checkpoints/last.pt.json`: `76f3ef6db1e4ecc702d1369ff9ad8290e490646d112c422883a4703a4a63cca6`
- `checkpoints/best.pt`: `5dea8eb1a53bc719c380c0d189718921d0a60976b7485983f9c92eef30081d08`
- `checkpoints/best.pt.json`: `bb12da36b95a7cd6a29d1a43868590021b7d8d8b833d138e69264edac0163d09`

The source-val values above are checkpoint-selection diagnostics, not independent
control/test performance. Neither stopped run is eligible for `finalize-training`
or `export-frozen` unless exact training is resumed and reaches its locked end.

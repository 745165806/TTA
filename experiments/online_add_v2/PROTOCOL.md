# Online ADD Baselines v2 — locked four-domain protocol

Status: protocol locked before formal target scoring. The formal run configuration is [`config.json`](config.json). This is a development benchmark; the selected LA21/DF21 records come from the official 2021 eval release and are **not** represented as untouched final holdouts.

## Data and order

| Domain | Fixed count | Selection | Runtime input |
|---|---:|---|---|
| ITW | 3,178 | Existing target10 selected assignment | Existing three-view LL cache |
| WaveFake | 4,096 | Existing 2,048-content-pair development selection | Existing three-view LL cache |
| LA21 | 4,096 | 2,048 bonafide + 2,048 spoof from official CM key, seed 2026 | New selected-only LL cache |
| DF21 | 4,096 | Same selection rule on distinct DF official CM key | New selected-only LL cache |

`build_asv21_subsets.py` is the only code that reads the official LA/DF label keys. It writes a label-free runner manifest and a separate evaluator-only label sidecar. ITW and WaveFake evaluator-only labels are prepared by `build_eval_labels.py`; WaveFake labels come from the explicit `real_or_fake` parquet column, with the existing native mapping `R=0`, `WF1…WF7=1`. Runner modules do not import these label artifacts.

`make_streams.py` writes 30 immutable manifests under `streams_locked/`. S1–S4 are stationary ITW, WaveFake, LA21, DF21. S5 is ITW→LA21→WaveFake→DF21. S6 reverses that domain order. The order seeds are 2026–2030. Each stream segment shuffles sorted opaque IDs with a separate `random.Random(seed*1000 + stream_index*10 + segment_index)`. Labels, scores, attacks, paths, and group IDs do not determine order. The old `streams/` directory was a preformal draft whose equal-length domains shared a permutation; it is not used by formal runs.

## Predictor and online state

All streams start from the project-trained SSL-AASIST epoch 7 Frozen checkpoint. The generic XLS-R initialization is recorded separately in the bundle. Native logits are `[spoof, bonafide]`; a larger score means spoof, calculated as `logit[0]−logit[1]`. The frozen XLS-R+LL frontend is replayed from its actual `[201,128]` pre-backend tensor; the original AASIST backend runs on CUDA. The fixed audio probe creates clean, noise and FIR views before frozen LL extraction. Batch size is 16.

For each arrived batch, `run.py` saves its first online score rows and flushes them before calling `adapt` on that batch. A method may use only the arrived batch and its own prior state. Future audio/features and evaluator labels are unavailable to the method. A new method object, optimizer, teacher, memory, BN buffers and Fisher reference is created for every independent stream. Domain changes do not reset state. LAME is the documented exception: it refines the current batch output from the current batch's penultimate features and probabilities without changing parameters or retaining memory.

| Method | Adaptation mechanism | Source-only input | State |
|---|---|---|---|
| Frozen | None, original BN and weights | Checkpoint | None |
| BN-only | Current B16 BN statistics, fixed affine | Checkpoint | None |
| TENT-Audio | Current BN statistics and entropy-gradient BN affine update | Source-select LR | BN affine + SGD |
| EATA-Audio | Reliable entropy filter, cosine redundancy filter, source-fit Fisher penalty | Source-select LR/diversity; fit Fisher | BN affine + SGD + probability EMA |
| RoTTA-Audio | Robust BN, class-balanced uncertainty/timeliness memory, student/EMA teacher, temporal cross-entropy | Source-select LR | 64 audio records + teacher + Adam |
| LAME-Audio | Symmetric current-batch kNN graph and Laplacian probability refinement | None | None |

RoTTA uses the author's 64-record memory cap and 64-arrival update frequency, below the v2 maximum of 256. Its stored records contain the three LL views of each audio; noise and FIR were applied to waveform before SSL feature extraction. No visual-image augmentation is applied directly to LL features. EATA's 2-class entropy threshold is `0.4*log(2)`. EATA's Fisher is constructed from 2,000 source-fit records, and its cosine threshold is the 95th percentile of source-select clean probability similarity. These are source-only choices and are recorded under `results/<run_id>/source_only/`.

Author mechanism references: [TENT](https://github.com/DequanWang/tent/blob/master/tent.py), [EATA](https://github.com/mr-eggplant/EATA/blob/main/eata.py), [RoTTA](https://github.com/BIT-DA/RoTTA/blob/main/core/adapter/rotta.py), [LAME](https://github.com/fiveai/LAME/blob/master/src/adaptation/lame.py). This project ports the relevant mechanisms to its task-trained backend; it does not claim bitwise equivalence to the original visual benchmark code.

## Selection, evaluation and exclusions

TENT/EATA/RoTTA LR candidates are author LR divided by 3, author LR, and author LR multiplied by 3. Only the fixed source-select clean/noise/FIR pseudo-target streams choose LR: lowest mean EER, then highest AUC, then author default. The shared operating threshold `−4.770049095153809` is the existing source cal0 Frozen threshold, used unchanged for all spoof-oriented scores. No target-label hyperparameter tuning occurs.

The independent evaluator joins scores to labels only after each run's first online score file is durable and marked COMPLETE. Stationary metrics include AUC, EER, source-threshold FPR/FNR/balanced accuracy, and paired deltas to Frozen. Dynamic metrics are per domain segment, segment macro, first 256 after each switch, and within-segment rolling windows (256/128). A single-class window is NA. Five orders report mean, sample std, min and max. Stationary paired bootstrap uses 1,000 conditional resamples of realized trajectories, grouped by content pair for WaveFake and by sample ID elsewhere. This bootstrap does not rerun adaptation.

`adapted_samples` counts records selected for a post-prediction optimizer step. BN-only and LAME change current-batch inference without an optimizer step, so their adapted fraction is zero under this counter. `memory_size` counts retained target records; trainable parameter names, update counts, backend CUDA time and peak GPU allocation are recorded per run.

The backend time counter uses synchronized elapsed time around CUDA prediction and update calls. It excludes frozen frontend extraction, cache reads, score-file writes and explicit score persistence. It includes host launch overhead, so it is a backend wall-time proxy rather than a CUDA-event kernel-only measurement.

`target90` and the separately designated final holdout are excluded from every builder, cache, stream, method and evaluator entry point. The remaining LA/DF release records are also excluded from this v2 subset.

## Reproduction and status

Run all commands in conda environment `tta`, with `PYTHONPATH=src:.`, `OMP_NUM_THREADS=2`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The exact selected IDs, order manifests, configuration, source-only selection evidence, per-run command, first scores, run status and logs are retained. The launcher executes all order2026 tasks before any later order and never replaces an existing run attempt.

Bounded engineering checks are 32-item native/LL Frozen parity (logit max error ≤1e-5), a 64-item TENT prefix under two futures (exact same first scores), 64-item CUDA smoke for every method, and compileall. Results and any failed attempts stay in `results/<run_id>/validation/` and `results/<run_id>/smoke/`. Formal completion is based only on actual COMPLETE run statuses and exact score coverage; an unfinished report says INCOMPLETE.

# Audio-Native Standard TTA Contract

## Scientific identity

This study is the `audio_native_standard_tta` comparison track. Its method IDs
are `tent_audio_native_v1`, `sar_audio_native_v1`, and
`memo_audio_native_v1`; they are audio-native mappings, not claims of exact
TENT, SAR, or MEMO reproduction.

Historical outputs and implementations remain immutable and retain the names
`TENT-Episodic-Ref`, `SAR-Episodic-Ref`, and `MEMO-Audio-Ref`. They answer what
happens under a direct/reference-style image-TTA transfer. This track asks what
happens when the algorithmic core is preserved but mapped to SSL-AASIST's
audio and single-sample episodic semantics.

## Invariants

TENT keeps label-free test-time gradient adaptation, prediction-entropy
minimization, and a limited modulation-parameter update. SAR keeps label-free
test-time adaptation, its first and perturbed second reliable-entropy filters,
and parameter-bound SAM. MEMO keeps one test item, multiple label-preserving
audio views, a test-time update, and the entropy of the average softmax
probability. `mean(entropy(view_i))` is prohibited.

Every sample begins from an exact parameter-and-buffer snapshot. Adaptation is
never continual and is never reduced across samples or devices. Target workers
consume only the strict label-free waveform manifest. Labels are available only
to the post-hoc target10 development aggregator.

## Pre-registered audio mapping

The main scope `backend_norm_affine_v1` contains affine weight/bias parameters
of task-trained AASIST BatchNorm, LayerNorm, and GroupNorm modules found by the
model audit. Generic XLS-R parameters and the classification head are excluded.
All other parameters are frozen.

The model stays in normal `eval()` inference semantics. BatchNorm uses intact
source running mean and variance; only selected gamma/beta can update.
LayerNorm and GroupNorm, if present, retain native behavior. Running buffers
must be bitwise unchanged after each update.

TENT-Audio v1 uses Adam, lr 1e-3, one step, no weight decay, and mean prediction
entropy. SAR-Audio v1 uses exactly the same parameter/normalization mapping,
SGD momentum 0.9, the pre-existing audited batch-size-one audio lr 1.5625e-5,
SAM rho 0.05, one step, and margin `0.4*log(2)`.

MEMO-Audio v1 uses the same conservative backend scope, SGD lr 2.5e-4, one
iteration, and no weight decay. Its available views are original plus only the
candidates accepted by the fixed source-select audit in `config.json`. Target10
labels cannot alter candidates, thresholds, or acceptance.

For Task2.1 this method is reported as **MEMO-Audio-Limited**. The decomposition
adds `memo_audio_full_safeaug_v1` (**MEMO-FullSafeAug**), which uses the identical
source-audited original+FIR views, optimizer, objective, and source-BN `eval()`
semantics, but restores MEMO's full SSL-AASIST parameter scope. Thus
MEMO-Audio-Ref → FullSafeAug changes augmentation only, while FullSafeAug →
Limited changes parameter scope only. Every FullSafeAug episode restores all
parameters and buffers exactly.

## Frozen baselines and scientific gate

`Frozen-Cache` is retained only for continuity with P0/P1/P2 and numerical-path
diagnosis. Each method's own waveform `score_before_update` is its primary
`Frozen-Waveform` paired baseline. Adaptation gain/harm is never inferred from
cache-to-waveform differences. Before-update scores across methods must agree
within absolute tolerance 1e-5; failure raises
`AUDIO_NATIVE_BEFORE_PATH_PARITY_FAIL` and blocks scientific conclusions.

All main and ablation methods receive seed-2026, 2000-resample paired bootstrap
comparisons against their waveform baseline. Gain and recovery are distinct:
gain compares with Frozen-Waveform, while recovery compares an audio-native
mapping with its historical reference port.

## Mechanism ablation

One structural ablation is pre-registered:
`backend_norm_plus_graph_modulation_v1`. It adds only AASIST backend standalone
graph-attention modulation tensors (`att_weight*`, `pos_S`, `master1`,
`master2`) to the main normalization scope. It excludes XLS-R, Linear/Conv
weights, and the head. It is a mechanism diagnostic, not a best-method search.

## Interpretation boundary

Flags and conclusions are limited to the trained SSL-AASIST detector, target10
development split, episodic protocol, and mappings tested here. No result may
be generalized as “all standard TTA fails for audio.” Multi-domain confirmation
is separate and is not implied by this study.

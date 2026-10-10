# Stage 0 validation and resource gate (2026-10-08)

## Evidence and limits

The source assignment was reused unchanged: `fit=25,380`,
`source_val=5,654`, `select=11,723`, `cal0=7,467`; role IDs were disjoint.
The read-only audit command
`conda run -n tta python experiments/meta_audio_tta/audit_roles.py --config experiments/meta_audio_tta/config_stage1_20261008.json --output experiments/meta_audio_tta/results/source_role_audit_20261008.json`
exited 0 and saved the class counts and exact role paths.
The fit manifest and audio root were read; no target manifest was opened.
The historical project-trained epoch-7 checkpoint was used **only** to
exercise real SSL-AASIST code and size resources. It is not a stage-1 or
stage-2 training initialization.

Real-model smoke on the committed fixed-seed code:
`results/stage0_smoke_20261008g.json`, exit 0 on GPU 2. The earlier
`stage0_smoke_20261008f.json` is retained as the pre-fixed-seed run. The active
backbone scope is 30 BN affine tensors / 1,538 scalars; ten inactive author
BN tensors were excluded. The shared backbone receives a nonzero auxiliary
gradient and K=1 changes the spoof score. K=0 direct-path parity,
A→B→A reset, target EMA and BN buffer stability, parameter restoration,
CPU/CUDA RNG stability, Cross- and Same-sample meta gradients all passed.
The final real second-order gradient differs from the detached first-order
gradient by L1 `0.00146351`. This tests the Hessian contribution directly.

The bounded commands executed (all exit 0):

```bash
conda run -n tta python -m pytest tests/unit/test_meta_audio_tta.py -q
conda run -n tta python -m py_compile experiments/meta_audio_tta/core.py experiments/meta_audio_tta/smoke_real.py experiments/meta_audio_tta/measure_resources.py experiments/meta_audio_tta/stage1_source.py tests/unit/test_meta_audio_tta.py
conda run -n tta python experiments/meta_audio_tta/smoke_real.py --asset-root /media/dell/data/fakeAudioDection/TTA --fit-manifest /media/dell/data/fakeAudioDection/TTA/data/manifests_v2/asv2019_la/manifests/fit.jsonl --audio-root /media/dell/data/fakedata/asvspoof2019/LA --output experiments/meta_audio_tta/results/stage0_smoke_20261008f.json --device cuda:0
conda run -n tta python experiments/meta_audio_tta/smoke_real.py --asset-root /media/dell/data/fakeAudioDection/TTA --fit-manifest /media/dell/data/fakeAudioDection/TTA/data/manifests_v2/asv2019_la/manifests/fit.jsonl --audio-root /media/dell/data/fakedata/asvspoof2019/LA --output experiments/meta_audio_tta/results/stage0_smoke_20261008g.json --device cuda:2
conda run -n tta python experiments/meta_audio_tta/measure_resources.py --asset-root /media/dell/data/fakeAudioDection/TTA --fit-manifest /media/dell/data/fakeAudioDection/TTA/data/manifests_v2/asv2019_la/manifests/fit.jsonl --audio-root /media/dell/data/fakedata/asvspoof2019/LA --mode ce --batch-size 14 --output experiments/meta_audio_tta/results/resource_ce_b14_20261008a.json --device cuda:0
conda run -n tta python experiments/meta_audio_tta/measure_resources.py --asset-root /media/dell/data/fakeAudioDection/TTA --fit-manifest /media/dell/data/fakeAudioDection/TTA/data/manifests_v2/asv2019_la/manifests/fit.jsonl --audio-root /media/dell/data/fakedata/asvspoof2019/LA --mode joint --batch-size 14 --output experiments/meta_audio_tta/results/resource_joint_b14_20261008a.json --device cuda:0
```

The two resource JSON files show one optimizer step at batch 14 original
audios × 2 views: CE 17.80 GB allocated / 20.16 GB reserved / 1.652 s;
joint 21.27 GB allocated / 22.07 GB reserved / 1.761 s. Four RTX A6000
cards each have 49,140 MiB; devices 0 and 1 had 48,666 MiB free. Storage
had 678 GB free. A prior task-training epoch checkpoint is 3.6 GB; joint
epoch files are **estimated** at about 5 GB because of the second detector
and optimizer. These are measurements/estimates, not throughput guarantees.

## Fixed stage-1 budget and control

`config_stage1_20261008.json` fixes seed 13, eight epochs, 25,380 fit
records/epoch, batch 14, 1,813 optimizer steps/epoch, Adam `1e-6`, weight
decay `1e-4`, original+deterministic-FIR views, weighted CE, BYOL weight
`0.1`, EMA `0.996`. CE and joint receive exactly the same initial model
state, fit sampler order, two views and backbone optimizer-step count.
`source_val` alone selects epoch by EER; ties select the earliest epoch.
The CE-only selected checkpoint supplies a Frozen comparison. Stage-1
expected wall time is at least 7–8 hours per run including validation and
audio I/O; a two-GPU concurrent run is expected to fit in 49 GB/device.
Eight CE epoch files are estimated at 29 GB and eight joint files about
40–50 GB. The run paths below are exclusive and immutable.

```bash
conda run -n tta python experiments/meta_audio_tta/stage1_source.py --config experiments/meta_audio_tta/config_stage1_20261008.json --mode prepare --initial experiments/meta_audio_tta/runs/stage1_20261008/initial.pt --output experiments/meta_audio_tta/runs/stage1_20261008/unused --device cuda:0
conda run -n tta python experiments/meta_audio_tta/stage1_source.py --config experiments/meta_audio_tta/config_stage1_20261008.json --mode ce --initial experiments/meta_audio_tta/runs/stage1_20261008/initial.pt --output experiments/meta_audio_tta/runs/stage1_20261008/ce --device cuda:0
conda run -n tta python experiments/meta_audio_tta/stage1_source.py --config experiments/meta_audio_tta/config_stage1_20261008.json --mode joint --initial experiments/meta_audio_tta/runs/stage1_20261008/initial.pt --output experiments/meta_audio_tta/runs/stage1_20261008/joint --device cuda:1
```

Stage 2 uses the valid selected joint checkpoint only. Its Cross- and
Same-sample variants will share task order, initialization, BN scope,
auxiliary weight and outer optimizer-step count; stage-2 configuration and
measured task runtime must be fixed before it runs. Stage 3 remains
`NOT_RUN` until both stage-1 and stage-2 checkpoints pass their gates.

## Gate

Stage 0 engineering checks: **PASS**. Resource gate for stage 1: **PASS**.
Stage 1, stage 2 and stage 3 scientific outcomes: **NOT_RUN** at this report.
Stop on any source-role, state, metric or resource failure and retain the
failed command/log. A lack of gain later stops expansion of that version;
it does not refute meta-adaptation outside the tested conditions.

# WaveFake and DFADD embedded audio

The local WaveFake and DFADD releases embed audio in Parquet and Arrow files.
`python -m eptta.data.container_import --config CONFIG.json` converts selected
records into a **new** ordinary-audio directory plus `input.csv`; the normal
`prepare-data` command then writes separate label, group, and label-free
inference views. Neither dataset enters source `fit` or `source_val`.

`pyarrow==21.0.0` is installed in the single `tta` environment and recorded in
`environment.yml`. The import requires explicit source file patterns and
expected counts, raw label mapping, group assignment, input/output sample
rates, and output path. It rejects missing audio, wrong sample rates, unknown
labels, group-map gaps, split mismatches, and existing output directories.

## Observed local contracts

| Release | Local observed records | Labels | Group | Preprocess |
|---|---:|---|---|---|
| WaveFake local Parquet | 131 files; 104,800 rows from Parquet metadata | `R`→0, `WF1`…`WF7`→1 | 13,100 `audio_id` groups, each with exactly one of all eight variants | 22,050 Hz WAV → 16,000 Hz mono float WAV using `scipy.signal.resample_poly` with ratio 320/441 |
| DFADD local Arrow, official `test` | 2 shards; 3,755 rows | `real`→0, `spoofed`→1 | explicit `audio_name`→VCTK utterance map | 16,000 Hz original bytes |

The WaveFake local README's 64,800 example count disagrees with the actual 131
Parquet footers and the hosted dataset viewer, both of which show 104,800.
For DFADD `test`, all 3,755 observed names and labels matched the reviewed
`speaker_utterance[_generator].extension` grammar. The saved local mapping is
`configs/embedded/dfadd_test_group_map_20260927.csv`: 3,755 unique
audio names and 768 source-utterance groups. It is deliberately separate from
the inference manifest. The map groups apparent VCTK utterance variants;
cross-dataset source overlap has not been established, so these data do not
support a claim of independent source speakers.

## Verified small imports

Run from the repository root in `tta`:

```bash
python -m eptta.data.container_import --config configs/embedded/wavefake_local_smoke_20260927.json
python -m eptta.cli prepare-data --config configs/embedded/wavefake_prepare_smoke_20260927.json
python -m eptta.data.container_import --config configs/embedded/dfadd_test_local_smoke_20260927.json
python -m eptta.cli prepare-data --config configs/embedded/dfadd_prepare_smoke_20260927.json
```

These commands already exited 0 once and refuse to overwrite their output.
The smoke imports contain one complete WaveFake group (8 samples: 1 real,
7 fake) and one complete DFADD test group (6 samples: 1 real, 5 fake). Their
`target_test` inference files contain only ID, index, root key, relative audio
path, and role. This verifies data plumbing, not model extraction, TTA, or
detection quality.

Full import and prepare configs are provided as
`configs/embedded/{wavefake,dfadd_test}_{local,prepare}_full_20260927.json`.
They use new output directories, the reviewed contracts, and exact expected
counts. A full import is a large audio materialization and has **not** been
run locally. The existing WaveFake `generated_audio.log` path does not apply
to this Parquet release.

The original DFADD `train` and `valid` splits are available in Arrow form,
but they have not been imported or assigned to project roles. Their split
names do not authorize use as source training data.

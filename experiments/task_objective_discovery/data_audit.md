# Task-objective development coverage audit

| Domain | Explicit resource | Development coverage | Current state |
|---|---|---|---|
| In-the-Wild | `/media/dell/data/fakedata/release_in_the_wild` | Fixed target10-derived mechanism 512; canonical 0/1 selected audit | Two-class development available; existing cache |
| Codecfake | `/media/dell/data/fakedata/Codecfake_Xie/extracted/dev`; official dev protocol 92,596 rows | Fixed selected 512; `filename raw_label attack_id`, `real=0`, `fake=1` from existing explicit mapping | 32 WAV production cache smoke PASS; full 512 blocked: 222 at 16 kHz, 209 at 24 kHz, 52 at 44.1 kHz, 29 at 48 kHz |
| WaveFake | `/media/dell/data/fakedata/WaveFake/data` | 131 Parquet partitions; README train count 64,800 is unverified locally; `real_or_fake` string mapping unverified | `pyarrow`, `pandas`, `polars`, `duckdb` all missing in `tta`; no compatible cache; dependency blocker |
| ASVspoof2021 LA | `/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_LA_eval` | Historical mechanism 282, one class | `TWO_CLASS_DEV_UNAVAILABLE`; no explicit 2021 train/dev found |
| ASVspoof2021 DF | `/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_DF_eval` | Historical mechanism 370, one class | `TWO_CLASS_DEV_UNAVAILABLE`; no explicit 2021 train/dev found |

ASVspoof2019 LA train/dev exist locally, but they are a different release and cannot be renamed ASVspoof2021 development. No ASV mechanism assignments were changed. Prior scanning of full ASV eval label files is recorded in the boundary correction; no remaining eval pool is claimed as untouched final holdout. No new target90 or final-holdout label access occurred in this audit.

The production `load_audio` contract requires exactly 16 kHz and never resamples. The first 32 Codecfake selections happen to satisfy it; the 512-set failure is preserved at `results/codecfake_cache_512_20260927a/failure.json`. Its fixed selected IDs were not redrawn and no new preprocessing path was silently introduced. A full Codecfake objective comparison is therefore **NOT_RUN**.

An **auxiliary, separately named** development source exists: `/media/dell/data/fakedata/asvspoof2019/PA/ASVspoof2019_PA_dev`, with explicit official dev protocol `ASVspoof2019.PA.cm.dev.trl.txt`. Its format is `speaker sample_id attack_id environment bonafide|spoof`. The audio tree lacks some protocol files, so only one complete 270-record speaker group (`PA_0105`) satisfied the strict group and 16-kHz checks. The selected 270 records were fixed once from sorted explicit speaker groups with seed 2026; no label field entered the selection. This is **ASVspoof2019 PA dev**, not a repair of ASVspoof2021 LA/DF coverage. Class coverage will be checked after all objective scores are produced.

Post-score audit of the fixed PA dev group found **270 bonafide, 0 spoof**. This result cannot support PA EER/AUC or the required second two-class development domain. The saved group remains fixed; it is not redrawn based on its observed labels.

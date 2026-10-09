# Local data audit — 2026-09-27

| Domain | Observed source | Split and count | Label format | Sandbox state |
|---|---|---|---|---|
| In-the-Wild | `/media/dell/data/fakedata/release_in_the_wild` | 31,779 WAV files; existing target10=3,178 | Existing target10 audit JSON, integer 0/1 | 512 selected from target10; cached |
| WaveFake | `/media/dell/data/fakedata/WaveFake/data` | 131 Parquet partitions; local README declares train=64,800, row count unverified | Parquet `real_or_fake` string; mapping unverified | 0; present but blocked by missing reader and feature cache |
| Codecfake-Xie | `/media/dell/data/fakedata/Codecfake_Xie/extracted` | Official dev protocol and WAV both 92,596 | `filename raw_label attack_id`; declared `real→0`, `fake→1` | 512 official dev samples; no feature cache |
| ASVspoof2021 LA | `/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_LA_eval` | Official eval manifest/cache 148,176 | Separate canonical JSONL label artifact | 282 group-disjoint eval samples; cached |
| ASVspoof2021 DF | `/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_DF_eval` | Official eval manifest/cache 533,928 | Separate canonical JSONL label artifact | 370 group-disjoint eval samples; cached |

The WaveFake and Codecfake audio roots are present, so neither is called MISSING. Their current **feature paths** are unavailable for a production-identical guarded EP comparison. ASV evaluation-pool labels were not used to form the mechanism assignment. No dataset, model, or dependency was downloaded.

The historical `split_meta.json` printed aggregate target90 class counts during orientation. No target90 per-sample manifest, label, score, or metric was opened for this branch; the aggregate counts are excluded from all decisions.

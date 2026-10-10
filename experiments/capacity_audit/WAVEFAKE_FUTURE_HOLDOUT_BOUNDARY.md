# WaveFake future holdout boundary — 2026-09-27

The user explicitly requested a full local WaveFake resource audit before development. Accordingly, `audit_wavefake.py` read the Parquet schema, `audio_id`, `real_or_fake` codes and embedded WAV headers for **all 104,800 local rows**, although supervised training/held-out CV used only the fixed 4,096-row development assignment. No separate WaveFake final-holdout assignment or final metric was created or evaluated in this stage.

Because the remaining local rows' label codes and audio headers were audited, **those rows must not later be described as an untouched WaveFake final holdout** under the project's strict byte-access convention. A future final evaluation requires an independently protected resource/assignment with a truthful prior-access account. No existing assignment is redrawn or relabelled by this statement.

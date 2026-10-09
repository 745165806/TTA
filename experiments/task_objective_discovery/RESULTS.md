# Task objective discovery, first fixed-budget study

O1/O2/O3, Frozen, and production `ep_no_keep` shared the fixed 512 In-the-Wild mechanism selection, frozen SSL-AASIST features and source resources, three safe views, and original-view scoring. O1/O2/O3 used 5 steps, `lr=0.03`, `rho=0.1`, fresh 8×8 `R=0` per sample, and projected SGD. All 2,560 score rows completed before the existing selected-only audit labels were opened. Exact coverage and finite-value checks passed. The initial run `results/objective_inwild_512_20260927a/` used uncommitted new code; the committed-code replay `results/objective_inwild_512_20260927b/` reproduced its task metrics and is the source for the table below.

| Arm | EER | AUC | ΔAUC vs Frozen | Balanced accuracy | Mean evidence damage | Helpful / harmful flips |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 0.099010 | 0.957841 | 0 | 0.569355 | 0 | 0 / 0 |
| Base `ep_no_keep` | 0.103960 | 0.958783 | +0.000942 | 0.567742 | 0.170890 | 0 / 1 |
| O1 | 0.103960 | 0.958799 | +0.000958 | 0.606452 | 0.159813 | 23 / 0 |
| O2 | 0.103960 | 0.958895 | +0.001054 | 0.567742 | 0.199754 | 0 / 1 |
| O3 | 0.103960 | 0.958879 | +0.001038 | 0.593548 | 0.218358 | 16 / 1 |

All four paired 1,000-replicate development bootstrap ΔAUC intervals include zero; for O1 the interval is `[−0.000415, +0.002891]`. O1 raised fixed-threshold balanced accuracy by correcting 23 decisions, yet EER worsened and AUC moved minimally. This is evidence of threshold-sensitive movement, not repeatable ranking correction. It does not justify preservation, a gate, continual accumulation, or method lock. Only one independent two-class domain has full scores.

The Codecfake production 32-waveform cache smoke passed exact coverage and finite `3×160` features; the two-domain 32-sample objective smoke completed without labels. Its fixed 512 selection contains 222 waveforms at 16 kHz, 209 at 24 kHz, 52 at 44.1 kHz, and 29 at 48 kHz. Production `load_audio` rejects non-16-kHz input, so the full feature cache failed and remains in `results/codecfake_cache_512_20260927a/failure.json`. No Codecfake labels were consulted and no selected IDs were changed. WaveFake reader and ASV2021 two-class dev blockers remain.

## Auxiliary PA dev continuation

`results/objective_two_domain_20260927a/` scored the same five arms for In-the-Wild 512 and separately named ASVspoof2019 PA official dev 270 under committed objective code. The PA fixed complete speaker group proved all bonafide only after score generation. It therefore supplies source-damage and threshold-flip evidence but no second EER/AUC domain. O1 PA damage was 0.041150 versus Base 0.193014, with 7 helpful and 3 harmful bonafide threshold flips; no ranking claim follows. Full table and limitations are in `TWO_DOMAIN_RESULT.md`. The PA group will not be redrawn.

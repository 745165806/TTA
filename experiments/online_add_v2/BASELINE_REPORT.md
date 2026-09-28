# BASELINE_REPORT — Online ADD Baselines v2

**Stage:** Online ADD Baselines v2 · **Status:** INCOMPLETE
**Completed runs:** 36 / 180
**Target domains:** ITW = 3,178; WaveFake = 4,096; LA21 = 4,096; DF21 = 4,096.
**Methods:** frozen = INCOMPLETE (6/30); bn_only = INCOMPLETE (6/30); tent = INCOMPLETE (6/30); eata = INCOMPLETE (6/30); rotta = INCOMPLETE (6/30); lame = INCOMPLETE (6/30).

## Stationary mean over five orders

| Method | ITW EER% / AUC | WaveFake EER% / AUC | LA21 EER% / AUC | DF21 EER% / AUC |
|---|---:|---:|---:|---:|
| frozen | NA | NA | NA | NA |
| bn_only | NA | NA | NA | NA |
| tent | NA | NA | NA | NA |
| eata | NA | NA | NA | NA |
| rotta | NA | NA | NA | NA |
| lame | NA | NA | NA | NA |

## Dynamic mean over five orders

| Method | S5 macro EER% / AUC | S6 macro EER% / AUC | Worst rolling EER% / AUC | Switch first256 EER% / AUC |
|---|---:|---:|---:|---:|
| frozen | NA | NA | NA | NA |
| bn_only | NA | NA | NA | NA |
| tent | NA | NA | NA | NA |
| eata | NA | NA | NA | NA |
| rotta | NA | NA | NA | NA |
| lame | NA | NA | NA | NA |

## Efficiency

| Method | Updates / 100 | Adapted fraction | Backend ms/audio | Peak GPU MB | Memory records |
|---|---:|---:|---:|---:|---:|
| frozen | 0.0000 | 0.0000 | 0.69 | 1213 | 0 |
| bn_only | 0.0000 | 0.0000 | 0.69 | 1213 | 0 |
| tent | 6.2528 | 1.0000 | 2.75 | 1213 | 0 |
| eata | 4.7418 | 0.5471 | 2.64 | 1213 | 0 |
| rotta | 1.5576 | 0.9950 | 2.82 | 2094 | 64 |
| lame | 0.0000 | 0.0000 | 0.76 | 1213 | 0 |

Adapted fraction counts samples selected for a post-prediction optimizer update. BN-only uses current-batch statistics and LAME refines current-batch output without such an update, so both show zero in that column. Memory records counts retained target examples; model/optimizer state is described in `PROTOCOL.md` and per-run status.

## Main observations

1. Formal results remain incomplete; no cross-method conclusion is claimed.
2. Frozen parity and future-blind prefix passed on the locked stream; see validation artifacts.
3. Source-only LR selection and Fisher are fixed before formal target scoring.

**Which existing mechanisms appear useful for audio ADD online TTA?** Pending complete comparison.

**Which mechanisms fail or drift?** Pending complete comparison.

**Can this benchmark support designing a new method?** PARTIAL

**Target90 accessed:** NO · **Final holdout accessed:** NO

## Protocol and limitations

All scores are first predictions on B16 predict-then-adapt streams; LAME refines its current batch output. The four domains are fixed development resources. LA21/DF21 are selected from official eval releases, so they are development subsets, not untouched final holdouts. The same source-cal0 operating threshold is applied to all spoof-oriented scores. Backend timing excludes frozen frontend extraction, cache I/O and score-file writes.

## Stationary paired effects

Positive ΔAUC and positive EER gain favor the method over Frozen. Each cell averages available arrival orders on the same fixed domain subset.

| Method | Domain | Orders | Mean ΔAUC | Mean EER gain pp | AUC gains / orders |
|---|---|---:|---:|---:|---:|
| bn_only | ITW | 1/5 | -0.0247 | -2.5136 | 0/1 |
| bn_only | WaveFake | 1/5 | -0.0133 | -1.5137 | 0/1 |
| bn_only | LA21 | 1/5 | -0.0214 | -1.9043 | 0/1 |
| bn_only | DF21 | 1/5 | -0.0249 | -3.0762 | 0/1 |
| tent | ITW | 1/5 | -0.0256 | -2.5885 | 0/1 |
| tent | WaveFake | 1/5 | -0.0294 | -2.9297 | 0/1 |
| tent | LA21 | 1/5 | -0.0208 | -2.2461 | 0/1 |
| tent | DF21 | 1/5 | -0.0235 | -2.8320 | 0/1 |
| eata | ITW | 1/5 | -0.0250 | -2.5015 | 0/1 |
| eata | WaveFake | 1/5 | -0.0133 | -1.4160 | 0/1 |
| eata | LA21 | 1/5 | -0.0215 | -2.0020 | 0/1 |
| eata | DF21 | 1/5 | -0.0251 | -3.0762 | 0/1 |
| rotta | ITW | 1/5 | -0.0546 | -6.4564 | 0/1 |
| rotta | WaveFake | 1/5 | -0.0251 | -1.3184 | 0/1 |
| rotta | LA21 | 1/5 | -0.0183 | -2.1484 | 0/1 |
| rotta | DF21 | 1/5 | -0.0269 | -4.1504 | 0/1 |
| lame | ITW | 1/5 | -0.0108 | -0.5914 | 0/1 |
| lame | WaveFake | 1/5 | -0.0691 | -8.0078 | 0/1 |
| lame | LA21 | 1/5 | -0.0027 | -0.2930 | 0/1 |
| lame | DF21 | 1/5 | -0.0039 | -0.7324 | 0/1 |

## Run inventory

| Method | Stream | Order | Status | EER% | AUC | ΔAUC vs Frozen | EER gain pp vs Frozen |
|---|---|---:|---|---:|---:|---:|---:|
| frozen | S1 | 2026 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 |
| bn_only | S1 | 2026 | COMPLETE | 12.37 | 0.9386 | -0.0247 | -2.5136 |
| tent | S1 | 2026 | COMPLETE | 12.45 | 0.9377 | -0.0256 | -2.5885 |
| eata | S1 | 2026 | COMPLETE | 12.36 | 0.9383 | -0.0250 | -2.5015 |
| rotta | S1 | 2026 | COMPLETE | 16.31 | 0.9087 | -0.0546 | -6.4564 |
| lame | S1 | 2026 | COMPLETE | 10.45 | 0.9525 | -0.0108 | -0.5914 |
| frozen | S2 | 2026 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 |
| bn_only | S2 | 2026 | COMPLETE | 17.29 | 0.9017 | -0.0133 | -1.5137 |
| tent | S2 | 2026 | COMPLETE | 18.70 | 0.8856 | -0.0294 | -2.9297 |
| eata | S2 | 2026 | COMPLETE | 17.19 | 0.9017 | -0.0133 | -1.4160 |
| rotta | S2 | 2026 | COMPLETE | 17.09 | 0.8899 | -0.0251 | -1.3184 |
| lame | S2 | 2026 | COMPLETE | 23.78 | 0.8459 | -0.0691 | -8.0078 |
| frozen | S3 | 2026 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 |
| bn_only | S3 | 2026 | COMPLETE | 8.45 | 0.9533 | -0.0214 | -1.9043 |
| tent | S3 | 2026 | COMPLETE | 8.79 | 0.9540 | -0.0208 | -2.2461 |
| eata | S3 | 2026 | COMPLETE | 8.54 | 0.9532 | -0.0215 | -2.0020 |
| rotta | S3 | 2026 | COMPLETE | 8.69 | 0.9564 | -0.0183 | -2.1484 |
| lame | S3 | 2026 | COMPLETE | 6.84 | 0.9721 | -0.0027 | -0.2930 |
| frozen | S4 | 2026 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 |
| bn_only | S4 | 2026 | COMPLETE | 6.84 | 0.9666 | -0.0249 | -3.0762 |
| tent | S4 | 2026 | COMPLETE | 6.59 | 0.9680 | -0.0235 | -2.8320 |
| eata | S4 | 2026 | COMPLETE | 6.84 | 0.9664 | -0.0251 | -3.0762 |
| rotta | S4 | 2026 | COMPLETE | 7.91 | 0.9646 | -0.0269 | -4.1504 |
| lame | S4 | 2026 | COMPLETE | 4.49 | 0.9876 | -0.0039 | -0.7324 |
| frozen | S5 | 2026 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 |
| bn_only | S5 | 2026 | COMPLETE | 12.22 | 0.9400 | 0.0018 | 1.8965 |
| tent | S5 | 2026 | COMPLETE | 12.90 | 0.9256 | -0.0126 | 1.2235 |
| eata | S5 | 2026 | COMPLETE | 12.16 | 0.9401 | 0.0019 | 1.9573 |
| rotta | S5 | 2026 | COMPLETE | 13.16 | 0.9440 | 0.0058 | 0.9564 |
| lame | S5 | 2026 | COMPLETE | 13.67 | 0.9153 | -0.0230 | 0.4490 |
| frozen | S6 | 2026 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 |
| bn_only | S6 | 2026 | COMPLETE | 12.25 | 0.9405 | 0.0022 | 1.8720 |
| tent | S6 | 2026 | COMPLETE | 12.04 | 0.9404 | 0.0021 | 2.0807 |
| eata | S6 | 2026 | COMPLETE | 12.24 | 0.9399 | 0.0017 | 1.8843 |
| rotta | S6 | 2026 | COMPLETE | 13.56 | 0.9289 | -0.0093 | 0.5587 |
| lame | S6 | 2026 | COMPLETE | 13.96 | 0.9152 | -0.0230 | 0.1611 |
| frozen | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S1 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S2 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S3 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S4 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S5 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| tent | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| eata | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| rotta | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| lame | S6 | 2027 | NOT_RUN | NA | NA | NA | NA |
| frozen | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S1 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S2 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S3 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S4 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S5 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| tent | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| eata | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| rotta | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| lame | S6 | 2028 | NOT_RUN | NA | NA | NA | NA |
| frozen | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S1 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S2 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S3 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S4 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S5 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| tent | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| eata | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| rotta | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| lame | S6 | 2029 | NOT_RUN | NA | NA | NA | NA |
| frozen | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S1 | 2030 | NOT_RUN | NA | NA | NA | NA |
| frozen | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S2 | 2030 | NOT_RUN | NA | NA | NA | NA |
| frozen | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S3 | 2030 | NOT_RUN | NA | NA | NA | NA |
| frozen | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S4 | 2030 | NOT_RUN | NA | NA | NA | NA |
| frozen | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S5 | 2030 | NOT_RUN | NA | NA | NA | NA |
| frozen | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |
| bn_only | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |
| tent | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |
| eata | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |
| rotta | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |
| lame | S6 | 2030 | NOT_RUN | NA | NA | NA | NA |

Dynamic pooled values in the inventory are supplementary; compare S5/S6 by segment macro and within-segment windows in `summary.json`.

## Order sensitivity

Mean ± sample std (ddof=1), then min–max. Five orders are arrival-order repeats, not independent target datasets.

| Method | Stream | Orders | EER% mean ± std [min, max] | AUC mean ± std [min, max] |
|---|---|---:|---:|---:|
| frozen | S1 | 1/5 | 9.86 ± NA [9.86, 9.86] | 0.9633 ± NA [0.9633, 0.9633] |
| frozen | S2 | 1/5 | 15.77 ± NA [15.77, 15.77] | 0.9150 ± NA [0.9150, 0.9150] |
| frozen | S3 | 1/5 | 6.54 ± NA [6.54, 6.54] | 0.9747 ± NA [0.9747, 0.9747] |
| frozen | S4 | 1/5 | 3.76 ± NA [3.76, 3.76] | 0.9915 ± NA [0.9915, 0.9915] |
| frozen | S5 | 1/5 | 8.98 ± NA [8.98, 8.98] | 0.9611 ± NA [0.9611, 0.9611] |
| frozen | S6 | 1/5 | 8.98 ± NA [8.98, 8.98] | 0.9611 ± NA [0.9611, 0.9611] |
| bn_only | S1 | 1/5 | 12.37 ± NA [12.37, 12.37] | 0.9386 ± NA [0.9386, 0.9386] |
| bn_only | S2 | 1/5 | 17.29 ± NA [17.29, 17.29] | 0.9017 ± NA [0.9017, 0.9017] |
| bn_only | S3 | 1/5 | 8.45 ± NA [8.45, 8.45] | 0.9533 ± NA [0.9533, 0.9533] |
| bn_only | S4 | 1/5 | 6.84 ± NA [6.84, 6.84] | 0.9666 ± NA [0.9666, 0.9666] |
| bn_only | S5 | 1/5 | 11.51 ± NA [11.51, 11.51] | 0.9402 ± NA [0.9402, 0.9402] |
| bn_only | S6 | 1/5 | 11.29 ± NA [11.29, 11.29] | 0.9407 ± NA [0.9407, 0.9407] |
| tent | S1 | 1/5 | 12.45 ± NA [12.45, 12.45] | 0.9377 ± NA [0.9377, 0.9377] |
| tent | S2 | 1/5 | 18.70 ± NA [18.70, 18.70] | 0.8856 ± NA [0.8856, 0.8856] |
| tent | S3 | 1/5 | 8.79 ± NA [8.79, 8.79] | 0.9540 ± NA [0.9540, 0.9540] |
| tent | S4 | 1/5 | 6.59 ± NA [6.59, 6.59] | 0.9680 ± NA [0.9680, 0.9680] |
| tent | S5 | 1/5 | 11.39 ± NA [11.39, 11.39] | 0.9408 ± NA [0.9408, 0.9408] |
| tent | S6 | 1/5 | 11.25 ± NA [11.25, 11.25] | 0.9426 ± NA [0.9426, 0.9426] |
| eata | S1 | 1/5 | 12.36 ± NA [12.36, 12.36] | 0.9383 ± NA [0.9383, 0.9383] |
| eata | S2 | 1/5 | 17.19 ± NA [17.19, 17.19] | 0.9017 ± NA [0.9017, 0.9017] |
| eata | S3 | 1/5 | 8.54 ± NA [8.54, 8.54] | 0.9532 ± NA [0.9532, 0.9532] |
| eata | S4 | 1/5 | 6.84 ± NA [6.84, 6.84] | 0.9664 ± NA [0.9664, 0.9664] |
| eata | S5 | 1/5 | 11.55 ± NA [11.55, 11.55] | 0.9398 ± NA [0.9398, 0.9398] |
| eata | S6 | 1/5 | 11.38 ± NA [11.38, 11.38] | 0.9403 ± NA [0.9403, 0.9403] |
| rotta | S1 | 1/5 | 16.31 ± NA [16.31, 16.31] | 0.9087 ± NA [0.9087, 0.9087] |
| rotta | S2 | 1/5 | 17.09 ± NA [17.09, 17.09] | 0.8899 ± NA [0.8899, 0.8899] |
| rotta | S3 | 1/5 | 8.69 ± NA [8.69, 8.69] | 0.9564 ± NA [0.9564, 0.9564] |
| rotta | S4 | 1/5 | 7.91 ± NA [7.91, 7.91] | 0.9646 ± NA [0.9646, 0.9646] |
| rotta | S5 | 1/5 | 10.84 ± NA [10.84, 10.84] | 0.9445 ± NA [0.9445, 0.9445] |
| rotta | S6 | 1/5 | 10.51 ± NA [10.51, 10.51] | 0.9497 ± NA [0.9497, 0.9497] |
| lame | S1 | 1/5 | 10.45 ± NA [10.45, 10.45] | 0.9525 ± NA [0.9525, 0.9525] |
| lame | S2 | 1/5 | 23.78 ± NA [23.78, 23.78] | 0.8459 ± NA [0.8459, 0.8459] |
| lame | S3 | 1/5 | 6.84 ± NA [6.84, 6.84] | 0.9721 ± NA [0.9721, 0.9721] |
| lame | S4 | 1/5 | 4.49 ± NA [4.49, 4.49] | 0.9876 ± NA [0.9876, 0.9876] |
| lame | S5 | 1/5 | 11.25 ± NA [11.25, 11.25] | 0.9385 ± NA [0.9385, 0.9385] |
| lame | S6 | 1/5 | 11.14 ± NA [11.14, 11.14] | 0.9383 ± NA [0.9383, 0.9383] |

## Bootstrap uncertainty

Stationary paired 1,000-resample intervals are conditional on the realized online trajectories. No adaptation is rerun inside bootstrap. Per-order intervals are saved in `bootstrap.json`.
A positive-only 95% interval favors the method; a negative-only interval favors Frozen. Intervals crossing zero are inconclusive.
Bootstrap status: NOT_RUN.

## Reproduction

See `PROTOCOL.md`, `config.json`, `streams_locked/`, `results/<run_id>/source_only/`, each run’s `command.json`, `status.json`, `scores.jsonl`, and the launcher logs. First scores and large LL caches stay on this workstation; compact summary tables and the report are versioned.

## One next research step

Finish all available fixed-protocol stream runs and then inspect the complete five-order comparison.

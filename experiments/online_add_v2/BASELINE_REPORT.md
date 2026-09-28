# BASELINE_REPORT — Online ADD Baselines v2

**Stage:** Online ADD Baselines v2 · **Status:** COMPLETE
**Completed runs:** 180 / 180
**Target domains:** ITW = 3,178; WaveFake = 4,096; LA21 = 4,096; DF21 = 4,096.
**Methods:** frozen = COMPLETE (30/30); bn_only = COMPLETE (30/30); tent = COMPLETE (30/30); eata = COMPLETE (30/30); rotta = COMPLETE (30/30); lame = COMPLETE (30/30).
**Operating thresholds:** method-specific source-select calibration (v2.1 supplement; chronology disclosed below).

## Stationary mean over five orders

| Method | ITW EER% / AUC | WaveFake EER% / AUC | LA21 EER% / AUC | DF21 EER% / AUC |
|---|---:|---:|---:|---:|
| frozen | 9.86 / 0.9633 | 15.77 / 0.9150 | 6.54 / 0.9747 | 3.76 / 0.9915 |
| bn_only | 12.46 / 0.9378 | 17.44 / 0.9011 | 8.17 / 0.9551 | 6.77 / 0.9707 |
| tent | 12.56 / 0.9364 | 19.10 / 0.8835 | 8.20 / 0.9558 | 6.47 / 0.9724 |
| eata | 12.56 / 0.9371 | 17.44 / 0.9011 | 8.23 / 0.9549 | 6.77 / 0.9704 |
| rotta | 14.17 / 0.9312 | 17.53 / 0.8850 | 9.11 / 0.9525 | 7.56 / 0.9715 |
| lame | 10.51 / 0.9515 | 23.51 / 0.8442 | 6.58 / 0.9730 | 4.25 / 0.9874 |

## Dynamic mean over five orders

| Method | S5 macro EER% / AUC | S6 macro EER% / AUC | Worst rolling EER% / AUC | Switch first256 EER% / AUC |
|---|---:|---:|---:|---:|
| frozen | 8.98 / 0.9611 | 8.98 / 0.9611 | 20.93 / 0.8704 | 9.52 / 0.9586 (10/10) |
| bn_only | 11.20 / 0.9413 | 11.18 / 0.9411 | 24.24 / 0.8505 | 11.78 / 0.9389 (10/10) |
| tent | 11.16 / 0.9416 | 10.98 / 0.9433 | 25.00 / 0.8481 | 11.35 / 0.9427 (10/10) |
| eata | 11.25 / 0.9410 | 11.23 / 0.9408 | 23.74 / 0.8507 | 11.88 / 0.9386 (10/10) |
| rotta | 11.07 / 0.9441 | 10.44 / 0.9498 | 20.77 / 0.8869 | 10.22 / 0.9532 (10/10) |
| lame | 11.15 / 0.9390 | 11.18 / 0.9393 | 29.91 / 0.7635 | 12.10 / 0.9327 (10/10) |

## Efficiency

| Method | Updates / 100 | Adapted fraction | Backend ms/audio | Peak GPU MB | Memory records |
|---|---:|---:|---:|---:|---:|
| frozen | 0.0000 | 0.0000 | 0.70 | 1213 | 0 |
| bn_only | 0.0000 | 0.0000 | 0.70 | 1213 | 0 |
| tent | 6.2528 | 1.0000 | 2.75 | 1213 | 0 |
| eata | 4.7289 | 0.5444 | 2.64 | 1213 | 0 |
| rotta | 1.5576 | 0.9951 | 2.79 | 2094 | 64 |
| lame | 0.0000 | 0.0000 | 0.78 | 1213 | 0 |

Adapted fraction counts samples selected for a post-prediction optimizer update. BN-only uses current-batch statistics and LAME refines current-batch output without such an update, so both show zero in that column. Memory records counts retained target examples; model/optimizer state is described in `PROTOCOL.md` and per-run status.

## Main observations

1. Adapted methods gain AUC in 0/100 stationary and 0/50 dynamic segment-macro paired comparisons with Frozen.
2. Least negative stationary mean ΔAUC: bn_only -0.0200; least negative dynamic mean ΔAUC: rotta -0.0142.
3. Largest domain-mean AUC drop: lame on WaveFake (-0.0708). Five arrival orders are not independent target datasets.

**Which existing mechanisms appear useful for audio ADD online TTA?** None improves mean stationary or dynamic segment-macro AUC over Frozen. rotta is least damaging by dynamic mean ΔAUC (-0.0142), but that is not an overall benefit.

**Which mechanisms fail or drift?** bn_only, tent, eata, rotta, lame have negative mean stationary ΔAUC. lame on WaveFake has the largest domain-mean drop (-0.0708); switch and rolling metrics show its dynamic behavior. These are development-stream observations, not a final-holdout claim.

**Can this benchmark support designing a new method?** YES

**Target90 accessed:** NO · **Final holdout accessed:** NO

## Protocol and limitations

All scores are first predictions on B16 predict-then-adapt streams; LAME refines its current batch output. The four domains are fixed development resources. LA21/DF21 are selected from official eval releases, so they are development subsets, not untouched final holdouts. Each method uses its own source-select threshold for FPR/FNR/BA; the v2.1 supplement arrived after interim target metrics, so this is a disclosed source-only evaluator correction. The earlier shared-threshold metrics remain in historical files. Backend timing is synchronized prediction/update elapsed wall time, including host launch overhead but excluding frozen frontend extraction, cache I/O and score-file writes. See `ACCESS_BOUNDARY.md` for file and label visibility.

## Source-only operating points

The same fixed 1,024 source-select IDs (512 per class) are scored once per freshly reset method. The threshold minimizes |FPR−FNR|, then mean error, then the numeric threshold; no target score or label enters calibration.

| Method | Score form | Source-select threshold | Source-select EER% |
|---|---|---:|---:|
| frozen | native spoof-minus-bonafide logits | -4.283476 | 0.20 |
| bn_only | native spoof-minus-bonafide logits | 3.308916 | 0.20 |
| tent | native spoof-minus-bonafide logits | 2.902970 | 0.20 |
| eata | native spoof-minus-bonafide logits | 3.324137 | 0.20 |
| rotta | native spoof-minus-bonafide logits | -0.558506 | 0.20 |
| lame | post-LAME spoof/bonafide log-odds | -5.155588 | 0.20 |

## Stationary paired effects

Positive ΔAUC and positive EER gain favor the method over Frozen. Each cell averages available arrival orders on the same fixed domain subset.

| Method | Domain | Orders | Mean ΔAUC | Mean EER gain pp | AUC gains / orders |
|---|---|---:|---:|---:|---:|
| bn_only | ITW | 5/5 | -0.0256 | -2.5981 | 0/5 |
| bn_only | WaveFake | 5/5 | -0.0139 | -1.6699 | 0/5 |
| bn_only | LA21 | 5/5 | -0.0197 | -1.6309 | 0/5 |
| bn_only | DF21 | 5/5 | -0.0208 | -3.0078 | 0/5 |
| tent | ITW | 5/5 | -0.0269 | -2.7041 | 0/5 |
| tent | WaveFake | 5/5 | -0.0315 | -3.3301 | 0/5 |
| tent | LA21 | 5/5 | -0.0189 | -1.6602 | 0/5 |
| tent | DF21 | 5/5 | -0.0191 | -2.7148 | 0/5 |
| eata | ITW | 5/5 | -0.0262 | -2.7058 | 0/5 |
| eata | WaveFake | 5/5 | -0.0139 | -1.6699 | 0/5 |
| eata | LA21 | 5/5 | -0.0199 | -1.6895 | 0/5 |
| eata | DF21 | 5/5 | -0.0211 | -3.0078 | 0/5 |
| rotta | ITW | 5/5 | -0.0321 | -4.3081 | 0/5 |
| rotta | WaveFake | 5/5 | -0.0300 | -1.7578 | 0/5 |
| rotta | LA21 | 5/5 | -0.0222 | -2.5684 | 0/5 |
| rotta | DF21 | 5/5 | -0.0200 | -3.7988 | 0/5 |
| lame | ITW | 5/5 | -0.0118 | -0.6496 | 0/5 |
| lame | WaveFake | 5/5 | -0.0708 | -7.7344 | 0/5 |
| lame | LA21 | 5/5 | -0.0017 | -0.0391 | 0/5 |
| lame | DF21 | 5/5 | -0.0041 | -0.4883 | 0/5 |

## Dynamic segments

Within-domain rolling windows use 256 records and stride 128; single-class windows are NA. First256 applies only after a switch. Full per-order segment metrics, rolling counts and supplementary pooled metrics are in `metrics_v21.csv` and `summary_v21.json`.

| Method | Stream | Segment / domain | Orders | EER% / AUC | First256 EER% / AUC | Rolling mean EER% / AUC | Worst rolling EER% / AUC | Valid windows total |
|---|---|---|---:|---:|---:|---:|---:|---:|
| frozen | S5 | 0 / ITW | 5/5 | 9.86 / 0.9633 | NA | 10.01 / 0.9635 | 14.74 / 0.9408 | 115 |
| frozen | S5 | 1 / LA21 | 5/5 | 6.54 / 0.9747 | 6.99 / 0.9766 | 6.35 / 0.9748 | 10.40 / 0.9428 | 155 |
| frozen | S5 | 2 / WaveFake | 5/5 | 15.77 / 0.9150 | 15.43 / 0.9222 | 15.83 / 0.9148 | 20.93 / 0.8767 | 155 |
| frozen | S5 | 3 / DF21 | 5/5 | 3.76 / 0.9915 | 3.14 / 0.9945 | 3.74 / 0.9915 | 7.03 / 0.9721 | 155 |
| frozen | S6 | 0 / DF21 | 5/5 | 3.76 / 0.9915 | NA | 3.87 / 0.9915 | 7.20 / 0.9780 | 155 |
| frozen | S6 | 1 / WaveFake | 5/5 | 15.77 / 0.9150 | 15.21 / 0.9219 | 15.73 / 0.9150 | 20.31 / 0.8704 | 155 |
| frozen | S6 | 2 / LA21 | 5/5 | 6.54 / 0.9747 | 6.38 / 0.9721 | 6.46 / 0.9747 | 11.02 / 0.9501 | 155 |
| frozen | S6 | 3 / ITW | 5/5 | 9.86 / 0.9633 | 9.98 / 0.9645 | 10.01 / 0.9630 | 15.00 / 0.9385 | 115 |
| bn_only | S5 | 0 / ITW | 5/5 | 12.45 / 0.9385 | NA | 12.41 / 0.9392 | 18.82 / 0.9039 | 115 |
| bn_only | S5 | 1 / LA21 | 5/5 | 8.19 / 0.9554 | 8.12 / 0.9560 | 8.07 / 0.9561 | 12.31 / 0.9163 | 155 |
| bn_only | S5 | 2 / WaveFake | 5/5 | 17.44 / 0.9007 | 17.34 / 0.9050 | 17.40 / 0.9013 | 24.24 / 0.8505 | 155 |
| bn_only | S5 | 3 / DF21 | 5/5 | 6.71 / 0.9707 | 7.66 / 0.9672 | 6.50 / 0.9715 | 12.50 / 0.9280 | 155 |
| bn_only | S6 | 0 / DF21 | 5/5 | 6.59 / 0.9705 | NA | 6.47 / 0.9711 | 13.01 / 0.9417 | 155 |
| bn_only | S6 | 1 / WaveFake | 5/5 | 17.53 / 0.9003 | 16.46 / 0.9143 | 17.25 / 0.9018 | 22.22 / 0.8604 | 155 |
| bn_only | S6 | 2 / LA21 | 5/5 | 8.08 / 0.9558 | 8.13 / 0.9547 | 7.98 / 0.9565 | 13.14 / 0.9119 | 155 |
| bn_only | S6 | 3 / ITW | 5/5 | 12.52 / 0.9378 | 12.95 / 0.9364 | 12.51 / 0.9382 | 17.65 / 0.9003 | 115 |
| tent | S5 | 0 / ITW | 5/5 | 12.61 / 0.9377 | NA | 12.55 / 0.9385 | 19.41 / 0.9026 | 115 |
| tent | S5 | 1 / LA21 | 5/5 | 8.06 / 0.9565 | 8.09 / 0.9573 | 7.98 / 0.9574 | 12.69 / 0.9145 | 155 |
| tent | S5 | 2 / WaveFake | 5/5 | 18.40 / 0.8945 | 17.77 / 0.9045 | 17.44 / 0.9010 | 25.00 / 0.8481 | 155 |
| tent | S5 | 3 / DF21 | 5/5 | 5.57 / 0.9776 | 5.86 / 0.9775 | 5.49 / 0.9782 | 11.11 / 0.9430 | 155 |
| tent | S6 | 0 / DF21 | 5/5 | 6.46 / 0.9726 | NA | 6.16 / 0.9737 | 12.20 / 0.9482 | 155 |
| tent | S6 | 1 / WaveFake | 5/5 | 18.46 / 0.8943 | 17.17 / 0.9117 | 17.45 / 0.9013 | 24.60 / 0.8552 | 155 |
| tent | S6 | 2 / LA21 | 5/5 | 7.30 / 0.9615 | 7.11 / 0.9615 | 7.32 / 0.9626 | 12.61 / 0.9212 | 155 |
| tent | S6 | 3 / ITW | 5/5 | 11.71 / 0.9449 | 12.07 / 0.9436 | 11.74 / 0.9453 | 16.67 / 0.9096 | 115 |
| eata | S5 | 0 / ITW | 5/5 | 12.50 / 0.9380 | NA | 12.47 / 0.9387 | 18.82 / 0.9031 | 115 |
| eata | S5 | 1 / LA21 | 5/5 | 8.25 / 0.9550 | 8.23 / 0.9559 | 8.11 / 0.9558 | 12.31 / 0.9153 | 155 |
| eata | S5 | 2 / WaveFake | 5/5 | 17.44 / 0.9007 | 17.34 / 0.9047 | 17.35 / 0.9013 | 23.74 / 0.8507 | 155 |
| eata | S5 | 3 / DF21 | 5/5 | 6.80 / 0.9701 | 7.70 / 0.9666 | 6.58 / 0.9708 | 12.50 / 0.9272 | 155 |
| eata | S6 | 0 / DF21 | 5/5 | 6.58 / 0.9703 | NA | 6.49 / 0.9709 | 13.01 / 0.9411 | 155 |
| eata | S6 | 1 / WaveFake | 5/5 | 17.56 / 0.9003 | 16.54 / 0.9145 | 17.26 / 0.9017 | 23.02 / 0.8600 | 155 |
| eata | S6 | 2 / LA21 | 5/5 | 8.15 / 0.9555 | 8.09 / 0.9547 | 8.03 / 0.9563 | 13.18 / 0.9119 | 155 |
| eata | S6 | 3 / ITW | 5/5 | 12.64 / 0.9370 | 13.39 / 0.9352 | 12.58 / 0.9374 | 17.48 / 0.8997 | 115 |
| rotta | S5 | 0 / ITW | 5/5 | 14.32 / 0.9295 | NA | 11.67 / 0.9516 | 16.83 / 0.9233 | 115 |
| rotta | S5 | 1 / LA21 | 5/5 | 8.13 / 0.9566 | 8.62 / 0.9588 | 7.66 / 0.9586 | 12.40 / 0.9213 | 155 |
| rotta | S5 | 2 / WaveFake | 5/5 | 16.91 / 0.9069 | 14.80 / 0.9275 | 14.53 / 0.9265 | 20.15 / 0.8869 | 155 |
| rotta | S5 | 3 / DF21 | 5/5 | 4.92 / 0.9833 | 5.56 / 0.9824 | 4.60 / 0.9865 | 8.57 / 0.9610 | 155 |
| rotta | S6 | 0 / DF21 | 5/5 | 7.67 / 0.9688 | NA | 5.66 / 0.9824 | 10.57 / 0.9629 | 155 |
| rotta | S6 | 1 / WaveFake | 5/5 | 16.04 / 0.9123 | 13.96 / 0.9294 | 14.48 / 0.9267 | 20.77 / 0.8884 | 155 |
| rotta | S6 | 2 / LA21 | 5/5 | 7.46 / 0.9609 | 7.88 / 0.9604 | 7.29 / 0.9623 | 12.69 / 0.9224 | 155 |
| rotta | S6 | 3 / ITW | 5/5 | 10.58 / 0.9572 | 10.52 / 0.9604 | 10.58 / 0.9575 | 15.00 / 0.9304 | 115 |
| lame | S5 | 0 / ITW | 5/5 | 10.50 / 0.9514 | NA | 10.65 / 0.9514 | 15.79 / 0.9164 | 115 |
| lame | S5 | 1 / LA21 | 5/5 | 6.64 / 0.9730 | 6.65 / 0.9762 | 6.56 / 0.9729 | 10.14 / 0.9369 | 155 |
| lame | S5 | 2 / WaveFake | 5/5 | 23.20 / 0.8439 | 23.34 / 0.8506 | 23.14 / 0.8437 | 29.23 / 0.7819 | 155 |
| lame | S5 | 3 / DF21 | 5/5 | 4.24 / 0.9875 | 4.22 / 0.9896 | 4.20 / 0.9877 | 9.63 / 0.9586 | 155 |
| lame | S6 | 0 / DF21 | 5/5 | 4.28 / 0.9876 | NA | 4.16 / 0.9877 | 7.20 / 0.9688 | 155 |
| lame | S6 | 1 / WaveFake | 5/5 | 23.33 / 0.8455 | 21.72 / 0.8591 | 23.31 / 0.8454 | 29.91 / 0.7635 | 155 |
| lame | S6 | 2 / LA21 | 5/5 | 6.59 / 0.9733 | 6.36 / 0.9708 | 6.65 / 0.9734 | 11.02 / 0.9463 | 155 |
| lame | S6 | 3 / ITW | 5/5 | 10.53 / 0.9508 | 10.33 / 0.9497 | 10.41 / 0.9510 | 15.52 / 0.9171 | 115 |

## Run inventory

| Method | Stream | Order | Status | EER% | AUC | ΔAUC vs Frozen | EER gain pp vs Frozen | FPR% @ source τ | FNR% @ source τ | BA% @ source τ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| frozen | S1 | 2026 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 | 51.31 | 0.44 | 74.13 |
| bn_only | S1 | 2026 | COMPLETE | 12.37 | 0.9386 | -0.0247 | -2.5136 | 35.63 | 1.57 | 81.40 |
| tent | S1 | 2026 | COMPLETE | 12.45 | 0.9377 | -0.0256 | -2.5885 | 36.13 | 1.57 | 81.15 |
| eata | S1 | 2026 | COMPLETE | 12.36 | 0.9383 | -0.0250 | -2.5015 | 35.58 | 1.57 | 81.42 |
| rotta | S1 | 2026 | COMPLETE | 16.31 | 0.9087 | -0.0546 | -6.4564 | 76.44 | 0.26 | 61.65 |
| lame | S1 | 2026 | COMPLETE | 10.45 | 0.9525 | -0.0108 | -0.5914 | 27.99 | 2.26 | 84.87 |
| frozen | S2 | 2026 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 | 6.01 | 26.37 | 83.81 |
| bn_only | S2 | 2026 | COMPLETE | 17.29 | 0.9017 | -0.0133 | -1.5137 | 12.79 | 19.63 | 83.79 |
| tent | S2 | 2026 | COMPLETE | 18.70 | 0.8856 | -0.0294 | -2.9297 | 66.55 | 4.83 | 64.31 |
| eata | S2 | 2026 | COMPLETE | 17.19 | 0.9017 | -0.0133 | -1.4160 | 12.70 | 19.82 | 83.74 |
| rotta | S2 | 2026 | COMPLETE | 17.09 | 0.8899 | -0.0251 | -1.3184 | 85.74 | 3.91 | 55.18 |
| lame | S2 | 2026 | COMPLETE | 23.78 | 0.8459 | -0.0691 | -8.0078 | 1.37 | 44.19 | 77.22 |
| frozen | S3 | 2026 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 | 26.46 | 0.44 | 86.55 |
| bn_only | S3 | 2026 | COMPLETE | 8.45 | 0.9533 | -0.0214 | -1.9043 | 19.38 | 1.61 | 89.50 |
| tent | S3 | 2026 | COMPLETE | 8.79 | 0.9540 | -0.0208 | -2.2461 | 18.26 | 1.71 | 90.01 |
| eata | S3 | 2026 | COMPLETE | 8.54 | 0.9532 | -0.0215 | -2.0020 | 19.24 | 1.61 | 89.58 |
| rotta | S3 | 2026 | COMPLETE | 8.69 | 0.9564 | -0.0183 | -2.1484 | 44.38 | 0.15 | 77.73 |
| lame | S3 | 2026 | COMPLETE | 6.84 | 0.9721 | -0.0027 | -0.2930 | 14.16 | 1.95 | 91.94 |
| frozen | S4 | 2026 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 | 20.80 | 0.20 | 89.50 |
| bn_only | S4 | 2026 | COMPLETE | 6.84 | 0.9666 | -0.0249 | -3.0762 | 16.26 | 0.44 | 91.65 |
| tent | S4 | 2026 | COMPLETE | 6.59 | 0.9680 | -0.0235 | -2.8320 | 15.09 | 0.59 | 92.16 |
| eata | S4 | 2026 | COMPLETE | 6.84 | 0.9664 | -0.0251 | -3.0762 | 16.26 | 0.44 | 91.65 |
| rotta | S4 | 2026 | COMPLETE | 7.91 | 0.9646 | -0.0269 | -4.1504 | 41.06 | 0.10 | 79.42 |
| lame | S4 | 2026 | COMPLETE | 4.49 | 0.9876 | -0.0039 | -0.7324 | 13.18 | 0.63 | 93.09 |
| frozen | S5 | 2026 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S5 | 2026 | COMPLETE | 12.22 | 0.9400 | 0.0018 | 1.8965 | 20.98 | 6.58 | 86.22 |
| tent | S5 | 2026 | COMPLETE | 12.90 | 0.9256 | -0.0126 | 1.2235 | 16.46 | 10.41 | 86.57 |
| eata | S5 | 2026 | COMPLETE | 12.16 | 0.9401 | 0.0019 | 1.9573 | 21.06 | 6.58 | 86.18 |
| rotta | S5 | 2026 | COMPLETE | 13.16 | 0.9440 | 0.0058 | 0.9564 | 65.01 | 0.23 | 67.38 |
| lame | S5 | 2026 | COMPLETE | 13.67 | 0.9153 | -0.0230 | 0.4490 | 14.22 | 13.45 | 86.17 |
| frozen | S6 | 2026 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S6 | 2026 | COMPLETE | 12.25 | 0.9405 | 0.0022 | 1.8720 | 21.24 | 6.55 | 86.10 |
| tent | S6 | 2026 | COMPLETE | 12.04 | 0.9404 | 0.0021 | 2.0807 | 13.85 | 10.63 | 87.76 |
| eata | S6 | 2026 | COMPLETE | 12.24 | 0.9399 | 0.0017 | 1.8843 | 21.27 | 6.55 | 86.09 |
| rotta | S6 | 2026 | COMPLETE | 13.56 | 0.9289 | -0.0093 | 0.5587 | 66.47 | 0.84 | 66.34 |
| lame | S6 | 2026 | COMPLETE | 13.96 | 0.9152 | -0.0230 | 0.1611 | 14.14 | 13.75 | 86.05 |
| frozen | S1 | 2027 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 | 51.31 | 0.44 | 74.13 |
| bn_only | S1 | 2027 | COMPLETE | 13.05 | 0.9349 | -0.0284 | -3.1978 | 34.30 | 1.57 | 82.07 |
| tent | S1 | 2027 | COMPLETE | 12.88 | 0.9330 | -0.0303 | -3.0237 | 34.75 | 1.57 | 81.84 |
| eata | S1 | 2027 | COMPLETE | 13.21 | 0.9342 | -0.0291 | -3.3514 | 34.25 | 1.48 | 82.13 |
| rotta | S1 | 2027 | COMPLETE | 12.62 | 0.9404 | -0.0229 | -2.7626 | 77.53 | 0.35 | 61.06 |
| lame | S1 | 2027 | COMPLETE | 10.55 | 0.9511 | -0.0122 | -0.6900 | 28.14 | 2.79 | 84.54 |
| frozen | S2 | 2027 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 | 6.01 | 26.37 | 83.81 |
| bn_only | S2 | 2027 | COMPLETE | 17.68 | 0.9004 | -0.0146 | -1.9043 | 13.82 | 19.68 | 83.25 |
| tent | S2 | 2027 | COMPLETE | 19.38 | 0.8792 | -0.0358 | -3.6133 | 65.19 | 6.20 | 64.31 |
| eata | S2 | 2027 | COMPLETE | 17.68 | 0.9005 | -0.0145 | -1.9043 | 13.62 | 20.02 | 83.18 |
| rotta | S2 | 2027 | COMPLETE | 17.38 | 0.8865 | -0.0285 | -1.6113 | 86.47 | 4.20 | 54.66 |
| lame | S2 | 2027 | COMPLETE | 23.14 | 0.8484 | -0.0666 | -7.3730 | 1.71 | 43.80 | 77.25 |
| frozen | S3 | 2027 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 | 26.46 | 0.44 | 86.55 |
| bn_only | S3 | 2027 | COMPLETE | 7.96 | 0.9535 | -0.0212 | -1.4160 | 19.48 | 1.42 | 89.55 |
| tent | S3 | 2027 | COMPLETE | 7.71 | 0.9541 | -0.0206 | -1.1719 | 18.70 | 1.56 | 89.87 |
| eata | S3 | 2027 | COMPLETE | 8.01 | 0.9533 | -0.0215 | -1.4648 | 19.53 | 1.46 | 89.50 |
| rotta | S3 | 2027 | COMPLETE | 9.62 | 0.9499 | -0.0249 | -3.0762 | 45.95 | 0.20 | 76.93 |
| lame | S3 | 2027 | COMPLETE | 6.49 | 0.9730 | -0.0017 | 0.0488 | 13.62 | 1.95 | 92.21 |
| frozen | S4 | 2027 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 | 20.80 | 0.20 | 89.50 |
| bn_only | S4 | 2027 | COMPLETE | 6.84 | 0.9718 | -0.0197 | -3.0762 | 14.94 | 0.54 | 92.26 |
| tent | S4 | 2027 | COMPLETE | 6.49 | 0.9728 | -0.0187 | -2.7344 | 13.67 | 0.68 | 92.82 |
| eata | S4 | 2027 | COMPLETE | 6.88 | 0.9716 | -0.0199 | -3.1250 | 14.89 | 0.54 | 92.29 |
| rotta | S4 | 2027 | COMPLETE | 7.81 | 0.9716 | -0.0199 | -4.0527 | 34.91 | 0.05 | 82.52 |
| lame | S4 | 2027 | COMPLETE | 4.05 | 0.9885 | -0.0030 | -0.2930 | 13.13 | 0.59 | 93.14 |
| frozen | S5 | 2027 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S5 | 2027 | COMPLETE | 12.13 | 0.9400 | 0.0018 | 1.9847 | 21.00 | 6.68 | 86.16 |
| tent | S5 | 2027 | COMPLETE | 12.79 | 0.9267 | -0.0115 | 1.3266 | 17.18 | 9.82 | 86.50 |
| eata | S5 | 2027 | COMPLETE | 12.08 | 0.9402 | 0.0020 | 2.0433 | 21.12 | 6.53 | 86.18 |
| rotta | S5 | 2027 | COMPLETE | 13.53 | 0.9348 | -0.0034 | 0.5861 | 70.24 | 0.66 | 64.55 |
| lame | S5 | 2027 | COMPLETE | 13.85 | 0.9156 | -0.0227 | 0.2708 | 14.10 | 13.71 | 86.10 |
| frozen | S6 | 2027 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S6 | 2027 | COMPLETE | 12.20 | 0.9423 | 0.0040 | 1.9162 | 20.58 | 6.21 | 86.60 |
| tent | S6 | 2027 | COMPLETE | 12.33 | 0.9414 | 0.0032 | 1.7928 | 13.39 | 11.13 | 87.74 |
| eata | S6 | 2027 | COMPLETE | 12.18 | 0.9418 | 0.0036 | 1.9436 | 20.63 | 6.21 | 86.58 |
| rotta | S6 | 2027 | COMPLETE | 13.96 | 0.9222 | -0.0160 | 0.1611 | 67.60 | 0.59 | 65.90 |
| lame | S6 | 2027 | COMPLETE | 13.84 | 0.9165 | -0.0217 | 0.2814 | 14.14 | 13.48 | 86.19 |
| frozen | S1 | 2028 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 | 51.31 | 0.44 | 74.13 |
| bn_only | S1 | 2028 | COMPLETE | 12.01 | 0.9379 | -0.0254 | -2.1534 | 35.83 | 2.00 | 81.08 |
| tent | S1 | 2028 | COMPLETE | 12.10 | 0.9371 | -0.0262 | -2.2404 | 35.49 | 1.74 | 81.39 |
| eata | S1 | 2028 | COMPLETE | 12.18 | 0.9372 | -0.0261 | -2.3274 | 35.73 | 2.00 | 81.13 |
| rotta | S1 | 2028 | COMPLETE | 13.16 | 0.9382 | -0.0251 | -3.3021 | 88.71 | 0.09 | 55.60 |
| lame | S1 | 2028 | COMPLETE | 10.44 | 0.9531 | -0.0102 | -0.5868 | 28.29 | 2.44 | 84.64 |
| frozen | S2 | 2028 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 | 6.01 | 26.37 | 83.81 |
| bn_only | S2 | 2028 | COMPLETE | 17.48 | 0.8983 | -0.0167 | -1.7090 | 13.57 | 20.41 | 83.01 |
| tent | S2 | 2028 | COMPLETE | 19.34 | 0.8811 | -0.0339 | -3.5645 | 64.01 | 6.01 | 64.99 |
| eata | S2 | 2028 | COMPLETE | 17.53 | 0.8984 | -0.0166 | -1.7578 | 13.53 | 20.41 | 83.03 |
| rotta | S2 | 2028 | COMPLETE | 17.97 | 0.8812 | -0.0338 | -2.1973 | 87.65 | 4.20 | 54.08 |
| lame | S2 | 2028 | COMPLETE | 23.54 | 0.8424 | -0.0726 | -7.7637 | 1.42 | 43.80 | 77.39 |
| frozen | S3 | 2028 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 | 26.46 | 0.44 | 86.55 |
| bn_only | S3 | 2028 | COMPLETE | 8.06 | 0.9561 | -0.0186 | -1.5137 | 19.34 | 1.32 | 89.67 |
| tent | S3 | 2028 | COMPLETE | 7.86 | 0.9569 | -0.0178 | -1.3184 | 18.70 | 1.56 | 89.87 |
| eata | S3 | 2028 | COMPLETE | 8.11 | 0.9559 | -0.0189 | -1.5625 | 19.29 | 1.32 | 89.70 |
| rotta | S3 | 2028 | COMPLETE | 8.94 | 0.9527 | -0.0221 | -2.3926 | 49.56 | 0.24 | 75.10 |
| lame | S3 | 2028 | COMPLETE | 6.45 | 0.9727 | -0.0020 | 0.0977 | 13.77 | 1.81 | 92.21 |
| frozen | S4 | 2028 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 | 20.80 | 0.20 | 89.50 |
| bn_only | S4 | 2028 | COMPLETE | 6.79 | 0.9717 | -0.0198 | -3.0273 | 15.09 | 0.44 | 92.24 |
| tent | S4 | 2028 | COMPLETE | 6.40 | 0.9736 | -0.0179 | -2.6367 | 14.01 | 0.78 | 92.60 |
| eata | S4 | 2028 | COMPLETE | 6.79 | 0.9714 | -0.0201 | -3.0273 | 14.99 | 0.44 | 92.29 |
| rotta | S4 | 2028 | COMPLETE | 7.28 | 0.9725 | -0.0190 | -3.5156 | 44.68 | 0.20 | 77.56 |
| lame | S4 | 2028 | COMPLETE | 4.35 | 0.9861 | -0.0054 | -0.5859 | 12.84 | 0.59 | 93.29 |
| frozen | S5 | 2028 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S5 | 2028 | COMPLETE | 12.42 | 0.9399 | 0.0017 | 1.7007 | 20.98 | 6.44 | 86.29 |
| tent | S5 | 2028 | COMPLETE | 13.14 | 0.9248 | -0.0134 | 0.9838 | 16.69 | 10.37 | 86.47 |
| eata | S5 | 2028 | COMPLETE | 12.36 | 0.9400 | 0.0017 | 1.7619 | 20.93 | 6.43 | 86.32 |
| rotta | S5 | 2028 | COMPLETE | 13.22 | 0.9360 | -0.0022 | 0.9015 | 64.15 | 0.43 | 67.71 |
| lame | S5 | 2028 | COMPLETE | 13.85 | 0.9160 | -0.0222 | 0.2708 | 14.43 | 13.49 | 86.04 |
| frozen | S6 | 2028 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S6 | 2028 | COMPLETE | 12.30 | 0.9405 | 0.0023 | 1.8231 | 21.08 | 6.27 | 86.33 |
| tent | S6 | 2028 | COMPLETE | 12.24 | 0.9398 | 0.0016 | 1.8750 | 13.45 | 11.28 | 87.63 |
| eata | S6 | 2028 | COMPLETE | 12.28 | 0.9401 | 0.0019 | 1.8353 | 21.07 | 6.23 | 86.35 |
| rotta | S6 | 2028 | COMPLETE | 13.73 | 0.9271 | -0.0111 | 0.3942 | 67.69 | 0.51 | 65.90 |
| lame | S6 | 2028 | COMPLETE | 13.66 | 0.9169 | -0.0213 | 0.4627 | 14.07 | 13.31 | 86.31 |
| frozen | S1 | 2029 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 | 51.31 | 0.44 | 74.13 |
| bn_only | S1 | 2029 | COMPLETE | 12.27 | 0.9411 | -0.0222 | -2.4150 | 35.68 | 1.65 | 81.33 |
| tent | S1 | 2029 | COMPLETE | 12.57 | 0.9400 | -0.0233 | -2.7107 | 35.58 | 1.57 | 81.42 |
| eata | S1 | 2029 | COMPLETE | 12.45 | 0.9402 | -0.0231 | -2.5885 | 35.63 | 1.74 | 81.31 |
| rotta | S1 | 2029 | COMPLETE | 14.83 | 0.9333 | -0.0300 | -4.9778 | 88.66 | 0.00 | 55.67 |
| lame | S1 | 2029 | COMPLETE | 10.65 | 0.9508 | -0.0125 | -0.7886 | 28.24 | 2.79 | 84.49 |
| frozen | S2 | 2029 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 | 6.01 | 26.37 | 83.81 |
| bn_only | S2 | 2029 | COMPLETE | 17.97 | 0.8998 | -0.0152 | -2.1973 | 14.06 | 20.56 | 82.69 |
| tent | S2 | 2029 | COMPLETE | 19.19 | 0.8852 | -0.0298 | -3.4180 | 66.80 | 3.96 | 64.62 |
| eata | S2 | 2029 | COMPLETE | 17.97 | 0.8997 | -0.0153 | -2.1973 | 13.96 | 20.61 | 82.71 |
| rotta | S2 | 2029 | COMPLETE | 17.29 | 0.8848 | -0.0302 | -1.5137 | 87.40 | 4.15 | 54.22 |
| lame | S2 | 2029 | COMPLETE | 23.93 | 0.8410 | -0.0740 | -8.1543 | 1.66 | 44.97 | 76.68 |
| frozen | S3 | 2029 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 | 26.46 | 0.44 | 86.55 |
| bn_only | S3 | 2029 | COMPLETE | 8.40 | 0.9538 | -0.0209 | -1.8555 | 19.19 | 1.66 | 89.58 |
| tent | S3 | 2029 | COMPLETE | 8.59 | 0.9545 | -0.0202 | -2.0508 | 18.60 | 1.76 | 89.82 |
| eata | S3 | 2029 | COMPLETE | 8.45 | 0.9534 | -0.0213 | -1.9043 | 19.14 | 1.66 | 89.60 |
| rotta | S3 | 2029 | COMPLETE | 9.57 | 0.9498 | -0.0250 | -3.0273 | 48.73 | 0.29 | 75.49 |
| lame | S3 | 2029 | COMPLETE | 6.54 | 0.9733 | -0.0014 | 0.0000 | 13.87 | 2.00 | 92.07 |
| frozen | S4 | 2029 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 | 20.80 | 0.20 | 89.50 |
| bn_only | S4 | 2029 | COMPLETE | 6.54 | 0.9739 | -0.0176 | -2.7832 | 15.72 | 0.63 | 91.82 |
| tent | S4 | 2029 | COMPLETE | 6.35 | 0.9763 | -0.0151 | -2.5879 | 14.16 | 0.68 | 92.58 |
| eata | S4 | 2029 | COMPLETE | 6.49 | 0.9736 | -0.0179 | -2.7344 | 15.67 | 0.63 | 91.85 |
| rotta | S4 | 2029 | COMPLETE | 7.62 | 0.9743 | -0.0172 | -3.8574 | 40.77 | 0.05 | 79.59 |
| lame | S4 | 2029 | COMPLETE | 4.00 | 0.9884 | -0.0031 | -0.2441 | 13.13 | 0.49 | 93.19 |
| frozen | S5 | 2029 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S5 | 2029 | COMPLETE | 12.29 | 0.9430 | 0.0048 | 1.8339 | 20.62 | 6.33 | 86.52 |
| tent | S5 | 2029 | COMPLETE | 12.79 | 0.9295 | -0.0087 | 1.3337 | 16.36 | 10.02 | 86.81 |
| eata | S5 | 2029 | COMPLETE | 12.32 | 0.9431 | 0.0049 | 1.7986 | 20.62 | 6.31 | 86.54 |
| rotta | S5 | 2029 | COMPLETE | 13.62 | 0.9348 | -0.0034 | 0.5039 | 62.90 | 0.60 | 68.25 |
| lame | S5 | 2029 | COMPLETE | 13.64 | 0.9155 | -0.0227 | 0.4772 | 14.24 | 13.34 | 86.21 |
| frozen | S6 | 2029 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S6 | 2029 | COMPLETE | 12.26 | 0.9414 | 0.0032 | 1.8598 | 21.08 | 6.29 | 86.31 |
| tent | S6 | 2029 | COMPLETE | 12.29 | 0.9411 | 0.0029 | 1.8339 | 13.67 | 10.79 | 87.77 |
| eata | S6 | 2029 | COMPLETE | 12.26 | 0.9409 | 0.0027 | 1.8598 | 21.12 | 6.31 | 86.29 |
| rotta | S6 | 2029 | COMPLETE | 14.01 | 0.9177 | -0.0205 | 0.1062 | 66.68 | 0.15 | 66.58 |
| lame | S6 | 2029 | COMPLETE | 13.77 | 0.9158 | -0.0224 | 0.3530 | 14.24 | 13.49 | 86.13 |
| frozen | S1 | 2030 | COMPLETE | 9.86 | 0.9633 | 0.0000 | 0.0000 | 51.31 | 0.44 | 74.13 |
| bn_only | S1 | 2030 | COMPLETE | 12.57 | 0.9362 | -0.0271 | -2.7107 | 35.34 | 1.48 | 81.59 |
| tent | S1 | 2030 | COMPLETE | 12.81 | 0.9344 | -0.0289 | -2.9571 | 35.83 | 1.39 | 81.39 |
| eata | S1 | 2030 | COMPLETE | 12.62 | 0.9356 | -0.0278 | -2.7600 | 35.19 | 1.48 | 81.67 |
| rotta | S1 | 2030 | COMPLETE | 13.90 | 0.9356 | -0.0277 | -4.0414 | 78.81 | 0.26 | 60.47 |
| lame | S1 | 2030 | COMPLETE | 10.45 | 0.9499 | -0.0134 | -0.5914 | 28.24 | 2.44 | 84.66 |
| frozen | S2 | 2030 | COMPLETE | 15.77 | 0.9150 | 0.0000 | 0.0000 | 6.01 | 26.37 | 83.81 |
| bn_only | S2 | 2030 | COMPLETE | 16.80 | 0.9053 | -0.0097 | -1.0254 | 12.94 | 19.58 | 83.74 |
| tent | S2 | 2030 | COMPLETE | 18.90 | 0.8864 | -0.0286 | -3.1250 | 68.80 | 4.79 | 63.21 |
| eata | S2 | 2030 | COMPLETE | 16.85 | 0.9053 | -0.0097 | -1.0742 | 12.70 | 19.68 | 83.81 |
| rotta | S2 | 2030 | COMPLETE | 17.92 | 0.8827 | -0.0323 | -2.1484 | 82.81 | 4.25 | 56.47 |
| lame | S2 | 2030 | COMPLETE | 23.14 | 0.8430 | -0.0720 | -7.3730 | 1.51 | 43.46 | 77.51 |
| frozen | S3 | 2030 | COMPLETE | 6.54 | 0.9747 | 0.0000 | 0.0000 | 26.46 | 0.44 | 86.55 |
| bn_only | S3 | 2030 | COMPLETE | 8.01 | 0.9587 | -0.0161 | -1.4648 | 19.53 | 1.51 | 89.48 |
| tent | S3 | 2030 | COMPLETE | 8.06 | 0.9594 | -0.0153 | -1.5137 | 18.65 | 1.61 | 89.87 |
| eata | S3 | 2030 | COMPLETE | 8.06 | 0.9586 | -0.0162 | -1.5137 | 19.53 | 1.51 | 89.48 |
| rotta | S3 | 2030 | COMPLETE | 8.74 | 0.9539 | -0.0209 | -2.1973 | 42.48 | 0.29 | 78.61 |
| lame | S3 | 2030 | COMPLETE | 6.59 | 0.9741 | -0.0006 | -0.0488 | 13.53 | 1.81 | 92.33 |
| frozen | S4 | 2030 | COMPLETE | 3.76 | 0.9915 | 0.0000 | 0.0000 | 20.80 | 0.20 | 89.50 |
| bn_only | S4 | 2030 | COMPLETE | 6.84 | 0.9695 | -0.0220 | -3.0762 | 15.67 | 0.59 | 91.87 |
| tent | S4 | 2030 | COMPLETE | 6.54 | 0.9712 | -0.0203 | -2.7832 | 14.79 | 0.68 | 92.26 |
| eata | S4 | 2030 | COMPLETE | 6.84 | 0.9691 | -0.0224 | -3.0762 | 15.62 | 0.59 | 91.89 |
| rotta | S4 | 2030 | COMPLETE | 7.18 | 0.9745 | -0.0170 | -3.4180 | 38.92 | 0.10 | 80.49 |
| lame | S4 | 2030 | COMPLETE | 4.35 | 0.9863 | -0.0052 | -0.5859 | 13.28 | 0.54 | 93.09 |
| frozen | S5 | 2030 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S5 | 2030 | COMPLETE | 12.35 | 0.9399 | 0.0017 | 1.7653 | 20.81 | 6.49 | 86.35 |
| tent | S5 | 2030 | COMPLETE | 12.86 | 0.9272 | -0.0110 | 1.2602 | 17.06 | 9.85 | 86.55 |
| eata | S5 | 2030 | COMPLETE | 12.30 | 0.9400 | 0.0018 | 1.8231 | 20.89 | 6.42 | 86.35 |
| rotta | S5 | 2030 | COMPLETE | 13.19 | 0.9389 | 0.0007 | 0.9299 | 66.21 | 0.32 | 66.74 |
| lame | S5 | 2030 | COMPLETE | 13.93 | 0.9149 | -0.0233 | 0.1885 | 14.30 | 13.64 | 86.03 |
| frozen | S6 | 2030 | COMPLETE | 14.12 | 0.9382 | 0.0000 | 0.0000 | 26.09 | 7.65 | 83.13 |
| bn_only | S6 | 2030 | COMPLETE | 12.22 | 0.9400 | 0.0018 | 1.8965 | 21.04 | 6.29 | 86.33 |
| tent | S6 | 2030 | COMPLETE | 12.26 | 0.9386 | 0.0004 | 1.8598 | 13.31 | 11.28 | 87.70 |
| eata | S6 | 2030 | COMPLETE | 12.23 | 0.9395 | 0.0013 | 1.8888 | 21.00 | 6.27 | 86.37 |
| rotta | S6 | 2030 | COMPLETE | 14.93 | 0.9025 | -0.0357 | -0.8125 | 70.72 | 0.43 | 64.43 |
| lame | S6 | 2030 | COMPLETE | 13.75 | 0.9156 | -0.0226 | 0.3667 | 14.07 | 13.49 | 86.22 |

Dynamic pooled values in the inventory are supplementary; compare S5/S6 by segment macro and within-segment windows in `summary_v21.json`.

## Order sensitivity

Mean ± sample std (ddof=1), then min–max. Five orders are arrival-order repeats, not independent target datasets.

| Method | Stream | Orders | EER% mean ± std [min, max] | AUC mean ± std [min, max] |
|---|---|---:|---:|---:|
| frozen | S1 | 5/5 | 9.86 ± 0.00 [9.86, 9.86] | 0.9633 ± 0.0000 [0.9633, 0.9633] |
| frozen | S2 | 5/5 | 15.77 ± 0.00 [15.77, 15.77] | 0.9150 ± 0.0000 [0.9150, 0.9150] |
| frozen | S3 | 5/5 | 6.54 ± 0.00 [6.54, 6.54] | 0.9747 ± 0.0000 [0.9747, 0.9747] |
| frozen | S4 | 5/5 | 3.76 ± 0.00 [3.76, 3.76] | 0.9915 ± 0.0000 [0.9915, 0.9915] |
| frozen | S5 | 5/5 | 8.98 ± 0.00 [8.98, 8.98] | 0.9611 ± 0.0000 [0.9611, 0.9611] |
| frozen | S6 | 5/5 | 8.98 ± 0.00 [8.98, 8.98] | 0.9611 ± 0.0000 [0.9611, 0.9611] |
| bn_only | S1 | 5/5 | 12.46 ± 0.39 [12.01, 13.05] | 0.9378 ± 0.0023 [0.9349, 0.9411] |
| bn_only | S2 | 5/5 | 17.44 ± 0.44 [16.80, 17.97] | 0.9011 ± 0.0026 [0.8983, 0.9053] |
| bn_only | S3 | 5/5 | 8.17 ± 0.23 [7.96, 8.45] | 0.9551 ± 0.0023 [0.9533, 0.9587] |
| bn_only | S4 | 5/5 | 6.77 ± 0.13 [6.54, 6.84] | 0.9707 ± 0.0028 [0.9666, 0.9739] |
| bn_only | S5 | 5/5 | 11.20 ± 0.20 [10.97, 11.51] | 0.9413 ± 0.0018 [0.9402, 0.9445] |
| bn_only | S6 | 5/5 | 11.18 ± 0.18 [10.93, 11.36] | 0.9411 ± 0.0011 [0.9395, 0.9422] |
| tent | S1 | 5/5 | 12.56 ± 0.31 [12.10, 12.88] | 0.9364 ± 0.0028 [0.9330, 0.9400] |
| tent | S2 | 5/5 | 19.10 ± 0.29 [18.70, 19.38] | 0.8835 ± 0.0032 [0.8792, 0.8864] |
| tent | S3 | 5/5 | 8.20 ± 0.47 [7.71, 8.79] | 0.9558 ± 0.0024 [0.9540, 0.9594] |
| tent | S4 | 5/5 | 6.47 ± 0.10 [6.35, 6.59] | 0.9724 ± 0.0031 [0.9680, 0.9763] |
| tent | S5 | 5/5 | 11.16 ± 0.19 [10.88, 11.39] | 0.9416 ± 0.0017 [0.9406, 0.9445] |
| tent | S6 | 5/5 | 10.98 ± 0.19 [10.72, 11.25] | 0.9433 ± 0.0008 [0.9424, 0.9442] |
| eata | S1 | 5/5 | 12.56 ± 0.39 [12.18, 13.21] | 0.9371 ± 0.0023 [0.9342, 0.9402] |
| eata | S2 | 5/5 | 17.44 ± 0.44 [16.85, 17.97] | 0.9011 ± 0.0026 [0.8984, 0.9053] |
| eata | S3 | 5/5 | 8.23 ± 0.25 [8.01, 8.54] | 0.9549 ± 0.0023 [0.9532, 0.9586] |
| eata | S4 | 5/5 | 6.77 ± 0.16 [6.49, 6.88] | 0.9704 ± 0.0028 [0.9664, 0.9736] |
| eata | S5 | 5/5 | 11.25 ± 0.19 [11.06, 11.55] | 0.9410 ± 0.0018 [0.9398, 0.9442] |
| eata | S6 | 5/5 | 11.23 ± 0.21 [10.92, 11.41] | 0.9408 ± 0.0012 [0.9391, 0.9419] |
| rotta | S1 | 5/5 | 14.17 ± 1.46 [12.62, 16.31] | 0.9312 ± 0.0129 [0.9087, 0.9404] |
| rotta | S2 | 5/5 | 17.53 ± 0.39 [17.09, 17.97] | 0.8850 ± 0.0034 [0.8812, 0.8899] |
| rotta | S3 | 5/5 | 9.11 ± 0.45 [8.69, 9.62] | 0.9525 ± 0.0028 [0.9498, 0.9564] |
| rotta | S4 | 5/5 | 7.56 ± 0.32 [7.18, 7.91] | 0.9715 ± 0.0040 [0.9646, 0.9745] |
| rotta | S5 | 5/5 | 11.07 ± 0.62 [10.53, 12.08] | 0.9441 ± 0.0043 [0.9376, 0.9493] |
| rotta | S6 | 5/5 | 10.44 ± 0.22 [10.19, 10.67] | 0.9498 ± 0.0019 [0.9476, 0.9524] |
| lame | S1 | 5/5 | 10.51 ± 0.09 [10.44, 10.65] | 0.9515 ± 0.0013 [0.9499, 0.9531] |
| lame | S2 | 5/5 | 23.51 ± 0.36 [23.14, 23.93] | 0.8442 ± 0.0030 [0.8410, 0.8484] |
| lame | S3 | 5/5 | 6.58 ± 0.15 [6.45, 6.84] | 0.9730 ± 0.0008 [0.9721, 0.9741] |
| lame | S4 | 5/5 | 4.25 ± 0.21 [4.00, 4.49] | 0.9874 ± 0.0011 [0.9861, 0.9885] |
| lame | S5 | 5/5 | 11.15 ± 0.11 [10.97, 11.25] | 0.9390 ± 0.0005 [0.9384, 0.9397] |
| lame | S6 | 5/5 | 11.18 ± 0.08 [11.10, 11.29] | 0.9393 ± 0.0009 [0.9383, 0.9406] |

## Bootstrap uncertainty

Stationary paired 1,000-resample intervals are conditional on the realized online trajectories. No adaptation is rerun inside bootstrap. Per-order intervals are saved in `bootstrap.json`.
A positive-only 95% interval favors the method; a negative-only interval favors Frozen. Intervals crossing zero are inconclusive.

| Method | Domain | ΔAUC CI positive / negative / crosses 0 | EER-gain CI positive / negative / crosses 0 |
|---|---|---:|---:|
| bn_only | ITW | 0 / 5 / 0 | 0 / 5 / 0 |
| bn_only | WaveFake | 0 / 5 / 0 | 0 / 5 / 0 |
| bn_only | LA21 | 0 / 5 / 0 | 0 / 5 / 0 |
| bn_only | DF21 | 0 / 5 / 0 | 0 / 5 / 0 |
| tent | ITW | 0 / 5 / 0 | 0 / 5 / 0 |
| tent | WaveFake | 0 / 5 / 0 | 0 / 5 / 0 |
| tent | LA21 | 0 / 5 / 0 | 0 / 5 / 0 |
| tent | DF21 | 0 / 5 / 0 | 0 / 5 / 0 |
| eata | ITW | 0 / 5 / 0 | 0 / 5 / 0 |
| eata | WaveFake | 0 / 5 / 0 | 0 / 5 / 0 |
| eata | LA21 | 0 / 5 / 0 | 0 / 5 / 0 |
| eata | DF21 | 0 / 5 / 0 | 0 / 5 / 0 |
| rotta | ITW | 0 / 5 / 0 | 0 / 5 / 0 |
| rotta | WaveFake | 0 / 5 / 0 | 0 / 5 / 0 |
| rotta | LA21 | 0 / 5 / 0 | 0 / 5 / 0 |
| rotta | DF21 | 0 / 5 / 0 | 0 / 5 / 0 |
| lame | ITW | 0 / 5 / 0 | 0 / 2 / 3 |
| lame | WaveFake | 0 / 5 / 0 | 0 / 5 / 0 |
| lame | LA21 | 0 / 1 / 4 | 0 / 0 / 5 |
| lame | DF21 | 0 / 5 / 0 | 0 / 3 / 2 |

Across 100 stationary method/domain/order comparisons, AUC intervals are positive-only in 0, negative-only in 96, and cross zero in 4.

## Reproduction

See `PROTOCOL.md`, `ACCESS_BOUNDARY.md`, `config.json`, `streams_locked/`, `results/<run_id>/run_index.csv`, `results/<run_id>/final_audit.json`, `results/<run_id>/source_only/`, each run’s `command.json`, `status.json`, `scores.jsonl`, and the launcher logs. First scores and large LL caches stay on this workstation; compact summary tables and the report are versioned.

## One next research step

Investigate the lame failure on WaveFake (-0.0708 mean paired ΔAUC) on the fixed development subset, with Frozen as the reference, before proposing a new online rule.

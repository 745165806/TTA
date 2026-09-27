# Matched supervised one-step head oracle

**THIS IS SUPERVISED DEVELOPMENT ORACLE DIAGNOSIS. NOT A TTA METHOD. NOT FINAL PERFORMANCE.**

Each B128 update uses the same batch's true selected development labels and is evaluated on that batch. This is an optimistic protocol capacity check, not held-out head fitting.

| Domain | Arm | AUC | EER | ΔAUC | ΔEER | Mean angle (deg) | Mean step norm | Mean abs score movement |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| itw | Frozen | 0.963309 | 0.098571 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 |
| itw | SUP-normalized-0.01 | 0.963325 | 0.098346 | +0.000016 | -0.000224 | 0.563576 | 0.008476 | 0.140530 |
| itw | SUP-normalized-0.03 | 0.963383 | 0.099064 | +0.000074 | +0.000493 | 1.697449 | 0.025428 | 0.421589 |
| itw | SUP-normalized-0.10 | 0.963406 | 0.099217 | +0.000097 | +0.000646 | 5.714295 | 0.084758 | 1.405296 |
| itw | O2-normalized-0.01 | 0.963308 | 0.098346 | -0.000000 | -0.000224 | 0.489137 | 0.008476 | 0.077709 |
| itw | O2-normalized-0.03 | 0.963294 | 0.098346 | -0.000015 | -0.000224 | 1.483629 | 0.025428 | 0.233128 |
| itw | O2-normalized-0.10 | 0.963239 | 0.098571 | -0.000070 | +0.000000 | 5.125888 | 0.084758 | 0.777092 |
| wavefake | Frozen | 0.915003 | 0.157715 | +0.000000 | +0.000000 | 0.000000 | 0.000000 | 0.000000 |
| wavefake | SUP-normalized-0.01 | 0.914314 | 0.157227 | -0.000689 | -0.000488 | 0.560524 | 0.008476 | 0.158668 |
| wavefake | SUP-normalized-0.03 | 0.912740 | 0.159180 | -0.002263 | +0.001465 | 1.689386 | 0.025428 | 0.476005 |
| wavefake | SUP-normalized-0.10 | 0.905041 | 0.169922 | -0.009962 | +0.012207 | 5.700345 | 0.084758 | 1.586684 |
| wavefake | O2-normalized-0.01 | 0.914854 | 0.157227 | -0.000149 | -0.000488 | 0.490222 | 0.008476 | 0.088089 |
| wavefake | O2-normalized-0.03 | 0.914368 | 0.158203 | -0.000635 | +0.000488 | 1.486985 | 0.025428 | 0.264267 |
| wavefake | O2-normalized-0.10 | 0.910994 | 0.160156 | -0.004009 | +0.002441 | 5.136130 | 0.084758 | 0.880891 |

| Domain | δ | SUP−O2 ΔAUC | SUP−O2 ΔEER |
|---|---:|---:|---:|
| itw | 0.01 | +0.000016 | +0.000000 |
| itw | 0.03 | +0.000089 | +0.000717 |
| itw | 0.10 | +0.000167 | +0.000646 |
| wavefake | 0.01 | -0.000540 | +0.000000 |
| wavefake | 0.03 | -0.001628 | +0.000977 |
| wavefake | 0.10 | -0.005953 | +0.009766 |

Decision: **SINGLE_STEP_HEAD_PROTOCOL_INSUFFICIENT**. Paired 1,000-draw development intervals, including direct same-δ SUP versus O2 contrasts, are in `bootstrap.csv`.

Implementation check: every supervised candidate lowers its own batch BCE versus Frozen in all 25 ITW and 32 WaveFake buffers. The loss decreases while pooled WaveFake AUC worsens, so the negative ranking result is not explained by a missing gradient step. This check does not isolate whether cross-buffer score calibration or another part of the one-step protocol causes the ranking loss.

Bootstrap intervals resample the already fitted score pairs; they do not refit a supervised head inside each resample. They describe development score uncertainty conditional on this fixed oracle execution, not independent generalization uncertainty.

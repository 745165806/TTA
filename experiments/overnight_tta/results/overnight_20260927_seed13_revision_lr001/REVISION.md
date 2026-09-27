# Development-informed revision

Primary seed13 source training remains at `overnight_20260927_seed13`; source checkpoints are read by absolute path. The first target scoring at 1e-4 was retained. Source-select pseudo-domains tied at AUC=1/EER=0 for all three predeclared rates. This single revision uses 1e-3, the upper predeclared rate, to test whether the near-zero ranking change at 1e-4 reflected too-small adaptation. Source thresholds are recalibrated on source select. The choice used ITW/WaveFake development feedback and is not independent selection.

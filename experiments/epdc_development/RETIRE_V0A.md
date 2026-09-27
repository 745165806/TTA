# Decision-order v0-A retired — 2026-09-27

`v0a_20260927T041709Z` finished scores and post-score audit on the fixed 512/282/370 mechanism samples. The paired cross-class source-order loss was approximately zero (macro mean `9.60e-8` for Base, `9.53e-8` for Preserve). It therefore did not control normalized source-anchor margin damage (`0.247588` vs `0.247582` macro). On In-the-Wild, Base and Preserve had identical EER `0.1039603960` and AUC `0.9587831364`; Frozen was EER `0.0990099010`, AUC `0.9578409454`. ASV LA/DF were single-class, so EER/AUC remain undefined there.

This preservation module is **not retained as an active EPDC component**. Its source and result remain in the research branch solely to reproduce the negative experiment and explain the next design. No reliability gate, drift controller, or continual state is built on it. The next prototype replaces—not stacks on top of—this loss with a normalized source-anchor margin deficit tied to the source threshold.

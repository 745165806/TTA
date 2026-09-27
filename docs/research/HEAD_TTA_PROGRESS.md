# Head-adaptation research progress — 2026-09-27

Branch: `exp-head-tta`. The supervised capacity audit showed a target-domain
linear-readout gap, so this branch first isolated head geometry, then measured
the production R adapter's effective-head space, and finally tested one
predeclared label-free target-head prototype. The [append-only research ledger](RESEARCH_LEDGER.md)
and [decision log](DECISION_LOG.md) contain the experiment-level interpretation.

| Stage | Result | Decision |
|---|---|---|
| Five-fold supervised head geometry | ITW Frozen/H2/H3 AUC 0.963309/0.967247/0.973159; WaveFake 0.915003/0.934795/0.951711. H2 improves all five folds but recovers only 40%/54% of the H3 AUC gain. | H2≈H3 criterion not met; supervised H3 remains a development upper bound. |
| Production R expressibility | Direct/derived score parity maximum 1.907e-6. R span explains 17.26%/17.14% of normalized supervised head displacement; radius-0.1 reachable fraction 3.38%/3.31%. | `R_PARAMETERIZATION_MISMATCH` confirmed for these target corrections. |
| Label-free H-UA1 B256 | ITW best ΔAUC +0.000204 with EER worsening; WaveFake best ΔAUC +0.003375. All alpha arms have source-threshold balanced accuracy 0.5. | `HEAD_ADAPTATION_NOT_YET_ACTIONABLE`; candidate NONE, prototype stopped. |

The complete small research package is in the
[head-geometry summary](../../experiments/head_capacity_geometry/summary/head_geometry_20260927a/)
and [H-UA1 summary](../../experiments/head_tta/summary/hua1_development_20260927a/).
Those folders include reports, per-fold or per-buffer metrics,
fixed-fold assignments, run configurations, H-UA1 provenance, score-completion
records, and a [verification log](../../experiments/head_tta/summary/hua1_development_20260927a/VERIFICATION_LOG.md).
Mathematical contracts are in the corresponding experiment directories.
Per-sample scores, feature caches, source checkpoint, and large runtime output
remain in their original ignored workstation result directories.

The target90 labels and metrics were not opened in this branch, and no final
held-out metric or method lock was produced. The earlier capacity branch's
strict target90 *feature-byte* access correction remains in force: its old
reader loaded full cache chunks before selecting target10. The new head worker
loads selected feature values only, although it scans cache ID metadata for
coverage. Local WaveFake rows outside the fixed development set were already
metadata/header-audited and are not an untouched final holdout.

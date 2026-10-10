# H-UA1 selected-development audit

Supervised Linear is a development upper bound, not a TTA result.

| Domain | Arm | AUC | EER | ΔAUC | EER improvement | AUC recovery |
|---|---|---:|---:|---:|---:|---:|
| itw | Frozen | 0.963309 | 0.098571 | +0.000000 | +0.000000 | NA |
| itw | H-UA1_alpha_0.25 | 0.963513 | 0.099556 | +0.000204 | -0.000986 | 0.020684549730010632 |
| itw | H-UA1_alpha_0.5 | 0.963347 | 0.100049 | +0.000039 | -0.001479 | 0.003919177843573013 |
| itw | H-UA1_alpha_0.75 | 0.962031 | 0.100087 | -0.001278 | -0.001516 | NA |
| wavefake | Frozen | 0.915003 | 0.157715 | +0.000000 | +0.000000 | NA |
| wavefake | H-UA1_alpha_0.25 | 0.918378 | 0.151367 | +0.003375 | +0.006348 | 0.09193767333710047 |
| wavefake | H-UA1_alpha_0.5 | 0.918357 | 0.157715 | +0.003354 | +0.000000 | 0.09137909757539149 |
| wavefake | H-UA1_alpha_0.75 | 0.911802 | 0.160645 | -0.003201 | -0.002930 | NA |

Decision: **HEAD_ADAPTATION_NOT_YET_ACTIONABLE**; candidate: NONE.

The label-free worker completed 3,178 ITW and 4,096 WaveFake scores in 13 and
16 independent buffers, respectively. Both domains had exact selected-ID
coverage, finite scores, zero reported numeric failures, and zero Frozen-score
parity difference. The preceding 32-row/domain smoke passed without loading
development labels. Labels were opened only by the separate analyzer after
both complete score files passed coverage validation.

The largest H-UA1 ITW AUC gain is +0.000204 at alpha 0.25 (2.07% of the
supervised linear AUC gap), while EER worsens by 0.000986. WaveFake's largest
AUC gain is +0.003375 at alpha 0.25 (9.19% of that gap), with EER improving by
0.006348. Neither reaches the predeclared domain threshold. Alpha 0.75 lowers
AUC in both domains. All three alphas predict every development sample as spoof
at the fixed source threshold, yielding balanced accuracy 0.5. This is a
calibration failure in addition to inadequate ranking correction.

Buffer diagnostic means: estimated unlabeled spoof prior 0.4763 ITW / 0.3651
WaveFake; target-head angle to source 79.90° / 79.11°; target-head bias 1.2004
/ 1.8863 versus source bias 0.1090. Mean absolute alpha-0.5 score movement
is 1.3128 / 1.2804, so the non-result is not a frozen or inactive estimator.
The post-hoc WaveFake development class fraction is 0.5, unlike the soft prior
estimate; no true class ratio was used in adaptation. These diagnostics locate
a possible source-affinity/calibration weakness but do not authorize changing
temperature, covariance, prior, alpha, or buffer size after seeing labels.

The supervised linear probe remains a held-out, label-using development upper
bound. It is not an unsupervised TTA result. H-UA1 is retired under the fixed
promotion rule. Target90 labels/metrics and final held-out metrics were not read.

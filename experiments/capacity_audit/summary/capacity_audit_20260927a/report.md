# Representation sufficiency and bounded-adapter capacity audit

**Decision: `R_PARAMETERIZATION_BOTTLENECK` for the present frozen classifier plus radius-0.1 8×8 R.** This is a supervised development diagnosis, not an unsupervised TTA method, a final benchmark, or a claim about every possible adapter. The Local-TTA mainline remains `CLOSED`, candidate `NONE`, and no method lock or target90/final metric was used.

## Held-out capacity ladder

Each primary domain used five fixed stratified outer folds; WaveFake kept each real/generated `audio_id` pair together in both outer and inner splits. All headline scores below pool predictions from the fold that never trained on that row. Within-fold directions were checked separately. C1 trains only R with frozen U/w/b and `||R||_F≤0.1`; C2 changes only the linear readout of original-view 160D frozen features; C3 is a 160→32→1 probe.

| Primary development domain | C0 Frozen AUC/EER | C1 R AUC/EER | C2 linear AUC/EER | C3 nonlinear AUC/EER |
|---|---:|---:|---:|---:|
| ITW target10, 3178 | 0.963309 / 0.098571 | 0.963385 / 0.099064 | 0.973159 / 0.085292 | 0.974185 / 0.082681 |
| WaveFake paired development, 4096 | 0.915003 / 0.157715 | 0.912394 / 0.161621 | 0.951711 / 0.114258 | 0.952616 / 0.113281 |

ITW C1 has ΔAUC `+0.000076` and EER worsens `0.000493`; WaveFake C1 has ΔAUC `−0.002609` and EER worsens `0.003906`. Both are below the preregistered actionable cutoff. The linear C2 **representation gap** is ITW ΔAUC `+0.009850` (MODERATE by AUC) and EER improvement `+0.013279` (ACTIONABLE); WaveFake ΔAUC `+0.036708` and EER improvement `+0.043457` (ACTIONABLE). C2 improves AUC and EER over Frozen in all five held-out folds in both domains. C3 adds only `+0.001026/+0.000906` AUC beyond C2 on ITW/WaveFake, so this evidence does not require a nonlinear readout.

## C1 optimization and geometry check

C1 Adam reaches `||R||_F≈0.1` in every fold. Since its score change is algebraically `Δs=(zU)·q` with `q=Rᵀ(Uᵀw)` and `||q||≤0.1||Uᵀw||`, the same bounded score family is an eight-dimensional convex logistic problem. A separately preregistered SciPy SLSQP solve on each outer training fold converged and remained on the same radius. Its held-out ITW ΔAUC was `+0.000027` with EER change `0`; WaveFake ΔAUC was `−0.006956` and EER worsened `0.009277`. Thus the original C1 non-result is not explained by its Adam early-stop schedule. This checks the BCE training objective; it is not an exhaustive upper bound for every possible supervised ranking objective. The current *bounded* R plus fixed classifier is the supported limitation, without isolating radius from subspace as separate causes.

## Cross-domain and source-artifact limits

The supervised C2 head direction did not transfer: ITW-trained linear probe on WaveFake AUC/EER `0.903675/0.172363` versus WaveFake Frozen `0.915003/0.157715`; WaveFake-trained probe on ITW `0.960763/0.100049` versus ITW Frozen `0.963309/0.098571`. Both are **development cross-domain diagnostics**, not final evaluation. The recoverable readout appears domain dependent; a source-domain head swap is not justified by this result.

WaveFake's 13,100 local content IDs have complete R/WF1–WF7 rows, and all 104,800 source WAVs are mono 22,050-Hz PCM16, so class is not separated by rate/codec. A few-millisecond real/generated duration offset remains, with no independent transcript/speaker/language metadata. As a post-hoc check, among 1,722 content pairs where both files are longer than the production 64,600-sample crop, WaveFake C2 AUC/EER is `0.961800/0.099303` versus Frozen `0.923785/0.152149`; the linear gap is not confined to short clips that require waveform tiling. This does not rule out other dataset-specific artifacts. Codecfake 5000 remains `RATE_CLASS_CONFOUNDED`; ASVspoof2019 LA is `NEAR_CEILING`, and neither is used to establish this two-domain conclusion.

## Scientific decision

The current frozen 160D features contain recoverable development-domain spoof-discriminative information for a supervised linear readout; the present frozen head plus bounded 8×8 R does not expose an actionable correction in the same held-out tests. This is **Case B** under the preregistered interpretation. The next method research direction, after this audit, is to study whether label-free target signals can support *controlled classifier/head or more expressive low-rank adaptation*, benchmarked against these supervised upper bounds. No new TTA method is implemented or claimed here. A universal shared spoof direction and final held-out improvement remain unproven.

Source files: `summary/itw_capacity_20260927a/`, `summary/wavefake_capacity_20260927a/`, `summary/c1_convex_20260927a/`, and `summary/crossdomain_20260927a/`. Full folds, logs and never-trained-on predictions remain in their unique workstation `results/` runs.

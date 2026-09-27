# O1 — Source-anchor soft affinity, v0

For frozen source anchors `a_i` with source labels `y_i ∈ {0,1}`, frozen detector score `s(z)=wᵀz+b`, and source subspace `U`, define `g(z)=[zU / σ_U, s(z)/σ_s]`. `σ_U` is the median nonzero pairwise source-anchor distance in `U`; `σ_s` is the source-anchor score standard deviation. Both are fixed from source data only. For adapted target view `z_v(R)`, the class log affinity is

`A_c(v,R) = logmeanexp_{i:y_i=c}[-||g(z_v(R))-g(a_i)||² / (2T)]`.

`p_v(R)=softmax(A_0,A_1)` is continuous; `p̄(R)=mean_v p_v(R)`. The loss is

`L_O1(R)=mean_v KL(stopgrad(p̄(R)) || p_v(R)) + 0.1 H(p̄(R))`.

The first term aligns safe views in source-labelled class-affinity space. The small entropy term encourages a decisive *soft* source-affinity consensus without threshold `tau0`, hard pseudo-labels, or target labels. The source anchors and their labels never change. `T=1` is fixed. This objective may still amplify incorrect source affinities; that is an empirical risk to assess after all scores are generated. Score direction remains larger = spoof.

Common experiment space: fresh 8×8 `R=0` per sample, 5 projected SGD steps, `lr=0.03`, Frobenius `rho=0.1`, original-view score, production three-view feature cache, frozen detector and anchors. No guard or preservation is added at this stage. No scale sweep is planned.

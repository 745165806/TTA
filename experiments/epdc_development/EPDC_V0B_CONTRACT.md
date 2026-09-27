# EPDC-TTA v0-B: normalized threshold-margin evidence

This replaces the retired paired-order evidence term. Same source model, three-view feature cache, fresh episodic `R=0`, view-variance Base Adapt (`ep_no_keep`), SGD, Frobenius projection, original-view score, and fixed B setting: K=5, lr=0.03, rho=0.1. No hard guard, target label, target attack, or cross-sample state enters the update. `lambda_preserve=0` is exactly the production Base Adapt path.

For source anchor `j`, fixed source score `s_j^0`, label `y_j`, source threshold `tau0`, and positive margin `m_j^0=(2y_j-1)(s_j^0-tau0)`, define adapted margin `m_j(R)=(2y_j-1)(s_j(R)-tau0)` and

\[
L_{evidence}(R)=\frac{1}{M}\sum_j\left[
\frac{\max(0,\eta m_j^0-m_j(R))}{m_j^0}\right]^2,
\qquad L=V(Z_x(R))+\lambda_{preserve}L_{evidence}(R).
\]

Initial fixed test: `eta=0.9`, `lambda_preserve=1.0`. The normalization emphasizes relative damage to low-margin anchors, the source evidence most vulnerable to crossing the decision threshold. Individual features and logits may change; the loss softly penalizes threshold-relevant margin destruction. It is distinct from the old unnormalized margin regularizer and from the retired paired-order loss. No grid search is performed.

This is a development prototype. The only active comparison is Frozen → Base Adapt → Base + Preserve. A reliability gate and continual drift control remain deferred until the preservation contrast shows task-useful correction and lower damage. The first gate is a 32-sample/domain label-free smoke, followed by the unchanged mechanism sandbox.

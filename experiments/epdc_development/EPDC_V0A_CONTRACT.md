# EPDC-TTA v0-A: soft decision-order evidence

This is a **prototype**, not a locked method. First comparison: Frozen, Base Adapt (`ep_no_keep`), Base + Preserve. All use the same three-view cached features, frozen source checkpoint, source-only anchors and head, original-view score, per-sample fresh `R=0`, SGD, and Frobenius projection. Initial predeclared setting is B: K=5, lr=0.03, rho=0.1. Base loss is production `view_loss` with `lambda_keep=0` and no hard source-margin guard.

Sort the 128 source bonafide anchors from highest (hardest) source score down, and the 128 source spoof anchors from lowest (hardest) source score up. Pair equal ranks. With source score gap `g_i^0=s_{1,i}^0-s_{0,i}^0>0` and adapted gap `g_i(R)=s_{1,i}(R)-s_{0,i}(R)`, define

\[
L_{evidence}(R)=\frac{1}{128}\sum_{i=1}^{128}
\left[\frac{\max(0,\eta g_i^0-g_i(R))}{g_i^0}\right]^2,
\qquad L=V(Z_x(R))+\lambda_{preserve}L_{evidence}(R).
\]

The first test fixes `eta=0.9`, `lambda_preserve=1.0`; no grid search. This penalizes source cross-class decision-order compression gradually. It does not require an individual feature or score to equal Frozen. The pairs, labels, and gaps come solely from the trained source anchor memory. The target input supplies only three unlabeled feature views; no target label, attack, group, or path is read by the optimizer.

`lambda_preserve=0` dispatches the exact production `ep_no_keep` Base Adapt implementation. `lambda_preserve>0` adds only the equation above to the same optimizer/projection/reset geometry. The controller, reliability gate, drift policy, and continual accumulation are **not implemented** in v0-A. After 32 real samples on every cached sandbox domain, a fixed mechanism-dev comparison will decide whether to keep this preservation term. A gain in target view consistency alone is insufficient; ranking and harmful drift must be checked where both labels exist.

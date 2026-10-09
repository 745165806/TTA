# O2 — Decision-sensitive multi-view consistency, v0

For each safe view `v` of the same target audio, compute the adapted detector logit `s_v(R)=wᵀz_v(R)+b`. Let `σ_s` be the frozen *source-anchor* score standard deviation. The loss is

`L_O2(R)=mean_v [(s_v(R)-mean_j s_j(R))/σ_s]²`.

Only the detector decision direction is constrained. The sign and class of the target are never supplied. This differs from production Base Adapt, which minimizes variance across all embedding coordinates. The score scale is source-only and fixed. A sample whose safe views have the same score has zero O2 gradient; this is a meaningful limitation, not a reason to add a target pseudo-label.

Common experiment space and fixed budget are exactly as in `OBJECTIVE_O1.md`. No guard, preservation, or reliability gate is present.

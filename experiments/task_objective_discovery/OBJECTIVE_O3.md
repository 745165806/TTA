# O3 — Reliability-weighted soft affinity, v0

Compute O1 frozen (`R=0`) class-affinity probabilities `p_v^0`. Let `agreement=clamp(1-std_v(p_{v,1}^0),0,1)` and `gap=|mean_v(p_{v,1}^0-p_{v,0}^0)|`. Define a continuous, detached source-affinity weight

`q=sigmoid(4(agreement-0.5)) × sigmoid(4(gap-0.5))`.

The adapted objective is

`L_O3(R)=q L_O1(R) + L_O2(R)`.

Thus both high multi-view agreement and high source-affinity separation increase the task-directed O1 term. Low-reliability samples still receive decision-sensitive consistency. `q` uses only frozen target views and source-labelled anchor geometry; it never uses target audit labels, correctness, or a hard pseudo-label. O3 is objective weighting, not an accept/reject deployment gate. The fixed budget and feature path match O1/O2.

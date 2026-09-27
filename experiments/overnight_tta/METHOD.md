# Overnight cached-feature method contract (fixed before target scoring)

Start: 2026-09-27 14:56 UTC. Stop starting new experiments after 22:56 UTC. GPU check: `nvidia-smi` failed to contact the driver; `tta` torch 2.1.0 reports no CUDA device. Waveform/earlier-layer route B is `BLOCKED_RESOURCE`; do not substitute final 160D features for it.

## Route A: source-trained residual feature adapter

The project-trained SSL-AASIST encoder stays frozen. On each 160D view, use `h(z)=z+W2 GELU(W1 z)` with `160→32→160`; `W2` begins at zero. A linear spoof-positive head begins at the audited source head and is trained with the adapter on source fit. For each balanced source batch, the same original and existing noise view are used for both classes. Source objective: mean BCE on both views plus `0.05 ×` supervised contrastive loss on their normalized adapted embeddings, temperature `0.1`. The matched static linear ERM sees the same batches, views, number of optimizer steps and BCE supervision. Train both for 10 full passes through the spoof fit examples; repeat bonafide examples by balanced deterministic sampling, and record presentations separately from unique IDs. Select checkpoints using source select clean EER, then perturbed-view EER and clean AUC as tie breakers. Report each view's EER/AUC.

At target time, freeze the trained head and update only the residual adapter. Source-fit class prototypes in adapted 160D space are retained; spoof prototypes remain separated by official A01–A06 attack family. For each target original embedding, obtain a **detached** pseudo-class confidence from the source-only checkpoint. High-confidence target examples (`p≥0.9` spoof, `p≤0.1` bonafide) are matched to their corresponding source class bank: bonafide mean or nearest of six spoof means. Missing high-confidence classes are skipped, not filled using target labels or assumed proportions. The loss is

`L = 0.1 L_anchor + 0.05 L_view + 0.01 L_move`,

where `L_anchor` is mean squared normalized 160D distance of selected original-view adapted embeddings to detached source prototype; `L_view` is mean squared difference between original and two existing perturbation-view logits; `L_move` is mean squared parameter displacement from the source-trained adapter, normalized by source parameter scale. These coefficients are fixed before target scoring, not chosen from target results. Minibatch SGD runs two complete unlabeled target passes, and **the final shared adapter** scores every domain sample. Each domain restarts from its own source checkpoint; no state crosses domains. Only target learning rate differs among first-round configurations `{1e-4, 3e-4, 1e-3}`. Choose it on source-select pseudo-domains before target scoring. Source fit examples and attack labels are retained only as a six-prototype resource; target adaptation receives neither target labels nor target attack IDs.

The mechanism differs from earlier 8×8 R/view-variance objectives: a task-trained nonlinear 160D residual map, supervised multi-view class structure, and class-conditioned source prototypes replace the generic view-variance direction. It is TTAC-inspired, not an implementation of TTAC or evidence of the paper's reported performance.

## T3A-batch port

Use the exact two native `out_layer` rows (`spoof=0`, `bonafide=1`) from the same frozen detector, verified against the exported spoof-minus-bonafide head. Follow the [T3A author implementation](https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py): initialize templates from classifier rows, append unlabeled target embeddings with native argmax pseudo-labels and entropy, retain the lowest-entropy `K=100` templates per class, normalize template supports and class weights, then score the full domain. This offline template estimate and rescore is named **T3A-batch port**; it is not the author's online prediction protocol. `K` remains fixed; no target-label tuning.

## Selection and promotion

The first route A run is seed 13, at most three target learning rates. Replicate seeds 29/47/71 only if one domain improves by at least 0.005 AUC or 0.005 EER without equal deterioration in the other, and the adapted checkpoint improves on its own source-only score. All target development comparisons are explicitly development selection. Final labels and target90 stay closed. WaveFake uncertainty resamples the saved `audio_id` content pair; ITW uses audio IDs because no stronger reliable group accompanies target10.

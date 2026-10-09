# Task objective v0: fixed In-the-Wild plus auxiliary PA dev

Run: `results/objective_two_domain_20260927a/` (`d723f8c` objective-code commit at score generation). Both domains used the same frozen SSL-AASIST source, production three-view caches, original-view score, episodic 8×8 R, K=5, lr=0.03, rho=0.1, and projected SGD. All 3,910 score rows passed exact coverage and finite checks before selected audit labels were read. The PA cache is `results/pa_cache_270_20260927a/`, with 270/270 finite 3×160 features and exact ID coverage.

| Domain | Class counts | Arm | EER | AUC | Mean update norm | Mean source damage | Helpful / harmful flips |
|---|---:|---|---:|---:|---:|---:|---:|
| In-the-Wild | 310 bona / 202 spoof | Frozen | 0.099010 | 0.957841 | 0 | 0 | 0 / 0 |
| In-the-Wild | 310 / 202 | Base `ep_no_keep` | 0.103960 | 0.958783 | 0.011714 | 0.170890 | 0 / 1 |
| In-the-Wild | 310 / 202 | O1 | 0.103960 | 0.958799 | 0.016058 | 0.159813 | 23 / 0 |
| In-the-Wild | 310 / 202 | O2 | 0.103960 | 0.958895 | 0.011529 | 0.199754 | 0 / 1 |
| In-the-Wild | 310 / 202 | O3 | 0.103960 | 0.958879 | 0.016713 | 0.218358 | 16 / 1 |
| ASVspoof2019 PA dev | 270 bona / 0 spoof | Frozen | undefined | undefined | 0 | 0 | 0 / 0 |
| ASVspoof2019 PA dev | 270 / 0 | Base `ep_no_keep` | undefined | undefined | 0.011743 | 0.193014 | 0 / 16 |
| ASVspoof2019 PA dev | 270 / 0 | O1 | undefined | undefined | 0.010731 | 0.041150 | 7 / 3 |
| ASVspoof2019 PA dev | 270 / 0 | O2 | undefined | undefined | 0.010221 | 0.236152 | 0 / 23 |
| ASVspoof2019 PA dev | 270 / 0 | O3 | undefined | undefined | 0.012116 | 0.192282 | 1 / 14 |

On In-the-Wild, O1 moves bonafide scores down by `−0.073774` on average and spoof scores up by `+0.044334`, increasing the *mean* class-score gap by `0.118108`. Yet EER worsens and paired bootstrap ΔAUC 95% development interval `[-0.000415, +0.002891]` includes zero. O1 objective reduction correlates weakly with signed class-correct score movement (Spearman `0.2385`). O2's mean class-gap movement is negative (`−0.049100`) and it causes 23 harmful threshold flips on the single-class PA group. The PA single-class result cannot validate ranking and does not satisfy objective promotion.

The fixed PA group was selected from official dev by speaker and audio availability before selected labels were read. It must remain fixed. Codecfake 512 requires non-production sample-rate handling, WaveFake lacks a reader in `tta`, and ASVspoof2021 LA/DF saved mechanism groups are single-class. **No Base Objective is promoted; no preservation, gate, continual control, or method lock follows from this study.** The remaining need is a genuinely independent two-class development resource compatible with the unchanged Frozen feature path, established without changing any existing assignments or using final data.

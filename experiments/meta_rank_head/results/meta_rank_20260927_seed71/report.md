# Meta-Rank Head-TTT — 首轮开发报告

协议：离线 batch-transductive；每个目标域从项目自训练 SSL-AASIST source head 重置，用该域无标签 160D 三视图特征更新完整线性 head 五步。目标标签仅用于保存分数后的开发评价。

| 域 | 方法 | AUC | EER | ΔAUC vs Frozen | ΔAUC vs ERM |
|---|---|---:|---:|---:|---:|
| itw_target10 | frozen | 0.963309 | 0.098571 | +0.000000 | — |
| itw_target10 | erm | 0.963516 | 0.098346 | +0.000207 | +0.000000 |
| itw_target10 | meta_bce | 0.963328 | 0.098346 | +0.000019 | -0.000188 |
| itw_target10 | meta_rank | 0.963301 | 0.098346 | -0.000007 | -0.000214 |
| wavefake_dev | frozen | 0.915003 | 0.157715 | +0.000000 | — |
| wavefake_dev | erm | 0.917494 | 0.155273 | +0.002491 | +0.000000 |
| wavefake_dev | meta_bce | 0.912706 | 0.160156 | -0.002297 | -0.004788 |
| wavefake_dev | meta_rank | 0.912947 | 0.160645 | -0.002056 | -0.004547 |

阈值仅由 source select 设定；表中 AUC/EER 与阈值无关。WaveFake 置信区间按 audio_id content-pair 重采样，ITW 按样本重采样；见 metrics.csv。

本轮只有 seed 71；三 seed 稳定性尚未验证。target90 与最终 holdout 未运行。

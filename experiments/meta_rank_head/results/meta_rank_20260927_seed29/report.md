# Meta-Rank Head-TTT — 首轮开发报告

协议：离线 batch-transductive；每个目标域从项目自训练 SSL-AASIST source head 重置，用该域无标签 160D 三视图特征更新完整线性 head 五步。目标标签仅用于保存分数后的开发评价。

| 域 | 方法 | AUC | EER | ΔAUC vs Frozen | ΔAUC vs ERM |
|---|---|---:|---:|---:|---:|
| itw_target10 | frozen | 0.963309 | 0.098571 | +0.000000 | — |
| itw_target10 | erm | 0.963511 | 0.098346 | +0.000202 | +0.000000 |
| itw_target10 | meta_bce | 0.963265 | 0.098346 | -0.000044 | -0.000246 |
| itw_target10 | meta_rank | 0.963244 | 0.098346 | -0.000065 | -0.000268 |
| wavefake_dev | frozen | 0.915003 | 0.157715 | +0.000000 | — |
| wavefake_dev | erm | 0.917603 | 0.155762 | +0.002600 | +0.000000 |
| wavefake_dev | meta_bce | 0.908999 | 0.166016 | -0.006004 | -0.008604 |
| wavefake_dev | meta_rank | 0.907632 | 0.168457 | -0.007371 | -0.009971 |

阈值仅由 source select 设定；表中 AUC/EER 与阈值无关。WaveFake 置信区间按 audio_id content-pair 重采样，ITW 按样本重采样；见 metrics.csv。

本轮只有 seed 29；三 seed 稳定性尚未验证。target90 与最终 holdout 未运行。

# Distribution-Conditioned Target Head

协议：离线 batch-transductive；冻结项目自训练 SSL-AASIST encoder。每域无标签原视图 160D 特征均值和标准差形成 320D 描述符，MLP 一次生成共享线性 head；测试时无梯度或优化器。

源伪域：官方 ASVspoof2019 LA CM 协议的 A01–A06 spoof attack，各配随机 bonafide；每个 episode 随机选择同一三视图条件，support/query 各 32+32 且原始音频 ID 不重叠。共 6 个攻击家族，18 个攻击×视图条件。

20 epochs、96 episode、Adam 1e-3、regularizer 0.01、hidden 128；ERM 与 DCH 共享 episode 日程及 64 个有标签 query/episode，均只用 source select 选 checkpoint 和工作阈值。

| 域 | 方法 | AUC mean (range) | EER mean (range) |
|---|---|---:|---:|
| itw_target10 | frozen | 0.963309 (0.963309–0.963309) | 0.098571 (0.098571–0.098571) |
| itw_target10 | erm | 0.963539 (0.963508–0.963563) | 0.098402 (0.098346–0.098571) |
| itw_target10 | dch | 0.963325 (0.963029–0.963455) | 0.098838 (0.098346–0.099217) |
| wavefake_dev | frozen | 0.915003 (0.915003–0.915003) | 0.157715 (0.157715–0.157715) |
| wavefake_dev | erm | 0.921766 (0.920582–0.922981) | 0.152344 (0.150391–0.154785) |
| wavefake_dev | dch | 0.924103 (0.921548–0.925744) | 0.149414 (0.147949–0.152344) |

## 每 seed DCH

| 域 | seed | AUC | EER | ΔAUC vs Frozen | ΔAUC vs ERM | EER gain vs Frozen | EER gain vs ERM | ||delta_w|| | delta_b | angle (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| itw_target10 | 13 | 0.963029 | 0.099217 | -0.000280 | -0.000480 | -0.000646 | -0.000870 | 0.244867 | -0.012277 | 14.9080 |
| itw_target10 | 29 | 0.963454 | 0.098346 | +0.000145 | -0.000109 | +0.000224 | +0.000224 | 0.189437 | -0.004064 | 9.8100 |
| itw_target10 | 47 | 0.963361 | 0.098571 | +0.000052 | -0.000171 | +0.000000 | -0.000224 | 0.282735 | +0.004354 | 15.2368 |
| itw_target10 | 71 | 0.963455 | 0.099217 | +0.000146 | -0.000097 | -0.000646 | -0.000870 | 0.307726 | +0.003717 | 15.4575 |
| wavefake_dev | 13 | 0.925744 | 0.147949 | +0.010741 | +0.004114 | +0.009766 | +0.003906 | 0.237496 | -0.011686 | 14.4936 |
| wavefake_dev | 29 | 0.921548 | 0.152344 | +0.006545 | +0.000966 | +0.005371 | +0.002441 | 0.194919 | -0.004257 | 10.0639 |
| wavefake_dev | 47 | 0.924624 | 0.148438 | +0.009622 | +0.002755 | +0.009277 | +0.003906 | 0.236164 | +0.006168 | 13.1388 |
| wavefake_dev | 71 | 0.924496 | 0.148926 | +0.009493 | +0.001514 | +0.008789 | +0.001465 | 0.317153 | +0.005292 | 15.8695 |

## 目标 head 位移（四 seed 均值）

| 域 | ||delta_w|| | delta_b | angle (deg) | DCH ΔAUC vs Frozen | DCH EER gain vs Frozen | DCH ΔAUC vs ERM | DCH EER gain vs ERM |
|---|---:|---:|---:|---:|---:|---:|---:|
| itw_target10 | 0.256191 | -0.002067 | 13.8531 | +0.000016 | -0.000267 | -0.000214 | -0.000435 |
| wavefake_dev | 0.246433 | -0.001121 | 13.3914 | +0.009100 | +0.008301 | +0.002337 | +0.002930 |

WaveFake 成对 bootstrap 以 audio_id content-pair 为单位；ITW 以原始音频 ID 为单位。逐 seed 区间在 metrics.csv。目标标签只在分数落盘后用于开发评价。关闭 delta 与 Frozen 分数完全一致。

已有 Meta-Rank 四 seed 结果见 `experiments/meta_rank_head/report.md`；本轮未重训该方法。

Decision: **DISTRIBUTION_CONDITIONING_NOT_ACTIONABLE**

仅完成 ITW target10 与 WaveFake development；target90 labels/metrics accessed = NO；final holdout accessed = NO。

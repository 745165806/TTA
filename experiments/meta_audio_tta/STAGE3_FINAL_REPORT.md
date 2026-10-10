# SSL-AASIST 逐样本音频 TTA：阶段 3 四域开发评价最终报告

日期：2026-10-09。状态：**完成，disclosed-access target development evaluation**。本报告仅对冻结版本和固定四域开发清单成立；没有进行新训练、算法修改、学习率搜索、目标重评分，亦未访问 target90 或 final holdout。

## 1. 冻结身份与访问顺序

- 分支 `exp/meta-audio-tta`；评分时 Git HEAD 始终是 `fae1e70da8ece85f63e8b9aa92d7f53c337dbc9f`。
- 原始 [freeze.json](runs/stage3_20261009_freeze/freeze.json) SHA-256 `befec35093223b9ebbc4fd5df5858d4f8e79dfc1be42dcae239c59c502706c9a`；含四个 checkpoint、36 个输入文件、四域无标签 ID、源 `select`/`cal0` 记录及评价代码哈希。原始 [冻结报告](runs/stage3_20261009_freeze/FREEZE_REPORT.md) 保留。
- [score_set_seal.json](runs/stage3_20261009_freeze/score_set_seal.json) SHA-256 `3eb3f7069a2717d1b1c32519b54be5c9547cd49827f74977ffecb27f8de0d588`，状态 `ALL_EXPECTED_SCORES_COMPLETE`。
- 评分前只读[复核](runs/stage3_20261009_freeze/execution_20261009/preflight.log)退出 **0 / PASS**：Git HEAD、freeze 哈希、36/36 文件、四个 checkpoint、source `select`/`cal0` 与 16 个尚不存在的目标分数路径均匹配。
- 三张卡 GPU 0/1/3 分别运行一个评分进程；启动时各有约 48 GiB 空闲显存，GPU 2 留给原桌面任务。磁盘空闲约 601 GiB，内存可用约 80 GiB；监控未发现严重争抢。16 条评分命令均退出 **0**，每项的原始命令、GPU、退出码与日志见 [执行目录](runs/stage3_20261009_freeze/execution_20261009/) 内 `*.started.json`、`*.finished.json`、`*.log`。没有失败评分、补跑或覆盖。
- 评分后先执行**无标签**[全量审计](runs/stage3_20261009_freeze/execution_20261009/preseal_audit.log)：16/16 文件、共 **61,864** 条记录、四个变体各四域、精确 ID 顺序和数量、有限值、checkpoint 完整 marker，以及 CE `k1=null` 与其余三组完整 K=0/K=1，退出 **0 / PASS**。之后才运行 `seal-scores`，退出 **0**；随后独立评价器才打开四域开发标签，16 条评价命令均退出 **0**。
- 原[2026-10-08 目标标签访问事件](STAGE3_ACCESS_INCIDENT_20261008.md) 保持原样，未删除或改写。评分期间有一次对 evaluator 配置中角色与标签文件**路径文本**的静态查看；未在 seal 前打开标签 sidecar 或读取标签值，评分子进程也从未接收 `--labels-config`。本轮结果严格标为 *disclosed-access target development evaluation*，不得声称满足原始 never-opened-label 条件。
- 最终只读[工件审计](runs/stage3_20261009_freeze/execution_20261009/final_audit.log)退出 **0 / PASS**：Git HEAD、freeze、seal、36 个冻结输入、16 个无标签分数文件、16 个主指标、12 个配对分析及全部命令退出码一致；评分行没有标签、攻击或组字段。
- 对报告中的 CE K=0 相对排序、12 组平均分数变化方向和耗时倍率又做了独立[声明核查](runs/stage3_20261009_freeze/execution_20261009/scientific_claim_audit.log)，退出 **0 / PASS**。
- 所有评分、seal、评价和最终审计先在上述冻结 HEAD 下完成，之后才提交本报告和结果。结果提交会产生新的 Git HEAD；原 `freeze.json` 此后只作为这次已完成评价的来源记录，不能拿来追加评分。

## 2. 固定协议和实际命令

CE 选中 Stage 1 epoch 6，仅 Frozen K=0；Joint 选中 Stage 1 epoch 2，Cross/Same 均选中 Stage 2 epoch 1 且来源为该 Joint checkpoint。Joint/Cross/Same 在 source `select` 两档学习率的 EER 并列时按预定规则选择 `0.0003`；CE 不设 BYOL 更新。各方法固定 `cal0` 阈值分别为 CE `-0.10068067908287048`、Joint `0.9968128502368927`、Cross `3.648584246635437`、Same `2.8675442934036255`。每项适应增益只比较**同一个 checkpoint、同一条音频**的 K=0 与 K=1。

实际入口和退出码：

```bash
conda run -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/preflight.py
# exit 0；preflight.log

conda run --no-capture-output -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/run_scores.py
# exit 0；launcher.log；内含冻结的 16 条 target-scores 命令和独立退出码

conda run -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/preseal_audit.py
# exit 0；preseal_audit.log

conda run -n tta python experiments/meta_audio_tta/stage3_audit.py seal-scores --freeze /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/stage3_20261009_freeze/freeze.json --output /media/dell/data/fakeAudioDection/TTA/.worktrees/exp-meta-audio-tta/experiments/meta_audio_tta/runs/stage3_20261009_freeze/score_set_seal.json
# exit 0；seal_scores.log

conda run --no-capture-output -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/run_evaluator.py
# exit 0；evaluator_launcher.log；内含 16 条 evaluate-target 命令和独立退出码

conda run --no-capture-output -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/paired_analysis.py
# exit 0；paired_analysis.log

conda run -n tta python experiments/meta_audio_tta/runs/stage3_20261009_freeze/execution_20261009/final_audit.py
# exit 0；final_audit.log
```

评分子进程均只使用冻结 checkpoint、source `select`、label-free runner、`freeze.json`；完整参数保存在每项 `*.started.json`。所有输出均使用新路径和独占创建，没有复制 K=0 伪造 K=1。四域样本数分别为 ITW 3,178（真人 2,029、伪造 1,149），WaveFake、LA21、DF21 各 4,096（两类各 2,048）。分数越大越偏 spoof。

## 3. 配对不确定性和解释边界

[次级分析方法](runs/stage3_20261009_freeze/execution_20261009/PAIRED_ANALYSIS_METHOD.md)与[代码](runs/stage3_20261009_freeze/execution_20261009/paired_analysis.py)在本轮 seal 与标签读取前登记并暂存，哈希见 [登记文件](runs/stage3_20261009_freeze/execution_20261009/paired_method_registration.json)。主 EER、AUROC、固定源阈值错误率、逐样本判决翻转和 fake–real pair 排序翻转仍由冻结的 `stage3_protocol.py` 计算。次级分析对各域每类分别有放回抽样，K=0/K=1 使用同一组 ID 索引，固定种子规则，1,000 次 bootstrap 给出 K=1−K=0 的 2.5%–97.5% 分位区间。区间未针对 12 组比较作多重性校正，也不覆盖语料内相关性、checkpoint 选择或既有开发标签访问。

CE 无受训 BYOL 头，因此 K=1、适应收益与配对区间均为 **NOT_RUN**。Joint/Cross/Same 的 K=1 分数几乎每条都发生非零变化，平均快权重 L1 更新也非零；**分数变化并不等于判决或排序改善**。所有 12 组的平均 K=1−K=0 分数为负，即整体略向 bonafide 方向移动。逐样本翻转清单保存在 evaluator-only 目录的 `*.decision_flips.jsonl`，分数变化的 5%/95% 分位数及完整区间见 [paired_analysis.json](runs/stage3_20261009_freeze/execution_20261009/evaluator_only/paired_analysis.json)。

四域中，Joint 的 EER 与固定阈值错误率除 WaveFake 两个 spoof 从正确转错误外均未改变；Cross 在 WaveFake、DF21 各有一个判决改善，在 LA21 有一个判决恶化，ITW EER 略降但 AUROC 略降；Same 在四域没有阈值判决翻转。真假样本对有益/有害排序翻转同时存在，不能与逐样本阈值翻转混为一谈。Joint DF21 的 AUROC 增量约 `0.000033`，本次未校正 bootstrap 区间 `[0.000009, 0.000062]`；量级很小，且 EER/固定错误率不变，不能据此宣称通用收益。

CE Frozen 的 EER 与 AUROC 在四域均优于 Joint/Cross/Same 各自 K=0，但固定源阈值错误率的相对次序并不总相同，尤其 ITW/DF21 存在阈值迁移差异。跨 checkpoint 的描述性差异**不是**同 checkpoint 的 TTA 收益。ITW 的固定阈值错误率显著高于相应 EER，也提示源阈值在该开发域的校准问题；本轮没有使用目标标签调整阈值。

联合/元训练模型的每条 K=0 前向约 0.026–0.029 秒，额外更新与 K=1 前向约 0.109–0.119 秒；记录的合计约为 K=0 路径的 **5.09–5.15 倍**。计时从视图生成之后开始，不含磁盘读取与音频预处理；峰值 CUDA 约 2.709 GiB，不能把不同 checkpoint 的峰值差全归因于内层更新。

**科研结论：**在这四个已披露访问状态的固定开发域、单次逐样本 BN/BYOL 更新、所选 checkpoint 与学习率下，适应确实改变鉴伪分数，但没有呈现跨域一致且相称于计算成本的 EER、AUROC 或固定阈值错误率收益。保留所有微小正向、负向和零变化，不扩大本版本种子、目标评价或学习率搜索。这一负面结果只限定当前实现与数据条件，不否定元适应理论或其他音频辅助任务。

## 4. 完整结果表

下表均为比例而非百分数。CE K=1 `NOT_RUN` 是协议规定，并非缺失数据。真假 pair 的“有益/有害”表示错误排序转正确、正确转错误；括号内是 tie 状态变化。逐样本翻转的“有益/有害”表示固定源阈值下错误转正确、正确转错误。

## Primary metrics (higher AUROC is better; lower EER/error is better)
| Domain | Model | n | K0 EER | K1 EER | K0 AUROC | K1 AUROC | K0 fixed error | K1 fixed error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ITW | ce | 3178 | 0.074914 | NOT_RUN | 0.975354 | NOT_RUN | 0.207363 | NOT_RUN |
| ITW | joint | 3178 | 0.124456 | 0.124456 | 0.946926 | 0.946973 | 0.223726 | 0.223726 |
| ITW | cross | 3178 | 0.128808 | 0.128635 | 0.945482 | 0.945461 | 0.193833 | 0.193833 |
| ITW | same | 3178 | 0.127156 | 0.127156 | 0.946286 | 0.946263 | 0.196665 | 0.196665 |
| WaveFake | ce | 4096 | 0.113770 | NOT_RUN | 0.953624 | NOT_RUN | 0.170898 | NOT_RUN |
| WaveFake | joint | 4096 | 0.143555 | 0.143555 | 0.936526 | 0.936533 | 0.264404 | 0.264893 |
| WaveFake | cross | 4096 | 0.140625 | 0.140625 | 0.937366 | 0.937358 | 0.319824 | 0.319580 |
| WaveFake | same | 4096 | 0.141113 | 0.141113 | 0.938002 | 0.938007 | 0.312988 | 0.312988 |
| LA21 | ce | 4096 | 0.066406 | NOT_RUN | 0.976708 | NOT_RUN | 0.074707 | NOT_RUN |
| LA21 | joint | 4096 | 0.104004 | 0.104004 | 0.954356 | 0.954357 | 0.099854 | 0.099854 |
| LA21 | cross | 4096 | 0.110352 | 0.110840 | 0.940716 | 0.940712 | 0.099121 | 0.099365 |
| LA21 | same | 4096 | 0.107422 | 0.107422 | 0.945863 | 0.945871 | 0.101318 | 0.101318 |
| DF21 | ce | 4096 | 0.052734 | NOT_RUN | 0.988307 | NOT_RUN | 0.118408 | NOT_RUN |
| DF21 | joint | 4096 | 0.071289 | 0.071289 | 0.976190 | 0.976223 | 0.103271 | 0.103271 |
| DF21 | cross | 4096 | 0.078125 | 0.078613 | 0.973260 | 0.973229 | 0.090088 | 0.089844 |
| DF21 | same | 4096 | 0.070801 | 0.070801 | 0.976000 | 0.975970 | 0.091309 | 0.091309 |

## Within-checkpoint K1 − K0 with paired 95% bootstrap intervals
| Domain | Model | Δ EER [95% CI] | Δ AUROC [95% CI] | Δ fixed error [95% CI] |
| --- | --- | --- | --- | --- |
| ITW | joint | 0.000000 [-0.000873, 0.001620] | 0.000047 [-0.000047, 0.000206] | 0.000000 [0.000000, 0.000000] |
| ITW | cross | -0.000173 [-0.000792, 0.000981] | -0.000022 [-0.000119, 0.000049] | 0.000000 [0.000000, 0.000000] |
| ITW | same | 0.000000 [-0.000870, 0.000960] | -0.000023 [-0.000083, 0.000033] | 0.000000 [0.000000, 0.000000] |
| WaveFake | joint | 0.000000 [-0.000488, 0.000977] | 0.000007 [-0.000081, 0.000082] | 0.000488 [0.000000, 0.001221] |
| WaveFake | cross | 0.000000 [-0.000977, 0.000977] | -0.000008 [-0.000064, 0.000044] | -0.000244 [-0.000732, 0.000000] |
| WaveFake | same | 0.000000 [-0.000488, 0.000488] | 0.000005 [-0.000088, 0.000103] | 0.000000 [0.000000, 0.000000] |
| LA21 | joint | 0.000000 [-0.000488, 0.000000] | 0.000002 [-0.000048, 0.000043] | 0.000000 [0.000000, 0.000000] |
| LA21 | cross | 0.000488 [-0.000488, 0.000977] | -0.000005 [-0.000041, 0.000033] | 0.000244 [0.000000, 0.000732] |
| LA21 | same | 0.000000 [-0.000488, 0.000488] | 0.000008 [-0.000074, 0.000071] | 0.000000 [0.000000, 0.000000] |
| DF21 | joint | 0.000000 [-0.001465, 0.000000] | 0.000033 [0.000009, 0.000062] | 0.000000 [0.000000, 0.000000] |
| DF21 | cross | 0.000488 [-0.000977, 0.000977] | -0.000031 [-0.000067, 0.000003] | -0.000244 [-0.000732, 0.000000] |
| DF21 | same | 0.000000 [-0.000977, 0.000977] | -0.000030 [-0.000067, 0.000017] | 0.000000 [0.000000, 0.000000] |

## Decisions, fake–real pair order, and score changes
| Domain | Model | Helpful/harmful decision flips | Helpful/harmful pair flips (ties changed) | Mean Δ score | Median Δ score | Mean absolute Δ | Positive/negative/zero Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ITW | joint | 0/0 | 358/249 (0) | -0.002569 | -0.001679 | 0.003655 | 567/2610/1 |
| ITW | cross | 0/0 | 255/306 (0) | -0.003126 | -0.000955 | 0.004931 | 995/2183/0 |
| ITW | same | 0/0 | 249/303 (1) | -0.002706 | -0.001280 | 0.004269 | 918/2259/1 |
| WaveFake | joint | 0/2 | 606/576 (0) | -0.002725 | -0.002203 | 0.003752 | 318/3778/0 |
| WaveFake | cross | 1/0 | 569/602 (0) | -0.004855 | -0.004930 | 0.006859 | 640/3456/0 |
| WaveFake | same | 0/0 | 719/699 (0) | -0.004193 | -0.004048 | 0.005694 | 512/3584/0 |
| LA21 | joint | 0/0 | 333/326 (0) | -0.001382 | -0.001173 | 0.002516 | 754/3342/0 |
| LA21 | cross | 0/1 | 359/378 (0) | -0.002139 | -0.001115 | 0.003889 | 961/3135/0 |
| LA21 | same | 0/0 | 428/395 (0) | -0.001805 | -0.001168 | 0.003457 | 944/3152/0 |
| DF21 | joint | 0/0 | 292/153 (1) | -0.001253 | -0.001105 | 0.002425 | 714/3382/0 |
| DF21 | cross | 1/0 | 245/377 (0) | -0.001672 | -0.000844 | 0.003833 | 1068/3027/1 |
| DF21 | same | 0/0 | 209/335 (0) | -0.001645 | -0.000852 | 0.003183 | 1089/3007/0 |

## Mean per-item compute time and update magnitude
| Domain | Model | K0 seconds | Update + K1 seconds | Total / K0 | Mean fast-weight Δ L1 | Peak CUDA GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ITW | ce | 0.028248 | 0.000000 | NOT_RUN | 0.000000 | 1.376 |
| ITW | joint | 0.028236 | 0.116918 | 5.14× | 0.006767 | 2.709 |
| ITW | cross | 0.028158 | 0.115899 | 5.12× | 0.007225 | 2.709 |
| ITW | same | 0.027358 | 0.111961 | 5.09× | 0.007139 | 2.709 |
| WaveFake | ce | 0.028669 | 0.000000 | NOT_RUN | 0.000000 | 1.376 |
| WaveFake | joint | 0.028761 | 0.118737 | 5.13× | 0.007257 | 2.709 |
| WaveFake | cross | 0.027419 | 0.112923 | 5.12× | 0.008882 | 2.709 |
| WaveFake | same | 0.026839 | 0.110626 | 5.12× | 0.008356 | 2.709 |
| LA21 | ce | 0.029058 | 0.000000 | NOT_RUN | 0.000000 | 1.376 |
| LA21 | joint | 0.028126 | 0.116616 | 5.15× | 0.006194 | 2.709 |
| LA21 | cross | 0.026964 | 0.110681 | 5.10× | 0.006486 | 2.709 |
| LA21 | same | 0.027262 | 0.112020 | 5.11× | 0.006513 | 2.709 |
| DF21 | ce | 0.028612 | 0.000000 | NOT_RUN | 0.000000 | 1.376 |
| DF21 | joint | 0.027996 | 0.115873 | 5.14× | 0.006009 | 2.709 |
| DF21 | cross | 0.027222 | 0.111821 | 5.11× | 0.006338 | 2.709 |
| DF21 | same | 0.026395 | 0.108576 | 5.11× | 0.006375 | 2.709 |

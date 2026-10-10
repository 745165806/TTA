# MORNING_REPORT — 2026-09-27 overnight TTA

**找到可复现开发候选了吗？NO。** 符合“超过 Frozen、同预算静态 ERM、同一模型关闭适配”三重比较的 TTA 方法：**NONE**。本轮最强的 WaveFake 模型是**不做目标适配的静态源训练 ERM**（AUC 0.933126、EER 14.0625%）；ITW 则没有可观的改进。历史 DCH 的 WaveFake 四 seed 正信号经固定源描述符对照后，不能归因于目标分布条件化。

**下一步只推荐一项：停止本轮残差锚定与 DCH 目标条件化路线，不进入 target90/最终 holdout。** 本轮无候选值得独立验证。ITW target10 与 WaveFake development 均为开发评价，所有结论限于这些缓存、模型和预算。

## 本轮实际执行范围与资源

- 独立 worktree/branch：`/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-overnight-tta`，`exp-overnight-tta`；基于历史 DCH 分支 `fa37e9a`。原工作区及旧实验结果未覆盖。
- `nvidia-smi` 返回 9，无法与驱动通信；`tta` 中 torch 2.1.0 的 CUDA available=false、device_count=0。真正较早层的 waveform 方法 B 为 **BLOCKED_RESOURCE**，没有用最终 160D 缓存冒充它，也没有更改驱动或环境。所有本轮计算使用 CPU，GPU 峰值 0 B。
- 只用项目自训练 Frozen SSL-AASIST 的 160D 三视图 source fit/select 与现有 target selected-only 缓存。ITW target10=3,178；WaveFake development=4,096（2,048 content pairs）。两个目标域各自从相同源 checkpoint 重启，域内最终模型共享；目标适配只接收无标签三视图特征。开发标签在逐样本分数写入后才用于指标。target90 标签/指标访问：**NO**；最终 holdout 访问：**NO**。
- 路线 A：160→32→160 残差特征映射、源监督 BCE+SupCon、目标类条件源原型锚定+三视图一致性+参数位移约束。源 fit 原始/noise 视图同时训练匹配静态线性 ERM。目标是离线 batch-transductive 2 轮、每域一个最终 adapter；源样本和 A01–A06 源攻击原型保留到目标阶段，因此**不属于 source-free**。
- 成熟对照：按 [T3A 作者实现](https://github.com/matsuolab/T3A/blob/master/domainbed/adapt_algorithms.py) 的原生两类 classifier 行、伪标签、熵排序、每类 K=100 和归一化模板构造了 **T3A-batch port**。它先用整域无标签特征估计模板再统一重评分，与作者在线协议不同。原生 spoof-minus-bonafide head 与冻结导出 head 精确一致。

## 真正的源训练与选参

seed 13 的残差和 ERM 各完整训练 10 个 source epoch、各 1,790 个优化步、458,240 次样本呈现，覆盖全部 25,380 个唯一 source fit ID；source select 为 11,723 条。两臂使用相同批次、标签、两个视图和优化步预算；残差另有预先固定的 SupCon 项。fit/select 原始音频 ID 无交集。CPU 训练 9.24 秒，数值失败 0。逐 epoch clean/noise/FIR 验证曲线见 `training_curve.csv`。按 source select clean EER、扰动 EER、clean AUC 顺序选出 ERM epoch 10（1,790 步）与残差 epoch 4（716 步）；两臂都实际训练到 epoch 10，不把选中的 epoch 当作总训练量。source select clean AUC 接近 1，导致适配 LR 的六个源伪域验证 episode 对三个预设候选都饱和为 AUC=1/EER=0。并列规则选择 1e-4。工作阈值只从 source select 得到，包括适配分数自身的源校准。

第一轮只用 seed 13；预先写下的晋级条件要求目标适配优于自身关闭版本，并有一个主域达到至少 0.005 AUC 或 EER 改善且另一域无同量级退化。首轮未达到，故**没有启动 seeds 29/47/71 的新训练**。不能以单 seed 冒称跨 seed 稳定。历史 DCH 已有四 seed，下面单列。

## 首轮主评价（seed 13，LR=1e-4）

EER 改善单位为百分点（pp），正值为好；相对 EER 下降以 Frozen 为基准。FPR/FNR 使用各方法的 source-select 工作阈值。AUC/EER 扫描仅用于开发指标，未用目标标签标定阈值。

| 域 | 方法 | AUC | EER % | EER改善 vs Frozen pp | 相对EER下降 % | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| ITW target10 | Frozen | 0.963309 | 9.857 | 0 | 0 | 0.2597 | 0.0305 |
| ITW target10 | 同预算 ERM | 0.964594 | 9.857 | 0 | 0 | 0.2454 | 0.0305 |
| ITW target10 | 残差 source-only | 0.961182 | 10.005 | −0.148 | −1.50 | 0.2780 | 0.0209 |
| ITW target10 | 残差 target-adapted | 0.961181 | 10.251 | −0.394 | −4.00 | 0.1237 | 0.0870 |
| ITW target10 | T3A-batch port | 0.963041 | 10.009 | −0.152 | −1.54 | 0.2592 | 0.0296 |
| WaveFake dev | Frozen | 0.915003 | 15.771 | 0 | 0 | 0.0073 | 0.5059 |
| WaveFake dev | 同预算 ERM | **0.933126** | **14.062** | **+1.709** | **+10.84** | 0.0039 | 0.5269 |
| WaveFake dev | 残差 source-only | 0.931893 | 14.502 | +1.270 | +8.05 | 0.0117 | 0.4292 |
| WaveFake dev | 残差 target-adapted | 0.931925 | 14.502 | +1.270 | +8.05 | 0.0034 | 0.5454 |
| WaveFake dev | T3A-batch port | 0.920668 | 15.137 | +0.635 | +4.02 | 0.0073 | 0.4980 |

**残差目标适配的独有收益：** ITW 对自身 source-only 的 ΔAUC −0.0000004、EER **退化 0.246 pp**；对 Frozen 的 ΔAUC −0.002128、EER 退化 0.394 pp；对 ERM 的 ΔAUC −0.003413、EER 退化 0.394 pp。WaveFake 对自身的 ΔAUC +0.000032、EER **0.000 pp**；对 Frozen 的 ΔAUC +0.016922、EER 改善 1.270 pp，但对 ERM 的 ΔAUC −0.001201、EER **退化 0.439 pp**。WaveFake 相对 Frozen 的收益来自源训练，不能记作 TTA 收益。

WaveFake 使用 content-pair 配对 bootstrap（400 次），残差开启相对关闭的 ΔAUC 95% 区间约 [−0.000130,+0.000196]、EER 改善比例区间 [−0.001953,+0.001953]；相对 ERM 的 ΔAUC 区间 [−0.002159,−0.000211]。ITW 没有可信的更强 group，只按 audio ID bootstrap，不能解释为说话人/会话独立性。完整三个对照的配对区间逐行保存在 `metrics.csv`。

目标适配 ITW 26 步、0.060 秒，WaveFake 32 步、0.076 秒；两域数值失败和回退均 0，GPU 峰值 0 B。最终全域推理的 20 次中位数：ITW Frozen/ERM/残差关闭/残差开启/T3A 模板+评分分别为约 0.023/0.021/0.428/0.441/1.009 ms；WaveFake 为 0.033/0.033/0.570/0.571/1.159 ms。计时不含缓存读取，CPU 线程数 2。目标 adapter 参数位移范数 ITW 0.203、WaveFake 0.263；确实发生更新，但排名收益缺失。保存的最终目标 adapter 重放后与逐样本分数**逐值完全一致**。

## 一次开发反馈修订（保留原结果）

源伪域 LR 选择完全饱和，首轮 1e-4 的残差目标分数排序几乎不变。因此只做了一次明示的 **development-informed** 修订：在原先三个候选范围内改用 1e-3，沿用同一源 checkpoint，重新从 source select 校准该适配分数阈值；原 1e-4 的分数和指标保留。ITW 残差开启 AUC 0.939755、EER 12.666%，相对自身关闭退化 2.661 pp；WaveFake AUC 0.933157、EER 14.551%，相对自身关闭 EER 退化 0.049 pp，且低于 ERM 的 14.062%。ITW 更新 26 步、0.057 秒；WaveFake 更新 32 步、0.075 秒，数值失败 0。该修订没有候选价值，**不再搜索学习率、loss 或 buffer**。它使用了开发反馈，不能称为独立验证或纯 source-only 选择。

## 历史 DCH 的必要固定源描述符对照

复用历史 `exp-distribution-conditioned-head` 的四个已训练 checkpoint，没有在今晚重训或修改它们。固定描述符只从 source fit 计算：bonafide/spoof 各 50% 质量，spoof 六个官方源攻击家族等权；把同一个 320D source 描述符输入每个 DCH 网络生成**静态源 head**。目标描述符使用完整目标域无标签特征。逐样本两套分数先落盘，之后打开开发标签。结果是四 seed 的配对比较：

| 域 | seed | 固定源 AUC/EER | 目标描述符 AUC/EER | 目标−固定 ΔAUC | EER改善 pp |
|---|---:|---:|---:|---:|---:|
| ITW | 13 | 0.962968 / 9.922% | 0.963029 / 9.922% | +0.000060 | 0.000 |
| ITW | 29 | 0.963450 / 9.835% | 0.963454 / 9.835% | +0.000003 | 0.000 |
| ITW | 47 | 0.963343 / 9.835% | 0.963361 / 9.857% | +0.000018 | −0.022 |
| ITW | 71 | 0.963444 / 9.922% | 0.963455 / 9.922% | +0.000010 | 0.000 |
| WaveFake | 13 | 0.926425 / 14.844% | 0.925744 / 14.795% | −0.000681 | +0.049 |
| WaveFake | 29 | 0.921280 / 15.332% | 0.921548 / 15.234% | +0.000268 | +0.098 |
| WaveFake | 47 | 0.926557 / 14.600% | 0.924624 / 14.844% | −0.001933 | −0.244 |
| WaveFake | 71 | 0.924934 / 14.746% | 0.924496 / 14.893% | −0.000438 | −0.146 |

四 seed 平均 ITW 目标条件化 ΔAUC +0.000023、EER 改善 **−0.006 pp**；WaveFake ΔAUC **−0.000696**、EER 改善 **−0.061 pp**。WaveFake 相对 Frozen 的历史 DCH 四 seed 正收益是**固定源 head 已能达到的源训练效应**；目标分布描述符没有稳定增加价值。DCH 既有训练量为 96 episode，不能冒称全源遍历 20 次。这里的 WaveFake 区间继续按 content-pair 分组；逐 seed 区间在消融 `metrics.csv`。

## 工件、复现和限制

- 本轮代码与预设约束：[experiments/overnight_tta/METHOD.md](experiments/overnight_tta/METHOD.md)、[config.json](experiments/overnight_tta/config.json)、[train.py](experiments/overnight_tta/train.py)、[adapt_eval.py](experiments/overnight_tta/adapt_eval.py)。`tta` 编译检查、分数覆盖/finite、固定目标 ID、训练源 fit/select ID 无重叠、目标状态精确重放均通过；没有运行 target90/holdout。
- 主 run：`/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-overnight-tta/experiments/overnight_tta/results/overnight_20260927_seed13/`。其中有精确命令、配置、训练/评价/状态日志、10 epoch 曲线、source 选模、source LR 选择、工作阈值、逐样本无标签分数、指标、时延、source 和目标 checkpoint。源训练启动时快照中的 ITW 缓存路径是早先检查用的宽缓存路径；实际评价使用的独立 target10-only 缓存在 `evaluation_config.json` 明确记录，源训练未使用 ITW。
- 唯一修订 run：`.../results/overnight_20260927_seed13_revision_lr001/`，保留完整原结果，不覆盖。固定 DCH 消融：`.../results/dch_fixed_descriptor_20260927/`。大 checkpoint 和逐样本分数留工作站；小型 CSV/JSON/代码随分支提交。各 run 的 `COMMANDS.md` 给出真实命令。历史 DCH 四 seed checkpoint 位于 `/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-distribution-conditioned-head/experiments/distribution_conditioned_head/results/`。
- 本轮没有跨 seed 的新残差候选稳定性结论；T3A 与残差只完成 seed 13。已训练的历史 DCH 四 seed 固定源描述符消融显示 WaveFake 条件化增量为负。因 GPU 不可用，较早层路线 B 没有数值结果，不能据此否定较早层方法。日志中的 torch TypedStorage deprecation 是非致命警告；计算数值失败为 0。

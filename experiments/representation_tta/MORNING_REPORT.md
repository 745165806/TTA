# MORNING_REPORT — representation-level SSL-AASIST TTA

**找到有效候选了吗？BLOCKED。** 最佳方法：**NONE（尚无有效科学评价）**。自动审批审查两次拒绝了完整 source fit/select 与 ITW target10/WaveFake development 的 GPU 较早层特征提取，因此真实源训练和两个开发域评价没有执行。**不能**把 16 条工程 smoke 当成方法收益或报告 NO 的负实验结论。

| 要求 | 当前证据 |
|---|---|
| ITW：Frozen / Static source-only / Adapter OFF / Adapter ON | Frozen 历史参考 AUC 0.963309、EER 9.857%；本分支其余三项 **NOT_RUN**。本分支的 TTA 独有 ΔAUC/EER 改善 **未定义**。|
| WaveFake：Frozen / Static source-only / Adapter OFF / Adapter ON | Frozen 历史参考 AUC 0.915003、EER 15.771%；本分支其余三项 **NOT_RUN**。本分支的 TTA 独有 ΔAUC/EER 改善 **未定义**。|
| 多 seed | seed13 全量训练/评价未运行；29/47/71 按预设晋级条件尚不应启动。|
| 真正的 representation adapter 位置 | 项目训练的 SSL-AASIST `LL` 输出 `[B,201,128]`，位于 max pooling、RawNet2、图注意力和最终 160D embedding 之前；128→64→128 residual，16,576 参数。|
| 使用 GPU | 工作站四张 RTX A6000；`tta` 在获准 GPU 上下文的小张量前后向 PASS。一条真实音频完整模型与 LL 回放 logits 精确一致；16 条源音频三视图抽取和四步无标签更新 PASS。完整 GPU 抽取被自动审批拒绝。|
| 最佳 checkpoint | 仅原项目 Frozen epoch 7；**无源训练后的 adapter checkpoint**。|
| Decision | **BLOCKED_RESOURCE**（自动审批拒绝完整固定范围 GPU 提取）。|
| target90 accessed / final holdout accessed | **NO / NO**。|

## 已完成的可复核工作

1. 保护主工作区：原项目 `/media/dell/data/fakeAudioDection/TTA` 存在未提交修改；从 `f58f3c2` 建立独立 `exp-representation-tta` worktree，未迁移或覆盖旧实验。
2. 测量实际模型结构：作者 `model.py` 的 XLS-R 输出 `[1,201,1024]`，任务训练 LL 输出 `[1,201,128]`。单条真实 ASVspoof2019 LA 音频完整 waveform 前向与原后端从 LL 回放的两个 logits 最大误差 `0.0`。详见 `probe_model.json` 和 [probe_model.py](probe_model.py)。
3. 已实现真实 waveform 到 LL 的 GPU 抽取器 [extract_ll.py](extract_ll.py)，仅读取现有 fit/select、ITW target10 及 WaveFake 4096 固定 assignment；WaveFake 复用先前经审计的 16kHz 派生波形，不访问生成器标签。该缓存每条为三视图 `[3,201,128]` float32。44,377 条总量估计 12.76 GiB，磁盘空余 692.7 GiB；16 条 smoke GPU 峰值约 2.12 GiB。见 `resource_estimate.json`。
4. 初次 16 条 smoke 发现与既有 Frozen 160D 缓存最大分数差 0.0337，原因是新脚本未遵守生产提取器禁用 TF32 的数值模式。保留初次缓存，关闭 CUDA matmul/cuDNN TF32 后以独立 `ll_smoke_20260928b` 重测，差降至 0.000486（跨 batch 数值差）；同一 LL 张量关闭 adapter 则**精确恢复**冻结后端分数。真实 LL adapter BCE 反向梯度范数 4.266，有限非零。见 `smoke_adapter.json`。
5. 用合成工程原型检查 [adapt.py](adapt.py) 的标签隔离和梯度路径：16 条真实 LL 特征上四次无标签更新，参数位移范数 0.03256、最大分数变化 0.0593、无数值失败。合成原型不构成源训练或目标性能证据。见 `smoke_adapt.json`。
6. 已准备源标签/攻击合同、严格缓存读取、[train.py](train.py) 的同批次同预算 BCE 静态对照与 BCE+SupCon adapter 训练、[evaluate.py](evaluate.py) 的四臂比较、源阈值、WaveFake content-pair 配对 bootstrap、最终共享适配器分数和 checkpoint。源训练和评价代码**尚未经全量真实运行验证**，不能标 PASS。参数与晋级规则见 [METHOD.md](METHOD.md) 和 [config.json](config.json)。

## 阻塞与下一步

自动审批审查拒绝的动作是：在四张 GPU 上为固定 25,380 source fit、11,723 source select、3,178 ITW target10 和 4,096 WaveFake development 提取 LL 三视图缓存，再进行源训练/目标开发评价。审查理由是 `AGENTS.md` 默认禁止本机全量缓存和目标评价；重试时给出用户当前明确授权、目标范围、非覆盖路径和 12.76 GiB/692.7 GiB 的资源证据后仍被拒绝，并要求不要通过间接执行绕过。当前用户任务明确要求这些全量研究步骤，但自动审批阻止了实际运行。

**唯一下一步：取得该固定范围 GPU 提取及随后的源训练/双域开发评价的明确审批，然后直接运行已准备的路线 A。** 审批前不启动全量缓存或用最终 160D 特征替代。若路线 A 首轮 seed13 无价值，再按 [METHOD.md](METHOD.md) 的最多两路线规则决定第二个较早层变体。target90 和最终 holdout 保持关闭。

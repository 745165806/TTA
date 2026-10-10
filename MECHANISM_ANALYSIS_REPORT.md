# SSL-AASIST episodic TTA v1 机制分析

日期：2026-10-09。分支：`codex/meta-audio-tta-v2-mechanism`，起点 `98df7f255e956f98a33da9dbf575edfbeb7e9303`。本报告只用既有 source roles、v1 checkpoint/日志/分数和本分支新增的有限源端 CPU 前向/梯度记录；没有训练新模型、没有新增目标评分，也没有读取 target90 或 final holdout。四域 Stage 3 是已披露访问状态的目标开发反馈，不能称为 never-opened-label 评价。

## 结论摘要

**本轮证据支持停止把当前无约束 BYOL 梯度当作默认单样本适应方向；证据不足以判定 BYOL 这类辅助任务原则上无效。**有限真实模型诊断中，BYOL 与同一共享 BN 仿射子空间内的监督 CE 梯度在 24 个 checkpoint×样本计算里总体负相关（Joint −0.340、Cross −0.210、Same −0.327；它们重复使用 8 个 source audio ID，并非 24 个独立样本）；在同更新范数的源标签 oracle 对照下，CE loss 下降，而 BYOL 单步后这 8 个挑选样本的 CE loss 上升。与此同时，v1 原有 64 条 source_val 全集上三种 BYOL checkpoint 的平均 CE 又都微幅下降，且 EER/AUROC 从 K=0 到 K=1 仍分别为 0/1、没有阈值判决或 pair 排序翻转。这一反例说明不能把 8 条机制子集的负向趋势外推为总体因果结论。

更稳妥的解释是：**所测源端已经接近性能天花板，单样本的 BYOL 更新幅度小，且更新方向对具体音频/声学视图不稳定；meta 初始化并未在这些设置下把这种更新转成可见的稳健分类收益。**方法层面还有任务构造差异：原 MABN 将 source domain 作为任务，MT3 明确用任务共享的分布变换制造任务间 shift；v1 的 source episode 是单音频两视图，独立 query B 仍来自相同源采样分布，没有模拟共享声学域偏移。它能测试逐音频更新协议，却没有复现原方法的 domain-task 训练分布。

## 证据账本

| 层级 | 已观察事实 | 尚不能推出 |
|---|---|---|
| 阶段 1 | CE 与 CE+BYOL 从同一 XLS-R 初始化，使用相同 fit、增强、batch/epoch 预算，各 8 epoch、14,504 主干步。两者选中 source_val EER 均为 0；CE 选 epoch 6，Joint 选 epoch 2。BYOL loss 随训练下降。 | 因 source_val AUC 与攻击/条件分层未按 epoch 保存，不能声称 Joint 全面降低了 source 泛化；也不能由不同选中 epoch 的跨 checkpoint 结果单独归因 BYOL。 |
| 阶段 2 | Cross 与 Same 均完成 2 epoch，各选 epoch 1；固定 256 条验证子集的 K=0/K=1 EER 0、AUC 1，分数有变化。 | 该子集饱和，且 Cross 每轮曝光 512 个不同 fit 音频、Same 256 个，未等曝光；Cross/Same 差异不能单因果解释。 |
| source_val K 配对 | 已有固定 64 条（32/class）K=0/K=1：Joint、Cross、Same 三者均为 EER 0、AUC 1；各自 64/64 分数变化，但正确/错误判决翻转 0、1,024 个 fake-real pair 排序翻转 0。平均绝对分数改变量分别 0.001969、0.002456、0.002495。 | 这 64 条属于曾用于 source_val epoch 选择的角色，不是独立确认集；EER/AUC 饱和不证明全源数据无效或有效。 |
| fit 声学扰动 | 每方法固定 16 个 fit 样本的原音频与 FIR 条件，K=0/K=1 两边 EER 0、AUC 1、无判决或 pair 翻转；每条件分数均发生非零变化。原音频/FIR 间更新方向符号改变 Joint 4/16、Cross 4/16、Same 6/16。 | FIR 只是一个很轻、确定性的条件，不代表真实信道、codec 或攻击变化；这些结果不能选目标超参数。 |
| 四域开发评价 | 对同 checkpoint 的 K=0/K=1，12 个方法×域组合中有微小正、负和零变化；Joint DF21 AUROC +0.000033（本次未校正 paired bootstrap CI [0.000009, 0.000062]），但 EER 和固定源阈值错误率不变；其余增量亦很小且无跨域一致方向。每条 TTA 路径约为 Frozen 的 5.09–5.15×。 | 这是 disclosed-access target development feedback，不能用于声称无目标标签访问的最终泛化。不得继续反复开放开发域来调当前版本。 |

## 真实 SSL-AASIST 梯度与 oracle 诊断

新增诊断加载四个真实 checkpoint（CE epoch 6、Joint epoch 2、Cross epoch 1、Same epoch 1），使用同一批 8 个已有 source_val ID：在 CE Frozen 的既有每类 32 条固定子集内，按 `|spoof_logit - bonafide_logit|` 每类预先取 2 条最小 margin 与 2 条最大 margin。另对相同音频套用已登记的 deterministic FIR 变换。此分层仅用于机制诊断，不是重划数据或选参。样本本来就都分类正确；“hard”代表相对小 margin，最小绝对 margin 仍约 2.87，因此不是接近决策边界的困难样本。

每个 BYOL checkpoint 的 8 条音频、每个声学条件均真实计算 1,538 个共享骨干 BN affine 参数上的 BYOL 梯度、监督 CE 梯度、L2 模长、点积与余弦。内层步长使用 v1 固定 `0.0003`。oracle 用同一 CE 标签和同一 BN 子空间更新；为隔离方向和模长影响，主要比较将 oracle 梯度缩放到与 BYOL 相同的 BN 更新 L2 范数。未对 aux projector/predictor 的 BN 参数做 CE oracle 更新，因为主任务 CE 图对这些分支无梯度；主分数不受辅助 head 参数直接影响。

| Checkpoint | 条件 | cos 均值 [min,max] | BYOL / CE 梯度 L2 均值 | BYOL 更新后同样本 CE Δ | 同范数 CE oracle 后 CE Δ |
|---|---|---:|---:|---:|---:|
| Joint | 原音频 | −0.340 [−0.644, 0.084] | 0.624 / 11.622 | +0.001632 | −0.002116 |
| Cross | 原音频 | −0.210 [−0.708, 0.455] | 0.746 / 12.887 | +0.000945 | −0.003005 |
| Same | 原音频 | −0.327 [−0.749, 0.200] | 0.810 / 12.579 | +0.001005 | −0.003021 |
| Joint | FIR | −0.365 [−0.683, 0.104] | 0.621 / 11.671 | +0.000771 | −0.002053 |
| Cross | FIR | −0.200 [−0.715, 0.469] | 0.750 / 12.816 | +0.000835 | −0.002990 |
| Same | FIR | −0.307 [−0.749, 0.174] | 0.806 / 12.538 | +0.001047 | −0.003775 |

更新范数匹配前的 oracle（同用 lr 0.0003）下降更多：原音频平均 ΔCE 分别 −0.0386、−0.0526、−0.0496。该对照因步长下 CE 梯度比 BYOL 梯度大约 15–20 倍，不作为公平的效能比较；应以表中的同范数结果为主要 oracle 参照。BN 更新 L2 均值在原音频只有 0.000187–0.000243。余弦为负表明**该子集上首步方向冲突**；但均值、范围和逐点正余弦反例同时表明并非每条音频都冲突。样本极少，区间没有统计推断含义。

CE checkpoint 没有受训 BYOL branch，故对 CE epoch 6 只计算 K=0 及 BN 空间 CE oracle；本报告没有把随意附加一个未训练的 BYOL head 冒充 CE 的 BYOL 梯度结果。该 checkpoint 的 1,538 维 BN CE 梯度 L2 均值为 12.607（原音频 n=8，范围 12.108–13.439）；同样 lr 0.0003 的源标签 oracle 更新 L2 均值为 0.003782、同样本 CE 平均下降 0.05007。此步长没有和其它 checkpoint 的 BYOL 更新 norm 匹配，不能作跨模型效能比较；它只说明冻结 detector 的 BN 参数并非没有主任务 CE 梯度。

按 canonical label（0=bonafide、1=spoof）分开后，方向差异更明显：

| Checkpoint | 条件 | bona cos / BYOL ΔCE / matched oracle ΔCE | spoof cos / BYOL ΔCE / matched oracle ΔCE |
|---|---|---:|---:|
| Joint | 原音频 | −0.535 / +0.002970 / −0.002265 | −0.144 / +0.000294 / −0.001968 |
| Cross | 原音频 | −0.580 / +0.002172 / −0.004047 | +0.161 / −0.000283 / −0.001963 |
| Same | 原音频 | −0.591 / +0.001864 / −0.003353 | −0.063 / +0.000146 / −0.002690 |
| Joint | FIR | −0.561 / +0.001185 / −0.002267 | −0.169 / +0.000358 / −0.001840 |
| Cross | FIR | −0.562 / +0.001945 / −0.003955 | +0.163 / −0.000275 / −0.002025 |
| Same | FIR | −0.559 / +0.001733 / −0.003358 | −0.055 / +0.000361 / −0.004193 |

每个 class×checkpoint×条件只有 4 条（hard 2、easy 2）。同一 source ID 会在三个 checkpoint 和两种条件重复，不能将表内格子当独立样本。它提示 v1 梯度冲突在 bonafide 侧较一致，而 spoof 侧 Cross 有正余弦反例；仍只能视作下一轮分层验证的先验。

![源端真实模型梯度方向和同样本 CE 损失变化](experiments/meta_audio_tta/runs/mechanism_cpu_20261009_normmatched_retry1/figures_normmatched/gradient_alignment_and_loss.svg)

### CE、margin、EER/AUROC 和排序

| 模型/源集合 | K=0 平均 CE | K=1 平均 CE | EER K0→K1 | AUC K0→K1 | 判决翻转 | pair 排序翻转 |
|---|---:|---:|---:|---:|---:|---:|
| CE epoch 6，64 条 source_val | 0.000566 | NOT_RUN | 0→NOT_RUN | 1→NOT_RUN | NOT_RUN | NOT_RUN |
| Joint，64 条 source_val | 0.010438 | 0.010423 | 0→0 | 1→1 | 0 | 0/1,024 |
| Cross，64 条 source_val | 0.004198 | 0.004181 | 0→0 | 1→1 | 0 | 0/1,024 |
| Same，64 条 source_val | 0.005419 | 0.005419 | 0→0 | 1→1 | 0 | 0/1,024 |

源端全 64 样本的 K1 平均 CE 变化是 Joint −0.0000151、Cross −0.0000165、Same −0.0000002；这与上面的八样本 hard/easy 梯度诊断相反或接近于零。分层子集 K=0/K=1 margin 都没有跨过 0，因此它自己的 EER/AUC 与排序仍饱和；oracle 同样本 CE 改善不能算作泛化收益。现有 source_val 记录未按攻击/真实录音来源做经过核验的分组字段，本轮不猜攻击列或目录语义。

fit 中 16 条固定样本按类别拆分后的平均 `K1−K0` spoof margin 变化（每格 n=8）再次显示条件敏感，不代表性能变化：

| Checkpoint | 条件 | bonafide 平均分数变化 | spoof 平均分数变化 |
|---|---|---:|---:|
| Joint | 原音频 | +0.001869 | −0.000067 |
| Joint | FIR | −0.001378 | −0.000559 |
| Cross | 原音频 | −0.002781 | +0.000223 |
| Cross | FIR | +0.004099 | −0.002361 |
| Same | 原音频 | −0.002642 | +0.000240 |
| Same | FIR | −0.000610 | −0.000352 |

## 收敛曲线及性能天花板

阶段 1 训练日志的匹配 epoch EER 如下（数据量每 epoch 1,813 步；EER 由整套 5,654 条 source_val 计算，AUC 没保存）：

| Epoch | CE EER | Joint EER | CE 源训练 loss | Joint CE loss |
|---:|---:|---:|---:|---:|
| 1 | 0.000201 | 0.000604 | 0.504878 | 0.490190 |
| 2 | 0.000805 | 0.000000 | 0.119776 | 0.119268 |
| 3 | 0.000201 | 0.001458 | 0.057015 | 0.057115 |
| 4 | 0.000201 | 0.000000 | 0.031415 | 0.034307 |
| 5 | 0.001458 | 0.000000 | 0.020978 | 0.021901 |
| 6 | 0.000000 | 0.000000 | 0.018253 | 0.016347 |
| 7 | 0.000000 | 0.002915 | 0.009433 | 0.012026 |
| 8 | 0.000604 | 0.000000 | 0.009138 | 0.008878 |

整集 EER 的小幅波动和几乎重合的训练 CE 不支持“Joint 已明显更好”或“Joint 明显损害 source”任一说法。新增匹配 epoch 推理使用固定 8 条源端样本、CE epoch6 margin 排出的 hard/easy 组；命令完成后 summary/逐样本记录在 `runs/matched_epoch_cpu_20261009/`。该小探针只作为困难 margin 描述，不取代完整验证，也不参与 epoch 选择。

四域 K=0 跨 checkpoint 对比中，CE Frozen 的 EER/AUROC 在四域都比 Joint/Cross/Same 各自 K=0 好；例如 ITW CE `0.074914 / 0.975354`，Joint `0.124456 / 0.946926`，WaveFake CE `0.113770 / 0.953624`，Joint `0.143555 / 0.936526`。但 checkpoint 和训练方式不同，所以这是**选择强 CE Frozen 为后续对照起点的动机**，并非同 checkpoint TTA 的因果对照。四域结果属于 disclosed-access feedback。

## 机制判断

1. **主任务已接近天花板。**本报告固定源样本以及原 v1 source_val/FIR 诊断的 EER=0、AUC=1；在此范围，最多能看到 loss 和 margin 变化，很难通过错误率下降展现收益。四域 CE Frozen 强于三种 BYOL checkpoint 的 K0 也说明后续适应应以强 CE detector 为主基线。
2. **当前 BYOL 梯度不是主任务的稳定下降方向。**共享 BN 仿射空间的有限子集平均余弦为负、BYOL 后同样本 CE 增加；同范数 oracle 下降。它支持“辅助目标方向失配”的可检验解释，但 n=8/class-balanced-source subset 太小，且 source labels 已用于分层与 oracle，因此不构成无标签适应收益结论。
3. **更新很小且声学敏感。**源端每条分数虽然都有变化，full subset 的最大指标效应几乎为零；FIR 对上某些样本的更新变化方向不稳定。辅助分支表征维度方差不为零，v1 记录的在线/目标 projection cosine 约 0.94–0.99，未发现整体全零表示塌缩证据；小样本统计不能排除局部退化。
4. **元目标 task 不等于声学域 shift task。**v1 Cross 外层 query 来自独立音频，Same 用 support 的标签，均来自原 source fit 分布。MABN 原文把 source domain 作为 task；MT3 还为每个 task 固定一个跨该任务样本共享的 batch augmentation 来显式产生任务间分布变化。V1 的固定原音频/FIR 视图一致性没有相同的任务结构。这是实现分布上的差异，不是“元学习无效”的证明。
5. **计算成本不匹配微小变化。**v1 四域 TTA约慢 5.1 倍；pair 排序翻转有益与有害并存，K1 平均分数普遍往 bonafide 方向移动。按当前版本停止扩展种子和目标评价是合理的，不应靠反复目标筛选找正收益。

## 文献对照及差异

| 文献 | 原方法/依据 | 与 v1 的对应以及启示 |
|---|---|---|
| MABN, AAAI 2024 | 只更新 BN affine、冻结 source statistics，用标签无关 SSL 辅助分支，并以 bi-level 元目标让辅助更新受主任务 query 监督。论文目标是 unseen **domain** 的 few-shot test-time adaptation。 | v1 移植了 BN/SSL/meta 的机制，但把任务改为单音频 episode；Cross/Same 是音频协议对照，不是原 domain-level MABN 复现。负结果不能否定其原始域级设定。依据：[论文](https://ojs.aaai.org/index.php/AAAI/article/view/29527)、[作者代码](https://github.com/ynanwu/MABN)。 |
| MT3, AISTATS 2022 | BYOL 内层适应、监督外层准确率/meta loss；训练时以不同分布定义 tasks，并以 task 内共享随机 batch augmentation 构造 shift；论文特意指出简单 joint supervised+SSL training 并不自动得到可快速适应模型。 | 证明“BYOL+meta 可学会某种 shift 分布下的适应”，不保证音频伪造检测的每音频任务有效。v1 数据角色与计算图参照其 bi-level 思路，但任务变换覆盖太窄。依据：[原文](https://proceedings.mlr.press/v151/bartler22a.html)、[作者实现](https://github.com/AlexanderBartler/MT3)。 |
| TTT, ICML 2020 | 联合训练主任务和旋转预测辅助任务，测试时用单样本旋转伪标签更新共享特征，再对同样本推理。 | 辅助损失能工作的关键是预训练让该信号与主任务泛化相关；从图像旋转或 BYOL 的论文成功不能推出任意音频一致性梯度有益。依据：[论文](https://proceedings.mlr.press/v119/sun20b.html)。 |
| BYOL, NeurIPS 2020 | online predictor 对齐 EMA target 的另一增强视图表征，stop-gradient/slow target 是其架构组成。 | v1 的目标 branch、predictor 和 EMA 语义源于此；BYOL 的表示学习有效性不等于其更新方向降低 spoof CE。依据：[论文](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)。 |
| RawBoost, ICASSP 2022 | 对 raw waveform 做多种信号处理扰动，提升 anti-spoof 学习的声学多样性；作者维护了参考实现。 | 给 v2 构造受控声学因素的依据；它是源训练数据 boosting/augmentation，不是自监督适应目标有效的直接证据。依据：[论文/项目页](https://www.eurecom.fr/en/publication/6736)、[作者代码](https://github.com/TakHemlata/RawBoost-antispoofing)。 |
| On Pitfalls of TTA, ICML 2023 | 指出选参困难、TTA 效果依赖基础模型质量，且有些 shift 即便调到较优也仍难适应。 | 支持强 Frozen 基线、固定源端 select 和少看目标的协议；本项目 CE checkpoint 好于其他 K0 的跨 checkpoint 现象不能当适应增益。依据：[论文](https://proceedings.mlr.press/v202/zhao23d.html)。 |
| ALDEN, ACM MM 2025 | 面向跨 vocoder 泛化的双层解耦：低层声码器特定/无关表征，高层分离内容/说话人因素，再做 vocoder-agnostic meta learning。 | 这是监督/训练阶段的跨声码器泛化，不是单音频 TTA；其问题分解提醒“声学/说话人不变性”和“伪造线索保留”要明确定义。其重建/解耦模型成本显著高于本轮有限 TTA，当前不应直接搬进 v2。依据：[论文页面](https://doi.org/10.1145/3746027.3754741)、[作者代码](https://github.com/Beyond0814/ALDEN)。 |
| T²A, IJCAI 2025 | 面向 deepfake detector 的 online TTA，强调不确定样本优先、negative learning 和梯度 masking，而非只依赖初始预测自训练。 | 直接说明 deepfake TTA 中适应信号可能不可靠；但 online 状态跨样本，与本项目 EP-reset 合同不同。这里只作为风险参考，不把其算法并入候选方法。依据：[论文](https://www.ijcai.org/proceedings/2025/854)。 |

## 局限与访问边界

- 新梯度研究在真实 SSL-AASIST CPU 上完成：`nvidia-smi` 可见 GPU 0/1/3 空闲、GPU2 有约 28% 负载，但 `tta` 进程的 `torch.cuda.is_available()` 为 false 且无 `/dev/nvidia*`。CPU 只影响速度，不改变 checkpoint 和前向/梯度数学；不把本次 CPU 时间作为 GPU 成本估算。
- 梯度夹角诊断每个 BYOL checkpoint 每条件 n=8，且所有样本正确分类；没有置信区间或总体分布推断。EER/AUC 的稳健数值来自既有 n=64、但已用于 source_val 选 epoch 的诊断集。
- 假标签/真实标签泄漏：BYOL update 函数自身不接标签。源标签只由独立分析读取，用于 same-sample CE 诊断、oracle 和报告分层。此处 oracle 仅机制上界，不是部署对照。
- FIR 是预登记的轻微声学探针；没有 claim 它覆盖真实域偏移。没有尝试新声学攻击或多参数搜索。
- v1 的四域评价保留 2026-10-08 访问事件并标为 disclosed-access；本轮没有打开目标评价标签、重新跑目标评分，或触碰 target90/final holdout。

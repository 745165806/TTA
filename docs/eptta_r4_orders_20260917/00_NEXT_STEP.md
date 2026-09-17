# EP-TTA v0.1.0：AASIST FINALIZED 后的下一步

日期：2026-09-17  
本轮范围：**R4 冻结导出和真实一致性验收；R5 只准备计划。**  
本文件是给项目内编码 AI 的接续方案，依据用户提供的最新报告编写；本轮没有访问远程源码或 checkpoint，也没有执行项目训练、导出或 GPU 验收。

## 1. 当前不再做什么

不重新划分已审计合格的源数据，不从头 smoke，不重试已失败的 exact resume，不自动续训80–99，不等待 SSL-AASIST，不新增目标评分。

AASIST 已被报告为正式 FINALIZED，但 FrozenModelBundle 及其真实 parity 尚未产生。下一步是把选好的底座转为可以安全供后续缓存/EP 使用的固定模型，而不是立刻测试 EP 的效果。

训练轨迹复现与冻结推理一致性是不同验收。获准的非 exact continuation 必须持续披露，但它不在逻辑上意味着同一份固定权重无法通过推理导出验证。

## 2. 必须绑定的模型身份

| 项目 | 用户提供的值，仍需现场核验 |
|---|---|
| Run ID | `source-run-91cd36fae7770aca4b51` |
| Finalized ID | `training-final-a095c4459547c47c9481` |
| 选择的 checkpoint | epoch69，不能用epoch79的last替代 |
| checkpoint SHA-256 | `076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a` |
| 实际累计训练终点 | epoch0–79，共80轮，其中child为14–79共66轮 |
| 最终global step | 42,320；这是训练终点，不应未经核对当成best checkpoint的step |
| 学习率日程 | 100-epoch cosine horizon |
| source_val EER原始值 | `0.00040257648953301306`，单位待实际实现确认 |
| continuation | 用户授权的非exact parent→child |

### 两个必须分开写的数字

**实际训练80轮**与**scheduler horizon=100**不是同一个概念。核验批准的child plan确实允许epoch79结束。若一致，继续R4；若合同没有接受该终点，报告资格冲突，不自动补训练、不改原finalized记录，也不直接绕过R4门禁。

若EER是0–1比例，乘100后为约**0.04026%**。不要把原始小数直接加百分号。若实际字段已经定义成百分比，遵循其真实语义。

## 3. 本轮怎样验证

| 验收项 | 需要的真实证据 |
|---|---|
| 身份正确 | finalized、选模记录、实际epoch69文件和完整hash闭合；保留非exact历史 |
| 拆分正确 | 导出的embedding确实是原head输入，head和native logits对应，类别/score映射不变 |
| 保存加载正确 | 两个独立实例：原绑定代码加载selected checkpoint vs 新进程加载export bundle |
| 行为冻结 | 关闭评估增强；参数和持久buffer不变；首次/重复/A→B→A行为稳定 |
| 输入不变 | 相同采样率、裁剪/补齐、归一化与输入张量，生产I/O也对齐 |
| 数值一致 | 逐样本embedding、logits、score在预先固定容差内，不能只比较EER或shape |
| 梯度边界正确 | head参数冻结，但其输入上的独立诊断梯度通路可用；不更新真实模型或拟合EP |
| 完整源验证 | source_val按锁定清单完整覆盖，核实指标单位；不重新挑checkpoint |
| 发布正确 | 只有真实必需项通过才原子发布可进入R5的bundle；候选/失败不能冒充PASS |

默认固定128条fit音频做细致验收，再检查完整source_val（报告为5,654条，以实际snapshot核对）。不使用select调试选参数，不使用cal0校准，更不读取control_test/外部目标效果。

真实特征提取可以用于R4一致性诊断，但不是R5正式批量cache；其产物不能绕过后续用途/身份门禁。

### 实现时容易遗漏的事项

PyTorch的`.eval()`、参数`requires_grad=False`、`no_grad()`/`inference_mode()`控制的内容不同。尤其未来需要对适配变量求导时，不应把整个路径放进禁用梯度的上下文；使用inference mode提取后的张量也需要核验与后续梯度通路的兼容性。[S1]

作者公开AASIST代码中，forward有显式`Freq_aug`开关，最后返回`last_hidden`和分类输出；卷积实现还存在普通tensor属性及前向构造的滤波器属性。执行AI必须核对项目pinned版本是否一致，不能将公开main当成本地版本，也不能只查注册buffer就结束状态审计。[S2]

不同batch组织、精度或设备可能导致数值差异；应在运行前固定各场景的容差和支持范围，而不是要求所有CPU/GPU结果逐位相同，或在失败后任意放宽。[S3]

## 4. 对编码 AI 的开发范围

读取仓库后将能力分为“可复用、缺真实验收、缺实现、外部阻塞”。优先使用现有factory/export/wrapper/FrozenModelBundle和R4→R5 gate。

确实缺失时，只补最小导出入口、独立parity runner、source_val复算与覆盖检查、状态/梯度边界诊断和反例测试。新增字段必须贯通schema、解析、序列化、hash和测试。不为本轮新增部署框架、模型结构、缓存体系或调度平台。

不修改训练产物以迁就新代码。训练代码身份与导出代码身份分别记录。源checkpoint只读，导出产物独立保存，不自动commit/push/tag，版本保持0.1.0。

## 5. 资源与停止点

本轮建议单卡GPU资源上限为**0.5 GPUh**，包含预热、失败和重试，仅用于R4推理验收及诊断。这是待用户发送工单/既有审批接受的上限，不是预计完成时间；不假定上一次7.3 GPUh训练预算可继续使用。

已有有效批准覆盖R4时直接执行，不重复申请。若合同必须另行绑定具体计划hash，而当前尚无批准，则先完成代码、CPU验收、计划固化和完整hash，只在GPU启动前提出一次准确批准模板。资源不足时如实PARTIAL/NOT_RUN，不能擅自扩预算。

R4通过后交付bundle、真实parity、source_val覆盖/复算、命令/退出码、源码身份和成本。**停止在R5执行前。**

## 6. R4之后的最短路径

**R5：**用合格bundle建立小规模真实cache，验证原路径/缓存parity与K=0接线，再按已有合同准备fit/cal0/select缓存和U/M/Fisher/static R/tau0。

**R6：**在源select上完成Frozen、同视图平均不更新、完整EP及已有P0/P1/关键消融，得到第一张真实适配表。

**R7–R9：**后续补齐既定底座/seed/五项方法，封存后按四个目标范围执行评分与独立评价。本轮不提前执行。

当前尚未完成SSL资格，不阻塞AASIST先导；但不能据此把未来主实验悄悄删为只有AASIST。四个主要目标以及2019已查看的历史保持不变。

## 7. 使用方式

把本文件与`01_R4_START_NOW.txt`一起交给项目编码AI。TXT是本轮完整指令，不是shell脚本，且不是已经执行过的结果报告。后续实际路径和CLI由编码AI从仓库查明，不编造未实现命令。

本轮期待的核心交付只有：**与指定epoch69权重绑定、真实验收通过、可合法进入R5的FrozenModelBundle。** 没有通过就交付具体失败证据，不把合成测试数量当成R4资格。

## 技术依据

以下只说明通用实现风险，不证明项目已完成任何验收。资料核对日期为2026-09-17；项目运行仍使用其已绑定环境，不因官方文档版本较新就升级环境。

[S1] PyTorch, Autograd mechanics，关于eval与梯度模式、inference mode限制。
https://docs.pytorch.org/docs/2.14/notes/autograd.html

[S2] AASIST作者公开源码，Model.forward、最后分类头和CONV实现；仅供执行AI对照其本地pinned revision。
https://raw.githubusercontent.com/clovaai/aasist/main/models/AASIST.py

[S3] PyTorch, Numerical accuracy，关于batch计算、设备及有限精度差异。
https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html

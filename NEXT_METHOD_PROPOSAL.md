# v2 候选方法与源端验证方案

状态：**提案，未实现/未训练/未作新目标评价。**本提案根据 v1 四域 disclosed-access 开发反馈和 2026-10-09 source-only 机制诊断形成，应登记为新版本。没有任何候选被本报告称为已有效。

## 决策

- **继续 BYOL 的条件：**不再把原始 BYOL 梯度作为默认主路径。只值得做一次小预算、事先限定的对照，测试一个源端学出的低维更新方向变换能否修复已观察到的梯度冲突。若 source `select` 不优于强 CE Frozen，当前 BYOL 路线停止，不进入目标开发评价。
- **保留 BN 仿射白名单：**第一轮 v2 仍只更新经验证的 1,538 个共享 backbone BN affine；冻结 source running statistics 和其他 detector 参数。真实同样本 CE oracle 在这组参数上能下降 CE，说明它有主任务可达方向，但不证明测试时自监督梯度可以找到该方向。没有证据支持扩到全模型或放开 BN running statistics。
- **强 CE Frozen 作为新起点和主比较：**用 v1 CE epoch 6 checkpoint 的结构/权重作为 K=0 基线。任何需 BYOL 的模型都须在同一 CE 初始化上重新训练/拟合辅助 head 或 adaptation map；不得把 Joint head 静默拼到 CE backbone 后当成已匹配对照。至少保留“现有 CE Frozen”“同一 CE 初始化的辅助分支 K=0”“同 checkpoint K=1”三项身份记录。
- **研究优先级：**先测证据约束的更新方向（候选 A），再考虑音频特定辅助目标（候选 B）。选择性适应先作为固定计算成本控制/消融：单样本若两个声学视图给出冲突更新就回退 Frozen；它能减少无谓更新并限制损害，但目前没有证据证明可提高 AUROC/EER。候选 B 只有在 A 不工作、并且源端能证明变换预测信号与检测 query 方向关联时才启动。

## 候选 A：Source-meta-learned BN update transport

### 假设及方法差异

v1 小型真实模型诊断中，共享 BN 空间的 BYOL 梯度与 same-sample CE 梯度余弦多数为负；同范数 CE oracle 的 CE 下降，而 BYOL 一步在该子集使 CE 略升。候选 A 不更换 BYOL 表征任务，而在 BYOL 梯度进入 BN 参数前学习一个**按 BN block 分组、可带符号的低维比例变换**：

```text
g_BYOL,b  →  α_b · g_BYOL,b  →  一次 episodic BN affine 更新
```

每个 BN block 一个有界标量 `α_b`（或一个全局标量作为最简对照），其值只由 source fit 的 bi-level CE query 元梯度学习；推理时输入样本无标签，加载固定 α。负 `α_b` 被允许但受界限约束，因为 v1 oracle 对照显示方向而不仅仅步长可能失配。它不是把 oracle CE 梯度带到测试时，也不使用目标标签、伪标签或在线记忆。

该候选借用 MABN/MT3 的辅助内层更新与主任务外层监督，但研究的具体变化是学一个低维**梯度传输映射**，不再只让 meta 初始化隐式吸收任务差异。仅凭本报告不能声称方法新颖；实施前需进一步查证 Meta-SGD、learned optimizer、gradient surgery/projection 及音频 TTA 的优先工作，并把相关相似项写进版本记录。

### 源端最小验证

1. 固定 CE epoch 6 为 backbone 和检测 head 起点。只用 `fit` 训练一个兼容 BYOL head/transport；共享 BN source buffers 固定。没有完整 XLS-R 重训练。
2. 依既有 assignment，从 fit 音频确定若干**任务共享**的声学变换：任务内 support A 和独立 query B 应经同一预注册变换族/参数抽样，以在 episode 内共享 channel shift；对照 episode 用原流程同分布视图。变换只在 fit 上构造，视图 recipe、seed、级别一并固定。不得按 source_val 结果反复新增扰动。
3. 外层损失仅为 B 的源端 CE（Cross-sample），验证时单独列 Same-sample 作为诊断；比较同 backbone、同 BYOL head、同 BN 参数、同任务顺序与更新次数的 raw BYOL `α=1` 和 learned `α_b`。以 source_val 选 epoch，`select` 只选预先限定的一个全局 clip 上限/门控阈值，`cal0` 只定阈值。
4. 先登记输出，再运行固定 source_val 子集的 K=0/K=1：报告 EER/AUROC、logit margin、CE、真假 pair 排序翻转、按 label 与固定 margin 难度分层、original/FIR 方向一致率；逐条统计 BN 更新范数、接受率和运行时间。
5. 只做一个 seed。若 learned map 的 source_val CE/EER 或预定声学条件无可辨的可靠改善，或普通音频损害超过预设界限，则停止此候选，不扩大 epoch、参数组或扰动搜索。

### 对照公平性

| 变体 | 初始化/可训练参数 | K=0/K=1 |
|---|---|---|
| CE Frozen | v1 CE epoch 6；没有 BYOL head | 仅 K=0 |
| CE + fixed raw BYOL | 相同 CE epoch 6 起点、同一辅助 head 训练预算、`α=1` | 同 checkpoint pair |
| CE + learned transport | 与 raw BYOL 完全相同初始化、head、数据、views、meta exposure、BN 范围；只多出分组 α | 同 checkpoint pair |
| BN CE oracle | 同一 CE backbone/BN 范围，使用源标签仅作机制上界 | 仅源端，不与无标签测试方法混称 |

先以两个独立 run 保存同一 CE 初始化引用；若辅助头训练导致共同 K=0 分类器状态不同，就同时给出各自 K=0/K=1 和共同 CE Frozen 主参照，不能只报 K=1。

## 候选 B：Audio transform prediction episodic adaptation

### 假设及方法差异

RawBoost 说明 raw waveform 的滤波、噪声及非线性扰动能丰富反欺骗训练，但它没有证明这些扰动适合作为测试时的梯度目标。候选 B 仅在候选 A 被 source gate 否决后，测试一个替代辅助目标：向每条 source fit 音频施加已知且固定的声学变换，辅助 head 预测“本视图用了哪种变换/强度”；内层仍每样本重置，只更新同一 BN affine，外层由独立、同变换任务的 source query CE 训练初始化。测试时自造规定视图，因此变换类别作为辅助伪标签来自本程序的变换操作，不读取样本真标签。

这和 TTT 的人为视图辅助标签思路相近，和 BYOL 的表示一致性不同；原创性与性能均未验证。与 RawBoost 源端监督分类增强的区别是：这里变换 ID 是内层自监督信号，主任务 query 只在源端元训练时指导更新。主要风险是扰动会抹掉 anti-spoof 线索，导致对 codec/fake traces 的不当不变性。

### 独立验证和停止条件

- 不叠加候选 A 的 α transport；直接比较 `CE Frozen`、同起点/同 head 预算的 `transform prediction` 和已有 raw BYOL 基线。
- 只选两至三种有明确物理解释的 deterministic perturbation 家族；幅度从既有 `config_source_gpu_diagnostics_20261008.json`/RawBoost 参考实现可支持的范围预先锁定。压缩 codec 依赖未核验时不启用。
- 对每种变换报告同 ID 原音频及受扰音频的 margin、CE、EER/AUC、pair 翻转和适应方向；分真/假、margin 难度报告。任何总体 source-val 改善若以某类/某声学条件明显恶化为代价，停止。
- GPU 成本与候选 A 同阶起步；如需要比 v1 Stage 2 更复杂的 waveform 增强/解码，必须先做小样本时间和峰值显存测量，再决定预算。未量测前不得把推测写为硬件实测。

## 共用协议和预计资源

### 固定角色和访问边界

- 不重新划分 assignment。`fit` 是唯一用于源端训练、元训练反向传播和机制 oracle 的 role；`source_val` 只选 epoch；`select` 只选择预先限定的方法参数；`cal0` 校准阈值。
- 固定 target90/final holdout 为不可访问。本提案本身无目标评分。后续若获得源端 gate PASS，再单独审查一次 v2 disclosed-access 四域开发评价，公开本轮看过的 v1 指标和版本差异，不能把 v2 称为未看目标标签的 confirmatory test。
- 不以源端指标单点正增益开始目标评价。至少要求同 checkpoint 配对下，source_val hard-margin 层和预注册声学扰动层都无明显退化，更新具有可复现的方向稳定性，并且输出/样本覆盖审计通过。

### 预期 GPU 与计算预算

v1 Stage 2 的真实日志记录 Cross/Same 每个 256 task epoch 各约 120 秒；阶段 3 逐音频 TTA 峰值 CUDA 记录约 2.709 GiB，但这不是新 meta 训练峰值。v1 full XLS-R Stage 1 曾每 epoch 约 2,000–2,600 秒/模型，v2 明确不重跑这项。

| 工作 | 初始上限 | 资源估计/状态 |
|---|---:|---|
| 固定 CE 初始化的辅助 head/source fit 预热 | 1 epoch，上限 1,813 update | 预计 2–5 分钟，需首次实测；此为估计，不是实测。 |
| 一个候选的 source meta 更新 | 2 epoch × 256 task | 依据 v1 Stage 2，约 4 分钟 GPU 时间；新代码先做单 epoch smoke 并重估。 |
| K=0/K=1 source probe | 固定 64 条 × 2 路径 | v1 实测 TTA 相对 Frozen 约 5.1×单样本时间；合计通常为分钟级，须记实际计时。 |
| GPU 显存预算 | 单卡，不和已有训练争抢 | 新训练 peak 尚未实测；启动前预留至少 24 GiB 空闲显存，先记录真实峰值再继续。该数值是保守准入门槛，不是模型峰值断言。 |

候选 B 若需更长任务或新依赖，不自行安装或下载；标 `NOT_RUN` 并报告资源/合同阻塞。任一 GPU 不可用时先保持 source-only CPU 可以完成的静态工作，完整二阶反向或长音频诊断标 `WAITING/NOT_RUN`。

## 进入任何后续目标开发评价前的必要条件

1. 新候选的实现和配置独立于 v1，并与现有 CE Frozen 文件、版本、数据角色逐项对齐；老结果只读复用。
2. `fit` 反向传播、`source_val` epoch 选择、`select` 参数选择、`cal0` 阈值校准的访问边界通过源端审计；target labels 完全在评分结束后处理。
3. 同一 checkpoint 的 K0/K1 成对分数完整，保存每样本 margin/score delta、判决翻转、fake-real pair 排序翻转、BN 更新 L2、耗时和退出码；CE-only 仍只有 K0。
4. 源端具有非饱和的预登记诊断层（使用现有角色而不重划），方法相对 CE Frozen 和 `α=1` raw BYOL 有可解释改善；效果不能只来自统一分数平移或工作点变化。报告配对不确定性、按类及声学条件的有害变化。
5. 停止条件事先固定：工程错立即停；资源门不通过则 WAITING；CE Frozen 已饱和且新方法无source端可靠变化就不开放目标开发；结果为负保留并停止扩展当前方法。
6. 若将来进入四域 target10 开发，须明确标为 **disclosed-access target development evaluation**，只执行一次冻结版本评价，不临时调参；target90/final holdout 仍不可访问。

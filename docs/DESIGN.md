# EP-TTA：AI Coding 设计与实施方案 v0.1.0

## 本机开发 · 远程数据接入与自主训练 · Ubuntu 双卡 A6000 · 适应、基线与实验验收

> **日期**：2026-09-13  
> **项目 / 规范 / 接口初始版本**：`0.1.0`；展示为 `v0.1.0`。方法标识固定为 `ep_tta`，版本单独写入元数据。  
> **当前阶段**：`NOT_STARTED`。AI Coding 尚未开始；本文是首版开发合同，不是已实现代码、已训练权重或已完成实验。  
> **权重要求**：AASIST、SSL-AASIST 等鉴伪检测器由本项目在远程服务器自行训练；结构依据原论文与作者开源实现。不得以下载他人已训练的鉴伪 checkpoint 代替本项目训练。SSL-AASIST 可以使用通用语音 SSL 预训练参数初始化前端，再训练本项目鉴伪模型；不要求从零预训练通用 XLS-R。  
> **已知资源**：本机只写代码；真实数据、预处理、训练、缓存、评分与评价在远程 Ubuntu 完成；2 × A6000，每卡 48 GB。CPU、内存、磁盘、驱动、互联和具体路径待远程预检。  
> **候选数据**：ASVspoof 2019 LA、ASVspoof 2021 DF（另预留 LA）、WaveFake、Codecfake（Xie）、In-the-Wild。实际目录、label/协议格式、来源映射、预处理需求、可用规模及增量新增规则尚未核验。  
> **设计原则**：现在固定内部接口、权限和科学合同；服务器上的真实布局与参数通过可审核配置补全，不猜格式、不伪造资源、不把缺本机数据当成开发阻塞。  
> **交付语言**：设计、进展和实验报告为中文 Markdown；代码与配置键为英文；关键术语附英文。  
> **单一依据**：本文件保存为 `docs/DESIGN.md`，独立可用。所有阶段状态初始为 `TODO`，只有后续真实执行证据才能变为 `PASS`。

<a id="navigation"></a>
## 阅读导航与立即开始

| 当前任务 | 先读内容 | 首次交付 |
|---|---|---|
| 本机从零开始 | [边界](#s0)、[架构](#s2)、[接口](#s7)、[任务](#s13)、[启动指令](#s15) | 项目骨架、配置、数学与接口测试；不访问真实资源 |
| 设计数据未知项和增量接入 | [数据合同](#s3)、[配置](#s8)、[增量缓存](#s9) | 插件、待核验模板、模拟协议、快照与分组检查 |
| 自行训练检测器 | [模型训练](#s3-training)、[训练接口](#s7-training)、[训练配置](#s8-training)、[训练硬件策略](#s10-training) | AASIST / SSL-AASIST 的架构、训练入口、选择和导出合同 |
| 实现 EP-TTA 和机制基线 | [数学](#s4)、[机制对照](#s5)、[方法移植](#s6)、[参考代码](#appendix-a) | 串行参考、批量等价、独立方法注册 |
| 远程实际补全与执行 | [任务](#s13)、[命令](#s14)、[远程指令](#s15) | 数据合同核验、训练、自有 checkpoint、冻结导出、后续实验 |
| 审核项目交付 | [验收](#s13)、[初始发布](#s16)、[测试模板](#appendix-d) | 实际命令、日志、未运行项及科学结论边界 |

**唯一默认推进顺序**：本机接口与数学开发 → 远程资源盘点 → 协议/标签/预处理/来源分组审核 → 训练前封存划分 → 自行训练源检测器 → 源验证集选 checkpoint → 冻结和导出 → 特征缓存 → EP-TTA 与基线先导 → 封存 → 最终评价。

**第一轮 AI Coding 只完成 L0–L2。** 不要求一次写完所有训练器和25项方法；但源模型训练与远程数据插件属于必须规划并实现的主流程，不是“可选的预训练权重缺失回退”。

---

<a id="s0"></a>
## 0. 执行边界、优先级与状态

### 0.1 科学目标与非目标

先依据论文/作者代码自行训练源域鉴伪检测器，再实现可插拔的测试时适应模块（test-time adaptation, TTA）：在训练和选模结束后冻结最终语句嵌入提取器与线性分类头，仅在源域处理响应子空间内做逐样本、小幅度更新。研究目标是检验这种调整能否缓解正常声学变化，同时减少未知伪造被错误适应为真实的风险。

第一版不包含持续学习、跨样本目标记忆、局部篡改定位、新的跨层检测器、测试伪标签训练、LLM 解释或智能体。允许不同方法作为**明确注册的基线**使用其原有机制，不能把基线限制偷换成 EP-TTA 的限制，也不能把扩展偷偷加到主算法。

### 0.2 必须通过机器检查的合同

| 合同 | EP-TTA / 同适应器主通路要求 | 验收依据 |
|---|---|---|
| 标签与分数 | `0=bonafide`、`1=spoof`；`s>tau` 判伪造 | 映射测试；禁止按目标性能反转分数 |
| 更新对象 | 每个检测单元一个 `R:[r,r]`；无偏置 | 参数枚举；每条恰为 r² 个元素 |
| 适应器 | `z + U R Uᵀ z`；固定正交 U | 行列向量与零初始化测试 |
| 主目标 | `L_view + lambda_keep*L_margin`；只做 Frobenius 球投影 | 不额外叠加 L2、熵、KL、伪标签 |
| 冻结范围 | E0、h0、U、源锚点和所有运行统计冻结 | 参数/buffer/mode 前后核验 |
| 重置 | 每个样本 R=0，优化器和所有目标状态清空 | A→B→A、打乱、尾批次测试 |
| 信息权限 | 目标标签、攻击 ID、干净版本、历史和整体统计不可见 | 类型隔离、访问审计、标签防火墙 |
| 视图 | 原视图加两个固定探测视图；每个 episode 只编码一次 | 种子与缓存 ID；不逐步重采样 |
| 主输出 | K 步后对原输入视图评分 | 不平均视图，不挑最佳步数 |
| 停止/回退 | 固定 K；只对规定的数值异常回退原分数 | 不按预测变化、熵或审核结果回退 |
| 并行 | 多个独立 R；不做跨样本/跨卡梯度归约 | 串行、批量、分片结果对应 |
| 比较身份 | 机制对照、已发表算法移植和原论文结果分开 | `comparison_track`、审计表 |
| 源模型 | 鉴伪任务参数由本项目训练；SSL通用预训练参数与任务参数分开 | 训练run、数据快照、初始化来源、checkpoint hash |
| 数据接入 | 真实格式须经远程审核；未核验字段不能变成假默认值 | contract状态；raw→canonical映射与拒绝记录 |
| 增量数据 | 不重划已有ID；新来源冲突先隔离；产物按依赖失效 | snapshot diff、来源组冲突和血缘审计 |
| 环境 | 本机不需要数据、权重、CUDA 或服务器连接 | 本机 profile 禁止真实 I/O/自动 SSH |
| 结果 | 未运行写 `not_run`；错误和负结果不删除 | 覆盖率、运行状态、原始分数封存 |

上述冻结和 R-only 要求**不约束**需要更新上游参数的已发表方法移植；这些方法走 §6 的独立计算通路与状态合同。全项目共同遵守标签隔离、数据范围、无虚假结果和权限规则。

### 0.3 优先级和进展状态

优先级：`P0` 最小闭环；`P1` 创新性机制比较；`P2` 已发表方法竞争比较；`P3` 额外数据/第二模型。它们表示执行次序，不表示 P2 可以在宣称优于已有方法时省略。

项目整体初始状态为 `NOT_STARTED`。任务状态统一为 `TODO / IN_PROGRESS / PASS / FAIL / DEFERRED_REMOTE / BLOCKED_CONTRACT / BLOCKED_RESOURCE / NOT_SELECTED / NOT_RUN`。`PASS` 必须附真实命令、退出码和日志位置。缺少本机真实数据是 `DEFERRED_REMOTE`；只有远程已选任务缺必需资源才是 `BLOCKED_RESOURCE`。

尚未核验的真实数据格式、label映射、预处理或划分阻塞真实执行时记 `BLOCKED_CONTRACT`；不是代码失败，不允许自动猜测消除该状态。工程实现、实验完成、科学假设被支持分别记录。程序正确且结果为负，不应被当作必须修到结果变好的 bug。

---

<a id="s1"></a>
## 1. 研究假设与必须排除的替代解释

| ID | 待检验主张 | 必要比较 | 不足以支持该主张的结果 |
|---|---|---|---|
| H1 | 源域处理响应方向有独立价值 | 响应 U / 源嵌入 PCA-U / 多种子随机 U | 只优于某一个随机种子 |
| H2 | 单侧间隔保持更好地协调纠错与损害 | 无保持 / L2 / logit / Fisher / margin | 保持项没激活；或普通知识保持同样好 |
| H3 | 逐样本适应有必要 | 固定衰减 / 固定源适应器 / 多视图融合 | 静态方案解释全部收益 |
| H4 | r² 个自由度有必要 | 完整矩阵 / 动态标量 / 可选对角矩阵 | 单标量已达到同样效果 |
| H5 | 收益不只是工作点改变 | 固定分数偏移、固定阈值 FPR/TPR 与 AUROC/EER | 共同压低伪造分数，未知攻击检出下降 |
| H6 | 相比成熟 TTA 有额外价值 | Tent、SAR、MEMO、EATA、T²A 音频移植 | 只比较无更新模型和本文消融 |
| H7 | 没有抹除正常重建与生成的区分 | 来源可核验的 CoRS/CoSG 控制 | 两组一起被判为真实 |

H7 是扩展，不要求现有 Codecfake 数据天然满足所需来源匹配。文献新颖性（novelty）还需由文献与方法差异说明，不能由一张性能表“证明”。MEMO、EATA 与 T²A 分别已有单样本多视图适应、抗遗忘保持和鉴伪专用适应思想。[R4–R6]

**最小有意义先导**：先回答 H1–H5，确认不是静态抑制、额外源监督或更多视图造成的表面改善，再投入大型模型更新基线。

---

<a id="s2"></a>
## 2. 系统架构与产物依赖

### 2.1 本机与远程职责

```text
本机：代码 / 论文与开源结构审计 / 配置模板 / 合成协议 / CPU测试
                              │ 固定commit或源码包
                              ▼
远程：只读原始数据、协议、通用SSL初始化权重 + 可写训练/产物目录
  资源盘点 → 原始格式适配 → 标签/预处理/分组审核 → 不可变数据快照
                              │ 训练前封存source_val与所有测试角色
                              ▼
  自行训练AASIST或SSL-AASIST → 仅source_val选模 → 自有checkpoint
                              │ 禁止读取目标测试结果挑模型
                              ▼
  冻结E0/h0 → 原架构前向/wrapper等价 → 导出头与baseline_id
       ├── GPU0：冻结模型，独立分片提取
       └── GPU1：冻结模型，独立分片提取
                              │ 最终语句特征缓存
                              ▼
  CPU/GPU：U/tau0/anchors → EP与机制基线 → 无标签分数封存
  独立GPU：需要梯度的已发表算法音频移植 → 无标签分数封存
                              ▼
  独立评价器读取目标标签 → 指标/翻转/置信区间/Markdown报告
```

自有训练输出和通用SSL初始化权重分目录；不覆盖原始数据、协议或预训练权重。大文件、源特征、私有路径及SSH凭证不进Git，不上传外部服务。本机允许读取公开论文或小型作者源码进行结构核验，但不自动下载模型/数据或连接服务器。

### 2.2 三种计算通路

| `route` | 可用缓存 | 可更新对象 | 使用场景 |
|---|---|---|---|
| `feature_cache` | 最终语句嵌入 `[N,d]` | R 或其限定形式 | EP、熵/保持替换、静态与标量基线 |
| `frozen_frontend` | 严格位于第一个可更新模块之前的特征 | 后端、归一化等 | 经审计的受限音频移植 |
| `waveform_update` | 波形/同一探测参数；不得把最终嵌入当梯度路径 | 已声明的前端或整个模型 | 完整参数更新及端到端成本 |

缓存与在线数值等价不意味着信息权限扩大：提前提取目标特征后，每个适应调用仍只能读自己的样本。不能对全目标缓存估计均值、PCA、聚类、阈值或 Fisher。

### 2.3 环境隔离

推荐至少区分两种环境职责：`eptta-core` 负责数值、配置、缓存评分和报告；`eptta-baseline` 负责构建、训练和加载本项目 AASIST/SSL-AASIST。可交换 `.npy/.npz + JSON`，不跨环境交换任意 Python pickle 对象。

核心工程的 Python 语法基线为3.10及以上；现有 fairseq/检测器环境可能更旧，**不能默认在旧环境安装并导入整个新核心包**。为此增加独立进程桥接（subprocess bridge）：核心 CLI 解析配置和产生无标签作业 JSON，调用已选 `EPTTA_BASELINE_PY workers/baseline_bridge.py --job ...`；旧环境只加载其兼容的模型/解码/处理模块，通过 `.npy/.npz + JSON` 写回结果。

bridge 及其兼容模块以 Python 3.7可解析语法为最低合同，不使用`list[str]`、`Tensor | None`、match/case或核心包依赖；核心通过子进程退出码/日志/产物hash验收，而不是跨环境传递可微Tensor。源模型训练与完整模型更新基线的前向、反向、优化器和检查点操作全部在同一所属worker环境完成，不通过IPC“传梯度”。bridge应一次加载模型、处理多个块，不每条音频启动进程。

本地基础 CLI、配置/清单模块使用轻依赖和延迟导入（lazy import）；没有 torch 时仍可做结构检查，张量测试标 `NOT_RUN`。真实版本组合由远程导入与前向验收后生成 lock，不把“最新版本”写成已验证兼容。源码不得自动更改驱动、系统 Python 或系统 CUDA。

### 2.4 有向产物依赖：先训练，再适应

```text
raw_inventory + approved_raw_contract + label/group_policy → canonical_snapshot
canonical_snapshot + approved_split_plan → fit / source_val / select / cal0 / audit / tests
架构源码commit + training_recipe + fit + source_val + 可选SSL初始化 → training_run
training_run + source_val选模规则 → selected_checkpoint + training_manifest
selected_checkpoint + approved_eval_profile + wrapper → frozen_bundle + baseline_id
fit + source_probe + baseline_id → source_pair_cache → U
cal0 + baseline_id → frozen_scores → tau0
fit候选 + head + tau0 → anchors(M, labels, s0, m0)
anchors + U + head → Fisher；fit源配对 + U + M → fixed_source_adapter
select → 适应方法参数；cal1(可选) → 管线固定后的tau_j
不可变资源包 + 当前样本视图 → scores → 评价标签 → metrics
```

`source_val`只负责源模型早停/epoch选择；`select`负责EP与基线参数选择，两者职责不同。`cal0`用于tau0和锚点定义，不是从未访问的评价数据。所有角色在源训练前划定；样本量不足时提交明确的合并角色建议和独立性限制，不在代码中自动合并。

任何节点写出父节点hash。更换源checkpoint后，U、头、锚点、阈值、Fisher、固定适应器和相应特征/分数不能继续按旧bundle使用；无需为每个TTA超参数重新训练源模型。

---

<a id="s3"></a>
## 3. 远程数据合同、增量接入与源检测器自主训练

### 3.1 候选数据与可变的实际使用方案

| `dataset_id` | 建议用途，不等于已经可用 | 远程需要核验的关键内容 |
|---|---|---|
| `asvspoof2019_la` | 首选源训练、源验证、适应选择、校准与受控测试 | 实际train/dev/eval位置；官方协议或用户转换label；跨角色原始来源组 |
| `asvspoof2021_df` | 首选外部确认，不默认参与源训练 | DF轨道、keys/subset和音频对应；跨年份来源映射 |
| `asvspoof2021_la` | 可选信道压力测试，默认关闭 | LA专用协议；不能套DF解析规则 |
| `in_the_wild` | 外部采集条件验证 | 实际音频布局、元数据列、身份/来源映射与标签语义 |
| `wavefake` | 扩展声码器/语料变化验证 | 发行版本、生成来源；匹配的LJSpeech/JSUT真人资源是否存在 |
| `codecfake_xie` | 编解码相关任务与压力测试 | Xie发行范围、协议、真假标签语义、codec/生成来源证据 |

输入别名`asvspoff2019 / asvspoog2021 / CodecFake_ Xie / In-the-wild`可规范化并提示，内部使用表中ID。数据顺序建议2019 LA→In-the-Wild/2021 DF→WaveFake/Codecfake；并非五个数据集都必须先下载齐。除经明确登记的后续多源训练实验外，外部集合不自动加入`fit`。

ASVspoof 2021区分LA/PA/DF，提供keys和来源映射入口；WaveFake真假对照应匹配其原语料；数据集年份或名称不同不证明独立。[R7–R9] 上述事实用于设计适配器，不能证明服务器已有相同目录或列格式。实际资源分配必须在查看对应测试性能之前确认。

### 3.2 已确定的内部合同，与待核验的外部信息

**现在可固定**：内部真假编码`0=bonafide, 1=spoof`；分数方向；规范清单字段；角色权限；模型和缓存的血缘；不泄漏目标信息；增量快照不可改写。**现在不能固定为“真实值”**：远程文件名、分隔符、列号、原始0/1的含义、采样率、声道、音频格式、补齐/截断策略、源组映射、最终划分比例和预处理产物是否已存在。

| 未知项 | 现在预留 | 远程补全产物 | 缺失时行为 |
|---|---|---|---|
| 目录/协议来源 | `LayoutInspector`，root和协议候选配置 | `inventory.json`、文件计数与抽样记录 | 仅阻塞选中数据，不阻塞本机 |
| label原始格式 | `ProtocolAdapter`，columns/encoding/delimiter/schema | `raw_contract.lock.json` | 可以盘点；不猜列后训练 |
| 真假语义 | `LabelMapper`，原值→规范值映射 | `label_policy.lock.json`、映射覆盖表 | 未知值隔离；不能默认算spoof |
| 内容/身份/来源 | `ProvenanceResolver`，原文档或映射文件 | 来源关联表、`group_quality` | 限定泛化声明；跨角色风险阻止封存 |
| 预处理 | `PreprocessPolicy`，训练/验证/推理分离 | `preprocess.lock.json`及音频统计 | 可结构校验；不能正式训练/缓存 |
| 初始与新增划分 | `SplitPlanner`、`IncrementalReconciler` | split plan、快照diff、冲突清单 | 新数据先隔离，不自动重划 |
| 模型训练配方 | `TrainingRecipeResolver`、作者commit与patch | `training_recipe.lock.json` | 架构测试可做；完整训练需锁定 |

外部合同状态为`UNRESOLVED → PROPOSED → APPROVED → LOCKED`。探测程序可以提出列映射或预处理候选，**不能自行将推测当已核验**。批准需有检查记录、抽样证据、检查者和时间；这属于项目流程记录，不是密码学信任证明。纯本机结构验证允许这些字段为`null`；真实运行的`validate --stage ...`只检查该阶段必需的已选合同，并一次列出未解决项。

### 3.3 原始记录→规范记录→分组快照→推理输入

使用四层数据边界，而不是在Dataset类中混合读label、猜路径、划分和增强：

```text
原始协议/CSV/TSV/JSON(L)/用户sidecar（只读）
  → RawRecord + 原文件位置/行号/原字段 + 解析问题
  → CanonicalRecord + 映射依据/标签策略/来源质量
  → StagingSnapshot（暂未分配角色）
  → 不可变DatasetSnapshot + 已审核角色分配
  → SourceTrainManifest / SourceEvalManifest / TargetInputManifest
```

规范主清单仅供数据构建、源训练和评价侧读取。下例为**模拟fixture**，不是服务器格式或已核验样本：

```json
{"schema_version":"0.1.0","sample_id":"opaque-sample-001","dataset_id":"asvspoof2019_la","dataset_release":null,"snapshot_id":"fixture-snapshot","root_key":"asvspoof2019_la_train","audio_relpath":"example/path.flac","raw_record_ref":{"file_key":"protocol_fixture","row":1},"original_label":"bonafide","canonical_label":0,"label_policy_id":"source_authenticity","label_mapping_status":"verified_fixture_only","official_split":"train","split_role":"fit","source_group_id":"fixture-group-001","group_quality":"synthetic_fixture","speaker_id":null,"generator_id":null,"generator_family":null,"parent_id":null,"treatment_id":"identity","codec_id":null,"input_sha256":null,"status":"fixture_only"}
```

`StagingSnapshot`允许`split_role=unassigned/quarantine`，仅供审核和划分，不允许进入训练或适应。`DatasetSnapshot`经审核发布后才绑定科学角色；角色校验按阶段区分，避免还没有规范数据就要求先给出split plan的循环依赖。

真实的无标签或不能识别标签记录使用`canonical_label=null`并标原因。标签缺失不等于真实。训练只允许标签明确且角色合规的记录；预先声明的无标签评分可以使用null标签，但不可生成EER、FPR等需标签指标。研究方新造的模拟文件必须留`fixture_only`，禁止混进正式snapshot。

推理清单投影为下列结构；示例hash为占位，真实执行前替换：

```json
{"schema_version":"0.1.0","sample_id":"opaque-sample-001","root_key":"asvspoof2021_df","audio_relpath":"example/path.flac","input_sha256":null,"decode_profile_id":"ssl_aasist_eval_unit","probe_profile_id":"probe_default"}
```

剥离目标标签、原始协议行、生成器/说话人、处理强度、原始干净音频路径和父指针。适应核只接收本条`features + opaque ID + 固定源context`，不接收root或整个manifest；I/O进程可以看路径，但不得以路径类别分支选择方法。

`sample_id`由登记系统稳定分配，不编码真假/攻击，不因移动根路径而变化；相同内容的别名通过dedup表关联。路径必须安全相对化，拒绝`..`或解析后越界链接。大清单用流式迭代/SQLite索引，不让每个loader worker复制全量字典。

### 3.4 可插拔协议、标签与预处理接口

至少提供`official_asvspoof_candidate`、通用分隔文本、JSON/JSONL和外部sidecar适配接口；CSV列名与无表头列序号分别处理。先在本机用合成fixture覆盖，不预设实际使用哪一个。

`RawContract`须记录：`format / encoding / delimiter / header / columns或json_paths / record_id / audio_path_rule / label_field / allowed_values / missing_policy`。某数据需要从目录推导信息时，只能使用经过审核的规则，记录来源；不能把“目录名字含fake”作为未经批准的label规则。

`LabelMapper`必须保留`original_label、canonical_label、mapping_reason、policy_id、policy_hash`。检查一对多映射冲突、未映射值、同一内容真假冲突及映射覆盖比例。输出`parse_errors.jsonl、label_conflicts.jsonl、quarantine.jsonl`；不能用`except: continue`丢弃失败数据。严格模式有错误就阻塞；允许隔离后缩小范围时，必须重新批准清单并报告排除分布。

`PreprocessPolicy`区分：

- `decode`：采样率读取、解码器、声道处理、重采样算法、幅度规则；音频内容不能被原地覆盖。
- `train_unit`：训练采样窗口、补齐、随机种子与作者配方中的数据增强（data augmentation）。
- `eval_unit`：固定检测单元、确定性截断/补齐/长音频处理；用在source_val、校准、审计和测试。
- `source_probe / target_probe`：只在固定检测单元上构造EP探测视图，不等于训练增强。
- `evaluation_treatment`：受控压力测试的处理，先生成观测文件；其干净父样本和处理参数不传给适应器。

16 kHz、64600样点是参考作者代码的**候选默认**，不是对远程原始格式的断言。[R1,R16–R17] 编码入口若需要该长度，应由已锁定的profile实现；禁止不同方法各自裁剪成不同内容。训练随机裁剪与推理确定性裁剪可以不同，但必须分别写入配方。静音、坏音频、多声道、超长/超短、削波和原始归一化状态要有明确处理与统计。

物化预处理文件是可选加速模式：输出到`work/derived_audio/<profile_hash>/`，保留原始父hash和profile；默认按需解码，评估CPU/I/O后再决定物化。更改预处理不得复用旧数值缓存。

### 3.5 训练前划分、角色权限与选模隔离

| `split_role` | 唯一主要用途 | 禁止事项 |
|---|---|---|
| `fit` | 源检测器梯度训练；训练完成后估计U/M/Fisher等 | 与最终测试同源而声称独立 |
| `source_val` | 源模型checkpoint/epoch选择和预先规定的早停 | 梯度训练；访问外部目标；用作最终测试 |
| `select` | EP及基线超参数、视图候选与开发诊断 | 重新训练源模型、混入U/M、当最终结果 |
| `cal0` | 源模型冻结后确定tau0 | 选择源checkpoint或适应超参数 |
| `audit` | 仅评价未缓存源样本的受损情况 | 进入loss、回退或选步数 |
| `control_test` | 受控A–D确认性评价 | 调模型、调方法 |
| `target_test` | 外部或明确未见来源的确认性评价 | 源训练、选模、重估U/M/阈值 |
| `cal1` | 可选，方法固定后的各管线工作点校准 | 再选择方法 |

建议从2019 LA官方train提供`fit`候选，从合适的源开发资源中预留其余源角色；**不在本机写死比例**。`split_ratios=null`保持待核验；服务器根据类别数、真实来源组、身份可分性和校准样本需要生成提案，再批准。选择一份合法的较小先导划分优于强行凑所有角色。

必须在第一次源模型训练前封存`split_plan.lock.json`。如果`source_val`和`select`确因样本不足需要共用，先明确登记为共享开发设置，报告限制；`cal0 / audit / control_test / target_test`不能随意共用。整个dev已经用于选模后，不能事后切出“独立校准/测试”并恢复独立性。

同源真人及生成/增强/codec派生样本组成关联组；跨数据集也合并实际同源组。缺来源映射时保留`group_quality=unknown`，不要给每条新ID后宣称同源独立；身份互斥只有核验后才声明。检查train/source_val/select是否包含拟宣称留出的攻击、生成器或家族；未知性区分`attack_id_holdout / generator_holdout / generator_family_holdout / unknown_metadata`。SSL预训练语料暴露与下游鉴伪训练暴露分别记录，不承诺无法验证的完全无预训练重叠。

### 3.6 增量新增：追加快照，不静默重划既有实验

**这里的增量是数据管理（incremental data ingestion），不是EP测试时持续学习。** EP仍每条重置，不因后来加入数据而使用跨条记忆。

接口采用`inspect → parse → normalize → deduplicate → reconcile → propose → approve → commit_snapshot`，其中只读阶段不修改已封存产物。

1. **只处理新增或变化条目**。记录`arrival_batch_id、source_release、file_checksum、raw_contract_hash`；相同文件hash和同一协议记录重复导入应幂等（idempotent）。路径移动只更新资源绑定，不产生重复样本。
2. **既有分配保持稳定**。旧snapshot的成员、label、组和角色不可覆写；新snapshot以`parent_snapshot_id`记录父子关系，不对全量数据再次随机划分。新组才按已审核规则分配；默认新数据进入`unassigned/quarantine`，不是默认进入fit。
3. **已知组新增派生样本**。原则上继承已批准的组角色，但新增数据集的角色权限与组角色冲突时隔离，不跨集自动放宽权限。例如新外部测试别名实际指向fit来源，就不能把它算独立外测。
4. **新证据合并旧组时**。若新映射发现fit与target_test、source_val或cal0实际同源，标记`lineage_conflict`；不自动搬动旧样本。封存实验登记潜在泄漏及受影响产物，停止相应强独立性声明；另建经审核的新方案。
5. **标签更正单独事件**。记录old/new、理由、证据和受影响sample IDs；不原地改旧label文件。依赖该标签的选模、阈值、anchors、Fisher及指标需要重新评估；原分数可在同数值身份下保留，但结果报告必须引用新标签snapshot并说明更正。
6. **划分策略不能随新增数据自动改变**。保持测试成员不变的`append_source`与保持训练模型不变的`append_external_test`是不同操作；需要重划时创建新的`experiment_id`和split snapshot。已经看过结果的测试集合保留访问记录，不能因快照名称变化变成新测试。

| 变化 | 可复用 | 必须新建/核验 |
|---|---|---|
| 新增外测数据，模型/预处理不变 | 原模型、U/M/阈值、已有数值特征 | 新输入特征、coverage、测试清单与新增评价计划 |
| fit新增但暂不重训 | 已冻结实验完整bundle | 新数据仅在新计划中待用；不自动污染旧U/M |
| fit新增并重训 | 原始音频与无关解码缓存 | 新checkpoint→全部模型相关特征/头/U/M/阈值等 |
| source_val改变 | 不受影响的原始数据 | 选模范围和checkpoint选择重新审计；需新训练/选择run身份 |
| 只改路径 | 内容和数值缓存 | 安全路径解析、资源hash和存在性 |
| 只改标签 | 数值条件相同的冻结分数/特征 | 角色/任务映射；依赖标签的源产物或指标 |
| 改decode/probe | 不变原始音频 | 受影响视图与特征及后继产物 |

提交snapshot使用临时目录→校验→原子发布，输出`added/unchanged/duplicate/conflict/quarantined/relabeled`计数和hash；失败不出现半成品active清单。别名索引、来源组与父子关系放CPU/磁盘索引，不需要加载GPU。

### 3.7 Codecfake_Xie的独立任务语义

此ID只对应Xie作者项目，不用其他同名资源替换。[R10] `codecfake_xie_official`保留已审核发行版原标签；`source_authenticity`表示本文“允许正常处理下真人来源 vs 目标生成/转换”的任务。二者label policy版本均由`0.1.0`起步，但政策不同不能混算一张EER。

将真实codec重建标成CoRS，需要证明该操作属于任务允许的处理且不改变关注的身份/内容；将样本标成同codec生成CoSG，需要生成流程证据。普通TTS经codec再编码不能自动当作同内部解码过程。缺失这些证据时保留官方任务，H7标`not_run_missing_provenance`；不得为满足反证实验编造配对或静默改标签。

<a id="s3-training"></a>
### 3.8 源检测器必须自行训练：初始化不是现成鉴伪权重

本项目有两个独立阶段：**源域监督训练（source-supervised training）**与**目标无标签测试时适应**。只有第二阶段执行E0/h0冻结规则；不能把冻结要求错误应用到尚未训练的源模型。

| 模型/阶段 | 结构依据 | 初始化 | 本项目必须执行的训练 |
|---|---|---|---|
| `aasist_source` | AASIST原论文、`clovaai/aasist` | 按作者结构初始化可训练参数；保留其固定或解析初始化组件 | 用批准fit训练源任务参数，用source_val选checkpoint |
| `ssl_aasist_source` | Tak等人的SSL-AASIST论文、`TakHemlata/SSL_Anti-spoofing` | 通用XLS-R/wav2vec SSL权重初始化前端；后端/分类头按作者实现初始化 | 主配方联合微调SSL前端及鉴伪后端/头，使用本项目fit/source_val |
| `ssl_aasist_frozen_frontend` | 同源结构的受限训练变体 | 通用SSL前端固定，后端/头初始化 | 可作为资源受限先导；必须独立model_id，不替代联合微调主配方 |
| 后续其他检测器 | `ModelFactory + TrainingRecipe`插件 | 显式声明pretrained/random部分 | 同样要求自有训练记录与可追溯checkpoint |

**允许**下载/挂载可信的通用语音预训练权重作为初始化，但须显式配置、授权和核验hash；**不允许**将作者提供的`best_SSL_model*.pth`、现成AASIST鉴伪权重等当成本项目已训练模型。主实验默认`allow_external_task_checkpoint=false`。自己训练得到的中断checkpoint允许恢复；不得将resume接口用作不明任务权重注入。

“自行训练SSL-AASIST”不等于从零预训练大型XLS-R。原SSL-AASIST研究使用通用SSL前端并进行任务微调；AASIST作者仓库也提供训练入口。[R1,R16–R17] 两者分开记录`pretraining_provenance`和`task_training_provenance`。现阶段这些任务均为`TODO`，不存在已完成权重。

### 3.9 架构复用与训练配方审计

本机实现两个训练插件，读取原论文、作者`model/config/train`源码，固定仓库commit并保存最小patch。不凭方法名称从头猜网络，不以简化MLP冒充AASIST，不把图像AASIST同名网络误接进来。

每个`docs/models/<model_id>.md`至少记录：原论文/仓库、完整commit、文件和许可；前端与后端组成；模型输入和输出；原始类别次序；实际embedding维度；损失、采样、优化器、scheduler、增强、epoch、checkpoint选择；与作者实现的协议/工程偏离及理由。

**结构与训练配方是不同对象**：可以为了防泄漏替换作者数据加载与选模入口，但保留结构；不能声称更改了验证角色、精度、有效batch后仍逐项严格复现作者所有结果。作者配置中训练期间访问eval的开关必须禁用；本项目训练worker永远不接受control_test/target_test，不能每个epoch打印最终目标EER。

可记录以下已核对的作者**启动参考**，不是实际服务器已批准配方：AASIST配置包含batch 24、100 epochs、Adam基础学习率1e-4、weight decay 1e-4和cosine schedule；SSL训练入口提供batch 14、100 epochs、学习率1e-6、weight decay 1e-4与带权交叉熵，增强选择由脚本参数决定。[R16–R17] 本项目最终的`training_recipe.lock.json`需在服务器根据固定commit、数据合同和内存预检确认。学习率和训练预算只能在允许的源开发范围选择；不得根据2021 DF或In-the-Wild表现挑训练增强。

原仓库的native标签可能与本项目相反。训练先将`canonical_label`按`class_index_map`转成native index，再计算对应类别权重；不要直接复制`[0.1,0.9]`这样的向量而不审计其语义。建议recipe按类别名保存权重，再由adapter按真实索引生成向量。checkpoint必须包含映射；导出分数始终为`fake_logit-real_logit`。

### 3.10 训练器的功能与硬性放行条件

统一流程：

```text
resolve-training-recipe → validate fit/source_val permissions → init-model
 → train-smoke（独立测试目录） → 正式训练 → source_val选模
 → finalize-training → export-frozen → verify wrapper/head parity
```

`train-smoke`完成少量真实fit样本前后向、损失与梯度检查、checkpoint写读和验证流程；少量训练不能冒称基线训练完成。正式训练遵循封存recipe的最大epoch/step及早停规则。默认checkpoint选择为**source_val上最小EER**，相同值选较早epoch；这是项目选模规则，不声称与所有作者代码相同。EER评价使用完整无重复source_val，验证无随机增强，loss按全样本/权重正确累计。类别不全导致EER未定义时阻塞选模，不偷偷换训练准确率。

`fit`可以有随机增强，随机性按training seed/epoch/sample/augmentation实例控制并记录；保存训练抽样实际来源与重复/丢弃政策。训练日志包含step、epoch、train loss、source_val loss/EER、LR、梯度/溢出、训练时长、每卡显存、样本覆盖和checkpoint选择原因。不得把`select/cal0/audit`用于反向、训练早停或隐藏的checkpoint筛选。

**检查点（checkpoint）**至少包含：model state、optimizer/scheduler、AMP scaler（启用时）、epoch/global step、随机数状态、sampler/数据顺序状态、训练配方hash、架构commit和patch hash、数据snapshot/split/preprocess/label policy hash、class index map、初始化来源和训练seed。每个完成验证的epoch必须写入独立且不可覆盖的`checkpoints/epoch-NNNN.pt`及hash sidecar，不自动裁剪；`last.pt`用于恢复，`best.pt`用于操作便利，两者是指向相应epoch内容的原子别名。选模记录和导出必须引用不可变epoch路径及其确定hash，不使用可变`last/best/latest`路径作为科学身份。

恢复默认保证同一配方和相同数据快照下的**epoch边界恢复**；step级恢复须实现并测试sampler、增强和worker状态，否则明确记为非逐步等价恢复。fit成员改变或代码改变不能用`resume-exact`，需新run和显式warm-start记录。不要把近似恢复写成位级可重复保证。

所有TTA方法在同一模型/seed实验中共享同一个自有checkpoint。初轮每个模型一个训练seed；正式关键结果可增加独立训练seed，并与U/M/probe种子分开。不能为每个TTA方法挑不同源checkpoint，也不能只让主方法获得更好的底座。

### 3.11 冻结导出与EP接入

训练完成后生成只读`FrozenModelBundle`，包含`model_id、baseline_id、selected_checkpoint_sha256、training_run_id、fit/source_val_snapshot_hash、recipe_hash、init_provenance、eval_preprocess_hash、class_index_map、head、embedding_dim`。该产物通过审计后才允许构建U、M、tau0和目标特征。

适应位置为**最后线性层的实际输入**。若`logits=zWᵀ+b`，导出`w=W_fake−W_real`、`b0=b_fake−b_real`。比较同一自有checkpoint在原架构前向、wrapper、导出线性头和R=0路径的logits/分数一致性。不用重新训练的线性探针代替原分类头，也不默认将1024维SSL帧输出当作最终语句嵌入。

非线性最终头必须另列扩展；主版只支持已验证线性头。冻结后的参数、BN运行统计、Dropout和SSL内部mode必须固定；注意原SSL代码中设备/精度切换可能改变内部train/eval状态，补丁需兼容训练阶段与冻结阶段，而不是永远强制eval。[R1]

---

<a id="s4"></a>
## 4. 唯一数学合同：`ep_tta`

### 4.1 源域候选响应子空间

同一检测单元处理前后差分：

\[
\Delta z_{i,j}=E_0(T_jx_i)-E_0(x_i).
\]

主估计为类×处理家族平衡的**非中心化二阶矩（uncentered second moment）**：

\[
C_N=\frac{1}{2J}\sum_{y=0}^{1}\sum_{j=1}^{J}
\frac{1}{|S_{y,j}|}\sum_{i\in S_{y,j}}\Delta z_{i,j}\Delta z_{i,j}^{\top}.
\]

主版先按来源组固定等量单元采样，再对采样后的单元按上式计算；同一组的多个派生版本不能无意获得更大权重。另用组均值策略时显式记录估计版本，不将两种估计混用。cell 缺失报错，不自动变成另一估计。FP64 累积、对称化后 `eigh` 取最大 r 个特征向量；U 满足 UᵀU=I。禁止默认对差分去均值、单位化或白化。

响应能量近零或有效秩<r 时拒绝该配置或明确登记降低 r；不随机补齐。报告能量覆盖、每处理残差和源组 bootstrap 稳定性。比较投影矩阵/主夹角，不逐列比较有符号特征向量。

### 4.2 适应器与行向量约定

\[
A_R(z)=z+URU^\top z,\qquad R_0=0,\quad \|R\|_F\leq\rho<1.
\]

数学 z 是列向量；程序 `Z:[N,d]` 是行向量：

```python
Z_adapted = Z + ((Z @ U) @ R.T) @ U.T
```

R 只有 r² 个参数，默认 r=8 但不硬编码。不加入偏置，不后接未声明的特征归一化，不读取上一条 R。双侧形式不读取正交补来生成修正，不能换回旧单侧 `z+UBz`。

### 4.3 当前样本视图一致性

N 表示含原视图的总视图数，默认 N=3，V=N−1。不再在多个配置位置分别保存 N/V，防止不一致。

\[
\mu_R=\frac1N\sum_v A_R(z_v),\qquad
L_{view}=\frac1{Nd}\sum_v\|A_R(z_v)-\mu_R\|_2^2.
\]

这是总体均方差，分母 Nd，不是 N、Nr 或 N−1。均值来自当前适应后视图，不截断其梯度。视图固定，K 步不重新采样或重新编码。

### 4.4 源分类间隔保持

源锚点 \(\mathcal M=\{z_i,y_i,m_i^0,s_i^0\}_{i=1}^{M}\) 来自 fit；两类平衡，原模型在 tau0 下正确，\(m_i^0>\epsilon\)。保留难易间隔分层，不只挑最容易样本。

\[
s_i^0=h_0(z_i),\qquad m_i^0=(2y_i-1)(s_i^0-\tau_0),
\]
\[
m_i(R)=(2y_i-1)(h_0(A_R(z_i))-\tau_0),
\quad L_{margin}=\frac1M\sum_i
\left[\max\{0,(1-\gamma)m_i^0-m_i(R)\}\right]^2.
\]

\(0\leq\gamma<1\)，m0 和 tau0 固定。不得替换成归一化间隔、交叉熵或双向 MSE 后仍使用主方法名。锚点不足不可重复抽同一条凑数；显式改变 M 或阻塞该配置。

### 4.5 更新、投影与主输出

\[
L=L_{view}+\lambda_{keep}L_{margin},\quad
\widetilde R=R-\eta\nabla_RL,
\]
\[
R\leftarrow\widetilde R\min\{1,\rho/\|\widetilde R\|_F\},\qquad
s_{EP}=h_0(A_{R_K}(z_0)).
\]

使用无动量 SGD，`weight_decay=0`；零范数直接保持零。投影是矩阵整体 Frobenius 范数，不是梯度裁剪、逐行裁剪或谱范数约束。固定 K；K=0 作为测试合法但不作为有适应的主结果。

主法最后只用原视图；多视图 logit 融合另注册。最后一步后重新计算最终保持项/违反比例，不能把末步更新前的值当成最终约束状态。

### 4.6 必须验证的性质和能力边界

R=0 与 frozen 完全一致；保持项初值及梯度为零，所以 **K=1 时改变 lambda_keep 不改变结果**。K>1 也可能从未激活保持项；必须记录激活率，不能把此时收益归因于保持。

令 \(Q_c=ZU-\operatorname{mean}(ZU)\)，则

\[
\nabla_R L_{view}(0)=\frac{2}{Nd}Q_c^\top Q_c.
\]

这说明第一步倾向于收缩不稳定坐标，并非恢复干净录音。N=3 时视图中心化协方差秩至多2，不能声称识别了任意8维信道。

线性头下：

\[
(I-UU^\top)A_R(z)=(I-UU^\top)z,
\]
\[
|h_0(A_R(z))-h_0(z)|\leq\rho\|U^\top w\|_2\|U^\top z\|_2.
\]

若 Uᵀw=0，任何允许更新都不改变分数；不能同时声称严格零空间保持和分类提升。固定视图/锚点、线性头时辅助目标是 R 上凸函数；有限 K 步不保证最优，辅助目标最优也不保证鉴伪风险降低。有限源锚点保持不等于未知伪造安全保证。

### 4.7 等价的低维批处理

对 B 条独立样本设 `R:[B,r,r]`。定义 `Q=ZU:[B,N,r]`、`Qa=ZaU:[M,r]`、`c=Uᵀw:[r]`。每条可写：

\[
L_{view}=\frac{\|Z_c-Q_cU^\top\|_F^2+\|Q_c+Q_cR^\top\|_F^2}{Nd},
\]
\[
\Delta s_i=Q_{a,i}R^\top c,\quad m_i(R)=m_i^0+(2y_i-1)\Delta s_i,
\quad s_{EP}=s_0+Q_0R^\top c.
\]

这样无需实例化 `[B,M,d]` 的锚点张量。原维分母 Nd 与正交补常数必须保留；若优化时去掉常数，日志要加回并核验等价。

每条损失为 `loss_b:[B]`，反传 **`loss_b.sum()`**。不能用 mean 将每条梯度缩小 B 倍；不能给 B 条共用一个 R；每条单独投影最后两维。纯 SGD 下 sum 对应串行更新，换优化器不得跳过等价测试。

批次非有限可使用同一固定输入、原参数串行重放隔离故障；不能让一个坏样本使整批都被丢弃或统一回退。重放不更换 seed、步数、阈值或学习率，开销单独记录。

---
<a id="s5"></a>
## 5. 必须实现的机制基线：定义、预算与主张对应

### 5.1 方法注册，不允许“同名不同实现”

每种方法注册 `method_id / comparison_track / route / parameterization / objective / regularizer / subspace / solver / source_resources / reset_policy / final_output`。不支持的组合直接报错。注册 ID 和解析后的全部配置写入分数元数据；方法名称不根据最终效果改变。

`comparison_track` 仅允许：`reference`、`mechanism`、`published_port`、`streaming_extension`。原论文公开指标若引用，必须单列 `reported_in_original_paper`，不能混入本项目实测。

### 5.2 P0：最小闭环，先做这六项

| `method_id` | 定义 | 必须回答的问题 |
|---|---|---|
| `frozen` | 原视图 `h0(z0)` | 原模型基准 |
| `multiview_mean` | 同一 N 个视图的 logit 分数均值 | 是否只是多次观察？ |
| `static_subspace` | 所有样本共享 `R=-aI`，a 源选择后固定 | 是否静态抑制即可？ |
| `ep_no_keep` | 主方法去掉间隔保持 | 保持机制是否参与？ |
| `ep_random_U` | 同 r 的预登记随机正交 U，其余同主法 | 是否只需低维限制？ |
| `ep_tta` | §4 原式 | 完整候选方案 |

`static_subspace` 的公平半径是 `0≤a≤rho/sqrt(r)`，因为 `||-aI||_F=a*sqrt(r)`。不能用 a=rho 冒称同预算，也不能只用 a=1 的完全删除作弱基线。首轮固定候选比例 `[0,0.25,0.5,1] * rho/sqrt(r)`，由 select 选择。

### 5.3 P1：辅助目标×保持的六格实验

令 `s_v=h0(A_R(z_v))`、`p_v=sigmoid(s_v)`。熵使用自然对数，二分类最大熵为 ln2；不沿用千类图像的原始阈值。[R2–R4]

\[
L_{EM}=\frac1N\sum_vH(p_v),\qquad
L_{MEMO}=H\left(\frac1N\sum_v p_v\right).
\]

**概率先平均再求熵不等于各视图熵的平均，也不等于平均 logit 后求熵。** 机制版 MEMO 借鉴边际熵目标，不等于原方法的全部参数更新。[R4]

| 辅助目标 | 无保持 | 同一源间隔保持 |
|---|---|---|
| 特征均方差 | `ep_no_keep` | `ep_tta` |
| 各视图熵的均值 | `entropy_same_adapter_no_keep` | `entropy_same_adapter` |
| 平均概率的熵 | `memo_same_adapter_no_keep` | `memo_same_adapter_keep` |

同一 U、M、视图、参数维度和投影；各目标给予同等源选择预算。`entropy_same_adapter` 不叫 Tent，MEMO 两格不叫完整 MEMO。若某基线不使用 M，记录没有使用，不能为了所谓统一接口私下加入源约束。

数值实现优先 log-sigmoid 与 logsumexp；不要先把概率 clip 到粗糙常数造成饱和区梯度被无意清零。附录 C 给出稳定目标参考。

### 5.4 P1：保持方式替换而非累加

| `method_id` | 替换项 | 具体计算 |
|---|---|---|
| `ep_keep_l2` | 参数 L2 | `mean(R**2)` |
| `ep_keep_logit` | 源 logit 保持 | `mean((s_anchor(R)-s_anchor(0))**2)` |
| `ep_keep_fisher` | 源经验对角 Fisher | `mean(F * R**2)` |
| `source_ce_only` | 仅源交叉熵，无目标适应信号 | 从 R=0 在相同 M 上做固定步更新，原视图评分 |

前三项都保留 L_view 与投影，但**替换** margin；不能同时加 margin 后仍称替换。各自权重在 select 选择，不因量纲不同而强行共用一个数值。

Fisher 使用同一允许的源锚点与真实源标签，在 R=0 估计：

\[
F_{ab}=\frac1M\sum_i\left[\frac{\partial\ell_i(R)}{\partial R_{ab}}\Big|_{R=0}\right]^2,
\quad \ell_i=\operatorname{BCEWithLogits}(s_i(R),y_i).
\]

先逐样本平方再平均，不是平均梯度平方。线性头时令 `c=U.T@w`、`q_i=z_i@U`，则

\[
\nabla_R\ell_i(0)=(\sigma(s_i^0)-y_i)cq_i^\top.
\]

这用于快速实现与自动微分交叉测试。F 不归一化、不无声加大常数；接近零需报告。**不能对初值为零的 margin loss 求梯度来计算 F**，否则会错误得到全零。

这是 EP 参数化下的 Fisher 对照，不是 EATA 全部算法。其对坐标选择和标签使用的依赖也应说明，不能当作坐标旋转不变的通用保持。[R5]

只有 margin 保持、无视图项、R 从零开始应恒等于 frozen；这是实现测试，不伪装成一个有训练收益的基线。`source_ce_only` 不同，它可能从第一步就更新，可排除追加源训练的影响。

### 5.5 P1：子空间与最小复杂度

**`ep_feature_pca_U`**：同源内容池的未额外处理嵌入，按相同类/来源权重估计中心化协方差并取 r 维。中心化只用于估计该对照 U；分类器输入仍为原 z，不能新增中心化层。响应 U 则保持非中心化差分二阶矩，两者 artifact 键不同。

**`fixed_source_adapter`**：用与响应 U 对应的 fit 视图池和同一 M，离线学一个共享 R，目标阶段不更新。可充分训练至预定步数/源选择停止，不能为了弱化基线只给3步。记录总源训练预算，匹配源信息权限而不伪称训练开销完全相同。

**`ep_scalar_adaptive`**：每条只允许 `R=-a_x I`，在 `[0,rho/sqrt(r)]` 的17个固定均匀网格点计算相同 L_view+lambda*L_margin。取最小损失，严格相等时取较小 a；主版只处理严格浮点等值，不新增近似平局容差。它是本项目的简单竞争基线，不是外部论文，也不是完全无测试优化。记录17次前向目标评价的成本。

**`ep_no_projection`**：移除投影，其他不变；不再具有 rho 界，损失数值失败按原规则记录。**`ep_diagonal_R`** 为 P3 可选：r 个对角自由度，不当作同参数量比较。

随机 U 由至少3个预登记构造种子复核，不能挑最差种子；PCA 与响应估计使用同源池。r/rho 改变时静态和标量半径随之重算。

### 5.6 工作点校正基线

`frozen_source_shift`：只在 select 上依据 §11 的相同选择准则选一个全局偏移 b，之后所有目标 `s=s0+b`。候选偏移可由源分数尺度生成并封存，不从目标统计估计。

严格递增的全局分数变换不改变排序型指标，固定阈值结果可变。若另做各方法 cal1 校准，阈值也须一致映射；不能把这类变化写成新的可辨识信息。数值 ties 单独处理，不因 epsilon 随意扰动目标分数。

### 5.7 对照预算分层

“同一视图”不等于“完全同算力”。所有比较披露 `source_access / target_history / update_scope / num_views / forwards / backwards / objective_evaluations / wall_time / memory`。

机制表优先匹配信息、结构与选择预算；完整算法表保留其算法定义并披露差异；成本效率另列。禁止为了严格凑同一步数阉割 SAR 的第二次计算、MEMO 的联合目标或 EATA/T²A 的关键选择组件。

---

<a id="s6"></a>
## 6. P2：已有算法的音频移植与资源合同

### 6.1 必须预留并逐一审计的五项

| ID | 原方法核心 | 本项目身份与关键检查 |
|---|---|---|
| `tent_audio_ep` | 测试预测熵最小化 | 归一化参数/统计、音频视图 batch、逐条重置；不等于 R-only EM [R2] |
| `sar_audio_ep` | 稳定/尖锐度相关的适应与筛选 | 保留两阶段计算与恢复行为；记录无样本入选情况 [R3] |
| `memo_audio_ep_full` | 单样本多视图边际熵、模型参数适应 | 审计全参数更新；原视图最终重算；不能用最终特征缓存 [R4] |
| `eata_audio_ep` | 可靠/非冗余样本选择＋重要参数保持 | Fisher 的计算范围、源信息与选择历史；全部 episode 状态重置 [R5] |
| `t2a_audio_ep` | 鉴伪负学习、样本优先、梯度掩码 | 作者算法部件、二分类处理、逐条设定相对原在线协议的差异 [R6] |

每个移植在 `docs/baselines/<id>.md` 保存：原论文、作者代码、实际 Git commit、许可、最小代码补丁、更新参数名/数目、BN/LN 策略、视图、预测时机、source/Fisher 来源、重置、目标历史、原协议差异和实测状态。参考仓库分支 URL 不是 commit 锁。

已经给出源码接口不等于复现完成。缺少可验证作者算法/许可证时标阻塞，并说明不足；不自行猜公式后贴原方法名。首轮只实现接口与审计骨架，远程再做真实 parity/集成。

### 6.2 原方法移植和机制对照的边界

主比较采用当前样本预测前适应、逐检测单元重置。若原方法的多个独立目标样本被替换为同一音频的多视图，明确写 `episodic audio port`。它检验这一受限场景，不表示推翻原论文原设置。

T²A/EATA 等在线版若另保留历史，应注册 `*_audio_stream`，放 `streaming_extension`，不得与 episodic 主表混排；必须按封存流顺序执行，不能把同一流随意分到两卡后丢失历史。[R5–R6]

EATA 原 Fisher 估计与本文真实源标签 Fisher 未必相同；记录真实访问条件。需要匹配源信息时增设显式变体，而不偷偷把原方法变成有标签源锚点更新。

### 6.3 梯度与冻结切分

源 detector 的前后端分界必须由结构审计决定。更新 SSL 前端任何层时，输入需走可反向的波形路径。只更新图后端时可复用前端冻结表示，但它通常比最终语句嵌入大，另做容量规划，默认按 batch 临时计算/丢弃，不预存全部 SSL 帧。

验收时检查所有被声明更新参数：`requires_grad=True`、能收到梯度、实际优化器引用正确；没有梯度不能计为“完成适应”。已发表方法也不得利用目标标签决定更新、筛选或返回哪一步。

### 6.4 两卡上的可行策略，不能用降级冒充复现

完整模型更新通常明显重于缓存 R-only。默认每卡一个独立基线作业，episode batch=1，先只在 select 上测显存与延迟。两卡不自动形成一块96GB可用显存；不默认模型并行。

若 MEMO 全参数版本无法在实测预算内运行：优先采用不改目标的视图 microbatch、验证过的梯度检查点/重算、受控参数快照；仍无法满足时标 `BLOCKED_RESOURCE`，另登记 `memo_audio_ep_backend`，明确只更新后端，不能把后者表述为完整 MEMO。

边际熵跨视图梯度不能把每个视图独立求熵再平均。分块实现需先得到联合平均概率与系数，再重放完全相同视图并累积正确梯度；随机层 RNG、BN buffer 以及模型参数必须与对应整批计算一致。未通过整批—分块梯度测试前不启用这一优化。

归一化统计依赖整批时，简单拆分 batch 不一定等价。需要固定、可说明的统计策略或停在支持的 batch；不能在 OOM 后自动改变统计含义。每 episode 恢复参数、优化器、筛选历史和 buffer，不能留下前一目标的信息。

最终报告所有被列为必需但未完成的基线，不通过删除行掩盖。P1 已完成但 P2 未完成时，只能作机制先导结论，不宣称完整优于成熟 TTA。

---

<a id="s7"></a>
## 7. 仓库结构、接口与依赖边界

### 7.1 目标结构

项目尚未开始。首次建立以下逻辑结构；若工作目录已有用户其他文件，先盘点再新增，不破坏性覆盖。所有条目都是待实现交付：

```text
ep-tta/
├── AGENTS.md
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_STATUS.md
├── docs/
│   ├── DESIGN.md                  # 本文件，唯一主规范
│   ├── DECISIONS.md               # 设计决策，不重复拷贝规范
│   ├── TEST_REPORT.md
│   ├── REMOTE_RUNBOOK.md
│   ├── baselines/                 # 各published port审计
│   ├── models/                    # 原论文/作者架构与训练配方审计
│   └── DATA_CONTRACT_REVIEW.md     # 远程未知项、审核与补全记录
├── configs/
│   ├── base.yaml
│   ├── profiles/{local_dev,remote_a6000}.yaml
│   ├── paths.remote.yaml.example
│   ├── models/registry.yaml
│   ├── datasets/registry.yaml
│   ├── data_contracts/*.yaml.example
│   ├── preprocess/*.yaml.example
│   ├── splits/{source,incremental}.yaml.example
│   ├── training/{aasist,ssl_aasist}.yaml
│   ├── source_training_plan.yaml
│   ├── methods/registry.yaml
│   ├── suites/{smoke,mechanism,published,confirmatory}.yaml
│   └── experiments/{source_pilot,controlled,external}.yaml
├── src/eptta/
│   ├── cli.py
│   ├── config/{schema,resolve,validate}.py
│   ├── data/{records,roles,groups,manifests,permissions,contracts,snapshots,incremental}.py
│   ├── data/adapters/{base,delimited,json_records,sidecar,labels,provenance}.py
│   ├── data/parsers/{asvspoof2019,asvspoof2021,wavefake,codecfake_xie,in_the_wild}.py
│   ├── audio/{decode,unit,probes,quality,seeding}.py
│   ├── models/{registry,factory,base,ssl_aasist,aasist,linear_head,toy}.py
│   ├── training/{contracts,recipe,dispatch,artifacts,selection}.py
│   ├── offline/{subspace,calibration,anchors,fisher,static_adapter}.py
│   ├── adaptation/{types,adapter,objectives,regularizers,projection,episode,batch}.py
│   ├── baselines/{registry,static,scalar,dispatch}.py
│   ├── baselines/ports/{tent,sar,memo,eata,t2a}.py
│   ├── cache/{schema,keys,writer,reader,merge,recovery}.py
│   ├── execution/{plan,preflight,scheduler,locks,extract,score,seal}.py
│   ├── evaluation/{join,metrics,flips,bootstrap,audit,report}.py
│   └── utils/{hashing,atomic,logging,versions}.py
├── workers/
│   ├── source_train_bridge.py     # 自行训练/验证/恢复/选模，不导入目标评价器
│   ├── baseline_bridge.py         # 自有checkpoint冻结推理入口，不导入eptta核心
│   ├── published_bridge.py        # 所属基线环境内完成前向/反向/重置
│   └── compat/                    # 兼容当前选择环境的架构/训练循环/解码/写出
├── tests/{unit,contracts,integration,remote}/
├── tests/fixtures/                # 小型模拟协议/张量，不含真实资源
├── deployment/{sync.exclude,remote.env.example,launch_extract.sh,launch_source_train.sh}
└── third_party/                  # 仅授权代码/patch与固定来源，不放权重
```

真实 `snapshots / manifests / derived_audio / training_runs / trained_models / artifacts / feature_cache / runs / reports / logs` 放远程 work root。源码目录不成为默认数据盘。

### 7.2 最小接口

下列为接口合同，Coding 工具需实现校验和工程字段；省略号不代表功能已完成。

```python
from dataclasses import dataclass
from typing import Protocol
from torch import Tensor

@dataclass(frozen=True)
class TargetViews:
    sample_id: str
    features: Tensor                 # [N,d]，无目标标签
    feature_artifact_id: str

@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    comparison_track: str
    route: str
    parameterization: str
    objective: str
    regularizer: str
    subspace: str
    reset_policy: str
    final_output: str

@dataclass(frozen=True)
class FrozenResources:
    U: Tensor
    w: Tensor
    b: Tensor
    anchors_z: Tensor
    anchors_y: Tensor                 # 明确允许的源标签
    anchors_m0: Tensor
    anchors_s0: Tensor
    tau0: float
    artifact_bundle_id: str

class FrozenDetector(Protocol):
    def encode(self, waveform: Tensor) -> Tensor: ...
    def score(self, embedding: Tensor) -> Tensor: ...
    def metadata(self) -> dict: ...

class EpisodeRunner(Protocol):
    def run(self, target: TargetViews, resources: FrozenResources,
            method: MethodSpec, params: dict) -> dict: ...

class PublishedPort(Protocol):
    def snapshot_source_state(self) -> None: ...
    def reset_episode(self) -> None: ...
    def adapt_and_predict(self, waveform_views: Tensor) -> dict: ...
    def audit_state(self) -> dict: ...
```

适应模块不得导入目标评价器或协议解析器。`resources` 只读，无最近目标缓存。生产类型进一步用枚举/验证约束字符串值；禁止 `params` 隐含传入标签或全体目标索引。

### 7.3 冻结 wrapper 验收

比较同一自有checkpoint在原作者架构前向的logits、`head(encode(x))`、导出的 `w,b` 分数及 R=0 分数。先在同设备 FP32 上用源 fixture 验证，建议起始容差 atol=rtol=1e−5；改内核后误差须独立记录并冻结，不按目标效果放宽。

已核验的 SSL_Anti-spoofing 代码在设备/dtype 改变时可能调用内部 `train()`；先迁移到最终 device/dtype 再 eval，并检查每次 forward 前后所有 module mode 和 running buffers。最小补丁必须区分source training与frozen inference：训练时允许正确的train状态，冻结导出后强制保持eval；路径、设备和状态修补保留diff，不顺便改网络。[R1]

提取使用 `no_grad`；对 R 的前向、冻结 head 输入和 loss 必须启用 autograd。`eval()` 不等于关闭梯度，参数冻结也不阻止对输入求梯度；不要给整个适应器加 no_grad 或在全局 inference mode 中创建后续反向所需状态。[R12]

### 7.4 跨环境工作进程合同

作业按用途分三种，不复用一个拥有全部标签权限的万能JSON：

- `SourceTrainJob`：只能引用批准的fit/source_val清单、架构/配方、可选通用SSL初始化、输出和恢复状态；允许这些源标签，拒绝其他角色。
- `InferenceJob`：job/schema版本、物理绑定、自有frozen bundle、无标签推理清单、decode/probe、分片、数值模式、输出及允许源资源。目标标签/攻击ID/干净路径不进入。
- `EvaluationJob`：在相应分数封存之后引用label sidecar；不持有适应模型，不反馈当前run超参数。

所有作业格式初始版本`0.1.0`。CLI在核心环境编译计划，worker按自己的job_type做二次权限检查。

核心CLI始终在core环境运行；`train-source / resume-source / finalize-training`委托source_train_bridge；`inspect-model / preflight / extract`委托baseline_bridge；published的run-suite委托published_bridge。bridge若与core使用同一环境也遵守该作业合同；只有经过导入与parity核验才允许直接进程内wrapper快捷通路。

兼容层的decode/probe必须与规范一致，不能为了旧环境方便改用另一音频库或随机数实现；两环境实现若重复，应有同一fixture与系数/seed测试。桥接协议写出错误码、状态、哈希和环境版本，不默默回退toy模型。没有必要的旧环境时本机完成接口并标DEFERRED_REMOTE。

### 7.5 配置实现原则

基础依赖优先 Python 标准库、NumPy、PyYAML；数值核与测试使用 PyTorch/pytest；音频与旧模型依赖延迟导入。具体 lock 在对应环境实测后生成，不在本规范中宣称一组新版本可兼容旧 fairseq。

使用**单向配置解析**，不实现任意 `inherit` 或动态 Python 表达式：

`base → selected experiment → selected profile → private resource paths → registered method → explicit allowed params`。

源训练采用独立解析链：`model registry → source training plan → selected training recipe → approved data/preprocess/split locks → runtime profile → private paths`；不得把EP的R-only或regularizer默认值注入训练器。最终training recipe lock记录各字段来源。

profile 只能改运行/权限项，paths 只能改位置，method 只能指定注册算法与其允许参数。未知键、重复键、非法覆盖直接失败。YAML Loader 必须检测重复 key，而不是依赖默认“后者覆盖前者”。枚举值和每次覆盖的来源写入 `resolved_config.json`。

本机shape-only验证允许未填真实路径和未批准合同；远程实运行按stage仅检查**选中**资源的必需字段。data inspection不要求已训练checkpoint；source training不要求U/M；EP评分必须有finalized自有frozen bundle。输出 help/plan 不导入大模型、不联网。

<a id="s7-training"></a>
### 7.6 数据与训练插件接口：先实现协议，不猜真实文件

以下类型是核心侧合同，worker侧采用同结构JSON且按其Python版本实现。`...`表示待实现接口，不是成功返回或伪造默认数据。

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

@dataclass(frozen=True)
class RawRecord:
    record_ref: str
    fields: Mapping[str, Any]
    source_file_sha256: str

@dataclass(frozen=True)
class ContractIssue:
    code: str
    field: str
    record_ref: str | None
    message: str
    blocking: bool

class DatasetAdapter(Protocol):
    def inspect(self, roots: Mapping[str, Path],
                hints: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def propose_contract(self, inventory: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def iter_raw(self, approved_contract: Mapping[str, Any]) -> Iterable[RawRecord]: ...
    def normalize(self, record: RawRecord, label_policy: Mapping[str, Any],
                  group_policy: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None,
                                                          list[ContractIssue]]: ...

class SplitPlanner(Protocol):
    def propose(self, snapshot_ref: str, policy: Mapping[str, Any],
                prior_assignments_ref: str | None) -> Mapping[str, Any]: ...
    def validate(self, proposal_ref: str) -> list[ContractIssue]: ...
    def commit(self, proposal_ref: str, approval_ref: str) -> str: ...

class IncrementalReconciler(Protocol):
    def diff(self, parent_snapshot_ref: str,
             candidate_records_ref: str) -> Mapping[str, Any]: ...
    def reconcile(self, diff_ref: str, grouping_policy_ref: str,
                  role_policy_ref: str) -> Mapping[str, Any]: ...
    def commit_snapshot(self, reviewed_delta_ref: str, approval_ref: str) -> str: ...

@dataclass(frozen=True)
class SourceTrainSpec:
    schema_version: str
    model_id: str
    recipe_lock_ref: str
    fit_manifest_ref: str
    source_val_manifest_ref: str
    init_artifact_ref: str | None
    output_dir: str
    training_seed: int
    resume_artifact_ref: str | None = None

class SourceTrainer(Protocol):
    def validate_job(self, spec: SourceTrainSpec) -> list[ContractIssue]: ...
    def train_smoke(self, spec: SourceTrainSpec) -> Mapping[str, Any]: ...
    def train(self, spec: SourceTrainSpec) -> Mapping[str, Any]: ...
    def resume(self, spec: SourceTrainSpec) -> Mapping[str, Any]: ...
    def finalize(self, run_ref: str) -> Mapping[str, Any]: ...
    def export_frozen(self, finalized_training_ref: str) -> str: ...
```

`normalize`失败返回问题或抛可识别异常，不返回填了随机label的“正常样本”。source training plan中的role allowlist固定为`fit/source_val`；数据盘点和标签审核可以看原label，TTA作业不能，因此不应把整个数据服务对象注入`EpisodeRunner`。

### 7.7 模型训练插件和产物交接

`ModelFactory`负责用已锁定作者架构构建网络、只加载允许的通用初始化、提供native类别映射；`SourceTrainer`实现train/validation/checkpoint；`FrozenDetector`接收finalized产物做冻结前向。三者不共享隐式全局“当前模型”变量。

核心侧`training/dispatch.py`编译`SourceTrainJob`，调用`workers/source_train_bridge.py`。训练worker持有GPU计算图与优化器；核心只读取状态、日志与finalized产物。AASIST和SSL-AASIST可以注册不同环境入口，不要求共用同一旧fairseq依赖。

`TrainingResult`只在训练和选模完成后写`status=FINALIZED`；此前可以是`RUNNING / INTERRUPTED / FAILED`。`export-frozen`拒绝仅完成smoke、缺训练manifest、缺source_val选择依据或初始化为外部鉴伪权重的产物。reference tensor测试用`synthetic_test`通道绕过真实训练时必须标识，禁止导入正式模型registry。


---

<a id="s8"></a>
## 8. 配置模板：唯一字段、路径占位与实验注册

### 8.1 `configs/base.yaml`

下例是需要实现的配置结构。r 只定义在 `defaults.rank`；N 只定义在 `probe.num_views`，其余由解析器派生。所有候选值都是启动值，不是实测最优。

```yaml
schema_version: "0.1.0"
design_version: "0.1.0"
project_version: "0.1.0"
method_contract_version: "0.1.0"
project: ep_tta
project_status: NOT_STARTED
phase: development
protocol:
  canonical_labels: {bonafide: 0, spoof: 1}
  score_direction: larger_is_spoof
  decision_rule: strict_greater
  episode_unit: baseline_detection_unit
  adaptation_timing: pre_prediction
  target_labels_visible: false
  target_history_visible: false
  target_population_statistics_visible: false
roles:
  fit: fit
  source_validation: source_val
  selection: select
  threshold: cal0
  audit: audit
  final_control: control_test
  final_external: target_test
  optional_method_calibration: cal1
model:
  selected_id: ssl_aasist_source
  require_linear_final_head: true
  allow_random_weight_fallback: false
  source_checkpoint_policy: train_in_project
  allow_external_task_checkpoint: false
  require_finalized_training: true
probe:
  profile_id: probe_default
  num_views: 3
  view0: identity
  transformed_views:
    - {family: additive_noise, snr_db: [25.0, 35.0]}
    - {family: mild_fir_eq, max_gain_db: 2.0}
  apply_after_baseline_unit: true
  regenerate_per_step: false
  shared_across_methods: true
  seed: 13
subspace:
  estimator: balanced_uncentered_second_moment
  accumulator_dtype: float64
  source_role: fit
  centered: false
  normalize_delta: false
  weighting: equal_class_family_group_balanced_sampling
  pair_seed: 13
  pilot_max_source_groups_per_class: 512
memory:
  source_role: fit
  size: 256
  balanced_classes: true
  positive_margin_only: true
  margin_epsilon: 1.0e-6
  margin_bins: 4
  seed: 13
calibration:
  mode: empirical_real_quantile
  alpha_cal: 0.05
  primary_threshold: tau0
  enable_cal1: false
defaults:
  rank: 8
  steps: 3
  lr: 0.01
  rho: 0.2
  gamma: 0.1
  regularizer_weight: 1.0
  momentum: 0.0
  weight_decay: 0.0
  static_fraction: 0.5
  scalar_grid_size: 17
  score_shift: 0.0
selection:
  max_candidates_per_method: 12
  role: select
  delta_fpr_allowance: 0.01
  delta_tpr_allowance: 0.01
  rule: source_constrained_macro_balanced_accuracy
  target_metrics_for_selection: false
cache:
  format: sharded_npy
  feature_dtype: float32
  layout_shards: 32
  block_units: 8192
  store_waveforms: false
  store_ssl_frame_features: false
  verify_completed_blocks: true
  atomic_commit: true
evaluation:
  score_comparator: strict_greater
  metrics: [fpr, tpr, fnr, auroc, eer, harmful_flips, helpful_flips]
  bootstrap_repeats: 1000
  confidence_level: 0.95
  group_bootstrap: true
  read_labels_after_score_seal: true
  incomplete_run_policy: mark_incomplete
logging:
  overwrite: false
  pilot_trace_fraction: 1.0
  confirmatory_trace_fraction: 0.01
  diagnostic_selection_seed: 13
  audit_target_fraction: 0.1
  keep_full_R: diagnostic_ids_only
safety:
  automatic_download: false
  unresolved_contract_execution: false
  automatic_ssh: false
  destructive_sync: false
  trust_unverified_pickle: false
  silent_algorithm_fallback: false
```

源选择容忍度1个百分点是先导设定，必须在看最终测试前决定是否合适；不是统计保证。其他保持权重使用同一 `regularizer_weight` 但独立源选择，不共享“同数值即公平”的假设。

### 8.2 profiles

**`configs/profiles/local_dev.yaml`**：

```yaml
profile_id: local_dev
environment: local
permissions:
  real_data_io: false
  model_weight_io: false
  network_download: false
  remote_submission: false
  real_source_training: false
compute:
  device: cpu
  physical_gpu_ids: []
  extraction_workers: 0
  episode_batch_size: 1
  encoder_batch_units: 1
  num_workers_per_process: 0
  prefetch_factor: null
  pin_memory: false
  persistent_workers: false
  precision: float32
  tf32: false
validation:
  require_real_paths: false
  allow_synthetic_fixtures: true
```

**`configs/profiles/remote_a6000.yaml`**：

```yaml
profile_id: remote_a6000
environment: remote
permissions:
  real_data_io: true
  model_weight_io: true
  network_download: false
  remote_submission: false
  real_source_training: true
compute:
  device: cpu
  physical_gpu_ids: [0, 1]
  extraction_workers: 2
  processes_per_gpu: 1
  synchronize_episode_gradients: false
  encoder_batch_units: 1
  encoder_batch_candidates: [1, 2, 4, 8]
  num_workers_per_process: 4
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true
  multiprocessing_context: spawn
  episode_batch_size: 128
  episode_batch_candidates: [1, 32, 128, 256]
  encoder_memory_cap_gib: 40.0
  free_memory_fraction: 0.85
  precision: float32
  adaptation_precision: float32
  tf32: false
  amp: false
  compile_model: false
  oom_policy: split_batch_without_changing_semantics
validation:
  require_real_paths: true
  allow_synthetic_fixtures: false
  require_preflight: true
```

profile 中 `device=cpu` 是缓存适应默认设备；编码 worker 按物理卡独立绑定，并在各进程内使用 `cuda:0`。不得把 `device=cpu` 错解释为禁止远程编码 GPU。实际 profile 覆盖到 `runtime_profile.lock.json` 后才扩大 batch；worker=0 时清除 prefetch/persistent_workers。

### 8.3 远程路径模板：初始化输入与自有训练输出分开

**`configs/paths.remote.yaml.example`**：路径尚未知，字段只预留，不访问占位地址。

```yaml
path_schema_version: "0.1.0"
roots:
  work: ${EPTTA_WORK_ROOT}
  pretrained: ${EPTTA_PRETRAINED_ROOT}
source_repos:
  aasist: ${AASIST_SOURCE_REPO}
  ssl_aasist: ${SSL_AASIST_SOURCE_REPO}
dataset_roots:
  asvspoof2019_la_train: ${ASV2019_LA_TRAIN}
  asvspoof2019_la_dev: ${ASV2019_LA_DEV}
  asvspoof2019_la_eval: ${ASV2019_LA_EVAL}
  asvspoof2021_df: ${ASV2021_DF_ROOT}
  asvspoof2021_la: ${ASV2021_LA_ROOT}
  wavefake_generated: ${WAVEFAKE_ROOT}
  ljspeech_real: ${LJSPEECH_ROOT}
  jsut_real: ${JSUT_ROOT}
  codecfake_xie: ${CODECFAKE_XIE_ROOT}
  in_the_wild: ${IN_THE_WILD_ROOT}
protocol_files:
  asv2019_train: ${ASV2019_TRAIN_PROTOCOL}
  asv2019_dev: ${ASV2019_DEV_PROTOCOL}
  asv2019_eval: ${ASV2019_EVAL_PROTOCOL}
  asv2021_df: ${ASV2021_DF_KEYS}
  asv2021_la: ${ASV2021_LA_KEYS}
  asv_source_mapping: ${ASV_SOURCE_MAPPING}
  wavefake: ${WAVEFAKE_METADATA}
  codecfake_xie: ${CODECFAKE_XIE_PROTOCOL}
  in_the_wild: ${IN_THE_WILD_METADATA}
initialization_files:
  xlsr_300m: ${XLSR_300M_PRETRAINED}
artifact_bindings:
  frozen_model_bundle: null
  parent_dataset_snapshot: null
  selected_source_checkpoint: null
worker_python:
  aasist_train: ${EPTTA_AASIST_PY}
  ssl_aasist_train: ${EPTTA_SSL_PY}
  frozen_inference: ${EPTTA_BASELINE_PY}
  published_ports: ${EPTTA_PUBLISHED_PY}
```

`artifact_bindings`是运行后生成的**产物引用**，不是要求现在提供作者已训练checkpoint。由`finalize-training / export-frozen / bind-artifacts`填写私有resolved配置；后续运行可显式选择已完成的本项目training_run，而不是手工猜文件名。只要构建阶段不需要，该字段保持null合法。

实际输出统一在工作根下派生，不要求用户把已有数据挪目录：

```text
$EPTTA_WORK_ROOT/
  inventory/ contracts/ snapshots/ manifests/ derived_audio/
  training_runs/<model_id>/<run_id>/{checkpoints,logs,metrics}/
  trained_models/<baseline_id>/{model_state,head,model_manifest}/
  artifacts/<bundle_id>/ feature_cache/ runs/ reports/ logs/
```

只支持字面`${ENV_NAME}`展开，不执行shell表达式。未选择的数据和可选SSL初始化允许未设置；选用AASIST时不强制要求XLS-R。`local_dev`永远不解引用真实路径。

`deployment/remote.env.example`列上述变量及`EPTTA_CORE_PY`；都留空或标`/replace/...`。真实.env和私有paths不进Git。数据/协议/通用初始化只读；train/checkpoint输出可写，不复用同一个根作为输入和覆盖目标。

### 8.4 模型与数据 registry

**`configs/models/registry.yaml`**：这是模型构建与训练注册，不是现成权重目录。

```yaml
registry_version: "0.1.0"
models:
  aasist_source:
    architecture_plugin: aasist_author
    repo_url: https://github.com/clovaai/aasist
    paper_ref: R16
    repo_key: aasist
    repo_commit: null
    architecture_config_ref: null
    training_recipe_ref: configs/training/aasist.yaml
    initialization: {task_weights: native_initialization, pretrained_frontend: null}
    allow_external_task_checkpoint: false
    source_training_required: true
    checkpoint_artifact_ref: null
    frozen_bundle_ref: null
    wrapper: aasist
    head_type: linear_two_logits
    embedding_dim: null
    class_index_map: null
    eval_profile_ref: null
    implementation_status: TODO
    training_status: NOT_STARTED
    remote_integration_status: DEFERRED_REMOTE
  ssl_aasist_source:
    architecture_plugin: ssl_aasist_author
    repo_url: https://github.com/TakHemlata/SSL_Anti-spoofing
    paper_ref: R17
    repo_key: ssl_aasist
    repo_commit: null
    architecture_config_ref: null
    training_recipe_ref: configs/training/ssl_aasist.yaml
    initialization: {task_weights: native_initialization, pretrained_frontend: xlsr_300m}
    pretrained_scope: generic_ssl_frontend_only
    training_regime: joint_finetune
    allow_external_task_checkpoint: false
    source_training_required: true
    checkpoint_artifact_ref: null
    frozen_bundle_ref: null
    wrapper: ssl_aasist
    head_type: linear_two_logits
    embedding_dim: null
    class_index_map: null
    eval_profile_ref: null
    implementation_status: TODO
    training_status: NOT_STARTED
    remote_integration_status: DEFERRED_REMOTE
```

实际build与训练必须补repo commit、architecture/recipe/preprocess lock；**不必先补checkpoint**。EP阶段必须再补finalized自有训练产物。`inspect-model --mode architecture`允许检查未训练结构，`--mode frozen_bundle`才要求训练权重；前者不能给出真实性能。自有checkpoint若已包含SSL参数，应优先支持无需再读取初始化文件的导出构造；作者构造器确有要求时记录并核验，禁止静默联网下载。

**`configs/datasets/registry.yaml`**：

```yaml
registry_version: "0.1.0"
datasets:
  asvspoof2019_la:
    parser: asvspoof2019
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    track: LA
    roles: [fit, source_val, select, cal0, audit, control_test, target_test]
    label_policy_id: source_authenticity
    root_keys: [asvspoof2019_la_train, asvspoof2019_la_dev, asvspoof2019_la_eval]
  asvspoof2021_df:
    parser: asvspoof2021
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    track: DF
    roles: [target_test]
    label_policy_id: source_authenticity
    root_keys: [asvspoof2021_df]
  asvspoof2021_la:
    parser: asvspoof2021
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    track: LA
    roles: [target_test]
    enabled_by_default: false
    root_keys: [asvspoof2021_la]
  wavefake:
    parser: wavefake
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    roles: [target_test]
    root_keys: [wavefake_generated, ljspeech_real, jsut_real]
    require_matching_reference_corpus: true
  codecfake_xie:
    parser: codecfake_xie
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    roles: [target_test]
    root_keys: [codecfake_xie]
    label_policy_id: codecfake_xie_official
    alternate_policy_requires_provenance_review: true
  in_the_wild:
    parser: in_the_wild
    raw_contract_status: UNRESOLVED
    raw_contract_ref: null
    snapshot_ref: null
    preprocess_contract_ref: null
    roles: [target_test]
    root_keys: [in_the_wild]
    label_policy_id: source_authenticity
```

roles 表示可分配范围，不表示一个样本可以同时拥有多个角色；`build-manifests` 按封存 split plan 分配。真实run还需绑定已锁定raw/label/group/preprocess合同和snapshot。`null`保留的是服务器待补全接口，不是让Coding工具猜一个可运行路径。

### 8.5 方法 registry 与兼容规则

避免重复编写同一目标。registry 解析器根据显式字段组成模块，不做隐式继承。以下每行的简写字段含义固定：`param`=parameterization，`obj`=objective，`reg`=regularizer，`U`=subspace；解析后保存全名。

```yaml
registry_version: "0.1.0"
methods:
  frozen: {track: reference, route: feature_cache, param: none, obj: none, reg: none, U: none, output: original}
  multiview_mean: {track: reference, route: feature_cache, param: none, obj: none, reg: none, U: none, output: mean_logits}
  static_subspace: {track: reference, route: feature_cache, param: fixed_scalar, obj: none, reg: none, U: response, output: original}
  fixed_source_adapter: {track: reference, route: feature_cache, param: fixed_matrix, obj: none, reg: none, U: response, output: original}
  frozen_source_shift: {track: reference, route: feature_cache, param: fixed_score_shift, obj: none, reg: none, U: none, output: original}
  ep_tta: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: margin, U: response, output: original}
  ep_no_keep: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: none, U: response, output: original}
  ep_random_U: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: margin, U: random, output: original}
  ep_feature_pca_U: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: margin, U: source_pca, output: original}
  ep_no_projection: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: margin, U: response, projection: none, output: original}
  entropy_same_adapter_no_keep: {track: mechanism, route: feature_cache, param: matrix, obj: mean_entropy, reg: none, U: response, output: original}
  entropy_same_adapter: {track: mechanism, route: feature_cache, param: matrix, obj: mean_entropy, reg: margin, U: response, output: original}
  memo_same_adapter_no_keep: {track: mechanism, route: feature_cache, param: matrix, obj: marginal_entropy, reg: none, U: response, output: original}
  memo_same_adapter_keep: {track: mechanism, route: feature_cache, param: matrix, obj: marginal_entropy, reg: margin, U: response, output: original}
  ep_keep_l2: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: l2, U: response, output: original}
  ep_keep_logit: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: logit, U: response, output: original}
  ep_keep_fisher: {track: mechanism, route: feature_cache, param: matrix, obj: view_variance, reg: fisher, U: response, output: original}
  ep_scalar_adaptive: {track: mechanism, route: feature_cache, param: scalar_grid, obj: view_variance, reg: margin, U: response, output: original}
  ep_diagonal_R: {track: mechanism, route: feature_cache, param: diagonal, obj: view_variance, reg: margin, U: response, output: original}
  source_ce_only: {track: mechanism, route: feature_cache, param: matrix, obj: source_bce, reg: none, U: response, output: original}
  tent_audio_ep: {track: published_port, route: waveform_update, implementation: tent, requires_audit: true}
  sar_audio_ep: {track: published_port, route: waveform_update, implementation: sar, requires_audit: true}
  memo_audio_ep_full: {track: published_port, route: waveform_update, implementation: memo_full, requires_audit: true}
  eata_audio_ep: {track: published_port, route: waveform_update, implementation: eata, requires_audit: true}
  t2a_audio_ep: {track: published_port, route: waveform_update, implementation: t2a, requires_audit: true}
```

默认矩阵/对角/标量动态方法使用逐条重置和 Frobenius 球；只有 `ep_no_projection` 显式取消。固定方法无优化状态。published-port 的参数/目标/归一化不由 R defaults 自动补齐，必须根据审计专属配置解析；否则拒绝计划。参数改变不能重命名既有方法来绕过合同。

### 8.6 suites 与实验入口

**`configs/suites/smoke.yaml`**：

```yaml
suite_id: smoke
suite_kind: cached_pilot
methods: [frozen, multiview_mean, static_subspace, ep_no_keep, ep_random_U, ep_tta]
require_source_only: true
allow_unimplemented_methods: false
```

**`configs/suites/mechanism.yaml`**：

```yaml
suite_id: mechanism
methods:
  - frozen
  - multiview_mean
  - static_subspace
  - fixed_source_adapter
  - frozen_source_shift
  - ep_no_keep
  - ep_tta
  - ep_random_U
  - ep_feature_pca_U
  - entropy_same_adapter_no_keep
  - entropy_same_adapter
  - memo_same_adapter_no_keep
  - memo_same_adapter_keep
  - ep_keep_l2
  - ep_keep_logit
  - ep_keep_fisher
  - ep_scalar_adaptive
  - source_ce_only
  - ep_no_projection
require_source_only: true
allow_unimplemented_methods: false
```

`suite_kind`是执行任务分类，不是方法的`comparison_track`；两者使用不同枚举，不能互相填充。published suite 包含五个已发表移植及 frozen/EP，共同底座、单独展示。confirmatory suite 在源选择后显式列方法，不使用 `all` 或根据目标结果自动删方法。

**`configs/experiments/source_pilot.yaml`**：

```yaml
experiment_id: source_pilot
stage: pilot
suite_id: smoke
selected_datasets: [asvspoof2019_la]
model_id: ssl_aasist_source
input_role: select
allow_target_test: false
subset:
  rule: source_group_stratified
  max_groups_per_class: 256
  seed: 13
seeds:
  probe: [13]
  source_artifacts: [13]
  random_subspace: [13]
reports:
  compare_all_to: frozen
  require_coverage_report: true
```

正式确认扩为预登记种子 `[13,37,73]`；明确哪些种子改变 probe、U/M 或随机 U。相同缓存重复三次不是三个随机训练/适应种子。不能只因时间有限而让所有超参数都在所有外部数据上跑一遍再选。

---
### 8.7 服务器待补的原始数据合同模板

**`configs/data_contracts/asvspoof2019_la.yaml.example`**。其他数据复用同接口，不能复制其列位置假设。

```yaml
schema_version: "0.1.0"
dataset_id: asvspoof2019_la
contract_status: UNRESOLVED
dataset_release: null
layout:
  root_keys: [asvspoof2019_la_train, asvspoof2019_la_dev, asvspoof2019_la_eval]
  audio_globs: null
  path_resolver: null
protocol:
  adapter: null
  file_keys: [asv2019_train, asv2019_dev, asv2019_eval]
  format: null
  encoding: null
  delimiter: null
  header: null
  columns: null
  json_paths: null
  quoting: null
  join_key: null
labels:
  field: null
  raw_to_canonical: null
  unknown_policy: quarantine
  policy_id: source_authenticity
provenance:
  resolver: null
  source_mapping_ref: null
  group_quality: unknown
  speaker_field: null
  generator_field: null
approval:
  reviewer: null
  report_ref: null
  raw_schema_hash: null
  approved_at: null
```

此模板的null不能在加载时替换成“看起来像官方格式”的默认列。`inspect-data`的候选列只写proposed文件；审核后的lock才供parse命令使用。server上如果已有用户转换后的CSV或label文本，新增adapter配置，不要求还原成作者目录树。

### 8.8 预处理与增量划分模板

```yaml
schema_version: "0.1.0"
preprocess_id: source_detector_io
status: UNRESOLVED
reference_candidate:
  sample_rate_hz: 16000
  eval_num_samples: 64600
  origin: author_code_candidate_not_remote_fact
decode:
  library: null
  mono_policy: null
  resample_backend: null
  amplitude_policy: null
train_unit:
  crop_policy: null
  pad_policy: null
  num_samples: null
  augmentation_recipe_ref: null
eval_unit:
  crop_policy: null
  pad_policy: null
  num_samples: null
quality:
  corrupt_audio_policy: quarantine_and_report
  silent_policy: retain_with_flag
materialization:
  mode: on_demand
  output_root_key: work
lock_hash: null
```

```yaml
schema_version: "0.1.0"
split_policy_id: source_and_external
status: UNRESOLVED
seed: 13
role_candidates: [fit, source_val, select, cal0, audit, control_test, target_test]
ratios: null
counts: null
group_resolver_ref: null
holdout_policy_ref: null
existing_assignment_ref: null
allow_automatic_role_merge: false
incremental:
  parent_snapshot_ref: null
  preserve_existing_assignments: true
  new_group_default_role: unassigned
  known_group_policy: inherit_if_role_permissions_compatible
  conflicting_group_policy: quarantine_and_invalidate_claims
  relabel_policy: append_correction_event
  automatic_retrain: false
  automatic_resplit: false
approval_ref: null
```

可用`schema_version=0.1.0`配合不同snapshot/contract hash表示不同数据内容；**数据到达一次不需要把项目release虚增一次**。raw mapping/recipe产生真实语义变化时写变更记录与新hash，不沿用旧批准状态。

<a id="s8-training"></a>
### 8.9 源模型训练计划和配方模板

**`configs/source_training_plan.yaml`**：训练入口先解析本文件，再合入选中model recipe。不会因base默认选择SSL而禁止先做AASIST工程闭环。

```yaml
schema_version: "0.1.0"
project_version: "0.1.0"
plan_status: DRAFT
execution_environment: remote_only
source_dataset_id: asvspoof2019_la
stages:
  - {model_id: aasist_source, purpose: pipeline_validation_and_reference}
  - {model_id: ssl_aasist_source, purpose: primary_detector}
fit_snapshot_ref: null
source_val_snapshot_ref: null
preprocess_lock_ref: null
split_plan_lock_ref: null
allow_external_task_checkpoint: false
train_seed: 13
automatic_followup_full_training: false
```

**`configs/training/aasist.yaml`**，作者值是待核验起点；recipes的科学字段与runtime批量配置分别记录：

```yaml
schema_version: "0.1.0"
recipe_id: aasist_source_training
recipe_status: DRAFT
model_id: aasist_source
reference: R16
source_training: true
init_mode: native_initialization
trainable_scope: all_native_trainable_parameters
fit_role: fit
validation_role: source_val
selection:
  metric: eer
  direction: min
  tie_break: earliest_epoch
  final_test_access: false
training:
  max_epochs: 80
  optimizer: adam
  lr: 0.0001
  weight_decay: 0.0001
  scheduler: cosine_author_audited
  loss: categorical_cross_entropy
  class_weights_by_name: null
  sampler_policy: null
  augmentation_recipe_ref: null
  early_stopping: {enabled: false, patience: null}
reference_batch: 24
runtime:
  strategy: single_gpu
  per_gpu_batch_size: 1
  grad_accum_steps: 1
  precision: float32
  activation_checkpointing: false
  tune_only_on_source_preflight: true
checkpoint:
  save_last: true
  save_best: true
  resume_scope: same_recipe_same_snapshot
  required_resume_granularity: epoch_boundary
recipe_lock_ref: null
```

**`configs/training/ssl_aasist.yaml`**：

```yaml
schema_version: "0.1.0"
recipe_id: ssl_aasist_source_training
recipe_status: DRAFT
model_id: ssl_aasist_source
reference: R17
source_training: true
init_mode: generic_ssl_frontend_plus_native_backend
init_artifact_key: xlsr_300m
trainable_scope: joint_ssl_frontend_backend_head
fit_role: fit
validation_role: source_val
selection:
  metric: eer
  direction: min
  tie_break: earliest_epoch
  final_test_access: false
training:
  max_epochs: 80
  optimizer: adam
  lr: 0.000001
  weight_decay: 0.0001
  scheduler: null
  loss: weighted_categorical_cross_entropy
  class_weights_by_name: null
  sampler_policy: null
  augmentation_recipe_ref: null
  early_stopping: {enabled: false, patience: null}
reference_batch: 14
runtime:
  strategy: single_gpu
  per_gpu_batch_size: 1
  grad_accum_steps: 1
  precision: float32
  activation_checkpointing: false
  tune_only_on_source_preflight: true
checkpoint:
  save_last: true
  save_best: true
  resume_scope: same_recipe_same_snapshot
  required_resume_granularity: epoch_boundary
recipe_lock_ref: null
```

`scheduler=null`等字段表示尚未完成配方审计，不等于自动使用默认scheduler；锁定时需要把“无scheduler”明确写为`none`，或选定已审核配置。class weights、sampler、训练增强、preprocess与native index mapping必须在训练前解决。`reference_batch`是文献/源码参照，不是实际batch承诺；runtime预检后决定实际microbatch/累积，并写入recipe lock和偏离记录。自2026-09-16起新模型recipe模板默认总预算为80轮；此前已锁定的100轮recipe保持原字节身份，不能用模板更新倒写其合同或冒充exact resume。

建议先独立smoke 10–20个optimizer steps与一次完整验证管线，不将smoke输出注册为正式底座；首次正式运行只执行一个训练seed。用户是否启用额外长训练由已批准计划控制，不因一个smoke命令成功就自动耗满服务器。


---

<a id="s9"></a>
## 9. 离线准备、视图和不可变缓存

### 9.1 预处理由已批准的训练/推理合同决定

架构候选的解码、声道、重采样、截断/补齐、幅度规则须先在服务器审核，再锁定为训练与推理profile。SSL_Anti-spoofing作者示例的16kHz、64600样点仅是参考；不等于用户远程文件已按此预处理。[R1] 未批准profile不得用于正式训练或建立可复用特征；本机可以按合成profile测试接口。

先固定原模型检测单元，再在同一单元上生成 probe。不同视图不得独立随机裁剪；基线只看前约4秒时不得宣称对整条长录音完成了局部检测。长音频分块是另立协议，不在默认流程中悄悄改变。

**评价信道与 probe 分开**：构建器先对原音频施加评价信道得到观测文件；推理只读取该观测文件，再施加预定轻微 probe。隐藏干净版本和评价处理参数不进入适应器。

### 9.2 Probe 的实现要求

N=3 默认 `identity + additive_noise + mild_fir_eq`；可选 RIR 必须独立注册并核对来源/对齐，不因缺 RIR 阻塞初版。

`probe_default`的可执行默认定义：噪声采用CPU侧`numpy.random.Generator(PCG64(seed))`产生与检测单元等长的标准高斯白噪声，SNR由同一生成器在[25,35]dB均匀采样，不下载噪声库。顺序固定为先采样SNR，再采样噪声数组。EQ使用65抽头、Hann窗的`scipy.signal.firwin2`；归一化频率结点为[0,0.125,0.25,0.5,1]，逐结点dB增益独立采样于[-2,2]，线性增益为`10**(gain_db/20)`。完整卷积后取`full[32:32+L]`补偿线性相位延迟，不再次幅度归一化。系数设计和卷积用float64，最后转float32；NumPy/SciPy版本和实际系数进入probe元数据与数值身份。这个具体probe选择是待源域先导验证的启动设计，不是音频取证标签保持的已证明定理。

`source_probe`默认使用同样两种家族，为每个fit单元同时构造两个配对差分；加入其他噪声、RIR或EQ版本必须更新source_probe身份。变换参数不读取真假标签，两类/各家族平衡由离线抽样实现。`RMS(x)<1e-6`的近静音噪声视图默认等于原视图并标记退化，不能静默删除该音频。人工合成fixture测试应确认上述随机顺序、卷积边界和形状。

噪声按 `scale = RMS(x)/(RMS(noise)*10**(SNR/20))` 构造；有限信号、零能量噪声、近静音有明确分支。近静音不随机补内容，记录 probe 退化。FIR 记录系数、设计版本、长度和延迟补偿，避免所谓 EQ 实际成为强低通。

禁止无记录地逐视图峰值归一化或硬裁剪；记录长度、RMS、实测 SNR、峰值与范围溢出。处理强度和类别无关。噪声或增益被前处理完全消除时记录零响应，不作为有效新证据。

种子由 `SHA256(probe_seed, opaque_sample_id, input_sha256, probe_profile_hash, view_index)` 截取得到。不使用 Python `hash()`，不含进程号、GPU号、顺序、方法名、标签或攻击类型。所有方法共享同一 N 个视图。

主版禁止强去噪、神经重建、变声、词语删除和声源分离作为默认 probe；这些操作可能改变任务标签或取证信息，需另设研究。

### 9.3 tau0 与锚点生成

本项目源训练、source_val选模、冻结导出均完成后，在cal0真人分数升序数组上：

\[
k=\lceil(1-\alpha)n\rceil,\qquad \tau_0=s_{(k)},\quad 0<\alpha<1.
\]

程序下标 `sorted_scores[k-1]`；规则是严格 `s>tau` 判伪造。相同分数不随机拆开。它只控制本校准样本的经验 FPR，不是任意域部署保证。

先保存 tau0，再用 fit 原分数选 M。若有效锚点不足，报告类别/间隔分布和不足数量；不从 select/audit/target 偷加。U 和 M 可共用允许的 fit 池；M 的类、来源、生成器和间隔覆盖要独立汇总。

只有选择完成后才能建立可选 cal1 的各方法 tau_j；这些分数必须通过各自完整适应流程生成，而不是仅重用 frozen 分数。

### 9.4 缓存格式与体积

默认保存最终语句嵌入 `[unit,N,d]`，float32；大数据用分块 `.npy` 与 JSON/SQLite 索引，U/M/Fisher/头用小 `.npz`。读取 `allow_pickle=False`，数组禁止 object dtype。不每条保存一个 `.pt`，不将百万样本压成单个必须全解压的 `.npz`。

按实际 d 规划：`feature_bytes=units*N*d*4`。例如仅在 d=160、N=3 时，100万条约1.79GiB 特征；该算术不包括索引、多种子、失败块或模型，也不宣称真实样本量。波形、SSL帧缓存不属于默认产物。

缓存 key 至少包含：输入内容hash、自有baseline/selected_checkpoint/线性头身份、wrapper数值版本、锁定decode/probe、seed、dtype和数值模式；另存所属snapshot/split/label-policy的血缘，不把路径当数据身份。改变 r、rho、gamma、K、loss 权重或 M 抽样**不重跑冻结编码器**；改变 U 的输入源池只重估其依赖资源。

### 9.5 三类 hash，避免迁移路径使所有缓存失效

| hash | 包含 | 用途 |
|---|---|---|
| `scientific_hash` | 方法、数据角色/ID、标签策略、目标/约束/参数、视图规则 | 比较身份与封存 |
| `numerical_hash` | 权重、wrapper、解码/精度/算子约定、实际内容 hash | 判断数值缓存兼容 |
| `resource_hash` | 路径、设备、worker、分片布局、环境快照 | 执行与恢复审计 |

更换绝对路径不改变相同内容的 sample_id 或 probe seed。layout 改变只重建索引，不将同一数值块误认成新科学实验。环境和设备差异也可能影响数值，不能仅因语义 hash 相同跳过跨环境 parity。

### 9.6 原子提交与恢复

先生成固定输入 ID 清单，`layout_shard=stable_hash(id)%num_layout_shards`，默认32个逻辑分片；物理 worker 只是领取这些分片，不把 GPU 数写入样本随机性。

每个 worker 独占一个块，写 `*.partial` → flush/fsync/close → 计算 hash → 同文件系统原子 rename → 最后提交 sidecar。sidecar 包含预计/实际行数、ID集合校验、形状、dtype、N、父资源 hash。只有 sidecar和数据校验共同通过才是完成。

恢复只跳过已验证完成块；未完成块重新计算。合并只合并索引，并检查每个预期 ID 恰出现一次且有全部视图。不能默认用会补齐重复样本的训练 sampler。禁止并发写同一 HDF5/同一 mmap 区间。

运行锁记录 run_id/PID/主机与心跳；不因锁存在就永久阻塞，也不自动删别人的锁。失效锁需明确恢复命令确认归属。优先磁盘余量≥预计新增产物的1.2倍并保留系统余量；预检不足就缩减**预登记数据范围**或更换路径，不在运行中丢数据。

### 9.7 新数据与新模型的按依赖刷新

新输入追加到新的cache index，仅对新内容与缺失视图编码；旧数值块以content-address引用，不覆盖。一个旧块能在多个snapshot引用时需验证相同输入内容、模型、decode和probe，而不是仅看sample_id相同。

正式训练期间不使用最终语句特征缓存加速正在更新的模型；该缓存只在checkpoint冻结后产生。冻结SSL前端变体可选择更早切分点缓存，但必须有单独冻结范围和训练配方，不能冒充联合微调。

新模型导出后，即使网络名称仍为SSL-AASIST，也生成新的baseline_id和资源bundle。只新增外测音频不重估源U/M；新增fit并决定重训则整条模型依赖链重新生成。`refresh-plan --delta`必须先列reusable/rebuild/blocked及理由，默认dry-run，不能自动触发长训练。

---

<a id="s10"></a>
## 10. 双卡 A6000 远程执行、性能与故障

### 10.1 预检按阶段进行，不要求先有鉴伪checkpoint

R0检查GPU/驱动、Python/PyTorch/CUDA、CPU/内存、数据与工作盘读写和空间，记录互联；不根据A6000型号推测NVLink或系统内存。R1先补完数据合同与划分；训练预检R2只需架构、通用初始化（选用SSL时）、fit/source_val和锁定recipe。

自有训练完成后R4运行frozen bundle的wrapper parity。R5再测试冻结提取batch units 1→2→4→8，N=3时实际波形数量为units×3；结果决定部署batch，不改输入长度或视图数。

每卡预算上限`min(40GiB, 0.85*preflight_free_GiB)`是保守策略而非可运行保证；训练要测完整前向、反向、optimizer step和保存/恢复，不能仅测inference。记录峰值显存、延迟、吞吐与OOM；不在最终目标集上挑精度和batch。

### 10.2 两张卡的任务策略

源训练阶段见§10.6，允许按已锁定配方使用DDP；以下规则只描述训练完成后的计算。

编码阶段：GPU0/GPU1 各一个冻结模型副本，按计划领取不同逻辑分片。进程用 `CUDA_VISIBLE_DEVICES=<physical_id>`，内部 `cuda:0`。不把两个独立 R 包装进 DDP；DDP 的梯度同步语义不符合逐样本独立适应。[R13]

缓存评分：先测试 CPU 批量128；再比 CPU256或单GPU。小矩阵计算可能不是系统瓶颈，不为“用满双卡”增加重复编码或进程。双卡也可分别跑不同已封存参数/基线作业，单个作业不共享目标状态。

完整算法移植：每卡一个独立方法/分片作业。原在线流的历史扩展不能任意拆分；只可在独立流或独立实验间并行。

### 10.3 CPU/I/O 与精度

每编码进程起始4个 DataLoader worker，但由实测CPU资源限制；worker=0时不传 prefetch，关闭 persistent_workers。解码 worker 只处理CPU张量，不创建CUDA模型。联合限制BLAS/OpenMP线程，避免两进程各自开满所有核心。[R14]

默认 FP32，TF32/AMP关闭；U累计FP64。需要加速精度时建立独立 numerical profile，以源 fixture 检查原分数、适应分数、决策/翻转差异。半精度提取后 cast float32 不是原FP32提取；必须使用不同缓存身份。

计时包含 warm-up记录与必要CUDA同步；吞吐计数用实际检测单元数、视图数、forward/backward数，不能仅数Python调用。缓存适应耗时单列，端到端成本必须计入视图构造与编码。

### 10.4 OOM、数值故障与数据故障分开

| 情形 | 行为 | 结果记录 |
|---|---|---|
| 编码 OOM | 同一输入拆小 microbatch；不改长度、N、dtype或seed | retry、真实batch、额外耗时 |
| batch=1仍OOM | 停止相关任务并标 `BLOCKED_RESOURCE` | 不换随机模型或删样本 |
| 更新中非有限但原分数有限 | 本条回退原分数，R清零 | `fallback_numeric`，保留在主指标 |
| 批量非有限 | 固定输入串行重放隔离 | 只处理失败条，计重放成本 |
| 原音频/原特征/原分数非法 | 不伪造分数；保留ID和错误 | `invalid_input`，全run覆盖不完整 |
| 配置、标签越权、hash不匹配 | run级错误，立即停止 | `FAIL`，不得视为坏样本忽略 |
| published筛选为空 | 按已审计原规则不更新/返回合法分数 | `no_selected_update`、选择率0 |

更新后锚点 loss 大、预测翻转或可疑分数降低，均不是主法的数值回退条件。它们可能是真实失败，必须进入分析。最终分数如果有限但效果不好不能重跑不同参数再覆盖。

### 10.5 运行快照与恢复命令

核心调度进程与其模型worker分开记录PID、环境和错误；GPU显存归属以实际模型worker为准。每个作业路径形如 `runs/<experiment_id>/<run_id>/`，run_id由封存配置hash与启动随机/时间标识生成，所有方法分目录但共享协议ID。禁止覆盖已存在run。

```text
run_manifest.json       # 全部hash、父产物、环境、访问权限
resolved_config.json    # 覆盖完成后的唯一配置
job_status.json         # 状态、心跳、完成块、失败ID
scores/                 # 每method每shard无标签分数
scores.seal.json        # 评分完成、不可更改的hash清单
coverage.json
runtime.json
traces/                 # 只保存预登记诊断ID，不持有计算图
metrics/                # 由独立评价命令生成
report.md
```

`resume --run` 只能恢复相同科学/数值配置的未完成作业；参数、精度或方法变化创建新run。恢复不重新选择方法或阈值。共享缓存只读，半成品不能被下游方法使用。

<a id="s10-training"></a>
### 10.6 自行训练时的双卡策略：与EP独立更新严格区分

| 阶段 | 默认策略 | 梯度同步 |
|---|---|---|
| 单模型训练预检 | 单卡、逐级增大microbatch | 无跨卡同步 |
| 两个独立底座/seed | 每卡一个独立训练run，资源允许时并行 | 不共享优化器/梯度 |
| 单个SSL-AASIST联合训练 | 可选两卡数据并行（DDP），一进程一卡 | 同一个源训练模型需要同步 |
| 冻结特征提取 | 每卡一个冻结副本、不同分片 | 无梯度 |
| EP批量适应 | 每条独立R和投影 | 严禁跨样本/跨卡归约 |

不能把“EP不使用DDP”扩展成禁止源监督训练DDP。PyTorch DDP按进程同步同一模型的梯度；它不会把两张48GB显存自动合并为一张96GB卡。[R13] 旧环境是否支持当前launcher/API要预检，支持时使用其兼容入口；不因要用新launcher自动升级整个fairseq环境。

源训练初始`microbatch=1`，在fit小样本完整train step上逐级试1/2/4/8，记录optimizer状态分配后的峰值；无OOM才提高。根据选定reference有效batch和资源选择梯度累积（gradient accumulation）。完整等长窗口、每rank相同样本数时`effective_batch = world_size × microbatch × accumulation_steps`；尾窗口必须另按实际样本/权重处理并记录。改变BN的microbatch统计、Dropout/增强和精度意味着训练轨迹可能不同，不能以有效batch相同声称严格单卡等价。

**带权交叉熵的特别约束**：不能简单将各microbatch的mean loss除以累积步数，尤其类别比例不同时分母不同。先累积未归一化加权loss梯度，再按该更新窗口的总权重规范化；DDP还要处理其梯度平均的world_size因子。应写与大batch的梯度对照测试（使用不含BN/随机性的toy网络），确认数值含义。尾批与多个rank样本数不同时不能默默沿用固定分母。

DDP训练需：按epoch设置sampler seed、明确训练尾样本drop/repeat政策、epoch统计用全局有效计数、rank0写checkpoint、所有rank一致决定早停/失败。验证默认所有rank先同步，rank0使用解除DDP包装的底层模型在eval/no_grad下跑完整source_val并广播结果，其他rank等待，避免只在rank0调用DDP forward触发collective死锁；也可使用不重复填充的验证分片并精确合并ID；不能因DistributedSampler填充导致验证样本重复。checkpoint保存每rank必要随机状态；恢复世界大小改变时不再声称逐步完全等价。

### 10.7 训练显存优化与停止条件

优先在不改变模型结构的前提下调microbatch、梯度累积和CPU加载。训练自动混合精度（automatic mixed precision, AMP）可单独启用，但其数值配置必须锁定，验证有限loss/梯度/溢出处理和源验证表现；它不是FP32训练的位级等价替换。[R18] FP16启用合适scaler；BF16是否可用由真实框架/模型算子预检决定。激活检查点（activation checkpointing）仅在该SSL实现支持且前后向验证后使用；不硬编码一个不兼容的最新版API。

主SSL配方不能因为资源不足静默冻结前端。合法处理为：减microbatch、审核累积/精度/重计算→仍不足则`BLOCKED_RESOURCE`；另申请独立的`ssl_aasist_frozen_frontend`先导，不修改主配方名。以AASIST自有训练先完成工程闭环；是否足以支持主结论由实际实验决定。

训练失败不跳过“最难的样本”继续选最优checkpoint；坏输入提前在snapshot阶段检查，运行期发现的新坏输入使训练run和数据覆盖报告显式更新。OOM恢复从合规checkpoint开始，不能自动更短音频；改变科学配方必须创建新run。

训练预算单列（训练GPU小时/峰值显存/epoch、预处理开销和checkpoint磁盘），不能只报告EP的几步更新时间。底座训练完成后，多数机制基线共享同一checkpoint和最终特征缓存；不要为25项注册方法分别重训底座，也不要默认对全部五数据集做全量超参网格。


---

<a id="s11"></a>
## 11. 实验路线：先导、基线、封存与停止条件

### 11.1 分阶段推进：先有自有源模型，再做适应比较

| 阶段 | 数据与任务 | 完成标准 |
|---|---|---|
| E0 数学/协议 | 纯张量、模拟清单、小型合成音频 | §13 单测通过；不产生鉴伪性能声明 |
| E-source 自行训练 | 合同已核验、fit/source_val先封存；AASIST训练闭环与SSL-AASIST主配方 | 自有checkpoint、训练/选择血缘、冻结parity通过 |
| E1 源域机制先导 | 自有模型固定后，fit准备U/M，select跑smoke和机制基线 | H1–H5有可解释诊断；消融作用确实存在 |
| E2 已有方法移植 | 同底座、同select环境完成P2 | 五项审计、梯度/重置和资源验收；未完成明确标注 |
| E3 确认性评价 | 封存A–D，2019评估及选中外部集 | 固定配置/清单；分数先封存后评价 |

只有源域先导未被完全简单基线解释，才扩大计算。外部顺序建议 In-the-Wild、2021 DF优先，再WaveFake和Codecfake专项；不需要每个消融都跑完整五集，需提前声明哪些比较在哪些集运行。

### 11.2 A–D：生成来源×信道

| cell | 伪造来源 | 信道 | 目标 |
|---|---|---|---|
| A | 源模型见过 | 熟悉/无额外处理 | 原条件不应明显受损 |
| B | 源模型见过 | 留出处理 | 是否修复正常声学变化 |
| C | 源模型未见 | 熟悉/无额外处理 | 是否抹去未知伪造证据 |
| D | 源模型未见 | 留出处理 | 双重变化中的纠错—损害权衡 |

每格配相应真人参照，对真假施加相同分布的评价处理。真人本身没有“生成器已见/未见”标签；A/C共享真人时汇总去重，统计不能当作两套独立真人证据。

区分新噪声实例、全新处理家族、强度外推、处理组合。源阶段已见噪声而目标只换随机种子不能叫未知信道家族。对未知生成器缺元数据的外部集合，只报告跨数据集，不升级主张。

### 11.3 调参与选择规则

所有候选只在 select 上运行，M/U/Fisher不从select重建。默认每方法至多12组候选，不穷举 r×rho×gamma×lambda×lr×K。先在fit/源开发诊断 probe幅度、损失量纲和梯度是否合理，发现实现错误则修复后版本化重新运行，而不是混合旧结果。

建议固定选择规则：相对同底座frozen，在每个预登记源开发环境满足 `FPR_m−FPR_0≤delta_fpr_allowance` 与 `TPR_m−TPR_0≥−delta_tpr_allowance`；可行候选中最大化各环境等权 balanced accuracy=`0.5*(TPR+1−FPR)`，再按更低有害翻转、更低更新成本和固定配置ID打破平局。

delta的默认0.01是先导容忍度而非置信保证。无可行候选标 `no_feasible_source_candidate`，按预先规定的最小约束违反程度保留一个候选用于失败分析，但不能改写为通过安全条件；确认性主张受限。不能看到目标结果后放宽容忍度。

目标函数不同可以独立源选择学习率/正则权重，但信息、候选试验数和选择规则相同。fixed_source_adapter按其离线训练需要配置合理步数；标量17点是每样本求解预算而不是17次目标标签调参。

### 11.4 种子分离与输入比例

分别保存 `probe_seed / source_pair_seed / anchor_seed / random_U_seed / model_training_seed`。同一自有底座权重上的U/M种子复核不是独立源训练seed复核；第二模型和新增训练seed在预算允许时分阶段扩展。首轮一个种子；正式主方法/关键对照至少3个预登记构造种子。可以固定probe仅改变U/M做低成本稳定性测试，但必须这样命名。

episodic正确实现时，改变顺序或把同一批ID组成不同真假比例**不应改变这些ID各自分数**；它检验状态污染/跨样本统计，不是额外泛化能力。流式扩展才将比例/顺序作为方法依赖因素。单一类别区间不计算AUROC/EER。

### 11.5 测试封存与声明条件

进入 E3 前生成 `preregistration.json`：数据ID/版本/标签策略、全部方法、已选参数与种子、checkpoint训练和选择范围、U/M/头/阈值、cell划分、处理和视图、主指标/置信区间、允许失败策略及主张接受条件。

`freeze` 输出 token 为此文档hash的引用，不是密码。确认性评分命令必须携带该token，参数不符拒绝。这种工程封存不是公共预注册或防恶意篡改证明，只提供本项目可追溯边界。

确认结果不理想也保留。修改设计后建立新探索实验；已经看过的目标集标为已访问，不能继续称为全新封存检验。必须先完成分数再由独立进程读取目标标签；运行期不看目标指标调迭代/阈值。

### 11.6 支持/收窄/停止规则

静态方法相当：不支持逐样本训练必要；标量相当：复杂R的必要性不足；PCA-U相当：响应特异性不足；logit/Fisher相当：单侧间隔的独特性不足。FPR下降但未知伪造TPR明显受损：证据保持目标未实现。

上述情况不是要求删除研究或保证可以发表；应据真实效果收窄方法和主张。没有全部指标第一不自动否定价值，但不能用平均数掩盖直接违背主张的失败。

---

<a id="s12"></a>
## 12. 评价、统计与必须生成的报告

### 12.1 分数文件与标签后置

每行至少：`sample_id, method_id, score_before, score_after, status, steps_completed, r_fro, keep_active_fraction_final, runtime_ref, scientific_hash, numerical_hash`。没有R的方法这些R字段为null，不填0伪装发生过适应。trace独立保存，不把大型张量写入CSV。

评价 join 要求预期ID唯一、无重复、无越界、无未声明缺失；同源派生的ID差异由mapping说明。`fallback_numeric`用原分数计入，不能删；`invalid_input`无分数时标不完整并报告每类覆盖及共同有效集比较，不用交集指标替代完整率。

`evaluate`是唯一读取目标标签的程序；`score`不能接 `--labels`。操作系统层可将评价sidecar置于独立位置/权限，接口隔离不是所有恶意读取的安全保证。

### 12.2 指标定义与分数方向

\[
FPR=P(s>\tau\mid y=0),\quad TPR=P(s>\tau\mid y=1),\quad FNR=1-TPR.
\]

主表使用tau0；另表可使用cal1生成的方法tau_j。AUROC正类为spoof，较大分数更伪造。EER基于同方向ROC/DET并按相同分数成组，固定插值/求交规则；不得按样本文件顺序拆ties。

官方评测脚本可能采用“分数大为bonafide”的约定，接入前显式转换并在合成分数验证，不根据目标EER试两个方向取更好。保留项目tie-aware EER与官方复核定义的区别，不无声切换。[R15]

至少单测完美排序、完全反向、全相等、单类、阈值等值、重复分数以及严格单调变换。单类AUROC/EER返回null与原因，不能用0代表完美。

### 12.3 按类别分解有害/有益翻转

令 \(\hat y_0\) 与 \(\hat y_m\) 为**同一主阈值**下冻结与方法结果，\(n_y\) 为类y总数：

\[
H_y=\frac{\#\{y_i=y,\hat y_{0,i}=y,\hat y_{m,i}\ne y\}}{n_y},
\quad G_y=\frac{\#\{y_i=y,\hat y_{0,i}\ne y,\hat y_{m,i}=y\}}{n_y}.
\]

必须核对 `FPR_m-FPR_0=H0-G0`、`TPR_m-TPR_0=G1-H1`。同时可报告按“原本正确/错误样本数”归一的条件率，但使用不同字段名，分母为0输出null。

低H不一定好：完全不更新即可低伤害，但也可能无纠错。比较G/H、FPR/TPR与成本，而不是只优化一个有害翻转率。

### 12.4 统计区间与有效独立单位

同一方法对的置信区间使用配对来源组bootstrap，默认1000次、95%；同一组及其多个处理视图一起重采样。不能把相同原语音派生样本当独立观测；跨cell共享真人/来源也要一并处理。

生成某bootstrap重复只有一类时，该重复的AUROC/EER无定义，记录跳过数量及可用重复，不反复采到结果好才接受。来源映射不完整时声明分组退化与潜在区间乐观性。种子波动和数据采样不确定性分开，不把三个种子当三个独立大数据集。

数据集间优先分别报告及预设等权macro汇总，不默认按大小混合；不同标签策略禁止合并总EER。多方法/多条件主次指标预先区分，避免只挑显著结果。

### 12.5 五个必交表与机制诊断

1. **完整方法表**：相同底座、协议、信息权限、FPR/TPR/EER/AUROC、成本；published-port与stream分开。
2. **目标×保持六格表**：说明特征一致性是否优于熵目标，以及源保持带来多少变化。
3. **替代解释表**：PCA/随机U、logit/L2/Fisher、固定源/静态/标量。
4. **A–D与按类翻转表**：说明修复何种变化、伤害何种来源。
5. **覆盖和资源表**：样本数、无效/回退、源数据权限、视图、前后向、峰值显存和端到端时间。

机制日志包括末步后实际margin违反率、累计/末步keep激活、梯度范数、投影触发、R大小、源audit退化，以及预登记样本的分数轨迹。audit不进入损失或回退。

图由真实分数生成，不能手填“预期改进曲线”。输出 `report.md`，实验未做写`not_run`。报告必须把“文献已发表结果”“本项目移植实测”“CPU数学测试”分开。

---

<a id="s13"></a>
## 13. 验收测试与 AI Coding 任务拆分

### 13.1 数学与机制测试清单

| 编号 | 必测项目 | 验收要求 |
|---|---|---|
| T01 | R=0、K=0 | frozen分数一致 |
| T02 | 行列矩阵方向 | 显式 U R Uᵀ 与实现一致 |
| T03 | 视图分母 | 使用Nd；总体方差而非N−1 |
| T04 | 首步梯度 | 与 §4.6 解析式一致 |
| T05 | 间隔初值/保持单独更新 | loss、gradient为0；不更新 |
| T06 | K=1 lambda不变性 | 多lambda输出完全对应 |
| T07 | K>1激活fixture | 构造确实违反锚点间隔的情况，保持产生作用 |
| T08 | Frobenius投影 | 每条≤rho；零范数合法 |
| T09 | 正交补与线性分数界 | 数值检查；Uᵀw=0退化 |
| T10 | U→UQ正交换基 | 适应器/主法在对应旋转下等价；不要求diagonal-Fisher此性质 |
| T11 | 非中心化二阶矩 | 常量非零差分仍保留响应；类/家族/来源权重正确 |
| T12 | PCA对照 | 只估基时中心化，不改变分类器输入 |
| T13 | 逐样本/批量 | B=1/2/7/128、尾batch、顺序、不同chunk一致 |
| T14 | 批量反传/投影 | sum与串行对应；故意mean/sharedR错误被测试检出 |
| T15 | 两种熵不同 | 对相反类别视图，EM低、MEMO边际熵高 |
| T16 | 稳定熵 | 大正负logit及梯度有限，联合视图梯度正确 |
| T17 | 替换保持 | L2/logit/Fisher不残留margin |
| T18 | Fisher | 逐样本梯度平方均值与解析式对应；不来自零margin梯度 |
| T19 | 标量基线 | 17点包括0，半径合法，等值规则确定 |
| T20 | 翻转恒等式与ROC ties | §12全部边界有单测 |

### 13.2 工程和远程验收

| 编号 | 项目 | 不通过时的行为 |
|---|---|---|
| C01 | local profile不访问真实资源、不导入大模型 | 修复权限/延迟导入，不以缺资源跳过代码 |
| C02 | 配置重复键、未知键、非法覆盖/角色 | fail-fast |
| C03 | 标签防火墙：改变目标sidecar不改变score | 视为协议错误，不继续结果 |
| C04 | 同源跨角色与历史checkpoint访问审计 | 限定主张或重新划分，不能静默忽略 |
| C05 | 相同ID/输入hash视图跨顺序/worker/方法相同 | 修复随机键 |
| C06 | wrapper/导出头/R0与原模型一致 | 停止真实实验 |
| C07 | E0/h0参数、mode和buffer冻结 | 修复隐式train或统计更新 |
| C08 | cache hash与形状/身份、重定位、失效 | 拒绝错缓存 |
| C09 | 进程中断、partial、原子提交、缺片/重复 | 恢复完整清单；不删除难样本 |
| C10 | 数值失败单条隔离、冻结回退、覆盖统计 | 无效不得记为正确 |
| C11 | published-port梯度路径/完整重置；core↔bridge协议与旧语法解析 | 不宣称复现完成 |
| C12 | 参数范围、选择历史和原算法差异审计 | 标为未完成，而非自动降级 |
| R01 | 单卡/双卡与batch数值对应 | 先回正确通路，不混不同数值结果 |
| R02 | OOM拆分保持N/长度/精度 | 单条不行则资源阻塞 |
| R03 | 视图forward、backward、求解次数、总耗时 | 不只报R参数量 |
| R04 | scores封存后才读取标签 | 不接受运行期目标调参 |

文末提供的是待抽取测试模板，不把本项目任何T/C/R任务预填为PASS。Coding工具应将各项映射到真实pytest文件，记录实际执行和跳过原因；合成关系不能证明鉴伪有效。

### 13.3 数据未知项、增量和源训练的专门验收

| ID | 必测合同 | 关键反例/验收 |
|---|---|---|
| D01 | 本机未解析null不阻塞结构开发 | shape-only可过；真实train/score必须拒绝缺必需lock |
| D02 | 多格式原始协议 | CSV/TSV/空白分隔/JSONL、带/无表头模拟fixture；不猜列 |
| D03 | 标签映射 | 相反0/1、未知标签、空值、同内容冲突；未映射不能默认spoof |
| D04 | source_val与select分离 | 训练job含select/cal0/target引用必须失败 |
| D05 | 增量幂等 | 同批重复导入不新增样本、不变旧ID/角色 |
| D06 | 冲突组与泄漏 | 新来源关联跨fit/test时隔离并输出影响范围，不自动移动 |
| D07 | 标签更正 | 新事件与新snapshot；旧label/指标不可原地覆写 |
| D08 | 原子snapshot | 中断/失败不发布active半成品；父snapshot仍可读 |
| D09 | 依赖刷新 | 新外测只补编码；新checkpoint导致全部模型相关缓存失效 |
| D10 | 预处理接口 | 不同train/eval策略均记录；profile变更拒用旧缓存；不覆盖原音频 |
| S01 | 架构与初始化身份 | AASIST/SSL-AASIST插件可建模；只加载允许通用SSL参数 |
| S02 | 源训练参数更新 | joint模式前端/后端/头有合法梯度；frozen变体另名 |
| S03 | native类别与权重映射 | canonical↔native双向对应；CE权重按类别名一致 |
| S04 | fit梯度/source_val选模 | 验证不更新参数/BN；目标侧路径不进入worker |
| S05 | smoke不冒充训练完成 | export拒绝synthetic/smoke/未finalized权重 |
| S06 | checkpoint读写/恢复 | 同snapshot/recipe的epoch边界恢复；改变数据拒绝exact resume |
| S07 | 验证无重复/无目标选择 | source_val每个ID恰一次；EER无定义不选best |
| S08 | 微批和DDP归一化 | 不含BN随机性的toy模型上加权梯度与大batch对应 |
| S09 | 训练→冻结导出 | selected checkpoint与head/wrapper/R0一致，初始化血缘完整 |
| S10 | 所有方法共享底座 | 同一比较组使用相同checkpoint；不按目标效果挑seed |

这些都是**待实现、待执行的测试**。本机的模拟fixture可以验证接口与数学关系；真实格式、训练收敛、数据来源与GPU行为要在服务器补验。测试名称或参考代码的存在不能预填PASS。

### 13.4 本机任务：从空项目开始

| 阶段 | 依赖 | 必交代码/文件 | 完成条件 |
|---|---|---|---|
| L0 项目初始化 | 无 | v0.1.0骨架、README、AGENTS、状态表、依赖策略 | 不假设已有实现/权重；不覆盖用户其他文件 |
| L1 配置和合同 | L0 | model/data/recipe/Method registry、raw/preprocess/split模板、schema | null可用于草稿；选阶段检查；未知键/越权失败 |
| L2 数学核心 | L1可部分并行 | 串行EP、独立批量、目标/保持/投影、数值测试 | T系列相关单测；本机无需真实模型 |
| L3 数据接入与增量 | L1 | 五类插件框架、通用适配器、label/group、snapshot/delta/审批/清单 | D01–D10合成fixture通过；真实格式DEFERRED_REMOTE |
| L4 源模型训练链 | L1/L3 | 两种架构factory、原文审计、source trainer/worker、resume/export、recipe模板 | S系列toy/结构/权限测试；真实训练NOT_RUN |
| L5 适应基线与评价 | L2/L3 | P0/P1基线、P2五个端口合同、缓存、指标、封存 | T/C对照测试；不把机制目标冒充完整算法 |
| L6 调度和交付 | 所需前序 | training/extract launcher、CLI、恢复、远程runbook、版本检查 | 合成端到端；源码包与所有未运行项明确 |

L0–L2完成先报告，不自动做真实训练。L4可以与L5并行开发，不必等25个方法完成才让服务器开始源模型训练。第三方旧依赖不能阻塞数据合同与纯张量模块。

### 13.5 远程任务：未知信息在这里补全

| 阶段 | 前置 | 执行内容 | 放行条件 |
|---|---|---|---|
| R0 环境/资源盘点 | 所需本机接口 | 数据root、协议、通用初始化、代码、环境与GPU/磁盘 | 只查选中资源；不要求已训练任务checkpoint |
| R1 数据合同与划分 | L3/R0 | layout/label/预处理/来源审核，生成snapshot和训练前split lock | D系列真实核验；train/source_val/cal/test边界明确 |
| R2 架构/训练smoke | L4/R1 | 自行建模、native标签、完整train step、恢复、显存预检 | S01–S08；smoke不能导出正式底座 |
| R3 自主正式训练 | R2/已批准recipe | AASIST工程参考与SSL-AASIST主配方；source_val选checkpoint | 训练manifest、完整日志、自有权重FINALIZED |
| R4 冻结导出 | R3 | frozen bundle、线性头、wrapper/mode/buffer parity | C06–C07/S09；无外部任务权重替代 |
| R5 特征与源资源 | R4 | 单卡/双卡缓存预检、分片恢复，U/tau0/M/Fisher | 正确数值、完整ID与资源hash |
| R6 源先导与方法移植 | R5/L5 | smoke→机制比较→五项已有算法 | 真实激活/收缩/负结果；每项日志与权限 |
| R7 参数封存 | R6 | select结果、方法完成状况、preregistration | 不在目标上挑模型/超参 |
| R8 确认性评分 | R7 | A–D、预选外部集；先score seal再读label | 覆盖完整、类别/来源主张受证据支持 |
| R9 汇总 | R8 | 指标/翻转/区间/成本/失败分析 | 可由分数重建Markdown报告 |
| R-data 新批到达 | L3，独立于run | 增量盘点→diff→来源/label冲突→批准新snapshot | 不自动重训、不覆盖已封存实验 |

R6若静态/标量已解释全部收益，可停止复杂化并报告；五项已有方法未完成时不能宣布完整竞争性验证。训练未收敛或只能跑AASIST先导时，明确底座范围与未完成SSL任务，不以下载作者checkpoint跳过。

### 13.6 阶段交付格式与初始状态

`IMPLEMENTATION_STATUS.md`按L/R/D/S/T/C任务ID记录：状态、文件、实际命令、退出码、日志、输入hash/commit、未执行项及原因。首次所有实现任务`TODO`，远程核验`DEFERRED_REMOTE`或`NOT_RUN`；项目整体`NOT_STARTED`。

`PASS`必须来自后续实际执行，不继承本对话任何草稿检查为项目代码已通过。每完成一个任务更新状态，不在文档初始化时填满完成日期。

---

<a id="s14"></a>
## 14. CLI 与远程命令合同

**以下命令是 Coding 工具需要实现并测试的接口，不是本文件已经提供的可直接安装软件。** 所有命令支持 `--help`；全局选项 `--config / --profile / --paths` 位于子命令之前。`--profile`使用ID，由`configs/profiles/<id>.yaml`解析。

### 14.1 CLI合同：接口需要开发，不是现成命令

| 子命令 | 必要输入/选项 | 输出与限制 |
|---|---|---|
| `validate` | `--level structure|resources --stage STAGE` | 按阶段检查；structure允许null，不读服务器 |
| `plan` | `--experiment FILE --task TASK --out FILE [--bundle FILE] [--dry-run]` | DAG与未解决项；preview不执行 |
| `inspect-data` | `--datasets ID[,ID] --out DIR` | inventory与格式候选；不自动批准label |
| `propose-data-contract` | `--inventory FILE --out FILE` | 原始协议、label、group候选；status=PROPOSED |
| `approve-contract` | `--kind raw|label|group|preprocess|split|recipe --proposal FILE --review FILE --out FILE` | 有核验记录才发布lock；不把propose等同approve |
| `stage-data` | `--inventory FILE --raw-contract FILE --label-policy FILE --group-policy FILE --out DIR` | 规范化到staging snapshot；不做科学角色分配 |
| `propose-splits` | `--staging FILE --policy FILE --out FILE` | 基于真实类别/来源组计数提出划分；不自动批准 |
| `build-manifests` | `--staging FILE --split-plan FILE --out DIR`；或使用inventory/raw/label/group四项重建同一staging | 已批准split绑定为发布snapshot；二选一输入，不混用 |
| `ingest-delta` | `--parent FILE --inventory FILE --contract FILE --out DIR [--dry-run]` | 只计算增量和冲突；不提交不重训 |
| `commit-snapshot` | `--delta FILE --review FILE --out DIR` | 只提交已批准增量，原子写入新snapshot |
| `refresh-plan` | `--delta FILE --artifacts FILE --out FILE [--dry-run]` | reusable/rebuild/blocked依赖清单 |
| `resolve-training-recipe` | `--model-id ID --snapshot FILE --preprocess FILE --out FILE` | 待批准recipe；不自动开始训练 |
| `inspect-model` | `--model-id ID --mode architecture|frozen_bundle [--bundle FILE] [--source-fixture FILE] --out DIR` | 未训练架构检查与冻结权重parity分开 |
| `preflight` | `--plan FILE --stage source_training|frozen_extract|published --out DIR` | 每阶段真实前后向/显存/环境锁 |
| `train-source` | `--recipe FILE --phase smoke|full --run-output DIR` | 仅fit/source_val；full须recipe LOCKED；smoke无正式导出资格 |
| `resume-source` | `--run DIR --checkpoint FILE --mode epoch_boundary` | 同snapshot/配方；不自动适配新数据 |
| `finalize-training` | `--run DIR --out FILE` | 根据source_val锁定选模并写FINALIZED manifest |
| `export-frozen` | `--training-manifest FILE --out DIR` | 自有模型/头/baseline_id/parity待审计产物 |
| `bind-artifacts` | `--bundle FILE --paths-template FILE --out FILE` | 创建私有绑定，不改Git模板或模型科学身份 |
| `extract` | `--plan FILE --worker-slot I --worker-count W` | 只读自有冻结bundle，分片特征 |
| `merge-cache` | `--plan FILE --out FILE` | 检查所有ID恰一次及完整视图 |
| `build-artifacts` | `--plan FILE --cache-index FILE --out DIR` | fit/cal0许可下U/M/tau0/Fisher等 |
| `run-suite` | `--plan FILE --suite-id ID --phase select|confirmatory --run-output DIR [--frozen-spec FILE]` | 确认性运行必须已封存；只读自身目标样本 |
| `seal-scores` | `--run DIR` | 分数与覆盖校验，无目标标签 |
| `evaluate` | `--run DIR --labels FILE --out DIR` | seal后读评价标签；不反馈当前run优化 |
| `evaluate-checkpoint` | `--model-id ID --checkpoint FILE --protocol FILE --audio-dir DIR --out DIR [...]` | 评测任意能严格加载到已审计AASIST/SSL-AASIST结构的参数；不要求训练完成或FINALIZED |
| `select-methods` | `--metrics FILE --plan FILE --out FILE` | 仅source-select角色；不选择源训练checkpoint |
| `freeze` | `--plan FILE --selection FILE --out FILE` | 确认性计划hash，不是公共注册或安全凭证 |
| `resume` | `--run DIR [--check-only]` | 恢复同科学/数值身份的提取/适应作业 |
| `report` | `--run DIR --out FILE` | 由真实产物生成中文Markdown；未执行写not_run |

公共退出码：0成功；2配置/未批准合同/权限错误；3选中资源缺失；4数据/缓存不完整；5数值/模型/训练失败；130中断。CLI输出机器可读code与未解决字段；不能用`|| true`吞掉错误。

`evaluate-checkpoint`不以训练是否完成、是否存在训练sidecar、参数来源或是否FINALIZED作为数值执行门槛。它必须记录参数、协议、worker和分数hash，保存逐条无标签分数、聚合指标、攻击类型分解、运行环境和资源开销，并禁止覆盖既有结果目录。参数按`model_state`、`state_dict`、原生state dict或DDP `module.`前缀格式解析，之后必须`strict=True`加载。该入口的结果可由用户标注用途，但不得反馈改变已封存训练split或recipe。

`STAGE`至少枚举`development / inventory / data_build / source_training / frozen_extract / adaptation / evaluation`。阶段数据权限分别检查：data_build能读取raw label；source_training只能fit/source_val；adaptation绝不能读取目标label；evaluation只有在封存之后。为解析/审计看到标签不构成适应器获得标签的权限。

### 14.2 本机命令

```bash
python -m eptta.cli --config configs/base.yaml --profile local_dev validate --level structure --stage development
python -m pytest tests/unit tests/contracts -q
python -m eptta.cli --config configs/base.yaml --profile local_dev plan \
  --experiment configs/experiments/source_pilot.yaml --task source_prepare \
  --out plans/source_prepare.preview.json --dry-run
```

没有CPU torch时第二条标待运行，不自动安装GPU依赖。preview允许资源占位，不能冒称远程资源已检查；不从preview直接执行确认性任务。

### 14.3 远程启动：先数据合同，再自有训练，最后提取

以下是需要实现的CLI用法。真实路径、审核记录及相应合同由R0–R2补全；示例不能被当作已存在文件。每段通过后才执行下一段，**不会在盘点后自动假定label已知或checkpoint已存在**。

```bash
: "${EPTTA_CORE_PY:?set core Python}"
: "${EPTTA_WORK_ROOT:?set writable remote work root}"
mkdir -p "$EPTTA_WORK_ROOT/inventory" "$EPTTA_WORK_ROOT/contracts" "$EPTTA_WORK_ROOT/plans"

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  validate --level resources --stage inventory

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  inspect-data --datasets asvspoof2019_la --out "$EPTTA_WORK_ROOT/inventory"

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  propose-data-contract --inventory "$EPTTA_WORK_ROOT/inventory/inventory.json" \
  --out "$EPTTA_WORK_ROOT/contracts/raw.proposed.json"
```

此后先批准raw/label/group/preprocess合同；`stage-data`生成未划分规范记录，`propose-splits`据实际计数提出source_val/cal/test等划分，审核后再锁定split。`build-manifests`可以直接引用staging，或用同一raw合同复用/重建其缓存。清单构建失败就处理错误，不向下运行。已锁定文件通过环境变量引用，以下不是作者预训练任务权重路径：

```bash
: "${EPTTA_RAW_CONTRACT:?set approved raw contract}"
: "${EPTTA_LABEL_POLICY:?set approved label policy}"
: "${EPTTA_GROUP_POLICY:?set approved group policy}"
: "${EPTTA_SPLIT_PLAN:?set approved split plan}"
: "${EPTTA_TRAIN_RECIPE:?set approved source training recipe}"
: "${EPTTA_TRAIN_RUN:?set new, non-existing full-training run directory}"

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  build-manifests --inventory "$EPTTA_WORK_ROOT/inventory/inventory.json" \
  --raw-contract "$EPTTA_RAW_CONTRACT" --label-policy "$EPTTA_LABEL_POLICY" \
  --group-policy "$EPTTA_GROUP_POLICY" --split-plan "$EPTTA_SPLIT_PLAN" \
  --out "$EPTTA_WORK_ROOT/manifests/source_initial"

# R2: 独立smoke。通过后报告，不自动把smoke当作正式训练。
"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  train-source --recipe "$EPTTA_TRAIN_RECIPE" --phase smoke \
  --run-output "${EPTTA_TRAIN_RUN}.smoke"
```

`resolve-training-recipe`、架构检查和训练preflight在正式训练前完成，`EPTTA_TRAIN_RECIPE`必须指向最终recipe lock，不能指草稿模板。只有已批准该训练阶段后，单独执行：

```bash
"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  train-source --recipe "$EPTTA_TRAIN_RECIPE" --phase full \
  --run-output "$EPTTA_TRAIN_RUN"

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  finalize-training --run "$EPTTA_TRAIN_RUN" \
  --out "$EPTTA_TRAIN_RUN/finalized.json"

"$EPTTA_CORE_PY" -m eptta.cli --config configs/base.yaml \
  --profile remote_a6000 --paths configs/paths.remote.yaml \
  export-frozen --training-manifest "$EPTTA_TRAIN_RUN/finalized.json" \
  --out "$EPTTA_TRAIN_RUN/frozen_export"
```

R4核验`frozen_export/bundle.json`后，通过`bind-artifacts`或`plan --bundle`绑定后续提取计划；再执行§14.4双卡提取和§14.5适应实验。每个模型分别有其training_run与bundle；不得拿AASIST训练manifest去填SSL-AASIST注册项。

核心进程调用训练/模型worker，实际GPU操作留在已选择的兼容环境。训练恢复使用`resume-source`，EP/缓存恢复使用`resume`；两种状态不能混用。

### 14.4 `deployment/launch_extract.sh` 的最小模板

实际脚本还需接入计划中的GPU租约、profile和完成索引；下面展示不共享梯度、保留错误退出的启动结构。GPU0/1来自已知双卡设定，实际调度可由环境覆写。

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${EPTTA_CORE_PY:?missing core Python}"
: "${EPTTA_BASELINE_PY:?missing baseline worker Python}"
: "${EPTTA_WORK_ROOT:?missing work root}"
plan="${1:?usage: launch_extract.sh PLAN PATHS}"
paths="${2:?usage: launch_extract.sh PLAN PATHS}"
mkdir -p "$EPTTA_WORK_ROOT/logs"
log_dir="$(mktemp -d "$EPTTA_WORK_ROOT/logs/extract.XXXXXX")"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap 'cleanup; exit 130' INT TERM
for slot in 0 1; do
  if [ "$slot" -eq 0 ]; then gpu="${EPTTA_GPU0:-0}"; else gpu="${EPTTA_GPU1:-1}"; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$EPTTA_CORE_PY" -m eptta.cli \
    --config configs/base.yaml --profile remote_a6000 --paths "$paths" \
    extract --plan "$plan" --worker-slot "$slot" --worker-count 2 \
    >"$log_dir/worker-$slot.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
printf 'logs=%s\n' "$log_dir"
exit "$status"
```

核心调度器必须转发SIGINT/SIGTERM并回收自己的bridge及其解码worker；否则只终止核心PID会留下占显存的孤儿进程。Ubuntu可让bridge在独立且由本作业拥有的进程组内启动，清理仅针对该组，正常退出也需wait回收。仅可终止本脚本创建的子进程；不使用`pkill python`或杀别人的GPU作业。索引合并必须在worker全部成功或确认有效恢复后进行。

### 14.5 源域实验与确认性实验的依赖

```text
finalized_source_training → frozen_bundle与parity
  → source_cache提取 → merge-cache(source_prepare)
  → build-artifacts(fit/cal0; U/M/Fisher/静态资源)
  → plan(source_pilot, task=select)
  → run-suite(phase=select, suite=smoke/mechanism/published)
  → seal-scores
  → evaluate(只读源select标签)
  → select-methods
  → plan(controlled/external, task=confirmatory)
  → freeze(完整已选计划+选择结果)
  → extract/merge-cache(按所需外部输入；不重估源产物)
  → run-suite(phase=confirmatory, frozen-spec=已封存文件)
  → seal-scores
  → evaluate(现在才读目标标签)
  → report
```

`build-artifacts`中固定源adapter的训练参数也要由fit与select明确安排，不能与confirmatory发生循环依赖。确认计划生成时可绑定尚未提取的输入清单身份，最终数值缓存需满足其固定视图/模型合同；资源可在之后物化，算法不能再变。

同一参考frozen/multiview分数可复用，但必须检查输入、probe、模型和precision一致。方法之间不能共用上一个方法留下的模型状态或优化器。

---

<a id="s15"></a>
## 15. 可直接复制给 AI Coding 工具的启动指令

### 15.1 本机首次执行：只完成L0–L2

```text
请读取 docs/DESIGN.md（EP-TTA v0.1.0），从尚未开始的项目建立实现。
本机只开发；数据、真实预处理、模型训练和实验在远程Ubuntu双卡A6000运行。
不下载真实音频/大模型，不自动SSH，不安装CUDA，不访问占位资源。

本轮只做L0–L2：
1. 创建项目、README/AGENTS/IMPLEMENTATION_STATUS。项目和自有schema版本均0.1.0。
   不写任何历史迁移、已有实现或已通过实验的假设；不覆盖目录中用户其他文件。
2. 实现config/schema/registry/CLI骨架和stage-aware校验。
   本机允许raw label格式、目录、preprocess、split和训练recipe的null占位；
   真实执行必须明确拒绝未审核合同，不能猜列或默认填标签。
3. 预留DatasetAdapter、LabelMapper、SplitPlanner、IncrementalReconciler接口，
   以及ModelFactory、SourceTrainer、SourceTrainJob、FrozenModelBundle接口。
   SSL-AASIST/AASIST鉴伪权重由本项目后续训练，不依赖作者现成任务checkpoint。
   通用SSL前端预训练初始化允许；与下游任务权重严格区分。
4. 实现EP数学核心：z+URU^Tz、Nd视图方差、单侧源间隔、投影SGD、
   每条reset和原视图score。串行作为参考，独立batch通过数值对照。
5. 写合成张量/协议和权限测试；可运行则记录真实命令/日志，缺依赖写NOT_RUN。
6. 提交实际改动、测试、未解决项与下一阶段；所有远程真实项目仍未运行。

完成本轮后停止并报告。不启动正式训练，不假称完整仓库或论文复现已完成。
```

### 15.2 本机后续实现：L3–L6

```text
按本项目实际状态继续，不跳过源模型训练链。
先实现远程格式插件/label mapping/分组/增量snapshot与源训练manifest；
对未知列格式仅提供候选和审核接口，不假造服务器事实。
实现AASIST、SSL-AASIST作者架构与训练worker、source_val选模、checkpoint恢复、
自有模型finalize/export，再接EP特征缓存、全部必要机制对照与published-port合同。
训练可以用DDP；EP的每条R不能DDP归约。所有方法共用固定自有底座。
新数据默认待分配，旧split不自动重洗；冲突数据隔离，依赖产物按hash刷新。
同时编写相应D/S/T/C测试、CLI和远程runbook；真实适配与训练保留待远程核验。
```

### 15.3 远程数据补全与训练

```text
当前在Ubuntu服务器，2×A6000，每卡48GB。读取docs/DESIGN.md、实际状态表和私有路径。
按R0–R4执行：资源盘点 → 原始label/预处理/来源组审核 → 训练前封存划分 →
架构与训练smoke → 已批准的正式源训练 → 仅source_val选模 → 自有模型冻结导出。

禁止用下载作者已训练鉴伪权重代替训练。SSL通用初始化可在核验后使用。
不把source_val/select/cal0/test合成一个dev；训练worker只能fit/source_val。
作者脚本若训练期间读eval，必须禁用并记录补丁；不依据目标EER选epoch/增强。
数据新增通过delta/reconcile/approval生成新snapshot，不能自动重划/改旧label/重训。
显存不足先审查microbatch/累积/精度/重计算；不悄悄冻结SSL前端或缩短输入。
所有阶段交付实际路径hash、运行命令、日志、失败与未执行内容。
```

### 15.4 远程适应实验：必须先有自有冻结底座

```text
仅在训练、选模与冻结parity通过后执行R5–R9。
先源缓存与U/M/tau0、P0/P1先导；随后五项已有算法的正确音频移植。
配置和方法在select（源域适应选择集）上确定，测试计划封存后评分，再由独立评价器读取标签。
目标label、生成器、隐藏干净版本不得进入EP；每条R独立重置。
保留负结果、有害翻转、coverage和真实总成本；缺数据或方法标未运行，不删行。
```

### 15.5 `AGENTS.md`的简要规则

只引用`docs/DESIGN.md`作为主合同；版本0.1.0；本机/远程分工；任务模型须自行训练；未知真实格式留接口而不猜；增量数据不改旧snapshot；源训练可DDP、EP不可跨条同步；目标标签禁止进适应；所有PASS需实际证据；不自动下载/SSH/长训练；Markdown报告不编造结果。AGENTS不重复复制另一套公式和默认值。

---

<a id="s16"></a>
## 16. v0.1.0首版初始化、风险与交付定义

### 16.1 版本与初始状态

项目、设计规范、Python包、配置schema、worker schema及项目自有policy初始版本统一`0.1.0`。文档标题/Git首个正式tag显示`v0.1.0`；方法ID不塞release号，主法为`ep_tta`。`pyproject.toml`的package version、`eptta.__version__`及CLI `--version`必须一致；测试检查不一致立即失败。

仓库初始不需要迁移器、旧版兼容层或虚构历史。`CHANGELOG.md`仅写`0.1.0 初始开发规划`，不能预填代码已交付。任务完成记录来自接下来的真实工作；第三方原仓库版本/论文年份/依赖锁保留实际值，不能为了统一项目版本篡改它们。

数据snapshot、模型run、缓存bundle用独立不可变ID/hash，不能把它们当项目release。数据更新不自动修改包版本；真实接口/算法后续变更应显式做语义版本管理，但当前不预填未来版本已存在。

### 16.2 实施风险与处理

| 风险 | 正确处理 |
|---|---|
| 没有任何任务模型权重 | 这是正常起点：实现并执行source trainer；不能换作者checkpoint |
| SSL通用初始化缺失 | 本机继续开发；远程选中SSL训练才阻塞；AASIST可独立推进 |
| 目录/label/预处理不明确 | 提出候选、审核后lock；不猜值或强迫搬成作者目录树 |
| 标签/同源组冲突 | 隔离、报告、新snapshot；不能静默修成有利结果 |
| 新数据影响已封存测试独立性 | 标记影响与访问历史，另做新实验方案 |
| 旧fairseq环境冲突 | 独立worker与兼容依赖；不升级一切或偷偷换架构 |
| 训练不收敛/SSL显存不足 | 先源smoke诊断，记录资源配方；受限变体独立命名 |
| EP信号弱或保持未激活 | 检查数学/静态对照，保留失败；不拿目标label调参 |
| 新checkpoint误用旧U/M | 按血缘校验拒绝混用，重建受影响资源 |
| 基线表现更好 | 收窄主张，保留表格与原始分数 |

### 16.3 资源安全

原始音频/协议/通用初始化只读；本项目checkpoint输出独立可写；许可证、源码commit和hash记录。旧格式序列化权重只在明确可信环境加载，不自动关闭安全校验。源特征不视为无隐私数据上传。

代码同步采用固定commit或校验源码包，不同步密钥/虚拟环境/大数据；不默认`rsync --delete`。不杀他人GPU任务，清理只针对本作业进程组。长训练、重训和最终测试都要求已选择、已批准的stage计划。

### 16.4 两种交付必须分开

**工程交付**：可执行代码、版本一致的配置与schema、数据未知项/增量接口、两类源训练插件、数学/协议测试、远程runbook和状态报告。

**实验交付**：远程核验的snapshot与split、真实源训练日志和自有checkpoint、冻结parity、数据缓存与方法结果、封存分数、评价与成本、明确未运行项和结论边界。

提供设计文档或通过合成测试不等于上述两项已完成。当前发布的是前者的执行规范，项目仍从`NOT_STARTED`起步。

---

<a id="s17"></a>
## 17. 依据、参考资料与版本核验

以下为模型结构、自行训练、方法身份和工程合同的一手入口，不证明EP-TTA有效。训练实施时固定完整仓库commit和patch；GitHub fetch得到的单文件blob SHA不是仓库commit。文献/源码核验不等于已在用户服务器完成训练或验证其数据格式。

**项目依据**

本文件是待开发项目的v0.1.0首版合同；算法、训练和数据接口在正文完整定义，不依赖其他设计文件。

**模型与方法**

- [R1] SSL_Anti-spoofing作者实现：<https://github.com/TakHemlata/SSL_Anti-spoofing>。重点`model.py`、`data_utils_SSL.py`；已核验model.py的Git blob为`0eb574ad892476090f7484e3863c384e255fade7`，这是文件blob而非仓库commit。第二模型：<https://github.com/clovaai/aasist>。
- [R2] Wang D, et al. *Tent: Fully Test-time Adaptation by Entropy Minimization*. ICLR 2021。<https://arxiv.org/abs/2006.10726>；作者代码：<https://github.com/DequanWang/tent>。
- [R3] Niu S, et al. *Towards Stable Test-Time Adaptation in Dynamic Wild World*. ICLR 2023。<https://arxiv.org/abs/2302.12400>；作者代码：<https://github.com/mr-eggplant/SAR>。
- [R4] Zhang M, Levine S, Finn C. *MEMO: Test Time Robustness via Adaptation and Augmentation*. NeurIPS 2022。<https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc28053a08f59fccb48b11f2e31e81c7-Abstract-Conference.html>；作者版本及code入口：<https://arxiv.org/abs/2110.09506>。
- [R5] Niu S, et al. *Efficient Test-Time Model Adaptation without Forgetting*. ICML 2022, PMLR 162:16888–16905。<https://proceedings.mlr.press/v162/niu22a.html>；作者代码：<https://github.com/mr-eggplant/EATA>。
- [R6] Nguyen-Le H H, et al. *Think Twice Before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation*. IJCAI 2025:7679–7687。<https://www.ijcai.org/proceedings/2025/854>；作者代码：<https://github.com/HongHanh2104/T2A-Think-Twice-Before-Adaptation>。

**数据与接口**

- [R7] ASVspoof 2019官方：<https://www.asvspoof.org/index2019.html>。
- [R8] ASVspoof 2021官方keys、LA/DF与跨年映射：<https://www.asvspoof.org/index2021.html>。
- [R9] WaveFake作者资源：<https://github.com/RUB-SysSec/WaveFake>；数据发行：<https://zenodo.org/records/5642694>。
- [R10] Xie等人的Codecfake：<https://github.com/xieyuankun/Codecfake>；论文：<https://arxiv.org/abs/2405.04880>。
- [R11] In-the-Wild作者入口：<https://deepfake-total.com/in_the_wild>。
- [R12] PyTorch自动微分、冻结/eval/no-grad/inference mode：<https://docs.pytorch.org/docs/stable/notes/autograd.html>。
- [R13] PyTorch DDP与CUDA语义：<https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>；<https://docs.pytorch.org/docs/stable/notes/cuda.html>。
- [R14] PyTorch DataLoader：<https://docs.pytorch.org/docs/stable/data.html>。文档当前版本不是本项目安装要求；以远程验证lock为准。
- [R15] ASVspoof官方评测代码：<https://github.com/asvspoof-challenge/2021>。
- [R16] Jung J, et al. *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks*. 原论文：<https://arxiv.org/abs/2110.01200>；作者代码：<https://github.com/clovaai/aasist>；训练配方入口：<https://github.com/clovaai/aasist/blob/main/config/AASIST.conf>。用于架构与源训练，不直接取现成任务权重。
- [R17] Tak H, et al. *Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation*. 原论文：<https://arxiv.org/abs/2202.12233>；作者训练入口：<https://github.com/TakHemlata/SSL_Anti-spoofing/blob/main/main_SSL_LA.py>；README：<https://github.com/TakHemlata/SSL_Anti-spoofing/blob/main/README.md>。区分通用SSL初始化与本项目鉴伪训练。
- [R18] PyTorch AMP与梯度累积官方说明：<https://docs.pytorch.org/docs/stable/notes/amp_examples.html>。实现使用服务器锁定版本对应API，不将在线文档版本当作已验证安装环境。

稳定分布外检测、持续适应等扩展文献不作为本版必须新增算法，避免首轮失去焦点。论文相关工作仍可另行补充，但不得把附属/预印本身份当作已经核验的主会发表。

---

<a id="appendix-a"></a>
## 附录 A. 主算法串行数学参考

**抽取路径**：`tests/reference/core_reference.py`。这是供首版实现使用的纯张量数学参考，用于作为生产实现的数值对照，不替代日志、模型、数据与访问权限模块。`CoreConfig.lambda_keep`对应解析配置的`defaults.regularizer_weight`，仅当主法使用margin时映射；不要由参考字段名推导给其他保持方式额外加入margin。

参考在float32/float64、同device的有限张量上使用；context检查一次后生产可缓存验证，不能为了吞吐移除资源身份核验。源锚点标签是允许的源监督，接口中没有目标标签。原分数本身非法应向上游报错，不返回虚假有效预测。

```python
# CORE_REFERENCE_BEGIN
import math
from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class CoreConfig:
    steps: int = 3
    lr: float = 0.01
    rho: float = 0.2
    gamma: float = 0.1
    lambda_keep: float = 1.0

@dataclass
class CoreOutput:
    R: Tensor
    score_before: float
    score_after: float
    status: str
    reason: Optional[str]
    trace: list[dict[str, float]]


def apply_adapter(Z: Tensor, U: Tensor, R: Tensor) -> Tensor:
    """Row features: [N,d]; mathematical R uses the column-vector convention."""
    return Z + ((Z @ U) @ R.T) @ U.T


def view_loss(Z_adapted: Tensor) -> Tensor:
    centered = Z_adapted - Z_adapted.mean(dim=0, keepdim=True)
    return centered.square().mean()  # denominator = number_of_views * d


def keep_loss(Za_adapted: Tensor, w: Tensor, b: float, ya: Tensor,
              m0: Tensor, tau0: float, gamma: float) -> Tensor:
    margin = (2 * ya.to(Za_adapted.dtype) - 1) * (Za_adapted @ w + b - tau0)
    deficit = torch.relu((1 - gamma) * m0 - margin)
    return deficit.square().mean()


@torch.no_grad()
def project_frobenius_(R: Tensor, rho: float) -> None:
    norm = torch.linalg.vector_norm(R)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("non-finite R norm")
    if float(norm) > rho:
        R.mul_(rho / norm)  # this branch cannot divide by zero for rho > 0


def run_core_episode(Z: Tensor, U: Tensor, w: Tensor, b: float,
                     Za: Tensor, ya: Tensor, m0: Tensor, tau0: float,
                     cfg: CoreConfig) -> CoreOutput:
    """Target Z contains no labels. All adaptation state is local to this call."""
    if Z.ndim != 2 or U.ndim != 2 or Za.ndim != 2 or w.ndim != 1:
        raise ValueError("invalid tensor ranks")
    n, d = Z.shape
    r = U.shape[1]
    m = Za.shape[0]
    if n < 2 or not 1 <= r <= d or m == 0:
        raise ValueError("invalid numbers of views, rank, or anchors")
    if U.shape[0] != d or Za.shape[1] != d or w.shape != (d,):
        raise ValueError("embedding/head dimensions disagree")
    if ya.shape != (m,) or m0.shape != (m,):
        raise ValueError("anchor arrays must be 1-D; refuse broadcasting")
    scalars = (b, tau0, cfg.lr, cfg.rho, cfg.gamma, cfg.lambda_keep)
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("configuration and head scalars must be finite")
    if not isinstance(cfg.steps, int):
        raise ValueError("steps must be an integer")
    if not (cfg.steps >= 0 and cfg.lr > 0 and 0 < cfg.rho < 1
            and 0 <= cfg.gamma < 1 and cfg.lambda_keep >= 0):
        raise ValueError("invalid adaptation configuration")
    for t in (Z, U, w, Za, m0):
        if t.dtype != Z.dtype or t.device != Z.device:
            raise ValueError("floating tensors must share dtype and device")
        if t.requires_grad or not bool(torch.isfinite(t).all()):
            raise ValueError("context must be detached and finite")
    if ya.device != Z.device or not bool(((ya == 0) | (ya == 1)).all()):
        raise ValueError("invalid source labels")
    eye = torch.eye(r, dtype=U.dtype, device=U.device)
    if not torch.allclose(U.T @ U, eye, atol=1e-5, rtol=1e-5):
        raise ValueError("U is not orthonormal")
    expected_m0 = (2 * ya.to(Z.dtype) - 1) * (Za @ w + b - tau0)
    if not bool((m0 > 0).all()) or not torch.allclose(
            m0, expected_m0, atol=1e-5, rtol=1e-5):
        raise ValueError("anchors are invalid or belong to a different head/threshold")

    R = nn.Parameter(torch.zeros((r, r), dtype=Z.dtype, device=Z.device))
    optimizer = torch.optim.SGD([R], lr=cfg.lr, momentum=0.0, weight_decay=0.0)
    before = float(Z[0] @ w + b)
    if not math.isfinite(before):
        raise ValueError("invalid frozen score")
    trace: list[dict[str, float]] = []
    status, reason = "ok", None

    with torch.enable_grad():
        for k in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            lv = view_loss(apply_adapter(Z, U, R))
            lk = keep_loss(apply_adapter(Za, U, R), w, b, ya, m0, tau0, cfg.gamma)
            total = lv + cfg.lambda_keep * lk
            if not bool(torch.isfinite(total)):
                status, reason = "fallback_numeric", "non-finite loss"
                break
            total.backward()
            if R.grad is None or not bool(torch.isfinite(R.grad).all()):
                status, reason = "fallback_numeric", "non-finite or missing R gradient"
                break
            grad_norm = float(torch.linalg.vector_norm(R.grad))
            if not math.isfinite(grad_norm):
                status, reason = "fallback_numeric", "non-finite gradient norm"
                break
            optimizer.step()
            if not bool(torch.isfinite(R).all()):
                status, reason = "fallback_numeric", "non-finite parameter"
                break
            try:
                project_frobenius_(R, cfg.rho)
            except FloatingPointError:
                status, reason = "fallback_numeric", "non-finite projection norm"
                break
            trace.append({"step": float(k + 1), "loss_view_before": float(lv.detach()),
                          "loss_keep_before": float(lk.detach()), "grad_norm": grad_norm,
                          "r_fro_after": float(torch.linalg.vector_norm(R.detach()))})

    with torch.no_grad():
        after = float((apply_adapter(Z[0:1], U, R) @ w + b)[0])
        if not math.isfinite(after):
            status, reason = "fallback_numeric", "non-finite final score"
        if status != "ok":
            R.zero_()
            after = before
    return CoreOutput(R.detach().clone(), before, after, status, reason, trace)
# CORE_REFERENCE_END
```

---

<a id="appendix-b"></a>
## 附录 B. 独立episode的等价批量参考

**抽取路径**：`tests/reference/batch_reference.py`。批量参考假定上游已通过与串行相同的context、shape与finite检查。生产代码还需逐条状态、原分数非法拒绝、数值异常隔离及同输入串行重放；不能只复制这段就声称容错链路完成。

每条loss独立，sum反传，R按样本单独投影；共享源锚点是固定上下文，不是共享目标统计。生产优化可使用§4.7的低维公式，但需与本参考逐样本和梯度对应。以下是显式张量参考而非最低显存实现。

```python
# BATCH_REFERENCE_BEGIN
import torch
from torch import Tensor


def run_batch_valid(Z: Tensor, U: Tensor, w: Tensor, b: float,
                    Za: Tensor, ya: Tensor, m0: Tensor, tau0: float,
                    cfg) -> tuple[Tensor, Tensor, Tensor]:
    """Valid finite inputs only; the caller must apply the Appendix A validator.

    Z: [B,N,d]. Each row has its own independent R. Plain projected SGD.
    This is a mathematical reference, not the full logging/failure executor.
    """
    if Z.ndim != 3 or Z.shape[0] < 1 or Z.shape[1] < 2:
        raise ValueError("expected Z=[B,N,d] with B>=1, N>=2")
    B, N, d = Z.shape
    r = U.shape[1]
    a = Z @ U
    ac = a - a.mean(dim=1, keepdim=True)
    zc = Z - Z.mean(dim=1, keepdim=True)
    pc = zc - ac @ U.T
    constant = pc.square().sum(dim=(1, 2)) / (N * d)
    aa = Za @ U
    c = U.T @ w
    sign = 2 * ya.to(Z.dtype) - 1
    before = Z[:, 0] @ w + b
    R = torch.zeros((B, r, r), dtype=Z.dtype,
                    device=Z.device, requires_grad=True)

    with torch.enable_grad():
        for _ in range(cfg.steps):
            adapted_a = a + torch.einsum("bnj,bij->bni", a, R)
            centered = adapted_a - adapted_a.mean(dim=1, keepdim=True)
            lv = constant + centered.square().sum(dim=(1, 2)) / (N * d)
            t = torch.einsum("bji,j->bi", R, c)  # R^T c, per episode
            delta_anchor = torch.einsum("mi,bi->bm", aa, t)
            margin = m0[None, :] + sign[None, :] * delta_anchor
            deficit = torch.relu((1 - cfg.gamma) * m0[None, :] - margin)
            lk = deficit.square().mean(dim=1)
            per_episode = lv + cfg.lambda_keep * lk
            if not bool(torch.isfinite(per_episode).all()):
                raise FloatingPointError("replay this batch serially to isolate failure")
            grad, = torch.autograd.grad(per_episode.sum(), R)
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("non-finite batch gradient: isolate serially")
            with torch.no_grad():
                R.add_(grad, alpha=-cfg.lr)
                norms = torch.linalg.vector_norm(R, dim=(-2, -1), keepdim=True)
                if not bool(torch.isfinite(norms).all()):
                    raise FloatingPointError("non-finite batch norm: isolate serially")
                scale = (cfg.rho / norms.clamp_min(torch.finfo(R.dtype).tiny)).clamp(max=1)
                R.mul_(scale)

    with torch.no_grad():
        t = torch.einsum("bji,j->bi", R, c)
        after = before + (a[:, 0] * t).sum(dim=-1)
        if not bool(torch.isfinite(after).all()):
            raise FloatingPointError("non-finite scores: isolate serially")
    return R.detach().clone(), before.detach(), after.detach()
# BATCH_REFERENCE_END
```

---

<a id="appendix-c"></a>
## 附录 C. 机制对照的关键数学原语

**抽取路径**：`tests/reference/baseline_primitives.py`，与附录A处于同一导入路径。本附录给出稳定二分类熵、源经验Fisher、保持替换和动态标量网格。它不包含Tent/SAR/MEMO/EATA/T²A的完整实现，也不替代§5与§6的身份审计。

`mean_binary_entropy`与`marginal_binary_entropy`支持最后一维为视图的张量；Fisher与scalar参考假定已通过核心context验证。未知regularizer应报错，不默默退化为none。保留项只做替换；source BCE使用原logit与y，而不是减tau0后冒称原模型概率。

```python
# BASELINE_PRIMITIVES_BEGIN
"""Finite-tensor reference primitives; not complete published algorithms."""
import math
from typing import NamedTuple
import torch
from torch import Tensor
import torch.nn.functional as F
from core_reference import apply_adapter, view_loss, keep_loss, CoreConfig


def _validate_scores(scores: Tensor) -> None:
    if scores.ndim < 1 or scores.shape[-1] < 1:
        raise ValueError("scores must have a nonempty final view dimension")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be floating and finite")


def mean_binary_entropy(scores: Tensor) -> Tensor:
    """Input [..., N] logit differences; output [...]. Natural logarithms."""
    _validate_scores(scores)
    lp, lq = F.logsigmoid(scores), F.logsigmoid(-scores)
    return -(lp.exp() * lp + lq.exp() * lq).mean(dim=-1)


def marginal_binary_entropy(scores: Tensor) -> Tensor:
    """H(mean(sigmoid(scores))), not mean(H) or H(sigmoid(mean(scores)))."""
    _validate_scores(scores)
    log_n = math.log(scores.shape[-1])
    lp = torch.logsumexp(F.logsigmoid(scores), dim=-1) - log_n
    lq = torch.logsumexp(F.logsigmoid(-scores), dim=-1) - log_n
    return -(lp.exp() * lp + lq.exp() * lq)


@torch.no_grad()
def empirical_fisher_R(Za: Tensor, ya: Tensor, U: Tensor,
                       w: Tensor, b: float) -> Tensor:
    """Source BCE per-example squared gradient at R=0; return [r,r]."""
    if Za.ndim != 2 or ya.shape != (Za.shape[0],) or Za.shape[0] == 0:
        raise ValueError("invalid source anchor shapes")
    if not bool(((ya == 0) | (ya == 1)).all()):
        raise ValueError("source labels must be canonical 0/1")
    q = Za @ U
    c = U.T @ w
    error = (Za @ w + b).sigmoid() - ya.to(Za.dtype)
    per_example_grad = error[:, None, None] * c[None, :, None] * q[:, None, :]
    result = per_example_grad.square().mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite source Fisher")
    return result


def preservation_penalty(kind: str, R: Tensor, Za: Tensor, U: Tensor,
                         w: Tensor, fisher: Tensor | None = None) -> Tensor:
    """Single R, valid context. Replacement penalties; no hidden margin term."""
    if kind == "none":
        return R.sum() * 0.0
    if kind == "l2":
        return R.square().mean()
    if kind == "logit":
        delta_scores = ((Za @ U) @ R.T) @ (U.T @ w)
        return delta_scores.square().mean()
    if kind == "fisher":
        if fisher is None or fisher.shape != R.shape:
            raise ValueError("Fisher shape must match R")
        return (fisher.detach() * R.square()).mean()
    raise ValueError(f"unsupported replacement: {kind}")


class ScalarResult(NamedTuple):
    a: float
    R: Tensor
    score: float
    objective: float
    evaluations: int


@torch.no_grad()
def scalar_grid_episode(Z: Tensor, U: Tensor, w: Tensor, b: float,
                        Za: Tensor, ya: Tensor, m0: Tensor, tau0: float,
                        cfg: CoreConfig, grid_size: int = 17) -> ScalarResult:
    """Assumes the same validated context as core_reference; source labels only.

    Exact floating-point ties select the first (smallest a) grid element.
    Production code must add the standard logging and numerical fallback.
    """
    if type(grid_size) is not int or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    r = U.shape[1]
    eye = torch.eye(r, dtype=Z.dtype, device=Z.device)
    grid = torch.linspace(0.0, cfg.rho / math.sqrt(r), grid_size,
                          dtype=Z.dtype, device=Z.device)
    matrices, losses = [], []
    for a in grid:
        R = -a * eye
        loss = view_loss(apply_adapter(Z, U, R))
        loss += cfg.lambda_keep * keep_loss(
            apply_adapter(Za, U, R), w, b, ya, m0, tau0, cfg.gamma)
        matrices.append(R)
        losses.append(loss)
    values = torch.stack(losses)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("non-finite scalar objective: use standard fallback")
    index = int(values.argmin())
    R = matrices[index]
    score = (apply_adapter(Z[:1], U, R) @ w + b)[0]
    if not bool(torch.isfinite(score)):
        raise FloatingPointError("non-finite scalar score: use standard fallback")
    return ScalarResult(float(grid[index]), R.clone(), float(score),
                        float(values[index]), grid_size)
# BASELINE_PRIMITIVES_END
```

---

<a id="appendix-d"></a>
## 附录 D. 待执行的参考测试与项目状态边界

### D.1 初始状态：不预填已完成验收

附录A–C提供可抽取的数学参考，下面给出pytest模板；这些代码片段不是已经建好的eptta项目。即使在文档检查环境执行了某些参考测试，也不自动为用户尚未开始的仓库创建PASS状态。需要Coding工具抽取、实现production-vs-reference对照，再在实际项目记录执行证据。

| 范围 | 项目初始状态 | 后续验收 |
|---|---|---|
| 项目骨架/配置/CLI | TODO | 本机实际构建和结构测试 |
| 数学/基线参考接入 | TODO | 抽取、单元测试、生产实现等价 |
| raw格式/label/预处理/来源组 | DEFERRED_REMOTE | 服务器抽样审核、lock和D系列测试 |
| 源模型训练与checkpoint | NOT_RUN | 自有训练日志、source_val选模、S系列测试 |
| 已发表算法音频移植 | TODO | 完整端口、权限/梯度/状态及真实运行 |
| 双卡显存/吞吐/恢复 | DEFERRED_REMOTE | 训练与推理分别测量 |
| 鉴伪效果与科学假设 | NOT_RUN | 封存后真实评分及统计分析 |

README和IMPLEMENTATION_STATUS不能把计划存在写成工程完成；所有输出为null/not_run而不是预期数值。

### D.2 可直接抽取的pytest文件

**抽取路径**：`tests/reference/test_references.py`。保留原reference函数，在生产代码出现时增加production-vs-reference测试，不仅测试reference自身。负面fixture应有确定触发条件，不用最终目标数据挑选“通过测试”的输入。

```python
# REFERENCE_TESTS_BEGIN
import math
from dataclasses import replace
import pytest
import torch
import torch.nn.functional as F
from core_reference import (CoreConfig, apply_adapter, view_loss, keep_loss,
                            run_core_episode, project_frobenius_)
from batch_reference import run_batch_valid
from baseline_primitives import (mean_binary_entropy, marginal_binary_entropy,
    empirical_fisher_R, preservation_penalty, scalar_grid_episode)

torch.set_num_threads(1)


def fixture(B=7, d=11, r=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(415)
    U = torch.linalg.qr(torch.randn(d, r, dtype=dtype, generator=g))[0]
    w = torch.randn(d, dtype=dtype, generator=g)
    Za = torch.randn(16, d, dtype=dtype, generator=g)
    ya = torch.arange(16) % 2
    b, tau = 0.13, -0.17
    signs = 2 * ya.to(dtype) - 1
    desired = tau + signs * torch.linspace(0.05, 1.6, 16, dtype=dtype)
    Za += ((desired - Za @ w - b) / w.square().sum())[:, None] * w
    m0 = signs * (Za @ w + b - tau)
    Z = torch.randn(B, 3, d, dtype=dtype, generator=g)
    return Z, U, w, b, Za, ya, m0, tau


@pytest.mark.parametrize('dtype',[torch.float32, torch.float64])
@pytest.mark.parametrize('B',[1,2,7,128])
@pytest.mark.parametrize('steps',[0,1,5])
def test_batch_serial(dtype,B,steps):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=B,dtype=dtype)
    cfg=CoreConfig(steps=steps,lr=0.15,lambda_keep=10)
    R,before,after=run_batch_valid(Z,U,w,b,Za,ya,m0,tau,cfg)
    serial=[run_core_episode(z,U,w,b,Za,ya,m0,tau,cfg) for z in Z]
    tol=3e-5 if dtype==torch.float32 else 1e-9
    torch.testing.assert_close(R,torch.stack([o.R for o in serial]),atol=tol,rtol=tol)
    torch.testing.assert_close(after,torch.tensor([o.score_after for o in serial],dtype=dtype),atol=tol,rtol=tol)


@pytest.mark.parametrize('d,r',[(7,2),(160,8)])
def test_first_gradient_and_structure(d,r):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1,d=d,r=r)
    z=Z[0]; R=torch.zeros(r,r,dtype=z.dtype,requires_grad=True)
    grad,=torch.autograd.grad(view_loss(apply_adapter(z,U,R)),R)
    ac=z@U-(z@U).mean(0)
    torch.testing.assert_close(grad,2*ac.T@ac/(z.shape[0]*d))
    R=(torch.randn_like(R)*0.03).detach()
    adapted=apply_adapter(z,U,R)
    torch.testing.assert_close(adapted,(z.T+U@R@U.T@z.T).T)
    perp=torch.eye(d,dtype=z.dtype)-U@U.T
    torch.testing.assert_close(adapted@perp,z@perp)
    delta=((adapted-z)@w).abs()
    bound=torch.linalg.vector_norm(R)*torch.linalg.vector_norm(U.T@w)*torch.linalg.vector_norm(z@U,dim=-1)
    assert bool((delta<=bound+1e-10).all())


def test_initial_keep_and_k1():
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    R=torch.zeros(U.shape[1],U.shape[1],dtype=Z.dtype,requires_grad=True)
    loss=keep_loss(apply_adapter(Za,U,R),w,b,ya,m0,tau,0.1)
    grad,=torch.autograd.grad(loss,R)
    assert float(loss.detach())==0 and float(grad.abs().sum())==0
    a=run_core_episode(Z[0],U,w,b,Za,ya,m0,tau,CoreConfig(steps=1,lambda_keep=0))
    c=run_core_episode(Z[0],U,w,b,Za,ya,m0,tau,CoreConfig(steps=1,lambda_keep=999))
    torch.testing.assert_close(a.R,c.R,atol=0,rtol=0)


def test_keep_active_multistep():
    dtype=torch.float64
    U=torch.eye(2,dtype=dtype); w=torch.tensor([1.,0.],dtype=dtype)
    Za=torch.tensor([[-1.,0.],[1.,0.]],dtype=dtype); ya=torch.tensor([0,1]); m0=torch.ones(2,dtype=dtype)
    Z=torch.tensor([[[1.,0.],[2.,0.],[3.,0.]]],dtype=dtype)
    a=run_batch_valid(Z,U,w,0.,Za,ya,m0,0.,CoreConfig(steps=3,lr=.1,gamma=0,lambda_keep=0))[0]
    b=run_batch_valid(Z,U,w,0.,Za,ya,m0,0.,CoreConfig(steps=3,lr=.1,gamma=0,lambda_keep=10))[0]
    assert not torch.allclose(a,b)


@pytest.mark.parametrize('norm',[0.0,0.1,9.0])
def test_projection(norm):
    R=torch.eye(3,dtype=torch.float64)*norm
    project_frobenius_(R,.2)
    assert torch.linalg.vector_norm(R)<=.2+1e-12


def test_order_chunk_and_rotation():
    Z,U,w,b,Za,ya,m0,tau=fixture(B=7)
    cfg=CoreConfig(lr=.1,steps=5)
    full=run_batch_valid(Z,U,w,b,Za,ya,m0,tau,cfg)[2]
    order=torch.tensor([3,1,6,2,4,0,5])
    shuffled=run_batch_valid(Z[order],U,w,b,Za,ya,m0,tau,cfg)[2]
    torch.testing.assert_close(shuffled,full[order])
    tail=torch.cat([run_batch_valid(z,U,w,b,Za,ya,m0,tau,cfg)[2] for z in Z.split(3)])
    torch.testing.assert_close(tail,full)
    Q=torch.linalg.qr(torch.randn(U.shape[1],U.shape[1],dtype=Z.dtype))[0]
    rotated=run_batch_valid(Z,U@Q,w,b,Za,ya,m0,tau,cfg)[2]
    torch.testing.assert_close(rotated,full,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize('kind',['none','l2','logit','fisher'])
def test_preservation_zero(kind):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    R=torch.zeros(3,3,dtype=Z.dtype,requires_grad=True)
    fi=empirical_fisher_R(Za,ya,U,w,b)
    loss=preservation_penalty(kind,R,Za,U,w,fi)
    grad,=torch.autograd.grad(loss,R)
    assert float(loss.detach())==0
    torch.testing.assert_close(grad,torch.zeros_like(grad),atol=0,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fisher_per_sample(dtype):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1,dtype=dtype)
    grads=[]
    for i in range(len(ya)):
        R=torch.zeros(3,3,dtype=dtype,requires_grad=True)
        score=(apply_adapter(Za[i:i+1],U,R)@w+b)[0]
        loss=F.binary_cross_entropy_with_logits(score,ya[i].to(dtype))
        g,=torch.autograd.grad(loss,R); grads.append(g)
    manual=torch.stack(grads).square().mean(0)
    computed=empirical_fisher_R(Za,ya,U,w,b)
    tol=1e-6 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(computed,manual,atol=tol,rtol=tol)
    assert bool((computed>=0).all())
    assert not torch.allclose(computed,torch.stack(grads).mean(0).square())


@pytest.mark.parametrize('values',[[-10.,10.],[.4,2.,-1.2],[10000.,-10000.],[10000.,10000.]])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_entropy_finite(values,dtype):
    x=torch.tensor(values,dtype=dtype,requires_grad=True)
    e=mean_binary_entropy(x); m=marginal_binary_entropy(x)
    assert bool(torch.isfinite(e)) and bool(torch.isfinite(m))
    g,=torch.autograd.grad(e+m,x)
    assert bool(torch.isfinite(g).all())
    assert float(m.detach())>=float(e.detach())-1e-6


def test_entropy_definitions_and_gradient():
    x=torch.tensor([[-9.,9.],[.4,2.]],dtype=torch.float64,requires_grad=True)
    assert float(mean_binary_entropy(x)[0].detach()) < .01
    assert float(marginal_binary_entropy(x)[0].detach()) > .69
    p=x.sigmoid().mean(-1)
    direct=-(p*p.log()+(1-p)*(1-p).log())
    stable=marginal_binary_entropy(x)
    torch.testing.assert_close(stable,direct)
    gs,=torch.autograd.grad(stable.sum(),x,retain_graph=True)
    gd,=torch.autograd.grad(direct.sum(),x)
    torch.testing.assert_close(gs,gd)
    wrong=mean_binary_entropy(x.mean(-1,keepdim=True))
    assert not torch.allclose(stable,wrong)


@pytest.mark.parametrize('grid_size',[2,17,33])
def test_scalar(grid_size):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    cfg=CoreConfig()
    out=scalar_grid_episode(Z[0],U,w,b,Za,ya,m0,tau,cfg,grid_size)
    assert 0<=out.a<=cfg.rho/math.sqrt(U.shape[1])+1e-12
    assert torch.linalg.vector_norm(out.R)<=cfg.rho+1e-12
    assert out.evaluations==grid_size
    assert out.objective<=float(view_loss(Z[0]))+1e-12


def test_scalar_tie_and_invalid_grid():
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    Z.zero_()  # exact arithmetic ties, not merely near-equal floating values
    cfg=CoreConfig(lambda_keep=0)
    out=scalar_grid_episode(Z[0],U,w,b,Za,ya,m0,tau,cfg)
    assert out.a==0
    with pytest.raises(ValueError):
        scalar_grid_episode(Z[0],U,w,b,Za,ya,m0,tau,cfg,1)


@pytest.mark.parametrize('bad',['nan','shape','orthogonal','margin','config'])
def test_core_rejects_invalid(bad):
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    cfg=CoreConfig()
    if bad=='nan': Z[0,0,0]=float('nan')
    elif bad=='shape': ya=ya[:,None]
    elif bad=='orthogonal': U=U*2
    elif bad=='margin': m0=m0+1
    elif bad=='config': cfg=replace(cfg,rho=2)
    with pytest.raises(ValueError): run_core_episode(Z[0],U,w,b,Za,ya,m0,tau,cfg)


def test_numeric_fallback_and_reset():
    Z,U,w,b,Za,ya,m0,tau=fixture(B=1)
    z=Z[0].float()*1e20
    out=run_core_episode(z,U.float(),w.float(),b,Za.float(),ya,m0.float(),tau,CoreConfig())
    assert out.status=='fallback_numeric'
    assert out.score_after==out.score_before
    assert float(out.R.abs().sum())==0
    x=run_core_episode(Z[0],U,w,b,Za,ya,m0,tau,CoreConfig())
    y=run_core_episode(Z[0],U,w,b,Za,ya,m0,tau,CoreConfig())
    assert x.score_after==y.score_after


def test_label_flip_identity():
    y=torch.tensor([0,0,0,0,1,1,1,1])
    p0=torch.tensor([0,0,1,1,0,0,1,1])
    pm=torch.tensor([0,1,0,1,0,1,0,1])
    for cls in [0,1]:
        mask=y==cls
        H=((p0[mask]==cls)&(pm[mask]!=cls)).double().mean()
        G=((p0[mask]!=cls)&(pm[mask]==cls)).double().mean()
        delta=pm[mask].double().mean()-p0[mask].double().mean()
        torch.testing.assert_close(delta,H-G if cls==0 else G-H)
# REFERENCE_TESTS_END
```

从文档抽取四段代码后，可在本机已有CPU PyTorch/pytest环境执行：

```bash
PYTHONPATH=tests/reference python -m pytest tests/reference/test_references.py -q
```

bridge语法兼容、真实数据适配、增量snapshot、源训练、配置防护、缓存恢复和指标边界都要另写生产测试。参考测试不覆盖这些任务，不能因一组数学检查通过就批量标PASS。

---

**执行结点**：先复制§15.1指令完成L0–L2；随后依实际状态推进。没有目标结果也能完成代码；没有真实执行证据不能宣布实验完成。


### 首次开发的最终检查单

开始实现前核对：标题、包和schema初始版本均0.1.0；没有已完成训练或已有任务checkpoint假设；raw/label/preprocess/split未知字段可明确占位；source training与target adaptation权限分开；incremental ingestion不改变EP重置式定义；所有正式数值结果保持未运行。

完成每个阶段时再生成实际TEST_REPORT。真实服务器尚未核验的内容始终保留状态和责任接口，不由Coding工具用猜测补成“完成”。

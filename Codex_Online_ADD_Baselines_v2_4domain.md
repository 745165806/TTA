# Codex 执行任务：Online ADD Baselines v2（四域固定协议）

## 任务目标

本阶段只做一件事：**固定 Online Audio Deepfake Detection 的实验范式，并在统一协议下完成基础 baseline 的 GPU 实验。**

不要设计新方法，不重新训练 source detector，不增加复杂工程框架，不以“baseline 必须超过 Frozen”为完成条件。

项目：`/media/dell/data/fakeAudioDection/TTA`
环境：`tta`
建议独立分支 / worktree：`exp-online-add-baselines`

最终交付：
- 4 个 stationary target domains；
- 2 个 dynamic multi-domain streams；
- 6 个 baseline；
- 5 个固定顺序种子；
- 共 `6 × 6 × 5 = 180` 次独立流运行；
- 统一逐样本首次在线分数；
- `BASELINE_REPORT.md`；
- 可复现命令、配置与小型结果；
- target90 和最终 holdout 均不运行。

如果个别 baseline 因模型结构或必要依赖确实无法保真移植，标记 `BLOCKED`，完成其余方法，不得删掉关键机制后仍沿用原方法名。

---

# 1. 固定实验范式：predict-then-adapt online TTA

这里的“causal”只表示**时间上不看未来样本**，与因果推断无关。本文档统一使用：

> **predict-then-adapt online protocol**

对第 `t` 个 batch：

```text
batch t 到达
    ↓
用当前状态 θ_t 产生并保存 batch t 的预测
    ↓
不读取标签
    ↓
使用 batch t 的无标签信息更新 TTA 状态
    ↓
得到 θ_{t+1}
    ↓
处理 batch t+1
```

固定规则：

- 所有方法从同一个原始 project-trained Frozen SSL-AASIST checkpoint 开始；不是历史 adapter / ERM checkpoint。
- 优先复用 `exp-representation-tta` 已验证的较早层 LL `[201,128]` 三视图缓存；冻结 XLS-R/LL，LL 后原 AASIST backend 在 GPU 上前向/反向。
- `batch_size = 16`。
- 每条 stream 只遍历一次。
- 参数型方法：**先保存当前 batch 的预测，再用当前 batch 更新；更新后的参数只能影响未来 batch。**
- 不允许用最终模型重新评分过去样本。
- 不允许提前读取未来 batch、完整目标域统计、完整目标域原型或未来伪标签。
- 流内跨域不重置；每条新的独立 stream 从同一个 source 状态完整重置。
- 所有 optimizer / BN / teacher / memory / Fisher / running statistics 都属于方法状态，stream 开始时一起重置。
- 允许当前已经到达的 B16 样本共同参与 BN 统计或当前 batch 图推断；不允许查看后续 batch。
- target label 永远不进入 TTA 方法，只能由独立 evaluator 在逐样本分数全部保存后关联。
- 主赛道 test time 不读取 raw source audio；允许模型权重及预先固定的小型 source summaries（例如 Fisher、BN statistics）。逐方法如实披露。

## LAME 例外

LAME 不更新参数，而是在当前 batch 内直接利用样本关系修正当前 batch 输出。因此：

- LAME 可以利用当前 B16 的 feature/probability graph；
- 不能看未来 batch；
- 不能一次对整个目标域构图；
- 保存的是 LAME 修正后的当前 batch 输出。

不要为了统一接口把 LAME 改成 Frozen 预测。

---

# 2. 固定四个 target development domains

本轮只使用以下四个 development domains：

## D1 — In-the-Wild

- 使用现有固定 `ITW target10`；
- `n = 3178`；
- 不重新划分；
- 继续使用 selected-only 资源。

## D2 — WaveFake

- 使用现有固定 development selection；
- `n = 4096`；
- 2,048 content pairs；
- 不重新划分；
- content-pair 信息仅供 evaluator 分组统计，不进入 TTA。

## D3 — ASVspoof 2021 LA development benchmark subset

一次性建立固定 subset：

```text
4096 total
2048 bonafide
2048 spoof
```

要求：

- 只在 benchmark builder 中读取官方 label；
- 使用固定 seed `2026`；
- 在每个类别内部按稳定 ID 列表 + `random.Random(2026)` 抽样；
- 一次生成 manifest 后永久固定，不因方法结果重抽；
- TTA runner 只能看到 `sample_id + waveform/feature path`，不能看到 label；
- label 单独写到 evaluator-only artifact；
- 如果任一类别少于 2048，停止并报告 `LA_SUBSET_CONSTRUCTION_BLOCKED`，不得静默缩小或换数据。

该资源属于 development benchmark，不声称 untouched final holdout。

## D4 — ASVspoof 2021 DF development benchmark subset

同样固定：

```text
4096 total
2048 bonafide
2048 spoof
```

规则与 LA 完全相同：固定 seed、一次性 manifest、label 与 runner 分离、不可根据实验结果重抽。

该资源也属于 development benchmark，不声称 untouched final holdout。

## 为什么本轮只锁四域

四个域承担不同 shift：

```text
ITW      : real-world uncontrolled shift
WaveFake : clean synthetic / generator shift
LA21     : telephony / channel / codec-related shift
DF21     : deepfake + compression / algorithm shift
```

暂不增加 Codecfake、新下载数据、语言扩展数据。Codecfake 仅保留历史辅助证据，不进入 v2 主 benchmark。

---

# 3. 固定六种 stream

所有 stream 都只由固定 manifest 生成。

每个 order seed：

```text
2026
2027
2028
2029
2030
```

只在每个 domain / segment 内打乱样本顺序，不使用 label、score、attack type 或文件名类别语义排序。

## Stationary streams

```text
S1 = ITW
S2 = WaveFake
S3 = ASV2021-LA
S4 = ASV2021-DF
```

分别回答：单一目标域持续到来时，各 TTA baseline 是否有稳定价值。

## Dynamic streams

### S5

```text
ITW → LA21 → WaveFake → DF21
```

### S6

```text
DF21 → WaveFake → LA21 → ITW
```

规则：

- 每个 segment 内按当前 order seed 独立确定顺序；
- 算法不知道 domain 名和切换点；
- 域边界不重置模型；
- 先完整拼接 stream，再按 B16 切 batch；
- 边界处允许出现跨域混合尾 batch，不人为清空；
- 每条音频在同一 stream 中只出现一次。

v2 暂不增加 gradual shift、burst stream、recurring same-sample replay 或 class-prior grid。

## 总实验量

```text
6 methods
× 6 streams
× 5 orders
= 180 independent stream runs
```

五个 order 是顺序敏感性重复，不是五个模型训练 seed，也不是五个独立 target domains。

---

# 4. 六个 baseline

只复现最必要、机制互补的 6 个 baseline。

## 4.1 Frozen

- 原 source checkpoint；
- 原 BN / weights；
- 不适配；
- 作为所有流的统一参考。

统一 score：优先使用 native 两类 logits 的 `spoof - bonafide`，先做列语义核对。

## 4.2 BN-only

目的：区分“当批 normalization statistics”与“真正梯度适配”。

- 使用与 TENT 相同的 test-time BN 行为；
- 不更新 affine 参数；
- 无 optimizer。

参考：TENT 作者实现 `norm.py` / `tent.py`。

## 4.3 TENT-Audio

保留：

- entropy minimization；
- BN affine update；
- 当前 batch statistics；
- 状态在 stream 内持续。

参考：
- Paper: *Tent: Fully Test-Time Adaptation by Entropy Minimization*, ICLR 2021
- https://arxiv.org/abs/2006.10726
- https://github.com/DequanWang/tent

不要把历史“固定 source BN statistics”的自定义 TENT 直接当作作者 TENT。

## 4.4 EATA-Audio

保留核心：

- reliable sample filtering；
- redundant sample filtering；
- Fisher-based anti-forgetting regularization。

参考：
- *Efficient Test-Time Model Adaptation without Forgetting*, ICML 2022
- https://proceedings.mlr.press/v162/niu22a.html
- https://github.com/mr-eggplant/EATA

二分类 audio-port：

- entropy threshold 使用 `0.4 * log(2)`；
- diversity / cosine threshold 只允许 source-select 标定一次；
- Fisher 从 source fit 固定最多 2000 条构造；
- 目标域不能重新调 threshold。

如果删除 Fisher，则必须标 `ETA-Audio`，不能仍叫 EATA。

## 4.5 RoTTA-Audio

保留核心：

- robust BN；
- memory；
- uncertainty / timeliness / class-balance selection；
- teacher-student；
- temporal weighting。

参考：
- *Robust Test-Time Adaptation in Dynamic Scenarios*, CVPR 2023
- https://arxiv.org/abs/2303.13899
- https://github.com/BIT-DA/RoTTA

固定 target memory 上限：

```text
256 audio records
```

如果作者默认更小则沿用作者值；不得超过 256。

已有三视图 audio perturbations 可作为音频增强映射；不能直接把视觉 augmentation 套到 LL feature。

## 4.6 LAME-Audio

- Frozen network；
- 当前 B16 feature + probability graph；
- parameter-free output refinement；
- 无 optimizer；
- 无跨 batch memory。

参考：
- *Parameter-free Online Test-time Adaptation*, CVPR 2022
- https://arxiv.org/abs/2201.05718
- https://github.com/fiveai/LAME

v2 固定：

```text
kNN k = min(5, batch_size - 1)
```

其他求解保持作者默认。

输出统一转成 spoof-oriented continuous score，不做 hard label 后再算 AUC。

## 暂不加入

```text
CoTTA
SAR
AdaContrast
新方法
```

只有 v2 baseline 表完整后，再决定是否需要 v3 扩 baseline。

---

# 5. 模型适配范围与 source-only 参数选择

## 参数范围

TENT / EATA：

- 只更新 LL 后原 AASIST backend 中实际存在的 BN1d/BN2d affine；
- XLS-R / LL 冻结；
- 非 BN 参数冻结。

RoTTA：

- 按作者机制替换 / 包装对应 BN；
- 不改变 source classifier task definition。

LAME / Frozen / BN-only：

- 不训练新参数。

如果当前模型结构缺少某个算法不可替代的必要机制，标 `INCOMPATIBLE`，不得悄悄换成另一个方法。

## 超参数选择

不做大网格。

梯度算法只允许学习率三选一：

```text
lr_author / 3
lr_author
lr_author * 3
```

选择只使用 source-select 的固定 clean/noise/FIR pseudo-target stream。

选择标准：

```text
mean EER 最低
→ AUC 打破平局
→ 完全并列则作者默认
```

EATA 的二分类 threshold 也只在 source-select 标定一次。

RoTTA 的 teacher / memory / robust-BN 核心常数优先作者默认，不做 target tuning。

所有最终配置必须在四个 target domain 正式打分前保存到 `config.json`。

---

# 6. ASV2021 LA / DF subset 与 LL cache 准备

只新增必要数据准备，不建设复杂数据框架。

建议实现：

```text
experiments/online_add_v2/build_asv21_subsets.py
```

输出：

```text
manifests/asv2021_la_online_dev.jsonl
manifests/asv2021_df_online_dev.jsonl
private_eval_labels/asv2021_la_online_dev_labels.jsonl
private_eval_labels/asv2021_df_online_dev_labels.jsonl
```

runner manifest 不含 label；label sidecar 只允许 `summarize.py` 使用。

如果 LA / DF 的 LL `[201,128]` 三视图缓存尚不存在：

- 复用 `representation_tta` 的 GPU LL extraction 路径；
- 只提取上述固定 4096 IDs；
- 四 GPU 分片；
- 保存 selected-only cache；
- 不为完整 LA/DF 全量数据建立新缓存。

最低检查：

```text
4096 unique IDs
2048/2048 class composition（仅 builder/evaluator 知道）
3 views / sample
all finite
```

---

# 7. GPU 调度

四卡并行单位是：

```text
method × stream × order
```

一条完整 stream 不能拆给多 GPU，因为会破坏 online state。

推荐：

```text
GPU0: TENT
GPU1: EATA
GPU2: RoTTA
GPU3: Frozen / BN-only / LAME
```

某卡完成后继续领取任务队列中未运行的 stream。

LA / DF LL cache 提取优先在 baseline 正式运行前四卡并行完成。

不使用分布式训练平台、数据库、Web dashboard。

---

# 8. 最小代码结构

建议：

```text
experiments/online_add_v2/
    PROTOCOL.md
    config.json
    build_asv21_subsets.py
    make_streams.py
    methods.py
    run.py
    summarize.py
    launch.sh
```

可以复用已有作者核心文件，不要求一定全部塞进 `methods.py`。

CLI 最少支持：

```bash
python experiments/online_add_v2/build_asv21_subsets.py --config ...
python experiments/online_add_v2/make_streams.py --config ...
python experiments/online_add_v2/run.py --method tent --stream S1 --order-seed 2026 --gpu 0 --run-id <id>
bash experiments/online_add_v2/launch.sh --gpus 0,1,2,3
python experiments/online_add_v2/summarize.py --root <result_root>
```

不要重构 `src/`，不要增加审批、proposal、数据库、hash framework 或大量 toy tests。

---

# 9. 只做三项必要检查

## Check 1 — Frozen parity

固定 32 条真实样本：

- native model logits；
- LL cache replay logits；
- spoof-oriented score；

float32 最大绝对误差要求 `<= 1e-5`。

## Check 2 — Future-blind prefix

相同 stream prefix、两个不同 future suffix：

- prefix 已产生的分数必须逐值相同；
- 证明 runner 没有提前使用未来样本或完整目标统计。

## Check 3 — 每方法 64 条 CUDA smoke

验证：

- finite score；
- trainable parameter scope 正确；
- update count 合理；
- memory <= 256；
- predict-then-adapt 时序正确；
- stream reset 可重放。

再执行一次：

```bash
python -m compileall -q experiments/online_add_v2
```

不要建设更复杂测试体系。

---

# 10. 固定评价指标

## Stationary S1–S4

每条 stream 报告：

```text
AUC
EER (%)
ΔAUC vs Frozen
EER improvement vs Frozen (pp)
FPR @ source threshold
FNR @ source threshold
Balanced Accuracy @ source threshold
```

## Dynamic S5–S6

必须逐 segment 报告：

```text
ITW segment
LA segment
WaveFake segment
DF segment
```

同时报告：

```text
segment macro AUC/EER
每次切换后前256条表现
```

全流 pooled AUC/EER 只作附加值，不用它单独宣称跨域提升。

## Rolling stability

只在同一 domain segment 内：

```text
window = 256
stride = 128
```

报告：

```text
mean rolling EER/AUC
worst valid window
valid window count
```

单类别窗口写 `NA`，不填 0，不为了双类别重排 stream。

## Order sensitivity

五个 order 分别保存，再汇总：

```text
mean
std (ddof=1)
min/max
```

不能把五个 order 当作五个独立 target datasets。

## 资源指标

至少记录：

```text
updates / 100 samples
adapted-sample fraction
peak GPU memory
method state / memory size
backend CUDA time
ms / audio（明确不含冻结前端和缓存I/O）
```

---

# 11. 统计与 bootstrap：最小化

主比较首先依赖：

```text
5 orders 的方向一致性
+ effect size
```

只对 stationary S1–S4 的最终逐样本在线分数做 1000 次 paired bootstrap：

- WaveFake：按 content-pair；
- ITW：按 audio ID；
- LA/DF：按 sample ID，除非已有可靠 group 可直接复用。

bootstrap 不重跑 online adaptation，因此明确标为：

> conditional uncertainty of the realized online trajectories

动态 S5/S6 第一版不额外做复杂 bootstrap；优先保证 180 个正式流全部完成。

---

# 12. 实际执行顺序

## Phase A — Protocol lock

1. 检查 Git / GPU / LL cache。
2. 写 `PROTOCOL.md` / `config.json`。
3. 固定 LA/DF subset manifest。
4. 生成全部 S1–S6 × orders2026–2030 stream manifest。
5. 从此不改变样本构成和协议。

## Phase B — Data cache

若 LA/DF 没有 LL cache：

- 四卡提取固定 subset；
- 只做一次。

## Phase C — Baseline implementation

优先顺序：

```text
Frozen
BN-only
TENT
LAME
EATA
RoTTA
```

完成一个 baseline 的 64-sample smoke 后直接进入正式实验，不做额外工程扩展。

## Phase D — order2026 全覆盖

先完成全部 6 methods × 6 streams 的 order2026。

检查：

- 没有 future leak；
- 没有 label leak；
- 没有 NaN；
- 方法行为与作者机制一致。

只要工程正确，不论结果好坏，都继续。

## Phase E — orders2027–2030

补齐剩余 4 个顺序。

目标：

```text
180 / 180 completed
```

如某方法正式 `BLOCKED`，报告其缺失数量，其他方法继续完成。

## Phase F — 汇总

生成：

```text
experiments/online_add_v2/BASELINE_REPORT.md
```

---

# 13. BASELINE_REPORT.md 必须第一屏回答

```text
Stage: Online ADD Baselines v2

Completed runs:
xxx / 180

Target domains:
ITW = 3178
WaveFake = 4096
LA21 = 4096
DF21 = 4096

Methods:
Frozen = COMPLETE/BLOCKED
BN-only = ...
TENT = ...
EATA = ...
RoTTA = ...
LAME = ...

Stationary mean over 5 orders:
Method | ITW EER/AUC | WaveFake EER/AUC | LA21 EER/AUC | DF21 EER/AUC

Dynamic:
Method | S5 segment macro | S6 segment macro | worst-window | switch-first256

Efficiency:
Method | updates/100 | ms/audio backend | peak GPU MB | memory size

Main observations:
1. ...
2. ...
3. ...

Which existing mechanisms appear useful for audio ADD online TTA?
...

Which mechanisms fail or drift?
...

Can this benchmark support designing a new method?
YES / PARTIAL / NO

Target90 accessed = NO
Final holdout accessed = NO
```

最后只给出 **一个** 后续研究建议，不列十个方向。

---

# 14. Git 与工件

- 使用独立分支 / worktree；
- 不覆盖历史结果；
- 每个 run 保存 config、command、log、scores、metrics / failure；
- 大 LL cache 留工作站；
- 小型代码、manifest、结果表和报告正常 commit/push；
- 不 force push；
- 不访问 target90 / final holdout；
- 负结果完整保留。

---

# 15. 给 Codex 的执行要求

本任务是**实验执行任务，不是规划任务**。

开始后：

- 自主完成数据固定、最小移植、GPU 实验和汇总；
- 普通工程问题自行处理；
- 不因 baseline 输给 Frozen 而停止；
- 不擅自增加新方法；
- 不扩大数据集；
- 不改变成功标准；
- 不使用目标标签调参；
- 个别 baseline 阻塞时记录 BLOCKED 并继续其余任务。

完成定义：

> **协议固定、四域数据固定、baseline 保真、180 条流尽可能完整、结果可公平比较。**

不是必须发现某个 baseline 有效，也不是证明所有 online TTA 无效。

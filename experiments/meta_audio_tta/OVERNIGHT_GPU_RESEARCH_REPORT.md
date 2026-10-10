# SSL-AASIST MABN 启发式逐样本音频 TTA：夜间 GPU 源端研究报告

状态截点：2026-10-08 19:03 CST（11:03 UTC）。分支：`exp/meta-audio-tta`。本报告覆盖固定源端任务的完整运行；阶段 3 目标评分和指标均为 **NOT_RUN**。所有成功与失败尝试保留在各自 run 目录，原始 [目标访问事件记录](STAGE3_ACCESS_INCIDENT_20261008.md) 未修改。

## 1. 资源、数据和执行边界

| 观测时点 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
| --- | --- | --- | --- | --- |
| Joint 最后一轮期间 | 13 MiB，0% | 24,718 MiB，约 96%（Joint） | 977 MiB，约 28%（桌面） | 276 MiB，0%（空闲 ComfyUI 服务） |
| Cross/Same 启动后 | 3,440 MiB，41%（Cross） | 3,438 MiB，44%（Same） | 857 MiB，23%（桌面） | 276 MiB，0% |
| 两种 Meta 和源端诊断完成后 | 10 MiB，0% | 10 MiB，0% | 854 MiB，约 23% | 276 MiB，0% |

GPU 0/1 始终由原 `continue_stage12.py` 续跑器管理；新增评分只用 GPU 3，GPU 2 未占用。资源门要求 GPU 空闲显存 ≥20 GiB、已有 GPU 利用率 ≤20%、可用 RAM ≥16 GiB、磁盘空闲 ≥100 GiB、标准化 CPU load ≤0.75、10 秒 I/O pressure ≤5%。成功重试前的实测为 GPU 3 空闲 **47.01 GiB**、利用率 **0%**、RAM **80.41 GiB**、磁盘 **600.39 GiB**、CPU load **0.0534**、I/O pressure **0.27%**。本次没有完整峰值显存记录，不能据启动时显存推断峰值。

沿用现有 assignment，未重新划分：`fit=25,380`、`source_val=5,654`、`select=11,723`、`cal0=7,467`，角色 ID 不重叠，见 [源角色审计](results/source_role_audit_20261008.json)。仅 fit 用于训练反向传播；source_val 用于 epoch 选择和本报告固定诊断。新诊断使用每类按 sample ID 排序后的 **128:160** 位置，共 32 真人、32 伪造，与阶段 2 已用的每类前 128 条验证样本无交集；声学检查使用 fit 每类前 8 条、epoch 0 裁剪。没有使用 select 选新超参数或用 cal0 校准本诊断阈值。

## 2. 阶段 1/2 实际完成状态

| 阶段 | 固定预算与选择 | 结果及证据 |
| --- | --- | --- |
| CE 源训练 | 8/8 epoch、每轮 1,813 主干步，共 14,504 步；选中 epoch 6 | 选中 source_val EER **0.0**；[history](runs/stage1_20261008/ce/history.jsonl) 与 [selection](runs/stage1_20261008/ce/selection.json) 完整；原会话退出码 **0** |
| CE+BYOL 源训练 | 与 CE 同一 XLS-R 初始化、fit、增强和主干预算；8/8 epoch、14,504 步；选中 epoch 2 | 选中 source_val EER **0.0**；第 8 轮 CE loss **0.00887790**、BYOL loss **0.252821**；[history](runs/stage1_20261008/joint/history.jsonl) 与 [selection](runs/stage1_20261008/joint/selection.json) 完整。训练进程自身退出码未单独持久记录，标 **NOT_RECORDED**；后续真实门检退出 0 |
| 阶段 1 真实模型门检 | 原续跑器在 GPU 0 运行 | 命令退出 **0**、[gate](runs/stage1_20261008/gate.json) 状态 **PASS**；30 个有效骨干 BN 张量、34 个总快权重张量；BYOL 骨干梯度 L1 **12.1699**，K=1−K=0 鉴伪分数 **−0.000492096**，Cross/Same 元梯度均非零 |
| Cross Meta | 2/2 epoch，每轮 256 support 任务、256 外层步、512 个不同 fit 音频曝光；选中 epoch 1 | 两轮固定 256 条 source_val 上 K=0/K=1 EER **0/0**、AUC **1/1**，每轮 256 条分数变化；进程退出 **0** |
| Same Meta | 2/2 epoch，每轮 256 support 任务、256 外层步、256 个不同 fit 音频曝光；选中 epoch 1 | 两轮固定 256 条 source_val 上 K=0/K=1 EER **0/0**、AUC **1/1**，每轮 256 条分数变化；进程退出 **0** |

原续跑器在阶段 1 门检通过后才启动 Cross/Same，并在两组退出 0、选中 checkpoint 存在后写出 `STAGE2_COMPLETE`。命令、时间和退出码见 [阶段 2 事件日志](runs/stage2_20261008_gpu/launch.jsonl)，完整每轮数据见 [Cross 历史](runs/stage2_20261008_gpu/cross/history.jsonl) 与 [Same 历史](runs/stage2_20261008_gpu/same/history.jsonl)。Cross 与 Same 的独立 fit 音频曝光量不同，后续差异不能单独归因于外层目标。阶段 1 的整套 source_val 未保存 AUC，不能从其 EER 推断 AUC；阶段 2 上述 AUC 来自另一固定 256 条子集。

## 3. 新增诊断的实际命令、失败和恢复

[固定配置](config_source_gpu_diagnostics_20261008.json)只允许 GPU 2/3、源角色、学习率 **0.0003**。每条音频从选中 checkpoint 的固定初值执行两视图 K=1 更新；K=0/K=1 同 checkpoint、同一原音频。CE 无受训 BYOL 头，因此只算 K=0。判决翻转用原生 spoof logit 差的 **0** 边界；这是机制诊断边界，**不是** cal0 固定阈值，也不用于目标选参。

| 独立 run | 实际执行和退出码 | 保留的工件 |
| --- | --- | --- |
| `run_fixed` | CE 子任务退出 **1**：Cross/Same 刚启动时 I/O pressure 超过固定 5% 资源门；CE 分数未开始写出 | [事件](runs/source_gpu_diagnostics_20261008/run_fixed/launch.jsonl)、[拒绝日志](runs/source_gpu_diagnostics_20261008/run_fixed/ce.log) |
| `run_fixed_retry1` | 资源恢复后原配置重试；CE 退出 **0**。Joint 计算完成但汇总写入遇到 `numpy.int64` JSON 序列化错误，退出 **1**；其部分 summary **无效**，逐样本文件保留但不参与下表 | [事件](runs/source_gpu_diagnostics_20261008/run_fixed_retry1/launch.jsonl)、[有效 CE 结果](runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce/summary.json)、[Joint 错误日志](runs/source_gpu_diagnostics_20261008/run_fixed_retry1/joint.log) |
| `run_fixed_repair1` | 仅修正汇总 JSON 类型，固定模型、数据、学习率和资源门不变；复用已完成 CE。Joint、Cross、Same 各退出 **0**，控制器事件 `COMPLETE`、会话退出 **0** | [事件和精确子命令](runs/source_gpu_diagnostics_20261008/run_fixed_repair1/launch.jsonl)、各变体 `run_config.json`、逐样本 JSONL、summary 和日志 |
| 独立工件审计 | `audit_source_gpu_diagnostics.py` 退出 **0**、状态 **PASS** | [最终审计 JSON](runs/source_gpu_diagnostics_20261008/audit_v2.json)、[审计日志](runs/source_gpu_diagnostics_20261008/audit_v2.log)；首次审计 `audit_v1` 原样保留 |

第一次资源拒绝是按设计保护已有训练；第二次是工程输出错误，已停止受影响任务、保留部分文件并在新目录修复重跑。没有覆盖旧结果，也没有重新计算有效 CE 分数。修复后 `tta` 环境的 **8 项单测通过**，原始输出见 [pytest 日志](runs/source_gpu_diagnostics_20261008/preflight_tests_v6.log)。独立审计从逐样本文件复核 64 条精确 ID/类别覆盖、四个 checkpoint 血缘、CE 无 K=1、三组 K=0/K=1 指标和 1,024 个真假 pair 的排序翻转、16×2 fit 声学配对及其 EER/AUC、排序和更新方向、分数及 BYOL/BN 更新有限值；任何不符都会失败。审计脚本仅打开源端工件。

复现入口：`conda run -n tta python experiments/meta_audio_tta/continue_source_diagnostics.py --stage1-root experiments/meta_audio_tta/runs/stage1_20261008 --stage2-root experiments/meta_audio_tta/runs/stage2_20261008_gpu --stage1-gate experiments/meta_audio_tta/runs/stage1_20261008/gate.json --output-root <新且不存在的 run 目录> --device cuda:3 --wait-hours 8 --resume-after-ce --prior-ce-summary experiments/meta_audio_tta/runs/source_gpu_diagnostics_20261008/run_fixed_retry1/ce/summary.json`。实际每个子进程的完整参数和退出码以本次 `run_fixed_repair1/launch.jsonl` 为准；此入口仅供复现，不应原地重跑历史目录。

## 4. 固定 source_val：Frozen 与同 checkpoint TTA

下表来自每类 32 条、共 64 条**相同 ID**。EER/AUC 分开列出；错误数与逐样本判决翻转使用原生 0 边界。真假 pair 总数每组为 **32×32=1,024**。

| Checkpoint | K=0 EER / AUC / 错误 | K=1 EER / AUC / 错误 | 正确→错误 / 错误→正确 | 有害 / 有益 pair 翻转 | 非零分数变化 | 平均绝对分数变化 | 平均实际骨干 BN 更新 L1 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| CE epoch 6 | 0 / 1 / 0 | **NOT_RUN**（无 BYOL 头） | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Joint epoch 2 | 0 / 1 / 0 | 0 / 1 / 0 | 0 / 0 | 0 / 0 | 64/64 | 0.00196946 | 0.00412596 |
| Cross epoch 1 | 0 / 1 / 0 | 0 / 1 / 0 | 0 / 0 | 0 / 0 | 64/64 | 0.00245599 | 0.00450770 |
| Same epoch 1 | 0 / 1 / 0 | 0 / 1 / 0 | 0 / 0 | 0 / 0 | 64/64 | 0.00249517 | 0.00492763 |

三组 BYOL 模型均有 64/64 次非零骨干梯度、实际 BN 参数更新及鉴伪分数变化；目标分支和 checkpoint 参数/buffer 在诊断前后相同。BYOL 的改动不是只停留在辅助分支。但在此近乎完全分离的固定源子集，**没有可观察的 EER、AUC、判决或排序收益，也没有恶化**。分数平均绝对变化相对 K=0 分数标准差约为 **0.04%–0.05%**；这只是该子集的尺度描述，不能证明所有数据上都过小。

## 5. fit 声学扰动、BYOL 表征和初步机制

fit 每类 8 条分别在原音频与既有 deterministic FIR 条件下评分；每条件 16 条，K=0/K=1 的 EER 均为 **0**、AUC 均为 **1**、原生 0 边界错误均为 **0**，每条件 16/16 条鉴伪分数非零变化。原音频与 FIR 条件不是独立样本。

| 方法 | 原音频 K=1−K=0 平均绝对变化 | FIR 条件平均绝对变化 | 同一样本两条件适应方向符号改变 | source_val BYOL 归一化表征维度标准差均值 | source_val 在线/目标投影余弦均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint | 0.00219330 | 0.00104910 | 4/16 | 0.04267 | 0.99236 |
| Cross | 0.00197214 | 0.00513184 | 4/16 | 0.04877 | 0.94451 |
| Same | 0.00157952 | 0.00182101 | 6/16 | 0.04803 | 0.97488 |

三组在源端的 BYOL loss、在线/目标投影范数、梯度、实际 BN 差值和分数均为有限值；64/64 条有非零共享骨干 BN 更新。固定 FIR 扰动下，部分样本的更新方向改变，说明局部更新对声学条件有敏感性；这里没有性能翻转，不能据符号变化判断有益或有害。表征维度标准差非零，所测投影没有整体全零的迹象；这项小样本标量检查不能排除其他形式的表征塌缩。不同方法的投影余弦和分数变化不能直接作因果比较，因为 checkpoint 与 Cross/Same 的 fit 曝光量均不同。

本版只在一个 seed、一个固定内层学习率和容易分离的源子集上观察到**有效但无可测收益的适应**。64 条诊断音频虽然避开了阶段 2 的 256 条验证音频，却仍属于阶段 1 用于选 epoch 的整套 source_val；这些结果不能当独立确认集。不扩大搜索或目标评价来制造正结果；这也不能推论元适应原理无效。下一步研究应首先关注分数 margin 与更新幅度的关系、FIR 下方向不稳定的样本特征，以及 Cross/Same 曝光量差异对比较的限制。

## 6. 阶段 3 预审和访问边界

今晚只进行了既有 label-free 阶段 3 配置的静态审查：四域预期分数文件共 **16** 个，语义为 CE 的四个 K=0 文件，以及 Joint/Cross/Same 各域成对 K=0/K=1；不能复制 K=0 伪造 CE 的 K=1。既有合成边界测试日志 [boundary_test_v2.log](runs/stage3_20261008/boundary_test_v2.log) 为退出 **0、8 passed**，但不等于真实目标评分。正式冻结、目标评分、完整性封存及独立目标指标评价均 **NOT_RUN**。

本次源端诊断没有读取目标 runner、音频、标签或 evaluator sidecar；没有访问 target90 或 final holdout。原始 [访问事件](STAGE3_ACCESS_INCIDENT_20261008.md) 保留，今后若执行四域开发评价，必须标为 `disclosed-access target development evaluation`，不能宣称原始 never-opened-label 条件。今晚的源结果不用于调整目标超参数。

## 7. 明日优先检验的假设与停止点

1. 当前分数 margin 远大于 BN 更新引起的分数变化，导致源子集的判决和 pair 排序完全不变；先做源端 margin 分层核查，而不开放更多目标样本。
2. FIR 扰动下 4/16、4/16、6/16 个样本的适应方向翻转可能与声学变化或 BYOL 梯度几何有关；先检查现有逐样本记录，不追加训练。
3. Cross/Same 在已测源端同为饱和指标，且 fit 独立音频曝光量不同；若比较机制，需要在下一版本事先控制曝光量并登记版本差异，不能用本版结果作单因果结论。

固定的阶段 1/2 和源端诊断均已完成，工程错误及负结果保留。今晚到此停止：**不启动新训练、不扩大学习率或目标搜索、不执行阶段 3**。

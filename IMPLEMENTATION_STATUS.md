# EP-TTA v0.1.0 实际实现状态

日期：2026-09-14。项目整体：**IN_PROGRESS**。L0–L4 工程接口、冻结 cache 与机制对照已实现并通过当前 Linux 合成验收；真实数据已完成 inventory/统一标签，但 raw/label/group/preprocess/split 尚未发布正式人工审核 lock。无自有任务权重，未启动 GPU 训练或适配实验。

版本：项目 / 包 / 配置 / 自有合同均 `0.1.0`。本目录未创建 Git commit 或发布 tag；可追溯输入使用文件 SHA-256 清单。

## 本轮交付及证据

| 项目 | 状态 | 实际文件 / 范围 | 验证证据 |
|---|---|---|---|
| L0 项目初始化 | PASS | pyproject、README、AGENTS、状态/决策/测试文档；构建 wheel | 本地 pytest 的版本/CLI 检查；wheel_build_retry.log，退出码 0 |
| L1 配置与合同骨架 | PASS（本轮定义范围） | 严格 schema、分层/独立训练解析、registry、7 阶段校验、CLI 预览与拒绝执行；接口/角色/初始化合同 | pytest.log / pytest.xml：121 passed；source_audit.log，退出码 0 |
| L1 普通 YAML 加载分支 | PASS（合成） | SafeLoader 重复键拒绝；内置 JSON 语法 YAML 路径 | 当前 PyYAML 环境已执行，无 skip |
| L2 EP 数学实现与数值验收 | PASS（合成） | adapter/math/serial/batch/reset/final diagnostics；修复 K=0 零乘法导致的末位漂移 | 20260914-l3/pytest.xml：float32/64、B=1/2/7/128 等均执行 |
| L3 数据接入与增量 | PASS（接口与统一标签构建） | 多格式 adapter、ALLM 兼容统一标签、显式 label/group、staging、分组 split proposal、人工 lock、不可变 snapshot、delta/reconcile/refresh、preprocess cache identity | `tests/contracts/test_data_l3.py`、`tests/contracts/test_unified_labels.py`；全套 173 passed；真实 group/preprocess/split 合同仍未 LOCKED |
| L4 源模型训练链 | IMPLEMENTED / DEFERRED_REMOTE | 两套作者 commit/hash、factory、语义类别映射、source worker、DDP、source_val EER、checkpoint/resume/finalize/export parity | AASIST 本地真实结构 `[1,160]→[1,2]`；SSL/训练/恢复/export 待远程资源核验 |
| L5 缓存、机制与评价 | IMPLEMENTED_SYNTHETIC_VERIFIED | pickle-free 分块 cache、P0/P1 对照、offline U/M/tau/Fisher/static R、score seal 和二分类指标 | 合成测试通过；真实 feature/score 未产生；published 五项仅审计合同 |
| L6 调度与部署 | PARTIAL | source/extract worker、单卡/DDP 启动、分片 merge、远程 runbook | 不自动 SSH/下载；run-suite/build-artifacts 等高层编排仍显式阻塞 |

真实命令、退出码、环境、失败修复与跳过原因见 [TEST_REPORT.md](docs/TEST_REPORT.md)。本次证据目录为 [docs/test_logs/20260914-l3](docs/test_logs/20260914-l3)；首轮证据仍保留在 `20260913-local`。

## 验收 ID 映射

| ID | 当前范围与状态 |
|---|---|
| T01–T10、T13–T14 | PASS（合成）：生产/参考 PyTorch 对照已实际执行；含原视图、Nd、梯度、margin、投影、换基、reset、batch 和错误反例 |
| T11–T12、T15–T20 | PASS（合成）：稳定 entropy/MEMO、Fisher 解析/自动微分、subspace/17点 scalar/cache 与封存后指标；真实数据效果 NOT_RUN |
| C01–C02 | PASS（合成）：轻依赖 CLI、本机真实 I/O 拒绝、JSON/YAML 重复键、未知键、覆盖和非法范围 |
| C03 | PASS（合成）：manifest 白名单、目标 sidecar 不可见及标签字段拒绝 |
| C04–C09 | PASS（合成合同）：来源冲突、hash、原子 snapshot/cache、固定 checkpoint 比较组、smoke/export 与 resume 身份；真实中断和 SSL mode/buffer 为 DEFERRED_REMOTE |
| C10 | PASS（合成）：数值失败单条隔离并回退冻结分数；真实覆盖仍未验 |
| C11–C12 | published port 合同/label-free bridge 反例 PASS；五项算法 parity 仍 BLOCKED_AUDIT |
| D01 | PASS：结构 null 合法，真实阶段列缺项并拒绝，本地不解引用占位 |
| D02 | PASS（合成）：CSV/TSV/空白分隔/JSON/JSONL/sidecar；带/无表头，列/path 均显式 |
| D03–D04 | PASS（合成）：反向标签、未知隔离、同 ID/路径冲突和源训练角色防火墙 |
| D05–D10 | PASS（合成）：幂等 delta、跨角色组拒绝、更正不覆写、原子发布、依赖刷新、预处理 cache 身份 |
| S01–S10 | 接口/合成合同已实现；AASIST 作者结构前向 PASS。SSL 初始化、真实参数更新、DDP、resume、source_val 选模和冻结 parity 仍须远程执行，不能由合成测试替代 |
| R01–R04 | DEFERRED_REMOTE：无单/双卡、OOM、真实成本或评价封存测试 |

## 所有远程状态

- R0 数据目录只读盘点与标签预处理：IN_PROGRESS。当前绑定下曾发现 915,694 个音频文件和 7 个协议/sidecar 候选；随后按 17 个显式来源生成 1,415,725 条有效统一标签记录并全量检查音频存在性，173,551 条 Codecfake 缺音频记录隔离。详见 `docs/LABEL_PACK_REPORT.md`。GPU/环境/磁盘预检未完成。
- R1 数据/来源/预处理/划分审核：BLOCKED_CONTRACT。标签包已可读且 hash 绑定，但 `split_role=unassigned`，无 LOCKED group/preprocess/split 合同或正式 DatasetSnapshot。
- R2 源模型 smoke、R3 自主正式训练、R4 冻结导出：NOT_RUN。
- R5 特征与 U/M/tau0、R6 先导/基线、R7 封存、R8 确认性评分、R9 评价：NOT_RUN。
- R-data 增量接入：NOT_RUN。
- 所有真实模型/数据/实验输出、效果指标和科学假设 H1–H7：未获得结果；不得宣称支持。

## 未解决项与下一阶段

1. 审核 `configs/data_contracts/asvspoof2019_la.observed.yaml.example` 的三类合同，不选择同目录的 2021 key 或转换 sidecar；按 runbook 生成 staging，检查全部诊断和覆盖。
2. 审核预处理及 `configs/split_policies/asvspoof2019_la.yaml.example`，确认来源组、类别数与角色边界后发布 snapshot。
3. 使用已固定的 AASIST commit 先做 remote smoke，核验音频 worker、GPU、显存、完整 source_val、last/best 与恢复；通过后单独启动 full。
4. SSL 作者 commit/class index/160维 embedding 已源码核对；通用 XLS-R 文件和兼容 fairseq 环境仍未绑定。没有自动下载、SSH、CUDA 安装或正式训练。

**训练/缓存/机制工程接口已交付；正式数据 lock、自有底座训练和论文实验仍未完成。**

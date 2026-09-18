# EP-TTA v0.1.0 实际实现状态

日期：2026-09-18。项目整体：**IN_PROGRESS**。审批/角色消费门禁、统一 suite 派发、数值回退、最终诊断、通用 AASIST R4、小样本采样、选择/冻结/报告和 review-only 计划生成已修复。现有 ASV2019 R4/R5 工件保持只读；没有启动新训练、GPU 适配或目标评价。最新证据见 [REVIEW_REGRESSION_REPORT_20260918.md](docs/REVIEW_REGRESSION_REPORT_20260918.md)。

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
| L5 缓存、机制与评价 | IMPLEMENTED_SYNTHETIC_VERIFIED | R4→R5 冻结门禁、按 purpose/role 的 pickle-free 分块 cache、P0/P1、offline U/M/tau/Fisher/static R、confirmatory 计划 hash、score/coverage seal、按类翻转指标 | 211 项合成测试通过；真实 feature/score 未产生；published 五项仅审计合同 |
| L6 调度与部署 | PARTIAL | source/extract worker、分片 merge、source artifacts/cache suite；`select-methods/freeze/report` 与 review-only `make_plans.py` 已实现 | 不自动 SSH/下载/训练；目标 snapshots 与 published ports 仍阻塞 |

真实命令、退出码、环境、失败修复与跳过原因见 [TEST_REPORT.md](docs/TEST_REPORT.md)。最新门禁证据目录为 [docs/test_logs/20260915-r5-r9-gates](docs/test_logs/20260915-r5-r9-gates)；此前 R0/R1、L3/L4 与本地证据均原样保留。

## 验收 ID 映射

| ID | 当前范围与状态 |
|---|---|
| T01–T10、T13–T14 | PASS（合成）：生产/参考 PyTorch 对照已实际执行；含原视图、Nd、梯度、margin、投影、换基、reset、batch 和错误反例 |
| T11–T12、T15–T20 | PASS（合成）：稳定 entropy/MEMO、Fisher 解析/自动微分、subspace/17点 scalar/cache、完整 coverage seal 与按类有益/有害翻转；真实数据效果 NOT_RUN |
| C01–C02 | PASS（合成）：轻依赖 CLI、本机真实 I/O 拒绝、JSON/YAML 重复键、未知键、覆盖和非法范围 |
| C03 | PASS（合成）：manifest 白名单、目标 sidecar 不可见及标签字段拒绝 |
| C04–C09 | PASS（合成合同）：来源冲突、hash、原子 snapshot/cache、固定 checkpoint 比较组、smoke/export 与 resume 身份；真实中断和 SSL mode/buffer 为 DEFERRED_REMOTE |
| C10 | PASS（合成）：数值失败单条隔离并回退冻结分数；score seal 要求 coverage=1；真实覆盖仍未验 |
| C11–C12 | published port 合同/label-free bridge 反例 PASS；缺失 port 保留 `NOT_RUN/BLOCKED_AUDIT` 行；五项算法 parity 仍 BLOCKED_AUDIT |
| D01 | PASS：结构 null 合法，真实阶段列缺项并拒绝，本地不解引用占位 |
| D02 | PASS（合成）：CSV/TSV/空白分隔/JSON/JSONL/sidecar；带/无表头，列/path 均显式 |
| D03–D04 | PASS（合成）：反向标签、未知隔离、同 ID/路径冲突和源训练角色防火墙 |
| D05–D10 | PASS（合成）：幂等 delta、跨角色组拒绝、更正不覆写、原子发布、依赖刷新、预处理 cache 身份 |
| S01–S10 | 接口/合成合同已实现；AASIST 作者结构前向 PASS。SSL 初始化、真实参数更新、DDP、resume、source_val 选模和冻结 parity 仍须远程执行，不能由合成测试替代 |
| R01–R04 | DEFERRED_REMOTE：无单/双卡、OOM、真实成本或评价封存测试 |

## 所有远程状态

- R0 资源与数据只读盘点：PASS。实测 4×RTX A6000（每卡 49140 MiB）、NV4 配对拓扑、125 GiB RAM、数据盘可用 1.2 TiB；py310/py38 均可见四卡。ASV2019 inventory、源码和统一标签 hash 已记录。详见 `docs/REMOTE_R0_R4_20260914.md`。
- R1 数据/来源/预处理/划分审核：PARTIAL / BLOCKED_CONTRACT。审核人 `mm` 已明确批准并发布 raw/label/group/preprocess lock。staging `staging-71275b8662b3509ce95a` 含 121,461 条，parse error、label conflict、quarantine 均为 0。split proposal `e07ddd4d…941db` 已完成全覆盖/分组审计；等待独立 split 批准，因此仍无正式 DatasetSnapshot。
- 历史 AASIST R2–R5：已有只读 full 训练选择、R4 bundle、R5 fit/cal0/select cache 与源资源；其报告和负结果保持原样。新 seed/新 source 的小型真实 AASIST 导出验收仍 NOT_RUN，SSL-AASIST parity 仍 NOT_RUN。
- 本补丁后的 R6/R7/R8/R9：NOT_RUN。入口现强制复核 approval、manifest role/scope、cache identity、精确 ID seal 与 fallback 门限；select 与 confirmatory 角色分离。真实总成本、GPU 峰值、端到端编码成本、新目标结果和五项 published-port 仍未运行/未核验。
- R-data 增量接入：NOT_RUN。
- 所有真实模型/数据/实验输出、效果指标和科学假设 H1–H7：未获得结果；不得宣称支持。

## 未解决项与下一阶段

1. 审核 `/media/dell/data/fakedata/eptta_work/splits/asvspoof2019_la.proposal.json` 的具体分布，尤其 source_val 5组/5,654条、select 8组/11,520条；同意后单独发布 split lock。
2. split lock 后发布不可变 DatasetSnapshot，并封存 fit/source_val manifests；不自动重划旧分配。
3. 使用已固定的 AASIST commit 先做 remote smoke，核验音频 worker、GPU、显存、完整 source_val、last/best 与恢复；通过后单独启动 full。
4. SSL 作者 commit/class index/160维 embedding 已源码核对；通用 XLS-R 文件和兼容 fairseq 环境仍未绑定。没有自动下载、SSH、CUDA 安装或正式训练。
5. 仅在 R4 自有 full checkpoint、source_val 选模和冻结 parity 全部通过后进入 R5；随后依次做源 cache/U/M/tau0、P0/P1 select 先导、五项忠实音频移植、confirmatory 计划封存、无标签评分与独立评价。缺数据/方法保留 NOT_RUN 行，不据目标结果删行。

**训练/缓存/机制工程接口已交付；正式数据 lock、自有底座训练和论文实验仍未完成。**

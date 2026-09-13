# EP-TTA v0.1.0 实际实现状态

日期：2026-09-13。项目整体：**IN_PROGRESS**。本轮 L0–L2 开发交付已停止；L2 数学代码已写，数值验收因缺少 PyTorch 为 **NOT_RUN**。本轮从仅有设计文档的目录开始，保留原 `docs/DESIGN.md`；没有已有实现、已有任务权重或实验成功的假设。

版本：项目 / 包 / 配置 / 自有合同均 `0.1.0`。本目录未创建 Git commit 或发布 tag；可追溯输入使用文件 SHA-256 清单。

## 本轮交付及证据

| 项目 | 状态 | 实际文件 / 范围 | 验证证据 |
|---|---|---|---|
| L0 项目初始化 | PASS | pyproject、README、AGENTS、状态/决策/测试文档；构建 wheel | 本地 pytest 的版本/CLI 检查；wheel_build_retry.log，退出码 0 |
| L1 配置与合同骨架 | PASS（本轮定义范围） | 严格 schema、分层/独立训练解析、registry、7 阶段校验、CLI 预览与拒绝执行；接口/角色/初始化合同 | pytest.log / pytest.xml：121 passed；source_audit.log，退出码 0 |
| L1 普通 YAML 加载分支 | NOT_RUN | SafeLoader 重复键拒绝实现；内置 JSON 语法 YAML 路径已测 | PyYAML 未安装；单项 skip |
| L2 EP 数学实现 | IN_PROGRESS | adapter/math/serial/batch/reset/final diagnostics/单条数值隔离已实现；设计 A/B 参考已原样抽取 | 语法检查通过；不能据此标数学 PASS |
| L2 数值验收 | NOT_RUN | tests/unit/test_ep.py，float32/64、B=1/2/7/128、K=0/1/5 等合成对照 | CPU PyTorch 未安装；整个模块 skip，未执行任何张量测试 |
| L3 数据接入与增量 | TODO | 仅 DatasetAdapter/LabelMapper/SplitPlanner/IncrementalReconciler 协议与少量合同原语 | 全格式 parser、来源分组、snapshot/delta/原子发布未实现 |
| L4 源模型训练链 | TODO | 仅 ModelFactory/SourceTrainer/SourceTrainJob/FrozenModelBundle 接口及拒绝条件 | 网络、作者源码审计、worker、checkpoint/resume/export 未实现 |
| L5 基线与评价 | TODO | 其他方法只登记 TODO，禁止静默退化 | 无基线/指标/封存实验 |
| L6 调度与部署 | TODO | CLI 名称/help 和停止入口；远程顺序文档 | 无 SSH、worker、双卡 launcher 或恢复执行器 |

真实命令、退出码、环境、失败修复与跳过原因见 [TEST_REPORT.md](docs/TEST_REPORT.md)。证据目录：[docs/test_logs/20260913-local](docs/test_logs/20260913-local)。输入哈希见该目录 delivery_manifest.json；它不包含自身以避免自引用。

## 验收 ID 映射

| ID | 当前范围与状态 |
|---|---|
| T01–T10、T13–T14 | 合成生产/参考测试已写，NOT_RUN（缺 PyTorch）。含原视图、Nd、梯度、margin、投影、换基、reset、batch 对照和故意错误实现反例 |
| T11–T12、T15–T20 | TODO，后续 offline/基线/评价实现；没有纳入本轮 PASS |
| C01–C02 | PASS：轻依赖 CLI、禁止本机真实 I/O、JSON 重复/未知键、覆盖和非法范围；普通 YAML 分支单列 NOT_RUN |
| C03 | PASS：manifest 白名单及目标标签字段拒绝；数值标签防火墙对照 NOT_RUN |
| C04–C09 | 完整链路 TODO/DEFERRED_REMOTE；只有路径/角色原语已测。C07 合成源张量不变测试 NOT_RUN，真实模型 mode/buffer 冻结完全未验 |
| C10 | 数值隔离测试已写，NOT_RUN |
| C11–C12 | TODO，未实现 published port/bridge/选择审计 |
| D01 | PASS：结构 null 合法，真实阶段列缺项并拒绝，本地不解引用占位 |
| D02 | TODO：只有测试侧合成 CSV，不算完整多格式 DatasetAdapter |
| D03–D04 | 合同原语 PASS：显式反向标签、空/未知隔离、源角色与训练 job 防火墙；全量冲突/覆盖审计仍 TODO |
| D05–D10 | TODO：snapshot、增量冲突、更正、恢复和预处理未实现；真实资源 DEFERRED_REMOTE |
| S01–S10 | TODO / 真实训练 NOT_RUN。初始化范围、source job、resume 身份、smoke/finalized 导出条件仅为本轮合同测试，不能替代 S 系列架构和训练验收 |
| R01–R04 | DEFERRED_REMOTE：无单/双卡、OOM、真实成本或评价封存测试 |

## 所有远程状态

- R0 环境资源盘点、R1 数据/来源/预处理/划分审核：DEFERRED_REMOTE，实际未运行。
- R2 源模型 smoke、R3 自主正式训练、R4 冻结导出：NOT_RUN。
- R5 特征与 U/M/tau0、R6 先导/基线、R7 封存、R8 确认性评分、R9 评价：NOT_RUN。
- R-data 增量接入：NOT_RUN。
- 所有真实模型/数据/实验输出、效果指标和科学假设 H1–H7：未获得结果；不得宣称支持。

## 未解决项与下一阶段

1. 在已准备好的 CPU PyTorch 环境运行 `python -m pytest tests/unit/test_ep.py -q`，核验生产串行/batch 与设计参考；若失败先修复 L2，不推进真实评分。普通 YAML 分支也待有 PyYAML 时补跑。
2. L3 实现多格式原始协议、label/group 冲突、split/snapshot 与增量发布；目前不猜真实列、不自动审批。
3. L4 读取并固定作者架构与配方，训练任务权重由本项目后续完成；SSL 通用初始化来源另记。schema 字段齐全仍不能绕过尚未实现的证据审计和 worker。
4. 远程路径/资源、预处理、split、recipe、作者 commit、class index、实际 embedding 维数、通用 SSL 初始化全部待核验；没有自动下载、SSH、CUDA 安装或正式训练。

**本轮到此停止。完整仓库、训练底座与论文复现均未完成。**

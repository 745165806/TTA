# EP-TTA v0.1.0

本项目从 `docs/DESIGN.md` 的首版合同开始建立。本轮交付 **L0–L2 本地开发代码**：配置与权限骨架、数据/训练接口、合成特征上的 EP 数学核心。真实数据接入、检测器训练、冻结导出、缓存、完整基线及论文实验均未运行。

主规范：[DESIGN.md](docs/DESIGN.md)；实际进展：[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)；命令与证据：[TEST_REPORT.md](docs/TEST_REPORT.md)。项目、包和自有 schema 版本均为 `0.1.0`，方法 ID 为 `ep_tta`。

## 本机使用

Python >= 3.10。基础 CLI 无强制第三方依赖；提供的 `.yaml` 使用 JSON 语法（YAML 的子集），标准库即可读取。普通 YAML 需要显式安装的 PyYAML，加载时拒绝重复键、未知键和非法覆盖。PyTorch 仅由数值模块延迟导入；本轮没有安装依赖。

在项目根目录执行（PowerShell）：

```powershell
$env:PYTHONPATH = 'src'
py -3 -m eptta.cli --version
py -3 -m eptta.cli --config configs/base.yaml --profile local_dev validate --level structure --stage development
py -3 -m eptta.cli --config configs/base.yaml --profile local_dev plan --experiment configs/experiments/source_pilot.yaml --task source_prepare --out plans/source_prepare.preview.json --dry-run
py -3 -m pytest tests/unit tests/contracts -q
```

POSIX shell 用 `PYTHONPATH=src python -m eptta.cli ...`。若已由用户准备好构建依赖，可选择安装包并使用 `eptta` 命令；本轮没有执行安装。基础配置校验不访问任何数据、权重、占位路径或网络。plan 始终为预览，输出已存在时拒绝覆盖。

`validate --level resources --stage source_training` 会汇总本机权限和未锁定合同，返回退出码 2；此处 resources 仅检查阶段合同字段，**尚未实现远程文件存在性/审核证据核验**。即使字段齐全，也返回 `execution_ready=false`。所有真实执行子命令目前仅有 help 和明确拒绝入口，绝不会从预览启动训练或访问真实资源。

## 已实现范围

- `src/eptta/config`：严格 schema、分层解析、来源记录、独立源训练预览解析、七阶段校验；`configs` 包含模型/数据/方法 registry 及 null 合同模板。
- `src/eptta/data`：DatasetAdapter / LabelMapper / SplitPlanner / IncrementalReconciler 协议，显式标签映射原语、无标签清单投影、角色与路径约束。真实格式解析、来源分组及增量发布留待 L3。
- `src/eptta/training`、`src/eptta/models`：SourceTrainer / SourceTrainJob / ModelFactory / FrozenModelBundle 合同。AASIST、SSL-AASIST 网络和训练器尚未实现；通用 SSL 初始化与本项目任务权重分别建模。
- `src/eptta/adaptation`：固定三视图、双侧子空间适应器、Nd 视图方差、单侧源间隔、无动量投影 SGD、每条 reset、原视图最终评分。独立 batch 保留正交补常数、sum 反传、逐矩阵投影，数值异常串行重放；当前仅允许 `synthetic_test` 通道。
- `tests`：配置、CLI、协议/权限反例，以及生产串行/批量与设计附录 A/B 的数学对照。**PyTorch 缺失时数值测试为 NOT_RUN；代码存在不代表等价性已经验证。**

## 数据和模型边界

`0=bonafide, 1=spoof`，`score > tau` 判 spoof。目标标签、攻击 ID、干净父音频、历史和整体统计不进入 TargetViews；源锚点标签仅允许来自 fit。训练允许 fit 梯度与 source_val 选模，不能接受 select/cal0/test 引用。

真实 raw label 格式、目录、preprocess、split、作者架构 commit/native 类别映射和训练 recipe 保持未审核。`UNRESOLVED → PROPOSED → APPROVED → LOCKED` 需要 reviewer、时间、报告、抽样证据和 payload hash；本轮没有实现自动批准或真实 lock 发布。

SSL-AASIST / AASIST 鉴伪权重由本项目后续在远程 Ubuntu 双 A6000 自行训练。允许经授权审核的通用 SSL 前端预训练初始化；不允许用作者的任务 checkpoint 替代源训练。smoke、未 finalized、外部/合成任务权重均不能作为正式 frozen bundle。

## 下一阶段

先在已有 CPU PyTorch 环境补跑 L2 数值验收；随后由用户开启 L3：真实协议插件、来源分组、不可变 snapshot、增量冲突和审核发布。L4 再实现固定作者架构与训练链。所有远程 R0–R9 均保持未运行，本轮到此停止。

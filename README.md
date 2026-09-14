# EP-TTA v0.1.0

本项目以 `docs/DESIGN.md` 为唯一主合同。当前已完成 L0–L3，并实现了 L4 源训练/恢复/选模/冻结接口、pickle-free 特征缓存、P0/P1 机制对照和 published-port 审计门。真实 GPU 训练、SSL 初始化核验、冻结 parity、目标适配和论文实验仍未运行。

主规范：[DESIGN.md](docs/DESIGN.md)；实际进展：[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)；命令与证据：[TEST_REPORT.md](docs/TEST_REPORT.md)。项目、包和自有 schema 版本均为 `0.1.0`，方法 ID 为 `ep_tta`。

## 本机使用

Python >= 3.10。基础 CLI 无强制第三方依赖；提供的 `.yaml` 使用 JSON 语法（YAML 的子集），标准库即可读取。普通 YAML 需要显式安装的 PyYAML，加载时拒绝重复键、未知键和非法覆盖。PyTorch 仅由数值模块延迟导入；本轮没有安装依赖。

在项目根目录执行：

```bash
PYTHONPATH=src python -m eptta.cli --version
PYTHONPATH=src python -m eptta.cli --profile local_dev validate --level structure --stage development
python -m pytest -q
```

基础配置校验不访问任何数据、权重、占位路径或网络。`local_dev` profile 调用真实数据 I/O 会失败；真实 inventory/staging 必须显式选择 remote profile 与私有路径绑定。所有发布命令拒绝覆盖已有输出。

数据 CLI 已实现 `prepare-labels / validate-labels`、`inspect-data → propose-data-contract → approve-contract → stage-data → propose-splits → build-manifests → seal-source-manifests`，以及 `ingest-delta / commit-snapshot / refresh-plan`。`approve-contract` 要求单独人工审核并绑定 payload hash。源训练入口为 `inspect-model / resolve-training-recipe / train-source / resume-source / finalize-training / export-frozen`；每一步单独调用，smoke 不自动续跑 full。

## 已实现范围

- `src/eptta/config`：严格 schema、分层解析、来源记录、独立源训练预览解析、七阶段校验；`configs` 包含模型/数据/方法 registry 及 null 合同模板。
- `src/eptta/data`：CSV/TSV/空白分隔/JSON/JSONL/sidecar，显式列与 JSON path；标签映射、协议上下文到安全相对音频路径、来源组、隔离诊断、staging/split/snapshot、增量与预处理/cache 身份。六个 registry 数据 ID 均有候选插件，但真实合同仍逐数据集审核。
- `src/eptta/training`、`src/eptta/models` 与 `workers/`：固定两套作者 commit/hash，native 类别映射，AASIST/SSL-AASIST 构造，语义加权 CE，单卡/DDP worker，完整 source_val EER、last/best、RNG/采样器恢复、finalize/export parity。SSL 只接受显式通用前端初始化。
- `src/eptta/cache`、`execution`：冻结特征分块 `.npy`、`allow_pickle=False`、精确 ID 覆盖、原子发布/合并和 label-free extraction job。
- `src/eptta/adaptation`、`offline`、`baselines`：主 EP、批量 sum 梯度，以及 frozen/multiview/static、目标×保持六格、L2/logit/Fisher、response/PCA/random U、固定源 R、17 点 scalar、no-projection/source-CE。五项 published port 仅有审计合同且显式阻塞。
- `tests`：D/S/T/C 合同、缓存、机制、评价封存、训练产物和原数学参考。2026-09-14 在现有 Python 3.10/PyTorch 环境最新执行为 206 passed。

## 数据和模型边界

`0=bonafide, 1=spoof`，`score > tau` 判 spoof。目标标签、攻击 ID、干净父音频、历史和整体统计不进入 TargetViews；源锚点标签仅允许来自 fit。训练允许 fit 梯度与 source_val 选模，不能接受 select/cal0/test 引用。

`/media/dell/data/fakedata` 的私有路径绑定保存在被 Git 忽略的 `configs/paths.fakedata.private.yaml`，本机统一标签源说明保存在同样被忽略的 `configs/unified_labels.fakedata.private.json`，生成包写入 Git 忽略的 `fakedata/`。ASVspoof 2019 LA 的只读观测候选位于 `configs/data_contracts/asvspoof2019_la.observed.yaml.example`，仍为 PROPOSED；它不是审核结论。统一标签包可供数据检查和后续 loader 使用，但其中 `split_role=unassigned`，不能替代 group/split/preprocess 的正式审核。`UNRESOLVED → PROPOSED → LOCKED` 必须有 reviewer、带时区时间、报告、抽样证据和 payload hash。

SSL-AASIST / AASIST 鉴伪权重由本项目后续在远程 Ubuntu 双 A6000 自行训练。允许经授权审核的通用 SSL 前端预训练初始化；不允许用作者的任务 checkpoint 替代源训练。smoke、未 finalized、外部/合成任务权重均不能作为正式 frozen bundle。

## 当前启动边界

当前统一 label pack 已完成，但正式 group/preprocess/split lock 与 DatasetSnapshot 尚未发布，所以不能直接执行训练。按 [REMOTE_RUNBOOK.md](docs/REMOTE_RUNBOOK.md) 完成 snapshot 和 recipe 审核后，先跑 AASIST smoke；通过远程显存、覆盖、checkpoint 读写和完整 source_val 验证后，再单独启动 full。SSL 还必须补通用 XLS-R 文件绑定。不得以作者任务 checkpoint 或 toy 回退越过该步骤。

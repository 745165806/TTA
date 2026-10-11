# EP-TTA

本仓库保留原有 AASIST / SSL-AASIST、源资源、EP 更新、baseline 与评价语义，把日常流程收敛为：准备数据 → 训练或复用源模型 → 准备源资源并选参 → 无标签 TTA → 独立评价 → 汇总。数学、数据角色和泄漏边界见 [docs/DESIGN.md](docs/DESIGN.md)。历史审批记录仍可读，但 reviewer、锁定状态、阶段提案、人工 SHA 和 score seal 均不再是执行前提。

## 安装：唯一 `tta` 环境

```bash
conda env create -f environment.yml
conda activate tta
python -m pip install -e .
python -m eptta.cli --help
python -m pytest -q
```

训练、特征提取、TTA、评价和测试均使用当前 `tta` 环境的 `python`；子进程使用 `sys.executable`。仓库不会自动下载数据/权重、SSH、安装 CUDA 或修改已有环境。

`environment.yml` 固定的是依据作者依赖和本机现状整理的单环境候选配方。SSL adapter 明确使用该环境安装的 fairseq 0.12.2，不再切换到作者目录内另一套 Python/fairseq 环境。本机已有 `tta` 环境；两套真实模型在该配方下的权重加载、前向、反向与 CUDA 联合验证仍未完成，详见“验证范围”。

私有路径写入不提交的本地文件，例如：

```bash
cp configs/research/paths.local.yaml.example configs/research/paths.local.yaml
```

实验配置中的 `@path:key` 从 `--paths` 文件读取；`${ENV_NAME}` 做显式环境变量展开。缺失路径、未设置变量和 `REPLACE_WITH_...` 占位都会在实际使用时明确报错。

本机后续实验使用 `configs/local_v2/paths.local.yaml`、`data/manifests_v2/asv2019_la/` 和 `configs/local_v2/` 中的运行配置。当前 AASIST 与 SSL-AASIST 的 frozen bundle 和源资源分别位于 `outputs_v2/aasist/` 与 `outputs_v2/ssl_aasist/`；模型、U、anchors、Fisher 和 cal0 阈值须按同一源训练运行成套使用。`configs/experiment.private.json` 是旧 R4/R5 实验记录，不作为新实验入口；旧划分、评分记录和停用的 `experiment.private.env` 保存在忽略的 `local/archive/eptta_work_20261011/`。

## 1. 准备数据

```bash
python -m eptta.cli prepare-data \
  --config configs/research/asv2019.yaml \
  --paths configs/research/paths.local.yaml
```

输入必须是显式 CSV/JSONL manifest，不根据目录名猜标签、group、release 或 subset。必需列为 `sample_id`、`audio_relpath`、label、group；已有 `split_role` 原样复用，也可用 `split.assignments` 显式引用上次生成的 `assignments.csv`。既有 role 和 `sample_index` 保持不变；仅对尚无 assignment 的新 group，按排序后的 group 列表、固定 seed 和独立整数 PRNG 分配一次，并为新样本追加 index。已有划分和 ID 不会重建。

输出包括完整 `manifest.csv`、角色 manifest、无标签 `inference/<role>.jsonl`、独立 labels/groups 和 `summary.json`。TTA 只接收无标签 inference 视图。

本机 WaveFake Parquet 与 DFADD Arrow 的显式音频导入、预处理和分组配置见 [docs/EMBEDDED_DATASETS.md](docs/EMBEDDED_DATASETS.md)。

## 2. 训练或复用源模型

```bash
python -m eptta.cli train-source \
  --config configs/research/aasist.yaml \
  --paths configs/research/paths.local.yaml
```

`phase` 可为 `smoke` 或 `full`。

仓库同时提供互不覆盖的 smoke/full 配置：`aasist.yaml` 与
`aasist_full.yaml`，以及 `ssl_aasist.yaml` 与 `ssl_aasist_full.yaml`。例如启动
AASIST 完整训练：

```bash
python -m eptta.cli train-source \
  --config configs/research/aasist_full.yaml \
  --paths configs/research/paths.local.yaml
```

完整续训使用：

```bash
python -m eptta.cli train-source \
  --config configs/research/aasist_full.yaml \
  --paths configs/research/paths.local.yaml \
  --resume outputs/source/aasist-full-asvspoof2019train-random/checkpoints/last.pt
```

每个 epoch 原子写入不可覆盖的 `checkpoints/epoch_XXXX.pt`。它保存完整 `model.state_dict()`（参数与持久 buffer），并同时保存 optimizer、scheduler、AMP scaler、epoch/global_step、best 指标、RNG、必要 sampler/DDP 状态，因此当前实现的 epoch 文件是可续训全状态超集。`last.pt` 指向最新完整状态，`best.pt` 指向 source_val 最优的具体 epoch；`history.csv` 记录逐 epoch 指标。

旧 checkpoint 可继续评价或作为新运行初始化。若缺 optimizer/RNG 等恢复字段，或来自旧 SHA 派生裁剪规则，只能标为 warm start，不能宣称 exact resume。已有文件不会被批量转换或覆盖。

## 3. 准备源资源并选择参数

```bash
python -m eptta.cli prepare-source \
  --config configs/research/source.yaml \
  --paths configs/research/paths.local.yaml
```

该入口复用冻结导出、特征提取、U/anchors/margins/Fisher/static-R、cal0 校准和 source-only select 实现，只构建当前配置所需资源。既有 frozen bundle 或缓存需显式引用；缓存必须声明具体 source run、不可变 epoch checkpoint、dataset/role、manifest、完整预处理/视图配置、seed、dtype/数值模式，并通过精确 ID、shape、dtype 和有限值检查。缺失资源不会补零。

项目不再计算内容摘要，也不自动跨运行寻找“相同”缓存。程序因此不能识别被外部原地替换、但路径与普通元数据保持不变的文件；依赖不可覆盖的 epoch/cache 目录和显式 `cache_ref` 操作纪律。输入语义变化时创建新目录。

## 4. 无标签 TTA

```bash
python -m eptta.cli run-tta \
  --config configs/research/control_test_ep.yaml \
  --paths configs/research/paths.local.yaml
```

`run-tta` 不接受 labels 参数，也不会打开目标标签文件。EP 仍逐样本 reset；K=0、投影、损失、baseline、数值诊断和预设回退率限制保持不变。未知方法/错误配置不会伪装成数值回退，未实现 port 仍是 `NOT_RUN`。

普通输出为：

```text
outputs/<run_name>/
  config.yaml
  meta.json
  scores.csv
  metrics.json       # evaluate 后生成
  log.txt
```

`meta.json` 记录 run ID、具体模型/cache/data 路径、seed、环境、方法和选参来源。脏工作区另存 `code.diff`；Git commit 仅是来源记录，不是执行令牌。

## 5. 独立评价

```bash
python -m eptta.cli evaluate \
  --run outputs/control_test_ep \
  --labels data/manifests/asv2019_la/labels/control_test.jsonl
```

评分阶段先原子生成 `scores.csv`，评价阶段才读取 labels。评价要求 scores、expected IDs 和 label IDs 集合精确相等，并检查重复 ID 与 NaN/Inf；不通过 inner join 丢样本。固定阈值来自 cal0 记录，不用目标标签重新校准。EER 只在评价时按定义扫描；缺 tDCF/minDCF 所需输入时明确标不可用。

## 6. 汇总已有结果

```bash
python -m eptta.cli report --runs outputs --out outputs/summary.csv
```

`report` 只汇总已有 `metrics.json`，不重新评分，也不根据最终测试结果选择方法。

## 科学边界与随机规则迁移

- `0=bonafide, 1=spoof`，分数越大越偏 spoof；source native 类别映射不变。
- fit/source_val/source-select/cal0/control_test/target_test 的用途保持分离，audit/cal1 按既有协议保留。
- TTA 不接收目标标签、攻击、干净父音频、目标整体统计或跨样本历史；`sample_id` 只关联结果。
- 新训练裁剪与视图由持久化的整数 `sample_index`、seed、epoch、view 编号和 namespace 驱动独立 PRNG，不依赖读取顺序或全局 RNG；源响应子空间的组内抽样同样改用排序后的 group index、样本位置和整数 seed。各自的采样范围/分布和后续数学不变。
- 这些规则会改变旧 SHA 派生的裁剪、视图和源响应抽样序列，因此新旧训练轨迹与重建的 U 不保证逐位一致。已有 checkpoint/明确归属的源资源可只读评价；旧运行若无逐视图/抽样记录，不能恢复原序列或宣称 exact resume。使用新规则重建源资源时必须写入新目录。
- 模型结构、损失、预处理操作/增强分布、EP 数学、数据角色和指标定义未因该迁移改变。

## 验证范围

轻量测试命令为：

```bash
python -m pytest -q
```

它不访问网络、远程数据或 CUDA，覆盖 EP reference/梯度/投影、K=0、逐样本 reset、serial/batch、标签方向、指标边界、group 隔离、TTA 标签隔离、显式缓存归属、精确 ID、数值回退以及 epoch checkpoint 重载/续训状态合同。

2026-10-10 主线整合在本机 `tta` 环境运行了 `tests/unit tests/contracts tests/integration`，结果为 **286 passed、2 skipped**。两个跳过项分别缺少本 worktree 的真实 target90 manifest 和真实 SSL-AASIST 波形校验资源，均不能算作通过。仍未执行真实两模型的完整权重/CUDA 联合验证、完整训练、全量缓存或新的目标集评价。轻量工程测试不代表论文方法有效。

## 主线与研究归档

新的 EP-TTA 研究从最新 `origin/main` 创建独立分支和 worktree；不要在含有本地科研产物或未提交工作的历史目录中切换分支。例如：

```bash
git fetch origin main
git worktree add -b codex/ep-tta-next-YYYYMMDD \
  .worktrees/ep-tta-next-YYYYMMDD origin/main
```

[研究归档索引](RESEARCH_ARCHIVE_INDEX.md)记录历史分支、原始提交、归档 tag 和研究结果边界；[整合记录](docs/research/CONSOLIDATION_20261010.md)说明进入主线的代码与未运行的验证。[分支管理说明](docs/BRANCH_MANAGEMENT.md)给出新研究 worktree 和历史退出条件。Git tag 只保存已提交历史；重要且不可轻易重建的产物须有独立设备或可信远端的第二份备份及恢复验证。私有 Meta Audio 历史不得直接推送公开仓库。

[截至 2026-10-11 的研究进展与实验结果](docs/research/PROGRESS_20261011.md)汇总各协议 EER、方法结论和未完成事项。

截至 2026-10-11，公开代码 PR #2 与文档 PR #3 已合入主线，主工作目录已切换至干净的 `main`。历史普通实验、文档整合及两处私有 Meta Audio worktree 已按[归档索引](RESEARCH_ARCHIVE_INDEX.md)退出；私有 Git 历史和科研产物留在受限归档，独立 Candidate B 克隆仍保留。根目录约 406 GiB 的 ignored 科研原件仍原地保存；其完整异盘备份尚未完成，不能据此清理磁盘空间。

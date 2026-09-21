# Target10 Unsupervised Parameter Selection (isolated experiment)

完全隔离的新实验：在 In-the-Wild 的 **10% 无标签子集**上做 target adaptation
parameter selection（禁止读标签），用选出的固定参数测试 **剩余 90%**，并与现有
source-select EP baseline 对比。

本实验不修改任何已有代码 / config / checkpoint / results，所有新增内容都在
`experiments/target10_selection/` 下（见「文件布局」）。分支：`exp-target10-selection`。

## 文件布局

```
experiments/target10_selection/
├── run_target10_experiment.sh    # 统一入口（Step 0–7，set -euo pipefail，支持 --dry-run）
├── manifests/
│   ├── inwild_target10.json      # 10% 原始审计 manifest（含 label/path/source_labels，仅供派生）
│   ├── inwild_target10_select.json  # 10% 无标签 select manifest（唯一用于参数搜索）
│   ├── inwild_target90.json      # 90% 子集（含 label，仅最终评价可读）
│   └── split_meta.json           # 划分元数据 + 类别计数
├── scripts/
│   ├── _common.py                # 私有共享模块（加载工件 / 评分 / 指标 / select manifest 校验）
│   ├── preflight.py              # Step 1：生成/验证 select manifest + 无标签硬检查 + 报告
│   ├── split_target10.py         # 一次性 10/90 划分派生参考（已冻结，管线不再调用）
│   ├── select_target10_params.py # Step 2：无标签参数搜索（--group N 多卡拆分，只读 select manifest）
│   ├── select_best_param.py      # Step 3：汇总分组结果 + 冻结 best_param.json
│   ├── eval_target90.py          # Step 4：固定参数在 target90 上评估
│   ├── eval_source_select_baseline.py # Step 5：source-select EP baseline
│   └── compare_results.py        # Step 6：生成 comparison.csv
├── configs/
│   └── target10_selection_eval.yaml  # selected_K/lr/steps（search_forbidden: true）
├── logs/
│   ├── run_target10_YYYYMMDD_HHMMSS.log  # 每次运行的完整日志
│   ├── git_status_before/after.txt       # 运行前后 git 快照
│   └── git_protected_before/after.txt    # src/ configs/ 保护路径快照
└── results/
    ├── preflight.json           # 无标签协议预检报告
    ├── search_group_0..3.json   # 各 GPU 分组的候选指标
    ├── best_param.json          # 冻结的最佳参数
    ├── param_search.json        # 全量候选指标（含熵/一致性/稳定性）
    ├── target90_result.json     # 选出的固定参数在 target90 上的指标
    ├── baseline_source_select.json  # source-select EP 在 target90 上的指标
    └── comparison.csv           # 两行对比表
```

## 关键解释（spec 与本代码库的对应）

1. **K ≡ steps**：本代码库的 EP-TTA 只有 `EPConfig(steps=…)` 一个步数参数，
   `docs/DESIGN.md` 明确写「K 步后对原输入视图评分」（K 即自适应步数）。
   因此候选网格为 `K ∈ {0,1,3,5,10} × lr ∈ {1e-5,5e-5,1e-4}`（K=0=frozen，
   lr/steps 对 K=0 无意义，故 K=0 只保留一个候选，共 13 组）。输出里 `K` 与
   `steps` 记录同一个整数。

2. **minDCF = null**：t-DCF / minDCF 需要 ASV 分数，本流水线不提供
   （与 `eptta evaluate` 的 `unavailable_metrics` 一致）。EER / accuracy / AUC
   均可从二分类分数+标签精确计算，照常输出。

3. **文件放置**：spec 里 `scripts/…`、`configs/…`、`results/…` 按「所有新增内容
   放入 `experiments/target10_selection/`」这条要求，统一收进本目录的对应子目录。

4. **复用只读工件**：特征缓存 `outputs_v2/ssl_aasist/cache-target-in_the_wild`、
   frozen bundle `outputs_v2/ssl_aasist/frozen/bundle.json`、source resources
   `outputs_v2/ssl_aasist/resources` 全部只读复用（脚本按 sample_id 直接从缓存
   取特征，绕开 CLI 对「缓存↔manifest 完全一致」的校验，因为 target10/target90
   都是该缓存对应全量 manifest 的子集）。

## 无标签选择协议（label-free protocol）

- **原始审计 manifest** `inwild_target10.json` 保留 `label` / `path` /
  `source_labels` / `role=target_test`，仅作为派生来源，**绝不用于参数搜索**。
- **无标签 select manifest** `inwild_target10_select.json` 由原始 manifest 派生
  （不重新划分、不改变任何 sample membership）。顶层 `role=select`、无
  `source_labels`；每条 record 严格是 `TargetInputManifest` 的 6 个字段
  （`schema_version/sample_id/sample_index/root_key/audio_relpath/split_role`），
  且 `split_role=select`，无任何 label 字段。
- **硬性预检（preflight）**：参数搜索开始前强制校验
  `role==select`、`count==len(records)==3178`、字段严格匹配 `TargetInputManifest`、
  无 label 字段、无 `source_labels`、所有 `split_role==select`、sample_id/sample_index
  与原始 manifest 完全一致；同时校验 target90（`role==target_test`、`count==28601`）、
  target10∩target90 为空、union==31779、checkpoint/resources/cache 路径、GPU 数量、
  参数组合数（13）。任一失败立即退出，不启动 GPU 搜索。
- 参数搜索**只读** `inwild_target10_select.json`，选择指标只有 entropy reduction /
  prediction consistency / confidence stability（无标签）；禁止 EER/accuracy/AUC/minDCF
  作为 selection criterion。target90 的 label 仅在 `best_param.json` 冻结后用于最终
  EER/minDCF/AUC/accuracy 评价。

## 选择指标（全部无标签）

对 target10 每个样本，跑一次 per-sample reset 的 EP（与 `run-tta` 同一条生产路径
`run_method("ep_tta", …)`），得到 frozen 原始视图分数与自适应后原始视图分数，以及
3 个 probe 视图的自适应分数，然后聚合：

- **entropy reduction**（熵减）：`mean(H(p_before) − H(p_after))`，二值熵，越大越好。
- **prediction consistency**（预测一致性）：3 个视图自适应后硬预测中，与多数票一致的
  比例（`majority/3`）的样本均值，越大越好。
- **confidence stability**（置信度稳定性）：`mean(1 − std(conf)/0.5)`，
  `conf = |sigmoid(view_score) − 0.5|`，越大越好。

选择规则：`argmax mean(entropy_reduction/ln2, consistency, stability)`，
并列时先取更小 K 再取更小 lr（确定性）。**全程不读 target10 的 label。**

## 复现命令

```bash
git checkout -b exp-target10-selection
conda activate tta

# 预检（不启动 GPU，只做所有校验并生成/验证 select manifest）
bash experiments/target10_selection/run_target10_experiment.sh --dry-run

# 一键运行全部 Step 0–7（日志自动写入 logs/run_target10_*.log）
bash experiments/target10_selection/run_target10_experiment.sh
```

或分步手动执行（脚本支持 `--limit N` 只跑前 N 个样本调试）：

```bash
cd experiments/target10_selection
python scripts/preflight.py            # 生成/验证 select manifest + 硬检查
CUDA_VISIBLE_DEVICES=0 python scripts/select_target10_params.py --group 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/select_target10_params.py --group 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/select_target10_params.py --group 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/select_target10_params.py --group 3 &
wait
python scripts/select_best_param.py
CUDA_VISIBLE_DEVICES=0 python scripts/eval_target90.py
python scripts/eval_source_select_baseline.py
python scripts/compare_results.py
```

## 结果摘要

- 选择阶段（target10，无标签）：13 个候选中选出 `K=0`（frozen）。
  由于候选 lr 仅为 1e-5～1e-4，自适应几乎不动分数，所有候选的三个无标签信号
  与 frozen 几乎一致，且自适应使熵略增（entropy_reduction 微小为负），因此
  `argmax` 规则确定性地选择了 K=0。
- target90 固定参数结果（见 `results/target90_result.json`）：
  `EER=0.09468`，`AUC=0.96766`，`accuracy=0.44509`（在 source cal0 阈值 tau0
  处计算；该阈值跨域偏移，故 operating-point accuracy 偏低），`minDCF=null`。
- baseline source-select EP 结果（见 `results/baseline_source_select.json`）：
  `EER=0.09474`，`AUC=0.96768`，`accuracy=0.44509`。
- 对比表见 `results/comparison.csv`：两者 EER 差异在第 5 位小数（~5.6e-5），
  基本等价；这是「10% 无标签选择 + 固定参数」与「source-select EP」在当前 lr
  网格下的诚实结果。

详细候选指标见 `results/param_search.json`（含 entropy / consistency / stability
与 selection_score）。


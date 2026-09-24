# Target10 Oracle Capacity Diagnosis

本实验是 development / diagnosis only，不能作为论文 final evaluation。
target10 已经查看过标签；本实验明确允许事后 label-aware 全局参数选择。
target90 不读取、不评分、不调参；没有自动运行 target90 的入口。

## 固定设计

- 生产 `run_method("ep_tta_guarded", TargetViews, resources, EPConfig, {})`，逐样本重置 R。
- K=steps ∈ {0,1,3,5,10}；lr ∈ {0.0003,0.001,0.003,0.01,0.03,0.1,0.3}；rho ∈ {0.05,0.1,0.2,0.4}。
- gamma=0.1，lambda_keep=1.0。K=0 仅一个 Frozen identity，lr/rho 输出 null（内部合法占位 .003/.2，零步不使用）。
- 共 113 候选，全部 3178 个 target10 样本，沿用既有 membership，不重划 target10/90。
- 不提供 unguarded 或 waveform 新路径。本路径沿用 NumPy→CPU torch 的生产 feature-cache EP。

## 隔离与输入验证

worker 仅通过既有严格无标签 select manifest 获取 ID，通过 TargetViews 传入单样本特征。
复用 `_common.evaluate_candidate` 的 view_reduction、probability_consistency、source_margin_retention、selection_score 与 guard 指标。
共享 helper 只增加可选 rho 与 score sink；原 selector 默认值、label-free 行为和原输出均不改变。
源锚点标签属于 source fit，允许用于既有保持目标；target 标签绝不进入 loss、gradient 或单样本候选选择。

preflight 检查 raw labelled manifest 存在；从无标签 manifest 检查 3178 个唯一 ID。
为满足“所有评分完成后才读标签”，raw labelled manifest 的 count=3178、dataset/role、标签类型与精确覆盖核验延后至聚合。
不调用原 target10 preflight（它会读取 target90）。只复用 label-free helper。
共享 in-the-wild cache 可以含非 target10 特征；读取块时仅保留 target10 特征，不以其余样本统计进行适应。
检查 pickle 禁用、shape/dtype/finite、路径包含关系、cache/bundle 普通来源字段、preprocess、精确样本覆盖。
不计算内容摘要，也不声称能够检测同名文件的外部原地替换。
历史 v1 cache 按生产 reader 规则只读复用，以普通 baseline_id 绑定已验证 frozen bundle；
存在的普通字段不符即失败，不使用历史摘要字段。报告明确记录 v1 缺少可核验的原始 preprocess 内容，
不能把这种兼容读取描述为 v2 完整来源验证。v2 严格检查来源、数据角色与 preprocess。

## 执行顺序

`preflight → 32-sample × 113-candidate engineering smoke → full sweep → exact coverage checks → labels → aggregate`。
smoke 运行所有候选，验证分组、参数、有限分数、无 fallback、Frozen identity；不计算质量指标。
各组进程只通过 sys.executable 启动，正式运行必须使用 tta conda 环境。
NUM_GROUPS 默认 4（分配 29/28/28/28），不暗示使用四张 GPU。
OMP_NUM_THREADS、MKL_NUM_THREADS、OPENBLAS_NUM_THREADS 默认 1，可由用户环境覆盖。
run directory 必须是本实验 results 的新直接子目录；不覆盖已有结果。失败保留日志/部分分数，换新 run-id 重跑。

## 统计与解释

使用生产 binary_metrics，0=bonafide、1=spoof，分数越高越 spoof。
balanced_accuracy/FPR/FNR 均用源 cal0 的 tau0；EER 的计算阈值不被用于部署或目标阈值优化。
oracle 按最低 EER，其后更高 AUC、更小 K、更小 lr、更小 rho 确定唯一全局候选。
delta_EER_vs_Frozen=EER_candidate−EER_frozen；gain=EER_frozen−EER_oracle，同时输出 absolute 与 pp。

原 guarded-v2 已完成 25 候选 ranking 时，验证其完整性及无标签声明，读取 selected 参数（rho=.2），
使用本轮同一 bundle 的重算分数报告该参数的 EER；不混入历史 target90 评价。
旧 ranking 文件缺失则 unavailable；不完整/字段不符则失败。
额外在新 113 候选上复用相同 proxy 排序，完全相同 proxy 后以 K/lr/rho 递增打破平局。
两种选择分别命名、分别报告 regret；不得把扩大网格的选择假称为原历史选择。

非 Frozen 的 112 个候选上计算 Spearman(signal,-EER)。项目已有 SciPy；这里使用轻量 average-rank + Pearson，
测试与 scipy.stats.spearmanr 对照；常量序列返回 null/undefined，不添加依赖。

5-fold：排序明确 sample ID，再按 label 分层，各类别使用独立于全局状态的 random.Random(2026) 顺序 shuffle，
round-robin 分配至 5 folds；保存每个 ID 的 fold 与 held-out 分数。
每折仅使用另外四折标签计算 EER/AUC 并选择一个全局候选；该候选评分当前折。
拼接全部 held-out 分数后再计算 cross_validated_oracle_EER/AUC，不对 fold EER 简单平均。
这是 sample-level confirmation，不是独立 group-held-out 验证；仍为 diagnosis。

report 自动给 A/B/C 描述性提示，‘接近’明确固定为 0.001 EER（0.1 pp），不是显著性检验。
辅助输出 full gain、CV gain、regret 与相关性，不宣称所有 TTA 无效或论文结论成立。

## 输出与命令

`results/<run_id>/scores/*.jsonl` 保存每候选 sample_id/score/score_before；不保存模型。
`analysis/oracle_surface.csv` 保存完整 113 行参数面；`oracle_5fold.csv` 保存逐折选择；
`oracle_5fold_predictions.csv` 保存 assignment 和拼接分数；`summary.json` 与 `report.md` 汇总结果。
结果与日志默认不进入 Git。

```bash
conda activate tta
bash experiments/oracle_diagnosis/run_oracle_diagnosis.sh --dry-run
NUM_GROUPS=4 bash experiments/oracle_diagnosis/run_oracle_diagnosis.sh
# 可选 --run-id oracle_YYYYMMDD_run1；TTA_ASSET_ROOT 指向已有 outputs_v2 的项目根目录。
python -m pytest tests/unit/test_oracle_diagnosis.py tests/unit/test_target10_selection_audit.py -q
```

本机工程验证记录见 VALIDATION.md；正式工作站 smoke、sweep、科学结果均须由实际运行产生。

# Target10-Selection 系统诊断报告

- 项目根目录：`/media/dell/data/fakeAudioDection/TTA`
- conda 环境：`tta`（Python 3.10.20，torch 2.1.0+cu121）
- 开始 git HEAD：`d15fc48d2030775cf23ecc2c34c8c2e42949b121`
- 开始 `git status --short`：`?? experiments/target10_selection/archive/`（唯一既有未跟踪项）
- 本报告所有新增产物均在 `experiments/target10_selection/diagnostics/` 下；**未修改** `src/`、`configs/`、`data/manifests_v2/`、任何 checkpoint、已有 `outputs_v2` 或任何正式 target10 结果文件。
- 本报告对应历史 `target10-v0` 13-candidate 协议。仓库仅保留本报告、汇总 JSON 和可复现脚本；大体积逐样本 CSV / post-hoc JSON 不入库，可按附录中的脚本重新生成。

---

## 1. Executive summary

对已完成的 target10 无标签参数选择做了一次**只读诊断**（未改任何核心代码 / 正式结果）。结论可浓缩为：

1. **K=0 严格等于 Frozen**：`FROZEN_K0_EQUIVALENCE = PASS`。纯 frozen 推理与 `EPConfig(steps=0)` 在 target90 全量 28601 样本上逐样本分数差 `max=0.0`，EER 完全一致（`0.09468049514887922`）。
2. **EP 确实在更新参数，但数值上几乎不影响预测**：梯度非零（mean grad norm ≈ 0.098，max ≈ 1.25），R 的 Frobenius 范数随 `K×lr` 线性增长；但投影预算 `rho=0.2` 从未被触发（`projection_applied=false`），score 变化最大仅 ~1e-3（搜索空间内），500 样本上 hard flip rate **恰好为 0**。
3. **consistency 恒定是「硬标签」指标的必然结果**：consistency 是 3 个 probe view 的多数票一致率（每样本只可能取 2/3 或 1）。全量 3178 样本的 frozen consistency = `0.9544787077826`，正是 param_search 里的常数 `0.9544787078`；adaptation 未使任何 view 的硬预测越过 0.5 边界，故 13 个候选一致。
4. **post-hoc 真实 EER 在 13 个候选上完全相同**：target10 的 true EER 恒为 `0.09857072449482504`（13 个候选一致），AUC 只在小数点后第 5–6 位波动，tau0 处 accuracy 完全一致。因此 **unsupervised selection metric 与真实 EER 的 Spearman 相关性不存在（零方差，未定义）**——不是「选错」，而是「没有可预测的性能差异」。
5. **K×lr 主宰一切无标签信号**：`effective_strength = K×lr` 与 `selection_score / entropy / stability` 的 Spearman 均为 **-0.9959**（近完美单调）。`K=1,lr=5e-5` 与 `K=5,lr=1e-5` 等「同 K×lr」组合产出几乎逐位一致的结果，证实当前 EP 行为 ≈ 由总更新强度 `K×lr` 控制。

**总体诊断：MIXED（偏「EP 弱更新 + selection metric 无区分度」），无 negative adaptation 证据。**

---

## 2. Frozen vs K=0 equivalence

- 数据：target90 全量 28601 样本，同一 checkpoint / features / 预处理 / score 定义。
- 对比路径：
  - A. 纯 frozen：`run_method("frozen", ...)`
  - B. `EPConfig(steps=0)`：`run_method("ep_tta", steps=0, ...)`
- 逐样本 CSV：`diagnostics/results/frozen_vs_k0.csv`（`sample_id, frozen_score, k0_score, abs_difference`）

| 指标 | 值 |
|---|---|
| sample count | 28601 |
| mean abs score diff | 0.0 |
| median abs score diff | 0.0 |
| max abs score diff | 0.0 |
| prediction disagreement count (tau0) | 0 |
| frozen EER | 0.09468049514887922 |
| K=0 EER | 0.09468049514887922 |
| frozen AUC | 0.9676644535269782 |
| K=0 AUC | 0.9676644535269782 |

**判定：`FROZEN_K0_EQUIVALENCE = PASS`**（判据 `max_abs_score_diff < 1e-7`，实测 0.0）。

原因（读码确认）：`run_cache_method("ep_tta")` 在 `cfg.steps==0` 时循环不执行，`R=zeros`，且 `score = before if cfg.steps == 0 else ...`，其中 `before = Z[0] @ w + b` 与 `run_method("frozen")` 的 `before` 是同一表达式。因此两者逐位相同，且 `0.09468` 与 `results/target90_result.json` 的 EER 精确吻合。

---

## 3. EP update effectiveness

在固定 500 样本子集（`seed=2026`，ID 见 `diagnostic_sample_ids.json`）上用**忠实复刻的 instrumented loop**（与 `run_method("ep_tta")` 的 max 绝对分数差 = 0.0，验证通过）测量 5 个配置。

requires_grad 与 optimizer 事实：

- **唯一 `requires_grad=True` 的张量**是每样本新鲜初始化的 R 矩阵（8×8，64 个元素）。`U / w / anchors_*` 全部 detached、`requires_grad=False`。
- **没有 `torch.optim.Optimizer`**：生产路径是手写 SGD `parameter.add_(gradient, alpha=-lr)`，无 momentum / weight decay，随后 `project_frobenius_(R, rho=0.2)`。

| 配置 | mean grad norm | update norm (mean) | mean abs score delta | flip rate |
|---|---|---|---|---|
| A: K=0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B: K=1, lr=1e-5 | 0.09828 | 9.83e-7 | 3.32e-6 | 0.0 |
| C: K=5, lr=1e-4 | 0.09827 | 4.91e-5 | 1.61e-4 | 0.0 |
| D: K=10, lr=1e-4 | 0.09825 | 9.83e-5 | 3.22e-4 | 0.0 |
| E: K=1, lr=0.003 (source-select) | 0.09828 | 2.95e-4 | 9.66e-4 | 0.0 |

关键观察：

- **梯度非零且基本不随步数变化**：`view_variance` 目标给出真实梯度（mean≈0.098，median≈0.0106，max≈1.25，强右偏）。margin keep 正则项梯度 ≈ 0（见下）。
- **update norm ≈ lr × grad_norm × K**：`9.83e-7 = 1e-5×0.098×1`；`4.91e-5 = 1e-4×0.098×5`；`9.83e-5 = 1e-4×0.098×10`；`2.95e-4 = 0.003×0.098×1`。说明每步是干净的 SGD 一次更新。
- **投影从未激活**：所有配置 `projection_applied=false`，R 的范数（最大 ~2e-3，相对 rho 预算仅 ~0.5%）远小于 `rho=0.2`。
- **margin keep 正则项 ≈ 0**：`mean_keep_before=0`，`mean_keep_after` 为 `3e-10 ~ 3e-7`。即 source anchors 在冻结点已满足 margin（`anchors_m0` 全为正），keep 正则不贡献梯度。因此 EP 目标实际退化为单纯的 `view_variance` 最小化。

**判定：`EP_EFFECTIVE_UPDATE = WEAK`**。参数层面确有更新（grad 非零、R 随 K×lr 线性增长），但更新幅度远小于投影预算，且 score/prediction 层面影响可忽略（见第 4 节）。不是「无更新」，而是「更新幅度不足以影响预测」。

---

## 4. score / logit change

本工程是**单 logit 头**：`score == logit == Z[0]@w+b`，sigmoid 后为概率。因此「score 变化」与「logit 变化」是同一物理量。

| 配置 | mean abs score(=logit) delta | max abs score delta | mean abs prob delta |
|---|---|---|---|
| D: K=10, lr=1e-4 | 3.22e-4 | 8.01e-3 | 1.03e-5 |
| E: K=1, lr=0.003 | 9.66e-4 | 2.40e-2 | 3.09e-5 |

经验「score 灵敏度」≈ `mean_abs_score_delta / R_norm ≈ 3.28`（D、E 一致）。即 score 变化 ≈ 3.28 × ‖R‖。要产生 O(1) 的 score 变化需要 ‖R‖ ≈ 0.3，已超过 `rho=0.2` 预算；而当前搜索空间的最大 ‖R‖ 仅 ~1e-4 量级。这解释了为何 score 几乎不动、hard prediction 不翻转。

---

## 5. consistency metric diagnosis

**数学定义**（来自 `_common.evaluate_candidate`，生产路径）：

```
view_scores = apply_adapter(z, U, R) @ w + b     # 3 个 probe view
view_probs  = sigmoid(view_scores)
votes       = [1 if p > 0.5 else 0]
majority    = max(#votes==0, #votes==1)
consistency_sample = majority / 3                # 每样本只可能 2/3 或 1
consistency        = mean over samples           # 硬标签多数票一致率
```

回答 6 个问题：

1. **consistency 的数学定义**：3 个 view 的硬标签多数票一致比例（`majority/3`）。
2. **比较哪些 view**：自适应后的 3 个 probe view（`apply_adapter` 后的 3 路 logits）。
3. **是否 hard-label agreement**：是。只对 `p>0.5` 的符号敏感，对概率大小不敏感。
4. **为什么 13 候选完全相同**：见下。
5. **logits 明明变化、hard prediction 是否没 flip**：是。score/logit 变化 ~1e-4，远小于使 sigmoid 越过 0.5 所需的幅度，因此任何 view 的硬标签都不翻转。
6. **soft consistency（仅诊断）**：见下表，所有配置的 soft 指标在 4–5 位小数上基本不变（本质上是 frozen 三视图的属性）。

| 指标（500 样本） | A: K=0 | D: K=10,lr=1e-4 | E: K=1,lr=0.003 |
|---|---|---|---|
| hard consistency mean | 0.96 | 0.96 | 0.96 |
| flip rate (view vs frozen) | 0.0 | 0.0 | 0.0 |
| soft mean abs prob diff | 0.084573 | 0.084570 | 0.084564 |
| soft mean pairwise KL | 0.225228 | 0.225057 | 0.224716 |
| soft mean pairwise JS | 0.037751 | 0.037741 | 0.037721 |
| soft mean pairwise cosine | 0.944768 | 0.944803 | 0.944871 |

**consistency 恒定原因**：`frozen_consistency_full_3178 = 0.9544787077826209` 与 param_search 常数 `0.9544787077826683` 只差 5e-14（浮点求和顺序）。也就是说 **0.9544787 就是 frozen 三视图的硬标签一致率**；由于 adaptation 未让任何 view 翻转，13 个候选的 consistency 都等于 frozen 值，故恒等。soft 指标（prob diff ≈ 0.085、JS ≈ 0.038）虽非零，但也几乎不随候选变化，进一步说明 adaptation 对「视图间关系」的扰动极小。

---

## 6. 13 candidate post-hoc real performance

POST-HOC ONLY：读取 `inwild_target10.json` 的 label 仅用于事后评判，**未参与原 selection**。全部 13 候选 × 3178 样本逐样本分数见 `posthoc_group_*.json`，汇总见 `posthoc_candidate_metrics.csv`。

| K | lr | steps | K×lr | selection_score | true_EER | true_AUC | true_accuracy(tau0) |
|---|---|---|---|---|---|---|---|
| 0 | — | 0 | 0 | 0.6359394 | 0.09857072 | 0.9633088 | 0.4367527 |
| 1 | 1e-5 | 1 | 1e-5 | 0.6359393 | 0.09857072 | 0.9633090 | 0.4367527 |
| 1 | 5e-5 | 1 | 5e-5 | 0.6359388 | 0.09857072 | 0.9633088 | 0.4367527 |
| 1 | 1e-4 | 1 | 1e-4 | 0.6359382 | 0.09857072 | 0.9633084 | 0.4367527 |
| 3 | 1e-5 | 3 | 3e-5 | 0.6359391 | 0.09857072 | 0.9633088 | 0.4367527 |
| 3 | 5e-5 | 3 | 1.5e-4 | 0.6359375 | 0.09857072 | 0.9633096 | 0.4367527 |
| 3 | 1e-4 | 3 | 3e-4 | 0.6359356 | 0.09857072 | 0.9633135 | 0.4367527 |
| 5 | 1e-5 | 5 | 5e-5 | 0.6359388 | 0.09857072 | 0.9633088 | 0.4367527 |
| 5 | 5e-5 | 5 | 2.5e-4 | 0.6359362 | 0.09857072 | 0.9633126 | 0.4367527 |
| 5 | 1e-4 | 5 | 5e-4 | 0.6359331 | 0.09857072 | 0.9633156 | 0.4367527 |
| 10 | 1e-5 | 10 | 1e-4 | 0.6359382 | 0.09857072 | 0.9633084 | 0.4367527 |
| 10 | 5e-5 | 10 | 5e-4 | 0.6359331 | 0.09857072 | 0.9633156 | 0.4367527 |
| 10 | 1e-4 | 10 | 1e-3 | 0.6359267 | 0.09857072 | 0.9633229 | 0.4367527 |

**关键事实：`true_EER` 在 13 个候选上完全一致（0.09857072449482504），AUC 只在小数点第 5–6 位变化，tau0 处 accuracy 完全一致。**

（注：target10 的 frozen EER=0.09857 略高于 target90 的 0.09468，属 10%/90% 子集统计差异，与 adaptation 无关。）

---

## 7. unsupervised metric vs EER correlation

| 相关性（Spearman，vs −EER，越大越好） | 值 |
|---|---|
| selection_score vs −EER | **未定义（null）** |
| entropy vs −EER | **未定义（null）** |
| consistency vs −EER | **未定义（null）** |
| stability vs −EER | **未定义（null）** |

原因：`true_EER` 在 13 个候选上**零方差**，Spearman/Pearson 秩相关无定义。因此不能下结论说「selection metric 与真实 EER 相关/不相关」——**当前搜索空间内根本没有可预测的 EER 差异**。这是比「相关弱/反向」更强的结论：metric 的区分度问题被「真实性能本身不变」所掩盖，二者都无法互相验证。

（Pearson 同理为 null；`posthoc_correlation.json` 记录了全部 null 结果。）

---

## 8. K×lr effective-strength analysis

`effective_strength = K × lr`（K=0 记 0）。见 `effective_strength_analysis.csv`。

| Spearman（13 候选） | 值 |
|---|---|
| effective_strength vs selection_score | **-0.99587** |
| effective_strength vs entropy | **-0.99587** |
| effective_strength vs stability | **-0.99587** |
| effective_strength vs true_EER | **未定义（EER 恒定）** |

同 `K×lr` 组合几乎逐位一致：

- `K=1, lr=5e-5`（es=5e-5）vs `K=5, lr=1e-5`（es=5e-5）：
  - selection_score `0.635938796116201` vs `0.635938796124967`
  - entropy `-1.1651179565598005e-06` vs `-1.1651011629332726e-06`
- `K=1, lr=1e-4`（es=1e-4）vs `K=10, lr=1e-5`（es=1e-4）：
  - selection_score `0.6359381589010874` vs `0.6359381589337746`

**结论：当前 EP 行为 ≈ 仅由总更新强度 `K×lr` 决定**（`update norm ≈ K×lr×grad_norm`，score ≈ 3.28×‖R‖），与用户观察 3、4 完全一致。

---

## 9. Diagnosis：A / B / C / mixed

用实测数字逐条对照三种情形：

- **Case A（EP 几乎不改变参数/预测）**：grad ≈ 0.098（**非极小**），但 update norm ≈ 1e-4、score delta ≈ 1e-4~1e-3、flip=0。→ **部分命中**：参数有更新但预测无变化。归因不是「无梯度」，而是「lr 网格太小，从未触及 rho 预算」+「score 对 R 的灵敏度 ≈ 3.28，需要 ‖R‖≈0.3 才产生 O(1) score 变化」。
- **Case B（EP 明显改变预测 + metric 相关性弱/反向）**：预测**未**明显改变（flip=0），且 EER 零方差导致相关性无定义。→ **部分命中**：metric 确无区分度，但根因是「无性能差异可预测」而非「metric 判反」。
- **Case C（真实 negative adaptation，selector 正确选 frozen）**：`true_EER` 13 候选完全一致，K>0 并不差于 K=0。→ **不成立**，无 negative adaptation 证据。

**最终分类：MIXED。** 主因是「**EP 更新在参数层真实发生、在预测层数值可忽略**」叠加「**consistency 为硬标签指标、无区分度**」。既不是纯 Case A（有非零梯度与更新），也不是 Case B（无预测变化、无相关性可谈），更不是 Case C（无 negative adaptation）。

---

## 10. 当前结果能说明什么

1. K=0 与 frozen 在实现上、数值上严格等价，target90 EER=0.09468 就是 frozen 结果。
2. 当前 `rho=0.2` 的投影预算在搜索网格下**从未被触及**，EP 实际在「几乎不动的线性区」运行。
3. margin keep 正则项（source anchors 已满足 margin）在当前模型上 ≈ 0，EP 目标退化为纯 `view_variance`。
4. consistency 是硬标签多数票一致率，`0.9544787078` 即 frozen 三视图硬一致率；只要不翻转，13 候选必然恒等。
5. 无标签 selection_score 在 13 候选上只是 `K×lr` 的单调函数（Spearman -0.9959），未承载任何目标域性能信息。
6. 真实 EER 在 13 候选上完全一致，说明在该 lr/steps 网格下 adaptation 对 In-the-Wild 二分类性能无任何可测影响。

## 11. 当前结果不能说明什么

1. **不能**说「EP-TTA 整体无效」——只测了 `lr∈{1e-5,5e-5,1e-4}`、`K≤10`、`rho=0.2`、view_variance+margin 这一组，未覆盖更大 lr、更多步、不同 objective/正则、不同 rho。
2. **不能**说「In-the-Wild 上存在/不存在 negative adaptation」——搜索空间内 K>0 与 K=0 的 EER 相同，没有产生任何 adaptation 信号。
3. **不能**据此对 target-aware selection 做最终判死刑——因为候选空间内的真实性能没有差异，任何 selection（含 source-select）都无法被区分。
4. **不能**把 target10 的 post-hoc label 结论当作「新的 unlabeled 结果」；这些 label 只用于事后诊断，且自此 target10 应视为 development set（见 §13）。

## 12. 是否值得继续 target-aware selection

**值得继续，但必须先修搜索空间。** 当前失败不是「选择域」问题，而是「候选空间太弱、EP 根本没动」的问题：`K×lr` 上限 1e-3，距离触发投影（约需 `K×lr≈2`）差 ~3 个数量级。在 EP 无法改变预测的前提下，任何无标签选择指标都必然失效。因此：

- 先做 **matched search space** 的 source-select vs target10-select 对照（见 §9 公平性与 §13）。
- 在 EP 能产生可测预测变化（flip rate > 0）的候选空间内，再检验 consistency/entropy/stability 是否与真实 EER 相关。

## 13. 下一轮实验建议（NEXT EXPERIMENT PROPOSAL）

不修改本次正式结果，仅提出后续方向：

1. **扩大/对齐搜索空间**：把 lr 上界提高到能触达 `rho=0.2` 的量级（例如 `lr∈{1e-3,3e-3,1e-2,3e-2}`，K 覆盖到 `K×lr` 达到 O(1)），并确认 `projection_applied` 确实发生。
2. **MATCHED SEARCH SPACE 实验**：用完全相同的一组 candidate，分别由 source validation 与 target10 unlabeled 进行 selection，严格归因「selection domain」而非「search space」。
3. **换/加 soft selection metric**：既然 consistency 是硬标签指标，补 soft 版本（JS/KL/prob-spread）作为诊断对照（**不能自动替换正式准则**）。
4. **检查 objective/正则**：margin keep 当前 ≈0，可考虑 `view_variance` 之外的目标或让 keep 真正激活（例如更紧的 margin / 不同 anchors）。
5. **协议声明**：一旦用 target10 的 label 做任何方法选择，target10 即为 development set；论文需另立**独立、未参与开发**的协议数据。

## 14. 哪些结果可以进入论文

- K=0 == Frozen 的严格数值等价性验证（方法学小节）。
- EP 在 `rho` 预算下的「线性区」行为：`update norm ≈ K×lr×‖grad‖`、`score ≈ 3.28×‖R‖`、投影未触发。
- consistency 作为 hard-label agreement 指标的定义及其在「无翻转」场景下必然恒定的数学解释。
- target10 与 target90 的 frozen EER（0.09857 / 0.09468）及 source-select baseline 的等价性（差异 ~5.6e-5）。
- 无标签 metric 在「性能零方差」下无法验证的方法论结论（可作为「selection 需先在可测区运行」的论证）。

## 15. 哪些结果只能作为 diagnostic

- 全部基于 target10 label 的 post-hoc 分析（`posthoc_*`、true_EER/AUC/accuracy、相关性）——label 未参与原选择，只能事后诊断，**不得**称为新的 unlabeled selection 结果。
- 500 样本的 grad/update/score/flip 测量——用于定位实现行为，非正式评测。
- soft consistency（JS/KL/prob diff）——诊断对照，非正式 selection criterion。
- `effective_strength` 相关性——解释性分析，非新的选择规则。

---

## 附：输出文件清单

- 报告：`diagnostics/reports/diagnostic_report.md`
- 汇总：`diagnostics/results/summary.json`
- 脚本：`diagnostics/scripts/{_diag_common,diag1_frozen_vs_k0,diag2_adaptation,diag3_consistency,diag4_posthoc,diag4_aggregate,generate_summary}.py`
- 可重建结果（不入库）：`diagnostics/results/{frozen_vs_k0.csv/json, diagnostic_sample_ids.json, adaptation_diagnostics.csv/json, consistency_diagnostics.json, posthoc_group_0..3.json, posthoc_candidate_metrics.csv, posthoc_rank_comparison.csv, effective_strength_analysis.csv, posthoc_correlation.json}`

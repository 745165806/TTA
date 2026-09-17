# EP-TTA v0.1.0 — AASIST R6 源域真实先导报告

日期：2026-09-17
结论：**PASS（工程接线可靠），但科学结果为负：完整 EP 未能在 source select 上优于 Frozen。** K>0 优化/reset/保护接线通过独立 FP64 参考验证；真实 select 全量结果显示所有 EP 候选与 Frozen EER 相同，源域基线接近饱和。

## 0. 批准与身份

- proposal SHA-256 `1d1ad752041c5e8930f5e251cf4e473624927de57b651ce9913a0cc95e0c9e25`；用户批准 LOCK 并执行。
- 底座 bundle `d774f714…`/checkpoint `076ca355…`、resources `ep-resources-9133770434dc343e7060`（resources.json `a7d9bc3b…`）、select cache `features-7069e243…`、select labels `07e085f2…`。
- 设备 CPU float32，独立参考 FP64；29 个方法（2 底线 + 9 EP 网格 + 15 机制 + 3 random U），全部在 select 11520 条封存后独立评价。
- 测试 `pytest -q` → **272 passed**；未训练/SSL/目标评分/commit。

## 1. K>0 工程验证（第一段，全部通过）

| 检查 | 结果 |
|---|---|
| FP64 参考梯度 vs 生产梯度（R=0 首步） | 最大绝对误差 `1.25e-7`（FP32 下 PASS） |
| K=1/3/5 确实更新 R（非 K=0） | steps_completed=1/3/5，r_fro `0.000091/0.000274/0.000456` |
| 每条独立 R=0、A→B→A 复位 | 一致，无跨样本状态 |
| `run_batch` `loss_b.sum()` vs 串行 | 最大分数差 `0.0` |
| 源 U/M/tau0/w/b 不可修改 | 无 requires_grad，resources hash 未变 |

结论：K>0 优化、reset 与保护接线**可靠**；但真实更新量极小（r_fro ~1e-4，远小于 rho=0.2）。

## 2. 主结果（source select=11520，tau0=-3.8384 固定）

| 方法 | EER% | AUROC | FPR | FNR | ΔEER(pp) |
|---|---|---|---|---|---|
| **Frozen** | **0.7937** | 0.999357 | 0.048611 | 0.000095 | 0.0000 |
| ep_tta（9 候选 K∈{1,3,5}×η∈{0.003,0.01,0.03}） | 0.7937 | 0.999355–0.999357 | 0.048611 | 0.000095 | 0.0000 |
| ep_no_keep / ep_keep_l2/logit/fisher / ep_feature_pca_U / ep_random_U(0,1,2) / static_subspace / source_ce_only | 0.7937 | 0.999357 | 0.048611 | 0.000095 | 0.0000 |
| fixed_source_adapter | 0.7937 | **0.999367** | 0.046627 | 0.000095 | 0.0000（+2 有益翻转） |
| ep_scalar_adaptive | 0.7937 | 0.999345 | 0.050595 | 0.000095 | 0.0000（2 有害翻转） |
| memo_same_adapter(±keep) | 0.8276 | 0.999356 | 0.045635 | 0.000095 | +0.0340 |
| entropy_same_adapter(±keep) | 0.8371 | 0.999355 | 0.045635 | 0.000095 | +0.0435 |
| multiview_mean | 0.8929 | 0.999278 | 0.048611 | 0.000190 | +0.0992 |

主选择规则结果：**NO_FEASIBLE_CANDIDATE**（最佳 EP = Frozen EER 到 4 位小数，ΔEER=0.0；源域基线 AUROC 0.99936 接近饱和）。published 5 项为 NOT_RUN（CONTRACT_ONLY_BLOCKED_AUDIT）。

## 3. 诊断（更新有没有改变证据，保护有没有起作用）

- **更新量**：ep_tta 平均 `r_fro≈7.8e-5`(K1)→`2.3e-4`(K3)→`7.5e-4`(K5)，平均 `|Δscore|≈9.5e-4`(K1)→`9.4e-3`(K5)。远小于 rho=0.2；三视图（identity/noise30dB/FIR0.05）嵌入几乎相同，view-variance 目标几乎没有可优化量。
- **保持项激活率**：ep_tta 全部 11520 条 `keep_activated=0`（margin 锚点 m0∈[5.04,24.73] 远离边界，微小更新从不触发 margin deficit）。故“保持项是否起作用”在本先导中**无法被激活**，不能据此断言保持机制有效或无效——它只是未被触发。K=3/5 下仍零激活，说明不是 K=1 的初值现象。
- **源侧几何**：`‖Uᵀw‖/‖w‖ = 0.865`（w 范数 1.597），响应子空间确实捕获了头权重的大部分方向；几何不阻止分数改变，但更新量太小。
- **投影触发率**：ep_tta 均 0（r_fro 远小于 rho）；ep_scalar_adaptive 到 rho=0.2 边界（17 点网格最大量）。
- **数值回退**：全部候选 fallback=0（无非有限损失/坏样本污染）。

## 4. 结论与四个问题

1. **K>0 优化/reset/保护接线可靠**：是。FP64 参考、A→B→A、batch=serial、无跨样本状态、投影/有限性均通过。
2. **完整 EP 是否优于 Frozen**：**否**。9 候选 EER 与 Frozen 完全相同，无任何改善（也无明显劣化）；幅度≈0、风险≈0、成本≈0.1–0.5ms/sample 的 cache-only 适配。
3. **U/动态更新/保持项的直接支持**：动态更新是真实发生的（非 K=0），但量级太小（r_fro~1e-4）；保持项从未激活，因此本先导既不能支持也不能否定其作用；U 几何（0.865）不构成分数改变的限制。
4. **小/零提升的原因**：主要是**source select 基线接近饱和**（AUROC 0.99936，EER 0.79%，FNR 仅 0.0095%）——正常声学变化本身已几乎不破坏源域判定；同时 deterministic 三视图扰动过弱使 view-variance 目标几乎无信号。这是实测判断，不是猜测。

## 5. 成本

- 全量评分 + 封存 + 评价：`run-r6` 墙钟 **867.9s**（单线程 CPU，≈0.24 CPU-core-hour），29 方法 × 11520 样本。
- cache-only 适配约 0.03–0.15ms/sample（EP K=1/5）；无峰值显存（CPU 适配，无重编码）。
- 本轮新增工件：select 评分/诊断/评价/CSV ≈ 数 MB，远小于 2.0 GiB；GPU 未使用（0 GPUh，<1.0）。
- R5 全量编码计时（extract ≈ 0.01s/sample）仅用于端到端 ESTIMATED，非本轮实测适配成本。

## 6. 后续（最多一项范围明确的源域诊断，不自动启动）

负结果下唯一建议的源域诊断：**增大视图扰动幅度**（当前 noise 30dB/FIR 0.05 过弱，view-variance 无信号）以观察更新量是否可达到可测分数变化——这属于 probe 参数的源域诊断，不涉及弱化底座/重选 epoch/改 tau0/M/U 或无界搜索；需单独 proposal 与批准。多训练 seed/其他底座/published ports/外部目标均不启动。

历史不可覆盖，版本保持 0.1.0，未 commit/push/tag。

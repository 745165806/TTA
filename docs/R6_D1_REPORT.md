# EP-TTA v0.1.0 — AASIST R6-D1 更新尺度与源域受控偏移诊断报告

日期：2026-09-17
结论：**NO_GAIN_WITHIN_TESTED_DIAGNOSTIC。方差目标被有效优化（view_loss 下降 ~21%），但无论原输入还是 +20dB 受控偏移输入，检测收益均未出现、且大步长反而轻微恶化。根因不是“更新太小”也不是“没有偏移”，而是方差目标的下降方向与检测纠错方向不一致。**

## 0. 身份与语义审计

- proposal `4cb27b8a…dcd2`、condition `203dcba2…a6a1`；用户以“没问题”批准并 LOCK，GPU 免审批直接执行。
- 底座 bundle `d774f714…`/checkpoint `076ca355…`/resources `ep-resources-9133770434dc343e7060`；select 11520。
- `RESULT_SEMANTICS_AUDIT.json`（派生工件，未覆写 R6 29 方法封存评分）：
  - Frozen 精确值：EER ratio `0.007936507936507936` = **0.7936507936507936%**；AUROC `0.9993574048404242`；tau0=-3.8384 下 tp=10511/fp=49/tn=959/fn=1（bonafide 1008、spoof 10512）。
  - FNR=1/10512=9.51e-5（tau0 为 cal0 alpha=0.05 分位数，目标是 FPR≈5%）；FPR=49/1008=4.86%。FNR 微小不证明“无提升空间”——真正空间是 49 个被误拒的真人（FPR 4.86%）与 EER 0.79%。
  - 语义修正：原 `NO_FEASIBLE_CANDIDATE` 是语义误标（LOCK 的 hard gate 只要求 coverage=1、fallback≤1%，从未要求改善）；更正为 **FEASIBLE_BUT_NO_GAIN**，未重选、未改历史。

## 1. 实验 A：仅变步长（C0，K=5，eta∈{0.03,0.3,3.0}）

| 方法 | EER% | AUROC | fallback | max R_fro |
|---|---|---|---|---|
| Frozen | 0.7937 | 0.999357 | 0 | — |
| multiview_mean | 0.8929 | 0.999278 | 0 | — |
| ep_tta / ep_no_keep eta=0.03 | 0.7937 | 0.999355 | 0 | 4.80e-2 |
| ep_tta eta=0.30 | 0.8276 | 0.999322 | 0 | 2.00e-1 |
| ep_no_keep eta=0.30 | 0.8276 | 0.999319 | 0 | 2.00e-1 |
| ep_tta eta=3.00 | 0.8752 | 0.999346 | 0 | 2.00e-1 |
| ep_no_keep eta=3.00 | 0.8929 | 0.999147 | 0 | 2.00e-1 |

逐步轨迹（固定 32 样本）：step1 梯度范数 ≈3.36e-2（与 eta 无关）；step5 R 范数随 eta 增大 5.02e-3→4.78e-2→1.45e-1；view_loss 从 0.0183 降到 0.0180/0.0155/0.0144。**方差目标确实被优化，但 EER 在 eta≥0.3 反而上升**（0.7937→0.8276→0.8752%）。投影后有界（≤rho=0.2），无 fallback。

## 2. 实验 B：+20dB 受控偏移（C20）

C20 派生 cache/parity 均 PASS（重编码读回 embedding/head logits/score 最大绝对误差 0.0、roundtrip 逐位一致），condition 独立命名空间、parent UID 完整、11520 全成对覆盖。

| 方法 | C20 EER% | C20 AUROC | fallback | max R_fro |
|---|---|---|---|---|
| Frozen | **4.2659** | **0.992597** | 0 | — |
| multiview_mean | 4.4140 | 0.992269 | 0 | — |
| ep_tta / ep_no_keep eta=0.03 | 4.2659 | 0.992597 | 0 | 2.27e-3 |
| ep_tta eta=0.30 | 4.2713 | 0.992593 | 0 | 2.25e-2 |
| ep_tta eta=3.00 | 4.3094 | 0.992556 | 0 | 2.00e-1 |

C20 确实制造了可辨别偏移（Frozen 从 0.79% 退化到 4.27% EER，AUROC 0.9994→0.9926）。但 EP 在所有步长下均未改善：eta=0.03 持平，eta≥0.3 轻微恶化。注意 C20 上更新量反而**更小**（eta=0.03 的 max R_fro 2.27e-3 vs C0 的 4.80e-2），因为 20dB 噪声已使三视图差异饱和，view-variance 信号更弱而非更强。

## 3. 实验 C：保护可达性与纠错方向

- 保持项可达性：256/256 锚点在 rho=0.2 球内**均可**触发（c_norm=1.3811，min threshold/bound=0.189，即最小触发需 ‖R‖_F≥0.0378）。
- 但 eta=0.03 的实际 ‖R‖_F≈5e-3（C0）/2.3e-3（C20），远小于触发阈值；eta≥0.3 时 R 可达 0.2 但此时 view_loss 下降方向反而使 EER 上升。**保持项在此诊断中既未在小区间触发、也未被证明有效或无效**——它只是没有参与已观测的（有害）更新。
- 方向诊断：方差目标下降（view_loss −21%@eta3.0）主要放大已有错误（EER 上升、AUROC 下降），未带来有益翻转。

## 4. 四个结论（仅按证据）

1. **大步长后纠错改善？** 否。eta≥0.3 时 EER 反而上升（C0 0.79→0.83→0.88%）。
2. **C20 有稳定收益且不被多视图平均解释？** 否。C20 上 EP 无任何稳定收益（多视图平均更差）。
3. **更新与目标下降明显但无检测收益/更差？** **是 → NO_GAIN_WITHIN_TESTED_DIAGNOSTIC**。方差目标被优化（view_loss 下降 21%）却不带来纠错。
4. **变化仍小/不稳定，或 C20 无偏移？** 否。C20 偏移真实存在（Frozen 退化 5.4×），但 EP 未纠正它。

**核心判断**：不是“更新太小”（已试大步长到 rho 边界）、也不是“没有有效偏移”（C20 已制造偏移）、而是**view-variance 目标的下降方向与鉴伪纠错方向不一致**——它压缩三视图方差的同时放大了判定误差。这转向目标函数与纠错机制评审，而非继续增大扰动。

## 5. 成本与交付

- C0 矩阵 254.2s、C20 extract 196.8s + parity 207.6s、C20 矩阵 252.1s；GPU 累计 ≈0.11 GPUh（<0.5），CPU ≈0.28 core-hour（<8），工件 <2.0 GiB。`pytest -q` → **279 passed**。
- 交付：`R6_D1_REPORT.md`、`RESULT_SEMANTICS_AUDIT.json`、`R6_D1_COMPARISON.csv`（16 行）、`R6_D1_STEP_TRAJECTORIES.csv`、`R6_D1_KEEP_REACHABILITY.json`、C20 派生清单/cache/parity/hash、proposal/LOCK。

未训练、未 SSL、未读取任何目标效果、未进入 R7–R9、未 commit/push/tag；版本 0.1.0。

# EP-TTA v0.1.0 — AASIST R5 第二阶段执行报告（全量缓存 + 源资源拟合）

日期：2026-09-17
结论：**PASS。三角色全量缓存、完整 parity、K=0、U/tau0/M/Fisher/static R 拟合与 loader 验收全部通过。K>0/R6–R9 仍为 NOT_RUN。**

## 1. 批准与 LOCK

| 项 | 值 |
|---|---|
| 新 proposal SHA-256 | `1536090ddaca26c293486ed7ab0191280895d641fb7d4627c05f9dbad0f3b7cb` |
| LOCK 决定 | 用户明确选择“批准 LOCK，执行第二阶段” |
| lock.json | `2b0ad806d203d6de533ab25a8dc9e3a992a66aedbea6dd33944a8c77166076ad` |
| source_artifacts.lock.json | `2a9cb3f3bb810dab9d2b3076321c55d24a15115ae1a03c850689d2e8a1712c10` |
| LOCKED 计划 | fit `07f710f4…` / cal0 `cdff5224…` / select `41b035e4…` |

复用 R4 v4 身份现场复核：bundle `d774f714…`、checkpoint `076ca355…`、snapshot `50d89313…`；worker `58816de6…`（含 null 守卫历史）、parity worker `ad860968…`。数值代码未变，未重导出/重训练。

## 2. 三角色全量缓存（无 pickle、不可覆盖）

| role | 数量 | chunks（块边界） | cache_key | 状态 |
|---|---|---:|---|---|
| fit | 25380 | 100（99 完整块 + 36 尾块） | `features-b27b517a…` | LOCKED |
| cal0 | 4906 | 20（19 完整块 + 42 尾块） | `features-a9d2eccc…` | LOCKED |
| select | 11520 | 45（无尾块） | `features-7069e243…` | LOCKED |

合计 41806 条，块布局与计划一致；三视图 `[identity, deterministic_noise, deterministic_fir]`、z0=view0、seed13、16 kHz/64600/FP32/block_units=256。每角色独立 identity（绑定 bundle/checkpoint/preprocess/probe/worker/seed），未覆盖 128 小缓存。缓存约 1995 B/sample。

## 3. 全量 cache parity（独立重编码 + 落盘读回，atol=1e-6/rtol=1e-5 联合容差）

| role | embedding/head_logits/score 最大绝对误差 | 联合容差超限 | 非有限 | 覆盖 | roundtrip 逐位 | 状态 |
|---|---|---:|---:|---:|---:|---|
| fit | 0.0 / 0.0 / 0.0 | 0 | 0 | 完整 | true | **PASS** |
| cal0 | 0.0 / 0.0 / 0.0 | 0 | 0 | 完整 | true | **PASS** |
| select | 0.0 / 0.0 / 0.0 | 0 | 0 | 完整 | true | **PASS** |

批/序一致性（B=1/2/7/48、逆序、偶奇、A→B→A、重复读取）over_joint 全 0；B=1 的 matmul-vs-dot 内核差异仍在联合容差内。峰值显存 586.5 MB（模型加载主导）。TF32 口径 `tf32_matmul=false / tf32_cudnn=true`。

## 4. K=0（复用第一阶段生产入口，无新实现）

| role | 状态 | all_match | batch 一致 | k0_status | 数值 fallback |
|---|---|---|---|---|---|
| fit | **PASS** | true | true | no_adaptation | false |
| cal0 | **PASS** | true | true | no_adaptation | false |
| select | **PASS** | true | true | no_adaptation | false |

## 5. 源资源（依赖顺序 fit→U、cal0→tau0、fit+tau0→M；资源在 loader 中验收通过）

`artifact_bundle_id = ep-resources-9133770434dc343e7060`，`resources.json` SHA-256 `a7d9bc3b…`，LOCKED、无 pickle。

- **U**：`balanced_response_subspace`，4 单元格（类×noise/fir），每格 20 来源组 × 64 = 1280 单元，FP64 累积、对称化 eigh，rank=8；正交误差 `2.38e-7`；特征值谱已记录。
- **tau0**：`-3.838419198989868`（`empirical_real_quantile`，alpha=0.05，interpolation=higher，larger_is_spoof）；cal0 bonafide=574 / spoof=4332。
- **M**：256 锚点（128/类），margin 难易分层（margin_bins=4），`anchors_m0` 范围 `5.04–24.73`（全 >1e-6）。
- **Fisher**：对角 R=0 Fisher，min `2.96e-10` / max `1.89e-6` / mean `1.86e-7`（近零已报告）。
- **static R**：`fixed_R` Frobenius `0.038`（≤rho 0.2），steps=200 source-only，不依赖 select。
- **w/b**：`w[160]`、`b=-0.01322823017835617`；random U seeds [13,29,47]；feature-PCA U 对照单独封存。
- **loader 验收**：`load_frozen_resources(resources.json, bundle)` 成功，U 正交、anchors_z(256,160)、anchors_y 0/1 平衡，channel=frozen_bundle。

select 标签未进入任何源拟合；`generator_id`（A01–A06）仅用于六攻击覆盖审计，未当处理家族。

## 6. 命令、环境、退出码、计量

- 测试：`PYTHONPATH=src py310 -m pytest -q` → **259 passed**，exit 0。
- LOCK / 3×extract / 3×parity / 3×K=0 / build-artifacts：全部 exit 0。环境 py310 协调器（torch 2.1.0+cu121）+ py38 worker（torch 2.0.1+cu118）、GPU0 RTX A6000。
- **GPU**（含启动/重编码，非仅 kernel）：extract fit 421.7s + cal0 139.6s + select 325.7s = 887.0s；parity fit 442.1s + cal0 85.8s + select 199.6s = 727.5s；合计 **1614.5s ≈ 0.448 GPUh**，小于 1.0 GPUh 上限。build-artifacts 13.8s（CPU）、K=0 合计 9.0s（CPU）。
- **磁盘**：本轮阶段二新增工件合计 **114,546,726 B ≈ 0.107 GiB**，远小于 2.0 GiB；纯三视图 embedding 下限约 76.6 MiB（41806×1920 B），加索引/报告/临时副本约 0.11 GiB。

## 7. 未完成项

K>0、R6–R9、select 上的参数/方法选择、目标评分均为 NOT_RUN。Fisher/static R 已完成 source-only 拟合与资源身份验收，但“在 select 上选择其超参”属于 R6，本轮未做。未训练、未 SSL、未访问 source_val/audit/control_test 音频或任何目标效果、未 commit/push/tag。

## 8. R6 源先导 proposal 草案（仅计划，不执行）

基于本轮真实资源（U/tau0/M/Fisher/static R + select 全量缓存），R6 先导按序：

1. **有限真实 K=1 梯度/更新/逐样本 reset 检查**：在 select 上取小批样本，验证 `run_episode` 的 K=1 梯度、`loss_b.sum()`、逐样本 R=0 重置、投影、A→B→A 一致性；方法/参数由 DESIGN §4/§5 与 registry 展开，不新造实现。
2. **Frozen / 同视图平均 / 完整 EP / 既有 P0/P1 与关键消融**：`frozen`、`multiview_mean`、`ep_no_keep`、`ep_random_U`、`ep_tta` 及熵/MEMO/保持替换、static_subspace、fixed_source_adapter、ep_feature_pca_U 等，按 registry 已注册项跑分。
3. 方法与超参数在 select 上按 DESIGN 既定准则选择；不把原 probe 当另加开发环境，不改四个目标范围。

新 proposal 的完整 SHA-256、命令与预算将在 R6 开工前单独固化并申请一次批准。版本保持 0.1.0。

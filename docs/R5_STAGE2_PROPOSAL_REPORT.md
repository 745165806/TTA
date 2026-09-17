# EP-TTA v0.1.0 — AASIST R5 第二阶段 Proposal（待批准 / NOT_RUN）

日期：2026-09-17
状态：**PROPOSED / APPROVAL_REQUIRED / NOT_RUN**

具体 proposal：`artifacts/private/r5-stage2-full-proposal-20260917/proposal.json`
完整 SHA-256：`1536090ddaca26c293486ed7ab0191280895d641fb7d4627c05f9dbad0f3b7cb`

## 1. 接续与身份

前置证据（只读，不冒充新批准）：上层计划 `07024939…ac79d`、第一阶段 LOCK `e79f1d69…a42`。复用 R4 v4 bundle `d774f714…9684`、checkpoint epoch69 `076ca355…bf0a`、snapshot `50d89313…5fe0`。实际 worker `58816de6…`（含 input_sha256 null 守卫历史），parity worker `ad860968…`。数值代码未变，不重导出、不重训练。

数值口径（第一阶段实际）：dtype float32、cuda matmul TF32=false、cuDNN TF32=true，其余确定性/设备/环境沿用已封存证据。

## 2. 逐项缺口（可复用 / 缺真实验收 / 缺实现 / 外部阻塞）

- **可复用**：cache writer/reader、extract/parity/K=0 入口、`verify_frozen_export`、`build_source_resources` 骨架、`empirical_real_quantile`、`empirical_diagonal_fisher`、`fit_fixed_source_adapter`、角色防火墙、三视图 probe、label-free worker。
- **缺真实验收**：U/M/Fisher/static R 只在合成 fixture 上验证，未在真实 fit/cal0 全量缓存上跑；资源 loader（`load_frozen_resources`）未被真实消费入口端到端加载。
- **缺实现（本轮已补）**：U 的“类×处理家族平衡、来源组固定等量、FP64 累积、对称化+eigh”此前是等权 FP32；M 的“margin 难易分层”此前是随机抽取；label/group sidecar 生成；源 artifact plan 缺少 treatment_families/samples_per_group/pair_seed/margin_bins/margin_epsilon/minimum_cal0_bonafide 等字段。
- **外部阻塞**：无（GPU 经 danger-full-access 可达，数据/快照可读）。Fisher 的“源选择”、static R 的“源选择”均为 fit 内固定超参的 source-only 拟合，不依赖 select（见 §4）。

## 3. 全量缓存

| role | purpose | 数量 | 无标签 manifest SHA-256 | 块布局 |
|---|---|---:|---|---|
| fit | source_prepare | 25380 | `5ff18eee…ff04` | 99 完整块 + 36 尾块 = 100 |
| cal0 | source_prepare | 4906 | `ca5e9701…396b` | 19 完整块 + 42 尾块 = 20 |
| select | select | 11520 | `ef76e3c8…1807` | 45 完整块，无尾块 |

合计 41806 条，以锁定 manifest 核实。三视图 `[identity, deterministic_noise, deterministic_fir]`，z0=view0。probe seed13 / noise 30dB / FIR 0.05；16 kHz/64600/FP32/block_units=256。每角色独立 cache identity/output；全量新工件，不覆盖 128 条小缓存。

sidecar（协调器专用，不传给编码 worker）：fit labels `342e275d…` / groups `be91806d…`；cal0 labels `83b1a737…` / groups `858ec6bf…`；select labels `07e085f2…` / groups `c7120336…`。select 标签不进入任何源拟合。

## 4. 源资源参数（从 DESIGN 与实现核对，非凭名猜公式）

- **U**：`balanced_response_subspace`。单元格 = 类别(2) × 处理家族(2: noise/fir)，来源组 `source_group_id`（说话人）固定等量采样 `samples_per_group=64`，`pair_seed=13`，FP64 累积，对称化后 `eigh` 取 rank=8。攻击字段 `generator_id`（A01–A06）仅用于覆盖审计，不当处理家族。报告单元格覆盖/特征值/正交误差。
- **tau0**：`empirical_real_quantile`，alpha_cal=0.05，interpolation="higher"，larger_is_spoof；`minimum_cal0_bonafide=100`（实测 cal0 bonafide=574）。不改最小 EER、不因 M/select 移动。
- **M**：`build_anchor_memory`，anchor_per_class=128，margin_bins=4，margin_epsilon=1e-6，seed=13，两类平衡、m_i^0>ε、难易分层；不足即减规模或 BLOCKED，不带放回补样。
- **Fisher**：`empirical_diagonal_fisher`（R=0 逐样本平方再平均，对角），报告 min/max/mean 近零情况。
- **static R**：`fit_fixed_source_adapter`，steps=200/lr=0.01/rho=0.2/gamma=0.1/lambda_keep=1.0，纯 fit 源样本 source-only 拟合，仅更新静态适配变量，不更新 encoder/head，不依赖 select。
- **random U**：seeds [13, 29, 47]；feature-PCA U 沿用 `feature_pca_subspace`（对照用，与响应 U 分开）。

## 5. 预算与命令

硬上限：单卡累计 ≤1.0 GPUh、本轮工件峰值 ≤2.0 GiB（含临时副本/报告）。每角色一次全量 extract + 一次独立重编码 parity（生产 batch，不枚举所有 batch 组合）；K=0 复用第一阶段入口。命令见 proposal.json `commands_after_review_lock`（3 extract + 3 parity + 1 build-artifacts，py310 协调器 + py38 worker，CUDA_VISIBLE_DEVICES=0）。

外推（阶段一实测 ~0.0099 GPUh / 384 条重编码）：41806 条单次 extract ≈0.116 GPUh，parity 重编码另计 ≈0.116 GPUh，合计 ≈0.23 GPUh；叠加源拟合（U/M/Fisher/static R 的 CPU 统计 + fixed_R 200 步）预计总 <1.0 GPUh。纯三视图 FP32 embedding 下限 ≈76.6 MiB，加索引/报告/临时副本 <2.0 GiB。

## 6. 代码改动（本轮）

- `src/eptta/offline/subspace.py`：新增 `balanced_response_subspace`（类×家族×来源组平衡 + FP64 + 对称化 eigh）。
- `src/eptta/offline/anchors.py`：`build_anchor_memory` 增加 margin 难易分层（margin_bins，默认 1 保持兼容）。
- `src/eptta/offline/artifacts.py`：`build_source_resources` 接入平衡 U、分层 M、label/group sidecar、minimum_cal0_bonafide、fisher/anchors/orthogonality 诊断。
- `src/eptta/offline/__init__.py`：导出 `balanced_response_subspace`。
- `src/eptta/execution/r5_stage2.py`：`prepare_r5_stage2_proposal` + `lock_r5_stage2_proposal`。
- `src/eptta/cli.py`：新增 `prepare-r5-stage2-proposal` / `lock-r5-stage2-proposal`。
- `tests/unit/test_offline.py`：新增平衡 U、分层 M 反例。

测试：`pytest -q` → **259 passed**，exit 0。未训练、未 SSL、未目标评分、未 K>0、未 commit/push/tag。本 proposal 保持 PROPOSED，交给现有 compiler 会拒绝（要求 LOCKED）；在用户批准并 LOCK 前不启动任何 GPU 或真实源拟合。

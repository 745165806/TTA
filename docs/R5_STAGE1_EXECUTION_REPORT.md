# EP-TTA v0.1.0 — AASIST R5 第一阶段执行报告（小缓存 + cache parity + K=0）

日期：2026-09-17
结论：**PASS（LOCK、三角色真实小缓存、cache parity、K=0 接线全部通过）。全量缓存与 U/M/Fisher/static R/tau0/K>0/R6–R9 仍为 NOT_RUN。**

## 0. 批准与 LOCK 身份

| 项 | 值 |
|---|---|
| 上层计划 SHA-256 | `07024939e2a156e681e006ace20b1b542b2cfde811c30e87d4977ffa3ecac79d` |
| 具体 proposal SHA-256 | `3579192bc6fbc3507e0eb618c9bb36f2a56ee80f11a65641bc3505bfa11dd1bc` |
| LOCK 决定 | 用户于 GUI 明确选择“批准 LOCK，继续执行第一阶段” |
| lock.json | `e79f1d692285eecd9c80d86542f1ff6b5c7f3efff208fdc3a6be6118b2290a42` |
| LOCKED 计划 | `fit` `425f34b3…` / `cal0` `b65e9a09…` / `select` `446fd6c0…` |

复用 R4 v4 身份全部现场复核通过：bundle `d774f714…9684`、export manifest `8cd0c21d…10c56`、checkpoint epoch69 `076ca355…bf0a`、snapshot `50d89313…5fe0`、probe `505b00be…3547`。v1/v2/v3 与便利别名未进入作业。

## 1. 本轮代码修补（最小化，未改架构/预处理/分数语义）

| 文件 | 变更 | 说明 |
|---|---|---|
| `workers/baseline_bridge.py` | 修复 `extract` 的 `input_sha256` null 守卫 | 原实现对 `input_sha256:null` 恒判“音频已变化”；与 R4 `_read_unlabeled_manifest`/`r4_verify` 的 `is not None and` 语义对齐。仅改内容校验，不改 load_audio/views/forward/score。 |
| `src/eptta/adaptation/episode.py` | 新增 `run_k0_episode` | resource-free K=0 生产入口：R=0，直接返回原视图 `z0 @ w + b`，不加载 U/M/tau0，状态 `no_adaptation`，非数值 fallback。 |
| `src/eptta/execution/r5_stage1.py` | 新增 | LOCK、cache parity 作业编译/启动、K=0 验收。 |
| `workers/r5_parity_bridge.py` | 新增 | 独立重编码→落盘读回→按 UID/视图比较 + 分阶段计时/峰值显存。 |
| `src/eptta/cli.py` | 新增 `lock-r5-stage1-proposal` / `run-r5-stage1-parity` / `run-r5-stage1-k0` | 三个命令接线。 |
| `tests/contracts/test_r5_stage1_execution.py` | 新增 5 项 CPU 反例 | K=0 原视图分数/非均值、三视图校验、坏 head、LOCK 幂等/错 hash。 |

worker hash 变更（仅内容校验修补，数值实现不变）：proposal 绑定的 `e68ce62a…` → 实际执行 `58816de661b81f9695569f7a86f876388216d31f77dd3d004c69824378e6d1a8`。新 parity worker hash `49407254…d1192`。三份 cache identity 均绑定实际 worker hash `58816de6…`。

**TF32 说明**：proposal 的 `numerical_mode.tf32=false` 指 matmul（线性头路径，`tf32_matmul=false`）；conv 的 cudnn 沿用了 R4 冻结默认（`tf32_cudnn=true`），parity 报告与 R4 parity.json 一致记录，未强制改写。parity 重编码与 extract 同 torch 默认，逐位一致。

## 2. 三角色真实小清单审计

| role | purpose | 数量 | 类别 | 攻击字段 | 攻击 | speaker | 无标签 manifest SHA-256 |
|---|---|---:|---|---:|---|---|---|
| fit | source_prepare | 128 | 64/64 | generator_id | A01–A06 | 20 | `dfc3a526…94a5` |
| cal0 | source_prepare | 128 | 64/64 | generator_id | A01–A06 | 5 | `c7ba281f…9696` |
| select | select | 128 | 64/64 | generator_id | A01–A06 | 8 | `38eeaca5…b066b` |

- fit 精确复用 R4 v4 `fit128_uids.json` 的 UID 与顺序（现场核验 `==`）。cal0/select 由协调器按源元数据（speaker / `generator_id` 分层 + 文件大小两端交替 + seed13 UID hash 排序）固定选择，不看模型分数。
- 攻击字段为真实 `generator_id`（非 v3 的 `treatment_id`）；worker 无标签清单仅含 6 字段，无 label/attack/speaker；协调器 selection evidence 单独保存并标记 `coordinator_only_annotations=true`。
- 三份 manifest 各 128 唯一 UID、`split_role` 正确、无重复/缺失/注释字段。

## 3. 真实小缓存（无 pickle、sharded NPY、不可覆盖）

| role | cache_key | sample_count | shape/dtype | index.json SHA-256 |
|---|---|---|---|---|
| fit | `features-2b92bde7…d2054` | 128 | [128,3,160]/float32 | `c2c72c7b…964b` |
| cal0 | `features-0d6588ee…8252` | 128 | [128,3,160]/float32 | `e5e4a3d3…e630` |
| select | `features-e70f8db4…dec0` | 128 | [128,3,160]/float32 | `bed922c3…935e` |

identity 同时绑定 snapshot/role manifest、bundle、checkpoint、preprocess `6d1eaf49…28f6a`、probe `505b00be…3547`、worker `58816de6…`、seed13、dtype float32。`z0=identity=view_index_0`，三视图顺序 `[identity, deterministic_noise, deterministic_fir]`，未另存第四份 z0。每角色一个 tail chunk（`block_units=256`，128 条仅覆盖尾块；完整块分支由合成反例覆盖，见 §5）。

## 4. Cache parity 逐样本/汇总（FP32/CUDA，atol=1e-6 / rtol=1e-5 联合容差）

| role | embedding 最大绝对误差 | head logits 最大绝对误差 | score 最大绝对误差 | 联合容差超限样本 | 非有限 | 覆盖 | roundtrip 逐位 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fit | 0.0 | 0.0 | 0.0 | 0 | 0 | 完整 | true | **PASS** |
| cal0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 完整 | true | **PASS** |
| select | 0.0 | 0.0 | 0.0 | 0 | 0 | 完整 | true | **PASS** |

- 在线重编码（相同确定性视图，不重抽噪声）与落盘读回逐 UID/视图对齐：z0、三视图 embedding、head logits、score 均逐位一致（最大绝对误差 0.0）。
- 无损 FP32 写入→读回 roundtrip 逐位一致（`roundtrip_bit_exact_all=true`）。
- 批/序一致性：逆序、偶/奇分片、A→B→A、重复读取均 0.0；B=1 评分因 matmul-vs-dot 内核差异出现 `3.8e-6`（fit）/`1.9e-6`（cal0/select）的纯绝对差，但 `over_joint=0`（联合容差内，未放宽）；B=2/7/48 均为 0.0。
- 每角色 128 条唯一覆盖，无重复/缺失/非有限。parity 报告：fit `5c86933a…` / cal0 `3dcb5b51…` / select `fabc1c5f…`。

## 5. K=0 生产入口验证

| role | 状态 | all_match | batch 一致 | k0_status | 数值 fallback | 序一致性 |
|---|---|---|---|---|---|---|
| fit | **PASS** | true | true | no_adaptation | false | 串行/乱序/偶奇/A/B/A_repeat 全部一致 |
| cal0 | **PASS** | true | true | no_adaptation | false | 同上 |
| select | **PASS** | true | true | no_adaptation | false | 同上 |

- 通过真实 EP 入口 `run_k0_episode`（resource-free，R=0）执行 K=0：每条返回原视图 z0 分数（非三视图平均），`steps_completed=0`、`status=no_adaptation`，未记为数值失败 fallback。
- 与同一 z0、同一 head(w,b) 的 Frozen 路径逐样本相等（`score==frozen_score`），batch 化 frozen matmul 亦一致。
- 无 U/M/tau0 资源，未编造假资源，K=0 恒等路径合法短路；K>0 的原资源/hash 门禁保持未动。

## 6. 命令、环境、退出码

- 测试：`PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python -m pytest -q` → `253 passed`，exit 0。
- LOCK：`… -m eptta.cli … lock-r5-stage1-proposal --proposal …/proposal.json --proposal-sha256 3579192b…` → exit 0。
- extract（三角色各一，CUDA_VISIBLE_DEVICES=0，py310 协调器 + py38 worker）：exit 0。
- parity（三角色各一，GPU0，py38 worker）：exit 0。
- K=0（三角色各一，py310 协调器，CPU 读回）：exit 0。
- 环境：协调器 py310（torch 2.1.0+cu121）、worker py38（torch 2.0.1+cu118，soundfile 0.12.1）、GPU0 RTX A6000。

## 7. 计量

**GPU**（含失败与重试的墙钟，非仅 kernel）：首轮 fit extract 失败 1.68s + “输出已存在” 0.09s + 成功 4.08s、cal0 6.73s、select 6.39s、fit parity 失败 3.88s + 成功 4.15s、cal0 4.55s、select 4.19s，合计 **≈35.76s ≈ 0.0099 GPUh**，远小于硬上限 0.25 GPUh。峰值显存 `586508800 B`（约 559 MiB，模型加载主导，与 batch 无关）。

**磁盘**：三角色 cache 各 `264912 B`（= 纯 embedding 245760 B + chunk ids 9219 B + npy header/index），折合 **2069.6 B/sample**；parity 213853 B、k0 88908 B、locked 8762 B。本轮新增工件合计 **1,156,940 B ≈ 0.0011 GiB**，远小于 0.25 GiB。

**分阶段耗时**（parity 实测稳态，per sample）：decode 0.0014s、view 0.0019s、encode 0.0065s、write 0.00015s、read 4.5e-6s；端到端 wall（三角色 extract + parity）约 39.8s。

**全量外推**（fit=25380、cal0=4906、select=11520，共 41806 条）：
- 纯三视图 FP32 embedding 下限 `41806×1920 = 80,267,520 B ≈ 76.6 MiB`；加 ids/index/npy header 约 `41806×2069.6 ≈ 82.5 MiB`；不含 selection evidence、报告、临时副本与安全余量。
- GPU：单次 extract ≈ `41806×0.00995 ≈ 416s ≈ 0.116 GPUh`；若再加同规模 parity 重编码另计 ≈0.116 GPUh，合计 ≈0.23 GPUh，接近并可能触及 0.25 GPUh，因此**全量缓存需单独 GPU 预算批准**。

## 8. 下一阶段计划（仅计划，不执行）

按依赖组织，不按旧计划条目编号顺序：

1. **fit 全量缓存 → U**：先批准并生成 fit=25380 全量三视图 cache；再由 fit 差分构造响应子空间 U（非中心二阶矩、FP64、对称化/eigh、秩与能量门禁）。
2. **cal0 全量缓存 → 封存 tau0**：批准并生成 cal0=4906 全量 cache；先封存 tau0（`empirical_real_quantile`）。
3. **M 使用 fit 样本并按 tau0 筛选**：在 tau0 封存后，从 fit 锚点按 `m_i^0>ε`、两类平衡、margin 难易分层构建 M；不因条目编号先拟合依赖 tau0 的 M。
4. **Fisher / static R**：顺序与依赖仍以 DESIGN §4/§5 为准（Fisher 需 U+M+真实源标签；static R 需源选择）。

新完整 SHA-256 与真实命令将在下一阶段 proposal 固化后给出，规模/预算按上述外推单独申请。本轮保持 U/M/Fisher/static R/tau0、全量缓存、K>0、R6–R9 为 NOT_RUN；未读取 source_val/audit/control_test 音频或任何目标效果；未训练、未操作 SSL、未 commit/push/tag。

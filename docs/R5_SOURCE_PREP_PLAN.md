# EP-TTA v0.1.0 — R5 Source Preparation Plan（未执行）

状态：`APPROVAL_REQUIRED / NOT_RUN`  
前置 bundle：`baseline-d1f0d91901c73eb5027c`  
bundle SHA-256：`d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684`

## 1. 固定身份与用途

- seed：13；source snapshot：`snapshot-5731a8d70a8b8fa4997d` / canonical SHA-256 `50d893...15fe0`。
- `fit`（25380）：仅用于源响应二阶矩/子空间 U、fit anchors M、Fisher 对照及 static/source-only 对照需要的源统计；不用于目标评价。
- `cal0`（4906）：仅用于 tau0/源校准及其合法依赖；不训练 detector，不替代 select。
- `select`（11520）：仅用于 EP/基线参数、P0/P1 与方法选择；不回流 U/M/Fisher/tau0，不选 checkpoint。
- `source_val` 保持 R4/训练选模角色，不复用为 R5 参数选择。`audit` 和四个目标范围不进入本轮源资源作业。

## 2. 执行顺序与门禁

1. 复制 example 为新的 proposal，绑定上述 bundle、各 role manifest/hash、label sidecar 和路径；审核后才 LOCK。
2. 先从 fit/cal0/select 各取固定、分层的小清单做真实 cache 接线；worker 只接收无标签输入。验证同一音频的直接 frozen embedding/score 与 cache 解码结果一致。
3. 验证原视图 `z0` 与既定三视图 `[identity, deterministic noise, deterministic FIR]` 的索引、probe seed13、16 kHz/64600 输入、FP32、TF32=false、`block_units=256`。
4. 只有小缓存 PASS 后才分别生成 fit、cal0、select 全量 cache；不同 role 使用独立 output/cache identity，失败分片隔离，禁止用其他 role 补洞。
5. fit cache 构建 U：类×处理家族平衡、来源组固定等量、FP64 累积非中心二阶矩、对称化/eigh、秩与能量门禁；不得默认中心化/白化。
6. fit 构建 M：两类平衡、tau0 下正确、按 margin 难易分层；不足时显式减少 M 或阻塞，不复制样本。
7. cal0 只拟合/封存 tau0；Fisher、static R 和 source-only 对照严格按 DESIGN 既有定义与已有 P0/P1，不改变算法名称或损失。
8. select 上先做 K=0 和 frozen 接线，再做已批准源选择；本计划不授权任何目标评分。

## 3. 必须验收

- cache identity 同时绑定 snapshot、bundle/baseline、checkpoint、preprocess、probe、worker、seed、dtype/numerical mode。
- 完整 UID 覆盖、无重复/缺失、乱序与 1/2/7/生产 batch、尾批、分片重组一致。
- cache parity：原视图 embedding 与 R4 production wrapper 在固定容差内；head score 与直接 frozen score 一致；非有限值隔离并使角色作业 FAIL。
- K=0：逐样本 reset，`R=0`，最终原视图 score 与 frozen 一致；串行、乱序和分片结果按 UID 一致，不跨样本/跨卡归约 R。
- K>0 前置：每条独立 reset；失败只隔离该样本并记录标准 fallback；不得继承上条 optimizer/R/历史。
- U 正交、有效秩、能量与 bootstrap 稳定性；M/tau0 角色与标签访问审计；Fisher/static R 资源 hash；所有产物不可覆盖。

## 4. 规模与预算候选（待批准）

既有三视图、d=160、float32 的纯 embedding 下限约每样本1920 bytes：fit约46.5 MiB、cal0约9.0 MiB、select约21.1 MiB，合计约76.6 MiB；另计 UID/index/hash、临时分片、报告和至少一份安全余量，建议预留不低于0.25 GiB。若同时封存原始输入、logits或调试张量，必须另报空间，不能沿用该估算。

GPU 预算必须在小缓存计时后外推：记录 decode/forward/写盘各自耗时、峰值显存和端到端 wall time，再提交单卡/双卡方案。未经新批准，不启动全量 cache 或资源拟合。

## 5. 停止点

R5 首次授权应只覆盖“小缓存 + cache parity + K=0 接线”；通过后再单独批准分角色全量缓存和 U/M/Fisher/static R/tau0。继续沿用原 P0/P1、方法注册和四目标冻结规则，不读取2019 eval、2021 LA/DF、In-the-Wild或任何目标效果。

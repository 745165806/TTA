# EP-TTA v0.1.0 — AASIST R5 第一阶段具体 Proposal

日期：2026-09-17  
状态：**PROPOSED / APPROVAL_REQUIRED / NOT_RUN**

## 审批边界

绑定的 `R5_SOURCE_PREP_PLAN.md` SHA-256 已核验为 `07024939e2a156e681e006ace20b1b542b2cfde811c30e87d4977ffa3ecac79d`。现有 extraction compiler 只接受严格 `status=LOCKED`；该计划又明确规定具体 UID/probe proposal 审核后才可 LOCK。因此本轮没有把用户对上层计划的批准扩张为对事先未知 UID 的批准。

具体 proposal：`artifacts/private/r5-stage1-aasist-small-proposal-20260917/proposal.json`  
完整 SHA-256：`3579192bc6fbc3507e0eb618c9bb36f2a56ee80f11a65641bc3505bfa11dd1bc`

它保持 `status=PROPOSED`、`approval_required=true`、`locks_published=false`。把任一 extraction proposal 交给现有 compiler 会得到 `ContractError: inference plan must be a strict LOCKED v0.1.0 document`，因此当前工件不能启动正式 cache。

## 固定身份与 probe

- R4 v4 bundle：`baseline-d1f0d91901c73eb5027c`；bundle SHA-256 `d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684`。
- export manifest SHA-256：`8cd0c21d3d89b46dbf4d07f86df29bce8e18803b952ed58d5bb6441264210c56`。
- checkpoint SHA-256：`076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a`；seed 13。
- snapshot：`snapshot-5731a8d70a8b8fa4997d`；canonical SHA-256 `50d893131ce99752759fb14a121bd17c29a9782eddb0d8560ac458d336315fe0`。
- worker SHA-256：`e68ce62a7577a95a8ae2365866820d5e40d3001f6ba58b918e9d2411593cbe4a`。
- probe：3 views、seed13、noise SNR 30 dB、FIR side gain 0.05；probe SHA-256 `505b00be92d4efb5fc023b159a198157c420ec4bec53373fe9657e287f123547`。
- view 顺序：`[identity, deterministic_noise, deterministic_fir]`；`z0=identity=view_index_0`，不会另存第四份 z0。
- noise seed：`sha256(seed + NUL + sample_id)` 前64 bits 模 `2^63-1`，CPU `torch.Generator`；与 batch、乱序、分片和此前 RNG 消耗无关。
- FIR：reflect-pad 1 sample，kernel `[0.05, 1, -0.05]`。FP32、TF32=false、`block_units=256`。

`block_units` 在实际 worker 中表示每个落盘 chunk 的样本数，不是 batch size。每角色128条，因此真实执行只覆盖一个 tail chunk；完整块分支必须由合成反例覆盖，不能跨角色拼块。

## 三角色清单

| role | purpose | 数量 | 类别 | 攻击 | speaker | 无标签 manifest SHA-256 |
|---|---|---:|---|---|---:|---|
| fit | source_prepare | 128 | 64/64 | A01–A06 | 20 | `dfc3a526bfb04722e79eba5389eac5ba6718404951be53a60320fe8a66d694a5` |
| cal0 | source_prepare | 128 | 64/64 | A01–A06 | 5 | `c7ba281f617054e68610d5520c2524702cf7e710f863571591bb7f549e111696` |
| select | select | 128 | 64/64 | A01–A06 | 8 | `38eeaca53d590c47c3e41f87f5dae9993aeb97b18d48f4f204e88b36d27b066b` |

fit 精确复用 R4 v4 fit128 UID 及顺序。cal0/select 按源元数据固定选择：每类64，bonafide 按 speaker、spoof 按真实 `generator_id` A01–A06 轮转，并交替选择源文件大小两端；不看模型分数。三份 worker manifest 各128个唯一 UID，只含六个允许字段，无 label/attack/speaker。协调器 selection evidence 单独保存注释并明确不可传给 worker。

## 真实缺口与获批后工作

现有缓存 writer/reader 可复用，使用无 pickle 的 sharded NPY 和不可覆盖目录；role/purpose 防火墙也可直接复用。尚缺：

1. 独立 cache parity runner：相同确定性 views 在线重编码，与真实落盘后独立读回的三视图 embedding/head score 按 UID/view 对齐，并覆盖 B=1/2/7/48、尾批、逆序、分片和重复读取。
2. 严格的 resource-free K=0-only 生产入口。当前 suite 在 K=0 恒等短路前仍强制加载 U/M/tau0；DESIGN 明确 K=0 合法，因此可补最小入口，但 K>0 必须继续要求完整资源/hash。
3. decode/view/encode/write/read 分阶段耗时、预热/稳态、峰值显存和磁盘峰值计量。

未生成假 U/M/tau0，未把 synthetic fixture 当正式资源。

## 预算与本轮实耗

- proposal 当前实际大小：188246 bytes；没有 cache、临时分片或既有 bundle 副本。
- 每角色三视图纯 float32 embedding 下限：245760 bytes；三角色合计737280 bytes，不含 NPY header/index/report/临时副本。
- 待批准硬上限仍为单卡累计0.25 GPUh、工件峰值0.25 GiB。proposal 为三角色 extraction 各180秒上限、parity 180秒上限，总 GPU wall 候选720秒（0.20 GPUh），为失败/清理前计量保留0.05 GPUh余量。
- 本轮 GPU 实耗：0；正式 cache：0；LOCKED plan：0。

## 测试

- CLI `prepare-r5-stage1-proposal --help`：exit 0。
- 真实 proposal 编译：首次发现 cal0/select 原 manifest 本来就是无标签清单，修正为由协调器从 canonical 获取分层标签；原子输出未留下半成品。修正后 exit 0。
- PROPOSED extraction 被执行 compiler 拒绝：PASS。
- 最终回归：245 passed，exit 0。

未训练、未访问 source_val/audit/control_test 音频、未评分目标、未执行全量缓存/U/M/Fisher/static R/tau0/K>0/R6–R9，未操作 SSL，未 commit/push/tag。

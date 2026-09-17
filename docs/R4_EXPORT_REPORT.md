# EP-TTA v0.1.0 — AASIST R4 冻结导出报告

日期：2026-09-17  
结论：**PASS；AASIST R4 完成，bundle 可进入后续 R5 源资源准备。R5 尚未执行。**

## 1. 复用、缺口与修补

- 可复用：既有 `FrozenModelBundle`、作者 AASIST adapter、线性 head、source_val EER、不可覆盖写入和 R4→R5 检查入口。
- 原实现缺真实验收：只在同一实例和单条音频上比较作者 head 与自身输出；没有独立进程重载、全量 source_val、真实多 batch、梯度边界或 nonexact/终点披露。
- 本轮最小修补：新增 `export-frozen-r4` 入口、候选与验收双进程、严格 bundle schema、真实 UID/覆盖/EER 报告、六攻击门禁及 CPU 反例测试；未重建训练器、缓存或调度框架。
- 外部/后续项：AMP 和跨设备 parity 未验，不能宣称支持；不存在 epoch69 历史逐样本分数文件，因此本轮结果只能称 source_val 复算；K=0、cache parity、U/M/Fisher/static R/tau0 均留给 R5。

## 2. 训练身份与资格闭合

| 字段 | 现场核验值 |
|---|---|
| run / finalized | `source-run-91cd36fae7770aca4b51` / `training-final-a095c4459547c47c9481` |
| 选择工件 | 不可变 `checkpoints/epoch-0069.pt`，不是依赖便利别名 |
| checkpoint SHA-256 | `076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a`；`best.pt` 内容 hash 相同 |
| best 时点 | epoch69 / global_step37030 |
| 实际训练终点 | epoch79 / global_step42320；累计 epoch0–79 共80轮，child 为14–79共66轮 |
| scheduler | 100-epoch cosine horizon；不等同于完成100轮，未补跑80–99 |
| 路线 | 获准 parent→child nonexact；`exact_resume_claim=false`；AASIST/SSL strict resume comparison 仍为 FAIL |
| seed / init | training seed 13；AASIST native initialization |
| 作者源码 | commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`；entrypoint SHA-256 `9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50` |
| snapshot | `snapshot-5731a8d70a8b8fa4997d`；canonical SHA-256 `50d893131ce99752759fb14a121bd17c29a9782eddb0d8560ac458d336315fe0` |
| fit/source_val manifests | `550bcc0f...f4e03` / `4378b98f...31a73`；25380 / 5654 条 |
| preprocess | `6d1eaf49fe7fbf0fcf931572751c0bf6a9178ee89d04bd6b70f6f27f72528f6a` |
| training code/patch identity | combined SHA-256 `1cca8d88d52a73dd6c3833e65faeb51e6e249596c312871585c4ca721716fb54`；历史 worker SHA-256 `4343bac6...9ee4` |

`metrics.jsonl` 中可选记录为 parent epoch12/13 与 child epoch14/37/65/69；既有规则为 source_val EER 最小、完全并列时取较早 epoch。没有重新评价全部 epoch 或用其他 role 打破并列。原始 EER 字段由实际实现确认是 0–1 ratio：`0.00040257648953301306`，百分比为 `0.040257648953301306%`。

## 3. 冻结合同与真实验收

- embedding 是本地 pinned AASIST `out_layer` 的真实输入，形状 `[B,160]`；native logits 为 `[B,2]`。
- native 映射：spoof=0、bonafide=1。score 固定为 `logits[spoof]-logits[bonafide]`，方向 `larger_is_spoof`，类型 `logit_difference`，单位 dimensionless；未加 softmax、温度或归一化。
- 预处理固定为 soundfile float32、16 kHz、双声道取 mono mean、64600 samples、长音频从首部裁剪、短音频重复、无归一化、`Freq_aug=false`。
- reference 从 selected checkpoint 构造；export 从候选权重在新进程中独立重建。参数和持久 buffer 逐 tensor 相同；所有 module 保持 eval；参数 `requires_grad=false`。
- `conv_time.hsupp` 与 `band_pass` 是 pinned 构造器确定性生成的普通 CPU tensor；`filters` 是每次 forward 从 `band_pass` 在目标设备重建的派生缓存。首次/重复/A→B→A 后未出现跨样本学习状态，注册 buffer 不变。
- 真实 fit 128 条按 seed13 固定规则选择：64 bonafide、64 spoof，覆盖 A01–A06、20 名 speaker，并交替取源文件大小两端以覆盖长短候选；选择不看分数。推理 job 只含无标签清单和 UID。
- 执行 B=1/2/7/48、尾批、逆序、偶/奇分片及 A→B→A；再以 B=48 完整覆盖 source_val 5654 条，不 drop_last、不补零。
- FP32/CUDA 预先固定 `atol=1e-6, rtol=1e-5`。embedding/native logits/head logits 最大绝对误差均为0；score 最大绝对误差 `3.814697265625e-06`，p50 `9.5367431640625e-07`，p95/p99 `1.9073486328125e-06`。有1329个标量超过纯 absolute `atol`，但结合预先固定的 `atol + rtol*|reference|` 后超限元素/样本均为0；未放宽容差。
- 可丢弃 embedding clone 经冻结 head 的梯度探针 PASS；输入梯度有限且非零，模型/head 参数无梯度，无 optimizer/step。
- source_val reference/export EER 均为 `0.00040257648953301306`，与历史 raw 值差为0；这是本轮完整复算，不冒充不存在的历史逐样本文件复现。

## 4. 最终工件与 hash

- bundle id：`baseline-d1f0d91901c73eb5027c`
- bundle：`artifacts/private/r4-aasist-20260917-v4/published_bundle/bundle.json`
- bundle SHA-256：`d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684`
- 完整 manifest SHA-256：`8cd0c21d3d89b46dbf4d07f86df29bce8e18803b952ed58d5bb6441264210c56`
- detector state：`5e8b8bde532935580f0aa4d71444cc57bb9a7d9972d88988b1df3e001e11c0d4`
- linear head：`9644e4c55cc06abeb76d8eddf22724f8b8ffbdb2419a445d43a7b857e7f080bb`
- parity report：`69e51113d322308b4bbb1e8d76610725740490ec2a361133b2b2147a58df56f7`
- per-sample parity：`060d8138bf269b5ef7daad273d1e47f105153900325574231d541ebc0324b490`
- fit128 UID evidence：`e17354f77af3ed72a879a29a86afd97ef6283bb2c249fa1e3dc5dfa09b06d4b1`
- source_val recompute：`9bdb0b63a8d0beecd43e9ce5711dbea6c40eebc331718fc43f2aa92d3b123f7f`
- archived export worker：`e68ce62a7577a95a8ae2365866820d5e40d3001f6ba58b918e9d2411593cbe4a`
- archived compat adapter：`a135f6cb80f0e83f96681e8131556fce669a6c9736a7b2c1f6216c30a008c325`

`export_manifest.json` 对九个发布文件逐项封存。受控 model registry 已只绑定 v4 bundle。v1/v2 在 CPU 防泄漏门禁阶段失败、未用 GPU；v3 数值 PASS 后发现攻击字段误取 `treatment_id`，当前 R5 verifier 会明确拒绝它，且 registry 从未绑定 v3。三者均保留为诊断历史，不覆盖、不删除。

## 5. 命令、环境、退出码与资源

- `... eptta.cli export-frozen-r4 --help`：exit 0。
- 相同入口加 `--dry-run` 并绑定实际 finalized/output/Python/GPU：exit 0，无 CUDA、无输出目录写入。
- 定向 CPU 回归：54 passed，exit 0；新增 R4 反例：9 passed，exit 0；修正六攻击 gate 后相关回归：16 passed，exit 0。
- 全量回归（最终代码再次执行，见日志）：PASS，exit 0。
- 最终真实命令使用 `/home/dell/anaconda3/envs/py310` 协调器、`/home/dell/anaconda3/envs/py38` worker、CUDA FP32、GPU0；外部 `timeout` 为1700秒。
- v3 GPU `0.0180023851` 小时；最终 v4 GPU `0.0181100589` 小时；本轮累计 `0.0361124440` GPU 小时，小于硬上限0.5。v1/v2 GPU=0。

未启动训练、SSL 操作、目标评分、R5 cache、U/M/Fisher/static R/tau0 或 R6–R9；未读取 cal0/select/control_test/目标效果；未 commit/push/tag。R4 结果只证明冻结导出一致性，不证明 EP 提升或 H1–H7。

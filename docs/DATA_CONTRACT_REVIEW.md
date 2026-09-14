# 远程合同待审核清单 v0.1.0

2026-09-14 对用户指定的 `/media/dell/data/fakedata` 做了只读盘点，无审核、无 LOCKED 产物。私有路径写入 Git 忽略文件；公共配置的科学未知项继续保留 null 或 PROPOSED。

ASVspoof 2019 LA 绑定的 root 下观测到 915,694 个音频文件、119,176 个 `.npy` 和 26 个 `.txt`；协议目录有 7 个候选，其中同时存在三份 2019 官方 CM 协议、2021 LA key 和三份用户转换 label sidecar。该总数含 root 下嵌套资源，不是 2019 训练集规模。候选合同只选三份 2019 CM 协议，必须由审核者核对后才能 lock。

| 合同 | 待补证据 | 当前状态 |
|---|---|---|
| raw/layout | 已观测 root/协议候选；待审核文件选择、5 列语义、ID→三目录音频规则与全量覆盖 | PROPOSED |
| label | 候选 `bonafide→0, spoof→1`；待审核原始语义及全量未知/冲突覆盖 | PROPOSED |
| group | 候选 speaker field；跨集来源/派生/身份映射、来源质量与冲突仍待核验 | PROPOSED / BLOCKED_CONTRACT |
| preprocess | 独立 decode/train_unit/eval_unit/source_probe/target_probe/quality 合同 | DEFERRED_REMOTE |
| split | 训练前封存角色、来源组互斥、增量保持既有分配 | DEFERRED_REMOTE |
| architecture | 作者完整 commit、许可/patch、输入/输出、实际 embedding/native 类别映射 | TODO（L4 本地结构审计后远程核验） |
| recipe | 损失、类别权重、采样、增强、优化器、scheduler、训练预算、source_val 选模、运行参数 | DEFERRED_REMOTE |
| initialization | 仅 SSL 通用前端，hash 和预训练来源；不接受任务权重 | DEFERRED_REMOTE |

模板中的批准字段不是密码学信任证明。L3 已实现候选插件、通用多格式 adapter、snapshot 和 approval 发布流程；D01–D10 与统一标签合成测试已通过。真实统一标签包已生成并验证，详见 `LABEL_PACK_REPORT.md`；但 group/preprocess/split 未审核、正式 staging/snapshot 未运行，不能据此声称完整数据合同已 LOCKED。

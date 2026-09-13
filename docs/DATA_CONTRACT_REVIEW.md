# 远程合同待审核清单 v0.1.0

本轮无真实资源盘点、无审核、无 LOCKED 产物。configs 中的目录和所有科学未知项均保留 null。

| 合同 | 待补证据 | 当前状态 |
|---|---|---|
| raw/layout | 选中数据 root/协议、格式/编码/表头/列或 JSON 路径、ID 和音频路径规则、原始词汇、缺失策略 | DEFERRED_REMOTE |
| label | 原始真假含义、显式映射、未知和冲突覆盖报告 | DEFERRED_REMOTE |
| group | 跨集来源/派生/身份映射、来源质量与冲突 | DEFERRED_REMOTE |
| preprocess | 独立 decode/train_unit/eval_unit/source_probe/target_probe/quality 合同 | DEFERRED_REMOTE |
| split | 训练前封存角色、来源组互斥、增量保持既有分配 | DEFERRED_REMOTE |
| architecture | 作者完整 commit、许可/patch、输入/输出、实际 embedding/native 类别映射 | TODO（L4 本地结构审计后远程核验） |
| recipe | 损失、类别权重、采样、增强、优化器、scheduler、训练预算、source_val 选模、运行参数 | DEFERRED_REMOTE |
| initialization | 仅 SSL 通用前端，hash 和预训练来源；不接受任务权重 | DEFERRED_REMOTE |

模板中的批准字段不是密码学信任证明。本轮只校验其结构和 payload hash，不能据此声称真实合同已审核。五类真实协议适配、snapshot 和 approval 发布流程未实现。

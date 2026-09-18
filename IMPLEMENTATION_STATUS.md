# 精简迁移说明

日期：2026-09-18。

当前日常流程只有 `prepare-data`、`train-source`、`prepare-source`、`run-tta`、`evaluate`、`report`。旧 reviewer、提案/锁定、阶段 ticket、业务摘要链、独立 score seal 与跨 Python bridge 已退出执行路径；历史报告和工件保持只读，不批量改写。

新数据 assignment 保存明确 `sample_index`，训练裁剪与视图由整数 seed/sample_index/epoch/view 驱动。该规则保持采样分布但改变旧摘要派生序列；旧 checkpoint 可评价或 warm start，不能据此声称旧训练轨迹的 exact resume。

新训练每个 epoch 保存不可覆盖的完整模型与续训状态。缓存绑定明确 run、具体 epoch、数据角色、配置和精确 ID；不提供字节级原地替换检测。

目标运行环境统一为 conda `tta`。本次仅交付配方和代码，没有创建/破坏用户环境，也没有运行完整训练、全量缓存、目标评价或论文效果验证。实际轻量测试结果以本次交付说明中的真实命令为准。

# 远程后续顺序 v0.1.0（未执行）

本文件仅说明后续依赖，不是可直接启动训练的脚本。当前所有真实 CLI 入口返回 `NOT_IMPLEMENTED_STAGE`。

1. 在已有 CPU PyTorch 环境运行 L2 合成数值对照，确认串行、batch、reset 和回退合同。
2. 完成 L3 协议/来源分组/snapshot/增量实现和 D 系列测试；经用户开启远程阶段后，R0 盘点 Ubuntu 双 A6000 的真实路径、环境与资源。
3. R1 审核 raw、label、group、preprocess 与训练前 split，不从目标结果决定方案。
4. L4 固定作者架构 commit 和补丁、实现 SourceTrainer/worker、检查 native 类别顺序；R2 源训练 smoke，只读 fit/source_val。
5. 仅按已批准 recipe 和计划运行 R3 自主训练；source_val 选模，生成本项目 FINALIZED checkpoint。通用 SSL 初始化是独立输入。
6. R4 冻结导出和 wrapper/线性头/mode/buffer parity。随后才能 R5 构建特征与 U/M/tau0，R6–R9 实验封存、评分、评价。

不自动 SSH、不下载资源、不装 CUDA、不调度真实任务。源训练可后续审核 DDP；EP 的独立 R 不进行跨样本/跨卡归约。这里没有硬件吞吐、显存、训练收敛或鉴伪效果承诺。

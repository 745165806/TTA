# EP-TTA v0.1.0 本轮实现决策

1. **当前实现边界**：L0–L4 工程接口、冻结 cache 和机制对照已实现；真实训练/适配需远程证据。五个 published port 只实现审计门，缺固定源码/许可/parity 时明确 `BLOCKED_AUDIT`，不创建同名近似算法。
2. **轻依赖配置**：提供 JSON 语法的 YAML 文件。真正 YAML 通过可选 PyYAML 安全加载且检查重复键；标准库配置路径不导入 torch。`src/eptta/schemas` 是本轮实际支持字段的严格 schema；尚未支持的设计字段必须明确实现后才能加入，不能静默忽略。
3. **明确的配置结构**：运行项包在 `runtime`，选择项包在 `selection`，位置包在 `paths`，待审核合同包在 `contracts`。这是本轮确定的 schema，设计 §8 的展示模板不被当成另一套已实现解析器。profile 只能替换 runtime；私有路径只能替换 paths；experiment 只能替换 selection；EP 参数只允许白名单。每层都是完整受限 section，避免隐式深合并或继承。
4. **训练与适应解析分开**：源 recipe/job 只携带 fit/source_val，worker 二次检查并拒绝目标角色；EP 配置不会进入源训练。源训练 DDP 可做全局梯度归一，EP batch 始终每条新建 R、对 `loss_b.sum()` 反传且没有 collective。
5. **审核字段验证不等于真实审核**：LOCKED 状态、审核元信息及 payload hash 只是执行前提。当前数据合同仍待人工审核；worker/CLI 实现存在不代表 GPU、依赖、收敛或科学结果已通过。
6. **解析由合同驱动**：CSV/TSV/空白/JSON(L)/sidecar adapter 只接受显式列或 JSON path、协议 glob 和协议→音频上下文。插件只提供候选，不猜标签；未知值不是 spoof，空值不是 bonafide。
7. **数学实现与证据分开**：设计附录 A/B 为独立参考；2026-09-14 已在现有 PyTorch 环境完成合成对照。K=0 直接复用冻结分数，避免零矩阵乘法造成浮点末位变化。
8. **共享冻结底座**：response/PCA/random U、M/tau/Fisher/static R 和机制对照均绑定 baseline/checkpoint/cache hash；改变 checkpoint 生成新身份。published port 不能获得不同底座，scalar/static 的半径随 `rho/sqrt(r)` 计算。
9. **作者代码边界**：AASIST/SSL-AASIST 固定到实际本地 commit 和入口文件 hash。SSL 只允许通用 XLS-R 初始化；唯一运行时源码替换是把作者硬编码路径改成显式已校验路径，并记录 patch hash。作者任务权重不能进入 init/resume/export。
10. **文件保护**：原有 `docs/DESIGN.md` 保留。发布使用排他/原子写入，重复运行拒绝覆盖。没有建立虚构 Git commit/tag、任务 checkpoint 或实验结果。

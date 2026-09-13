# EP-TTA v0.1.0 本轮实现决策

1. **仅 L0–L2**：设计中的目录树是目标结构，不创建空的模型/基线实现来暗示完整复现。其他 24 个方法只登记 TODO；选择它们执行会失败。
2. **轻依赖配置**：提供 JSON 语法的 YAML 文件。真正 YAML 通过可选 PyYAML 安全加载且检查重复键；标准库配置路径不导入 torch。`src/eptta/schemas` 是本轮实际支持字段的严格 schema；尚未支持的设计字段必须明确实现后才能加入，不能静默忽略。
3. **明确的配置结构**：运行项包在 `runtime`，选择项包在 `selection`，位置包在 `paths`，待审核合同包在 `contracts`。这是本轮确定的 schema，设计 §8 的展示模板不被当成另一套已实现解析器。profile 只能替换 runtime；私有路径只能替换 paths；experiment 只能替换 selection；EP 参数只允许白名单。每层都是完整受限 section，避免隐式深合并或继承。
4. **训练与适应解析分开**：`resolve_training` 输出源训练预览，不注入 EP 超参数。正式 recipe 编译、训练作业生产、worker 二次检查和 Python 3.7 bridge 仍属于 L4，不能用类型检查代替真实权限审计。
5. **审核字段验证不等于真实审核**：本轮检查 LOCKED 状态、必填字段、审核元信息及 payload hash。报告/抽样证据的内容、资源存在性、同源分组与血缘仍未核验。所有 plan/validate 输出明确 `execution_ready=false`；真实执行入口永远拒绝。
6. **数据接口不是解析器**：ExplicitLabelMapper 仅实现基于已提供合同的显式映射和未知值隔离。测试 CSV 用测试侧解析器读取，未宣称 D02 的全部真实/模拟格式适配已完成。未知值不是 spoof，空值不是 bonafide。
7. **数学实现与证据分开**：设计附录 A/B 原样抽取为独立参考。生产串行使用手写纯 SGD 更新，batch 使用低维等价式；只有 R 求导，资源需 detached。最终原视图评分与最终间隔诊断重新计算。数值异常有明确原因和原分数回退；原分数非法直接拒绝。PyTorch 未装，数值对照未运行。
8. **本轮不构建 U**：响应子空间估计、PCA/Fisher/熵/静态与标量基线属于后续 offline/L5；L2 使用合成正交 U 验证 EP。T11/T12/T15–T20 不因接口或测试参考存在而标 PASS。
9. **文件保护**：原有 `docs/DESIGN.md` 保留。初始化脚本仅使用排他写入，重复运行会拒绝覆盖。没有建立虚构迁移史、Git commit/tag、任务 checkpoint 或实验结果。

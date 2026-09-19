# EP-TTA 精简开发规则

- `docs/DESIGN.md` 中的数学、数据角色、模型和指标语义仍是科学约束；历史 reviewer、审批状态、人工授权、阶段 ticket 和 score seal 不再约束日常实验。
- 项目业务代码不计算或校验 SHA、checksum、内容指纹或哈希链，也不以其他摘要算法重建门禁。Git 引用只可作为普通来源记录。
- 日常训练、提取、TTA、评价和 pytest 统一使用名为 `tta` 的 conda 环境；子进程只能使用 `sys.executable`，不得选择另一套 Python。
- 本机只做轻量、合成和可用资源范围内的验证；不自动 SSH、下载数据/权重、安装依赖、修改 CUDA，亦不启动完整训练、全量缓存或目标集评价。
- 不覆盖或迁移真实数据、私有配置、checkpoint、缓存、实验输出和历史负结果。新工件采用明确 run/epoch/cache 路径且默认不可原地覆盖。
- AASIST / SSL-AASIST 鉴伪参数必须由本项目训练；通用 SSL 前端初始化单独记录，不得替代任务训练。已有兼容 checkpoint、frozen bundle 和明确归属的缓存可只读复用。
- raw/label/group/preprocess/split/recipe 的科学必需字段必须明确；不猜列、不猜标签、不按目录名推断 release/subset/group。
- 源训练只接受 fit/source_val；选参只使用源域 select，阈值只使用 cal0。目标标签、攻击、路径语义、历史及整体统计不得进入适应接口。
- 优先复用已有 assignment。新数据按排序后的明确 group、固定 seed 和独立整数随机生成器生成一次并保存；不得重新划分已有数据或用 Python `hash()` 派生种子。
- EP 每条样本重置，独立 R 不跨样本/跨卡归约；不得把 episodic TTA 改为 continual 或 batch adaptation。源训练 DDP 是独立语义。
- `0=bonafide, 1=spoof`，分数越大越偏 spoof；保留 source native 类别映射，不为改善指标翻转分数。
- 保留 pickle 禁用、shape/dtype、有限值、唯一 ID、精确覆盖、角色隔离和路径检查。普通字段不匹配必须失败；无内容摘要时不声称能检测同名文件的外部原地替换。
- PASS 必须来自真实命令、退出码和日志；缺依赖或未执行标 NOT_RUN/未验证。工程验证、远程运行和科学结论分别记录。

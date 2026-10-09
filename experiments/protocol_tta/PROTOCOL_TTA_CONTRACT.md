# TENT reset protocol study v0.1.0

本轮独立研究轨道 `protocol_tta`：检验每条 reset 是否限制目标域状态积累。
这是对现有 audio-native TENT 的明确扩展，不改变 EP-TTA 的逐样本 R 合同，
也不修改旧 audio-native 脚本、方法、结果。真实 CUDA 实验与科学结论均为 NOT_RUN，
必须由工作站实际日志及聚合结果更新。这里的后续执行范围以本次用户明确指令为准。

## 固定因素与状态

唯一变化是 reset 边界。所有协议使用 config.json 的 TENT、Adam、lr=1e-3、
steps=1、weight_decay=0、backend_norm_affine_v1；model.eval()，源 BN running
statistics 保持冻结。TENT objective 复用现有 prediction_entropy，更新步骤与旧
tent_adapt 一致，不新增 gate、augmentation、scope、裁剪或优化器超参数。

reset 包含完整 source model state 和清空 Adam moments/step；窗口内 Adam 状态持续。
该定义比较的是整个 TTA 状态的重置协议，不能单独归因于参数记忆或 Adam moments。
episodic 每条新建 Adam 与旧实现一致；continual 不能每条重建 Adam。

| 协议 | 发生 reset 的零基 sequence index |
|---|---|
| episodic | 每条：0, 1, 2, ... |
| continual | 仅 0 |
| reset32 | 0, 32, 64, ... |
| reset128 | 0, 128, 256, ... |

模型仅构建一次、configure 一次、source snapshot 一次。适应循环只在上述显式
边界调用 reset_episode_state 与构建 Adam。任何数值、资源、音频或其他异常立即
终止该序列，不回退、隐式 reset 或跳过样本。failure.json 保留失败，runner 返回非零。

## 顺序与信息边界

只允许 target10，直接按 `inwild_target10_select.json` 的 records 顺序，shuffle=false。
sample_index 是处理顺序的零基下标；manifest_sample_index 另存原 manifest 的索引。
episode_index 从零开始；samples_since_reset 包含当前样本，因此 reset 条目为 1。
每个 worker 独占一张卡、处理整个有序序列，不分片，不 DDP，不跨卡交换适应状态。

沿用 TargetWaveformDataset 的严格 label-free 字段白名单及 load_audio。
传入模型/优化步骤的输入只有 waveform，不传 sample ID、路径、标签、攻击或分组。
worker 不加载 feature cache；目标标签只在 aggregate.py 完整性检查通过后读取。
target90 不运行、不读取标签、不做选择。阈值固定为源 resources.tau0。

## 三种分数与诊断

- source_frozen_score：在本 worker 开始任何适应之前，以原始 source 模型完成
  独立 waveform pass。无需第二个 GPU 模型；reference pass 不更新任何参数。
- current_pre_adapt_score = score_before_update：当前序列模型对本条样本更新前的分数。
  continual 从第二条起可能已经 drift，绝不是共同 Frozen baseline。
- score_after：对本条 waveform 做固定一次 TENT 更新后的分数。

所有分数沿用 spoof logit - bonafide logit，大于源 tau0 判 spoof。
四个 worker 的 source reference 必须在 1e-5 内一致；共同 Frozen-Waveform
采用 episodic worker 的 source reference。cache Frozen 不参与本研究。

drift_before = current_pre_adapt_score - source_frozen_score；
instant_update = score_after - current_pre_adapt_score；
total_delta = score_after - source_frozen_score。
aggregate 的 abs_score_delta、分类别 score_delta 与 helpful/harmful flips 均基于 total_delta。
熵使用现有 softmax prediction_entropy；grad_norm 为 step 前梯度 L2；
parameter_delta_norm 为本条更新前后选中参数 L2；parameter_distance_from_source
为每条更新后相对 source 选中参数的 L2。记录 parameter_distance_before 验证窗口内连续性。
norm 使用 double 累加做诊断，不转换模型精度、不影响更新；不要求每条都出现非零梯度。
adaptation_applied 仅在实测 parameter_delta_norm > 0 时为 true；没有更新则如实记录。

final/max/mean_parameter_drift 基于逐条更新后、下次 reset 前的距离。
Spearman 对 ties 用平均 rank，常数序列返回 null（未定义），不增加 scipy。
runtime 含本条 reset、载入、forward/backward 和诊断（CUDA 同步计时）；
source_reference_runtime 单独记录，启动/模型加载耗时不混入每条 runtime。

## 工作站流程

`bash experiments/protocol_tta/run_protocol_tta.sh --dry-run` 仅打印配置，无 Python、
模型载入或输出目录创建。实际运行要求四个不同 CUDA GPU，默认 GPU_LIST=0,1,2,3，
按 episodic/continual/reset32/reset128 一一卡分配。资产根来自 TTA_ASSET_ROOT，默认仓库根。
使用现有 outputs_v2/ssl_aasist/frozen/bundle.json 与 resources；不复制模型实现。

preflight → 唯一目录 → 四卡 32 条 smoke → 校验 PASS → 全新进程从 source 开始
四卡完整 target10 → 等待全部成功 → aggregate → metrics.csv/summary.json/report.md。
preflight 在导入 CUDA 前强制 manifest 顶层 role=select、count=3178、records 长度=3178，
并沿用逐条严格 label-free 白名单、split_role=select、sample_id 唯一检查。

episodic 的 32 条 smoke 之后，在同一卡、同一模型/source snapshot 上逐条 reset，
通过旧生产 `tent_audio.score_current` / `tent_audio.tent_adapt` 重放相同前32条 waveform。
结果保存为 `smoke/episodic/legacy_tent_scores.jsonl`，不加载 cache 参考、不另建模型、
不运行完整旧 TENT target10。check-smoke 必须验证 ID/顺序完全一致，并分别比较
source_frozen_score/score_before_update 与旧 waveform before、score_after 与旧 after。
三项绝对容差都固定 1e-5；缺结果、非有限值或超限均 SMOKE FAIL，禁止进入正式序列。
最大差值与实际验收结论写入 smoke_check.log；本机未运行此真实 CUDA parity。
smoke 参数与正式相同，只截取 manifest 前 32 条；因此 reset32/128 在 smoke 内只重置一次，
正式窗口边界由纯逻辑 260/300 条测试覆盖。smoke 不消费 target labels。

smoke/正式聚合均拒绝缺失/重复/重排 ID、非有限字段、错误 reset、连续性破坏、BN 改变、
四卡 checkpoint/config/reference 不一致和整个序列无适应。单条零更新允许如实记录。
run_config 保存 manifest path/count、checkpoint/detector state 路径、baseline/source run ID、
source tau0、scope 名称和全部参数。遵循本分支 AGENTS.md，不新增摘要算法或 SHA 门禁；
逐条精确比对 manifest ID 顺序，但不声称检测同名资产的外部原地替换。
runner 保存 Git commit、配置、preflight/smoke/各 worker/aggregate 日志与 status.txt。
失败不进入后续阶段；不支持部分序列恢复（恢复可能改变 Adam 和历史状态）。

本轮不宣称 target90 泛化或显著性。单一顺序的结果只能回答该固定顺序、该初始化和
TENT 配置下协议的影响；发现 drift 也不能自动解释为有效目标域学习。

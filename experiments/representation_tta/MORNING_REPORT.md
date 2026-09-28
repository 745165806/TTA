# MORNING_REPORT — earlier-layer SSL-AASIST TTA

**找到可复现开发候选了吗？NO。最佳 TTA 方法：NONE。** 在真实 source fit 上完成了较早层 adapter 和同预算静态对照的 GPU 训练，并在 ITW target10 与 WaveFake development 完成全量评价。预先固定的目标学习率 `1e-4` 在两个域均使同一 checkpoint 的 ON 表现低于 OFF。一次明确受开发反馈触发的 `1e-5` 修订缩小了退化，但没有提供跨域 TTA 收益。没有候选满足补齐 seeds29/47/71 的晋级条件；这三个 seed **NOT_RUN**，因此没有跨 seed 稳定性结论。

## 核心比较

AUC 越高越好；EER 为百分数，越低越好。`Static` 是相同 5 epoch、3,565 步的 BCE 源训练对照；`OFF` 是 BCE+SupCon 方法臂的源 checkpoint；`ON` 从同一 checkpoint 在每个无标签目标域单独更新。两次目标配置复用同一 source checkpoint。分数方向均为较大值偏 spoof。

| 域 | 配置 | Frozen AUC / EER% | Static AUC / EER% | OFF AUC / EER% | ON AUC / EER% | ON−OFF ΔAUC / EER改善pp |
|---|---|---:|---:|---:|---:|---:|
| ITW target10, n=3178 | 原始 LR 1e-4 | 0.963309 / 9.857 | 0.949575 / 11.237 | 0.958451 / 10.153 | 0.942183 / 11.237 | −0.016268 / −1.084 |
| WaveFake, n=4096 | 原始 LR 1e-4 | 0.915003 / 15.771 | 0.931225 / 14.795 | 0.932015 / 14.697 | 0.930777 / 14.990 | −0.001238 / −0.293 |
| ITW target10 | 开发反馈修订 LR 1e-5 | 同上 | 同上 | 同上 | 0.957643 / 10.096 | −0.000809 / +0.057 |
| WaveFake | 开发反馈修订 LR 1e-5 | 同上 | 同上 | 同上 | 0.931385 / 14.795 | −0.000630 / −0.098 |

原始配置 ON 相对 Frozen：ITW ΔAUC −0.021125、EER 改善 −1.380 pp（相对 EER 下降 −14.0%）；WaveFake ΔAUC +0.015774、EER 改善 +0.781 pp（相对下降 +4.95%）。相对匹配 Static：ITW ΔAUC −0.007392、EER 改善 0；WaveFake ΔAUC −0.000448、EER 改善 −0.195 pp。WaveFake 相对 Frozen 的正差来自源训练：ON 比自己的 OFF 更差，不能归因于测试时适配。

修订 ON 相对 Frozen：ITW ΔAUC −0.005666、EER 改善 −0.239 pp；WaveFake ΔAUC +0.016382、EER 改善 +0.977 pp。相对匹配 Static：ITW ΔAUC +0.008068、EER 改善 +1.141 pp；WaveFake ΔAUC +0.000160、EER 改善 0。但 ITW AUC 仍低于 Frozen，两个域相对自身 OFF 的 AUC 都下降，故修订也不构成有效 TTA 候选。

### 源阈值工作点与不确定性

每臂阈值仅由 source select 得到，目标标签仅在逐样本分数保存后用于开发评价。原始配置在 ITW 的 Frozen / Static / OFF / ON FPR、FNR 依次为 `0.2597/0.0305`、`0.3716/0.0104`、`0.3558/0.0113`、`0.3756/0.0104`；WaveFake 依次为 `0.0073/0.5059`、`0.0103/0.4404`、`0.0073/0.4634`、`0.0161/0.4150`。修订 ON 的 ITW FPR/FNR 为 `0.3864/0.0096`，WaveFake 为 `0.0200/0.3848`。完整 source 阈值、balanced accuracy 及每臂指标见两次 `metrics.csv`。

原始配置的配对 bootstrap：ITW ON−OFF ΔAUC 95% CI `[−0.01994,−0.01253]`，EER 改善比例 CI `[−0.01732,−0.00445]`；WaveFake ON−OFF ΔAUC CI `[−0.00179,−0.00065]`，EER 改善比例 CI `[−0.00537,+0.00146]`。修订后 ITW ON−OFF ΔAUC CI `[−0.00138,−0.00026]`；WaveFake `[−0.00104,−0.00022]`。WaveFake 按固定的 2,048 个 content-pair 组抽样；ITW 仅按音频 ID 抽样，未声称独立 speaker/session 区间。置信区间是同一开发集样本不确定性，不能代替多 seed 或独立目标域。

## 实际训练、资源和故障

- 真实冻结模型边界：项目任务训练的 XLS-R→LL 输出 `[B,201,128]`；一个真实 waveform 的完整 logits 与 LL 后端回放完全相同。新增 128→64→128 residual adapter 共 16,576 参数，位于 AASIST 时间聚合与最终 160D 表征之前。XLS-R、LL 和原后端冻结。
- GPU LL 提取：source fit 25,380、source select 11,723、ITW target10 3,178、WaveFake development 4,096，总计 44,377 条，每条三视图。四张 RTX A6000 分片，全部 PASS；771.7 秒，提取峰值约 2.27 GB/卡；新缓存约 13 GB，留在工作站。
- seed13 两臂各训练 5 epoch、3,565 optimizer steps、228,160 样本呈现、25,380 unique source fit ID；source select 11,723 条，仅用于选 checkpoint 和阈值。Static 和方法臂都选 epoch5；每臂约 568 秒、峰值约 2.74 GB、0 数值失败。`training_curve.csv` 保存 clean/noise/FIR 验证曲线。
- 测试协议为 **offline batch-transductive TTA**：每个域从同一 source checkpoint 重新开始，使用全部无标签三视图特征更新一个共享 adapter，最后用同一个最终状态重评全域。没有跨域状态，不是在线或单样本协议。测试时保留 source fit 攻击家族原型，故不声称严格 source-free。
- 原始配置 ITW/WaveFake 各 100/128 次目标更新，适配 14.10/17.94 秒，最终 ON 推理 1.11/1.43 秒，评价进程峰值约 4.08 GB，0 数值失败。adapter 参数相对源 checkpoint 的 L2 位移分别为 0.732/1.099。修订仍为 100/128 次更新，适配 14.05/18.02 秒，位移 0.102/0.148，0 数值失败。
- 第一次评价在 source select 阶段因 `inference_mode` 原型张量进入 `cdist` 反向而退出 1；失败日志和部分输出保留。适配入口复制常量原型后，重试完整 PASS。修复不改变目标标签隔离或源 checkpoint。
- 曾有自动审批以旧 `AGENTS.md` 默认本机限制拒绝全量提取两次；本次用户明确批准后才执行。先前 smoke/阻塞记录保留。主工作区及历史结果未覆盖。

## 已有成熟对照和方法边界

同一固定 ITW/WaveFake 开发资源上的历史 T3A-batch port 在 `experiments/overnight_tta/results/overnight_20260927_seed13/metrics.csv`：ITW AUC/EER `0.963041/10.009%`，WaveFake `0.920668/15.137%`。它按作者模板规则作离线整域估计与重评分，不代表原论文在线协议。此前 160D 源训练 residual 路线也已真实完成 10 epoch；它的目标更新在 ITW 降低表现、在 WaveFake 没有实质增益。本次较早层路线补齐了此前 GPU 受阻的独立位置比较，但同样未产生可信的 ON−OFF 收益。

本轮 seed13 首配置在两域均未达到预先写入 `config.json` 的自身 OFF 改善 `ΔAUC ≥ 0.005` 或 `EER 改善 ≥ 0.005`，修订也未达到。故 29/47/71 不运行，不能称本方法“跨 seed 不稳定”或“稳定无效”；准确结论仅限于实际 seed13、两个目标配置及固定开发集。修订使用了开发反馈，不是独立验证。此前 DCH 四 seed WaveFake 正结果另有固定 source descriptor 消融表明其增益主要来自静态源训练，详见历史报告；不将其作为本次 TTA 成功。

## 工件与运行命令

根目录：`/media/dell/data/fakeAudioDection/TTA/.worktrees/exp-representation-tta`，分支 `exp-representation-tta`。本轮唯一 run：`experiments/representation_tta/results/rep_ll_20260928_seed13/`。大 LL 缓存和 `.pt` checkpoint 留在该路径；主结果分别在 `evaluation_retry_inference_tensor_fix/` 和 `evaluation_revision_lr1e5/`，含 `metrics.csv`、ITW/WaveFake 逐样本分数、最终目标 adapter、源阈值、runtime、paired bootstrap、summary。`logs/` 保留四域分片、训练、失败评价、两次成功评价日志。精确环境：`/home/dell/anaconda3/envs/tta/bin/python`，`PYTHONPATH=src:.`，`OMP_NUM_THREADS=2`。

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/dell/anaconda3/envs/tta/bin/python experiments/representation_tta/extract_all.py --run-id rep_ll_20260928_seed13
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/dell/anaconda3/envs/tta/bin/python experiments/representation_tta/train.py --config experiments/representation_tta/config.json --run-id rep_ll_20260928_seed13 --arm static_bce --seed 13 --gpu 0
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/dell/anaconda3/envs/tta/bin/python experiments/representation_tta/train.py --config experiments/representation_tta/config.json --run-id rep_ll_20260928_seed13 --arm task_supcon --seed 13 --gpu 1
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/dell/anaconda3/envs/tta/bin/python experiments/representation_tta/evaluate.py --config experiments/representation_tta/config.json --run-id rep_ll_20260928_seed13 --seed 13 --gpu 0 --evaluation-id evaluation_retry_inference_tensor_fix
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/dell/anaconda3/envs/tta/bin/python experiments/representation_tta/evaluate.py --config experiments/representation_tta/config_revision_lr1e5.json --run-id rep_ll_20260928_seed13 --seed 13 --gpu 0 --evaluation-id evaluation_revision_lr1e5
```

上述原始命令创建独占目录；重现需指定新的 run/evaluation ID，不覆盖已存在输出。`python -m compileall -q experiments/representation_tta` 和报告数值核对在最终提交前执行。`target90` 标签/指标访问：**NO**。本分支最终 holdout 访问：**NO**。

**下一步只推荐一项：停止本轮无效的 residual-anchor TTA 路线，保留完整负结果，不用它开启目标最终评测。**

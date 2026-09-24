# 本机工程验证（2026-09-24）

本机 Windows，无 GPU、真实音频、checkpoint 或 feature cache；未运行真实 adaptation 或评价。
当前修改基于 ebe7afa，包含上一阶段 f8a7051（episodic-continuous protocol study）。
用户未跟踪 ZIP 保留，未加入提交；已有实验结果未改动。

| 命令 / 检查 | 状态 | 实际证据 |
|---|---|---|
| `python -m compileall -q src experiments workers tests` | PASS | exit 0，无错误输出；仅静态编译，系统 Python 3.14 |
| `python experiments/oracle_diagnosis/run.py --dry-run` | PASS | exit 0，candidate_count=113，target=target10，groups=4，target90_used=false，网格与合同一致；未加载科学运行库/资源 |
| Git Bash `bash -n experiments/oracle_diagnosis/run_oracle_diagnosis.sh` | PASS | exit 0；首次沙箱内 signal pipe 权限错误，随后沙箱外只读语法检查成功 |
| `git diff --check` | PASS | exit 0，无输出 |
| `conda env list` | 检查完成 | 仅 base、researchstudio，无 tta |
| `conda run -n tta python -m pytest tests/unit/test_oracle_diagnosis.py tests/unit/test_target10_selection_audit.py -q` | NOT_RUN | 启动 exit 1：NoWritableEnvsDirError；未执行 pytest。本机也未发现 tta 环境；未改用其他环境或安装依赖 |
| 工作站资源 preflight / smoke / 113×3178 sweep | NOT_RUN | 本机无真实资源，留工作站执行 |
| EER/AUC、5-fold、regret、相关性科学结果 | NOT_RUN | 没有生成或提交假实验结果 |

新增测试覆盖：候选数/唯一性、Frozen 唯一性、四组精确覆盖、EER/AUC/K/lr/rho tie-break、regret、
Spearman ties/常量与 SciPy 对照、分层固定 folds、仅训练折选参、分数精确覆盖、失败先于标签读取、
Frozen identity/非有限值/重复 ID 拒绝、路径与不可覆盖约束、历史 cache 普通字段检查、
synthetic 全流程报告，以及真实 production guarded EP 的 rho 和 score sink 对照。
这些测试代码存在不代表已运行通过。

工作站执行正式实验前建议先运行上述 tta pytest 命令。runner 自带 mandatory 32-sample × 113-candidate smoke，
检查成功才自动开始正式 sweep；所有日志保存到本次 run 目录，不覆盖历史输出。

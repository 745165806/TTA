# 本地开发验证 — 2026-09-24

工程检查与 CUDA 实验严格分开。未生成任何本地实验 result；测试合成分数仅写入
TemporaryDirectory，退出时删除。未执行真实 waveform、模型构建、训练或目标评价。

## Git 保护与基线

开始时实际运行 `git status`、`git branch -vv`、`git log -8 --oneline`、`git remote -v`。
当时处于 main，HEAD=8ed0a2f，origin=https://github.com/745165806/TTA.git。
工作树没有已跟踪修改，只有未跟踪的 `EP-TTA_v0.1.0_execution_pack.zip`，保留且不提交。
当时日志为 8ed0a2f、cebe3f8、285636c、e3d55bd、2fc5925。

初次 fetch 被 sandbox 拒绝写入 .git/FETCH_HEAD；提升权限后网络传输卡住。
本任务开始前另有 fetch PID 9604。用户明确授权停止后确认 Git 进程全部退出才恢复 checkout。
连接重置原因是 Git 未自动使用已配置的 Windows 代理；通过单次 `-c http.proxy=...`
使用该代理成功执行 fetch，不修改全局代理，不 force、不 reset、不改用户文件。
曾使用 partial/sparse checkout 避免大量历史对象传输；最终已恢复完整工作树，
core.sparseCheckout=false。partial clone 元数据仅为本机 Git 设置，不进入提交。

`exp-audio-native-tta` pull --ff-only 成功并与 origin 一致。
中断恢复中生成的空 `exp-next-stage-tta` 从 8ed0a2f fast-forward 到
cfc2be3c5e4df69ff574c6149377889f883cc4ba，无重写已有提交。
正式开发差异全部以此 base 审查。预期的 cfc2be3/fca51fc/2b70257/e46826b/a860c09 均已核对。

## 已执行的轻量检查

实际输出及退出码见 [local_checks.txt](local_checks.txt)。

| 命令 | 结果 |
|---|---|
| python -m compileall -q src experiments workers tests | PASS，exit 0 |
| python -m unittest discover -s tests/unit -p test_protocol_reset_schedule.py -v | PASS，11 tests，exit 0 |
| git diff --check | PASS，exit 0；暂存后另查 git diff --cached --check |
| Git Bash: bash -n experiments/protocol_tta/run_protocol_tta.sh | PASS，exit 0 |
| Git Bash: bash experiments/protocol_tta/run_protocol_tta.sh --dry-run | PASS，exit 0 |

unittest 使用本机已有 Python 3.14，没有安装依赖。初次运行受中断工作树和
Windows sandbox 临时目录权限影响失败，完成 checkout 并提升运行权限后重跑通过。
Bash 初次受 sandbox signal pipe 权限限制，提升权限后使用已安装 Git Bash 检查。
这些是开发工具权限处理，没有增加科研代码 CPU 分支；工作站仍使用 conda tta。

## 自审

- continual 模型只构建、configure、snapshot 一次；序列 index=0 reset，此后模型和 Adam 状态持续。
- episodic 每条 source reset + 新 Adam，使用旧 TENT 的 softmax entropy、更新顺序及 grad-enabled score path。
- reset32/128 边界覆盖 0/31/32/127/128/256，计数定义见合同；四协议共用 reset_info。
- 四卡精确比对 manifest 原始 ID 顺序、配置、参数名、源 tau0 和资产引用。
- model.eval + 原 source BN buffers；逐条调用已有 assert_bn_buffers_unchanged。
- worker 只读取严格无标签 manifest，objective 只接收 waveform；标签只在 aggregate 后验评价。
- source Frozen 是独立 source waveform pass；continual pre-adapt 不作为 Frozen。
- smoke 通过后自动启动全新正式序列；worker 非零退出导致整体 FAIL，不聚合不完整结果。
- 新代码无 Windows 固定路径；资产根读取 TTA_ASSET_ROOT；CUDA 不可用立即失败。
- 没有运行/读取 target90 标签、没有改旧实验结果、没有新增算法/scope/lr sweep。
- 新增内容仅 experiments/protocol_tta 与一个纯逻辑测试；旧 src、audio-native 脚本保持不变。

## 未验证

真实 CUDA 32 样本 smoke、正式四卡 target10、SSL-AASIST 实际梯度/显存、科学效果：
全部 NOT_RUN。必须在工作站运行 runner；本地 PASS 只表示以上已执行工程检查通过。
同名资产被外部原地替换的检测不在本项目现行非摘要合同保证范围内。

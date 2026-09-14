# EP-TTA v0.1.0 本轮真实测试报告

## 2026-09-14 L4/缓存/机制实现追加执行

| 检查 | 实际命令 | 退出码与结果 | 证据 |
|---|---|---|---|
| 全套 D/S/T/C 合成验收 | `PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python -m pytest -q --junitxml=docs/test_logs/20260914-l4/pytest-final4.xml` | 0；**206 passed in 3.97s** | [pytest-final4.log](test_logs/20260914-l4/pytest-final4.log)、[pytest-final4.xml](test_logs/20260914-l4/pytest-final4.xml) |
| core/schema/worker 语法审计 | `/home/dell/anaconda3/envs/py310/bin/python scripts/check_sources.py` | 0；core Python 3.10、worker Python 3.7 AST 目标、14 schemas PASS | [source_audit_final.log](test_logs/20260914-l4/source_audit_final.log) |
| py38 worker 编译 | py38 `py_compile` 三个 bridge/compat 文件 | 0；Python 3.8.20 三文件 PASS | [worker_py38_compile.log](test_logs/20260914-l4/worker_py38_compile.log) |
| pinned AASIST 结构前向 | Python 3.10 构建 commit `a04c...d1` 的作者 `Model`，CPU FP32 输入 `[1,64600]` | 0；embedding `[1,160]`、logits `[1,2]`、finite、无源码补丁 | [aasist_architecture_smoke.log](test_logs/20260914-l4/aasist_architecture_smoke.log) |

本轮合成 PASS 覆盖 source manifest 角色/哈希、native 类别/权重、EER 选模、smoke/export 拒绝、checkpoint 身份、DDP 加权梯度公式、pickle-free cache、exact shard coverage、P0/P1 机制、Fisher 自动微分对照、U/M/tau/static R、published-port 阻塞合同、label-free inference job、score seal 后评价。AASIST 结构前向使用真实作者源码但随机初始化，不能证明任务训练或性能。

SSL-AASIST 因 `generic_ssl_initialization=null` 且当前 py38 环境无系统 fairseq 安装（作者仓库虽含 vendored revision），结构/梯度/训练记 `NOT_RUN/DEFERRED_REMOTE`。本机无可见 CUDA，源训练、DDP 运行、checkpoint 真恢复、finalize/export parity、真实 cache/适配和 published port parity 均未执行，不计 PASS。

## 2026-09-14 统一标签追加执行

| 检查 | 实际命令 | 退出码与结果 | 证据 |
|---|---|---|---|
| 统一标签全量校验 | `PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml validate-labels --labels fakedata/current --check-audio` | 0；17 manifests，1,415,725 条 VALID | [validate_labels.log](test_logs/20260914-labels/validate_labels.log) |
| ALLM-DF 逐行兼容 | `/home/dell/anaconda3/envs/py310/bin/python scripts/check_allmdf_compat.py --labels fakedata/current --reference-dir /media/dell/data/fakeAudioDection/ALLM-DF/data/manifests` | 0；5 份共享清单 PASS | [allmdf_compat.log](test_logs/20260914-labels/allmdf_compat.log) |
| 全套验收 | `/home/dell/anaconda3/envs/py310/bin/python -m pytest -q` | 0；**173 passed in 3.89s** | [pytest.log](test_logs/20260914-labels/pytest.log) |
| 源码与 schema 审计 | `/home/dell/anaconda3/envs/py310/bin/python scripts/check_sources.py` | 0；PASS | [source_audit.log](test_logs/20260914-labels/source_audit.log) |

真实标签包及隔离边界见 [LABEL_PACK_REPORT.md](LABEL_PACK_REPORT.md)。这些结果证明数据读取与标签预处理工程状态，不代表 group/split/preprocess 已完成正式人工审核，也不代表源模型训练或科学实验已经执行。

## 2026-09-14 Linux L3 追加执行

| 检查 | 真实命令 | 退出码 / 结果 | 日志 |
|---|---|---|---|
| 全套合成验收 | `/home/dell/anaconda3/envs/py310/bin/python -m pytest -q --junitxml=docs/test_logs/20260914-l3/pytest.xml` | 0；**167 passed in 3.92s** | [pytest.log](test_logs/20260914-l3/pytest.log)、[pytest.xml](test_logs/20260914-l3/pytest.xml) |
| Python/schema 审计 | `/home/dell/anaconda3/envs/py310/bin/python scripts/check_sources.py` | 0；Python 3.10 语法、14 schema、依赖与 base 配置通过 | [source_audit.log](test_logs/20260914-l3/source_audit.log) |
| ASVspoof root 只读 inventory | `PYTHONPATH=src ... python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml inspect-data --datasets asvspoof2019_la --out /tmp/.../inventory3` | 0；915,694 音频，7 个协议/sidecar 候选；`label_contract_approved=false` | [inventory_cli.log](test_logs/20260914-l3/inventory_cli.log)、[inventory_summary.log](test_logs/20260914-l3/inventory_summary.log) |
| 旧 wheel 交付检查 | `/home/dell/anaconda3/envs/py310/bin/python scripts/check_delivery.py` | 1；当前 worktree 无 `dist/ep_tta-0.1.0-py3-none-any.whl`，未为本次数据接口重建 wheel | [delivery_check.log](test_logs/20260914-l3/delivery_check.log) |

本次 D01–D10 均为合成合同/工程验收；只读 inventory 证明当前路径可枚举，不证明 raw/label/group/preprocess/split 已审核。未生成真实 staging/snapshot，未启动训练。原 L2 环境本次有 PyTorch/PyYAML，因此旧报告中相应 NOT_RUN 由本次执行证据补充，不改写旧日志。

测试日期：2026-09-13；Windows 11 本机，合成 fixture 与配置文件。数据、音频、模型、远程服务器均未参与。所有 PASS 仅对应下列真实检查；缺依赖未被计入 PASS。

## 命令与结果

| 检查 | 真实命令 | 退出码 / 结果 | 日志 |
|---|---|---|---|
| 单元、协议和权限 | `py -3 -m pytest tests/unit tests/contracts -q --junitxml=docs/test_logs/20260913-local/pytest.xml` | 0；**121 passed, 2 skipped in 1.18s** | [pytest.log](test_logs/20260913-local/pytest.log)、[pytest.xml](test_logs/20260913-local/pytest.xml) |
| Python 3.10 语法与配置 schema | `py -3 scripts/check_sources.py` | 0；31 个 Python 文件语法通过、14 个 JSON schema 可读、base 严格校验通过 | [source_audit.log](test_logs/20260913-local/source_audit.log)；交付脚本后增补检查另见 source_audit_final.log |
| 本地 wheel 构建 | 使用已存在的 bundled Python 3.12.14 调用 `setuptools.build_meta.build_wheel('dist')`，完整绝对命令在日志 | 首次 1（沙箱临时目录权限）；授权后 0 | [wheel_build.log](test_logs/20260913-local/wheel_build.log)、[wheel_build_retry.log](test_logs/20260913-local/wheel_build_retry.log) |
| 打包内容与参考完整性 | `py -3 scripts/check_delivery.py` | 首次沙箱读取被拒；授权重试发现审计脚本未兼容 Windows METADATA 换行，修复为 splitlines；最终结果见日志和清单 | [delivery_manifest.json](test_logs/20260913-local/delivery_manifest.json)、[delivery_check_final.log](test_logs/20260913-local/delivery_check_final.log) |

主要测试解释器：`C:\Python314\python.exe`，Python 3.14.3。已有 pytest 可用；未安装 torch、numpy、PyYAML、setuptools。另一套已存在的 bundled Python 3.12.14 有 setuptools/wheel/NumPy，但也没有 PyTorch/PyYAML；仅用其执行本地打包，未安装到系统环境。

本地 pytest 的临时目录及构建工具最终临时目录受到沙箱限制，因此在工具请求授权后重跑对应本地命令；该权限未用于真实资源、网络或训练。早期 schema/测试守卫问题和临时目录失败均保留在 [development_attempts.md](test_logs/20260913-local/development_attempts.md)，不隐去失败运行。

## 两项跳过，明确记 NOT_RUN

1. `tests/unit/test_ep.py` 整个模块：缺少 CPU PyTorch。生产串行、batch、设计附录参考均**未执行**；不声明数值等价、梯度正确、数值隔离或性能通过。
2. `test_yaml_duplicate_if_available`：缺少 PyYAML。内置 `.yaml` 的 JSON 子集路径、重复键与非有限 JSON 数值拒绝已执行；普通 YAML Loader 分支没有执行。

重跑数学测试：`python -m pytest tests/unit/test_ep.py -q`。该模块定义了 T01–T10/T13–T14、C03/C07/C10 的合成检验：float32/64，B=1/2/7/128，K=0/1/5，三视图固定；行列方向、Nd 分母和首步解析梯度；单侧间隔初值/激活；Frobenius 投影；原视图评分；正交补/换基/零空间；A→B→A、乱序、尾批次、分片；sum/共享 R 错误反例；上下文冻结、数值回退与坏样本隔离。

语法扫描使用 `ast.parse(feature_version=(3,10))`，不是 Python 3.10 运行时测试；Python 3.7 worker 尚未实现，因此无 bridge 兼容性 PASS。单元测试的 FrozenModelBundle 样本只验证字段拒绝规则，不是正式冻结模型或血缘审核。

## 结论边界

本轮已验证基础 CLI、schema/配置、协议与权限原语，并构建项目 wheel。**L2 数值验收仍为 NOT_RUN；所有真实数据、预处理、源训练、冻结 parity、双 A6000、基线、最终评分与论文实验均未运行。** 未计算任何 EER/AUROC/FPR/TPR 或科学结论。

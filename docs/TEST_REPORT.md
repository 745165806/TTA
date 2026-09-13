# EP-TTA v0.1.0 本轮真实测试报告

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

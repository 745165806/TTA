# 最终验收前的开发运行记录

以下为本轮工具输出摘录与修复说明，不是逐字完整终端日志；完整最终运行另存 pytest.log / pytest.xml。

1. `py -3 -m pytest tests/unit tests/contracts -q`：退出码 1。报告 `27 failed, 32 passed, 2 warnings, 1 error`，并出现 pytest INTERNALERROR。原因：nullable enum 的 JSON Schema 未把 null 放进 enum；测试全局 monkeypatch Path.exists/stat 导致错误报告过程也被拦截。修复 schema 生成器与实际 schema，并把 stat 守卫缩到占位资源路径。
2. `py -3 -m pytest tests/unit tests/contracts -q --maxfail=3`：退出码 1。`118 passed, 1 skipped, 2 warnings, 2 errors`。剩余两项为 pytest 在临时目录 `pytest-of-m7451` 的 WinError 5 权限错误，另有 cache 权限警告。
3. 同一命令在用户已授权的沙箱外重跑：退出码 0。`120 passed, 1 skipped in 1.05s`；PyYAML 分支因缺依赖跳过。
4. 后续增加独立训练配置解析测试与张量测试定义后，执行最终 pytest 命令；结果见同目录 pytest.log 和 pytest.xml。数值模块因 PyTorch 缺失没有执行。
5. wheel 内容审计初次在沙箱内读取被拒；授权重试后审计脚本对 METADATA 使用 LF 子串断言，未兼容 Windows CRLF，触发 AssertionError。修复为 splitlines 行匹配；这不是项目版本不一致。两次失败日志为 delivery_check.log / delivery_check_retry.log，最终记录另存 delivery_check_final.log。

没有安装任何缺失依赖，也没有用沙箱外权限访问真实资源、联网或训练。

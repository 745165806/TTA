# EP-TTA v0.1.0 工单1核查与交付

日期：2026-09-16  
状态：已完成只读现场核查、必要代码修补、合成测试与工件归档；停止在恢复路线、GPU预算和新目标snapshot发布审批点。未启动GPU、未恢复训练、未产生新目标模型分数，未commit/push/tag。

## 1. 现场与训练身份

进程表未发现 `source_train_bridge`、对应run目录或`torch.distributed.run`训练进程。两个run均无`run.json`，故真实状态是STOPPED、未TRAINED、未FINALIZED；目录名不是已写入的科学run_id。按原worker算法可预计算完成时ID，但只能标作`deterministic_completion_run_id`，不能倒写为已记录run_id。没有`failure.json`或其他停止事件，停止原因保留`UNKNOWN_NO_STOP_RECORD`。

| 字段 | AASIST | SSL-AASIST |
|---|---|---|
| run目录 | `.../aasist_source/full-gpu2-b48-001` | `.../ssl_aasist_source/full-gpu1-b48-001` |
| seed / phase | 13 / full | 13 / full |
| 原锁定预算 | 100 epochs | 100 epochs |
| last边界 | epoch 13；14 epochs；global_step 7406 | epoch 90；91 epochs；global_step 48139 |
| 每epoch optimizer steps | 529 | 529 |
| recipe SHA-256 | `4fe26f8e2cd71f35db19d8aabe3bd3da90c55b60b9dc8e1faa7dc8a2fe182ec7` | `f378a02ec126634a30b63b27169793529991bd18bf8fdd06c8cc4f433b326646` |
| snapshot SHA-256 | `50d893131ce99752759fb14a121bd17c29a9782eddb0d8560ac458d336315fe0` | 同左 |
| 初始化 | native | generic XLS-R `b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9` |
| 原训练patch SHA-256 | `00aea39322b51c49a70d691d9daa8cad0be99da460a5043f566da1b016e721a7` | `292765fe9e6b468324264741a60473a191fd4942cf26e5c69e10ab2b299d1cde` |
| 原worker SHA-256 | `ff77caa3f20065e665bb8a1bedb4eb8dc13fa7609d62dfb451fa4ffbdbe7f631` | 同左 |
| 当前worker SHA-256 | `5a6e7658468cab0fac7c023e531d60ef0bfc46558984e1bed11244379d471b4c` | 同左 |
| best | epoch 12；`89cf9c65...da295`；source_val EER 0.0029154519 | epoch 5；`5dea8eb1...81d08`；source_val EER 0 |
| last | `d712df09...c3b19` | `efb3c259...c3b19` |
| scheduler | cosine；last_epoch/global_step=7406，optimizer LR与`_last_lr`一致 | none；LR 1e-6 |

两个last均实际含model、optimizer、scheduler、AMP scaler、epoch/global step、单rank Python/NumPy/Torch/CUDA RNG、sampler epoch/policy/world size、best选择历史所需的外部metrics及身份字段。AASIST global step=`14×529`，SSL=`91×529`，未发现重复epoch或漏step偏移。float32的scaler state为空字典，符合未启用FP16 scaler。原best/last文件及sidecar hash匹配；后来增加逐epoch不可变规则不否定旧run，也未补造旧epoch文件。

实际import路径门禁会同时校验字节hash和绝对路径。原worker字节仍可从Git HEAD精确恢复，但归档副本路径与锁定路径不同；当前路径上的worker字节又已经改变。因此路线A目前不能在不替换用户工作区文件的条件下安全执行。当前worker不能称原run exact resume。

## 2. 归档

独立普通文件副本位于忽略版本控制的 `artifacts/private/ticket1-20260916/`，共约7.1 GiB。原文件与归档checkpoint设备相同但inode不同，SHA-256一致，不是硬链接。归档包含两个完整run目录、Git HEAD源码tar、原worker、修补前工作区diff、预检/数据审计/曝光工件及hash清单。最终45文件清单`SHA256SUMS.final2.txt`自身SHA-256为`d264f69b3703fd5e4d6a11914b1df15ecd9d39a0e1dd3a1b7cdd2a755c369fcf`；当前代码逐文件清单自身SHA-256为`609509b227fab53cffaf2597d81e05ecf74fae909bb12e01a5d9b56f233a4291`。

## 3. 源snapshot审计

`snapshot-5731a8d70a8b8fa4997d`共121,461条，canonical SHA-256为`50d893...15fe0`。样本ID、音频相对路径、角色manifest均恰好覆盖一次；无跨角色speaker/group；官方边界为fit=train、source_val/select/cal0/audit=dev、control_test=eval。各角色类别计数与批准记录一致：fit 2,580/22,800；source_val 686/4,968；select 1,008/10,512；cal0 574/4,332；audit 280/2,484；control_test 7,355/63,882（canonical 0/1，即bonafide/spoof）。fit/dev覆盖A01–A06，control_test覆盖A07–A19。

限制与处理：

- 所有121,461条`input_sha256`均为null，`parent_id`也均为null；所以协议级唯一性通过，但内容重复和原录音家族关系必须写UNKNOWN，不能据此宣称跨数据集独立。
- `source_val`最大speaker组占46.66%，`cal0`53.77%，`select`22.90%；这是只有20个dev speaker且坚持组不拆分产生的代表性限制，不是发现跨角色泄漏。
- `audit`只有2个speaker，最大组占95.44%，对稳健损伤审计构成实质问题。现有训练只使用fit/source_val，故训练与已有选模证据仍可复用。建议另建child snapshot，仅重分select/cal0/audit并保持fit/source_val/control_test逐条不变；若批准，只需重建尚未运行的源资源/先导，不需要重训当前底座。未发布该候选。
- DESIGN规定audit只评价未缓存源样本受损，不得进loss、回退或步数选择。未找到运行工件引用audit manifest；这只能写“未观察到引用”，访问史仍为UNKNOWN，不能写“从未查看”。

结论：保留现有snapshot及既有批准作为当前训练身份；在R5/R6实际使用audit前解决上述下游角色候选，不执行旧40/20/40或8/4/8重划指令。

## 4. 四个主要目标与曝光

| 目标 | 本轮真实状态 | 血缘/曝光边界 |
|---|---|---|
| 2019 LA | control_test 71,237，官方eval全量，已存在两模型best诊断 | 已查看效果，永久作为已暴露标准参照；不得回包装为盲测 |
| 2021 DF | 音频及仅ID trial list 611,829条存在，trial list SHA-256 `d65befec...61d`；官方13列metadata未找到 | adapter已实现并有反例测试；真实metadata审计NOT_RUN，snapshot发布BLOCKED |
| In-the-Wild | `meta.csv` SHA-256 `2e404150...795`；完整31,779条，bonafide/spoof=19,963/11,816；音频缺失0 | 本地内容已绑定，但作者release/version标识仍UNKNOWN；与源各角色的speaker/record-ID重合为0，原录音家族UNKNOWN；未看模型效果 |
| 2021 LA | 官方metadata SHA-256 `6cc64a94...ba8`；精确eval=148,176，排除progress 16,464与hidden 16,926；bonafide/spoof=14,816/133,360；音频缺失0 | 与源开发角色speaker/record-ID重合为0；67个speaker与已暴露2019 control_test完全重合，全部148,176条属于这些speaker；确切原录音家族仍UNKNOWN；未看模型效果 |

2019曝光账本记录了两个best的model scoring与用户报告的effect view。另一个目录虽标记`ssl ... last-epoch90`，其真实`run.json`绑定`best.pt`的epoch-5 hash，metrics/scores亦与best运行相同，因此不能作为last epoch90效果证据。账本还记录了本轮LA/ITW metadata只读访问；7条事件使用hash链及独立head锚点，清空/截断测试会拒绝。

四个目标从本轮起均禁止参与模型、epoch、recipe、seed、EP参数和部署阈值选择。WaveFake与CodecFake_Xie仍是R7前扩展决策，不阻塞当前源训练处理。

## 5. 代码：复用、补丁与阻塞

复用：现有严格配置/schema框架、数据staging/snapshot、目标manifest白名单、训练worker、checkpoint状态、source_val选模、checkpoint evaluation及不可变epoch保存修改。

本轮新增/修补：

- `resume-preflight`：只读检查checkpoint/sidecar/job、代码hash与import路径、模型/optimizer/scheduler/scaler/RNG/sampler、epoch/global-step、LR与best证据；输出严格schema报告。
- `audit-source-snapshot`：官方partition×role×class、攻击、speaker/group、最大组占比、覆盖/重复/hash及UNKNOWN血缘。
- `audit-target-metadata`：LA严格8列、DF严格13列、In-the-Wild严格表头；只允许固定eval范围，管理员sidecar和无标签运行字段分离，并审计源/2019 speaker与record-ID关系。
- `record-exposure`：严格schema、追加式hash链、head锚点；不能静默清空历史。
- AASIST/SSL-AASIST后续模板默认由100改为80；旧锁定100轮recipe字节不变。
- 保留并验收用户已有的不可变epoch checkpoint与任意兼容checkpoint诊断入口；不覆盖旧文件。

尚缺实现：获批路线B的实际parent→child迁移执行器，以及SSL行政finalize迁移合同。本轮在路线审批点停止，未用宽松校验或force绕过。外部阻塞：DF官方metadata、In-the-Wild作者版本标识、跨数据集原录音家族映射、新目标snapshot审批。

## 6. 恢复路线建议与预算

AASIST推荐B：以last epoch13为parent，新child从epoch14执行到79，总预算80；恢复全部状态，但保持原100轮cosine horizon，避免把scheduler重建成80轮曲线造成LR跳变。epoch14起写不可变文件，绝不伪造0–13；child不得称原run exact resume。剩余66 epochs、34,914 optimizer updates，按现有日志估计约7.3 A6000 GPU小时，新checkpoint约0.25 GiB。

SSL-AASIST推荐B行政迁移：不继续训练。父run真实是100轮锁定合同下已完成91轮，新的80轮默认不能追溯套用；保留并披露父谱系，以原严格改善/最早并列规则选epoch5，建立新child finalization身份。若不批准该迁移，只能选A完成原100轮（还需9轮且使用旧身份执行链）或C从头80轮；不能把91/80写成已完成80轮run。

正式恢复前的有限真实测试必须在隔离目录比较：同一parent连续跑两个完整epoch，与跑一epoch→退出进程→恢复→再跑一epoch。当前合同只支持epoch边界，所以每epoch529 updates，每模型两条分支共2,116次update执行；预计两模型合计约1.5 GPU小时。比较样本/增强顺序、loss、参数/buffer、optimizer、scheduler/LR、scaler、RNG和global step。该测试及正式GPU预算均未获授权、NOT_RUN。

计划工件为 `docs/TICKET1_RECOVERY_PLAN_20260916.json`，SHA-256：`07e5318c87f286f71ec6df4e484431bc317c2ec6ba7b62df7ed01231b01785a6`。

## 7. 测试与入口

Python环境：`/home/dell/anaconda3/envs/py310/bin/python`，Python 3.10.20；worker额外用py38编译。实际命令与日志：

| 检查 | 退出码 | 结果/证据 |
|---|---:|---|
| `PYTHONPATH=src ... -m pytest -q --junitxml=.../pytest-final2.xml` | 0 | 227 passed；`docs/test_logs/20260916-ticket1/pytest-final2.log` |
| `... scripts/check_sources.py` | 0 | Python/schema/worker审计PASS，18 schemas；`check-sources-final2.log` |
| py38 `py_compile` 三个worker/compat | 0 | PASS；`py38-worker-compile-final2.log` |
| 三个新CLI的真实`--help` | 0 | `cli-*-help.log` |
| 真实source snapshot审计 | 0 | `source-snapshot-audit-final.log` |
| 真实2021 LA与ITW metadata审计 | 0 | `asvspoof2021-la-audit-final.log`、`in-the-wild-audit-final.log` |
| 两模型exact/current与child/80只读预检 | 0（报告自身为BLOCKED/APPROVAL_REQUIRED） | `*-exact-current-r3.log`、`*-child-80-final.log` |

覆盖的要求包括hash变化拒绝、缺恢复状态拒绝、epoch/global-step与LR/scheduler偏移、旧epoch文件不可覆盖、LA/DF错误schema/subset拒绝、标签极性、源训练角色越权、运行manifest标签隔离、曝光历史清空/截断拒绝。新增字段均经过严格schema、解析、运行、序列化/hash与反例测试。第一次全套复跑因合成resume fixture缺`phase`出现2个失败，修正fixture后最终227项全通过；失败日志保留为`pytest-final.log`。真实跨进程GPU恢复、正式训练、R4导出parity和新目标效果均为NOT_RUN。

## 8. 下一阶段与批准模板

获批后先实现并复审B迁移执行合同，再只运行有限真实恢复测试；通过后AASIST完成child 14–79，SSL只做行政迁移，随后FINALIZED与R4。R4只做冻结导出/parity，不看新目标效果。R5–R9和任何目标snapshot发布仍需后续独立批准。

批准模板（必须原样包含计划hash）：

> 我批准执行 `ticket1-recovery-20260916`，绑定计划 SHA-256 `07e5318c87f286f71ec6df4e484431bc317c2ec6ba7b62df7ed01231b01785a6`：采用AASIST路线B（parent epoch13→child总预算80，保留100轮scheduler horizon）和SSL-AASIST路线B行政finalize迁移（不再训练），先执行两模型合计不超过1.5 GPU小时的隔离恢复验收；验收通过后AASIST正式续训预算不超过7.3 GPU小时；不授权新目标snapshot发布、目标评分或R5–R9。

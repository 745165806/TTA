# 远程数据、源训练与冻结导出顺序 v0.1.0

数据、源训练、冻结导出和特征 cache worker 已实现；真实训练/适配仍须在 GPU 服务器逐阶段核验。本流程不会自动批准合同、下载数据/模型、SSH、安装 CUDA或串联启动 full training。

当前主机的私有路径示例为 `configs/paths.fakedata.private.yaml`（Git 忽略）。在 GPU 服务器路径不同时复制并修改私有文件，不能改公共模板来伪装路径一致。

## 统一标签预处理

当前主机另有 Git 忽略的 `configs/unified_labels.fakedata.private.json`，其中每个来源均显式记录协议列、标签映射和音频路径模板。以下命令生成不可覆盖的 CSV 标签包；前七列 `utt_id,path,label,attack,speaker,split,source` 兼容 ALLM-DF，后续列满足 EP-TTA 的稳定 ID、规范标签和协议血缘要求：

```bash
PYTHONPATH=src python -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml \
  prepare-labels --spec configs/unified_labels.fakedata.private.json \
  --out fakedata/unified_v0.1.0-r2

PYTHONPATH=src python -m eptta.cli validate-labels \
  --labels fakedata/unified_v0.1.0-r2
```

`manifests/*.csv` 只包含构建时确认存在且标签显式映射为 `0/1` 的音频；`quarantine.csv` 保存缺音频、未映射值及重复/冲突，不允许静默丢弃。`index.json` 绑定每个协议和输出清单的 SHA-256。该包保留官方 split，`split_role` 仍为 `unassigned`；训练前还需按主流程审核来源组、角色划分和预处理，不能直接把目标 eval 标签传入适应器。

```bash
export PYTHONPATH=src
export EPTTA_WORK_ROOT=/media/dell/data/fakedata/eptta_work
mkdir -p "$EPTTA_WORK_ROOT/inventory" "$EPTTA_WORK_ROOT/contracts" "$EPTTA_WORK_ROOT/reviews" "$EPTTA_WORK_ROOT/staging" "$EPTTA_WORK_ROOT/splits" "$EPTTA_WORK_ROOT/snapshots"

python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  inspect-data --datasets asvspoof2019_la --out "$EPTTA_WORK_ROOT/inventory/asv2019_la"

python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  propose-data-contract --inventory "$EPTTA_WORK_ROOT/inventory/asv2019_la/inventory.json" \
  --out "$EPTTA_WORK_ROOT/contracts/asv2019_la.generated.proposal.json"
```

本仓库的 `configs/data_contracts/asvspoof2019_la.observed.yaml.example` 是基于当前只读抽样形成的更具体候选，明确选择三份 2019 CM 协议，避开同目录的 2021 key 与用户转换 sidecar。审核者应核对完整列语义、音频解析、标签覆盖和来源组，并为仍为 null 的 `group.payload.source_mapping_ref` 填入真实证据引用；不补该字段时 lock 会按设计失败。将 `configs/reviews/approval.yaml.example` 复制到私有 work root，补齐证据且主动把 `accepted` 改为 true。然后分别发布 lock：

```bash
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  approve-contract --kind raw --proposal configs/data_contracts/asvspoof2019_la.observed.yaml.example \
  --review "$EPTTA_WORK_ROOT/reviews/asv2019_raw.json" --out "$EPTTA_WORK_ROOT/contracts/raw.lock.json"
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  approve-contract --kind label --proposal configs/data_contracts/asvspoof2019_la.observed.yaml.example \
  --review "$EPTTA_WORK_ROOT/reviews/asv2019_label.json" --out "$EPTTA_WORK_ROOT/contracts/label.lock.json"
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  approve-contract --kind group --proposal configs/data_contracts/asvspoof2019_la.observed.yaml.example \
  --review "$EPTTA_WORK_ROOT/reviews/asv2019_group.json" --out "$EPTTA_WORK_ROOT/contracts/group.lock.json"

python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  stage-data --inventory "$EPTTA_WORK_ROOT/inventory/asv2019_la/inventory.json" \
  --raw-contract "$EPTTA_WORK_ROOT/contracts/raw.lock.json" \
  --label-policy "$EPTTA_WORK_ROOT/contracts/label.lock.json" \
  --group-policy "$EPTTA_WORK_ROOT/contracts/group.lock.json" \
  --out "$EPTTA_WORK_ROOT/staging/asv2019_la"

python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  propose-splits --staging "$EPTTA_WORK_ROOT/staging/asv2019_la" \
  --policy configs/split_policies/asvspoof2019_la.yaml.example \
  --out "$EPTTA_WORK_ROOT/splits/asv2019_la.proposal.json"
```

审核 `counts`、类别/来源组分布及角色独立性后，再用 `approve-contract --kind split` 生成 `split.lock.json`，最后：

```bash
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  build-manifests --staging "$EPTTA_WORK_ROOT/staging/asv2019_la" \
  --split-plan "$EPTTA_WORK_ROOT/splits/split.lock.json" \
  --out "$EPTTA_WORK_ROOT/snapshots/asv2019_la_initial"
```

snapshot 中 fit/source_val manifest 含规范标签；其他角色的运行 manifest 不含标签，评价标签写到独立目录。任何 parse error、缺音频、越界路径、同 ID/同路径标签冲突均显式报告；严格合同下阻止发布。现有输出不可覆盖。

随后封存源训练入口：

```bash
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  seal-source-manifests --snapshot "$EPTTA_WORK_ROOT/snapshots/asv2019_la_initial" \
  --out "$EPTTA_WORK_ROOT/source_manifests/asv2019_la_initial"

# 将候选复制到 work root 后核对 decode/unit，再用独立 review 发布 preprocess lock。
python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml \
  approve-contract --kind preprocess --proposal configs/preprocess/source.yaml.example \
  --review "$EPTTA_WORK_ROOT/reviews/source_preprocess.json" \
  --out "$EPTTA_WORK_ROOT/contracts/source_preprocess.lock.json"
```

## 源模型结构、配方与训练

核心 CLI 使用 Python 3.10；作者/音频 worker 可绑定兼容环境。本机当前可用示例为 py38，但远程须以 preflight 的真实依赖为准：

```bash
export EPTTA_CORE_PY=/home/dell/anaconda3/envs/py310/bin/python
export EPTTA_SOURCE_PY=/home/dell/anaconda3/envs/py38/bin/python

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml inspect-model \
  --model-id aasist_source --mode architecture \
  --out "$EPTTA_WORK_ROOT/model_inspection/aasist"

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml resolve-training-recipe \
  --model-id aasist_source \
  --snapshot "$EPTTA_WORK_ROOT/snapshots/asv2019_la_initial" \
  --preprocess "$EPTTA_WORK_ROOT/contracts/source_preprocess.lock.json" \
  --out "$EPTTA_WORK_ROOT/contracts/aasist.recipe.proposal.json"

# 检查作者 commit、类别映射、batch/累积、单卡或 DDP、增强偏离后人工批准。
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml approve-contract --kind recipe \
  --proposal "$EPTTA_WORK_ROOT/contracts/aasist.recipe.proposal.json" \
  --review "$EPTTA_WORK_ROOT/reviews/aasist_recipe.json" \
  --out "$EPTTA_WORK_ROOT/contracts/aasist.recipe.lock.json"

# smoke 是独立 run，不能 finalize/export，也不会自动续跑 full。
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml train-source \
  --recipe "$EPTTA_WORK_ROOT/contracts/aasist.recipe.lock.json" --phase smoke \
  --run-output "$EPTTA_WORK_ROOT/training_runs/aasist_source/smoke-001" \
  --worker-python "$EPTTA_SOURCE_PY"

# smoke 日志/显存/覆盖审核通过后，才单独启动 full。
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml train-source \
  --recipe "$EPTTA_WORK_ROOT/contracts/aasist.recipe.lock.json" --phase full \
  --run-output "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001" \
  --worker-python "$EPTTA_SOURCE_PY"

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml finalize-training \
  --run "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001" \
  --out "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001/finalized.json"

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml export-frozen \
  --training-manifest "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001/finalized.json" \
  --out "$EPTTA_WORK_ROOT/trained_models/aasist_source/full-001" \
  --worker-python "$EPTTA_SOURCE_PY"
```

中断后只从同一 run 的 `checkpoints/last.pt` 恢复；recipe/snapshot/world size 任一变化都会拒绝 exact resume：

```bash
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml resume-source \
  --run "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001" \
  --checkpoint "$EPTTA_WORK_ROOT/training_runs/aasist_source/full-001/checkpoints/last.pt" \
  --mode epoch_boundary --worker-python "$EPTTA_SOURCE_PY"
```

每个完成验证的epoch都会保留为`checkpoints/epoch-NNNN.pt`，并同时保存同名`.json` hash sidecar；训练器不自动删除旧epoch。`last.pt`和`best.pt`只是原子更新的便利别名，分别指向最近epoch和source_val EER最优epoch的相同内容。`metrics.jsonl`引用不可变epoch文件，因此finalize/export不会以可变别名作为科学身份。规划磁盘时应按“单个完整checkpoint大小 × 最大epoch数”预留空间。

要启用源训练 DDP，在 **recipe proposal 审核前** 把 runtime 固定为 `strategy=ddp, world_size=2, physical_gpu_ids=[0,1]`；worker 用 `torch.distributed.run` 启动。训练采样策略须显式选择全局尾部 drop 或 repeat。验证由 rank 0 对解除 DDP 的模型完整执行并广播。该设置不传播到 EP：EP 每条 R 独立、`loss_b.sum()`，禁止跨卡归约。

SSL-AASIST 同样执行上述链路，但必须先在私有 paths 中填写 `generic_ssl_initialization`，其内容 hash 会进入 recipe。当前主机该字段为 null，所以 SSL 命令应阻塞；不得用仓库中的 `best_SSL_model*.pth` 或其他鉴伪 checkpoint 填充。

冻结 export 会用 source_val 中一条无目标侧信息的 fixture 核对作者 logits、`head(embedding)`、导出 `w,b` 和 R=0 分数，并检查 module mode/BN buffer 不变。通过后才能编写 LOCKED inference plan，使用：

```bash
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml extract --plan "$EPTTA_EXTRACT_PLAN" \
  --worker-slot 0 --worker-count 2 --worker-python "$EPTTA_SOURCE_PY"
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml extract --plan "$EPTTA_EXTRACT_PLAN" \
  --worker-slot 1 --worker-count 2 --worker-python "$EPTTA_SOURCE_PY"
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli merge-cache \
  --plan "$EPTTA_MERGE_PLAN" --out "$EPTTA_WORK_ROOT/feature_cache/merged"
```

cache 身份包含输入 manifest、自有 checkpoint/head、wrapper、preprocess、probe、seed、dtype 和数值模式。新目标数据只追加新 ID；checkpoint/hash 改变会生成新 cache identity，不覆盖旧 cache。

fit/cal0 编码前先用 `prepare-inference-manifest` 生成不含标签的只读视图；标签继续留在 snapshot 的 sidecar。分别合并 fit/cal0 cache 后，用 `configs/source_artifact_plan.yaml.example` 的私有 LOCKED 副本构建资源，再运行机制 suite：

```bash
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml prepare-inference-manifest \
  --snapshot "$EPTTA_WORK_ROOT/snapshots/asv2019_la_initial" --role fit \
  --out "$EPTTA_WORK_ROOT/inference_manifests/asv2019_fit"

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli build-artifacts \
  --plan "$EPTTA_SOURCE_ARTIFACT_PLAN" --cache-index "$EPTTA_FIT_CACHE" \
  --out "$EPTTA_WORK_ROOT/artifacts/ep_resources"

PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli run-suite \
  --plan "$EPTTA_SUITE_PLAN" --suite-id mechanism_v0.1.0 --phase select \
  --run-output "$EPTTA_WORK_ROOT/runs/mechanism-select"
```

`run-suite` 为每个 method 写独立无标签分数目录。每个目录必须先 `seal-scores --run DIR`，之后 `evaluate --run DIR --labels SIDECAR --threshold TAU --out FILE` 才能读取标签。confirmatory 阶段额外要求 `--frozen-spec` 且 method/参数逐项一致。

### 任意兼容参数的ASVspoof2019 LA eval评测

`evaluate-checkpoint`与训练完成状态解耦，可评测项目checkpoint（`model_state`）、原生state dict、`state_dict`字段或带`module.`前缀的DDP参数。模型参数必须严格匹配指定结构。输出目录包含`run.json`、`metrics.json`、`scores.jsonl`和`progress.jsonl`，并记录输入与产物hash；已存在目录拒绝覆盖。

```bash
PYTHONPATH=src "$EPTTA_CORE_PY" -m eptta.cli --profile remote_a6000 \
  --paths configs/paths.fakedata.private.yaml evaluate-checkpoint \
  --model-id aasist_source --checkpoint /absolute/path/to/weights.pt \
  --protocol /absolute/path/to/ASVspoof2019.LA.cm.eval.trl.txt \
  --audio-dir /absolute/path/to/ASVspoof2019_LA_eval/flac \
  --asv-scores /absolute/path/to/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
  --out /absolute/path/to/new/evaluation-output --gpu-id 2 --batch-size 48 \
  --evaluation-tag asvspoof2019-la-eval --worker-python "$EPTTA_SOURCE_PY"
```

评测分数固定为`spoof_logit - bonafide_logit`，因此越大越偏spoof。`metrics.json`报告EER、AUROC、阈值0下的混淆矩阵/平衡准确率及逐攻击EER；提供`--asv-scores`时还用hash绑定的ASVspoof2019实现报告官方EER百分比和min t-DCF。阈值0指标只是给定参数头的诊断，EER/AUROC不依赖该阈值。

以上均是可执行接口与远程核验顺序；当前没有 GPU 吞吐、显存、训练收敛、真实适配或鉴伪效果结论。

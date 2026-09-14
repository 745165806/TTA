# 统一标签包报告（2026-09-14）

## 当前可用产物

- 稳定入口：`fakedata/current` → `unified_v0.1.0-r2`
- 索引：`fakedata/current/index.json`
- 可读取清单：`fakedata/current/manifests/*.csv`
- 隔离清单：`fakedata/current/quarantine.csv`
- 格式：`eptta_unified_labels_csv_v1`
- 标签：`0=bonafide`，`1=spoof`

CSV 前七列为 `utt_id,path,label,attack,speaker,split,source`，与 ALLM-DF 的 CSV reader 兼容。扩展列记录 `sample_id、dataset_id、dataset_release、root_key、audio_relpath、original_label、label_mapping_status、source_group_id、group_quality、protocol_file、protocol_sha256、protocol_line、split_role、status、schema_version`。

## 有效记录

| dataset_id | bonafide (0) | spoof (1) | 可读取合计 | 隔离 |
|---|---:|---:|---:|---:|
| asvspoof2019_la | 12,483 | 108,978 | 121,461 | 0 |
| asvspoof2021_la | 18,452 | 163,114 | 181,566 | 0 |
| in_the_wild | 19,963 | 11,816 | 31,779 | 0 |
| codecfake_xie | 220,547 | 860,372 | 1,080,919 | 173,551 |
| **总计** | **271,445** | **1,144,280** | **1,415,725** | **173,551** |

Codecfake Xie 的隔离记录全部为协议中有标签但本机没有对应音频：A3 缺 49,274 条，C7 缺 124,277 条。它们未写入可读取 manifest，完整保留在 `quarantine.csv`，未静默丢弃。

以下当前候选尚未生成有效标签清单：

- `asvspoof2021_df`：本地可见 trial 文件只有 utterance ID，尚未绑定经审核的 DF label key。
- `wavefake`：当前只有压缩包和下载日志，尚未解压音频及权威元数据。

`unified_v0.1.0` 是首次审计产物，其中尚未纳入 In-the-Wild 的原始值 `bona-fide`，不得用于实验；`current` 只指向修正后的 `unified_v0.1.0-r2`。

## 验证证据

| 检查 | 命令 | 退出码 | 结果 | 日志 |
|---|---|---:|---|---|
| 全量 hash、ID、标签及音频存在性 | `PYTHONPATH=src python -m eptta.cli --profile remote_a6000 --paths configs/paths.fakedata.private.yaml validate-labels --labels fakedata/current --check-audio` | 0 | 17 manifests；1,415,725 条 VALID | `docs/test_logs/20260914-labels/validate_labels.log` |
| ALLM-DF 逐行兼容 | `python scripts/check_allmdf_compat.py --labels fakedata/current --reference-dir /media/dell/data/fakeAudioDection/ALLM-DF/data/manifests` | 0 | 5 个共享 manifest 的 ID/路径/标签/split 全部一致 | `docs/test_logs/20260914-labels/allmdf_compat.log` |
| 完整测试 | `python -m pytest -q` | 0 | 173 passed in 3.89s | `docs/test_logs/20260914-labels/pytest.log` |
| 源码/schema 审计 | `python scripts/check_sources.py` | 0 | PASS | `docs/test_logs/20260914-labels/source_audit.log` |

## 使用边界

这些 CSV 已可由 ALLM-DF 风格 loader 或 `eptta.data.iter_unified_manifest()` 流式读取。它们保存的是可信数据构建侧标签，不是允许送入目标适应器的输入。当前 `split_role=unassigned`；正式训练前仍需审核来源组、split policy 和 preprocess policy，并发布不可变 DatasetSnapshot。目标测试运行时必须使用去标签投影，评价器再从独立标签侧读取 `canonical_label`。

# EP-TTA v0.1.0 R1 合同批准、staging 与 split proposal 记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15T12:49:59+08:00
- Verification Status: VERIFIED（合同发布与 staging/split 工程产物；split 尚未批准）
- Version Label: r1_approval_split_proposal_v1

## 审批边界

用户明确批准：`raw/label/group/preprocess`，审核人 `mm`。该授权不包含 `split`、`recipe`、训练或适配。四类 review 使用各自的真实证据文件；锁定后 payload hash 与批准前 proposal 一致。

| kind | lock | 文件 SHA-256 | payload SHA-256 |
|---|---|---|---|
| raw | `/media/dell/data/fakedata/eptta_work/contracts/raw.lock.json` | `e451fc19e096f680971709030d1d03d45e12379ff9593604c4280a7a40100f1d` | `64b915bf168598e36242b0ce8688caf2880aee1b4469b0b386287b2923fc87a5` |
| label | `/media/dell/data/fakedata/eptta_work/contracts/label.lock.json` | `a639f5dc40db0efa3c2c77eab8541cca7ed4dfb63cd21ecf5ac8c4ba79fa1ecb` | `b382ec7e6939667dbfee30141515a5ce9a481ec8e198468d87529560c6a36dd1` |
| group | `/media/dell/data/fakedata/eptta_work/contracts/group.lock.json` | `c97b0dddd6a1ecafbd0fa43a1a92843b2f5ab5817e4b8b7d05648aa3905efc59` | `47f5904687248aa10973a75894b033dacf8bf718170d9473c42dc67324d9b775` |
| preprocess | `/media/dell/data/fakedata/eptta_work/contracts/source_preprocess.lock.json` | `85e7ce59184961d0358f25ef8df63f2b986d8086bfd1bcdfa67d1be5c3d8e250` | `6d1eaf49fe7fbf0fcf931572751c0bf6a9178ee89d04bd6b70f6f27f72528f6a` |

审批命令退出码均为 0；日志：`docs/test_logs/20260915-r1-approval-locks.log`，SHA-256 `b83a3ed3ac6bcbdd4782abd2a4ce3a892f586b378cb273e6464ddf4879893134`。

## Staging

- 路径：`/media/dell/data/fakedata/eptta_work/staging/asvspoof2019_la`
- 状态：`STAGED`
- staging ID：`staging-71275b8662b3509ce95a`
- 记录：121,461；`parse_errors=0`、`label_conflicts=0`、`quarantine=0`
- `staging.json` SHA-256：`35d08e21e85f17bccbc8239c213969638129b5e29dc28dc127502fd373c7a375`
- `records.jsonl` SHA-256：`1db0c6c435e4b550ed07ac7ca008abebfed81f0424497b86150aaf98740bf656`
- 命令退出码：0；日志 SHA-256：`75cacccfeeb97c686f3353cd683bbb07d991844df4bb32d20f505f823406f0b2`

## Split proposal（等待单独批准）

proposal：`/media/dell/data/fakedata/eptta_work/splits/asvspoof2019_la.proposal.json`，SHA-256 `e07ddd4d994317907872a75042a00b9064e1f442091ca076adf1b2567fc941db`。assignments SHA-256：`afcb818fcf4c07205edf1ce1f943c36f01adc00ccf88dcc750f2fd92ba06cd8f`。

| role | groups | bona fide | spoof | total |
|---|---:|---:|---:|---:|
| fit | 20 | 2,580 | 22,800 | 25,380 |
| source_val | 5 | 686 | 4,968 | 5,654 |
| select | 8 | 1,008 | 10,512 | 11,520 |
| cal0 | 5 | 574 | 4,332 | 4,906 |
| audit | 2 | 280 | 2,484 | 2,764 |
| control_test | 67 | 7,355 | 63,882 | 71,237 |

审计结果：121,461 条恰好一次分配；duplicate ID、缺失 staging ID、跨角色 source group 均为 0；train 只进入 fit，dev 只进入 source_val/select/cal0/audit，eval 只进入 control_test。由于只有 20 个 dev speaker group，按组 hash 分配不会精确达到样本级 40/30/20/10 比例，当前 `source_val` 小于 `select`；必须人工接受该具体分布后才能发布 split lock。

split 生成和审计命令退出码均为 0；日志：`docs/test_logs/20260915-r1-propose-splits.log`、`docs/test_logs/20260915-r1-split-audit.log`。尚无 DatasetSnapshot、source manifest、recipe 或训练 run。

"""Source-selected thresholds and offline batch-transductive target development evaluation."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from model import AuxiliaryLoss, adapt_head
from train_meta import read_jsonl, source_head


def read_selected(cache_path, selected_ids, checkpoint_ref, source_run_id):
    cache = FeatureCache(cache_path)
    ident = cache.index["identity"]
    if (ident["checkpoint_ref"] != checkpoint_ref or ident["source_run_id"] != source_run_id or
            cache.index["feature_dim"] != 160 or cache.index["num_views"] != 3):
        raise ValueError("target cache/frozen model mismatch")
    cache_data = {}
    wanted = set(selected_ids)
    if len(wanted) != len(selected_ids):
        raise ValueError("duplicate selected ID")
    for ids, block in cache.iter_chunks():
        cache_data.update((sid, block[i].copy()) for i, sid in enumerate(ids) if sid in wanted)
    if set(cache_data) != wanted:
        raise ValueError("selected target cache coverage differs")
    array = np.stack([cache_data[sid] for sid in selected_ids])
    if array.shape != (len(selected_ids), 3, 160) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("malformed selected target features")
    return torch.from_numpy(array), cache.cache_id


def score_arms(views, source_w, source_b, checkpoints, config):
    # This interface receives only frozen features, model state and fixed config.
    result = {"frozen": (views[:, 0] @ source_w + source_b).detach().numpy()}
    erm = checkpoints["erm"]
    result["erm"] = (views[:, 0] @ erm["weight"] + erm["bias"]).detach().numpy()
    for arm in ("meta_bce", "meta_rank"):
        state = checkpoints[arm]
        aux = AuxiliaryLoss(hidden=config["hidden"])
        aux.load_state_dict(state["aux"])
        aux.eval()
        w, b = adapt_head(aux, views, source_w, source_b, steps=config["inner_steps"],
                          lr=config["inner_lr"])
        result[arm] = (views[:, 0] @ w + b).detach().numpy()
        result[arm + "_off"] = result["frozen"].copy()
    if not all(np.isfinite(scores).all() for scores in result.values()):
        raise ValueError("nonfinite score")
    if not (np.array_equal(result["meta_bce_off"], result["frozen"]) and
            np.array_equal(result["meta_rank_off"], result["frozen"])):
        raise AssertionError("disabled adaptation did not restore source head")
    return result


def eer_threshold(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], labels[order]
    pos, neg = int(labels.sum()), int(len(labels) - labels.sum())
    if not pos or not neg:
        raise ValueError("source validation threshold needs both classes")
    tp = fp = 0
    best = (float("inf"), None)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        tp += int(sorted_y[index:end].sum())
        fp += end - index - int(sorted_y[index:end].sum())
        far, frr = fp / neg, (pos - tp) / pos
        threshold = float((sorted_scores[end - 1] + sorted_scores[end]) / 2) if end < len(scores) else float(sorted_scores[end - 1] - 1e-6)
        candidate = (abs(far - frr), threshold)
        if candidate < best:
            best = candidate
        index = end
    return best[1]


def bootstrap(scores, labels, groups, repeats, seed):
    by_group = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    keys = sorted(by_group)
    rng = random.Random(seed)
    aucs, eers = [], []
    for _ in range(repeats):
        indices = [i for group in rng.choices(keys, k=len(keys)) for i in by_group[group]]
        sampled_y = [labels[i] for i in indices]
        if min(sampled_y) == max(sampled_y):
            continue
        result = binary_metrics([float(scores[i]) for i in indices], sampled_y, 0.0)
        aucs.append(result["auroc"])
        eers.append(result["eer"])
    return (np.quantile(aucs, [0.025, 0.975]).tolist(),
            np.quantile(eers, [0.025, 0.975]).tolist())


def paired_bootstrap(rows, arm, reference, repeats, seed):
    by_group = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["content_group"]].append(index)
    keys = sorted(by_group)
    labels = [int(row["label"]) for row in rows]
    rng = random.Random(seed)
    auc_gain, eer_gain = [], []
    for _ in range(repeats):
        indices = [i for key in rng.choices(keys, k=len(keys)) for i in by_group[key]]
        y = [labels[i] for i in indices]
        if min(y) == max(y):
            continue
        a = binary_metrics([float(rows[i][arm]) for i in indices], y, 0.0)
        b = binary_metrics([float(rows[i][reference]) for i in indices], y, 0.0)
        auc_gain.append(a["auroc"] - b["auroc"])
        eer_gain.append(b["eer"] - a["eer"])
    return np.quantile(auc_gain, [0.025, 0.975]).tolist(), np.quantile(eer_gain, [0.025, 0.975]).tolist()


def summarize(run_ids, repeats):
    base = Path(__file__).parent / "results"
    metrics = {}
    for run_id in run_ids:
        with (base / run_id / "metrics.csv").open() as stream:
            metrics[run_id] = {(r["domain"], r["arm"]): r for r in csv.DictReader(stream)}
    lines = ["# Meta-Rank Head-TTT — 四 seed 开发汇总", "",
        "完整源训练、源域选模和目标开发评价：seed 13、29、47、71。每个 seed 的 ERM/Meta-BCE/Meta-Rank 使用相同 96 个训练 episode、96 次外层更新、32 个有标签 query/episode；两个 Meta 使用相同初始化、5 步无标签内层更新，仅外层排序权重不同。",
        "", "| 域 | 方法 | AUC 均值（范围） | EER 均值（范围） |", "|---|---|---:|---:|"]
    for domain in ("itw_target10", "wavefake_dev"):
        for arm in ("frozen", "erm", "meta_bce", "meta_rank"):
            auc = [float(metrics[run][domain, arm]["auc"]) for run in run_ids]
            eer = [float(metrics[run][domain, arm]["eer"]) for run in run_ids]
            lines.append("| %s | %s | %.6f (%.6f–%.6f) | %.6f (%.6f–%.6f) |" %
                (domain, arm, float(np.mean(auc)), min(auc), max(auc), float(np.mean(eer)), min(eer), max(eer)))
    lines.extend(["", "## 成对增益", "",
        "正 ΔAUC 表示排序改善；正 EER gain 表示错误率下降。WaveFake 的 95% bootstrap 区间以 audio_id content-pair 为重采样单位，ITW 以原始音频 ID 为单位。",
        "", "| 域 | seed | Meta-Rank 对照 | ΔAUC | 95% CI | EER gain | 95% CI |",
        "|---|---:|---|---:|---:|---:|---:|"])
    for domain in ("itw_target10", "wavefake_dev"):
        for run_id in run_ids:
            with (base / run_id / (domain + "_scores.csv")).open() as stream:
                rows = list(csv.DictReader(stream))
            rank = metrics[run_id][domain, "meta_rank"]
            for reference in ("frozen", "erm", "meta_bce"):
                ref = metrics[run_id][domain, reference]
                auc_ci, eer_ci = paired_bootstrap(rows, "meta_rank", reference, repeats, 2026)
                lines.append("| %s | %s | %s | %+.6f | [%+.6f, %+.6f] | %+.6f | [%+.6f, %+.6f] |" %
                    (domain, run_id.rsplit("seed", 1)[-1].replace("_v2", ""), reference,
                     float(rank["auc"]) - float(ref["auc"]), *auc_ci,
                     float(ref["eer"]) - float(rank["eer"]), *eer_ci))
    lines.extend(["", "## 判断", "",
        "源域元训练已完成，内层可微更新的辅助网络梯度非零且有限。Meta-Rank 没有稳定优于 Meta-BCE；两个目标域上均未胜过同预算 Source-augmented ERM。ITW 的 EER 小幅变化没有伴随稳定 AUC 增益，WaveFake 的 Meta-Rank 平均 AUC/EER 均劣于 Frozen。关闭适配恢复 source head，故 Meta 与 Frozen 的分数差确实来自测试时更新；目前没有可信的正收益。",
        "", "开发结论仅适用于既有 ITW target10 与 WaveFake 4,096 条 content-pair 开发缓存。模型/阈值选取只用 source select；target90 和最终 holdout 均未运行。GPU 驱动不可用，复用只读 160D 缓存并在 tta 环境 CPU 完成训练。",
        ""])
    destination = Path(__file__).parent / "report.md"
    destination.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "summary": str(destination), "runs": run_ids}), flush=True)


def run(config_path, run_id):
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(1)
    run_dir = Path(__file__).parent / "results" / run_id
    if not run_dir.is_dir():
        raise ValueError("training run absent")
    if (run_dir / "metrics.csv").exists():
        raise FileExistsError("evaluation already exists for this run")
    bundle, source_w, source_b = source_head(Path(config["bundle_ref"]))
    checkpoints = {arm: torch.load(run_dir / (arm + ".pt"), map_location="cpu", weights_only=True)
                   for arm in ("erm", "meta_bce", "meta_rank")}
    if any(state["config"] != config for state in checkpoints.values()):
        raise ValueError("checkpoint/config mismatch")
    root = Path(config["project_root"])
    source_labels = read_jsonl(root / "data/manifests_v2/asv2019_la/labels/select.jsonl")
    source_ids = sorted(source_labels)
    source_x, _ = read_selected(Path(config["select_cache"]), source_ids,
                                bundle["checkpoint_ref"], bundle["source_run_id"])
    source_scores = score_arms(source_x, source_w, source_b, checkpoints, config)
    y_source = [source_labels[sid]["canonical_label"] for sid in source_ids]
    thresholds = {arm: eer_threshold(source_scores[arm], y_source)
                  for arm in ("frozen", "erm", "meta_bce", "meta_rank")}
    (run_dir / "source_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    itw_manifest = json.loads(Path(config["itw_target10"]).read_text())
    itw_ids = [row["sample_id"] for row in itw_manifest["records"]]
    wave_manifest = json.loads(Path(config["wavefake_select"]).read_text())
    wave_ids = [row["sample_id"] for row in wave_manifest["records"]]
    domains = (("itw_target10", itw_ids, Path(config["itw_cache"]), itw_manifest),
               ("wavefake_dev", wave_ids, Path(config["wavefake_cache"]), wave_manifest))
    all_metrics, reports = [], []
    for domain, ids, cache_path, manifest in domains:
        x, cache_id = read_selected(cache_path, ids, bundle["checkpoint_ref"], bundle["source_run_id"])
        # One shared head per domain; every call restarts at the source head.
        scores = score_arms(x, source_w, source_b, checkpoints, config)
        # Target labels are opened only after every arm's score is fixed.
        if domain == "itw_target10":
            labels = [row["label"] for row in manifest["records"]]
            groups = ids
        else:
            # Audited WaveFake release mapping: R=real, WF1..WF7=fake.
            code_map = {"R": 0, **{"WF%d" % i: 1 for i in range(1, 8)}}
            labels = [code_map[row["sample_id"].rsplit(":", 1)[1]] for row in manifest["records"]]
            groups = [row["audio_id"] for row in manifest["records"]]
            if len(set(groups)) != manifest["content_groups"]:
                raise ValueError("WaveFake content-pair group mismatch")
        if len(ids) != manifest["count"] or set(labels) != {0, 1}:
            raise ValueError("development assignment coverage/labels invalid")
        score_file = run_dir / (domain + "_scores.csv")
        with score_file.open("x", newline="") as stream:
            fields = ["sample_id", "content_group", "label", "frozen", "erm", "meta_bce", "meta_rank", "meta_bce_off", "meta_rank_off"]
            writer = csv.DictWriter(stream, fields, lineterminator="\n")
            writer.writeheader()
            for i, sid in enumerate(ids):
                writer.writerow({"sample_id": sid, "content_group": groups[i], "label": labels[i],
                                 **{arm: float(values[i]) for arm, values in scores.items()}})
        metrics_by_arm = {}
        for arm in ("frozen", "erm", "meta_bce", "meta_rank"):
            metric = binary_metrics(scores[arm].tolist(), labels, thresholds[arm])
            ci_auc, ci_eer = bootstrap(scores[arm], labels, groups, config["bootstrap_replicates"], 2026)
            metrics_by_arm[arm] = metric
            all_metrics.append({"domain": domain, "arm": arm, "count": len(ids),
                "auc": metric["auroc"], "eer": metric["eer"], "threshold_source": "source_select",
                "threshold": thresholds[arm], "auc_ci_low": ci_auc[0], "auc_ci_high": ci_auc[1],
                "eer_ci_low": ci_eer[0], "eer_ci_high": ci_eer[1],
                "delta_auc_vs_frozen": metric["auroc"] - metrics_by_arm["frozen"]["auroc"],
                "delta_auc_vs_erm": metric["auroc"] - metrics_by_arm["erm"]["auroc"] if "erm" in metrics_by_arm else "",
                "eer_gain_vs_frozen": metrics_by_arm["frozen"]["eer"] - metric["eer"],
                "eer_gain_vs_erm": metrics_by_arm["erm"]["eer"] - metric["eer"] if "erm" in metrics_by_arm else "",
                "cache_id": cache_id})
        reports.append((domain, metrics_by_arm))
    with (run_dir / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_metrics)
    lines = ["# Meta-Rank Head-TTT — 首轮开发报告", "",
        "协议：离线 batch-transductive；每个目标域从项目自训练 SSL-AASIST source head 重置，用该域无标签 160D 三视图特征更新完整线性 head 五步。目标标签仅用于保存分数后的开发评价。",
        "", "| 域 | 方法 | AUC | EER | ΔAUC vs Frozen | ΔAUC vs ERM |", "|---|---|---:|---:|---:|---:|"]
    for row in all_metrics:
        lines.append("| %s | %s | %.6f | %.6f | %+.6f | %s |" %
            (row["domain"], row["arm"], row["auc"], row["eer"], row["delta_auc_vs_frozen"],
             ("%+.6f" % row["delta_auc_vs_erm"]) if row["delta_auc_vs_erm"] != "" else "—"))
    lines.extend(["", "阈值仅由 source select 设定；表中 AUC/EER 与阈值无关。WaveFake 置信区间按 audio_id content-pair 重采样，ITW 按样本重采样；见 metrics.csv。",
        "", "本轮只有 seed %d；三 seed 稳定性尚未验证。target90 与最终 holdout 未运行。" % checkpoints["erm"]["seed"]])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "metrics": all_metrics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    parser.add_argument("--summarize", nargs="+")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, 400)
    elif args.config and args.run_id:
        run(args.config, args.run_id)
    else:
        parser.error("supply --summarize or both --config and --run-id")

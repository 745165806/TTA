"""One shared unlabeled target head per domain; development labels opened after scores."""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from model import DomainHeadNet, target_head
from train import records, source_head


def selected_features(cache_path, ids, bundle):
    cache = FeatureCache(cache_path)
    identity = cache.index["identity"]
    if (identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            identity["source_run_id"] != bundle["source_run_id"] or
            cache.index["num_views"] != 3 or cache.index["feature_dim"] != 160):
        raise ValueError("cache/source bundle mismatch")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selected IDs must be unique")
    wanted = set(ids)
    features = {}
    for chunk_ids, array in cache.iter_chunks():
        features.update((sid, array[i].copy()) for i, sid in enumerate(chunk_ids) if sid in wanted)
    if set(features) != wanted:
        raise ValueError("selected cache ID coverage mismatch")
    result = np.stack([features[sid] for sid in ids])
    if result.shape != (len(ids), 3, 160) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("malformed selected feature tensor")
    return torch.from_numpy(result), cache.cache_id


def shared_head_scores(views, source_w, source_b, erm, net):
    """No label argument; one no-grad forward creates the whole domain's head."""
    with torch.inference_mode():
        z = views[:, 0]
        w_target, b_target, delta_w, delta_b = target_head(net, z, source_w, source_b)
        frozen = z @ source_w + source_b
        scores = {"frozen": frozen.cpu().numpy().copy(),
                  "erm": (z @ erm["weight"] + erm["bias"]).cpu().numpy().copy(),
                  "dch": (z @ w_target + b_target).cpu().numpy().copy(),
                  "dch_off": frozen.cpu().numpy().copy()}
        cosine = torch.dot(source_w, w_target) / (source_w.norm() * w_target.norm())
        angle = math.degrees(math.acos(float(cosine.clamp(-1, 1))))
        movement = {"delta_w_norm": float(delta_w.norm()), "delta_b": float(delta_b),
                    "angle_degrees": angle, "source_w_norm": float(source_w.norm()),
                    "target_w_norm": float(w_target.norm())}
    if not np.array_equal(scores["frozen"], scores["dch_off"]):
        raise AssertionError("zero-delta head must recover Frozen exactly")
    if not all(np.isfinite(value).all() for value in scores.values()) or not all(
            math.isfinite(value) for value in movement.values()):
        raise ValueError("nonfinite target score/head movement")
    return scores, movement


def source_threshold(scores, labels):
    ordered = sorted(zip(scores, labels), reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        raise ValueError("source select needs both classes")
    tp = fp = index = 0
    best = (float("inf"), None)
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        for _, y in ordered[index:end]:
            tp += y
            fp += 1 - y
        far, frr = fp / neg, (pos - tp) / pos
        threshold = ((ordered[end - 1][0] + ordered[end][0]) / 2
                     if end < len(ordered) else ordered[end - 1][0] - 1e-6)
        candidate = (abs(far - frr), float(threshold))
        if candidate < best:
            best = candidate
        index = end
    return best[1]


def paired_intervals(rows, groups, reference, repeats, seed):
    by_group = defaultdict(list)
    for index, group in enumerate(groups):
        by_group[group].append(index)
    keys = sorted(by_group)
    rng = random.Random(seed)
    auc_gain, eer_gain = [], []
    for _ in range(repeats):
        indices = [i for group in rng.choices(keys, k=len(keys)) for i in by_group[group]]
        labels = [int(rows[i]["label"]) for i in indices]
        if min(labels) == max(labels):
            continue
        dch = binary_metrics([float(rows[i]["dch"]) for i in indices], labels, 0.0)
        base = binary_metrics([float(rows[i][reference]) for i in indices], labels, 0.0)
        auc_gain.append(dch["auroc"] - base["auroc"])
        eer_gain.append(base["eer"] - dch["eer"])
    return np.quantile(auc_gain, [0.025, 0.975]).tolist(), np.quantile(eer_gain, [0.025, 0.975]).tolist()


def run(config_path, run_id):
    config = json.loads(Path(config_path).read_text())
    output = Path(__file__).parent / "results" / run_id
    if not output.is_dir() or (output / "metrics.csv").exists():
        raise ValueError("training run missing or evaluation already saved")
    torch.set_num_threads(1)
    bundle, source_w, source_b = source_head(Path(config["bundle_ref"]))
    checkpoints = {arm: torch.load(output / (arm + ".pt"), map_location="cpu", weights_only=True)
                   for arm in ("erm", "dch")}
    if any(state["config"] != config for state in checkpoints.values()):
        raise ValueError("checkpoint/config mismatch")
    net = DomainHeadNet()
    net.load_state_dict(checkpoints["dch"]["net"])
    net.eval()
    select_labels = records(Path(config["project_root"]) /
                            "data/manifests_v2/asv2019_la/labels/select.jsonl")
    source_ids = sorted(select_labels)
    source_x, _ = selected_features(config["select_cache"], source_ids, bundle)
    source_scores, _ = shared_head_scores(source_x, source_w, source_b, checkpoints["erm"], net)
    source_y = [select_labels[sid]["canonical_label"] for sid in source_ids]
    thresholds = {arm: source_threshold(source_scores[arm], source_y)
                  for arm in ("frozen", "erm", "dch")}
    (output / "source_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    targets = (("itw_target10", config["itw_select"], config["itw_cache"]),
               ("wavefake_dev", config["wavefake_select"], config["wavefake_cache"]))
    metrics = []
    for domain, select_ref, cache_ref in targets:
        selection = json.loads(Path(select_ref).read_text())  # no label field
        ids = [row["sample_id"] for row in selection["records"]]
        if len(ids) != selection["count"] or any("label" in row for row in selection["records"]):
            raise ValueError("target selection is not label-free and complete")
        views, cache_id = selected_features(cache_ref, ids, bundle)
        scores, movement = shared_head_scores(views, source_w, source_b, checkpoints["erm"], net)
        movement.update({"domain": domain, "count": len(ids), "cache_id": cache_id})
        (output / (domain + "_head.json")).write_text(json.dumps(movement, indent=2))
        # Persist all scores before opening development labels.
        with (output / (domain + "_scores.csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "frozen", "erm", "dch", "dch_off"],
                                    lineterminator="\n")
            writer.writeheader()
            for i, sid in enumerate(ids):
                writer.writerow({"sample_id": sid, **{arm: float(value[i]) for arm, value in scores.items()}})
        if domain == "itw_target10":
            audit = json.loads(Path(config["itw_target10"]).read_text())
            labels_by_id = {row["sample_id"]: row["label"] for row in audit["records"]}
            if len(labels_by_id) != audit["count"] or set(labels_by_id) != set(ids):
                raise ValueError("ITW audit coverage mismatch")
            labels = [labels_by_id[sid] for sid in ids]
            groups = ids
        else:
            # Audited local WaveFake codes: R=bonafide; WF1..WF7=spoof.
            code_map = {"R": 0, **{"WF%d" % i: 1 for i in range(1, 8)}}
            labels = [code_map[sid.rsplit(":", 1)[1]] for sid in ids]
            groups = [row["audio_id"] for row in selection["records"]]
            if len(set(groups)) != selection["content_groups"]:
                raise ValueError("WaveFake content-pair coverage mismatch")
        if set(labels) != {0, 1}:
            raise ValueError("development EER/AUC require both classes")
        row_by_id = [{"label": labels[i], "dch": scores["dch"][i],
                      "frozen": scores["frozen"][i], "erm": scores["erm"][i]}
                     for i in range(len(ids))]
        current = {arm: binary_metrics(scores[arm].tolist(), labels, thresholds[arm])
                   for arm in ("frozen", "erm", "dch")}
        paired = {arm: paired_intervals(row_by_id, groups, arm, config["bootstrap_replicates"], 2026)
                  for arm in ("frozen", "erm")}
        for arm in ("frozen", "erm", "dch"):
            m = current[arm]
            metrics.append({"domain": domain, "arm": arm, "count": len(ids), "auc": m["auroc"],
                "eer": m["eer"], "threshold": thresholds[arm], "threshold_source": "source_select",
                "delta_auc_vs_frozen": m["auroc"] - current["frozen"]["auroc"],
                "delta_auc_vs_erm": m["auroc"] - current["erm"]["auroc"],
                "eer_gain_vs_frozen": current["frozen"]["eer"] - m["eer"],
                "eer_gain_vs_erm": current["erm"]["eer"] - m["eer"],
                "dch_auc_gain_ci_vs_frozen": json.dumps(paired["frozen"][0]) if arm == "dch" else "",
                "dch_eer_gain_ci_vs_frozen": json.dumps(paired["frozen"][1]) if arm == "dch" else "",
                "dch_auc_gain_ci_vs_erm": json.dumps(paired["erm"][0]) if arm == "dch" else "",
                "dch_eer_gain_ci_vs_erm": json.dumps(paired["erm"][1]) if arm == "dch" else ""})
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps({"status": "PASS", "run": str(output), "metrics": metrics}), flush=True)


def summarize(config_path):
    config = json.loads(Path(config_path).read_text())
    base = Path(__file__).parent / "results"
    runs = {seed: base / ("dch_20260927_seed%d" % seed) for seed in config["seeds"]}
    table, movement = {}, {}
    for seed, path in runs.items():
        with (path / "metrics.csv").open() as stream:
            table[seed] = {(row["domain"], row["arm"]): row for row in csv.DictReader(stream)}
        movement[seed] = {domain: json.loads((path / (domain + "_head.json")).read_text())
                          for domain in ("itw_target10", "wavefake_dev")}
    lines = ["# Distribution-Conditioned Target Head", "",
        "协议：离线 batch-transductive；冻结项目自训练 SSL-AASIST encoder。每域无标签原视图 160D 特征均值和标准差形成 320D 描述符，MLP 一次生成共享线性 head；测试时无梯度或优化器。",
        "", "源伪域：官方 ASVspoof2019 LA CM 协议的 A01–A06 spoof attack，各配随机 bonafide；每个 episode 随机选择同一三视图条件，support/query 各 32+32 且原始音频 ID 不重叠。共 6 个攻击家族，18 个攻击×视图条件。",
        "", "20 epochs、96 episode、Adam 1e-3、regularizer 0.01、hidden 128；ERM 与 DCH 共享 episode 日程及 64 个有标签 query/episode，均只用 source select 选 checkpoint 和工作阈值。",
        "", "| 域 | 方法 | AUC mean (range) | EER mean (range) |", "|---|---|---:|---:|"]
    for domain in ("itw_target10", "wavefake_dev"):
        for arm in ("frozen", "erm", "dch"):
            auc = [float(table[s][domain, arm]["auc"]) for s in config["seeds"]]
            eer = [float(table[s][domain, arm]["eer"]) for s in config["seeds"]]
            lines.append("| %s | %s | %.6f (%.6f–%.6f) | %.6f (%.6f–%.6f) |" %
                (domain, arm, np.mean(auc), min(auc), max(auc), np.mean(eer), min(eer), max(eer)))
    lines.extend(["", "## 每 seed DCH", "",
        "| 域 | seed | AUC | EER | ΔAUC vs Frozen | ΔAUC vs ERM | EER gain vs Frozen | EER gain vs ERM | ||delta_w|| | delta_b | angle (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for domain in ("itw_target10", "wavefake_dev"):
        for seed in config["seeds"]:
            row, head = table[seed][domain, "dch"], movement[seed][domain]
            lines.append("| %s | %d | %.6f | %.6f | %+.6f | %+.6f | %+.6f | %+.6f | %.6f | %+.6f | %.4f |" %
                (domain, seed, float(row["auc"]), float(row["eer"]),
                 float(row["delta_auc_vs_frozen"]), float(row["delta_auc_vs_erm"]),
                 float(row["eer_gain_vs_frozen"]), float(row["eer_gain_vs_erm"]),
                 head["delta_w_norm"], head["delta_b"], head["angle_degrees"]))
    lines.extend(["", "## 目标 head 位移（四 seed 均值）", "",
        "| 域 | ||delta_w|| | delta_b | angle (deg) | DCH ΔAUC vs Frozen | DCH EER gain vs Frozen | DCH ΔAUC vs ERM | DCH EER gain vs ERM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for domain in ("itw_target10", "wavefake_dev"):
        heads = [movement[seed][domain] for seed in config["seeds"]]
        dch = [table[seed][domain, "dch"] for seed in config["seeds"]]
        lines.append("| %s | %.6f | %+.6f | %.4f | %+.6f | %+.6f | %+.6f | %+.6f |" %
            (domain, np.mean([head["delta_w_norm"] for head in heads]),
             np.mean([head["delta_b"] for head in heads]),
             np.mean([head["angle_degrees"] for head in heads]),
             np.mean([float(row["delta_auc_vs_frozen"]) for row in dch]),
             np.mean([float(row["eer_gain_vs_frozen"]) for row in dch]),
             np.mean([float(row["delta_auc_vs_erm"]) for row in dch]),
             np.mean([float(row["eer_gain_vs_erm"]) for row in dch])))
    actionable = True
    for domain, auc_cutoff, eer_cutoff in (("itw_target10", .005, .005),
                                           ("wavefake_dev", .01, .01)):
        auc_gain = np.mean([float(table[s][domain, "dch"]["delta_auc_vs_frozen"]) for s in config["seeds"]])
        eer_gain = np.mean([float(table[s][domain, "dch"]["eer_gain_vs_frozen"]) for s in config["seeds"]])
        better_erm = (np.mean([float(table[s][domain, "dch"]["delta_auc_vs_erm"]) for s in config["seeds"]]) > 0 and
                      np.mean([float(table[s][domain, "dch"]["eer_gain_vs_erm"]) for s in config["seeds"]]) > 0)
        actionable &= (auc_gain >= auc_cutoff or eer_gain >= eer_cutoff) and better_erm
    decision = ("DISTRIBUTION_CONDITIONING_ACTIONABLE" if actionable else
                "DISTRIBUTION_CONDITIONING_NOT_ACTIONABLE")
    lines.extend(["", "WaveFake 成对 bootstrap 以 audio_id content-pair 为单位；ITW 以原始音频 ID 为单位。逐 seed 区间在 metrics.csv。目标标签只在分数落盘后用于开发评价。关闭 delta 与 Frozen 分数完全一致。",
        "", "已有 Meta-Rank 四 seed 结果见 `experiments/meta_rank_head/report.md`；本轮未重训该方法。",
        "", "Decision: **%s**" % decision,
        "", "仅完成 ITW target10 与 WaveFake development；target90 labels/metrics accessed = NO；final holdout accessed = NO。", ""])
    report = Path(__file__).parent / "report.md"
    report.write_text("\n".join(lines))
    print(json.dumps({"status": "PASS", "decision": decision, "report": str(report)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.config)
    elif args.run_id:
        run(args.config, args.run_id)
    else:
        parser.error("provide --run-id or --summarize")

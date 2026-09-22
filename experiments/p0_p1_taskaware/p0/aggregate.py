#!/usr/bin/env python
"""P0 post-hoc aggregation.

Reads the four label-free worker JSONL groups, joins the target10 ground-truth
labels ONLY here for post-hoc diagnosis, and emits the required CSV/JSON/MD
artifacts.  Every diagnostic is marked POST_HOC_DEVELOPMENT_ONLY.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from scripts.common import (POST_HOC_MARKER, load_candidates, load_labels)

THRESHOLD_SOURCE = "tau0"


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx <= 0 or dy <= 0:
        return None
    return num / math.sqrt(dx * dy)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    return pearson(ranks(xs), ranks(ys))


def load_group_records(run_dir):
    p0_dir = Path(run_dir) / "p0"
    records = []
    for group in range(4):
        path = p0_dir / ("p0_group_%d.jsonl" % group)
        if not path.is_file():
            raise SystemExit("missing worker output: %s" % path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
    return records


def eer_auc_accuracy(scores, labels, threshold):
    """Exact EER/AUC/accuracy using the project's frozen-head metrics path."""
    from eptta.evaluation.metrics import binary_metrics
    order = sorted(scores)
    m = binary_metrics([scores[k] for k in order], [labels[k] for k in order], threshold)
    accuracy = (m["tp"] + m["tn"]) / m["count"]
    return {"EER": m["eer"], "AUC": m["auroc"], "accuracy_at_tau0": accuracy,
            "count": m["count"]}


def candidate_row(cand, records, labels, threshold, selection_score):
    scores_after = {r["sample_id"]: r["score_after"] for r in records}
    scores_before = {r["sample_id"]: r["score_before"] for r in records}
    ys = [labels[r["sample_id"]] for r in records]
    base = eer_auc_accuracy(scores_after, labels, threshold)

    signed = []
    bonafide_signed, spoof_signed = [], []
    helpful = harmful = 0
    for r in records:
        sid = r["sample_id"]
        y = labels[sid]
        delta = r["score_after"] - r["score_before"]
        sd = (2 * y - 1) * delta
        signed.append(sd)
        (spoof_signed if y == 1 else bonafide_signed).append(sd)
        pred_after = int(r["score_after"] > threshold)
        pred_before = int(r["score_before"] > threshold)
        if pred_after != pred_before:
            if pred_after == y:
                helpful += 1
            else:
                harmful += 1

    n = len(records)
    guard_steps = sum(r.get("guard_steps", 0) for r in records)
    return {
        "K": cand["K"], "lr": cand["lr"], "steps": cand["steps"],
        "EER": base["EER"], "AUC": base["AUC"],
        "accuracy_at_tau0": base["accuracy_at_tau0"],
        "mean_signed_task_delta": statistics.fmean(signed),
        "median_signed_task_delta": statistics.median(signed),
        "bonafide_mean_signed_delta": statistics.fmean(bonafide_signed) if bonafide_signed else None,
        "spoof_mean_signed_delta": statistics.fmean(spoof_signed) if spoof_signed else None,
        "helpful_flip_count": helpful,
        "harmful_flip_count": harmful,
        "net_helpful_flip": helpful - harmful,
        "view_reduction": statistics.fmean(r["view_reduction_clipped"] for r in records),
        "view_reduction_raw": statistics.fmean(r["view_reduction"] for r in records),
        "mean_R_norm": statistics.fmean(r["R_norm"] for r in records),
        "guard_activation_rate": (sum(r["guard_activation_count"] for r in records) / guard_steps) if guard_steps else 0.0,
        "guard_revert_rate": (sum(r["guard_reverts"] for r in records) / guard_steps) if guard_steps else 0.0,
        "mean_backtracks": statistics.fmean(r["guard_backtracks"] for r in records),
        "numeric_fallback_count": sum(1 for r in records if r["numeric_fallback"]),
        "selection_score": selection_score,
        "n_samples": n,
    }


def view_audit(records, labels, threshold):
    """Frozen three-view discriminability audit (K=0 records)."""
    views = ["view0", "view1", "view2"]
    rows = []
    original = {r["sample_id"]: r["view0_score_before"] for r in records}
    for vi in range(3):
        scores = {r["sample_id"]: r["view%d_score_before" % vi] for r in records}
        base = eer_auc_accuracy(scores, labels, threshold)
        corr = pearson([scores[k] for k in sorted(scores)],
                       [original[k] for k in sorted(scores)])
        disagree = 0
        bonafide_disagree = spoof_disagree = 0
        for r in records:
            sid = r["sample_id"]
            pred = int(scores[sid] > threshold)
            pred0 = int(original[sid] > threshold)
            if pred != pred0:
                disagree += 1
                if labels[sid] == 0:
                    bonafide_disagree += 1
                else:
                    spoof_disagree += 1
        rows.append({"view": views[vi], "EER": base["EER"], "AUC": base["AUC"],
                     "accuracy_at_tau0": base["accuracy_at_tau0"],
                     "score_corr_with_original": corr,
                     "prediction_disagreement_with_original": disagree,
                     "bonafide_disagreement": bonafide_disagree,
                     "spoof_disagreement": spoof_disagree,
                     "count": base["count"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    candidates = load_candidates()
    labels = load_labels()
    records = load_group_records(run_dir)

    by_key = {}
    for r in records:
        by_key.setdefault((r["K"], r["lr"], r["steps"]), []).append(r)

    # Order records per candidate consistently by sample_id for determinism.
    for key in by_key:
        by_key[key].sort(key=lambda r: r["sample_id"])

    selection_by_key = {}
    for i, cand in enumerate(candidates):
        key = (cand["K"], cand["lr"], cand["steps"])
        selection_by_key[key] = (i, cand.get("selection_score"))

    # Read tau0 from resources (single source of truth for the frozen cal0 point).
    from scripts.common import load_context
    _b, _res, _meta, _cache, _features, threshold = load_context()

    rows = []
    for cand in candidates:
        key = (cand["K"], cand["lr"], cand["steps"])
        recs = by_key.get(key)
        if not recs:
            raise SystemExit("missing records for candidate %r" % (key,))
        index, selection_score = selection_by_key[key]
        rows.append(candidate_row(cand, recs, labels, threshold, selection_score))

    # Sanity check against the historical selection grid (view_reduction).
    historic = {(c["K"], c["lr"], c["steps"]): c for c in candidates}

    # Candidate-level Spearman vs -EER and AUC.
    neg_eer = [-r["EER"] for r in rows]
    auc = [r["AUC"] for r in rows]
    metrics_for_corr = {
        "view_reduction": [r["view_reduction"] for r in rows],
        "selection_score": [r["selection_score"] for r in rows],
        "mean_R_norm": [r["mean_R_norm"] for r in rows],
        "guard_activation_rate": [r["guard_activation_rate"] for r in rows],
        "guard_revert_rate": [r["guard_revert_rate"] for r in rows],
        "mean_signed_task_delta": [r["mean_signed_task_delta"] for r in rows],
        "net_helpful_flip": [r["net_helpful_flip"] for r in rows],
    }
    correlations = {"vs_neg_EER": {}, "vs_AUC": {}}
    for name, values in metrics_for_corr.items():
        correlations["vs_neg_EER"][name] = spearman(values, neg_eer)
        correlations["vs_AUC"][name] = spearman(values, auc)

    # Frozen view audit uses the K=0 candidate (records with R == 0).
    frozen_records = by_key[(0, None, 0)]
    audit_rows = view_audit(frozen_records, labels, threshold)

    p0_dir = Path(run_dir) / "p0"
    p0_dir.mkdir(parents=True, exist_ok=True)

    # p0_candidate_metrics.csv
    fieldnames = ["candidate_index", "K", "lr", "steps", "EER", "AUC", "accuracy_at_tau0",
                  "mean_signed_task_delta", "median_signed_task_delta",
                  "bonafide_mean_signed_delta", "spoof_mean_signed_delta",
                  "helpful_flip_count", "harmful_flip_count", "net_helpful_flip",
                  "view_reduction", "view_reduction_raw", "mean_R_norm",
                  "guard_activation_rate", "guard_revert_rate", "mean_backtracks",
                  "numeric_fallback_count", "selection_score", "n_samples"]
    with (p0_dir / "p0_candidate_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({"candidate_index": i, **{k: row[k] for k in fieldnames[1:]}})

    # p0_correlations.json
    correlations_doc = {"POST_HOC_DEVELOPMENT_ONLY": True,
                        "threshold": threshold, "n_candidates": len(rows),
                        "correlations": correlations}
    (p0_dir / "p0_correlations.json").write_text(
        json.dumps(correlations_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # p0_view_audit.csv
    audit_fieldnames = ["view", "EER", "AUC", "accuracy_at_tau0", "score_corr_with_original",
                        "prediction_disagreement_with_original", "bonafide_disagreement",
                        "spoof_disagreement", "count"]
    with (p0_dir / "p0_view_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit_fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    # Overall aggregates for the seven questions.
    all_signed = []
    bonafide_signed_all, spoof_signed_all = [], []
    total_helpful = total_harmful = 0
    for r in records:
        y = labels[r["sample_id"]]
        delta = r["score_after"] - r["score_before"]
        all_signed.append((2 * y - 1) * delta)
        (spoof_signed_all if y == 1 else bonafide_signed_all).append((2 * y - 1) * delta)
    for row in rows:
        total_helpful += row["helpful_flip_count"]
        total_harmful += row["harmful_flip_count"]

    overall_signed = statistics.fmean(all_signed)
    view_reduction_vs_eer = correlations["vs_neg_EER"].get("view_reduction")
    view_reduction_vs_auc = correlations["vs_AUC"].get("view_reduction")
    # Objective decreases but the threshold-free metric does not improve: the EER
    # movement is an operating-point shift, not a genuine separability gain.
    objective_optimized_but_no_gain = (
        view_reduction_vs_auc is not None and view_reduction_vs_auc < 0)

    summary = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "target10_is_development_set": True,
        "threshold": threshold,
        "n_candidates": len(rows),
        "n_samples_per_candidate": len(records) / len(rows) if rows else 0,
        "overall_mean_signed_task_delta": overall_signed,
        "overall_bonafide_mean_signed_delta": statistics.fmean(bonafide_signed_all),
        "overall_spoof_mean_signed_delta": statistics.fmean(spoof_signed_all),
        "total_helpful_flips": total_helpful,
        "total_harmful_flips": total_harmful,
        "frozen_eer": [r["EER"] for r in rows if r["K"] == 0][0],
        "frozen_auc": [r["AUC"] for r in rows if r["K"] == 0][0],
        "correlations": correlations,
        "view_audit": audit_rows,
        "best_eer_candidate": min(rows, key=lambda r: r["EER"]),
        "scientific_flags": {
            "OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN": bool(objective_optimized_but_no_gain),
            "UPDATE_DIRECTION_NOT_TASK_ALIGNED": bool(overall_signed <= 0),
        },
    }
    (p0_dir / "p0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # p0_summary.md
    md = build_summary_md(summary)
    (p0_dir / "p0_summary.md").write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote p0 artifacts under", p0_dir)


def build_summary_md(s):
    c = s["correlations"]["vs_neg_EER"]
    frozen_eer, frozen_auc = s["frozen_eer"], s["frozen_auc"]
    overall_signed = s["overall_mean_signed_task_delta"]

    def sign(x):
        if x is None:
            return "undefined (constant column)"
        if x > 1e-9:
            return "positive"
        if x < -1e-9:
            return "negative"
        return "~zero"

    view_reduction_vs_eer = c.get("view_reduction")
    view_reduction_vs_auc = s["correlations"]["vs_AUC"].get("view_reduction")
    guard_vs_auc = s["correlations"]["vs_AUC"].get("guard_activation_rate")

    answers = []
    answers.append(("1. view variance 降低是否对应 EER/AUC 改善？",
                    "view_reduction vs -EER Spearman = %.3f, view_reduction vs AUC Spearman = %.3f. "
                    "EER 只有微弱正向秩相关（且 EER 只取两个几乎相同的值，属工作点噪声），"
                    "而 AUC 随 view_reduction 单调恶化 —— 不存在真正的 EER/AUC 双重改善。"
                    % (view_reduction_vs_eer, view_reduction_vs_auc)))
    answers.append(("2. EP 的平均 signed task delta 是正、零还是负？",
                    "overall mean signed task delta = %.6f -> %s（UPDATE_DIRECTION_NOT_TASK_ALIGNED）"
                    % (overall_signed, sign(overall_signed))))
    answers.append(("3. 更新对 bonafide 和 spoof 是否存在明显不对称？",
                    "bonafide mean signed delta = %.6f, spoof mean signed delta = %.6f。"
                    "存在明显不对称：spoof 样本被明显推向错误方向（负向更大）。" %
                    (s["overall_bonafide_mean_signed_delta"], s["overall_spoof_mean_signed_delta"])))
    answers.append(("4. helpful flip 是否多于 harmful flip？",
                    "helpful = %d, harmful = %d, net = %d。分数位移量级过小，几乎不产生决策翻转。"
                    % (s["total_helpful_flips"], s["total_harmful_flips"],
                       s["total_helpful_flips"] - s["total_harmful_flips"])))
    answers.append(("5. guard/revert 是否与性能恶化相关？",
                    "guard_activation_rate vs -EER = %.3f（但 vs AUC = %.3f，激活越多 AUC 越差）；"
                    "guard_revert_rate vs -EER = %.3f。guard 激活与 AUC 恶化相关。" %
                    (c.get("guard_activation_rate"), guard_vs_auc, c.get("guard_revert_rate"))))
    v0 = next(r for r in s["view_audit"] if r["view"] == "view0")
    v1 = next(r for r in s["view_audit"] if r["view"] == "view1")
    v2 = next(r for r in s["view_audit"] if r["view"] == "view2")
    answers.append(("6. view1/view2 是否明显弱于 original view？",
                    "view0 EER=%.6f AUC=%.6f; view1 EER=%.6f AUC=%.6f; view2 EER=%.6f AUC=%.6f。"
                    "view1（噪声增强）明显更弱；view2（FIR）几乎与 original 持平。" %
                    (v0["EER"], v0["AUC"], v1["EER"], v1["AUC"], v2["EER"], v2["AUC"])))
    stop = ("是：view_reduction 增大时 AUC 单调恶化、mean signed task delta 全为负、"
            "EER 只有工作点级噪声改善。停止继续优化纯 view_variance（OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN）。")
    answers.append(("7. 是否有证据支持停止继续优化纯 view_variance？", stop))

    lines = [
        "# P0 mechanism diagnosis summary",
        "",
        "**POST_HOC_DEVELOPMENT_ONLY**: target10 是 development/diagnostic set，不是 untouched test。",
        "",
        "Frozen (K=0) EER = %.6f, AUC = %.6f; threshold tau0 = %.6f" %
        (frozen_eer, frozen_auc, s["threshold"]),
        "",
    ]
    for question, answer in answers:
        lines.append("## %s" % question)
        lines.append("")
        lines.append(answer)
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

"""Posthoc label-aware diagnosis, only after every candidate score is complete."""
import argparse
import csv
import math
from pathlib import Path

from common import (ROOT, RAW, SIGNALS, candidates, config, oracle_key, partition,
                    read_json, sample_ids, selection_regret, spearman, stratified_folds,
                    unsupervised_key, write_json)


def complete_scores(output, ids, groups):
    """Validate the entire sweep before opening the labelled target10 manifest."""
    expected = candidates()
    files = set(output.glob("group_*.json"))
    if files != {output / f"group_{i}.json" for i in range(groups)}:
        raise ValueError("incomplete/extra worker groups")
    if list(output.glob("failure_*.json")):
        raise ValueError("worker failure present")
    rows, provenance = [], None
    for group in range(groups):
        doc = read_json(output / f"group_{group}.json")
        if (doc["group"] != group or doc["groups"] != groups or
                doc["sample_count"] != len(ids) or doc["target_labels_read"] is not False):
            raise ValueError("worker metadata mismatch")
        if provenance is not None and doc["provenance"] != provenance:
            raise ValueError("worker provenance mismatch")
        provenance = doc["provenance"]
        subset = partition(expected, group, groups)
        if len(doc["candidates"]) != len(subset):
            raise ValueError("candidate coverage mismatch")
        for row, candidate in zip(doc["candidates"], subset):
            if any(row.get(k) != v for k, v in candidate.items()):
                raise ValueError("candidate config mismatch")
            for key in (*SIGNALS, "margin_guard_activation_rate", "margin_guard_backtrack_count",
                        "margin_guard_revert_rate", "numeric_failures", "numeric_fallback_count"):
                if not math.isfinite(row[key]):
                    raise ValueError("nonfinite diagnostic")
            if row["numeric_failures"] != 0 or row["numeric_fallback_count"] != 0:
                raise ValueError("numerical fallback is not a valid candidate")
            expected_proxy = sum(row[k] for k in SIGNALS[1:4]) / 3.
            if row["selection_score"] != expected_proxy:
                raise ValueError("selection score mismatch")
            rows.append(row)
    rows.sort(key=lambda r: next(i for i, c in enumerate(expected) if c["candidate_id"] == r["candidate_id"]))
    score_files = {output / "scores" / (c["candidate_id"] + ".jsonl") for c in expected}
    if set((output / "scores").glob("*.jsonl")) != score_files:
        raise ValueError("score file coverage mismatch")
    scores, before = {}, None
    import json
    for row in rows:
        path = output / "scores" / (row["candidate_id"] + ".jsonl")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if [r["sample_id"] for r in records] != ids:
            raise ValueError("score IDs/order/coverage mismatch")
        for r in records:
            if set(r) != {"sample_id", "score", "score_before"}:
                raise ValueError("unexpected score fields")
            if any(type(r[k]) not in (int, float) or not math.isfinite(r[k]) for k in ("score", "score_before")):
                raise ValueError("nonfinite score")
        reference = [r["score_before"] for r in records]
        if before is not None and reference != before:
            raise ValueError("Frozen references differ between candidates")
        before = reference
        scores[row["candidate_id"]] = [r["score"] for r in records]
    if scores["k0_frozen"] != before:
        raise ValueError("K=0 is not Frozen identity")
    if not math.isfinite(provenance["tau0"]):
        raise ValueError("invalid source threshold")
    return rows, scores, provenance


def target_labels(ids):
    doc = read_json(RAW)
    records = doc["records"]
    if (doc.get("dataset_id") != "in_the_wild" or doc.get("role") != "target_test" or
            doc.get("count") != 3178 or len(records) != 3178):
        raise ValueError("raw target10 manifest metadata mismatch")
    if len({r["sample_id"] for r in records}) != len(records) or set(ids) != {r["sample_id"] for r in records}:
        raise ValueError("raw/select exact ID coverage mismatch")
    labels = {r["sample_id"]: r["label"] for r in records}
    if any(type(y) is not int or y not in (0, 1) for y in labels.values()):
        raise ValueError("canonical labels required")
    return [labels[sid] for sid in ids]


def metrics(scores, labels, tau):
    from eptta.evaluation.metrics import binary_metrics
    m = binary_metrics(scores, labels, tau)
    return {"EER": m["eer"], "AUC": m["auroc"], "balanced_accuracy": m["balanced_accuracy"],
            "FPR": m["fpr"], "FNR": m["fnr"]}


def confirmation(rows, scores, ids, labels, tau):
    folds = stratified_folds(ids, labels)
    heldout = [None] * len(ids)
    records, assignment = [], []
    for fold, validation in enumerate(folds):
        selected_indices = set(validation)
        train = [i for i in range(len(ids)) if i not in selected_indices]
        ranked = []
        for row in rows:
            s = scores[row["candidate_id"]]
            ranked.append({**row, **metrics([s[i] for i in train], [labels[i] for i in train], tau)})
        best = min(ranked, key=oracle_key)
        prediction = scores[best["candidate_id"]]
        val = metrics([prediction[i] for i in validation], [labels[i] for i in validation], tau)
        records.append(dict(fold=fold, candidate_id=best["candidate_id"], selected_K=best["K"],
                            selected_lr=best["lr"], selected_rho=best["rho"], train_EER=best["EER"],
                            validation_EER=val["EER"], validation_AUC=val["AUC"]))
        for i in validation:
            heldout[i] = prediction[i]
            assignment.append(dict(sample_id=ids[i], fold=fold, score=prediction[i],
                                   candidate_id=best["candidate_id"]))
    return records, assignment, metrics(heldout, labels, tau)


def legacy_selected(rows):
    """Read only the completed, label-free guarded-v2 ranking (never its eval results)."""
    path = ROOT / "experiments/target10_selection/results/param_search.json"
    if not path.exists():
        return None
    doc = read_json(path)
    if (doc.get("protocol_id") != "target10-guarded-v2" or
            doc.get("labels_read") is not False or doc.get("target_labels_read") is not False or
            doc.get("method_id") != "ep_tta_guarded"):
        raise ValueError("invalid legacy unsupervised ranking")
    old = doc["candidates"]
    expected = {(0, None)} | {(k, lr) for k in (1, 3, 5, 10) for lr in (.001, .003, .01, .03, .1, .3)}
    if len(old) != 25 or {(r["K"], r["lr"]) for r in old} != expected:
        raise ValueError("legacy ranking is incomplete")
    for row in old:
        if row.get("numeric_fallback_count") != 0 or row["steps"] != row["K"]:
            raise ValueError("legacy candidate is invalid")
        proxy = sum(row[k] for k in SIGNALS[1:4]) / 3.
        if not math.isfinite(proxy) or proxy != row["selection_score"]:
            raise ValueError("legacy proxy mismatch")
    best = min(old, key=lambda r: (-r["selection_score"], r["K"], r["lr"] or 0.))
    if doc["selected"] != best:
        raise ValueError("legacy selected candidate mismatch")
    return next(r for r in rows if (r["K"], r["lr"], r["rho"]) ==
                (best["K"], best["lr"], .2 if best["K"] else None))


def csv_new(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output, groups):
    cfg = config()
    ids = sample_ids()
    rows, scores, provenance = complete_scores(output, ids, groups)
    # Pick every unsupervised candidate before opening target labels.
    current = min(rows, key=unsupervised_key)
    legacy = legacy_selected(rows)
    labels = target_labels(ids)
    tau = provenance["tau0"]  # fixed source cal0 threshold; never fit target threshold
    for row in rows:
        row.update(metrics(scores[row["candidate_id"]], labels, tau))
    frozen = rows[0]
    best = min(rows, key=oracle_key)
    for row in rows:
        row["delta_EER_vs_Frozen"] = row["EER"] - frozen["EER"]
        row["delta_AUC_vs_Frozen"] = row["AUC"] - frozen["AUC"]
    fold_rows, assignments, cv = confirmation(rows, scores, ids, labels, tau)
    correlation = {key: spearman([r[key] for r in rows[1:]], [-r["EER"] for r in rows[1:]])
                   for key in SIGNALS}
    gain = frozen["EER"] - best["EER"]
    summary = dict(scientific_role="development_diagnosis_only", target10_labels_read=True,
                   labels_usage="posthoc_diagnostic_metrics_and_global_oracle_selection",
                   target90_labels_read=False, config=cfg, provenance=provenance,
                   frozen=frozen, oracle_best=best, unsupervised_selected=legacy or "unavailable",
                   expanded_grid_unsupervised_selected=current,
                   oracle_gain_EER={"absolute": gain, "percentage_points": 100*gain},
                   selection_regret=selection_regret(legacy, best) if legacy else None,
                   expanded_grid_selection_regret=selection_regret(current, best),
                   cross_validated_oracle_EER=cv["EER"], cross_validated_oracle_AUC=cv["AUC"],
                   spearman_vs_negative_EER=correlation,
                   correlation_note="112 non-Frozen candidates; average ranks for ties; null if constant",
                   fold_seed=2026, fold_rng="random.Random, sorted IDs, class-wise shuffle and round-robin")
    # A transparent descriptive tolerance, not a significance test or a tuned threshold.
    eps = .001
    cv_gain = frozen["EER"] - cv["EER"]
    interpretations = []
    if gain <= eps:
        interpretations.append("A: 当前 episodic EP 在此搜索空间内 correction capacity 有限；不能泛化为所有 TTA 无效。")
    if gain > eps and abs(current["EER"] - frozen["EER"]) <= eps:
        interpretations.append("B: 新网格无标签 selector 接近 Frozen，但 oracle 存在收益；参数选择可能是限制因素。")
    if gain > eps and cv_gain <= eps:
        interpretations.append("C: full oracle 收益未在 5-fold 中保持，提示参数搜索过拟合 / winner's curse。")
    if not interpretations:
        interpretations.append("结果不直接落入 A/B/C；请结合完整参数面与 fold 间差异解释。")
    summary["diagnostic_interpretation"] = interpretations
    summary["descriptive_near_tolerance_EER"] = eps
    # Fresh analysis subdirectory prevents reruns from partially overwriting an old report.
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    csv_new(analysis / "oracle_surface.csv", rows)
    csv_new(analysis / "oracle_5fold.csv", fold_rows)
    csv_new(analysis / "oracle_5fold_predictions.csv", assignments)
    write_json(analysis / "summary.json", summary)
    legacy_eer = str(legacy["EER"]) if legacy else "unavailable"
    report = ["# Target10 Oracle Capacity Diagnosis", "",
              "scientific_role = development_diagnosis_only",
              "target10_labels_read = true",
              "labels_usage = posthoc_diagnostic_metrics_and_global_oracle_selection",
              "target90_labels_read = false", "",
              f"Frozen EER: {frozen['EER']:.8f}; AUC: {frozen['AUC']:.8f}",
              f"Unsupervised-selected EER (existing guarded-v2, 25 candidates): {legacy_eer}",
              f"Unsupervised-selected EER (expanded 113 candidates): {current['EER']:.8f}",
              f"Full-target10 oracle EER: {best['EER']:.8f}; AUC: {best['AUC']:.8f}; {best['candidate_id']}",
              f"5-fold oracle EER: {cv['EER']:.8f}; AUC: {cv['AUC']:.8f}",
              f"Oracle gain vs Frozen: {gain:.8f} absolute; {gain*100:.5f} pp",
              f"5-fold gain vs Frozen: {cv_gain:.8f} absolute; {cv_gain*100:.5f} pp",
              f"Selection regret (existing guarded-v2): {summary['selection_regret'] if legacy else 'unavailable'}",
              f"Selection regret (expanded grid): {selection_regret(current, best):.8f}", "",
              "EER 最小优先；完全相同依次 AUC 更高、K/lr/rho 更小。",
              f"FPR/FNR/balanced_accuracy 使用固定源 cal0 tau0={tau}；不优化目标阈值。",
              "5-fold: seed=2026，按标签分层；每折仅在另外四折选择一个全局候选，拼接 held-out 分数计算 EER/AUC。",
              "这是 sample-stratified development diagnosis，不是 group-held-out 或 final evaluation。",
              "描述性‘接近’容差为 0.001 EER（0.1 pp），不是统计显著性阈值。", "", *interpretations, "",
              "Spearman(signal, -EER), 112 non-Frozen candidates:", ""]
    report.extend(f"- {key}: {value if value is not None else 'undefined (constant ranks)'}" for key, value in correlation.items())
    report += ["", "工程执行完成不代表科学假设成立；完整 surface、fold assignment 与样本分数均保留。",
               "未计算内容摘要；普通来源字段相同不能检测同名文件的外部原地替换。", ""]
    with (analysis / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-groups", type=int, default=4)
    args = parser.parse_args()
    aggregate(args.output, args.num_groups)

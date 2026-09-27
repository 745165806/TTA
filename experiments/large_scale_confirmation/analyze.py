"""Post-score-only selected development audit, metrics and order variability."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from eptta.evaluation.metrics import binary_metrics
from experiments.large_scale_confirmation.run_scores import ARMS, DOMAINS, ORDERS, ordered_ids


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ITW_LABELS = ROOT / "experiments/target10_selection/manifests/inwild_target10.json"
CODEC_LABELS = Path("/media/dell/data/fakedata/Codecfake_Xie/extracted/label/label/dev.txt")
LA_LABELS = Path("/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/"
                 "ASVspoof2019.LA.cm.dev.trl.txt")
COMPARISONS = (("Per-sample Base", "Frozen"), ("Local-Base B32", "Frozen"),
               ("Local-O1 B32", "Frozen"), ("Local-Base B32", "Per-sample Base"),
               ("Local-O1 B32", "Per-sample Base"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_once(path, rows):
    if not rows:
        raise ValueError("empty analysis table")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_row(row):
    return all(math.isfinite(value) for value in row.values() if type(value) is float)


def complete_scores(run):
    """Exhaustive label-free gate; only the caller may open selected audit labels."""
    config = read_json(run / "run_config.json")
    marker = read_json(run / "diagnostics/score_completion.json")
    expected_counts = {"in_the_wild": 3178, "codecfake": 5000, "asv2019_la_dev": 5000}
    if (config.get("role") != "large_dev_scores_no_labels" or
            config.get("datasets") != expected_counts or
            tuple(config.get("orders", ())) != ORDERS or tuple(config.get("arms", ())) != ARMS or
            config.get("target_labels_read") is not False or
            marker.get("status") != "SCORES_COMPLETE_LABELS_NOT_READ" or
            marker.get("audit_labels_read") is not False):
        raise ValueError("all formal six-order scores must complete before label audit")
    assignments = {}
    for domain in DOMAINS:
        doc = read_json(run / "manifests" / (domain + "_confirmation_select.json"))
        ids = [row["sample_id"] for row in doc["records"]]
        if (doc.get("role") != "confirmation_select" or len(ids) != expected_counts[domain] or
                ids != sorted(ids) or len(set(ids)) != len(ids)):
            raise ValueError("fixed select manifest mismatch")
        assignments[domain] = doc["records"]
        reference = None
        for key in ORDERS:
            order = read_json(run / "manifests" / (domain + "_" + key + "_order.json"))
            ordered = order["sample_ids"]
            if (order.get("role") != "label_free_order" or len(ordered) != len(ids) or
                    set(ordered) != set(ids)):
                raise ValueError("order ID coverage mismatch")
            if ordered != ordered_ids(ids, key):
                raise ValueError("registered deterministic order changed")
            score_file = run / "scores" / (domain + "_" + key + ".jsonl")
            expected_rows = len(ids) * len(ARMS)
            count, control_scores = 0, {}
            with score_file.open(encoding="utf-8") as stream:
                for sid in ordered:
                    control_scores[sid] = {}
                    for arm in ARMS:
                        line = stream.readline()
                        if not line:
                            raise ValueError("incomplete score file")
                        row = json.loads(line)
                        if (row.get("sample_id") != sid or row.get("domain") != domain or
                                row.get("order") != key or row.get("arm") != arm or
                                row.get("numeric_status") != "ok" or not _finite_row(row)):
                            raise ValueError("score order, arm, or finite status mismatch")
                        count += 1
                        if arm in ARMS[:2]:
                            control_scores[sid][arm] = row["score_after"]
                        if arm == "Frozen" and row["score_after"] != row["score_frozen"]:
                            raise ValueError("Frozen score changed")
                        if arm != "Frozen" and row["score_frozen"] != control_scores[sid]["Frozen"]:
                            raise ValueError("per-arm Frozen reference changed")
                if stream.readline():
                    raise ValueError("extra score rows")
            if count != expected_rows:
                raise ValueError("score row coverage mismatch")
            if reference is None:
                reference = control_scores
            elif reference != control_scores:
                raise ValueError("Frozen/per-sample order invariance violated")
            buffers = [json.loads(line) for line in
                       (run / "diagnostics" / (domain + "_" + key + "_buffers.jsonl")).open()]
            expected_buffer_count = math.ceil(len(ids) / 32)
            if len(buffers) != 2 * expected_buffer_count:
                raise ValueError("local buffer coverage mismatch")
            for arm_index, arm in enumerate(ARMS[2:]):
                part = buffers[arm_index * expected_buffer_count:(arm_index + 1) * expected_buffer_count]
                for index, buffer in enumerate(part):
                    expected_members = ordered[index * 32:(index + 1) * 32]
                    if (buffer.get("arm") != arm or buffer.get("order") != key or
                            buffer.get("buffer_index") != index or
                            buffer.get("sample_ids") != expected_members or
                            buffer.get("buffer_size") != len(expected_members) or
                            buffer.get("numeric_status") != "ok" or
                            len(buffer.get("trace", ())) != 5 or
                            buffer["trace"][0]["parameter_norm_before"] != 0 or
                            not _finite_row(buffer)):
                        raise ValueError("buffer membership/reset/numeric failure")
    return config, assignments


def selected_labels(assignments):
    """This function must only run after complete_scores has returned."""
    wanted = {domain: {row["sample_id"] for row in records}
              for domain, records in assignments.items()}
    raw = read_json(ITW_LABELS)
    if (raw.get("role") != "target_test" or raw.get("dataset_id") != "in_the_wild" or
            raw.get("count") != 3178):
        raise ValueError("ITW target10 labelled manifest mismatch")
    itw = {row["sample_id"]: row["label"] for row in raw["records"]}
    if len(itw) != 3178 or set(itw) != wanted["in_the_wild"]:
        raise ValueError("ITW selected label coverage mismatch")
    codec, codec_attack = {}, {}
    with CODEC_LABELS.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError("Codecfake official dev protocol format changed")
            sid = "dev/" + parts[0]
            if sid in wanted["codecfake"]:
                if sid in codec or parts[1] not in ("real", "fake"):
                    raise ValueError("Codecfake selected label duplicate/unknown")
                codec[sid] = 0 if parts[1] == "real" else 1
                codec_attack[sid] = parts[2]
    if set(codec) != wanted["codecfake"]:
        raise ValueError("Codecfake selected protocol coverage mismatch")
    la, la_attack = {}, {}
    with LA_LABELS.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("ASVspoof2019 LA official dev protocol format changed")
            sid = parts[1]
            if sid in wanted["asv2019_la_dev"]:
                if sid in la or parts[4] not in ("bonafide", "spoof"):
                    raise ValueError("ASVspoof2019 LA selected label duplicate/unknown")
                la[sid] = 0 if parts[4] == "bonafide" else 1
                la_attack[sid] = parts[3]
    la_unlabelled = sorted(wanted["asv2019_la_dev"] - set(la))
    if (not set(la) <= wanted["asv2019_la_dev"] or len(la) != 4972 or
            len(la_unlabelled) != 28):
        raise ValueError("ASVspoof2019 LA official protocol coverage differs from recorded correction")
    labels = {"in_the_wild": itw, "codecfake": codec, "asv2019_la_dev": la}
    if any(type(value) is not int or value not in (0, 1)
           for domain in labels for value in labels[domain].values()):
        raise ValueError("noncanonical selected label")
    coverage = {"asv2019_la_dev": {"scored_count": 5000,
                                    "official_protocol_labelled_count": 4972,
                                    "unlabelled_sample_ids": la_unlabelled,
                                    "policy": "retain_all_scores_analyze_only_official_protocol_covered_ids"}}
    return labels, {"codecfake": codec_attack, "asv2019_la_dev": la_attack}, coverage


def fast_eer_auc(scores, labels):
    """Vectorized equivalent of project EER/AUC for paired bootstrap draws."""
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives or not np.isfinite(values).all():
        raise ValueError("two finite classes required")
    order = np.argsort(-values, kind="mergesort")
    sorted_values, sorted_y = values[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_values) != 0) + 1, len(values)]
    cumulative_positive = np.cumsum(sorted_y, dtype=np.int64)[ends - 1]
    cumulative_negative = ends - cumulative_positive
    far = np.r_[0., cumulative_negative / negatives]
    tpr = np.r_[0., cumulative_positive / positives]
    frr = 1. - tpr
    auc = float(np.trapz(tpr, far))
    difference = far - frr
    crossing = int(np.searchsorted(difference, 0., side="left"))
    if difference[crossing] == 0:
        eer = float(far[crossing])
    else:
        left, right = crossing - 1, crossing
        fraction = abs(difference[left]) / (abs(difference[left]) + abs(difference[right]))
        far_cross = far[left] + fraction * (far[right] - far[left])
        frr_cross = frr[left] + fraction * (frr[right] - frr[left])
        eer = float((far_cross + frr_cross) / 2.)
    return eer, auc


def bootstraps(y, arm_scores):
    labels = np.asarray(y, dtype=np.int8)
    rng = np.random.default_rng(2026)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    if not len(class0) or not len(class1):
        return {}
    draws = {pair: ([], []) for pair in COMPARISONS}
    arrays = {arm: np.asarray(scores, dtype=np.float64) for arm, scores in arm_scores.items()}
    for _ in range(1000):
        selected = np.r_[rng.choice(class0, len(class0), replace=True),
                         rng.choice(class1, len(class1), replace=True)]
        metric = {arm: fast_eer_auc(scores[selected], labels[selected])
                  for arm, scores in arrays.items()}
        for pair, (eers, aucs) in draws.items():
            after, before = pair
            eers.append(metric[after][0] - metric[before][0])
            aucs.append(metric[after][1] - metric[before][1])
    return {pair: {"delta_EER_ci_low": float(np.quantile(eers, .025)),
                   "delta_EER_ci_high": float(np.quantile(eers, .975)),
                   "delta_AUC_ci_low": float(np.quantile(aucs, .025)),
                   "delta_AUC_ci_high": float(np.quantile(aucs, .975))}
            for pair, (eers, aucs) in draws.items()}


def read_order_scores(run, domain, key, ids):
    rows = [json.loads(line) for line in
            (run / "scores" / (domain + "_" + key + ".jsonl")).open(encoding="utf-8")]
    by_id = {sid: {} for sid in ids}
    for row in rows:
        if row["sample_id"] in by_id:
            by_id[row["sample_id"]][row["arm"]] = row
    return by_id


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values)


def analyze(run):
    config, assignments = complete_scores(run)  # Required before any selected label reader.
    labels, attacks, coverage = selected_labels(assignments)
    per_order, bootstrap_rows, subset_rows = [], [], []
    composition = {}
    for domain in DOMAINS:
        ids = [record["sample_id"] for record in assignments[domain]
               if record["sample_id"] in labels[domain]]
        y = [labels[domain][sid] for sid in ids]
        composition[domain] = {"assigned_and_scored": len(assignments[domain]),
                               "official_protocol_labelled": len(y),
                               "bonafide": len(y) - sum(y), "spoof": sum(y),
                               "attack_counts": dict(Counter(attacks.get(domain, {}).values()))}
        tau0 = read_json(run / "diagnostics" / (domain + "_provenance.json"))["tau0"]
        if not (0 < sum(y) < len(y)):
            composition[domain]["ranking_status"] = "SINGLE_CLASS_UNAVAILABLE"
            continue
        composition[domain]["ranking_status"] = "TWO_CLASS"
        for key in ORDERS:
            rows = read_order_scores(run, domain, key, ids)
            arm_scores = {arm: [rows[sid][arm]["score_after"] for sid in ids] for arm in ARMS}
            frozen = arm_scores["Frozen"]
            point = {}
            for arm in ARMS:
                values = arm_scores[arm]
                result = binary_metrics(values, y, tau0, frozen_scores=frozen)
                point[arm] = result
                records = [rows[sid][arm] for sid in ids]
                per_order.append({"domain": domain, "order": key, "arm": arm, "count": len(ids),
                                  "bonafide_count": len(y) - sum(y), "spoof_count": sum(y),
                                  "EER": result["eer"], "AUC": result["auroc"],
                                  "balanced_accuracy": result["balanced_accuracy"],
                                  "FPR": result["fpr"], "FNR": result["fnr"],
                                  "delta_EER_vs_Frozen": result["eer"] -
                                  binary_metrics(frozen, y, tau0)["eer"],
                                  "delta_AUC_vs_Frozen": result["auroc"] -
                                  binary_metrics(frozen, y, tau0)["auroc"],
                                  "mean_abs_score_delta": statistics.mean(
                                      abs(r["score_delta"]) for r in records),
                                  "mean_R_norm": statistics.mean(r["R_norm"] for r in records),
                                  "mean_source_evidence_damage": statistics.mean(
                                      r["source_evidence_damage"] for r in records),
                                  "numeric_failures": 0,
                                  "mean_runtime_seconds": statistics.mean(
                                      r["runtime_seconds"] for r in records)})
            intervals = bootstraps(y, arm_scores)
            for after, before in COMPARISONS:
                bootstrap_rows.append({"domain": domain, "order": key, "after": after,
                                       "before": before, "replicates": 1000, "seed": 2026,
                                       "delta_EER": point[after]["eer"] - point[before]["eer"],
                                       "delta_AUC": point[after]["auroc"] - point[before]["auroc"],
                                       **intervals[(after, before)]})
            if domain == "codecfake":
                membership = {r["sample_id"]: r["old_fixed512_member"] for r in assignments[domain]}
                for subset, wanted in (("prior_fixed512", True), ("additional4488", False)):
                    selected = [index for index, sid in enumerate(ids) if membership[sid] is wanted]
                    suby = [y[index] for index in selected]
                    if not (0 < sum(suby) < len(suby)):
                        continue
                    f = [frozen[index] for index in selected]
                    frozen_auc = binary_metrics(f, suby, tau0)["auroc"]
                    for arm in ARMS[2:]:
                        value = [arm_scores[arm][index] for index in selected]
                        subset_rows.append({"order": key, "subset": subset, "arm": arm,
                                            "count": len(selected), "bonafide_count": len(suby) - sum(suby),
                                            "spoof_count": sum(suby),
                                            "delta_AUC_vs_Frozen":
                                            binary_metrics(value, suby, tau0)["auroc"] - frozen_auc})
    if not per_order:
        raise ValueError("no two-class development domain")
    aggregate, variability = [], []
    for domain in DOMAINS:
        domain_rows = [r for r in per_order if r["domain"] == domain]
        if not domain_rows:
            continue
        for arm in ARMS:
            rows = [r for r in domain_rows if r["arm"] == arm]
            auc = [r["AUC"] for r in rows]
            eer = [r["EER"] for r in rows]
            delta = [r["delta_AUC_vs_Frozen"] for r in rows]
            delta_eer = [r["delta_EER_vs_Frozen"] for r in rows]
            mean_auc, std_auc = mean_std(auc)
            mean_eer, std_eer = mean_std(eer)
            mean_delta, std_delta = mean_std(delta)
            aggregate.append({"domain": domain, "arm": arm, "orders": len(rows),
                              "mean_AUC": mean_auc, "std_AUC": std_auc,
                              "min_AUC": min(auc), "max_AUC": max(auc),
                              "mean_EER": mean_eer, "std_EER": std_eer,
                              "min_EER": min(eer), "max_EER": max(eer),
                              "mean_balanced_accuracy": statistics.mean(
                                  r["balanced_accuracy"] for r in rows),
                              "mean_FPR": statistics.mean(r["FPR"] for r in rows),
                              "mean_FNR": statistics.mean(r["FNR"] for r in rows),
                              "mean_abs_score_delta": statistics.mean(
                                  r["mean_abs_score_delta"] for r in rows),
                              "mean_R_norm": statistics.mean(r["mean_R_norm"] for r in rows),
                              "mean_source_evidence_damage": statistics.mean(
                                  r["mean_source_evidence_damage"] for r in rows),
                              "numeric_failures": sum(r["numeric_failures"] for r in rows),
                              "mean_runtime_seconds": statistics.mean(
                                  r["mean_runtime_seconds"] for r in rows)})
            intervals = [r for r in bootstrap_rows if r["domain"] == domain and
                         r["after"] == arm and r["before"] == "Frozen"]
            variability.append({"domain": domain, "arm": arm,
                                "mean_delta_AUC": mean_delta, "std_delta_AUC": std_delta,
                                "min_delta_AUC": min(delta), "max_delta_AUC": max(delta),
                                "effect_to_order_variability": abs(mean_delta) / (std_delta + 1e-12),
                                "positive_order_count": sum(value > 0 for value in delta),
                                "bootstrap_positive_order_count": sum(
                                    r["delta_AUC_ci_low"] > 0 for r in intervals),
                                "mean_delta_EER": statistics.mean(delta_eer),
                                "max_delta_EER": max(delta_eer)})
    chosen = []
    for arm in ARMS[2:]:
        domains_passing = []
        for v in variability:
            if v["arm"] != arm:
                continue
            if (v["mean_delta_AUC"] >= .005 and v["positive_order_count"] >= 5 and
                    v["effect_to_order_variability"] >= 2 and
                    v["bootstrap_positive_order_count"] >= 4 and
                    v["mean_delta_EER"] <= .005 and v["max_delta_EER"] <= .010):
                domains_passing.append(v["domain"])
        chosen.append({"arm": arm, "passing_domains": domains_passing})
    if any(len(row["passing_domains"]) >= 2 for row in chosen):
        decision = "STRONG_REPLICATED_EFFECT"
    elif any(row["passing_domains"] for row in chosen):
        decision = "DOMAIN_SPECIFIC_EFFECT"
    elif all(next((v["mean_delta_AUC"] for v in variability if v["domain"] == "codecfake"
                   and v["arm"] == arm), 0) < .005 for arm in ARMS[2:]):
        decision = "SMALL_DEVELOPMENT_ARTIFACT"
    else:
        decision = "INCONCLUSIVE"
    analysis = run / "analysis"
    write_csv_once(analysis / "per_order_metrics.csv", per_order)
    write_csv_once(analysis / "aggregate_metrics.csv", aggregate)
    write_csv_once(analysis / "bootstrap.csv", bootstrap_rows)
    write_csv_once(analysis / "effect_variability.csv", variability)
    if subset_rows:
        write_csv_once(analysis / "codecfake_nested_subsets.csv", subset_rows)
    summary = {"status": "DEVELOPMENT_CONFIRMATION_ANALYZED", "run_id": config["run_id"],
               "role": "development_only", "decision": decision,
               "protocol_coverage_correction": coverage,
               "promoted_mechanism": [r for r in chosen if len(r["passing_domains"]) >= 2],
               "arm_domain_checks": chosen, "composition": composition,
               "resource_audit": read_json(HERE / "resource_audit.json"),
               "target90_metrics_accessed": False, "final_holdout_metrics_accessed": False,
               "labels_opened_after_all_scores": True,
               "local_O1_vs_per_sample_O1": "NOT_ESTIMABLE_FOUR_ARM_CONTRACT"}
    write_json_once(analysis / "summary.json", summary)
    with (analysis / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("# Large-scale development confirmation\n\n")
        stream.write("All label-free scores and buffer resets passed exact coverage before selected audit labels opened.\n\n")
        stream.write("| Domain | Arm | Mean AUC ± std | Mean EER ± std | Mean ΔAUC | ΔAUC order std |\n")
        stream.write("|---|---|---:|---:|---:|---:|\n")
        for row in aggregate:
            v = next(x for x in variability if x["domain"] == row["domain"] and x["arm"] == row["arm"])
            stream.write(f"| {row['domain']} | {row['arm']} | {row['mean_AUC']:.6f} ± {row['std_AUC']:.6f} | "
                         f"{row['mean_EER']:.6f} ± {row['std_EER']:.6f} | "
                         f"{v['mean_delta_AUC']:+.6f} | {v['std_delta_AUC']:.6f} |\n")
        stream.write(f"\nPreregistered decision: **{decision}**. Bootstrap intervals are development-only "
                     "diagnostics. No method is implemented in this stage.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    print(result["decision"], args.run)

#!/usr/bin/env python
"""Target10 labels are read only here, after all four sequences complete."""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from protocol import ROOT, PROTOCOLS, manifest_ids, validate_run
sys.path.insert(0, str(ROOT / "src"))
from eptta.evaluation.metrics import binary_metrics


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for k in range(i, j):
            result[ordered[k]] = (i + j - 1) / 2.0
        i = j
    return result


def spearman(before, after):
    a, b = ranks(before), ranks(after)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if not va or not vb:
        return None  # Undefined for constant ranks, never a fabricated zero.
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def summarize(protocol, records, labels, tau0, reference):
    before = [r["score_before_update"] for r in records]
    after = [r["score_after"] for r in records]
    metrics = binary_metrics(after, labels, tau0, frozen_scores=reference)
    instant = [a - b for a, b in zip(after, before)]
    total = [a - b for a, b in zip(after, reference)]
    abs_total = sorted(abs(x) for x in total)
    drifts = [r["parameter_distance_from_source"] for r in records]
    mean = statistics.fmean
    return {
        "protocol": protocol, "EER": metrics["eer"], "AUROC": metrics["auroc"],
        "balanced_accuracy": metrics["balanced_accuracy"], "FPR": metrics["fpr"], "FNR": metrics["fnr"],
        "helpful_flips": metrics["helpful_flips"], "harmful_flips": metrics["harmful_flips"],
        "mean_abs_score_delta": mean(abs_total), "median_abs_score_delta": statistics.median(abs_total),
        "p90_abs_score_delta": abs_total[max(0, math.ceil(0.9 * len(abs_total)) - 1)],
        "mean_abs_drift_before": mean(abs(b - f) for b, f in zip(before, reference)),
        "mean_abs_instant_update": mean(abs(x) for x in instant),
        "mean_abs_total_delta": mean(abs_total),
        "mean_entropy_before": mean(r["entropy_before"] for r in records),
        "mean_entropy_after": mean(r["entropy_after"] for r in records),
        "mean_parameter_delta_norm": mean(r["parameter_delta_norm"] for r in records),
        "bonafide_mean_score_delta": mean(x for x, y in zip(total, labels) if y == 0),
        "spoof_mean_score_delta": mean(x for x, y in zip(total, labels) if y == 1),
        "adaptation_coverage": mean(int(r["adaptation_applied"]) for r in records),
        "numeric_failures": sum(r["numeric_failure"] for r in records),
        "resource_failures": sum(r["resource_failure"] for r in records),
        "runtime_per_sample": mean(r["runtime"] for r in records),
        "source_reference_runtime_per_sample": mean(r["source_reference_runtime"] for r in records),
        "spearman_before_after": spearman(before, after),
        "final_parameter_drift": drifts[-1], "max_parameter_drift": max(drifts),
        "mean_parameter_drift": mean(drifts),
    }


def read_labels(path, ids):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["records"]
    result = {r["sample_id"]: r["label"] for r in rows}
    if (len(result) != len(rows) or set(result) != set(ids) or
            any(type(y) is not int or y not in (0, 1) for y in result.values())):
        raise ValueError("duplicate, noncanonical, or mismatched target10 labels")
    return [result[sid] for sid in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.run_dir / name for name in ("metrics.csv", "summary.json", "report.md")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite aggregate artifacts")
    ids = manifest_ids()
    records, config = validate_run(args.run_dir, ids)
    labels = read_labels(ROOT / "experiments/target10_selection/manifests/inwild_target10.json", ids)
    tau0 = config["tau0"]
    if isinstance(tau0, bool) or not isinstance(tau0, (float, int)) or not math.isfinite(tau0):
        raise ValueError("invalid source threshold")
    reference = [r["source_frozen_score"] for r in records["episodic"]]
    fm = binary_metrics(reference, labels, tau0)
    rows = [{"protocol": "Frozen-Waveform", "EER": fm["eer"], "AUROC": fm["auroc"],
             "balanced_accuracy": fm["balanced_accuracy"], "FPR": fm["fpr"], "FNR": fm["fnr"]}]
    rows.extend(summarize(p, records[p], labels, tau0, reference) for p in PROTOCOLS)
    summary = {"schema_version": "0.1.0", "comparison_track": "protocol_tta",
               "status": "COMPLETE", "scope": "target10 development; fixed manifest order",
               "primary_reference": "source_frozen_score from unadapted waveform path",
               "threshold_policy": "source tau0; no target threshold selection", "tau0": tau0,
               "sample_count": len(ids), "parameters": config["parameters"],
               "manifest_path": config["manifest_path"],
               "checkpoint_ref": config["checkpoint_ref"], "baseline_id": config["baseline_id"], "rows": rows,
               "scientific_conclusion": "Single fixed-order protocol comparison; no significance or target90 claim."}
    with outputs[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# TENT protocol study — target10", "",
             "固定 manifest 顺序；相同 checkpoint、TENT 参数、源 BN 统计与源阈值 tau0。",
             "唯一协议变量是模型与 Adam 状态恢复 source/empty 状态的边界。", "",
             "| Protocol | EER | AUROC | Balanced accuracy | FPR | FNR |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {protocol} | {EER:.6f} | {AUROC:.6f} | {balanced_accuracy:.6f} | {FPR:.6f} | {FNR:.6f} |".format(**row))
    lines += ["", "指标均为比例（不是百分数）。完整机制诊断见 metrics.csv / summary.json。",
              "score_before_update 是当前模型的本条更新前分数，不能用作 continual 的共同 Frozen。",
              "score delta、分类别 delta 与 helpful/harmful flips 均相对共同 Source Frozen。",
              "instant_update 相对当前更新前分数；parameter_delta_norm 也只计本条更新。",
              "参数 drift 是每条更新后与 source 选中参数的 L2 距离，final 为末条更新后、重置前。",
              "runtime 包含本条 reset、音频载入、适应与诊断；独立 Frozen pass 耗时另列。",
              "本次只比较一个固定顺序，不据此宣称统计显著或 held-out 泛化。", ""]
    with outputs[2].open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

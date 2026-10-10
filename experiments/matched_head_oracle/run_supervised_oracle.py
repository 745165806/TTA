"""Matched label-using B128 single-step linear-head development oracle.

This deliberately uses selected development labels for each batch's update and
then scores the same batch. It is not an unlabeled TTA worker or final result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eptta.evaluation.metrics import binary_metrics
from experiments.head_capacity_geometry.resources import ROOT, load_domain
from experiments.head_capacity_geometry.supervised_labels import load_labels
from experiments.o2_strength_audit.analyze import fast_eer_auc
from experiments.o2_strength_audit.run_scores import load_itw


HERE = Path(__file__).resolve().parent
O2_RUN = ROOT / "experiments/o2_strength_audit/results/o2_strength_dev_20260927a"
BATCH = 128
EPS = 1e-12
DELTAS = (0.01, 0.03, 0.10)
SUP = {delta: f"SUP-normalized-{delta:.2f}" for delta in DELTAS}
O2 = {delta: f"O2-normalized-{delta:.2f}" for delta in DELTAS}
ARMS = ("Frozen", *SUP.values(), *O2.values())
COUNTS = {"itw": 3178, "wavefake": 4096}


def write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv_new(path, rows):
    if not rows:
        raise ValueError("empty CSV")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def previous_o2_scores(domain, ids):
    completion = json.loads((O2_RUN / "score_completion.json").read_text(encoding="utf-8"))
    if (completion["complete"] is not True or completion["buffer_size"] != BATCH or
            completion["order"] != "fixed_manifest" or
            completion["domains"][domain]["count"] != len(ids) or
            completion["target_labels_read"] is not False):
        raise ValueError(f"{domain}: prior O2 completion incompatible")
    rows = [json.loads(line) for line in
            (O2_RUN / "scores" / f"{domain}.jsonl").open(encoding="utf-8")]
    if len(rows) != len(ids) or [row["sample_id"] for row in rows] != ids:
        raise ValueError(f"{domain}: prior O2 ID/order coverage mismatch")
    names = {"Frozen": "Frozen", "O2-normalized-0.01": "O2-normalized-small",
             "O2-normalized-0.03": "O2-normalized-medium",
             "O2-normalized-0.10": "O2-normalized-large"}
    if any(row["domain"] != domain or row["buffer_index"] != i // BATCH or
           not all(math.isfinite(float(row["scores"][old])) for old in names.values())
           for i, row in enumerate(rows)):
        raise ValueError(f"{domain}: prior O2 numeric/buffer mismatch")
    result = {new: np.asarray([row["scores"][old] for row in rows], dtype=np.float64)
              for new, old in names.items()}
    return result, completion


def one_supervised_buffer(views, labels, resources):
    z = torch.from_numpy(np.asarray(views, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    if z.ndim != 3 or z.shape[1:] != (3, 160) or y.shape != (len(z),):
        raise ValueError("supervised batch shape mismatch")
    w = resources.w.detach().clone().requires_grad_(True)
    b = float(resources.b)
    loss = F.binary_cross_entropy_with_logits(z[:, 0] @ w + b, y)
    gradient, = torch.autograd.grad(loss, w)
    source_norm = float(torch.linalg.vector_norm(resources.w))
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    if (not bool(torch.isfinite(gradient).all()) or
            source_norm <= 0 or gradient_norm <= 0 or not math.isfinite(gradient_norm)):
        raise FloatingPointError("invalid supervised gradient")
    original = z[:, 0].double()
    frozen = original @ resources.w.double() + b
    scores = {}
    diagnostics = {}
    for delta in DELTAS:
        arm = SUP[delta]
        head = resources.w.detach() - (
            delta * source_norm / (gradient_norm + EPS)) * gradient.detach()
        value = original @ head.double() + b
        step = head - resources.w
        cosine = float(torch.dot(head, resources.w) / (
            torch.linalg.vector_norm(head) * torch.linalg.vector_norm(resources.w)))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        if not bool(torch.isfinite(value).all()) or not math.isfinite(angle):
            raise FloatingPointError("nonfinite supervised candidate")
        scores[arm] = value.numpy()
        diagnostics[arm] = {
            "supervised_loss_before": float(loss.detach()),
            "supervised_gradient_norm": gradient_norm,
            "source_head_norm": source_norm,
            "head_angle_degrees": angle,
            "effective_step_norm": float(torch.linalg.vector_norm(step)),
            "mean_abs_score_delta": float((value - frozen).abs().mean()),
            "numeric_status": "ok"}
    return frozen.numpy(), scores, diagnostics


def bootstrap(domain, labels, scores):
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(2026)
    comparisons = [(SUP[d], "Frozen") for d in DELTAS] + [
        (O2[d], "Frozen") for d in DELTAS] + [(SUP[d], O2[d]) for d in DELTAS]
    draws = {pair: ([], []) for pair in comparisons}
    for _ in range(1000):
        selected = np.r_[rng.choice(class0, len(class0), replace=True),
                         rng.choice(class1, len(class1), replace=True)]
        measured = {arm: fast_eer_auc(values[selected], labels[selected])
                    for arm, values in scores.items()}
        for (after, before), (eers, aucs) in draws.items():
            eers.append(measured[after][0] - measured[before][0])
            aucs.append(measured[after][1] - measured[before][1])
    return [{"domain": domain, "after": after, "before": before,
             "replicates": 1000, "seed": 2026,
             "delta_auc_ci_low": float(np.quantile(draws[(after, before)][1], .025)),
             "delta_auc_ci_high": float(np.quantile(draws[(after, before)][1], .975)),
             "delta_eer_ci_low": float(np.quantile(draws[(after, before)][0], .025)),
             "delta_eer_ci_high": float(np.quantile(draws[(after, before)][0], .975))}
            for after, before in comparisons]


def classify(metrics):
    def meets(domain, arm):
        row = metrics[domain][arm]
        floor = .005 if domain == "itw" else .01
        return row["delta_auc"] >= floor or row["delta_eer"] <= -floor
    cross = [delta for delta in DELTAS if meets("itw", SUP[delta]) and
             meets("wavefake", SUP[delta]) and
             not meets("itw", O2[delta]) and not meets("wavefake", O2[delta])]
    if cross:
        return "UNSUPERVISED_DIRECTION_IS_PRIMARY_BOTTLENECK", cross
    if all(not meets(domain, SUP[delta]) for domain in COUNTS for delta in DELTAS):
        return "SINGLE_STEP_HEAD_PROTOCOL_INSUFFICIENT", []
    if (any(meets("itw", SUP[delta]) for delta in DELTAS) !=
            any(meets("wavefake", SUP[delta]) for delta in DELTAS)):
        return "DOMAIN_DEPENDENT_HEAD_CORRECTION_TIMESCALE", []
    return "INCONCLUSIVE", []


def run(run_id):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    torch.set_num_threads(1)
    out = HERE / "results" / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "scores").mkdir()
    try:
        wave_ids, wave_views, _, resources, wave_info = load_domain("wavefake")
        itw_ids, itw_views, itw_info = load_itw(resources)
        features = {"itw": (itw_ids, itw_views),
                    "wavefake": (wave_ids, wave_views)}
        old_scores = {domain: previous_o2_scores(domain, ids)
                      for domain, (ids, _) in features.items()}
        # Only selected development labels enter this explicitly supervised oracle.
        labels = {domain: load_labels(domain, ids)
                  for domain, (ids, _) in features.items()}
        all_scores = {}
        diagnostics = []
        for domain, (ids, views) in features.items():
            if len(ids) != COUNTS[domain] or views.shape != (len(ids), 3, 160) or \
                    views.dtype != np.float32 or not np.isfinite(views).all():
                raise ValueError(f"{domain}: selected feature coverage/schema failure")
            prior, _ = old_scores[domain]
            collected = {arm: [] for arm in SUP.values()}
            frozen = []
            for begin in range(0, len(ids), BATCH):
                start = time.perf_counter()
                part_frozen, part_scores, part_diag = one_supervised_buffer(
                    views[begin:begin+BATCH], labels[domain][begin:begin+BATCH], resources)
                runtime = time.perf_counter() - start
                frozen.extend(part_frozen.tolist())
                for arm in SUP.values():
                    collected[arm].extend(part_scores[arm].tolist())
                    diagnostics.append({"domain": domain, "buffer_index": begin // BATCH,
                        "buffer_size": len(part_frozen), "arm": arm,
                        "runtime_seconds_for_buffer": runtime, **part_diag[arm]})
            difference = float(np.max(np.abs(np.asarray(frozen) - prior["Frozen"])))
            if difference != 0:
                raise ValueError(f"{domain}: Frozen score parity failure {difference}")
            scores = {"Frozen": prior["Frozen"], **{
                arm: np.asarray(values, dtype=np.float64)
                for arm, values in collected.items()}, **{
                arm: prior[arm] for arm in O2.values()}}
            if any(len(values) != len(ids) or not np.isfinite(values).all()
                   for values in scores.values()):
                raise ValueError(f"{domain}: exact finite score coverage failure")
            with (out / "scores" / f"{domain}.jsonl").open("x", encoding="utf-8") as stream:
                for index, sample_id in enumerate(ids):
                    stream.write(json.dumps({"sample_id": sample_id, "domain": domain,
                        "buffer_index": index // BATCH,
                        "scores": {arm: float(scores[arm][index]) for arm in ARMS}},
                        allow_nan=False) + "\n")
            all_scores[domain] = scores
            print(f"{domain}: {len(ids)} matched oracle scores, Frozen parity {difference}", flush=True)
        write_csv_new(out / "diagnostics.csv", diagnostics)
        config = {"run_id": run_id, "role": "SUPERVISED_DEVELOPMENT_ORACLE_NOT_TTA",
            "branch": subprocess.check_output(["git", "branch", "--show-current"],
                cwd=ROOT, text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=ROOT, text=True).strip(),
            "command": [sys.executable, *sys.argv], "python": sys.version,
            "torch": torch.__version__, "sample_counts": COUNTS,
            "buffer_size": BATCH, "one_step": True, "order": "fixed_manifest",
            "reset_policy": "source_head_each_buffer", "encoder_frozen": True,
            "bias_frozen": True, "normalized_deltas": DELTAS,
            "o2_result_ref": str(O2_RUN), "itw_cache": itw_info,
            "wavefake_cache": wave_info, "target90_labels_or_metrics_accessed": False,
            "final_heldout_metrics_accessed": False}
        write_json_new(out / "run_config.json", config)
        threshold = float(wave_info["tau0"])
        table = []
        metrics = {}
        intervals = []
        for domain, (ids, _) in features.items():
            scores = all_scores[domain]
            y = labels[domain].tolist()
            frozen_metric = binary_metrics(scores["Frozen"].tolist(), y, threshold)
            metrics[domain] = {}
            for arm in ARMS:
                measure = binary_metrics(scores[arm].tolist(), y, threshold)
                diag_group = [row for row in diagnostics if row["domain"] == domain and
                              row["arm"] == arm]
                if arm == "Frozen":
                    geometry = {"mean_head_angle_degrees": 0.0,
                        "mean_effective_step_norm": 0.0,
                        "mean_abs_score_movement": 0.0}
                elif arm in SUP.values():
                    geometry = {field: sum(int(row["buffer_size"]) * float(row[key])
                                    for row in diag_group) / len(ids)
                                for field, key in (("mean_head_angle_degrees", "head_angle_degrees"),
                                    ("mean_effective_step_norm", "effective_step_norm"),
                                    ("mean_abs_score_movement", "mean_abs_score_delta"))}
                else:
                    old = json.loads((O2_RUN / "summary.json").read_text(encoding="utf-8"))
                    source_arm = {O2[.01]: "O2-normalized-small",
                                  O2[.03]: "O2-normalized-medium",
                                  O2[.10]: "O2-normalized-large"}[arm]
                    old_metric = old["metrics"][domain][source_arm]
                    if abs(measure["auroc"] - old_metric["auc"]) > 1e-12 or \
                            abs(measure["eer"] - old_metric["eer"]) > 1e-12:
                        raise ValueError(f"{domain}/{arm}: prior O2 metric parity failure")
                    geometry = {"mean_head_angle_degrees": old_metric["mean_head_angle_degrees"],
                        "mean_effective_step_norm": old_metric["mean_effective_step_norm"],
                        "mean_abs_score_movement": old_metric["mean_abs_score_delta"]}
                row = {"domain": domain, "arm": arm, "count": len(ids),
                    "auc": measure["auroc"], "eer": measure["eer"],
                    "delta_auc": measure["auroc"] - frozen_metric["auroc"],
                    "delta_eer": measure["eer"] - frozen_metric["eer"], **geometry}
                if not all(math.isfinite(value) for value in row.values()
                           if isinstance(value, float)):
                    raise FloatingPointError("nonfinite oracle metric")
                table.append(row)
                metrics[domain][arm] = row
            intervals.extend(bootstrap(domain, labels[domain], scores))
        matched = {domain: {str(delta): {
            "sup_minus_o2_auc": metrics[domain][SUP[delta]]["auc"] -
                                metrics[domain][O2[delta]]["auc"],
            "sup_minus_o2_eer": metrics[domain][SUP[delta]]["eer"] -
                                metrics[domain][O2[delta]]["eer"]}
            for delta in DELTAS} for domain in COUNTS}
        decision, qualifying = classify(metrics)
        summary = {"run_id": run_id, "role": "SUPERVISED_DEVELOPMENT_ORACLE_NOT_TTA",
            "decision": decision, "qualifying_deltas": qualifying,
            "metrics": metrics, "matched_sup_minus_o2": matched,
            "bootstrap": intervals,
            "labels_used_in_same_batch_gradient_and_evaluation": True,
            "new_tta_method_implemented": False,
            "target90_labels_or_metrics_accessed": False,
            "final_heldout_metrics_accessed": False}
        write_csv_new(out / "metrics.csv", table)
        write_csv_new(out / "bootstrap.csv", intervals)
        write_json_new(out / "summary.json", summary)
        with (out / "report.md").open("x", encoding="utf-8") as stream:
            stream.write("# Matched supervised one-step head oracle\n\n")
            stream.write("**THIS IS SUPERVISED DEVELOPMENT ORACLE DIAGNOSIS. NOT A TTA METHOD. NOT FINAL PERFORMANCE.**\n\n")
            stream.write("Each B128 update uses the same batch's true selected development labels and is evaluated on that batch. This is an optimistic protocol capacity check, not held-out head fitting.\n\n")
            stream.write("| Domain | Arm | AUC | EER | ΔAUC | ΔEER | Mean angle (deg) | Mean step norm | Mean abs score movement |\n")
            stream.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in table:
                stream.write(f"| {row['domain']} | {row['arm']} | {row['auc']:.6f} | {row['eer']:.6f} | {row['delta_auc']:+.6f} | {row['delta_eer']:+.6f} | {row['mean_head_angle_degrees']:.6f} | {row['mean_effective_step_norm']:.6f} | {row['mean_abs_score_movement']:.6f} |\n")
            stream.write("\n| Domain | δ | SUP−O2 ΔAUC | SUP−O2 ΔEER |\n|---|---:|---:|---:|\n")
            for domain in COUNTS:
                for delta in DELTAS:
                    contrast = matched[domain][str(delta)]
                    stream.write(f"| {domain} | {delta:.2f} | {contrast['sup_minus_o2_auc']:+.6f} | {contrast['sup_minus_o2_eer']:+.6f} |\n")
            stream.write(f"\nDecision: **{decision}**. Paired 1,000-draw development intervals, including direct same-δ SUP versus O2 contrasts, are in `bootstrap.csv`.\n")
        return {"decision": decision, "metrics": metrics}
    except BaseException:
        write_json_new(out / "failure.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    print(json.dumps(run(parser.parse_args().run_id), indent=2))

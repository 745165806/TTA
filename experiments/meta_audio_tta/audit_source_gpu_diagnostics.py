"""Verify fixed source-only score artifacts from completed real-model diagnostics."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eptta.evaluation.metrics import binary_metrics


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pair_flips(rows):
    real = [row for row in rows if row["canonical_label"] == 0]
    fake = [row for row in rows if row["canonical_label"] == 1]
    helpful = harmful = ties = 0
    for f in fake:
        for r in real:
            before = (f["k0"] > r["k0"]) - (f["k0"] < r["k0"])
            after = (f["k1"] > r["k1"]) - (f["k1"] < r["k1"])
            helpful += before < 0 and after > 0
            harmful += before > 0 and after < 0
            ties += before != after and (before == 0 or after == 0)
    return {"total_pairs": len(fake) * len(real),
            "incorrect_to_correct": int(helpful),
            "correct_to_incorrect": int(harmful),
            "tie_status_changed": int(ties)}


def audit_variant(variant, directory, checkpoint, expected_config=None):
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (config["variant"] != variant or summary["variant"] != variant or
            config["checkpoint"] != checkpoint or summary["checkpoint"] != checkpoint or
            summary["status"] != "PASS" or summary["checkpoint_state_unchanged"] is not True):
        raise ValueError(variant + " checkpoint or summary status mismatch")
    diagnostic_config = config["diagnostic_config"]
    if not diagnostic_config["source_only"] or (expected_config is not None and
            diagnostic_config != expected_config):
        raise ValueError(variant + " source-only diagnostic config mismatch")
    rows = read_jsonl(directory / "source_val.scores.jsonl")
    ids = [row["sample_id"] for row in rows]
    expected_ids = config["source_val_ids"]
    if (len(rows) != 64 or ids != expected_ids or len(set(ids)) != 64 or
            {row["canonical_label"] for row in rows} != {0, 1} or
            sum(row["canonical_label"] for row in rows) != 32):
        raise ValueError(variant + " source_val ID/class coverage mismatch")
    scores0 = [row["k0"] for row in rows]
    if any(not math.isfinite(value) for value in scores0):
        raise ValueError(variant + " nonfinite Frozen score")
    labels = [row["canonical_label"] for row in rows]
    k0 = binary_metrics(scores0, labels, 0.0)
    if k0 != summary["source_val"]["k0"]:
        raise ValueError(variant + " Frozen metric mismatch")
    result = {"variant": variant, "checkpoint": checkpoint,
              "source_val_count": len(rows), "frozen_eer": k0["eer"],
              "frozen_auc": k0["auroc"],
              "frozen_error_count": k0["fp"] + k0["fn"],
              "source_val_ids_match": True, "checkpoint_unchanged": True}
    if variant == "ce":
        if (any(row["k1"] is not None for row in rows) or
                not str(summary["source_val"]["k1"]).startswith("NOT_RUN") or
                (directory / "fit_acoustic.scores.jsonl").exists()):
            raise ValueError("CE-only Frozen artifact fabricated K=1 or fit acoustic scores")
        result["adapted"] = "NOT_RUN: CE-only has no trained BYOL head"
        return result, diagnostic_config, expected_ids
    scores1 = [row["k1"] for row in rows]
    if any(value is None or not math.isfinite(value) for value in scores1):
        raise ValueError(variant + " missing/nonfinite adapted score")
    k1 = binary_metrics(scores1, labels, 0.0, scores0)
    if k1 != summary["source_val"]["k1"]:
        raise ValueError(variant + " adapted metric mismatch")
    pairs = pair_flips(rows)
    if pairs != summary["source_val"]["pair_order_flips"]:
        raise ValueError(variant + " fake-real pair flip mismatch")
    nonzero_scores = sum(abs(a - b) > 1e-8 for a, b in zip(scores1, scores0))
    nonzero_backbone = sum(row["backbone_update_l1"] > 0 for row in rows)
    if (nonzero_scores != summary["source_val"]["score_change"]["nonzero_count"] or
            nonzero_backbone != summary["source_val"]["update"]["backbone_update_l1"]["nonzero_count"]):
        raise ValueError(variant + " detector score or backbone update mismatch")
    for row in rows:
        if any(not math.isfinite(row[key]) for key in
               ("byol_loss", "backbone_gradient_l1", "aux_gradient_l1",
                "backbone_update_l1", "aux_update_l1", "projection_norm",
                "target_projection_norm", "projection_cosine")):
            raise ValueError(variant + " nonfinite BYOL/update diagnostic")
    fit_rows = read_jsonl(directory / "fit_acoustic.scores.jsonl")
    fit_ids = config["fit_ids"]
    if (len(fit_ids) != 16 or len(set(fit_ids)) != 16 or
            set(fit_ids) & set(expected_ids) or len(fit_rows) != 32 or
            any([row["sample_id"] for row in fit_rows if row["condition"] == condition] != fit_ids
                for condition in ("original", "deterministic_fir"))):
        raise ValueError(variant + " fit acoustic pair coverage mismatch")
    paired = {}
    for condition in ("original", "deterministic_fir"):
        selected = [row for row in fit_rows if row["condition"] == condition]
        paired[condition] = {row["sample_id"]: row for row in selected}
        if any(not math.isfinite(row[key]) for row in selected
               for key in ("k0", "k1", "byol_loss", "backbone_gradient_l1",
                           "backbone_update_l1")):
            raise ValueError(variant + " nonfinite fit acoustic diagnostic")
        fit_labels = [row["canonical_label"] for row in selected]
        fit_k0 = [row["k0"] for row in selected]
        fit_k1 = [row["k1"] for row in selected]
        reported = summary["fit_acoustic"][condition]
        if (binary_metrics(fit_k0, fit_labels, 0.0) != reported["k0"] or
                binary_metrics(fit_k1, fit_labels, 0.0, fit_k0) != reported["k1"] or
                pair_flips(selected) != reported["pair_order_flips"]):
            raise ValueError(variant + " fit acoustic metric mismatch")
    def sign(value):
        return (value > 0) - (value < 0)
    original, fir = paired["original"], paired["deterministic_fir"]
    sign_changes = sum(sign(original[key]["k1"] - original[key]["k0"]) !=
                       sign(fir[key]["k1"] - fir[key]["k0"]) for key in fit_ids)
    shift_reversal = sum((fir[key]["k0"] - original[key]["k0"]) *
                         (fir[key]["k1"] - original[key]["k1"]) < 0
                         for key in fit_ids)
    reported_pair = summary["fit_acoustic"]["perturbation_pair"]
    if (reported_pair["count"] != 16 or
            reported_pair["adaptation_delta_sign_changed"] != sign_changes or
            reported_pair["k0_changed_direction"] != shift_reversal):
        raise ValueError(variant + " fit acoustic direction mismatch")
    representation = summary["representation"]
    if (representation["count"] != 64 or
            any(not math.isfinite(value) for value in representation.values())):
        raise ValueError(variant + " nonfinite representation summary")
    result.update({"adapted_eer": k1["eer"], "adapted_auc": k1["auroc"],
                   "adapted_error_count": k1["fp"] + k1["fn"],
                   "helpful_decision_flips": k1["helpful_flips"],
                   "harmful_decision_flips": k1["harmful_flips"],
                   "pair_flips": pairs, "nonzero_score_changes": nonzero_scores,
                   "nonzero_backbone_updates": nonzero_backbone,
                   "mean_abs_score_change": summary["source_val"]["score_change"]["mean_abs"],
                   "mean_backbone_update_l1": summary["source_val"]["update"]["backbone_update_l1"]["mean"],
                   "fit_acoustic_count_per_condition": 16,
                   "fir_adaptation_direction_changes": summary["fit_acoustic"]["perturbation_pair"]["adaptation_delta_sign_changed"]})
    return result, diagnostic_config, expected_ids


def main():
    parser = argparse.ArgumentParser()
    for name in ("ce", "joint", "cross", "same"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
    parser.add_argument("--stage1-gate", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("source diagnostic audit output already exists")
    gate = json.loads(args.stage1_gate.read_text(encoding="utf-8"))
    if gate["status"] != "PASS":
        raise ValueError("stage-1 real-model gate is not PASS")
    checkpoints = {name: gate[name]["checkpoint"] for name in ("ce", "joint")}
    for name in ("cross", "same"):
        selection = json.loads((args.stage2_root / name / "selection.json").read_text())
        checkpoints[name] = selection["selected_checkpoint"]
    results, diagnostic_config, expected_ids = {}, None, None
    for name in ("ce", "joint", "cross", "same"):
        result, config, ids = audit_variant(name, getattr(args, name + "_dir"),
                                            checkpoints[name], diagnostic_config)
        if expected_ids is not None and ids != expected_ids:
            raise ValueError("diagnostic source_val IDs differ by checkpoint")
        diagnostic_config, expected_ids = config, ids
        results[name] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"status": "PASS", "source_only": True, "results": results},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "variants": list(results)}, allow_nan=False))


if __name__ == "__main__":
    main()

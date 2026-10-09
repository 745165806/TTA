"""Read-only CPU geometry audit on the existing fixed 512-ID target10 subset.

No target labels, attacks, waveform paths, or previous target scores are loaded.
The output directory must be new.  Large per-sample rows stay local.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from eptta.adaptation.adapter import run_cache_method
from eptta.adaptation.math import apply_adapter, view_loss
from eptta.adaptation.types import EPConfig, TargetViews
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.models.linear_head import export_spoof_score_head
from eptta.offline.artifacts import load_frozen_resources
from eptta.offline.task_subspaces import source_mixed_subspace, source_task_subspace


def selected_ids(manifest):
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if (document.get("role") != "mechanism_select" or document.get("dataset_id") != "in_the_wild"
            or document.get("selection_seed") != 2026
            or document.get("selection_policy") != "fixed_target10_id_sample_without_repartition"
            or document.get("source_ref") !=
                "experiments/target10_selection/manifests/inwild_target10_select.json"
            or document.get("count") != 512 or len(document.get("records", [])) != 512):
        raise ValueError("expected the existing fixed 512-ID In-the-Wild mechanism manifest")
    forbidden = {"label", "canonical_label", "raw_label", "attack_id", "source_labels"}
    if forbidden.intersection(document):
        raise ValueError("target label field in manifest")
    records = document["records"]
    if any(forbidden.intersection(record) or record.get("original_split") != "existing_target10"
           for record in records):
        raise ValueError("target selection contains label fields or wrong assignment")
    ids = [record["sample_id"] for record in records]
    if len(set(ids)) != 512 or ids != sorted(ids):
        raise ValueError("selection IDs are not fixed, sorted, and unique")
    return ids


def selected_features(cache_ref, bundle, wanted):
    cache = FeatureCache(cache_ref)
    identity = cache.index["identity"]
    expected = {"source_run_id": bundle["source_run_id"],
                "checkpoint_ref": bundle["checkpoint_ref"],
                "preprocess": bundle["preprocess"],
                "dataset_id": "in_the_wild", "split_role": "target_test"}
    if (cache.index.get("format") != "sharded_npy_v2"
            or cache.index.get("num_views") != 3
            or cache.index.get("feature_dim") != 160
            or any(identity.get(key) != value for key, value in expected.items())):
        raise ValueError("cache and frozen bundle provenance differ")
    features, seen, wanted = {}, set(), set(wanted)
    for chunk in cache.index["chunks"]:
        ids_path = (cache.root / chunk["ids_ref"]).resolve()
        array_path = (cache.root / chunk["array_ref"]).resolve()
        for path in (ids_path, array_path):
            if not path.is_relative_to(cache.root.resolve()) or not path.is_file():
                raise ValueError("cache chunk escapes root or is missing")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if (len(ids) != chunk["count"] or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("cache chunk IDs are invalid")
        seen.update(ids)
        positions = [(index, sample_id) for index, sample_id in enumerate(ids) if sample_id in wanted]
        if not positions:
            continue
        with array_path.open("rb") as stream:
            array = np.load(stream, allow_pickle=False)
        if (list(array.shape) != chunk["shape"] or array.shape != (chunk["count"], 3, 160)
                or str(array.dtype) != chunk["dtype"]
                or array.dtype.hasobject or not np.isfinite(array).all()):
            raise ValueError("selected feature chunk violates shape, dtype, or finite contract")
        for index, sample_id in positions:
            features[sample_id] = array[index].copy()
    if len(seen) != cache.index["sample_count"] or set(features) != wanted:
        raise ValueError("cache or fixed selected ID coverage mismatch")
    return features, cache.cache_id


def geometry(Z, U, w, rho):
    R = torch.zeros((8, 8), dtype=Z.dtype, requires_grad=True)
    adapted = apply_adapter(Z, U, R)
    objective = view_loss(adapted)
    score = adapted[0] @ w
    grad_view, = torch.autograd.grad(objective, R, retain_graph=True)
    grad_score, = torch.autograd.grad(score, R)
    gv = float(torch.linalg.vector_norm(grad_view))
    gs = float(torch.linalg.vector_norm(grad_score))
    cosine = float((grad_view * grad_score).sum() / (gv * gs)) if gv * gs else 0.0
    bound = rho * float(torch.linalg.vector_norm(U.T @ w)) * float(torch.linalg.vector_norm(Z[0] @ U))
    return {"view_gradient_norm": gv, "score_gradient_norm": gs,
            "gradient_cosine": cosine, "score_change_bound": bound,
            "view_loss_initial": float(objective.detach())}


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize(rows, U, w):
    def column(key):
        return [row[key] for row in rows]
    delta = column("delta_score")
    absolute = [abs(value) for value in delta]
    traces = [trace for row in rows for trace in row["trace"]]
    return {
        "n": len(rows),
        "task_projection_ratio": float((U.T @ w).square().sum() / w.square().sum()),
        "u_orthogonality_max_abs": float((U.T @ U - torch.eye(8, dtype=U.dtype)).abs().max()),
        "score_bound_median": percentile(column("score_change_bound"), .5),
        "score_bound_p90": percentile(column("score_change_bound"), .9),
        "delta_score_mean": float(np.mean(delta)),
        "delta_score_abs_median": percentile(absolute, .5),
        "delta_score_abs_p90": percentile(absolute, .9),
        "delta_score_abs_max": max(absolute),
        "delta_over_bound_median": percentile(column("delta_over_bound"), .5),
        "view_gradient_norm_median": percentile(column("view_gradient_norm"), .5),
        "score_gradient_norm_median": percentile(column("score_gradient_norm"), .5),
        "gradient_cosine_median": percentile(column("gradient_cosine"), .5),
        "gradient_cosine_positive_fraction": float(np.mean([x > 0 for x in column("gradient_cosine")])),
        "first_step_predicted_delta_positive_fraction": float(np.mean([x < 0 for x in column("gradient_cosine")])),
        "view_loss_decreased_fraction": float(np.mean([row["view_loss_final"] < row["view_loss_initial"]
                                                       for row in rows])),
        "view_loss_median_reduction": percentile([row["view_loss_initial"] - row["view_loss_final"]
                                                   for row in rows], .5),
        "r_norm_median": percentile(column("r_norm"), .5),
        "r_norm_p90": percentile(column("r_norm"), .9),
        "projection_step_fraction": float(np.mean([step["projection_applied"] for step in traces])),
        "regularizer_active_step_fraction": float(np.mean([step["regularizer_gradient_norm"] > 0
                                                            for step in traces])),
        "regularizer_gradient_norm_p90": percentile([step["regularizer_gradient_norm"]
                                                       for step in traces], .9),
        "guard_step_fraction": float(np.mean([step.get("margin_guard_applied", False) for step in traces])),
        "guard_revert_step_fraction": float(np.mean([step.get("margin_guard_reverted", False)
                                                     for step in traces])),
        "guard_backtracks_per_step_mean": float(np.mean([step.get("margin_guard_backtracks", 0)
                                                          for step in traces])),
        "numeric_fallback_count": sum(row["status"] != "ok" for row in rows),
        "formula_max_abs_error": max(column("formula_abs_error")),
        "formula_max_rel_error": max(column("formula_rel_error")),
        "bound_max_excess": max(column("bound_excess")),
    }


def plot_svg(summary, output):
    # SVG avoids an optional plotting dependency; all bar heights come from summary.json.
    arms = list(summary)
    maximum = max(summary[arm]["delta_score_abs_median"] for arm in arms) or 1.0
    width, height = 880, 95 + 50 * len(arms)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="25" font-size="16">Median |score change| (logit), fixed unlabeled 512</text>']
    for index, arm in enumerate(arms):
        y = 55 + 50 * index
        value = summary[arm]["delta_score_abs_median"]
        bar = 460 * value / maximum
        lines.extend((f'<text x="20" y="{y + 16}" font-size="12">{arm}</text>',
                      f'<rect x="265" y="{y}" width="{bar:.2f}" height="22" fill="#326aa8"/>',
                      f'<text x="{275 + bar:.2f}" y="{y + 16}" font-size="12">{value:.6g}</text>'))
    lines.append('</svg>')
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    data_root = arguments.data_root.resolve()
    local_root = Path(__file__).resolve().parents[2]
    bundle_ref = data_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _, parity, selection = verify_frozen_export(bundle_ref)
    resources, _, metadata = load_frozen_resources(data_root / "outputs_v2/ssl_aasist/resources", bundle)
    if bundle["embedding_dim"] != 160 or resources.U.shape != (160, 8):
        raise ValueError("expected the trained 160D detector and original 8D subspace")
    state = torch.load(bundle_ref.parent / bundle["detector_state_ref"], map_location="cpu", weights_only=True)
    native = state["model_state"]
    head_w, head_b = export_spoof_score_head(native["out_layer.weight"],
                                               native["out_layer.bias"], bundle["class_index_map"])
    if not torch.allclose(head_w, resources.w, atol=1e-6, rtol=1e-6) or abs(float(head_b) - resources.b) > 1e-6:
        raise ValueError("resource head differs from native spoof-minus-bonafide final layer")
    ids = selected_ids(local_root / "experiments/multidomain_mechanism/manifests/in_the_wild_mechanism_select.json")
    features, cache_id = selected_features(data_root / "outputs_v2/ssl_aasist/cache-target-in_the_wild",
                                            bundle, ids)
    U_by_name = {"U-original": resources.U,
                 "U-task": source_task_subspace(resources.w, resources.anchors_z, resources.anchors_y),
                 "U-mixed": source_mixed_subspace(resources.U, resources.w,
                                                   resources.anchors_z, resources.anchors_y)}
    cfg = EPConfig(steps=10, lr=.3, rho=.2, gamma=.1, lambda_keep=1.0)
    k0_cfg = replace(cfg, steps=0)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    per_arm, all_rows = {}, []
    for name, U in U_by_name.items():
        arm_resources = replace(resources, U=U)
        method_specs = (("ep_tta_guarded", 10), ("ep_tta_guarded", 1),
                        ("ep_tta_guarded", 5), ("ep_tta", 10),
                        ("ep_no_projection", 10)) if name == "U-original" else (
                            ("ep_tta_guarded", 10), ("ep_tta", 10))
        for method, steps in method_specs:
            arm_name = name + "/" + method + ("_K%d" % steps if steps != 10 else "")
            run_cfg = replace(cfg, steps=steps)
            rows = []
            for sample_id in ids:
                Z = torch.from_numpy(features[sample_id])
                target = TargetViews(sample_id, Z, cache_id)
                if name == "U-original" and method == "ep_tta_guarded" and steps == 10:
                    frozen = run_cache_method("frozen", target, arm_resources, cfg)
                    k0 = run_cache_method("ep_tta_guarded", target, arm_resources, k0_cfg)
                    if frozen["score"] != k0["score"] or frozen["score"] != k0["score_before"]:
                        raise ValueError("Frozen/K=0 score parity failed")
                g = geometry(Z, U, resources.w, cfg.rho)
                result = run_cache_method(method, target, arm_resources, run_cfg)
                R = result["R"]
                actual = float(result["score"] - result["score_before"])
                formula = float((U.T @ resources.w) @ R @ (U.T @ Z[0]))
                adapted = apply_adapter(Z, U, R)
                direct = float((adapted[0] @ resources.w + resources.b) -
                               (Z[0] @ resources.w + resources.b))
                bound = g["score_change_bound"]
                row = {"sample_id": sample_id, "subspace": name, "method": method,
                       "steps": steps,
                       "status": result["status"], "delta_score": actual,
                       "formula_abs_error": abs(actual - formula),
                       "formula_rel_error": abs(actual - formula) /
                           max(1.0, abs(actual), abs(formula), abs(result["score_before"])),
                       "direct_abs_error": abs(actual - direct),
                       "bound_excess": max(0.0, abs(actual) - bound),
                       "delta_over_bound": abs(actual) / bound if bound else 0.0,
                       "r_norm": float(torch.linalg.vector_norm(R)),
                       "view_loss_final": float(view_loss(adapted)),
                       "trace": result["trace"], **g}
                if (row["formula_rel_error"] > 2e-5 or
                        (method != "ep_no_projection" and row["bound_excess"] > 2e-5)):
                    raise ValueError("score formula or projected bound failed for %s/%s: %s; actual=%g formula=%g R=%g error=%g excess=%g" %
                                     (name, method, sample_id, actual, formula, row["r_norm"],
                                      row["formula_abs_error"], row["bound_excess"]))
                rows.append(row)
                all_rows.append(row)
            per_arm[arm_name] = summarize(rows, U, resources.w)
    report = {"status": "COMPLETE", "scope": "unlabeled_fixed_target10_512_geometry_only",
              "baseline_commit": "66680217720532388ff241bbb88a8565788cf297",
              "bundle_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
              "selected_epoch": bundle["epoch"], "cache_id": cache_id,
              "class_index_map": bundle["class_index_map"],
              "head_weight_max_abs_error": float((head_w - resources.w).abs().max()),
              "head_bias_abs_error": abs(float(head_b) - resources.b),
              "frozen_parity_status": parity["status"], "source_selection_epoch": selection["selected_epoch"],
              "configuration": asdict(cfg), "arms": per_arm,
              "target_labels_read": False, "target90_labels_read": False,
              "final_holdout_labels_read": False}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "per_sample.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = [key for key in all_rows[0] if key != "trace"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row[key] for key in fields})
    plot_svg(per_arm, output / "score_movement.svg")
    print(json.dumps({"status": report["status"], "n_ids": len(ids), "arms": list(per_arm)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

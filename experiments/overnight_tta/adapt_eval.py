"""Source-selected residual TTA, T3A-batch port and two-domain development scores."""
import argparse
import copy
import csv
import json
import math
import random
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "distribution_conditioned_head"))
from experiments.distribution_conditioned_head.train import load_source, source_head
from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from residual import ResidualDetector


def selected_features(cache_ref, sample_ids, bundle):
    cache = FeatureCache(cache_ref)
    identity = cache.index["identity"]
    if (cache.index["sample_count"] != len(sample_ids) or cache.index["num_views"] != 3 or
            cache.index["feature_dim"] != 160 or identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            identity["source_run_id"] != bundle["source_run_id"]):
        raise ValueError("selected-only cache identity/shape differs from source bundle")
    cache.verify_expected_ids(sample_ids)
    feature = cache.load_by_id()
    array = np.stack([feature[sid] for sid in sample_ids])
    if array.shape != (len(sample_ids), 3, 160) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("malformed selected-only features")
    return torch.from_numpy(array), cache.cache_id


def _load_selected(run, bundle_ref):
    bundle, w, b = source_head(Path(bundle_ref))
    output = Path(__file__).parent / "results" / run
    selections = {arm: json.loads((output / f"{arm}_selection.json").read_text())
                  for arm in ("erm", "residual")}
    weights = {arm: torch.load(selections[arm]["checkpoint_ref"], map_location="cpu", weights_only=True)
               for arm in selections}
    model = ResidualDetector(w, b)
    model.load_state_dict(weights["residual"]["model"])
    model.eval()
    return bundle, w, b, model, weights["erm"], output, selections


@torch.no_grad()
def source_prototypes(model, fit):
    z = fit["x"][:, 0]
    h = model.embed(z)
    labels = fit["y"].numpy().astype(int)
    bona = h[labels == 0].mean(0)
    families = sorted({code for code in fit["attacks"] if code != "-"})
    if families != ["A01", "A02", "A03", "A04", "A05", "A06"]:
        raise ValueError("unexpected spoof family bank")
    spoof = torch.stack([h[[i for i, code in enumerate(fit["attacks"]) if code == family]].mean(0)
                         for family in families])
    scale = float(h.var(dim=0, unbiased=False).mean().clamp_min(1e-6))
    logit_scale = float(model(z).var(unbiased=False).clamp_min(1.0))
    return {"bonafide": bona, "spoof": spoof, "families": families,
            "scale": scale, "logit_scale": logit_scale}


def target_adapt(source_model, views, prototypes, config, lr, seed):
    """Only unlabeled 3-view embeddings and source resources enter adaptation."""
    if views.ndim != 3 or views.shape[1:] != (3, 160) or not torch.isfinite(views).all():
        raise ValueError("target views must be finite [N,3,160]")
    model = copy.deepcopy(source_model)
    model.train()
    for p in model.head.parameters():
        p.requires_grad_(False)
    base = [p.detach().clone() for p in source_model.w1.parameters()] + [
        p.detach().clone() for p in source_model.w2.parameters()]
    params = list(model.w1.parameters()) + list(model.w2.parameters())
    denom = sum(float(p.square().sum()) for p in base)
    optimizer = torch.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    updates = failures = skipped_bona = skipped_spoof = selected_bona = selected_spoof = 0
    losses = []
    started = time.monotonic()
    source_model.eval()
    bona_proto, spoof_proto = prototypes["bonafide"], prototypes["spoof"]
    for _ in range(config["target_epochs"]):
        order = list(range(len(views)))
        rng.shuffle(order)
        for start in range(0, len(order), config["target_batch_size"]):
            z = views[order[start:start + config["target_batch_size"]]]
            if len(z) < 2:
                continue
            with torch.no_grad():
                teacher = source_model(z[:, 0]).sigmoid()
                mask_bona = teacher <= 1 - config["confidence"]
                mask_spoof = teacher >= config["confidence"]
            optimizer.zero_grad(set_to_none=True)
            h0 = model.embed(z[:, 0])
            anchor_terms = []
            if mask_bona.any():
                anchor_terms.append((h0[mask_bona] - bona_proto).square().mean() / prototypes["scale"])
                selected_bona += int(mask_bona.sum())
            else:
                skipped_bona += 1
            if mask_spoof.any():
                hs = h0[mask_spoof]
                nearest = torch.cdist(hs, spoof_proto).argmin(1)
                anchor_terms.append((hs - spoof_proto[nearest]).square().mean() / prototypes["scale"])
                selected_spoof += int(mask_spoof.sum())
            else:
                skipped_spoof += 1
            anchor = torch.stack(anchor_terms).mean() if anchor_terms else h0.sum() * 0
            s0 = model.head(h0).squeeze(-1)
            s1 = model(z[:, 1])
            s2 = model(z[:, 2])
            consistency = ((s1 - s0).square().mean() + (s2 - s0).square().mean()) / (2 * prototypes["logit_scale"])
            move = sum((p - p0).square().sum() for p, p0 in zip(params, base)) / max(denom, 1e-6)
            loss = (config["anchor_weight"] * anchor + config["view_weight"] * consistency +
                    config["move_weight"] * move)
            if not torch.isfinite(loss):
                failures += 1
                raise ValueError("nonfinite target adaptation loss")
            loss.backward()
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
    model.eval()
    moved = math.sqrt(sum(float((p.detach() - p0).square().sum()) for p, p0 in zip(params, base)))
    info = {"updates": updates, "numerical_failures": failures, "selected_bonafide_presentations": selected_bona,
            "selected_spoof_presentations": selected_spoof, "skipped_bonafide_batches": skipped_bona,
            "skipped_spoof_batches": skipped_spoof, "adapter_parameter_delta_norm": moved,
            "mean_target_loss": float(np.mean(losses)), "adapt_seconds": time.monotonic() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0}
    return model, info


def t3a_batch_scores(views, native_w, native_b, k):
    """Offline adaptation of author T3A templates, then one final domain rescore."""
    z = views[:, 0]
    warmup = native_w
    warmup_logits = warmup @ native_w.T + native_b
    warmup_y = warmup_logits.argmax(1)
    if set(warmup_y.tolist()) != {0, 1}:
        raise ValueError("native warmup rows fail two-class T3A contract")
    native_logits = z @ native_w.T + native_b
    pseudo = native_logits.argmax(1)
    supports = torch.cat((warmup, z))
    labels = torch.cat((warmup_y, pseudo))
    entropy = -(torch.cat((warmup_logits, native_logits)).softmax(1) *
                torch.cat((warmup_logits, native_logits)).log_softmax(1)).sum(1)
    selected = []
    for cls in (0, 1):
        indices = torch.where(labels == cls)[0]
        order = torch.argsort(entropy[indices], stable=True)
        selected.append(indices[order[:k]])
    indices = torch.cat(selected)
    normalized = torch.nn.functional.normalize(supports[indices], dim=1)
    one_hot = torch.nn.functional.one_hot(labels[indices], num_classes=2).float()
    template_w = torch.nn.functional.normalize(normalized.T @ one_hot, dim=0)
    adjusted = z @ template_w
    return (adjusted[:, 0] - adjusted[:, 1]).numpy(), {
        "selected_native_spoof": int(len(selected[0])), "selected_native_bonafide": int(len(selected[1])),
        "pseudo_native_spoof": int((pseudo == 0).sum()), "pseudo_native_bonafide": int((pseudo == 1).sum())}


def _source_validation_episodes(val, seed):
    rng = random.Random(seed + 9051)
    bona = [i for i, y in enumerate(val["y"]) if y == 0]
    attacks = sorted({a for a in val["attacks"] if a != "-"})
    for attack in attacks:
        spoof = [i for i, a in enumerate(val["attacks"]) if a == attack]
        b, s = rng.sample(bona, 1152), rng.sample(spoof, 1152)
        support, query = b[:1024] + s[:1024], b[1024:] + s[1024:]
        if set(val["audio_ids"][i] for i in support) & set(val["audio_ids"][i] for i in query):
            raise AssertionError("source validation support/query overlap")
        yield attack, val["x"][support], val["x"][query, 0], val["y"][query]


def choose_target_lr(source_model, prototypes, config, val, seed, output):
    rows = []
    episodes = list(_source_validation_episodes(val, seed))
    for lr in config["target_lr_candidates"]:
        all_scores, all_y = [], []
        for attack, support, query, labels in episodes:
            adapted, info = target_adapt(source_model, support, prototypes, config, lr, seed)
            with torch.no_grad():
                scores = adapted(query).tolist()
            all_scores.extend(scores)
            all_y.extend(int(y) for y in labels.tolist())
            rows.append({"lr": lr, "attack": attack, "updates": info["updates"],
                         "adapt_seconds": info["adapt_seconds"], "adapter_delta_norm": info["adapter_parameter_delta_norm"]})
        metric = binary_metrics(all_scores, all_y, 0.0)
        for row in rows:
            if row["lr"] == lr:
                row["pooled_source_select_auc"] = metric["auroc"]
                row["pooled_source_select_eer"] = metric["eer"]
    with (output / "target_lr_source_selection.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    by_lr = {lr: next(row for row in rows if row["lr"] == lr) for lr in config["target_lr_candidates"]}
    best = min(config["target_lr_candidates"], key=lambda lr: (
        by_lr[lr]["pooled_source_select_eer"], -by_lr[lr]["pooled_source_select_auc"], lr))
    (output / "target_lr_selection.json").write_text(json.dumps({
        "selected_on": "source_select_only", "lr": best, "criterion": "EER_then_AUC_then_lower_lr",
        "episodes": len(episodes), "support_count_per_episode": 2048, "query_count_per_episode": 256}, indent=2))
    return best


def source_threshold(scores, labels):
    ordered = sorted(zip(scores, labels), reverse=True)
    pos, neg = sum(labels), len(labels) - sum(labels)
    tp = fp = i = 0
    best = (float("inf"), None)
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]: j += 1
        for _, y in ordered[i:j]:
            tp += y; fp += 1 - y
        threshold = (ordered[j - 1][0] + ordered[j][0]) / 2 if j < len(ordered) else ordered[j - 1][0] - 1e-6
        best = min(best, (abs(fp / neg - (pos - tp) / pos), float(threshold)))
        i = j
    return best[1]


def _pair_ci(scores, reference, labels, groups, repeats):
    by_group = defaultdict(list)
    for i, group in enumerate(groups): by_group[group].append(i)
    keys, rng = sorted(by_group), random.Random(2026)
    auc, eer = [], []
    for _ in range(repeats):
        ids = [i for group in rng.choices(keys, k=len(keys)) for i in by_group[group]]
        y = [labels[i] for i in ids]
        if min(y) == max(y): continue
        a = binary_metrics([float(scores[i]) for i in ids], y, 0.0)
        b = binary_metrics([float(reference[i]) for i in ids], y, 0.0)
        auc.append(a["auroc"] - b["auroc"])
        eer.append(b["eer"] - a["eer"])
    return np.quantile(auc, [.025, .975]).tolist(), np.quantile(eer, [.025, .975]).tolist()


def run(config_ref, run_id, seed):
    config = json.loads(Path(config_ref).read_text())
    deadline = datetime.fromisoformat(config["budget_deadline_utc"].replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("nightly budget expired; no new evaluation")
    torch.set_num_threads(2)
    bundle, source_w, source_b, source_model, erm, output, selections = _load_selected(run_id, config["bundle_ref"])
    if (output / "metrics.csv").exists():
        raise FileExistsError("do not overwrite an existing evaluation")
    fit, val = load_source(config, "fit", bundle), load_source(config, "select", bundle)
    prototypes = source_prototypes(source_model, fit)
    torch.save(prototypes, output / "source_prototypes.pt")
    lr = choose_target_lr(source_model, prototypes, config, val, seed, output)
    detector = torch.load(Path(config["bundle_ref"]).parent / bundle["detector_state_ref"],
                          map_location="cpu", weights_only=True)
    native_w = detector["model_state"]["out_layer.weight"].float()
    native_b = detector["model_state"]["out_layer.bias"].float()
    if (detector["class_index_map"] != {"spoof": 0, "bonafide": 1} or
            not torch.equal(native_w[0] - native_w[1], source_w) or
            not torch.equal(native_b[0] - native_b[1], source_b)):
        raise ValueError("native T3A classifier/source score parity mismatch")
    del detector
    with torch.no_grad():
        select_x = val["x"]
        source_select_scores = {
            "frozen": (select_x[:, 0] @ source_w + source_b).numpy(),
            "erm": (select_x[:, 0] @ erm["w"] + erm["b"]).numpy(),
            "residual_off": source_model(select_x[:, 0]).numpy(),
            "t3a_batch": t3a_batch_scores(select_x, native_w, native_b, config["t3a_k"])[0]}
    select_adapted, _ = target_adapt(source_model, val["x"], prototypes, config, lr, seed)
    with torch.no_grad():
        source_select_scores["residual_on"] = select_adapted(val["x"][:, 0]).numpy()
    labels = [int(y) for y in val["y"].tolist()]
    thresholds = {arm: source_threshold(scores.tolist(), labels) for arm, scores in source_select_scores.items()}
    (output / "source_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    targets = (("itw_target10", config["itw_select"], config["itw_cache"]),
               ("wavefake_dev", config["wavefake_select"], config["wavefake_cache"]))
    all_metrics = []
    for domain, select_ref, cache_ref in targets:
        selection = json.loads(Path(select_ref).read_text())
        if any("label" in row for row in selection["records"]):
            raise ValueError("target select contains label")
        ids = [row["sample_id"] for row in selection["records"]]
        views, cache_id = selected_features(cache_ref, ids, bundle)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        adapted, info = target_adapt(source_model, views, prototypes, config, lr, seed)
        started = time.monotonic()
        with torch.no_grad():
            z = views[:, 0]
            scores = {"frozen": (z @ source_w + source_b).numpy(),
                      "erm": (z @ erm["w"] + erm["b"]).numpy(),
                      "residual_off": source_model(z).numpy(),
                      "residual_on": adapted(z).numpy()}
            scores["t3a_batch"], template_info = t3a_batch_scores(views, native_w, native_b, config["t3a_k"])
        info.update({"inference_seconds": time.monotonic() - started, "cache_id": cache_id,
                     "source_residual_checkpoint": selections["residual"]["checkpoint_ref"],
                     "selected_target_lr": lr, "t3a": template_info,
                     "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        if not all(np.isfinite(v).all() for v in scores.values()) or not np.array_equal(
                scores["residual_off"], source_model(z).detach().numpy()):
            raise ValueError("nonfinite scores or disabled adaptation parity failure")
        # Score files have no labels. Labels are opened only after all scores are durable.
        with (output / f"{domain}_scores.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id"] + list(scores), lineterminator="\n")
            writer.writeheader()
            for i, sid in enumerate(ids):
                writer.writerow({"sample_id": sid, **{key: float(value[i]) for key, value in scores.items()}})
        (output / f"{domain}_runtime.json").write_text(json.dumps(info, indent=2))
        if domain == "itw_target10":
            audit = json.loads(Path(config["itw_audit"]).read_text())
            lab = {r["sample_id"]: r["label"] for r in audit["records"]}
            if len(lab) != len(ids) or set(lab) != set(ids):
                raise ValueError("ITW selected audit coverage mismatch")
            y, groups = [lab[sid] for sid in ids], ids
        else:
            mapping = {"R": 0, **{f"WF{i}": 1 for i in range(1, 8)}}
            y = [mapping[sid.rsplit(":", 1)[1]] for sid in ids]
            groups = [r["audio_id"] for r in selection["records"]]
            if len(set(groups)) != selection["content_groups"]:
                raise ValueError("WaveFake pair-group mismatch")
        metrics = {arm: binary_metrics(value.tolist(), y, thresholds[arm]) for arm, value in scores.items()}
        for arm, metric in metrics.items():
            row = {"domain": domain, "arm": arm, "seed": seed, "count": len(ids),
                   "auc": metric["auroc"], "eer": metric["eer"], "eer_percent": 100 * metric["eer"],
                   "fpr": metric["fpr"], "fnr": metric["fnr"], "balanced_accuracy": metric["balanced_accuracy"],
                   "source_threshold": thresholds[arm], "delta_auc_vs_frozen": metric["auroc"] - metrics["frozen"]["auroc"],
                   "delta_auc_vs_erm": metric["auroc"] - metrics["erm"]["auroc"],
                   "delta_auc_vs_own_off": metric["auroc"] - metrics["residual_off"]["auroc"] if arm == "residual_on" else "",
                   "eer_gain_pp_vs_frozen": 100 * (metrics["frozen"]["eer"] - metric["eer"]),
                   "eer_gain_pp_vs_erm": 100 * (metrics["erm"]["eer"] - metric["eer"]),
                   "eer_gain_pp_vs_own_off": 100 * (metrics["residual_off"]["eer"] - metric["eer"]) if arm == "residual_on" else "",
                   "relative_eer_drop_vs_frozen": (metrics["frozen"]["eer"] - metric["eer"]) / metrics["frozen"]["eer"],
                   "adapt_seconds": info["adapt_seconds"] if arm == "residual_on" else 0,
                   "inference_seconds": info["inference_seconds"], "peak_gpu_bytes": info["peak_gpu_bytes"],
                   "updates": info["updates"] if arm == "residual_on" else 0,
                   "numerical_failures": info["numerical_failures"]}
            if arm == "residual_on":
                for ref in ("frozen", "erm", "residual_off"):
                    ci = _pair_ci(scores[arm], scores[ref], y, groups, config["bootstrap_replicates"])
                    row[f"auc_gain_ci_vs_{ref}"] = json.dumps(ci[0])
                    row[f"eer_gain_ci_vs_{ref}"] = json.dumps(ci[1])
            all_metrics.append(row)
        print(json.dumps({"domain": domain, "scores_saved": len(ids), "metrics": metrics,
                          "adapt": info}), flush=True)
    columns = list(dict.fromkeys(key for row in all_metrics for key in row))
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(all_metrics)
    print(json.dumps({"status": "PASS", "run": str(output), "selected_lr": lr}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    run(args.config, args.run_id, args.seed)

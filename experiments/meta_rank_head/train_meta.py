"""Source-only meta-training of an unlabeled linear-head update rule."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.evaluation.metrics import binary_metrics
from model import AuxiliaryLoss, adapt_head, outer_loss


def read_jsonl(path):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate manifest ID")
    return {row["sample_id"]: row for row in rows}


def read_attack_protocol(path):
    result = {}
    with open(path) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5 or fields[1] in result or fields[4] not in ("bonafide", "spoof"):
                raise ValueError("invalid source attack protocol")
            result[fields[1]] = (fields[3], int(fields[4] == "spoof"))
    return result


def load_source(root, cache_path, role, protocol_path, bundle):
    base = root / "data/manifests_v2/asv2019_la"
    manifest = read_jsonl(base / "inference" / (role + ".jsonl"))
    labels = read_jsonl(base / "labels" / (role + ".jsonl"))
    groups = read_jsonl(base / "groups" / (role + ".jsonl"))
    if set(manifest) != set(labels) or set(manifest) != set(groups):
        raise ValueError("source manifest role coverage differs")
    protocol = read_attack_protocol(protocol_path)
    cache = FeatureCache(cache_path)
    identity = cache.index["identity"]
    if (identity["checkpoint_ref"] != bundle["checkpoint_ref"] or
            identity["source_run_id"] != bundle["source_run_id"] or
            identity["split_role"] != role or cache.index["feature_dim"] != 160 or
            cache.index["num_views"] != 3 or Path(identity["manifest_ref"]).resolve() !=
            (base / "inference" / (role + ".jsonl")).resolve()):
        raise ValueError("source feature provenance mismatch")
    cache.verify_expected_ids(manifest)
    features = cache.load_by_id()
    ids = sorted(manifest)
    y, attacks, group_ids = [], [], []
    for sample_id in ids:
        audio_id = Path(manifest[sample_id]["audio_relpath"]).stem
        if audio_id not in protocol:
            raise ValueError("source audio missing attack protocol")
        attack, label = protocol[audio_id]
        if label != labels[sample_id]["canonical_label"]:
            raise ValueError("source protocol label mismatch")
        y.append(label)
        attacks.append(attack)
        group_ids.append(groups[sample_id]["source_group_id"])
    x = np.stack([features[sample_id] for sample_id in ids])
    if x.shape != (len(ids), 3, 160) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("source feature shape/dtype/value mismatch")
    return {"ids": ids, "x": torch.from_numpy(x), "y": np.asarray(y, dtype=np.int64),
            "attack": attacks, "groups": group_ids, "cache_id": cache.cache_id}


def pools(data):
    labels = data["y"]
    bona = np.flatnonzero(labels == 0).tolist()
    result = {}
    for attack in sorted(set(data["attack"])):
        if attack == "-":
            continue
        spoof = [i for i, code in enumerate(data["attack"]) if code == attack and labels[i] == 1]
        if len(bona) >= 32 and len(spoof) >= 32:
            result[attack] = (bona, spoof)
    if not result:
        raise ValueError("no two-class attack episode pool")
    return result


def episode(data, attack_pools, rng, per_class):
    attack = rng.choice(sorted(attack_pools))
    condition = rng.randrange(3)
    bona, spoof = attack_pools[attack]
    chosen_b = rng.sample(bona, 2 * per_class)
    chosen_s = rng.sample(spoof, 2 * per_class)
    support_idx = chosen_b[:per_class] + chosen_s[:per_class]
    query_idx = chosen_b[per_class:] + chosen_s[per_class:]
    if set(data["ids"][i] for i in support_idx) & set(data["ids"][i] for i in query_idx):
        raise AssertionError("support/query original audio overlap")
    rng.shuffle(support_idx)
    rng.shuffle(query_idx)
    # Rotate the same existing augmentation condition for both classes.
    order = [condition] + [j for j in range(3) if j != condition]
    return (data["x"][support_idx][:, order], data["x"][query_idx][:, order],
            torch.as_tensor(data["y"][query_idx], dtype=torch.long), attack, condition)


def source_head(bundle_path):
    bundle, _, _, _ = verify_frozen_export(bundle_path)
    state = torch.load(bundle_path.parent / bundle["head_ref"], map_location="cpu", weights_only=True)
    if state["score_direction"] != "larger_is_spoof" or state["w"].shape != (160,):
        raise ValueError("incompatible source head")
    return bundle, state["w"].float(), state["b"].float()


def validation(data, attack_pools, aux, weight, bias, config, seed, arm):
    rng = random.Random(seed)
    all_scores, all_labels = [], []
    for _ in range(config["validation_episodes"]):
        support, query, labels, _, _ = episode(data, attack_pools, rng, config["per_class"])
        if arm.startswith("meta"):
            adapted_w, adapted_b = adapt_head(aux, support, weight, bias,
                steps=config["inner_steps"], lr=config["inner_lr"])
        else:
            adapted_w, adapted_b = weight, bias
        scores = (query[:, 0] @ adapted_w + adapted_b).detach().tolist()
        all_scores.extend(scores)
        all_labels.extend(labels.tolist())
    return binary_metrics(all_scores, all_labels, 0.0)


def run(config_path, run_id, seed):
    config = json.loads(Path(config_path).read_text())
    root = Path(config["project_root"])
    output = Path(__file__).parent / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({**config, "run_id": run_id, "seed": seed}, indent=2))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    bundle, source_w, source_b = source_head(Path(config["bundle_ref"]))
    fit = load_source(root, Path(config["fit_cache"]), "fit", Path(config["fit_protocol"]), bundle)
    val = load_source(root, Path(config["select_cache"]), "select", Path(config["select_protocol"]), bundle)
    fit_pools, val_pools = pools(fit), pools(val)
    if set(fit["ids"]) & set(val["ids"]):
        raise ValueError("fit/select audio overlap")
    rng = random.Random(seed)
    # One immutable episode schedule gives the two meta arms identical data/budget.
    schedule = [episode(fit, fit_pools, rng, config["per_class"])
                for _ in range(config["epochs"] * config["episodes_per_epoch"])]
    initial = AuxiliaryLoss(hidden=config["hidden"])
    base_state = {key: value.clone() for key, value in initial.state_dict().items()}
    curves = []
    best = {}
    for arm in ("erm", "meta_bce", "meta_rank"):
        aux = AuxiliaryLoss(hidden=config["hidden"])
        aux.load_state_dict(base_state)
        weight = source_w.detach().clone().requires_grad_(arm == "erm")
        bias = source_b.detach().clone().requires_grad_(arm == "erm")
        optimizer = torch.optim.Adam((weight, bias) if arm == "erm" else aux.parameters(),
                                      lr=config["outer_lr"])
        best_key = (float("inf"), float("inf"))
        for epoch in range(config["epochs"]):
            losses, grad_norms = [], []
            for support, query, labels, _, _ in schedule[
                    epoch * config["episodes_per_epoch"]:(epoch + 1) * config["episodes_per_epoch"]]:
                optimizer.zero_grad()
                if arm == "erm":
                    logits = query[:, 0] @ weight + bias
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())
                else:
                    adapted_w, adapted_b = adapt_head(aux, support, source_w, source_b,
                        steps=config["inner_steps"], lr=config["inner_lr"], create_graph=True)
                    logits = query[:, 0] @ adapted_w + adapted_b
                    loss = outer_loss(logits, labels, config["rank_weight"] if arm == "meta_rank" else 0.0)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                params = (weight, bias) if arm == "erm" else tuple(aux.parameters())
                norm = sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None) ** 0.5
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("meta/ERM gradient is zero or nonfinite")
                optimizer.step()
                losses.append(float(loss.detach()))
                grad_norms.append(norm)
            metrics = validation(val, val_pools, aux, weight.detach(), bias.detach(),
                                 config, seed + 1000, arm)
            record = {"arm": arm, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                      "gradient_norm": float(np.mean(grad_norms)),
                      "source_val_auc": metrics["auroc"], "source_val_eer": metrics["eer"]}
            curves.append(record)
            key = (metrics["eer"], -metrics["auroc"])
            if key < best_key:
                best_key = key
                best[arm] = record.copy()
                torch.save({"arm": arm, "epoch": epoch + 1, "seed": seed,
                            "weight": weight.detach(), "bias": bias.detach(),
                            "aux": aux.state_dict(), "config": config}, output / (arm + ".pt"))
            print(json.dumps(record), flush=True)
    with (output / "training_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    (output / "selection.json").write_text(json.dumps({"source_role": "select", "best": best,
        "fit_cache_id": fit["cache_id"], "source_val_cache_id": val["cache_id"],
        "support_query_overlap": 0, "meta_gradient": "nonzero_finite"}, indent=2))
    print(json.dumps({"status": "PASS", "run": str(output), "best": best}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    run(args.config, args.run_id, args.seed)

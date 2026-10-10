"""Source-only training of a domain-conditioned head and matched ERM head."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eptta.cache.reader import FeatureCache
from eptta.evaluation.metrics import binary_metrics
from eptta.models.frozen import verify_frozen_export
from model import DomainHeadNet, target_head


def records(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate source ID")
    return result


def protocol(path):
    table = {}
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 5 or fields[1] in table or fields[4] not in ("bonafide", "spoof"):
                raise ValueError("invalid official LA CM protocol")
            attack = fields[3]
            label = int(fields[4] == "spoof")
            if (label == 0 and attack != "-") or (label == 1 and attack not in
                    {"A01", "A02", "A03", "A04", "A05", "A06"}):
                raise ValueError("unexpected LA attack/label semantics")
            table[fields[1]] = (attack, label)
    return table


def source_head(bundle_path):
    bundle, _, _, _ = verify_frozen_export(bundle_path)
    state = torch.load(bundle_path.parent / bundle["head_ref"], map_location="cpu", weights_only=True)
    if (state["score_direction"] != "larger_is_spoof" or
            state["w"].shape != (160,) or state["b"].ndim != 0):
        raise ValueError("incompatible source head")
    return bundle, state["w"].float(), state["b"].float()


def load_source(config, role, bundle):
    root = Path(config["project_root"])
    base = root / "data/manifests_v2/asv2019_la"
    manifest_path = base / "inference" / (role + ".jsonl")
    manifest = records(manifest_path)
    labels = records(base / "labels" / (role + ".jsonl"))
    groups = records(base / "groups" / (role + ".jsonl"))
    if set(manifest) != set(labels) or set(manifest) != set(groups):
        raise ValueError("source role mismatch")
    attacks = protocol(config["fit_protocol"] if role == "fit" else config["select_protocol"])
    cache = FeatureCache(config["fit_cache"] if role == "fit" else config["select_cache"])
    ident = cache.index["identity"]
    if (ident["checkpoint_ref"] != bundle["checkpoint_ref"] or
            ident["source_run_id"] != bundle["source_run_id"] or
            ident["split_role"] != role or
            Path(ident["manifest_ref"]).resolve() != manifest_path.resolve() or
            cache.index["num_views"] != 3 or cache.index["feature_dim"] != 160):
        raise ValueError("source cache provenance mismatch")
    cache.verify_expected_ids(manifest)
    features = cache.load_by_id()
    ids = sorted(manifest)
    y, attack_codes, audio_ids = [], [], []
    for sample_id in ids:
        audio_id = Path(manifest[sample_id]["audio_relpath"]).stem
        if audio_id not in attacks or not groups[sample_id]["source_group_id"]:
            raise ValueError("missing source attack/group")
        attack, label = attacks[audio_id]
        if label != labels[sample_id]["canonical_label"]:
            raise ValueError("source label/protocol mismatch")
        y.append(label)
        attack_codes.append(attack)
        audio_ids.append(audio_id)
    if len(set(audio_ids)) != len(audio_ids):
        raise ValueError("duplicate original audio ID")
    x = np.stack([features[sid] for sid in ids])
    if x.shape != (len(ids), 3, 160) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("source feature shape/dtype/value mismatch")
    return {"ids": ids, "audio_ids": audio_ids, "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32), "attacks": attack_codes,
            "cache_id": cache.cache_id}


def episode_pools(data):
    bona = [i for i, label in enumerate(data["y"]) if label == 0]
    spoof = defaultdict(list)
    for i, attack in enumerate(data["attacks"]):
        if attack != "-":
            spoof[attack].append(i)
    if sorted(spoof) != ["A01", "A02", "A03", "A04", "A05", "A06"]:
        raise ValueError("expected six audited source attack families")
    if len(bona) < 64 or any(len(indices) < 64 for indices in spoof.values()):
        raise ValueError("insufficient two-class pseudo-domain data")
    return bona, spoof


def make_episode(data, pools, rng):
    bona, spoof = pools
    attack = rng.choice(sorted(spoof))
    condition = rng.randrange(3)
    b = rng.sample(bona, 64)
    s = rng.sample(spoof[attack], 64)
    support = b[:32] + s[:32]
    query = b[32:] + s[32:]
    if set(data["audio_ids"][i] for i in support) & set(data["audio_ids"][i] for i in query):
        raise AssertionError("support/query original audio overlap")
    rng.shuffle(support)
    rng.shuffle(query)
    return support, query, attack, condition


def source_validation(data, net, weight, bias, arm):
    z = data["x"][:, 0]
    if arm == "dch":
        weight, bias, _, _ = target_head(net, z, weight, bias)
    scores = (z @ weight + bias).detach().cpu().tolist()
    return binary_metrics(scores, [int(y) for y in data["y"].tolist()], 0.0)


def run(config_path, run_id, seed):
    config = json.loads(Path(config_path).read_text())
    output = Path(__file__).parent / "results" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps({**config, "seed": seed, "run_id": run_id}, indent=2))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, source_w, source_b = source_head(Path(config["bundle_ref"]))
    fit = load_source(config, "fit", bundle)
    val = load_source(config, "select", bundle)
    if set(fit["audio_ids"]) & set(val["audio_ids"]):
        raise ValueError("fit/select original audio overlap")
    pools = episode_pools(fit)
    rng = random.Random(seed)
    schedule = [make_episode(fit, pools, rng) for _ in range(config["total_episodes"])]
    fit_x, fit_y = fit["x"].to(device), fit["y"].to(device)
    val["x"], val["y"] = val["x"].to(device), val["y"].to(device)
    source_w, source_b = source_w.to(device), source_b.to(device)
    curves, selected = [], {}
    for arm in ("erm", "dch"):
        torch.manual_seed(seed)
        net = DomainHeadNet().to(device)
        weight = source_w.detach().clone().requires_grad_(arm == "erm")
        bias = source_b.detach().clone().requires_grad_(arm == "erm")
        optimizer = torch.optim.Adam((weight, bias) if arm == "erm" else net.parameters(),
                                      lr=config["learning_rate"])
        best_key = (float("inf"), float("inf"))
        for epoch in range(config["epochs"]):
            begin = epoch * config["total_episodes"] // config["epochs"]
            end = (epoch + 1) * config["total_episodes"] // config["epochs"]
            losses, bces, regs = [], [], []
            for support_ids, query_ids, _, condition in schedule[begin:end]:
                optimizer.zero_grad()
                support_z = fit_x[support_ids, condition]
                query_z = fit_x[query_ids, condition]
                query_y = fit_y[query_ids]
                if arm == "dch":
                    w, b, dw, db = target_head(net, support_z, source_w, source_b)
                    reg = config["lambda_reg"] * (dw.square().sum() / source_w.square().sum() + db.square())
                else:
                    w, b = weight, bias
                    reg = torch.zeros((), device=device)
                bce = F.binary_cross_entropy_with_logits(query_z @ w + b, query_y)
                loss = bce + reg
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                bces.append(float(bce.detach()))
                regs.append(float(reg.detach()))
            with torch.no_grad():
                metric = source_validation(val, net, weight.detach(), bias.detach(), arm)
            record = {"arm": arm, "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                      "train_bce": float(np.mean(bces)), "train_reg": float(np.mean(regs)),
                      "source_select_auc": metric["auroc"], "source_select_eer": metric["eer"]}
            curves.append(record)
            key = (metric["eer"], -metric["auroc"])
            if key < best_key:
                best_key = key
                selected[arm] = record.copy()
                torch.save({"arm": arm, "epoch": epoch + 1, "seed": seed, "config": config,
                            "weight": weight.detach().cpu(), "bias": bias.detach().cpu(),
                            "net": {name: tensor.detach().cpu() for name, tensor in net.state_dict().items()}},
                           output / (arm + ".pt"))
            print(json.dumps(record), flush=True)
    with (output / "training_curve.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    (output / "selection.json").write_text(json.dumps({"source_role": "select", "selected": selected,
        "fit_cache_id": fit["cache_id"], "select_cache_id": val["cache_id"],
        "pseudo_domain_attacks": sorted(pools[1]), "view_conditions": [0, 1, 2],
        "support_query_audio_overlap": 0, "device": str(device)}, indent=2))
    print(json.dumps({"status": "PASS", "run": str(output), "selected": selected}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.run_id, args.seed)

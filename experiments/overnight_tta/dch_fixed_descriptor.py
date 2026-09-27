"""Read-only four-seed ablation of historical DCH target descriptors."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "distribution_conditioned_head"))
from experiments.distribution_conditioned_head.model import DomainHeadNet, target_head
from experiments.distribution_conditioned_head.train import load_source, source_head
from eptta.evaluation.metrics import binary_metrics
from adapt_eval import _pair_ci, selected_features, source_threshold


def balanced_source_descriptor(fit):
    """Equal bona/spoof mass, with six source spoof families equally weighted."""
    z = fit["x"][:, 0]
    bona = z[fit["y"] == 0]
    families = sorted(set(fit["attacks"]) - {"-"})
    if families != [f"A{i:02d}" for i in range(1, 7)]:
        raise ValueError("source attack families differ")
    spoof_groups = [z[[i for i, attack in enumerate(fit["attacks"]) if attack == a]] for a in families]
    mean = (bona.mean(0) + torch.stack([g.mean(0) for g in spoof_groups]).mean(0)) / 2
    second = (bona.square().mean(0) +
              torch.stack([g.square().mean(0) for g in spoof_groups]).mean(0)) / 2
    return torch.cat((mean, (second - mean.square()).clamp_min(0).sqrt()))


def run(config_ref, output_ref):
    config = json.loads(Path(config_ref).read_text())
    out = Path(output_ref)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(2)
    bundle, w, b = source_head(Path(config["bundle_ref"]))
    fit = load_source(config, "fit", bundle)
    select = load_source(config, "select", bundle)
    descriptor = balanced_source_descriptor(fit)
    if descriptor.shape != (320,) or not torch.isfinite(descriptor).all():
        raise ValueError("bad source descriptor")
    target_data = {}
    for domain in ("itw_target10", "wavefake_dev"):
        selection = json.loads(Path(config[f"{domain}_select"]).read_text())
        if any("label" in row for row in selection["records"]):
            raise ValueError("target selection contains labels")
        ids = [r["sample_id"] for r in selection["records"]]
        views, cache_id = selected_features(config[f"{domain}_cache"], ids, bundle)
        target_data[domain] = (selection, ids, views, cache_id)
    rows = []
    for seed in (13, 29, 47, 71):
        checkpoint = Path(config["historical_dch_root"]) / f"dch_20260927_seed{seed}" / "dch.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net = DomainHeadNet()
        net.load_state_dict(state["net"])
        net.eval()
        with torch.inference_mode():
            correction = net.net(descriptor)
            fixed_w, fixed_b = w + correction[:160], b + correction[160]
            source_fixed = (select["x"][:, 0] @ fixed_w + fixed_b).numpy()
            source_y = select["y"].int().tolist()
            threshold = source_threshold(source_fixed.tolist(), source_y)
        for domain, (selection, ids, views, cache_id) in target_data.items():
            with torch.inference_mode():
                z = views[:, 0]
                target_w, target_b, _, _ = target_head(net, z, w, b)
                scores = {"source_fixed": (z @ fixed_w + fixed_b).numpy(),
                          "target_descriptor": (z @ target_w + target_b).numpy()}
            if not all(np.isfinite(s).all() for s in scores.values()):
                raise ValueError("nonfinite DCH ablation scores")
            score_ref = out / f"{domain}_seed{seed}_scores.csv"
            with score_ref.open("x", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(["sample_id", "source_fixed", "target_descriptor"])
                writer.writerows((sid, float(scores["source_fixed"][i]),
                                  float(scores["target_descriptor"][i])) for i, sid in enumerate(ids))
            # Score file is durable before any development label is opened.
            if domain == "itw_target10":
                audit = json.loads(Path(config["itw_audit"]).read_text())
                lab = {r["sample_id"]: r["label"] for r in audit["records"]}
                if set(lab) != set(ids):
                    raise ValueError("ITW target10 label coverage mismatch")
                labels, groups = [lab[sid] for sid in ids], ids
            else:
                label_map = {"R": 0, **{f"WF{i}": 1 for i in range(1, 8)}}
                labels = [label_map[sid.rsplit(":", 1)[1]] for sid in ids]
                groups = [r["audio_id"] for r in selection["records"]]
                if len(set(groups)) != selection["content_groups"]:
                    raise ValueError("WaveFake content-pair count mismatch")
            fixed = binary_metrics(scores["source_fixed"].tolist(), labels, threshold)
            target = binary_metrics(scores["target_descriptor"].tolist(), labels, 0)
            ci_auc, ci_eer = _pair_ci(scores["target_descriptor"], scores["source_fixed"],
                                    labels, groups, config["bootstrap_replicates"])
            rows.append({"domain": domain, "seed": seed, "count": len(ids), "cache_id": cache_id,
                         "source_fixed_auc": fixed["auroc"], "source_fixed_eer": fixed["eer"],
                         "source_fixed_threshold": threshold, "source_fixed_fpr": fixed["fpr"],
                         "source_fixed_fnr": fixed["fnr"], "target_descriptor_auc": target["auroc"],
                         "target_descriptor_eer": target["eer"],
                         "target_minus_source_fixed_auc": target["auroc"] - fixed["auroc"],
                         "source_fixed_minus_target_eer_pp": 100 * (fixed["eer"] - target["eer"]),
                         "auc_gain_ci": json.dumps(ci_auc), "eer_gain_ci": json.dumps(ci_eer),
                         "historical_checkpoint": str(checkpoint), "score_file": str(score_ref)})
    with (out / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "PASS", "output": str(out), "rows": rows}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)

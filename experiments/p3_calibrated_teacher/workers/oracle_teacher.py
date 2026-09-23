#!/usr/bin/env python
"""POST_HOC_ORACLE_ONLY / NOT_DEPLOYABLE P3 diagnostic."""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "scripts"))

import torch

from eptta.adaptation.calibrated_selective import _loss_components, _total, posterior_spoof
from eptta.adaptation.math import apply_adapter, project_frobenius_
from p3_common import (ep_config, load_context, load_ids, make_target, p3_params,
                       resolve_input)

torch.set_num_threads(1)


def load_posthoc_labels(config):
    doc = json.loads(resolve_input(config["inputs"]["target10_label_manifest"]).read_text(
        encoding="utf-8"))
    return {row["sample_id"]: int(row["label"]) for row in doc["records"]}


def frozen(sample_id, before, teacher, posterior, confidence, agreement, reason=None,
           safety=False, flips=0):
    return {
        "variant": "OracleTeacher", "method_id": "POST_HOC_ORACLE_ONLY",
        "sample_id": sample_id, "score_before": before, "score_after": before,
        "delta_score": 0.0, "adaptation_applied": False,
        "abstain_reason": reason, "teacher_label": teacher,
        "teacher_p_spoof": posterior, "gate_confidence": confidence,
        "gate_agreement": agreement, "safety_rejected": safety,
        "source_anchor_flip_count": flips, "final_source_anchor_flip_count": 0,
        "numeric_fallback": False, "POST_HOC_ORACLE_ONLY": True,
        "NOT_DEPLOYABLE": True,
    }


def run_one(target, resources, cfg, params, label):
    before = float(target.features[0] @ resources.w + resources.b)
    posterior = posterior_spoof(before, params)
    confidence = max(posterior, 1.0 - posterior)
    view_labels = target.features @ resources.w + resources.b >= params["tau_hat"]
    calibrated_teacher = int(posterior >= 0.5)
    agreement = float((view_labels == bool(calibrated_teacher)).to(target.features.dtype).mean())
    if confidence < params["confidence_threshold"]:
        return frozen(target.sample_id, before, label, posterior, confidence, agreement,
                      "low_teacher_confidence")
    if agreement < params["min_agreement"]:
        return frozen(target.sample_id, before, label, posterior, confidence, agreement,
                      "view_disagreement")
    rank = resources.U.shape[1]
    parameter = torch.zeros((rank, rank), dtype=target.features.dtype,
                            device=target.features.device, requires_grad=True)
    try:
        with torch.enable_grad():
            for _ in range(cfg.steps):
                loss = _total(_loss_components(parameter, target, resources, label, params), params)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite oracle objective")
                gradient, = torch.autograd.grad(loss, parameter)
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError("non-finite oracle gradient")
                with torch.no_grad():
                    parameter.add_(gradient, alpha=-cfg.lr)
                    project_frobenius_(parameter, cfg.rho)
        attempted = parameter.detach().clone()
        source_classes = ((apply_adapter(resources.anchors_z, resources.U, attempted)
                           @ resources.w + resources.b - resources.tau0) >= 0).to(torch.long)
        flips = int((source_classes != resources.anchors_y.to(torch.long)).sum())
        if flips:
            return frozen(target.sample_id, before, label, posterior, confidence, agreement,
                          safety=True, flips=flips)
        score = float((apply_adapter(target.features[:1], resources.U, attempted)
                       @ resources.w + resources.b)[0])
        result = frozen(target.sample_id, before, label, posterior, confidence, agreement)
        result.update(score_after=score, delta_score=score - before,
                      adaptation_applied=True, source_anchor_flip_count=0)
        return result
    except FloatingPointError:
        result = frozen(target.sample_id, before, label, posterior, confidence, agreement)
        result["numeric_fallback"] = True
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("refusing to overwrite: %s" % out)
    config, resources, _meta, cache, features = load_context()
    labels = load_posthoc_labels(config)
    ids = load_ids("evalU", args.manifest_dir)
    if args.limit is not None:
        ids = ids[:args.limit]
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("status") != "OK":
        raise SystemExit("OracleTeacher NOT_RUN: calibration fit failed")
    params = p3_params(config, calibration, True)
    cfg = ep_config(config)
    with out.open("w", encoding="utf-8") as stream:
        for sid in ids:
            record = run_one(make_target(sid, features, cache.cache_id), resources, cfg,
                             params, labels[sid])
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    print("OracleTeacher: wrote %d POST_HOC_ORACLE_ONLY records to %s" % (len(ids), out))


if __name__ == "__main__":
    main()

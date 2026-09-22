#!/usr/bin/env python
"""Load the frozen SSL-AASIST checkpoint and inventory adaptation candidates."""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers" / "compat"))

import torch

from author_training import build_author_model
from eptta.baselines.ports.audio_native import (SCOPE_A, SCOPE_B, audit_candidates,
                                                 preregistered_parameter_names)
from eptta.models.frozen import verify_frozen_export


def default_asset_root():
    explicit = os.environ.get("TTA_ASSET_ROOT")
    if explicit:
        return Path(explicit).resolve()
    if (ROOT / "outputs_v2/ssl_aasist/frozen/bundle.json").is_file():
        return ROOT
    sibling = ROOT.parent / "TTA"
    return sibling.resolve()


def load_model(asset_root, device):
    bundle_path = asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle, _manifest, _parity, _selection = verify_frozen_export(bundle_path)
    state = torch.load(bundle_path.parent / bundle["detector_state_ref"], map_location="cpu")
    construction = {
        "source_job": {"model_id": bundle["model_id"],
                       "initialization": bundle.get("initialization")},
        "execution": {"architecture": state["architecture"]},
    }
    adapter, patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    return adapter.model, bundle, patch, bundle_path


def build_audit(model, bundle, patch, bundle_path):
    candidates = audit_candidates(model)
    counts = collections.Counter(row["module_type"] for row in candidates
                                 if row["module_type"] != "StandaloneParameter")
    components = collections.Counter()
    for name, parameter in model.named_parameters():
        if name.startswith("ssl_model."):
            components["frontend"] += parameter.numel()
        elif name.startswith("out_layer."):
            components["head"] += parameter.numel()
        else:
            components["backend"] += parameter.numel()
    return {
        "schema_version": "0.1.0",
        "model_id": bundle["model_id"],
        "baseline_id": bundle["baseline_id"],
        "frozen_bundle_ref": str(bundle_path),
        "task_weight_origin": bundle.get("task_weight_origin"),
        "generic_frontend_initialization": bundle.get("initialization"),
        "load_patch": patch,
        "module_type_counts": {
            "BatchNorm1d": counts["BatchNorm1d"],
            "BatchNorm2d": counts["BatchNorm2d"],
            "LayerNorm": counts["LayerNorm"],
            "GroupNorm": counts["GroupNorm"],
            "other_affine_modulation": sum(
                row["module_type"] == "StandaloneParameter" for row in candidates),
        },
        "parameter_counts": dict(components),
        "scopes": {
            SCOPE_A: preregistered_parameter_names(model, SCOPE_A),
            SCOPE_B: preregistered_parameter_names(model, SCOPE_B),
        },
        "candidates": candidates,
    }


def write_outputs(audit, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_parameter_audit.json"
    md_path = output_dir / "model_parameter_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# SSL-AASIST model parameter audit", "",
        "- Model: `%s`" % audit["model_id"],
        "- Baseline: `%s`" % audit["baseline_id"],
        "- Task weights: `%s`" % audit["task_weight_origin"],
        "- Parameter counts: `%s`" % audit["parameter_counts"], "",
        "## Normalization inventory", "",
    ]
    for key, value in audit["module_type_counts"].items():
        lines.append("- %s: %d" % (key, value))
    lines.extend(["", "## Pre-registered scopes", ""])
    for scope, names in audit["scopes"].items():
        lines.append("### `%s` (%d parameters, %d tensors)" % (
            scope,
            audit["scope_parameter_counts"][scope],
            len(names)))
        lines.extend("- `%s`" % name for name in names)
        lines.append("")
    lines.extend(["## Candidate modules", "",
                  "| module | type | component | origin | parameter count | parameters |",
                  "|---|---|---|---|---:|---|"])
    for row in audit["candidates"]:
        lines.append("| `%s` | %s | %s | %s | %d | %s |" % (
            row["module_name"], row["module_type"], row["component"],
            row["weight_origin"], row["parameter_count"],
            ", ".join("`%s`" % name for name in row["parameter_names"])))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=default_asset_root())
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments/audio_native_tta/audits")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle, patch, bundle_path = load_model(args.asset_root, device)
    audit = build_audit(model, bundle, patch, bundle_path)
    # Add exact selected scalar counts after model construction.
    params = dict(model.named_parameters())
    audit["scope_parameter_counts"] = {
        scope: sum(params[name].numel() for name in names)
        for scope, names in audit["scopes"].items()
    }
    write_outputs(audit, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "module_type_counts": audit["module_type_counts"],
                      "scope_parameter_counts": audit["scope_parameter_counts"]}, indent=2))


if __name__ == "__main__":
    main()

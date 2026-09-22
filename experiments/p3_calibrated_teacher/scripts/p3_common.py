"""Shared label-free P3 worker plumbing."""
import json
from pathlib import Path

import torch

from eptta.adaptation.types import EPConfig, TargetViews
from eptta.cache.reader import FeatureCache
from eptta.models.frozen import verify_frozen_export
from eptta.offline.artifacts import load_frozen_resources


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments/p3_calibrated_teacher"
CONFIG = EXP_DIR / "configs/main.json"
FORBIDDEN = frozenset({"label", "canonical_label", "original_label", "target", "class", "y"})


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def resolve_input(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_ids(role):
    path = EXP_DIR / "manifests" / ("target10_%s.json" % role)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("role") != role or doc.get("protocol") != "unlabeled_domain_calibration_then_episodic_tta":
        raise RuntimeError("invalid P3 %s manifest protocol" % role)
    records = doc.get("records")
    if not isinstance(records, list) or len(records) != doc.get("count"):
        raise RuntimeError("P3 %s manifest count mismatch" % role)
    ids = []
    for row in records:
        for key in row:
            if key.lower() in FORBIDDEN or "label" in key.lower():
                raise RuntimeError("label field forbidden in P3 worker manifest")
        if row.get("split_role") != role:
            raise RuntimeError("P3 split role mismatch")
        ids.append(row["sample_id"])
    if len(ids) != len(set(ids)):
        raise RuntimeError("P3 IDs must be unique")
    return ids


def load_context():
    cfg = load_config()
    inputs = cfg["inputs"]
    bundle, _manifest, _parity, _selection = verify_frozen_export(resolve_input(inputs["frozen_bundle"]))
    resources, _extras, meta = load_frozen_resources(resolve_input(inputs["resources"]), bundle)
    cache = FeatureCache(resolve_input(inputs["target_cache"]))
    features = cache.load_by_id()
    return cfg, resources, meta, cache, features


def make_target(sample_id, features, cache_id):
    return TargetViews(sample_id, torch.from_numpy(features[sample_id]), cache_id)


def ep_config(config):
    p = config["objective"]
    return EPConfig(steps=p["steps"], lr=p["lr"], rho=p["rho"], gamma=p["gamma"],
                    lambda_keep=p["lambda_keep"])


def p1_params(config):
    p = config["objective"]
    return {
        "lambda_pseudo": p["lambda_pseudo"],
        "lambda_consistency": p["lambda_consistency"],
        "lambda_source": p["lambda_source"], "lambda_r": p["lambda_r"],
        "temperature": p["temperature"], "confidence_margin": 0.5,
        "min_agreement": 1.0,
    }


def p3_params(config, calibration, selection_enabled):
    p = config["objective"]
    return {
        **{key: calibration[key] for key in (
            "mu_bona", "mu_spoof", "var_bona", "var_spoof",
            "pi_bona", "pi_spoof", "tau_hat")},
        "lambda_pseudo": p["lambda_pseudo"],
        "lambda_consistency": p["lambda_consistency"],
        "lambda_source": p["lambda_source"], "lambda_r": p["lambda_r"],
        "temperature": p["temperature"],
        "confidence_threshold": config["gate"]["confidence_threshold"],
        "min_agreement": config["gate"]["min_view_agreement"],
        "selection_enabled": selection_enabled,
    }


def frozen_score(sample_id, features, resources):
    return float(torch.from_numpy(features[sample_id][0]) @ resources.w + resources.b)

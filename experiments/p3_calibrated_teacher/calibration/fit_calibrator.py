#!/usr/bin/env python
"""Fit the P3 label-free GMM from calU frozen original-view scores."""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from calibration.mixture import CalibrationFitError, fit_gmm
from p3_common import frozen_score, load_context, load_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("refusing to overwrite calibration result: %s" % out)
    config, resources, _meta, _cache, features = load_context()
    ids = load_ids("calU", args.manifest_dir)
    scores = [frozen_score(sid, features, resources) for sid in ids]
    gmm_cfg = config["gmm"]
    try:
        model = fit_gmm(scores, max_iter=gmm_cfg["max_iter"], tol=gmm_cfg["tol"],
                        variance_floor=gmm_cfg["variance_floor"])
        result = {
            "status": "OK", "protocol": config["protocol"],
            "fit_input": "calU_frozen_original_view_scores", "label_free": True,
            "calU_count": len(ids), **model.as_dict(),
        }
    except CalibrationFitError as exc:
        result = {
            "status": "CALIBRATION_FIT_FAIL", "protocol": config["protocol"],
            "fit_input": "calU_frozen_original_view_scores", "label_free": True,
            "calU_count": len(ids), "converged": False, "error": str(exc),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the P1 confirmatory config directory (locked task-aware params).

Called only after the pilot mechanism gate passes.  Configs copy the existing
ssl_aasist target/control structure but set method_id=ep_tta_taskaware_v1,
run_name=taskaware_v1, locked EP params and fresh output roots.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from eptta.config.schema import read_document

SOURCES = {
    "ssl_itw_taskaware.yaml": ("configs/local_v2/ssl_aasist_target/ssl_itw_ep.yaml",
                               "outputs_v2/p1_taskaware_v1/in_the_wild"),
    "ssl_asv2021_la_taskaware.yaml": ("configs/local_v2/ssl_aasist_target/ssl_asv2021_la_ep.yaml",
                                      "outputs_v2/p1_taskaware_v1/asv2021_la"),
    "ssl_asv2021_df_taskaware.yaml": ("configs/local_v2/ssl_aasist_target/ssl_asv2021_df_ep.yaml",
                                      "outputs_v2/p1_taskaware_v1/asv2021_df"),
    "ssl_control_taskaware.yaml": ("configs/local_v2/ssl_aasist_control_main/ep_tta.yaml",
                                   "outputs_v2/p1_taskaware_v1/control"),
}

LOCKED_CONFIG = {"steps": 3, "lr": 0.03, "rho": 0.05, "gamma": 0.1, "lambda_keep": 1.0}
LOCKED_PARAMS = {"lambda_pseudo": 1.0, "lambda_consistency": 0.25, "lambda_source": 1.0,
                 "lambda_r": 0.01, "temperature": 1.0, "confidence_margin": 0.5,
                 "min_agreement": 1.0}


def main():
    out_dir = ROOT / "configs/local_v2/ssl_aasist_target_taskaware"
    if out_dir.exists():
        raise SystemExit("refusing to overwrite existing config dir: %s" % out_dir)
    out_dir.mkdir(parents=True)
    for name, (source_ref, output_root) in SOURCES.items():
        cfg = read_document(ROOT / source_ref)
        cfg["run_name"] = "taskaware_v1"
        cfg["output_root"] = output_root
        cfg["method"] = {"method_id": "ep_tta_taskaware_v1",
                         "config": dict(LOCKED_CONFIG), "params": dict(LOCKED_PARAMS)}
        (out_dir / name).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
        print("wrote", out_dir / name)


if __name__ == "__main__":
    main()

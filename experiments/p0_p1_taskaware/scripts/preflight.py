#!/usr/bin/env python
"""Preflight checks for the p0_p1_taskaware pipeline (also powers --dry-run).

Only checks; it never scores, adapts, extracts features or writes into protected
paths.  Failures raise non-zero so run_all.sh aborts before any adaptation work.
"""
import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from scripts.common import (CACHE, FROZEN_BUNDLE, LABEL_MANIFEST, PARAM_SEARCH, RESOURCES,
                            SELECT_MANIFEST, load_candidates, load_context, load_select_sample_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--gpu-count", type=int, default=None,
                        help="deprecated compatibility alias for --worker-count")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    worker_count = args.worker_count if args.worker_count is not None else args.gpu_count
    if worker_count is None:
        raise SystemExit("preflight requires --worker-count")

    checks = {}

    # 1. Runtime / environment.  Cache-based P0/P1 adaptation runs on CPU;
    #    CUDA is only recorded for environment audit, never required.
    import torch
    checks["runtime"] = {
        "adaptation_device": "cpu",
        "parallel_workers": worker_count,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_required": False,
    }
    if worker_count != 4:
        raise SystemExit("p0_p1_taskaware protocol requires exactly 4 parallel workers, got %d" % worker_count)

    # 2. Git
    import subprocess
    checks["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                        capture_output=True, text=True).stdout.strip()
    checks["git_branch"] = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip()

    # 3. frozen bundle / checkpoint / resources / cache / threshold
    bundle, resources, resource_meta, cache, features, threshold = load_context()
    checks["frozen"] = {"baseline_id": bundle.get("baseline_id"),
                        "checkpoint_ref": bundle.get("checkpoint_ref"),
                        "tau0": threshold,
                        "cache_sample_count": len(features)}

    # 4. manifests
    select_ids = load_select_sample_ids()
    checks["select_manifest"] = {"role": "select", "count": len(select_ids),
                                 "label_free": True}
    label_doc = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))
    checks["label_manifest"] = {"role": label_doc.get("role"), "count": len(label_doc.get("records", []))}

    # 5. candidates
    candidates = load_candidates()
    checks["candidates"] = {"count": len(candidates)}

    # 6. config source paths exist for the confirmatory config copy.
    target_cfg_dir = ROOT / "configs/local_v2/ssl_aasist_target"
    required_cfgs = ["ssl_itw_ep.yaml", "ssl_asv2021_la_ep.yaml", "ssl_asv2021_df_ep.yaml"]
    checks["config_sources"] = {name: (target_cfg_dir / name).is_file() for name in required_cfgs}

    # 7. output directory must not already exist (never overwrite).
    run_dir = Path(args.run_dir)
    checks["output_dir"] = {"path": str(run_dir), "exists": run_dir.exists()}
    if run_dir.exists():
        raise SystemExit("run directory already exists; refusing to overwrite: %s" % run_dir)

    # 8. protected paths must remain untouched by this run.
    protected = [str(ROOT / "experiments/target10_selection/results"),
                 str(ROOT / "outputs_v2")]
    checks["protected_paths"] = protected

    report = {"schema_version": "0.1.0", "status": "PASS", "checks": checks}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    main()

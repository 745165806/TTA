"""One interpreter, bounded CPU workers, mandatory engineering smoke, then diagnosis."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from common import HERE, ROOT, RESULTS, candidates, config, partition, sample_ids, write_json


def sweep(output, asset_root, groups, limit=None):
    (output / "scores").mkdir(parents=True, exist_ok=False)
    processes, streams = [], []
    try:
        for group in range(groups):
            stream = (output / f"worker_{group}.log").open("x", encoding="utf-8")
            streams.append(stream)
            cmd = [sys.executable, str(HERE / "oracle_worker.py"), "--output", str(output),
                   "--asset-root", str(asset_root), "--group", str(group), "--num-groups", str(groups)]
            if limit:
                cmd += ["--limit", str(limit)]
            processes.append(subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        if any(codes):
            raise RuntimeError(f"worker exit codes {codes}; see {output}/worker_*.log")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        for stream in streams:
            stream.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-groups", type=int, default=int(os.environ.get("NUM_GROUPS", "4")))
    p.add_argument("--asset-root", type=Path, default=Path(os.environ.get("TTA_ASSET_ROOT", ROOT)))
    p.add_argument("--run-id", default="oracle_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = p.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id must be a simple directory name")
    output = RESULTS / args.run_id
    cfg = config()
    partition(candidates(), 0, args.num_groups)
    if args.dry_run:
        print("candidate_count=113\ntarget=target10\nlabels_used_for=posthoc_oracle_metrics_only\ntarget90_used=false")
        print(f"groups={args.num_groups}\noutput directory={output}")
        for key in ("K", "lr", "rho"):
            print(f"{key} grid={cfg[key]}")
        return
    if Path(sys.prefix).name != "tta":
        raise RuntimeError("activate the tta conda environment")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    from preflight import preflight
    from aggregate import aggregate, complete_scores
    evidence = preflight(output, args.asset_root, args.num_groups)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight.json", evidence)
    write_json(output / "run_config.json", {"config": cfg, "groups": args.num_groups,
               "interpreter": sys.executable, "asset_root": str(args.asset_root.resolve()),
               "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}})
    try:
        print("Starting 32-sample engineering smoke over all 113 candidates", flush=True)
        sweep(output / "smoke", args.asset_root, args.num_groups, limit=32)
        _, _, smoke_provenance = complete_scores(output / "smoke", sample_ids()[:32], args.num_groups)
        if smoke_provenance != evidence["provenance"]:
            raise ValueError("smoke/preflight provenance mismatch")
        write_json(output / "smoke_check.json", {"status": "PASS", "sample_count": 32,
                   "candidate_count": 113, "purpose": "engineering_only_no_metrics"})
        print("Smoke PASS; starting full 3178 x 113 sweep", flush=True)
        sweep(output, args.asset_root, args.num_groups)
        _, _, full_provenance = complete_scores(output, sample_ids(), args.num_groups)
        if full_provenance != evidence["provenance"]:
            raise ValueError("full/preflight provenance mismatch")
        aggregate(output, args.num_groups)
        print("Complete:", output / "analysis/report.md", flush=True)
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "FAIL", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()

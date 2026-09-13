"""L0-L2 CLI: structure validation and non-executing plans only."""
import argparse
import json
import sys
from pathlib import Path

from eptta import __version__
from eptta.config.resolve import load_resolved
from eptta.config.validate import STAGES, content_hash, report
from eptta.errors import EPTTAError, NotImplementedStage

# Future commands are named and documented but never dispatch a worker.
STUBS = {
    "inspect-data": ("inventory", "datasets out"),
    "propose-data-contract": ("inventory", "inventory out"),
    "approve-contract": ("data_build", "kind proposal review out"),
    "stage-data": ("data_build", "inventory raw-contract label-policy group-policy out"),
    "propose-splits": ("data_build", "staging policy out"),
    "build-manifests": ("data_build", "staging split-plan inventory raw-contract label-policy group-policy out"),
    "ingest-delta": ("data_build", "parent inventory contract out"),
    "commit-snapshot": ("data_build", "delta review out"),
    "refresh-plan": ("data_build", "delta artifacts out"),
    "resolve-training-recipe": ("source_training", "model-id snapshot preprocess out"),
    "inspect-model": ("source_training", "model-id mode bundle source-fixture out"),
    "preflight": ("source_training", "plan stage out"),
    "train-source": ("source_training", "recipe phase run-output"),
    "resume-source": ("source_training", "run checkpoint mode"),
    "finalize-training": ("source_training", "run out"),
    "export-frozen": ("frozen_extract", "training-manifest out"),
    "bind-artifacts": ("frozen_extract", "bundle paths-template out"),
    "extract": ("frozen_extract", "plan worker-slot worker-count"),
    "merge-cache": ("frozen_extract", "plan out"),
    "build-artifacts": ("frozen_extract", "plan cache-index out"),
    "run-suite": ("adaptation", "plan suite-id phase run-output frozen-spec"),
    "seal-scores": ("adaptation", "run"),
    "evaluate": ("evaluation", "run labels out"),
    "select-methods": ("adaptation", "metrics plan out"),
    "freeze": ("adaptation", "plan selection out"),
    "resume": ("adaptation", "run"),
    "report": ("evaluation", "run out"),
}
TASKS = {"source_prepare": "data_build", "source_training": "source_training",
         "frozen_extract": "frozen_extract", "adaptation": "adaptation", "evaluation": "evaluation"}


def parser():
    p = argparse.ArgumentParser(description="EP-TTA v0.1.0: local L0-L2 development, no real execution")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--profile", choices=("local_dev", "remote_a6000"), default="local_dev")
    p.add_argument("--paths", help="explicit private config file; resource bindings are not dereferenced")
    sub = p.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="validate structure or stage contract fields without resource I/O")
    val.add_argument("--level", choices=("structure", "resources"), default="structure")
    val.add_argument("--stage", choices=STAGES, default="development")
    plan = sub.add_parser("plan", help="write a preview; never execute or approve")
    plan.add_argument("--experiment", required=True)
    plan.add_argument("--task", choices=TASKS, required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--bundle", help="opaque optional reference, not loaded in a preview")
    plan.add_argument("--dry-run", action="store_true")
    for name, (_, options) in STUBS.items():
        cmd = sub.add_parser(name, help="reserved for L3-L6; raises NOT_IMPLEMENTED_STAGE")
        for option in options.split():
            cmd.add_argument(f"--{option}")
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--check-only", action="store_true")
    return p


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        profile = Path(args.config).resolve().parent / "profiles" / f"{args.profile}.yaml"
        cfg, sources = load_resolved(args.config, profile=profile, paths=args.paths,
                                    experiment=getattr(args, "experiment", None))
        if args.command == "validate":
            result = report(cfg, args.stage, args.level)
            result["field_sources"] = sources
            emit(result)
            return 2 if result["issues"] else 0
        if args.command == "plan":
            stage = TASKS[args.task]
            result = report(cfg, stage, "resources")
            result.update(status="PREVIEW_ONLY", task=args.task, dry_run=True,
                          requested_dry_run=args.dry_run, bundle_ref=args.bundle,
                          resolved_config=cfg, field_sources=sources, config_sha256=content_hash(cfg),
                          nodes=[{"id": s, "status": "NOT_RUN"} for s in STAGES[1:]],
                          edges=[[a, b] for a, b in zip(STAGES[1:-1], STAGES[2:])])
            dest = Path(args.out)
            if dest.exists():
                raise EPTTAError("plan output exists; overwrite is forbidden")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("x", encoding="utf-8") as output:
                json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
            emit({"schema_version": "0.1.0", "status": "PREVIEW_ONLY", "out": str(dest),
                  "execution_ready": False, "issues": result["issues"]})
            return 0
        stage = STUBS[args.command][0]
        # Listing options is supported; no unimplemented entry can return success.
        result = report(cfg, stage, "resources")
        emit({"schema_version": "0.1.0", "status": "NOT_RUN", "code": "NOT_IMPLEMENTED_STAGE",
              "command": args.command, "execution_ready": False,
              "message": "L0-L2 only; real worker/approval/resource I/O not implemented",
              "issues": result["issues"]})
        return 2
    except EPTTAError as exc:
        emit({"schema_version": "0.1.0", "status": "ERROR", "code": exc.code, "message": str(exc)})
        return exc.exit_code
    except (OSError, UnicodeError) as exc:
        emit({"schema_version": "0.1.0", "status": "ERROR", "code": "CONFIG_IO_ERROR", "message": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())

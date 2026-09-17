"""EP-TTA CLI: reviewed data, source training, frozen cache, and explicit stage gates."""
import argparse
import json
import sys
from pathlib import Path

from eptta import __version__
from eptta.config.resolve import load_resolved
from eptta.config.validate import STAGES, content_hash, report
from eptta.errors import EPTTAError, NotImplementedStage

# Remaining commands stay explicit refusal gates until their contracts exist.
STUBS = {
    "preflight": ("source_training", "plan stage out"),
    "bind-artifacts": ("frozen_extract", "bundle paths-template out"),
    "select-methods": ("adaptation", "metrics plan out"),
    "freeze": ("adaptation", "plan selection out"),
    "resume": ("adaptation", "run"),
    "report": ("evaluation", "run out"),
}
DATA_COMMANDS = ("inspect-data", "propose-data-contract", "approve-contract", "stage-data",
                 "propose-splits", "build-manifests", "ingest-delta", "commit-snapshot", "refresh-plan",
                 "prepare-labels", "validate-labels", "seal-source-manifests", "prepare-inference-manifest",
                 "audit-source-snapshot", "audit-target-metadata", "record-exposure")
TRAINING_COMMANDS = ("resolve-training-recipe", "inspect-model", "train-source",
                     "resume-source", "resume-preflight", "prepare-child-resume",
                     "compare-resume-validation", "finalize-training",
                     "finalize-administrative-child", "export-frozen", "export-frozen-r4")
EVALUATION_COMMANDS = ("seal-scores", "evaluate", "evaluate-checkpoint")
EXECUTION_COMMANDS = ("extract", "merge-cache", "build-artifacts", "prepare-r5-stage1-proposal",
                      "lock-r5-stage1-proposal", "run-r5-stage1-parity", "run-r5-stage1-k0",
                      "prepare-r5-stage2-proposal", "lock-r5-stage2-proposal",
                      "prepare-r6-proposal", "lock-r6-proposal", "run-r6")
ADAPTATION_COMMANDS = ("run-suite",)
TASKS = {"source_prepare": "data_build", "source_training": "source_training",
         "frozen_extract": "frozen_extract", "adaptation": "adaptation", "evaluation": "evaluation"}


def parser():
    p = argparse.ArgumentParser(description="EP-TTA v0.1.0: reviewed source-training and episodic experiment pipeline")
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
    inspect_data = sub.add_parser("inspect-data", help="read-only inventory; never approves labels")
    inspect_data.add_argument("--datasets", required=True)
    inspect_data.add_argument("--out", required=True)
    proposal = sub.add_parser("propose-data-contract", help="write unresolved proposals from inventory observations")
    proposal.add_argument("--inventory", required=True)
    proposal.add_argument("--out", required=True)
    approval = sub.add_parser("approve-contract", help="publish a reviewed immutable contract lock")
    approval.add_argument("--kind", choices=("raw", "label", "group", "preprocess", "split", "recipe"), required=True)
    approval.add_argument("--proposal", required=True)
    approval.add_argument("--review", required=True)
    approval.add_argument("--dataset-id")
    approval.add_argument("--out", required=True)
    stage_data = sub.add_parser("stage-data", help="parse reviewed data to a staging snapshot")
    for option in ("inventory", "raw-contract", "label-policy", "group-policy", "out"):
        stage_data.add_argument(f"--{option}", required=True)
    splits = sub.add_parser("propose-splits", help="create deterministic group-preserving assignments for review")
    for option in ("staging", "policy", "out"):
        splits.add_argument(f"--{option}", required=True)
    manifests = sub.add_parser("build-manifests", help="publish a reviewed immutable dataset snapshot")
    for option in ("staging", "split-plan", "out"):
        manifests.add_argument(f"--{option}", required=True)
    delta = sub.add_parser("ingest-delta", help="diff a candidate staging snapshot against an immutable parent")
    for option in ("parent", "inventory", "contract", "out"):
        delta.add_argument(f"--{option}", required=True)
    delta.add_argument("--dry-run", action="store_true")
    commit = sub.add_parser("commit-snapshot", help="append only approved non-conflicting delta records")
    for option in ("delta", "review", "out"):
        commit.add_argument(f"--{option}", required=True)
    refresh = sub.add_parser("refresh-plan", help="plan dependency reuse/rebuild after a delta")
    for option in ("delta", "artifacts", "out"):
        refresh.add_argument(f"--{option}", required=True)
    refresh.add_argument("--dry-run", action="store_true")
    labels = sub.add_parser("prepare-labels", help="build immutable ALLM-compatible canonical label CSV files")
    labels.add_argument("--spec", required=True, help="explicit private JSON/YAML source specification")
    labels.add_argument("--out", required=True)
    validate_labels = sub.add_parser("validate-labels", help="validate hashes, labels, IDs, and optional audio paths")
    validate_labels.add_argument("--labels", required=True, help="label-pack directory or index.json")
    validate_labels.add_argument("--check-audio", action="store_true")
    source_manifests = sub.add_parser("seal-source-manifests", help="validate and seal fit/source_val references")
    source_manifests.add_argument("--snapshot", required=True)
    source_manifests.add_argument("--out", required=True)
    inference_manifest = sub.add_parser("prepare-inference-manifest", help="strip labels into an immutable inference view")
    inference_manifest.add_argument("--snapshot", required=True)
    inference_manifest.add_argument("--role", required=True)
    inference_manifest.add_argument("--out", required=True)
    source_audit = sub.add_parser("audit-source-snapshot", help="read-only audit of an existing locked snapshot")
    source_audit.add_argument("--snapshot", required=True)
    source_audit.add_argument("--out", required=True)
    target_audit = sub.add_parser("audit-target-metadata", help="strictly parse one principal target scope without publishing it")
    target_audit.add_argument("--dataset-id", choices=("asvspoof2021_la", "asvspoof2021_df", "in_the_wild"), required=True)
    target_audit.add_argument("--metadata", required=True)
    target_audit.add_argument("--audio-dir", required=True)
    target_audit.add_argument("--release", required=True)
    target_audit.add_argument("--subset", choices=("eval", "progress", "hidden"), required=True)
    target_audit.add_argument("--source-snapshot")
    target_audit.add_argument("--out", required=True)
    exposure = sub.add_parser("record-exposure", help="append one hash-chained target access event")
    exposure.add_argument("--ledger", required=True)
    for option in ("recorded-at", "dataset-id", "scope", "event-type", "artifact-ref",
                   "artifact-sha256", "decision-impact"):
        exposure.add_argument("--" + option, required=True)
    exposure.add_argument("--effect-visible", choices=("true", "false"), required=True)
    resolve_recipe = sub.add_parser("resolve-training-recipe", help="bind author/data/preprocess identities for review")
    for option in ("model-id", "snapshot", "preprocess", "out"):
        resolve_recipe.add_argument("--" + option, required=True)
    inspect_model = sub.add_parser("inspect-model", help="verify pinned author architecture source without weights")
    inspect_model.add_argument("--model-id", required=True)
    inspect_model.add_argument("--mode", choices=("architecture", "frozen_bundle"), required=True)
    inspect_model.add_argument("--bundle")
    inspect_model.add_argument("--source-fixture")
    inspect_model.add_argument("--out", required=True)
    train_source = sub.add_parser("train-source", help="run the source worker on fit/source_val only")
    train_source.add_argument("--recipe", required=True)
    train_source.add_argument("--phase", choices=("smoke", "full"), required=True)
    train_source.add_argument("--run-output", required=True)
    train_source.add_argument("--worker-python")
    resume_source = sub.add_parser("resume-source", help="resume an exact epoch-boundary source checkpoint")
    resume_source.add_argument("--run", required=True)
    resume_source.add_argument("--checkpoint", required=True)
    resume_source.add_argument("--mode", choices=("epoch_boundary",), required=True)
    resume_source.add_argument("--worker-python")
    resume_preflight = sub.add_parser("resume-preflight", help="read-only checkpoint and identity audit")
    resume_preflight.add_argument("--run", required=True)
    resume_preflight.add_argument("--checkpoint", required=True)
    resume_preflight.add_argument("--worker", required=True)
    resume_preflight.add_argument("--route", choices=("exact", "child"), required=True)
    resume_preflight.add_argument("--target-max-epochs", type=int, required=True)
    resume_preflight.add_argument("--out", required=True)
    prepare_child = sub.add_parser("prepare-child-resume", help="publish an approved route-B child job without executing it")
    prepare_child.add_argument("--plan", required=True)
    prepare_child.add_argument("--plan-sha256", required=True)
    prepare_child.add_argument("--model-id", choices=("aasist_source", "ssl_aasist_source"), required=True)
    prepare_child.add_argument("--checkpoint", required=True)
    prepare_child.add_argument("--worker", required=True)
    prepare_child.add_argument("--run-output", required=True)
    prepare_child.add_argument("--purpose", choices=("resume_validation", "formal_continuation"), required=True)
    prepare_child.add_argument("--validation-epochs", type=int, choices=(1, 2))
    prepare_child.add_argument("--validation-report")
    compare_resume = sub.add_parser("compare-resume-validation", help="compare continuous and restarted epoch states exactly")
    compare_resume.add_argument("--model-id", choices=("aasist_source", "ssl_aasist_source"), required=True)
    compare_resume.add_argument("--continuous-checkpoint", required=True)
    compare_resume.add_argument("--restarted-checkpoint", required=True)
    compare_resume.add_argument("--continuous-log", required=True)
    compare_resume.add_argument("--restarted-log", required=True)
    compare_resume.add_argument("--out", required=True)
    finalize = sub.add_parser("finalize-training", help="select source checkpoint by source_val EER")
    finalize.add_argument("--run", required=True)
    finalize.add_argument("--out", required=True)
    admin_finalize = sub.add_parser("finalize-administrative-child", help="finalize the approved SSL no-training child migration")
    admin_finalize.add_argument("--plan", required=True)
    admin_finalize.add_argument("--plan-sha256", required=True)
    admin_finalize.add_argument("--validation-report", required=True)
    admin_finalize.add_argument("--out", required=True)
    export_frozen = sub.add_parser("export-frozen", help="legacy exporter; explicitly blocked by the R4 contract")
    export_frozen.add_argument("--training-manifest", required=True)
    export_frozen.add_argument("--out", required=True)
    export_frozen.add_argument("--worker-python")
    export_r4 = sub.add_parser("export-frozen-r4", help="AASIST R4 independent real-data parity and gated frozen publication")
    export_r4.add_argument("--training-manifest", required=True)
    export_r4.add_argument("--out", required=True, help="new R4 run directory; never overwritten")
    export_r4.add_argument("--worker-python")
    export_r4.add_argument("--gpu-id", type=int, default=0)
    export_r4.add_argument("--dry-run", action="store_true", help="read-only identity-bound preview; does not start CUDA")
    seal = sub.add_parser("seal-scores", help="immutably seal label-free method scores")
    seal.add_argument("--run", required=True)
    evaluate = sub.add_parser("evaluate", help="join labels only after score sealing")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--threshold", required=True, type=float)
    checkpoint_eval = sub.add_parser("evaluate-checkpoint", help="evaluate any architecture-compatible parameter file")
    checkpoint_eval.add_argument("--model-id", choices=("aasist_source", "ssl_aasist_source"), required=True)
    checkpoint_eval.add_argument("--checkpoint", required=True)
    checkpoint_eval.add_argument("--protocol", required=True)
    checkpoint_eval.add_argument("--audio-dir", required=True)
    checkpoint_eval.add_argument("--out", required=True)
    checkpoint_eval.add_argument("--batch-size", type=int, default=48)
    checkpoint_eval.add_argument("--num-workers", type=int, default=4)
    checkpoint_eval.add_argument("--threshold", type=float, default=0.0)
    checkpoint_eval.add_argument("--gpu-id", type=int)
    checkpoint_eval.add_argument("--evaluation-tag", default="unspecified")
    checkpoint_eval.add_argument("--asv-scores")
    checkpoint_eval.add_argument("--worker-python")
    extract = sub.add_parser("extract", help="run one label-free frozen feature shard")
    extract.add_argument("--plan", required=True)
    extract.add_argument("--worker-slot", required=True, type=int)
    extract.add_argument("--worker-count", required=True, type=int)
    extract.add_argument("--worker-python")
    merge = sub.add_parser("merge-cache", help="verify and merge feature-cache shards")
    merge.add_argument("--plan", required=True)
    merge.add_argument("--out", required=True)
    suite = sub.add_parser("run-suite", help="run label-free registered methods from a frozen feature cache")
    suite.add_argument("--plan", required=True)
    suite.add_argument("--suite-id", required=True)
    suite.add_argument("--phase", choices=("select", "confirmatory"), required=True)
    suite.add_argument("--run-output", required=True)
    suite.add_argument("--frozen-spec")
    artifacts = sub.add_parser("build-artifacts", help="build source-only U/M/tau/Fisher/static-R resources")
    artifacts.add_argument("--plan", required=True)
    artifacts.add_argument("--cache-index", required=True)
    artifacts.add_argument("--out", required=True)
    r5_proposal = sub.add_parser("prepare-r5-stage1-proposal", help="write a review-only R5 small-cache proposal; never lock or execute")
    r5_proposal.add_argument("--plan", required=True)
    r5_proposal.add_argument("--plan-sha256", required=True)
    r5_proposal.add_argument("--bundle", required=True)
    r5_proposal.add_argument("--snapshot", required=True)
    r5_proposal.add_argument("--out", required=True)
    r5_lock = sub.add_parser("lock-r5-stage1-proposal", help="publish LOCKED extraction plans after explicit review")
    r5_lock.add_argument("--proposal", required=True)
    r5_lock.add_argument("--proposal-sha256", required=True)
    r5_parity = sub.add_parser("run-r5-stage1-parity", help="re-encode deterministic views and verify cache read-back parity")
    r5_parity.add_argument("--plan", required=True)
    r5_parity.add_argument("--role", choices=("fit", "cal0", "select"), required=True)
    r5_parity.add_argument("--cache", required=True)
    r5_parity.add_argument("--out", required=True)
    r5_parity.add_argument("--worker-python")
    r5_k0 = sub.add_parser("run-r5-stage1-k0", help="verify the resource-free K=0 EP entrypoint against the frozen path")
    r5_k0.add_argument("--bundle", required=True)
    r5_k0.add_argument("--cache", required=True)
    r5_k0.add_argument("--role", choices=("fit", "cal0", "select"), required=True)
    r5_k0.add_argument("--out", required=True)
    r5_stage2 = sub.add_parser("prepare-r5-stage2-proposal", help="fix the full-cache + source-resource proposal; never lock or execute")
    r5_stage2.add_argument("--plan", required=True)
    r5_stage2.add_argument("--plan-sha256", required=True)
    r5_stage2.add_argument("--bundle", required=True)
    r5_stage2.add_argument("--snapshot", required=True)
    r5_stage2.add_argument("--out", required=True)
    r5_lock2 = sub.add_parser("lock-r5-stage2-proposal", help="publish LOCKED full-cache plans after explicit review")
    r5_lock2.add_argument("--proposal", required=True)
    r5_lock2.add_argument("--proposal-sha256", required=True)
    r6_proposal = sub.add_parser("prepare-r6-proposal", help="fix the R6 source-pilot proposal; never lock or execute")
    r6_proposal.add_argument("--plan", required=True)
    r6_proposal.add_argument("--bundle", required=True)
    r6_proposal.add_argument("--resources", required=True)
    r6_proposal.add_argument("--select-cache", required=True)
    r6_proposal.add_argument("--select-labels", required=True)
    r6_proposal.add_argument("--out", required=True)
    r6_lock = sub.add_parser("lock-r6-proposal", help="publish the LOCKED R6 proposal after explicit review")
    r6_lock.add_argument("--proposal", required=True)
    r6_lock.add_argument("--proposal-sha256", required=True)
    r6_run = sub.add_parser("run-r6", help="score the full select cache, seal and evaluate all R6 methods")
    r6_run.add_argument("--proposal", required=True)
    r6_run.add_argument("--out", required=True)
    for name, (_, options) in STUBS.items():
        cmd = sub.add_parser(name, help="reserved for L4-L6; raises NOT_IMPLEMENTED_STAGE")
        for option in options.split():
            cmd.add_argument(f"--{option}")
        cmd.add_argument("--dry-run", action="store_true")
        cmd.add_argument("--check-only", action="store_true")
    return p


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def _require_data_io(cfg, stage):
    from eptta.config.validate import report
    from eptta.errors import PermissionDenied
    readiness = report(cfg, stage, "resources")
    if readiness["issues"]:
        raise PermissionDenied("; ".join(f"{item['field']}: {item['message']}" for item in readiness["issues"]))


def _data_command(args, cfg):
    from eptta.config.schema import read_document
    from eptta.data.io import AtomicDirectory, write_json_new
    if args.command == "inspect-data":
        _require_data_io(cfg, "inventory")
        from eptta.data.inventory import inspect_selected
        requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = set(requested) - set(cfg["paths"]["dataset_roots"])
        if not requested or unknown:
            raise EPTTAError(f"unknown or empty dataset selection: {sorted(unknown)}")
        result = inspect_selected(requested, cfg["paths"]["dataset_roots"], cfg["paths"]["protocol_files"])
        with AtomicDirectory(args.out) as temporary:
            write_json_new(temporary / "inventory.json", result)
        return {"schema_version": "0.1.0", "status": "INVENTORIED", "out": str(Path(args.out) / "inventory.json"),
                "datasets": requested, "label_contract_approved": False}, 0
    if args.command == "propose-data-contract":
        from eptta.data.approval import propose_from_inventory
        result = propose_from_inventory(read_document(args.inventory))
        write_json_new(args.out, result)
        return {"schema_version": "0.1.0", "status": "PROPOSED", "out": args.out,
                "execution_ready": False}, 0
    if args.command == "approve-contract":
        from eptta.data.approval import approve
        result = approve(read_document(args.proposal), read_document(args.review), args.kind, args.dataset_id)
        write_json_new(args.out, result)
        return {"schema_version": "0.1.0", "status": "LOCKED", "kind": args.kind, "out": args.out,
                "content_sha256": result["approval"]["content_sha256"]}, 0
    if args.command == "stage-data":
        _require_data_io(cfg, "inventory")
        from eptta.data.staging import stage_dataset
        result = stage_dataset(read_document(args.inventory), read_document(args.raw_contract),
                               read_document(args.label_policy), read_document(args.group_policy), args.out)
        return result, 4 if result["status"] != "STAGED" else 0
    if args.command == "propose-splits":
        _require_data_io(cfg, "inventory")
        from eptta.data.splits import propose_splits
        result = propose_splits(args.staging, read_document(args.policy), args.out)
        return {"schema_version": "0.1.0", "status": "PROPOSED", "out": args.out,
                "counts": result["payload"]["counts"], "execution_ready": False}, 0
    if args.command == "build-manifests":
        _require_data_io(cfg, "inventory")
        from eptta.data.manifests import build_manifests
        result = build_manifests(args.staging, args.split_plan, args.out)
        return result, 0
    if args.command == "ingest-delta":
        _require_data_io(cfg, "inventory")
        from eptta.data.incremental import build_delta
        policy = read_document(args.contract)
        payload = policy.get("payload", policy)
        if payload.get("preserve_existing_assignments") is not True or payload.get("automatic_resplit") is not False:
            raise EPTTAError("incremental contract must preserve assignments and disable automatic resplit")
        if args.dry_run:
            return {"schema_version": "0.1.0", "status": "PREVIEW_ONLY", "parent": args.parent,
                    "candidate_staging": args.inventory, "execution_ready": False}, 0
        result = build_delta(args.parent, args.inventory, args.out)
        return result, 0
    if args.command == "commit-snapshot":
        _require_data_io(cfg, "inventory")
        from eptta.data.incremental import commit_delta
        result = commit_delta(args.delta, read_document(args.review), args.out)
        return result, 0
    if args.command == "refresh-plan":
        from eptta.data.incremental import refresh_plan
        delta_path = Path(args.delta)
        delta = read_document(delta_path / "delta.json" if delta_path.is_dir() else delta_path)
        result = refresh_plan(delta, read_document(args.artifacts))
        result["requested_dry_run"] = args.dry_run
        write_json_new(args.out, result)
        return {"schema_version": "0.1.0", "status": "PREVIEW_ONLY", "out": args.out,
                "execution_ready": False}, 0
    if args.command == "prepare-labels":
        _require_data_io(cfg, "inventory")
        from eptta.data.unified_labels import prepare_label_pack
        result = prepare_label_pack(read_document(args.spec), args.out)
        return {**result, "out": str(Path(args.out) / "index.json")}, 0
    if args.command == "validate-labels":
        if args.check_audio:
            _require_data_io(cfg, "inventory")
        from eptta.data.unified_labels import validate_label_pack
        return validate_label_pack(args.labels, check_audio=args.check_audio), 0
    if args.command == "seal-source-manifests":
        _require_data_io(cfg, "inventory")
        from eptta.data.source_manifests import seal_source_manifests
        return seal_source_manifests(args.snapshot, args.out), 0
    if args.command == "prepare-inference-manifest":
        _require_data_io(cfg, "inventory")
        from eptta.data.source_manifests import publish_label_free_manifest
        return publish_label_free_manifest(args.snapshot, args.role, args.out), 0
    if args.command == "audit-source-snapshot":
        _require_data_io(cfg, "inventory")
        from eptta.data.audit import audit_source_snapshot
        return audit_source_snapshot(args.snapshot, args.out), 0
    if args.command == "audit-target-metadata":
        _require_data_io(cfg, "inventory")
        from eptta.data.target_metadata import audit_target_metadata
        return audit_target_metadata(args.dataset_id, args.metadata, args.audio_dir,
                                     args.release, args.subset, args.out, args.source_snapshot), 0
    if args.command == "record-exposure":
        from eptta.data.exposure import append_exposure_event
        event = {"recorded_at": args.recorded_at, "dataset_id": args.dataset_id,
                 "scope": args.scope, "event_type": args.event_type,
                 "artifact_ref": args.artifact_ref, "artifact_sha256": args.artifact_sha256,
                 "effect_visible": args.effect_visible == "true",
                 "decision_impact": args.decision_impact}
        return append_exposure_event(args.ledger, event), 0
    raise EPTTAError(f"unknown data command: {args.command}")


def _training_command(args, cfg):
    if cfg["runtime"]["environment"] != "remote":
        from eptta.errors import PermissionDenied
        raise PermissionDenied("source architecture/data/weight operations require the remote profile")
    permissions = cfg["runtime"]["permissions"]
    if args.command in ("train-source", "resume-source") and not permissions.get("real_source_training"):
        from eptta.errors import PermissionDenied
        raise PermissionDenied("real_source_training permission is disabled")
    if args.command == "resolve-training-recipe":
        from eptta.registry import get_spec
        from eptta.training.recipe import resolve_training_recipe
        model = get_spec("models", args.model_id)
        repo_key = "ssl_aasist" if args.model_id == "ssl_aasist_source" else "aasist"
        repo = cfg["paths"]["source_repos"].get(repo_key)
        if not repo:
            raise EPTTAError("source repository binding is unresolved: " + repo_key)
        initialization = cfg["paths"].get("generic_ssl_initialization") if args.model_id == "ssl_aasist_source" else None
        result = resolve_training_recipe(args.model_id, args.snapshot, args.preprocess,
                                         model["training_recipe_ref"], repo,
                                         cfg["paths"]["dataset_roots"], initialization, args.out)
        return {"schema_version": "0.1.0", "status": "PROPOSED", "out": args.out,
                "execution_ready": False,
                "unresolved_fields": [key for key, value in result["payload"].items() if value is None]}, 0
    if args.command == "inspect-model":
        if args.mode != "architecture":
            raise NotImplementedStage("frozen bundle inspection requires export-frozen")
        from eptta.models.author import inspect_author_repository
        from eptta.data.io import write_json_new
        repo_key = "ssl_aasist" if args.model_id == "ssl_aasist_source" else "aasist"
        repo = cfg["paths"]["source_repos"].get(repo_key)
        if not repo:
            raise EPTTAError("source repository binding is unresolved: " + repo_key)
        result = inspect_author_repository(args.model_id, repo)
        write_json_new(Path(args.out) / "architecture.json", result)
        return {**result, "out": str(Path(args.out) / "architecture.json"), "weights_loaded": False}, 0
    if args.command == "train-source":
        from eptta.training.dispatch import compile_source_job, launch_source_job
        job = compile_source_job(args.recipe, args.phase, args.run_output)
        return launch_source_job(job, python_executable=args.worker_python), 0
    if args.command == "resume-source":
        from eptta.training.dispatch import launch_resume_job
        return launch_resume_job(args.run, args.checkpoint, python_executable=args.worker_python), 0
    if args.command == "resume-preflight":
        from eptta.training.resume import compile_resume_preflight
        return compile_resume_preflight(args.run, args.checkpoint, args.worker, args.route,
                                        args.target_max_epochs, args.out), 0
    if args.command == "prepare-child-resume":
        from eptta.training.resume import prepare_child_resume
        return prepare_child_resume(args.plan, args.plan_sha256, args.model_id,
                                    args.checkpoint, args.worker, args.run_output,
                                    args.purpose, args.validation_epochs,
                                    args.validation_report), 0
    if args.command == "compare-resume-validation":
        from eptta.training.resume import compare_resume_validation
        return compare_resume_validation(args.model_id, args.continuous_checkpoint,
                                         args.restarted_checkpoint, args.continuous_log,
                                         args.restarted_log, args.out), 0
    if args.command == "finalize-training":
        from eptta.training.artifacts import finalize_training
        return finalize_training(args.run, args.out), 0
    if args.command == "finalize-administrative-child":
        from eptta.training.artifacts import finalize_administrative_child
        return finalize_administrative_child(args.plan, args.plan_sha256,
                                             args.validation_report, args.out), 0
    if args.command == "export-frozen":
        from eptta.training.artifacts import launch_frozen_export
        return launch_frozen_export(args.training_manifest, args.out,
                                    python_executable=args.worker_python), 0
    if args.command == "export-frozen-r4":
        from eptta.training.r4 import compile_r4_preview, launch_r4_export
        if args.dry_run:
            return compile_r4_preview(args.training_manifest, args.out,
                                      python_executable=args.worker_python,
                                      gpu_id=args.gpu_id), 0
        return launch_r4_export(args.training_manifest, args.out,
                                python_executable=args.worker_python,
                                gpu_id=args.gpu_id), 0
    raise EPTTAError("unknown training command: " + args.command)


def _evaluation_command(args, cfg):
    if args.command == "seal-scores":
        from eptta.evaluation.seal import seal_scores
        return seal_scores(args.run), 0
    if cfg["runtime"]["environment"] != "remote":
        from eptta.errors import PermissionDenied
        raise PermissionDenied("real evaluation labels require the remote profile")
    if args.command == "evaluate-checkpoint":
        from eptta.evaluation.checkpoint import compile_checkpoint_evaluation, launch_checkpoint_evaluation
        job = compile_checkpoint_evaluation(args.model_id, args.checkpoint, args.protocol,
                                            args.audio_dir, args.out, cfg, args.batch_size,
                                            args.num_workers, args.threshold, args.gpu_id,
                                            args.evaluation_tag, args.asv_scores)
        return launch_checkpoint_evaluation(job, args.worker_python), 0
    from eptta.evaluation.seal import evaluate_sealed
    return evaluate_sealed(args.run, args.labels, args.out, args.threshold), 0


def _execution_command(args, cfg):
    if args.command == "prepare-r5-stage1-proposal":
        from eptta.execution.r5_proposal import prepare_r5_stage1_proposal
        return prepare_r5_stage1_proposal(args.plan, args.plan_sha256, args.bundle,
                                          args.snapshot, args.out), 0
    if args.command == "lock-r5-stage1-proposal":
        from eptta.execution.r5_stage1 import lock_stage1_proposal
        return lock_stage1_proposal(args.proposal, args.proposal_sha256), 0
    if args.command == "run-r5-stage1-parity":
        from eptta.execution.r5_stage1 import compile_parity_job, launch_parity
        job, worker = compile_parity_job(args.plan, args.role, args.cache, args.out)
        report = launch_parity(job, worker, args.worker_python)
        return report, 0
    if args.command == "run-r5-stage1-k0":
        from eptta.execution.r5_stage1 import run_k0_verification
        return run_k0_verification(args.bundle, args.cache, args.role, args.out), 0
    if args.command == "prepare-r5-stage2-proposal":
        from eptta.execution.r5_stage2 import prepare_r5_stage2_proposal
        return prepare_r5_stage2_proposal(args.plan, args.plan_sha256, args.bundle,
                                          args.snapshot, args.out), 0
    if args.command == "lock-r5-stage2-proposal":
        from eptta.execution.r5_stage2 import lock_r5_stage2_proposal
        return lock_r5_stage2_proposal(args.proposal, args.proposal_sha256), 0
    if args.command == "prepare-r6-proposal":
        from eptta.execution.r6 import prepare_r6_proposal
        return prepare_r6_proposal(args.plan, args.bundle, args.resources, args.select_cache,
                                   args.select_labels, args.out), 0
    if args.command == "lock-r6-proposal":
        from eptta.execution.r6 import lock_r6_proposal
        return lock_r6_proposal(args.proposal, args.proposal_sha256), 0
    if args.command == "run-r6":
        from eptta.execution.r6 import run_r6
        return run_r6(args.proposal, args.out), 0
    if args.command == "build-artifacts":
        if cfg["runtime"]["environment"] != "remote":
            from eptta.errors import PermissionDenied
            raise PermissionDenied("real source artifacts require the remote profile")
        from eptta.offline.artifacts import build_source_resources
        return build_source_resources(args.plan, args.cache_index, args.out), 0
    if args.command == "extract":
        if cfg["runtime"]["environment"] != "remote":
            from eptta.errors import PermissionDenied
            raise PermissionDenied("real frozen feature extraction requires the remote profile")
        from eptta.execution.extract import compile_inference_job, launch_extraction
        job = compile_inference_job(args.plan, args.worker_slot, args.worker_count)
        return launch_extraction(job, python_executable=args.worker_python), 0
    from eptta.config.schema import read_document
    from eptta.cache.merge import merge_feature_caches
    plan = read_document(args.plan)
    if set(plan) != {"schema_version", "status", "shards", "expected_ids_ref"} or plan.get("status") != "LOCKED":
        raise EPTTAError("cache merge plan must be a strict LOCKED v0.1.0 document")
    expected = read_document(plan["expected_ids_ref"])
    return merge_feature_caches(plan["shards"], args.out, expected), 0


def _adaptation_command(args, cfg):
    from eptta.execution.suite import run_suite
    return run_suite(args.plan, args.suite_id, args.phase, args.run_output, args.frozen_spec), 0


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
        if args.command in DATA_COMMANDS:
            payload, code = _data_command(args, cfg)
            emit(payload)
            return code
        if args.command in TRAINING_COMMANDS:
            payload, code = _training_command(args, cfg)
            emit(payload)
            return code
        if args.command in EVALUATION_COMMANDS:
            payload, code = _evaluation_command(args, cfg)
            emit(payload)
            return code
        if args.command in EXECUTION_COMMANDS:
            payload, code = _execution_command(args, cfg)
            emit(payload)
            return code
        if args.command in ADAPTATION_COMMANDS:
            payload, code = _adaptation_command(args, cfg)
            emit(payload)
            return code
        stage = STUBS[args.command][0]
        # Listing options is supported; no unimplemented entry can return success.
        result = report(cfg, stage, "resources")
        emit({"schema_version": "0.1.0", "status": "NOT_RUN", "code": "NOT_IMPLEMENTED_STAGE",
              "command": args.command, "execution_ready": False,
              "message": "L4-L6 worker path is not implemented; L3 data commands are separate",
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

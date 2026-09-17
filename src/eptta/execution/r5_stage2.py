"""R5 stage-2: fix the full-cache + source-resource proposal (never lock or execute here)."""
from __future__ import annotations

import json
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export

R5_PLAN_SHA256 = "07024939e2a156e681e006ace20b1b542b2cfde811c30e87d4977ffa3ecac79d"
STAGE1_LOCK_SHA256 = "e79f1d692285eecd9c80d86542f1ff6b5c7f3efff208fdc3a6be6118b2290a42"
BUNDLE_SHA256 = "d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684"
EXPORT_MANIFEST_SHA256 = "8cd0c21d3d89b46dbf4d07f86df29bce8e18803b952ed58d5bb6441264210c56"
SNAPSHOT_SHA256 = "50d893131ce99752759fb14a121bd17c29a9782eddb0d8560ac458d336315fe0"
ROLES = ("fit", "cal0", "select")
ROLE_COUNTS = {"fit": 25380, "cal0": 4906, "select": 11520}


def _jsonl_new(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _unlabeled(row):
    return {key: row[key] for key in ("schema_version", "sample_id", "root_key", "audio_relpath",
                                      "input_sha256", "split_role")}


def prepare_r5_stage2_proposal(plan_ref, expected_plan_sha256, bundle_ref, snapshot_ref, output):
    plan_path, bundle_path, snapshot_root = Path(plan_ref), Path(bundle_ref), Path(snapshot_ref)
    if expected_plan_sha256 != R5_PLAN_SHA256 or sha256_file(plan_path) != R5_PLAN_SHA256:
        raise ContractError("R5 source-preparation plan SHA-256 changed")
    if sha256_file(bundle_path) != BUNDLE_SHA256:
        raise ContractError("only the approved R4 v4 bundle may enter this proposal")
    if sha256_file(bundle_path.parent / "export_manifest.json") != EXPORT_MANIFEST_SHA256:
        raise ContractError("R4 v4 export manifest changed")
    bundle, _manifest, _parity, _selection = verify_frozen_export(bundle_path)
    if bundle["baseline_id"] != "baseline-d1f0d91901c73eb5027c" or bundle[
            "selected_checkpoint_sha256"] != "076ca355cec3358c7181769459de27279609ab1cda35f186670bc49e9a1cbf0a":
        raise ContractError("R4 v4 identity mismatch")
    snapshot = read_json(snapshot_root / "snapshot.json")
    if (snapshot.get("status") != "LOCKED" or snapshot.get("snapshot_id") !=
            "snapshot-5731a8d70a8b8fa4997d" or snapshot.get("canonical_sha256") != SNAPSHOT_SHA256 or
            sha256_file(snapshot_root / snapshot["canonical_ref"]) != SNAPSHOT_SHA256):
        raise ContractError("approved source snapshot identity changed")
    data_roots = {"asvspoof2019_la": "/media/dell/data/fakedata/asvspoof2019/LA"}
    worker = Path(__file__).parents[3] / "workers/baseline_bridge.py"
    parity_worker = Path(__file__).parents[3] / "workers/r5_parity_bridge.py"
    probe = {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": 0.05}
    numerical = {"dtype": "float32", "tf32_matmul": False, "tf32_cudnn": True, "block_units": 256}
    destination = Path(output).resolve()
    with AtomicDirectory(destination) as temporary:
        role_records = {}
        for role in ROLES:
            manifest = snapshot_root / "manifests" / (role + ".jsonl")
            expected = snapshot["role_manifest_hashes"][role + ".jsonl"]
            if sha256_file(manifest) != expected:
                raise DataError(role + " manifest changed")
            rows = list(iter_jsonl(manifest))
            if len(rows) != ROLE_COUNTS[role] or len({row["sample_id"] for row in rows}) != len(rows):
                raise DataError(role + " full manifest count/unique mismatch")
            unlabeled_path = temporary / (role + ".unlabeled.jsonl")
            label_path = temporary / (role + ".labels.jsonl")
            group_path = temporary / (role + ".groups.jsonl")
            _jsonl_new(unlabeled_path, [_unlabeled(row) for row in rows])
            # canonical_label and source_group_id live only in the canonical record.
            canonical = {row["sample_id"]: row for row in
                         iter_jsonl(snapshot_root / snapshot["canonical_ref"])
                         if row.get("split_role") == role}
            if set(canonical) != {row["sample_id"] for row in rows}:
                raise DataError(role + " canonical coverage mismatch")
            _jsonl_new(label_path, [{"schema_version": "0.1.0", "sample_id": row["sample_id"],
                                     "canonical_label": canonical[row["sample_id"]]["canonical_label"]}
                                    for row in rows])
            _jsonl_new(group_path, [{"schema_version": "0.1.0", "sample_id": row["sample_id"],
                                     "source_group_id": canonical[row["sample_id"]]["source_group_id"]}
                                    for row in rows])
            purpose = "select" if role == "select" else "source_prepare"
            extraction = {"schema_version": "0.1.0", "status": "PROPOSED", "purpose": purpose,
                          "input_role": role, "frozen_bundle_ref": str(bundle_path.resolve()),
                          "manifest_ref": str((destination / unlabeled_path.name).resolve()),
                          "manifest_sha256": sha256_file(unlabeled_path), "data_roots": data_roots,
                          "probe": probe, "numerical_mode": numerical,
                          "output_root": str((destination / "caches" / role).resolve())}
            extraction_path = temporary / (role + ".extraction.proposal.json")
            write_json_new(extraction_path, extraction)
            role_records[role] = {"purpose": purpose, "count": len(rows),
                                  "unlabeled_manifest_ref": unlabeled_path.name,
                                  "unlabeled_manifest_sha256": sha256_file(unlabeled_path),
                                  "labels_ref": label_path.name,
                                  "labels_sha256": sha256_file(label_path),
                                  "groups_ref": group_path.name,
                                  "groups_sha256": sha256_file(group_path),
                                  "extraction_proposal_ref": extraction_path.name,
                                  "extraction_proposal_sha256": sha256_file(extraction_path)}
        artifacts_plan = {"schema_version": "0.1.0", "status": "PROPOSED",
                          "frozen_bundle_ref": str(bundle_path.resolve()),
                          "fit_role": "fit", "fit_manifest_sha256": role_records["fit"]["unlabeled_manifest_sha256"],
                          "fit_labels_ref": str((destination / "fit.labels.jsonl").resolve()),
                          "fit_labels_sha256": role_records["fit"]["labels_sha256"],
                          "fit_groups_ref": str((destination / "fit.groups.jsonl").resolve()),
                          "fit_groups_sha256": role_records["fit"]["groups_sha256"],
                          "calibration_role": "cal0",
                          "calibration_manifest_sha256": role_records["cal0"]["unlabeled_manifest_sha256"],
                          "calibration_cache_ref": str((destination / "caches/cal0/shard-00000-of-00001").resolve()),
                          "calibration_labels_ref": str((destination / "cal0.labels.jsonl").resolve()),
                          "calibration_labels_sha256": role_records["cal0"]["labels_sha256"],
                          "source_snapshot_hash": SNAPSHOT_SHA256, "rank": 8, "alpha_cal": 0.05,
                          "anchor_per_class": 128, "seed": 13, "random_seeds": [13, 29, 47],
                          "treatment_families": ["noise", "fir"], "samples_per_group": 64,
                          "pair_seed": 13, "margin_bins": 4, "margin_epsilon": 1e-6,
                          "minimum_cal0_bonafide": 100,
                          "fixed_adapter": {"steps": 200, "lr": 0.01, "rho": 0.2, "gamma": 0.1,
                                            "lambda_keep": 1.0}}
        artifacts_path = temporary / "source_artifacts.proposal.json"
        write_json_new(artifacts_path, artifacts_plan)
        root_text = "/media/dell/data/fakeAudioDection/TTA"
        out_text = str(destination)
        commands = {}
        for role in ROLES:
            commands["extract_" + role] = (
                "env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python "
                "-m eptta.cli --config configs/base.yaml --profile remote_a6000 "
                "--paths configs/paths.fakedata.private.yaml extract --plan " + out_text +
                "/locked/" + role + ".extraction.lock.json --worker-slot 0 --worker-count 1 "
                "--worker-python /home/dell/anaconda3/envs/py38/bin/python")
            commands["parity_" + role] = (
                "env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python "
                "-m eptta.cli --config configs/base.yaml --profile remote_a6000 "
                "--paths configs/paths.fakedata.private.yaml run-r5-stage1-parity --plan " + out_text +
                "/locked/" + role + ".extraction.lock.json --role " + role + " --cache " + out_text +
                "/caches/" + role + "/shard-00000-of-00001 --out " + out_text + "/parity/" + role +
                " --worker-python /home/dell/anaconda3/envs/py38/bin/python")
        commands["build_artifacts"] = (
            "env PYTHONPATH=src /home/dell/anaconda3/envs/py310/bin/python -m eptta.cli "
            "--config configs/base.yaml --profile remote_a6000 --paths configs/paths.fakedata.private.yaml "
            "build-artifacts --plan " + out_text + "/locked/source_artifacts.lock.json --cache-index " +
            out_text + "/caches/fit/shard-00000-of-00001 --out " + out_text + "/resources")
        proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                    "locks_published": False, "proposal_id": "r5-stage2-full-20260917",
                    "parent_plan_ref": str(plan_path.resolve()), "parent_plan_sha256": R5_PLAN_SHA256,
                    "stage1_lock_sha256": STAGE1_LOCK_SHA256,
                    "bundle_ref": str(bundle_path.resolve()), "bundle_sha256": BUNDLE_SHA256,
                    "bundle_id": bundle["baseline_id"], "export_manifest_sha256": EXPORT_MANIFEST_SHA256,
                    "checkpoint_sha256": bundle["selected_checkpoint_sha256"], "training_seed": 13,
                    "source_snapshot_ref": str(snapshot_root.resolve()),
                    "source_snapshot_sha256": SNAPSHOT_SHA256,
                    "worker_ref": str(worker.resolve()), "worker_sha256": sha256_file(worker),
                    "parity_worker_ref": str(parity_worker.resolve()),
                    "parity_worker_sha256": sha256_file(parity_worker),
                    "views": ["identity", "deterministic_noise", "deterministic_fir"],
                    "view_index": {"z0": 0, "identity": 0, "deterministic_noise": 1, "deterministic_fir": 2},
                    "probe": probe, "probe_sha256": content_hash(probe), "numerical_mode": numerical,
                    "roles": role_records,
                    "source_artifacts": artifacts_plan,
                    "source_artifacts_sha256": sha256_file(artifacts_path),
                    "block_layout": {"fit": "99 full blocks + 36 tail = 100", "cal0": "19 full + 42 tail = 20",
                                     "select": "45 full, no tail"},
                    "commands_after_review_lock": commands,
                    "budget": {"gpu_hour_hard_cap": 1.0, "artifact_peak_gib_hard_cap": 2.0,
                               "bootstrap_repeats": 1000, "static_fit_steps": 200},
                    "forbidden": ["K_gt_0", "training", "SSL", "target_scoring", "R6_to_R9",
                                  "select_based_method_selection", "parameter_search"]}
        write_json_new(temporary / "proposal.json", proposal)
    return {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
            "proposal_ref": str((destination / "proposal.json").resolve()),
            "proposal_sha256": sha256_file(destination / "proposal.json"),
            "sample_counts": dict(ROLE_COUNTS), "gpu_started": False, "locks_published": False}


def lock_r5_stage2_proposal(proposal_ref, expected_sha256):
    import datetime
    proposal_path = Path(proposal_ref)
    if sha256_file(proposal_path) != expected_sha256:
        raise ContractError("R5 stage-2 proposal SHA-256 changed")
    proposal = read_json(proposal_path)
    if (proposal.get("schema_version") != "0.1.0" or proposal.get("status") != "PROPOSED" or
            proposal.get("approval_required") is not True or proposal.get("locks_published") is not False):
        raise ContractError("only a PROPOSED/approval_required stage-2 proposal may be locked")
    proposal_dir = proposal_path.parent
    locked_dir = proposal_dir / "locked"
    if locked_dir.exists():
        raise ContractError("locked proposal exists; overwrite is forbidden")
    role_records = proposal["roles"]
    locked_records = {}
    with AtomicDirectory(locked_dir) as temporary:
        for role in ROLES:
            record = role_records[role]
            source = proposal_dir / record["extraction_proposal_ref"]
            if sha256_file(source) != record["extraction_proposal_sha256"]:
                raise DataError(role + " extraction proposal changed")
            plan = read_json(source)
            if plan.get("status") != "PROPOSED" or plan.get("input_role") != role:
                raise ContractError("unexpected extraction proposal for " + role)
            locked = dict(plan)
            locked["status"] = "LOCKED"
            locked_path = temporary / (role + ".extraction.lock.json")
            write_json_new(locked_path, locked)
            locked_records[role] = {"ref": locked_path.name, "sha256": sha256_file(locked_path),
                                    "purpose": plan["purpose"], "input_role": plan["input_role"],
                                    "labels_sha256": record["labels_sha256"],
                                    "groups_sha256": record["groups_sha256"]}
        artifacts_source = proposal_dir / "source_artifacts.proposal.json"
        if sha256_file(artifacts_source) != proposal["source_artifacts_sha256"]:
            raise DataError("source artifacts proposal changed")
        artifacts_plan = read_json(artifacts_source)
        artifacts_plan["status"] = "LOCKED"
        artifacts_locked = temporary / "source_artifacts.lock.json"
        write_json_new(artifacts_locked, artifacts_plan)
        lock = {"schema_version": "0.1.0", "status": "LOCKED",
                "proposal_ref": str(proposal_path.resolve()), "proposal_sha256": expected_sha256,
                "proposal_id": proposal["proposal_id"],
                "parent_plan_sha256": proposal["parent_plan_sha256"],
                "stage1_lock_sha256": proposal["stage1_lock_sha256"],
                "bundle_sha256": proposal["bundle_sha256"],
                "checkpoint_sha256": proposal["checkpoint_sha256"],
                "source_snapshot_sha256": proposal["source_snapshot_sha256"],
                "worker_sha256": proposal["worker_sha256"],
                "parity_worker_sha256": proposal["parity_worker_sha256"],
                "probe_sha256": proposal["probe_sha256"],
                "approval": {"decision": "approved_lock_and_execute_stage2", "reviewer": "human_user",
                             "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "scope": "full_source_cache + cache_parity + source_resources"},
                "roles": locked_records,
                "source_artifacts_ref": artifacts_locked.name,
                "source_artifacts_sha256": sha256_file(artifacts_locked),
                "forbidden": proposal["forbidden"]}
        write_json_new(temporary / "lock.json", lock)
    return lock

"""Prepare a review-only R5 stage-1 source-cache proposal; never lock or execute it."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from eptta.config.schema import check
from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export


R5_PLAN_SHA256 = "07024939e2a156e681e006ace20b1b542b2cfde811c30e87d4977ffa3ecac79d"
BUNDLE_SHA256 = "d774f714c8a2b6ecdbe7147cb06d5d96e6563500f1b7eb8ec4224b8406319684"
EXPORT_MANIFEST_SHA256 = "8cd0c21d3d89b46dbf4d07f86df29bce8e18803b952ed58d5bb6441264210c56"
SNAPSHOT_SHA256 = "50d893131ce99752759fb14a121bd17c29a9782eddb0d8560ac458d336315fe0"
ROLES = ("fit", "cal0", "select")


def _jsonl_new(path, rows):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _order(sample_id):
    return hashlib.sha256(("r5-stage1-small-v1\0seed13\0" + sample_id).encode()).hexdigest()


def _alternating_extremes(rows):
    ordered = sorted(rows, key=lambda row: (row["audio_bytes"], _order(row["sample_id"])))
    result, left, right = [], 0, len(ordered) - 1
    while left <= right:
        result.append(ordered[left])
        left += 1
        if left <= right:
            result.append(ordered[right])
            right -= 1
    return result


def _round_robin(groups, count):
    queues = {key: _alternating_extremes(rows) for key, rows in groups.items()}
    keys, result, seen_speakers, index = sorted(queues), [], set(), 0
    while len(result) < count and any(queues.values()):
        key = keys[index % len(keys)]
        index += 1
        if not queues[key]:
            continue
        choice = next((i for i, row in enumerate(queues[key])
                       if row["speaker_id"] not in seen_speakers), 0)
        row = queues[key].pop(choice)
        result.append(row)
        seen_speakers.add(row["speaker_id"])
    if len(result) != count:
        raise DataError("R5 small proposal cannot select the requested fixed count")
    return result


def _unlabeled(row):
    return {key: row[key] for key in ("schema_version", "sample_id", "root_key", "audio_relpath",
                                      "input_sha256", "split_role")}


def _select_role(role, manifest_rows, metadata, data_roots, fixed_ids=None, per_class=64,
                 expected_attacks=("A01", "A02", "A03", "A04", "A05", "A06")):
    by_id = {row["sample_id"]: row for row in manifest_rows}
    if len(by_id) != len(manifest_rows):
        raise DataError("duplicate source manifest UID")
    enriched = []
    for sample_id, row in by_id.items():
        item = metadata.get(sample_id)
        if (item is None or item.get("split_role") != role or
                ("canonical_label" in row and item.get("canonical_label") != row.get("canonical_label"))):
            raise DataError("canonical/source manifest mismatch for " + sample_id)
        path = Path(data_roots[row["root_key"]]) / row["audio_relpath"]
        if not path.is_file():
            raise DataError("source audio missing for proposal: " + sample_id)
        enriched.append({**row, "canonical_label": item.get("canonical_label"),
                         "speaker_id": item.get("speaker_id"),
                         "attack_id": item.get("generator_id"), "audio_bytes": path.stat().st_size})
    enriched_by_id = {row["sample_id"]: row for row in enriched}
    if fixed_ids is not None:
        if (len(fixed_ids) != 2 * per_class or len(set(fixed_ids)) != len(fixed_ids) or
                not set(fixed_ids).issubset(enriched_by_id)):
            raise ContractError("R4 parity sample cannot be reused as the R5 fit proposal")
        selected = [enriched_by_id[sample_id] for sample_id in fixed_ids]
    else:
        groups = {0: defaultdict(list), 1: defaultdict(list)}
        for row in enriched:
            label = int(row["canonical_label"])
            key = row["speaker_id"] if label == 0 else row["attack_id"]
            if not key:
                if expected_attacks is not None:
                    raise DataError("required speaker/attack metadata is missing")
                key = "metadata-unavailable:" + row["sample_id"]
            groups[label][key].append(row)
        selected = _round_robin(groups[0], per_class) + _round_robin(groups[1], per_class)
        selected.sort(key=lambda row: _order(row["sample_id"]))
    counts = {"bonafide": sum(row["canonical_label"] == 0 for row in selected),
              "spoof": sum(row["canonical_label"] == 1 for row in selected)}
    attacks = sorted({row["attack_id"] for row in selected if row["canonical_label"] == 1})
    if counts != {"bonafide": per_class, "spoof": per_class}:
        raise ContractError("R5 small proposal lacks the approved balanced parity budget")
    if expected_attacks is not None and attacks != list(expected_attacks):
        raise ContractError("R5 small proposal differs from approved attack/group coverage")
    return selected


def prepare_r5_stage1_proposal(plan_ref, expected_plan_sha256, bundle_ref, snapshot_ref, output):
    plan_path, bundle_path, snapshot_root = Path(plan_ref), Path(bundle_ref), Path(snapshot_ref)
    if sha256_file(plan_path) != expected_plan_sha256:
        raise ContractError("R5 source-preparation plan SHA-256 changed")
    legacy = plan_path.suffix.lower() != ".json"
    plan = {} if legacy else read_json(plan_path)
    bundle_sha256 = sha256_file(bundle_path)
    export_sha256 = sha256_file(bundle_path.parent / "export_manifest.json")
    if not legacy and (plan.get("bundle_sha256") != bundle_sha256 or
                       plan.get("export_manifest_sha256") != export_sha256):
        raise ContractError("R5 plan does not approve this frozen bundle/export")
    bundle, _manifest, _parity, _selection = verify_frozen_export(bundle_path)
    snapshot = read_json(snapshot_root / "snapshot.json")
    snapshot_sha256 = snapshot.get("canonical_sha256")
    if (snapshot.get("status") != "LOCKED" or sha256_file(snapshot_root / snapshot["canonical_ref"]) !=
            snapshot_sha256 or (not legacy and plan.get("source_snapshot_sha256") != snapshot_sha256)):
        raise ContractError("approved source snapshot identity changed")
    canonical = {role: {} for role in ROLES}
    for row in iter_jsonl(snapshot_root / snapshot["canonical_ref"]):
        if row.get("split_role") in canonical:
            canonical[row["split_role"]][row["sample_id"]] = row
    data_roots = ({"asvspoof2019_la": "/media/dell/data/fakedata/asvspoof2019/LA"} if legacy else
                  plan.get("data_roots"))
    if not isinstance(data_roots, dict) or not data_roots:
        raise ContractError("R5 plan must bind reviewed data_roots")
    r4_fit = read_json(bundle_path.parent / "fit128_uids.json")
    fixed_fit_ids = [item["sample_id"] for item in r4_fit["uids"]]
    policy = bundle["r4_validation"].get("parity_policy") or {
        "per_class_budget": 64, "expected_attack_ids": ["A01", "A02", "A03", "A04", "A05", "A06"]}
    per_class = policy["per_class_budget"]
    expected_attacks = policy.get("expected_attack_ids")
    destination = Path(output).resolve()
    worker = Path(__file__).parents[3] / "workers/baseline_bridge.py"
    probe = {"num_views": 3, "seed": 13, "noise_snr_db": 30.0, "fir_side_gain": 0.05}
    numerical = {"dtype": "float32", "tf32": False, "block_units": 256}
    with AtomicDirectory(destination) as temporary:
        role_records = {}
        for role in ROLES:
            manifest = snapshot_root / "manifests" / (role + ".jsonl")
            expected = snapshot["role_manifest_hashes"][role + ".jsonl"]
            if sha256_file(manifest) != expected:
                raise DataError(role + " manifest changed")
            rows = list(iter_jsonl(manifest))
            selected = _select_role(role, rows, canonical[role], data_roots,
                                    fixed_fit_ids if role == "fit" else None, per_class, expected_attacks)
            unlabeled_path = temporary / (role + ".unlabeled.jsonl")
            evidence_path = temporary / (role + ".selection.json")
            plan_path_out = temporary / (role + ".extraction.proposal.json")
            _jsonl_new(unlabeled_path, [_unlabeled(row) for row in selected])
            evidence = {"schema_version": "0.1.0", "status": "PROPOSED",
                        "coordinator_only_annotations": True, "role": role, "count": 2 * per_class,
                        "selection_rule": ("reuse_R4_v4_fit128_exact_UID_order" if role == "fit" else
                            "64_per_class; metadata_only; seed13 UID hash order; bonafide speaker and spoof generator_id strata; alternating source-file byte-length extremes; no scores"),
                        "class_counts": {"bonafide": per_class, "spoof": per_class},
                        "attack_field": "generator_id",
                        "attack_ids": sorted({row["attack_id"] for row in selected
                                              if row["canonical_label"] == 1 and row["attack_id"] is not None}),
                        "speaker_count": len({row["speaker_id"] for row in selected}),
                        "audio_bytes_min": min(row["audio_bytes"] for row in selected),
                        "audio_bytes_max": max(row["audio_bytes"] for row in selected),
                        "uids": [{"sample_id": row["sample_id"], "canonical_label": row["canonical_label"],
                                  "speaker_id": row["speaker_id"], "attack_id": row["attack_id"],
                                  "audio_bytes": row["audio_bytes"]} for row in selected]}
            write_json_new(evidence_path, evidence)
            purpose = "select" if role == "select" else "source_prepare"
            extraction = {"schema_version": "0.1.0", "status": "PROPOSED", "purpose": purpose,
                          "input_role": role, "frozen_bundle_ref": str(bundle_path.resolve()),
                          "manifest_ref": str((destination / unlabeled_path.name).resolve()),
                          "manifest_sha256": sha256_file(unlabeled_path), "data_roots": data_roots,
                          "probe": probe, "numerical_mode": numerical,
                          "output_root": str((destination / "caches" / role).resolve())}
            write_json_new(plan_path_out, extraction)
            role_records[role] = {"purpose": purpose, "count": 2 * per_class,
                                  "unlabeled_manifest_ref": unlabeled_path.name,
                                  "unlabeled_manifest_sha256": sha256_file(unlabeled_path),
                                  "selection_evidence_ref": evidence_path.name,
                                  "selection_evidence_sha256": sha256_file(evidence_path),
                                  "extraction_proposal_ref": plan_path_out.name,
                                  "extraction_proposal_sha256": sha256_file(plan_path_out)}
        root_text = "/media/dell/data/fakeAudioDection/TTA"
        out_text = str(destination)
        commands = {role: ("env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "
                    "/home/dell/anaconda3/envs/py310/bin/python -m eptta.cli --config configs/base.yaml "
                    "--profile remote_a6000 --paths configs/paths.fakedata.private.yaml extract --plan "
                    + out_text + "/locked/" + role + ".extraction.lock.json --worker-slot 0 --worker-count 1 "
                    "--worker-python /home/dell/anaconda3/envs/py38/bin/python") for role in ROLES}
        proposal = {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
                    "locks_published": False, "proposal_id": "r5-stage1-aasist-small-20260917",
                    "parent_plan_ref": str(Path(plan_ref).resolve()), "parent_plan_sha256": expected_plan_sha256,
                    "bundle_ref": str(bundle_path.resolve()), "bundle_sha256": bundle_sha256,
                    "bundle_id": bundle["baseline_id"], "export_manifest_sha256": export_sha256,
                    "checkpoint_sha256": bundle["selected_checkpoint_sha256"], "training_seed": 13,
                    "source_snapshot_ref": str(snapshot_root.resolve()), "source_snapshot_sha256": snapshot_sha256,
                    "worker_ref": str(worker.resolve()), "worker_sha256": sha256_file(worker),
                    "views": ["identity", "deterministic_noise", "deterministic_fir"],
                    "view_index": {"z0": 0, "identity": 0, "deterministic_noise": 1,
                                   "deterministic_fir": 2},
                    "probe": probe, "probe_sha256": content_hash(probe), "numerical_mode": numerical,
                    "transform_contract": {"noise_seed": "sha256(probe_seed + NUL + sample_id) first64bits mod (2^63-1)",
                        "noise": "CPU torch.Generator; RMS-scaled Gaussian at noise_snr_db; independent of order/batch/shard",
                        "fir": "reflect-pad one sample; conv1d kernel [fir_side_gain,1,-fir_side_gain]",
                        "view_order": "stack(identity,noise,fir)", "z0_storage": "view_index_0_not_a_fourth_copy"},
                    "roles": role_records, "commands_after_review_lock": commands,
                    "real_block_coverage": "one tail block per 128-sample role because block_units=256; full-block branch requires synthetic test",
                    "implementation_gaps_before_lock": [
                        "cache parity runner must independently re-encode the exact deterministic views and read cache from disk",
                        "production K=0 currently requires FrozenResources before its legal identity shortcut; add a strict resource-free K=0-only entrypoint while retaining K>0 resource gates",
                        "stage timing must separately record decode/view/encode/write/read and peak CUDA memory"],
                    "budget": {"gpu_hour_hard_cap": 0.25, "artifact_peak_gib_hard_cap": 0.25,
                               "extraction_timeout_seconds_each": 180, "parity_timeout_seconds": 180,
                               "planned_gpu_wall_seconds": 720,
                               "raw_array_bytes_per_role": 128 * 3 * 160 * 4,
                               "raw_array_bytes_all_roles": 3 * 128 * 3 * 160 * 4},
                    "forbidden": ["full_role_cache", "U", "M", "Fisher", "static_R", "tau0", "K_gt_0",
                                  "training", "SSL", "target_scoring", "R6_to_R9"]}
        check(proposal, "r5_stage1_proposal")
        write_json_new(temporary / "proposal.json", proposal)
    return {"schema_version": "0.1.0", "status": "PROPOSED", "approval_required": True,
            "proposal_ref": str((destination / "proposal.json").resolve()),
            "proposal_sha256": sha256_file(destination / "proposal.json"),
            "sample_counts": {role: 2 * per_class for role in ROLES}, "gpu_started": False,
            "formal_cache_written": False, "locks_published": False}

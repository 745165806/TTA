"""Pickle-free EP source-resource bundle writer/loader."""
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export


ARRAY_NAMES = ("U", "U_feature_pca", "U_random_0", "U_random_1", "U_random_2", "w",
               "anchors_z", "anchors_y", "anchors_m0", "anchors_s0", "fisher", "fixed_R")


def write_frozen_resources(output, baseline_id, selected_checkpoint_sha256, arrays, scalars,
                           source_snapshot_hash, source_role="fit", diagnostics=None):
    import numpy as np
    required = {"U", "w", "anchors_z", "anchors_y", "anchors_m0", "anchors_s0"}
    if not required.issubset(arrays) or set(arrays) - set(ARRAY_NAMES):
        raise ContractError("source resource arrays are incomplete or unknown")
    if set(scalars) != {"b", "tau0"}:
        raise ContractError("source resource scalars must be b/tau0")
    with AtomicDirectory(output) as temporary:
        files = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise DataError("resource arrays must be finite and pickle-free")
            path = temporary / (name + ".npy")
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            files[name] = {"ref": path.name, "sha256": sha256_file(path), "shape": list(array.shape),
                           "dtype": str(array.dtype)}
        basis = {"baseline_id": baseline_id, "checkpoint": selected_checkpoint_sha256,
                 "source_snapshot_hash": source_snapshot_hash,
                 "files": {name: item["sha256"] for name, item in files.items()}, "scalars": scalars}
        metadata = {"schema_version": "0.1.0", "status": "LOCKED",
                    "artifact_bundle_id": "ep-resources-" + content_hash(basis)[:20],
                    "baseline_id": baseline_id, "selected_checkpoint_sha256": selected_checkpoint_sha256,
                    "source_snapshot_hash": source_snapshot_hash, "source_role": source_role,
                    "files": files, "scalars": {key: float(value) for key, value in scalars.items()},
                    "allow_pickle": False, "immutable": True}
        if diagnostics:
            metadata["diagnostics"] = diagnostics
        write_json_new(temporary / "resources.json", metadata)
    return metadata


def load_frozen_resources(resource_ref, bundle):
    import numpy as np
    import torch
    path = Path(resource_ref)
    metadata_path = path / "resources.json" if path.is_dir() else path
    metadata = read_json(metadata_path)
    if metadata.get("status") != "LOCKED" or metadata.get("allow_pickle") is not False:
        raise DataError("source resource bundle is not locked/pickle-free")
    if metadata.get("baseline_id") != bundle.get("baseline_id") or metadata.get(
            "selected_checkpoint_sha256") != bundle.get("selected_checkpoint_sha256"):
        raise ContractError("source resources and frozen detector identities differ")
    arrays = {}
    for name, item in metadata["files"].items():
        array_path = metadata_path.parent / item["ref"]
        if sha256_file(array_path) != item["sha256"]:
            raise DataError("source resource changed: %s" % name)
        with array_path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
        if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
            raise DataError("source resource metadata mismatch: %s" % name)
        arrays[name] = torch.from_numpy(value)
    from eptta.adaptation.types import FrozenResources
    resources = FrozenResources(arrays["U"], arrays["w"], metadata["scalars"]["b"], arrays["anchors_z"],
                                arrays["anchors_y"], arrays["anchors_m0"], arrays["anchors_s0"],
                                metadata["scalars"]["tau0"], metadata["artifact_bundle_id"],
                                source_role=metadata["source_role"], channel="frozen_bundle")
    extras = {name: value for name, value in arrays.items() if name not in
              ("U", "w", "anchors_z", "anchors_y", "anchors_m0", "anchors_s0")}
    return resources, extras, metadata


def _labels(path):
    from eptta.data.io import iter_jsonl
    result = {}
    for row in iter_jsonl(path):
        if set(row) != {"schema_version", "sample_id", "canonical_label"} or row["canonical_label"] not in (0, 1):
            raise DataError("source artifact label sidecar is invalid")
        if row["sample_id"] in result:
            raise DataError("duplicate source artifact label")
        result[row["sample_id"]] = row["canonical_label"]
    return result


def _groups(path):
    from eptta.data.io import iter_jsonl
    result = {}
    for row in iter_jsonl(path):
        if set(row) != {"schema_version", "sample_id", "source_group_id"} or not row["source_group_id"]:
            raise DataError("source artifact group sidecar is invalid")
        if row["sample_id"] in result:
            raise DataError("duplicate source artifact group")
        result[row["sample_id"]] = row["source_group_id"]
    return result


def build_source_resources(plan_ref, fit_cache_ref, output):
    """Build response/PCA/random U, tau0, M, Fisher and fixed R from allowed source roles."""
    import numpy as np
    import torch
    from eptta.cache.reader import FeatureCache
    from eptta.offline.anchors import build_anchor_memory
    from eptta.offline.calibration import empirical_real_quantile
    from eptta.offline.fisher import empirical_diagonal_fisher
    from eptta.offline.static_adapter import fit_fixed_source_adapter
    from eptta.offline.subspace import (balanced_response_subspace, feature_pca_subspace,
                                        random_subspace)
    plan = read_json(plan_ref)
    required = {"schema_version", "status", "frozen_bundle_ref", "fit_role", "fit_manifest_sha256",
                "fit_labels_ref", "fit_labels_sha256", "fit_groups_ref", "fit_groups_sha256",
                "calibration_role", "calibration_manifest_sha256", "calibration_cache_ref",
                "calibration_labels_ref", "calibration_labels_sha256", "source_snapshot_hash",
                "rank", "alpha_cal", "anchor_per_class", "seed", "random_seeds",
                "treatment_families", "samples_per_group", "pair_seed", "margin_bins",
                "margin_epsilon", "minimum_cal0_bonafide", "fixed_adapter"}
    if set(plan) != required or plan.get("status") != "LOCKED" or plan.get("random_seeds") is None or len(
            plan["random_seeds"]) != 3:
        raise ContractError("source artifact plan must be strict, LOCKED, and contain three random seeds")
    if plan["fit_role"] != "fit" or plan["calibration_role"] != "cal0":
        raise ContractError("source artifacts may use labels only from fit and cal0")
    if type(plan["treatment_families"]) is not list or len(plan["treatment_families"]) != 2:
        raise ContractError("treatment families must list exactly two probe families (noise, fir)")
    if type(plan["samples_per_group"]) is not int or plan["samples_per_group"] < 1:
        raise ContractError("samples_per_group must be a positive integer")
    if type(plan["minimum_cal0_bonafide"]) is not int or plan["minimum_cal0_bonafide"] < 2:
        raise ContractError("minimum_cal0_bonafide must be at least 2")
    for ref_field, hash_field in (("fit_labels_ref", "fit_labels_sha256"),
                                  ("fit_groups_ref", "fit_groups_sha256"),
                                  ("calibration_labels_ref", "calibration_labels_sha256")):
        if sha256_file(plan[ref_field]) != plan[hash_field]:
            raise DataError("source artifact sidecar changed after lock: %s" % ref_field)
    bundle_path = Path(plan["frozen_bundle_ref"])
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_path)
    fit_cache = FeatureCache(fit_cache_ref)
    cal_cache = FeatureCache(plan["calibration_cache_ref"])
    if fit_cache.index["identity"]["input_manifest_sha256"] != plan["fit_manifest_sha256"] or cal_cache.index[
            "identity"]["input_manifest_sha256"] != plan["calibration_manifest_sha256"]:
        raise ContractError("source cache manifests differ from the locked fit/cal0 plan")
    if plan["source_snapshot_hash"] != bundle["fit_snapshot_hash"]:
        raise ContractError("source artifact snapshot differs from the trained frozen detector")
    for cache in (fit_cache, cal_cache):
        if cache.index["identity"]["baseline_id"] != bundle["baseline_id"] or cache.index["identity"][
                "selected_checkpoint_sha256"] != bundle["selected_checkpoint_sha256"]:
            raise ContractError("source cache and frozen bundle identity mismatch")
    fit = fit_cache.load_by_id()
    cal = cal_cache.load_by_id()
    fit_labels = _labels(plan["fit_labels_ref"])
    cal_labels = _labels(plan["calibration_labels_ref"])
    fit_groups = _groups(plan["fit_groups_ref"])
    if set(fit) != set(fit_labels) or set(cal) != set(cal_labels) or set(fit) != set(fit_groups):
        raise DataError("source feature/label/group ID coverage mismatch")
    fit_ids, cal_ids = sorted(fit), sorted(cal)
    fit_views = torch.from_numpy(np.stack([fit[key] for key in fit_ids]))
    fit_y = torch.tensor([fit_labels[key] for key in fit_ids], dtype=torch.long)
    fit_group_ids = [fit_groups[key] for key in fit_ids]
    cal_z = torch.from_numpy(np.stack([cal[key][0] for key in cal_ids]))
    cal_y = torch.tensor([cal_labels[key] for key in cal_ids], dtype=torch.long)
    head = torch.load(bundle_path.parent / bundle["head_ref"], map_location="cpu", weights_only=True)
    w, b = head["w"].to(fit_views.dtype), float(head["b"])
    cal_scores = cal_z @ w + b
    if int((cal_y == 0).sum()) < plan["minimum_cal0_bonafide"]:
        raise DataError("cal0 has too few bonafide samples for tau0")
    tau0 = empirical_real_quantile(cal_scores, cal_y, plan["alpha_cal"])
    rank = plan["rank"]
    U, sub_diag = balanced_response_subspace(
        fit_views, fit_y, fit_group_ids, rank, tuple(plan["treatment_families"]),
        plan["samples_per_group"], plan["pair_seed"])
    U_pca = feature_pca_subspace(fit_views[:, 0], rank)
    random_values = [random_subspace(fit_views.shape[-1], rank, seed, dtype=fit_views.dtype)
                     for seed in plan["random_seeds"]]
    memory = build_anchor_memory(fit_views[:, 0], fit_y, w, b, tau0, plan["anchor_per_class"],
                                 plan["seed"], plan["margin_bins"], plan["margin_epsilon"])
    from eptta.adaptation.types import FrozenResources
    resources = FrozenResources(U, w, b, memory["anchors_z"], memory["anchors_y"], memory["anchors_m0"],
                                memory["anchors_s0"], tau0, "build-pending")
    fisher = empirical_diagonal_fisher(memory["anchors_z"], memory["anchors_y"], U, w, b)
    fixed = plan["fixed_adapter"]
    fixed_R, _trace = fit_fixed_source_adapter(fit_views, resources, fixed["steps"], fixed["lr"],
                                                fixed["rho"], fixed["gamma"], fixed["lambda_keep"])
    diagnostics = {"tau0": float(tau0), "cal0_bonafide_count": int((cal_y == 0).sum()),
                   "cal0_spoof_count": int((cal_y == 1).sum()), "alpha_cal": float(plan["alpha_cal"]),
                   "subspace": sub_diag,
                   "orthogonality_error": float((U.T @ U - torch.eye(rank, dtype=U.dtype)).abs().max()),
                   "anchor_margins_min": float(memory["anchors_m0"].min()),
                   "anchor_margins_max": float(memory["anchors_m0"].max()),
                   "fisher_min": float(fisher.min()), "fisher_max": float(fisher.max()),
                   "fisher_mean": float(fisher.mean()),
                   "fixed_R_fro": float(torch.linalg.vector_norm(fixed_R))}
    arrays = {"U": U.numpy(), "U_feature_pca": U_pca.numpy(), "w": w.numpy(),
              "anchors_z": memory["anchors_z"].numpy(), "anchors_y": memory["anchors_y"].numpy(),
              "anchors_m0": memory["anchors_m0"].numpy(), "anchors_s0": memory["anchors_s0"].numpy(),
              "fisher": fisher.numpy(), "fixed_R": fixed_R.numpy()}
    arrays.update(("U_random_%d" % index, value.numpy()) for index, value in enumerate(random_values))
    return write_frozen_resources(output, bundle["baseline_id"], bundle["selected_checkpoint_sha256"],
                                  arrays, {"b": b, "tau0": tau0}, plan["source_snapshot_hash"],
                                  diagnostics=diagnostics)

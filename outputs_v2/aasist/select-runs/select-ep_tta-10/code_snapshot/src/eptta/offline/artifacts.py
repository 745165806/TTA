"""Pickle-free EP source-resource bundle writer/loader."""
from pathlib import Path

from eptta.data.io import AtomicDirectory, read_json, write_json_new
from eptta.errors import ContractError, DataError
from eptta.models.frozen import verify_frozen_export


ARRAY_NAMES = ("U", "U_feature_pca", "U_random_0", "U_random_1", "U_random_2", "w",
               "anchors_z", "anchors_y", "anchors_m0", "anchors_s0", "fisher", "fixed_R")


def write_frozen_resources(output, bundle, arrays, scalars, source_role="fit", diagnostics=None):
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
            files[name] = {"ref": path.name, "shape": list(array.shape), "dtype": str(array.dtype)}
        metadata = {"schema_version": "0.2.0", "status": "READY",
                    "artifact_bundle_id": bundle["baseline_id"] + ":source-resources",
                    "baseline_id": bundle["baseline_id"], "source_run_id": bundle["source_run_id"],
                    "checkpoint_ref": bundle["checkpoint_ref"], "source_role": source_role,
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
    if metadata.get("allow_pickle") is not False:
        raise DataError("source resource bundle must disable pickle")
    current = metadata.get("schema_version") == "0.2.0"
    if current:
        if metadata.get("status") != "READY":
            raise DataError("source resource bundle is incomplete")
        if (metadata.get("baseline_id") != bundle.get("baseline_id") or
                metadata.get("source_run_id") != bundle.get("source_run_id") or
                Path(metadata.get("checkpoint_ref", "")).resolve() != Path(bundle["checkpoint_ref"]).resolve()):
            raise ContractError("source resources and frozen detector references differ")
    else:
        if metadata.get("status") != "LOCKED" or metadata.get("baseline_id") != bundle.get("baseline_id"):
            raise ContractError("historical source resources do not name this frozen detector")
    arrays = {}
    for name, item in metadata["files"].items():
        array_path = metadata_path.parent / item["ref"]
        if not array_path.is_file():
            raise DataError("source resource is missing: %s" % name)
        with array_path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
        if list(value.shape) != item["shape"] or str(value.dtype) != item["dtype"]:
            raise DataError("source resource metadata mismatch: %s" % name)
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise DataError("source resource is object-valued or non-finite: %s" % name)
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
    """Build response/PCA/random U, tau0, anchors, Fisher and fixed R from source roles."""
    import numpy as np
    import torch
    from eptta.cache.reader import FeatureCache
    from eptta.offline.anchors import build_anchor_memory
    from eptta.offline.calibration import empirical_real_quantile
    from eptta.offline.fisher import empirical_diagonal_fisher
    from eptta.offline.static_adapter import fit_fixed_source_adapter
    from eptta.offline.subspace import balanced_response_subspace, feature_pca_subspace, random_subspace
    plan = read_json(plan_ref)
    required = {"schema_version", "status", "frozen_bundle_ref", "fit_role", "fit_manifest_ref",
                "fit_labels_ref", "fit_groups_ref", "calibration_role", "calibration_manifest_ref",
                "calibration_cache_ref", "calibration_labels_ref", "rank", "alpha_cal",
                "anchor_per_class", "seed", "random_seeds", "treatment_families",
                "samples_per_group", "pair_seed", "margin_bins", "margin_epsilon",
                "minimum_cal0_bonafide", "fixed_adapter"}
    if (set(plan) != required or plan.get("schema_version") != "0.3.0" or
            plan.get("status") != "READY" or not isinstance(plan.get("random_seeds"), list) or
            len(plan["random_seeds"]) != 3):
        raise ContractError("source artifact plan is invalid")
    if plan["fit_role"] != "fit" or plan["calibration_role"] != "cal0":
        raise ContractError("source artifacts may use labels only from fit and cal0")
    if type(plan["treatment_families"]) is not list or len(plan["treatment_families"]) != 2:
        raise ContractError("treatment families must list noise and FIR")
    if type(plan["samples_per_group"]) is not int or plan["samples_per_group"] < 1:
        raise ContractError("samples_per_group must be positive")
    if type(plan["minimum_cal0_bonafide"]) is not int or plan["minimum_cal0_bonafide"] < 2:
        raise ContractError("minimum_cal0_bonafide must be at least 2")
    for field in ("fit_manifest_ref", "fit_labels_ref", "fit_groups_ref",
                  "calibration_manifest_ref", "calibration_labels_ref"):
        if not Path(plan[field]).is_file():
            raise DataError("source artifact input is missing: %s" % field)
    bundle_path = Path(plan["frozen_bundle_ref"])
    bundle, _export, _parity, _selection = verify_frozen_export(bundle_path)
    fit_cache, cal_cache = FeatureCache(fit_cache_ref), FeatureCache(plan["calibration_cache_ref"])
    for cache, manifest, role in ((fit_cache, plan["fit_manifest_ref"], "fit"),
                                  (cal_cache, plan["calibration_manifest_ref"], "cal0")):
        identity = cache.index.get("identity", {})
        if cache.index.get("format") == "sharded_npy_v2" and (
                Path(identity.get("manifest_ref", "")).resolve() != Path(manifest).resolve() or
                identity.get("split_role") != role or identity.get("source_run_id") != bundle["source_run_id"] or
                Path(identity.get("checkpoint_ref", "")).resolve() != Path(bundle["checkpoint_ref"]).resolve()):
            raise ContractError("source cache declared provenance differs from the resource plan")
        if identity.get("baseline_id") and identity["baseline_id"] != bundle["baseline_id"]:
            raise ContractError("historical source cache and bundle identity differ")
    if fit_cache.index.get("format") == cal_cache.index.get("format") == "sharded_npy_v2":
        fit_identity, cal_identity = fit_cache.index["identity"], cal_cache.index["identity"]
        shared = ("source_run_id", "checkpoint_ref", "preprocess", "views", "seed", "dtype",
                  "numerical_mode")
        if any(fit_identity.get(field) != cal_identity.get(field) for field in shared):
            raise ContractError("fit and cal0 caches use different model/view/numerical settings")
    fit, cal = fit_cache.load_by_id(), cal_cache.load_by_id()
    fit_labels, cal_labels, fit_groups = (_labels(plan["fit_labels_ref"]),
                                          _labels(plan["calibration_labels_ref"]),
                                          _groups(plan["fit_groups_ref"]))
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
    U, sub_diag = balanced_response_subspace(fit_views, fit_y, fit_group_ids, rank,
                                              tuple(plan["treatment_families"]),
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
                   "fisher_mean": float(fisher.mean()), "fixed_R_fro": float(torch.linalg.vector_norm(fixed_R))}
    arrays = {"U": U.numpy(), "U_feature_pca": U_pca.numpy(), "w": w.numpy(),
              "anchors_z": memory["anchors_z"].numpy(), "anchors_y": memory["anchors_y"].numpy(),
              "anchors_m0": memory["anchors_m0"].numpy(), "anchors_s0": memory["anchors_s0"].numpy(),
              "fisher": fisher.numpy(), "fixed_R": fixed_R.numpy()}
    arrays.update(("U_random_%d" % index, value.numpy()) for index, value in enumerate(random_values))
    return write_frozen_resources(output, bundle, arrays, {"b": b, "tau0": tau0}, diagnostics=diagnostics)

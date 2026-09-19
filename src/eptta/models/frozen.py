"""Read-only semantic checks for frozen detector exports."""
from pathlib import Path

from eptta.config.schema import read_document
from eptta.errors import ContractError, DataError


def _contained_file(root, relative, field):
    if not isinstance(relative, str) or not relative:
        raise ContractError("frozen bundle %s is missing" % field)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError("frozen bundle %s escapes its export directory" % field) from exc
    if not candidate.is_file():
        raise DataError("frozen export file is missing: %s" % field)
    return candidate


def verify_frozen_export(bundle_ref):
    """Validate structure/parity without computing a project content digest.

    Historical bundles are opened read-only and normalized in memory. Their
    stored digest fields remain historical metadata and are not trusted or
    recomputed. Explicit paths plus strict state-dict loading provide migration
    compatibility; byte-level replacement detection is intentionally unavailable.
    """
    bundle_path = Path(bundle_ref).resolve()
    bundle = read_document(bundle_path)
    root = bundle_path.parent
    current = bundle.get("schema_version") == "0.3.0"
    if current:
        required = {"schema_version", "model_id", "baseline_id", "source_run_id",
                    "checkpoint_ref", "epoch", "class_index_map", "head_ref",
                    "detector_state_ref", "embedding_dim", "task_weight_origin",
                    "training_phase", "source_val_selection_ref", "preprocess",
                    "initialization", "parity_report_ref"}
        if set(bundle) != required:
            raise ContractError("frozen bundle has missing/unknown fields")
        if (bundle["task_weight_origin"] != "trained_in_project" or bundle["training_phase"] != "full" or
                bundle["class_index_map"].get("bonafide") == bundle["class_index_map"].get("spoof")):
            raise ContractError("frozen bundle training/class semantics are invalid")
        checkpoint = Path(bundle["checkpoint_ref"])
        if not checkpoint.is_file() or checkpoint.name in ("best.pt", "last.pt"):
            raise DataError("frozen bundle must reference an existing concrete epoch checkpoint")
    else:
        # Minimal read compatibility for completed v0.1 exports.
        required = {"model_id", "baseline_id", "training_run_id", "class_index_map",
                    "head_ref", "embedding_dim", "task_weight_origin", "training_phase",
                    "source_val_selection_ref", "parity_report_ref"}
        if not required.issubset(bundle):
            raise ContractError("unsupported historical frozen bundle")
        selection = read_document(bundle["source_val_selection_ref"])
        checkpoint_ref = selection.get("selected_checkpoint_ref")
        if not checkpoint_ref or not Path(checkpoint_ref).is_file():
            raise DataError("historical bundle no longer has its selected checkpoint")
        bundle = {**bundle, "source_run_id": bundle["training_run_id"],
                  "checkpoint_ref": checkpoint_ref, "epoch": selection.get("selected_epoch"),
                  "detector_state_ref": "detector_state.pt",
                  "preprocess": (bundle.get("model_contract") or {}).get("preprocess", {}),
                  "initialization": bundle.get("init_provenance")}
    _contained_file(root, bundle["detector_state_ref"], "detector_state_ref")
    _contained_file(root, bundle["head_ref"], "head_ref")
    parity = read_document(_contained_file(root, bundle["parity_report_ref"], "parity_report_ref"))
    if (parity.get("status") != "PASS" or parity.get("module_modes_stable") is not True or
            parity.get("buffers_stable") is not True):
        raise ContractError("frozen wrapper/head parity is missing or failed")
    selection_path = Path(bundle["source_val_selection_ref"])
    if not selection_path.is_file():
        raise DataError("source_val selection record is missing")
    selection = read_document(selection_path)
    selected_epoch = selection.get("selected_epoch")
    if selected_epoch is not None and bundle.get("epoch") is not None and selected_epoch != bundle["epoch"]:
        raise ContractError("frozen bundle and source_val selection name different epochs")
    manifest_path = root / "export_manifest.json"
    manifest = read_document(manifest_path) if manifest_path.is_file() else None
    return bundle, manifest, parity, selection

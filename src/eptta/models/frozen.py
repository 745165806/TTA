"""Read-only eligibility checks for production frozen detector exports."""
from pathlib import Path

from eptta.config.schema import read_document
from eptta.data.io import sha256_file
from eptta.errors import ContractError, DataError
from eptta.models.contracts import FrozenModelBundle


def _contained_file(root, relative, field):
    if not isinstance(relative, str) or not relative:
        raise ContractError("frozen bundle %s is missing" % field)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError("frozen bundle %s escapes the export directory" % field) from exc
    if not candidate.is_file():
        raise DataError("frozen export file is missing: %s" % field)
    return candidate


def verify_frozen_export(bundle_ref):
    """Require full in-project training, source_val selection and parity PASS.

    R5 and every later consumer call this function.  A bundle containing only
    plausible metadata is not eligible: every exported file is hash checked,
    the finalized source-selection record is re-opened, and the parity report
    must explicitly record stable module modes and buffers.
    """
    bundle_path = Path(bundle_ref).resolve()
    bundle = read_document(bundle_path)
    try:
        FrozenModelBundle(**bundle)
    except TypeError as exc:
        raise ContractError("frozen bundle fields do not match v0.1.0") from exc

    root = bundle_path.parent
    manifest_path = _contained_file(root, "export_manifest.json", "export_manifest")
    manifest = read_document(manifest_path)
    if (manifest.get("schema_version") != "0.1.0" or manifest.get("status") != "LOCKED" or
            manifest.get("immutable") is not True or not isinstance(manifest.get("files"), dict)):
        raise ContractError("frozen export manifest is not a LOCKED immutable v0.1.0 document")
    required_files = {"bundle.json", "detector_state.pt", bundle["head_ref"],
                      bundle["parity_report_ref"]}
    if not required_files.issubset(manifest["files"]):
        raise DataError("frozen export manifest omits a required artifact")
    for relative, digest in manifest["files"].items():
        path = _contained_file(root, relative, "manifest file")
        if sha256_file(path) != digest:
            raise DataError("frozen export file changed: %s" % relative)
    if manifest["files"]["bundle.json"] != sha256_file(bundle_path):
        raise DataError("frozen bundle changed after export")

    parity_path = _contained_file(root, bundle["parity_report_ref"], "parity_report_ref")
    parity = read_document(parity_path)
    if (parity.get("schema_version") != "0.1.0" or parity.get("status") != "PASS" or
            parity.get("module_modes_stable") is not True or parity.get("buffers_stable") is not True):
        raise ContractError("R5 requires frozen wrapper/head parity PASS with stable modes/buffers")

    selection_path = Path(bundle["source_val_selection_ref"]).resolve()
    if not selection_path.is_file():
        raise DataError("source_val selection record is missing")
    selection = read_document(selection_path)
    if bundle["task_training_provenance"].get("source_val_selection_sha256") != sha256_file(selection_path):
        raise DataError("source_val selection record changed after frozen export")
    expected = {"status": "FINALIZED", "training_phase": "full",
                "task_weight_origin": "trained_in_project",
                "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                "training_run_id": bundle["training_run_id"],
                "fit_snapshot_hash": bundle["fit_snapshot_hash"],
                "source_val_snapshot_hash": bundle["source_val_snapshot_hash"],
                "recipe_hash": bundle["recipe_hash"]}
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ContractError("frozen bundle disagrees with finalized source_val selection")
    return bundle, manifest, parity, selection

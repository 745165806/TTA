"""Source-val model selection and frozen-export eligibility."""
from pathlib import Path

from eptta.data.io import iter_jsonl, read_json, write_json_new
from eptta.errors import ContractError, DataError
from eptta.training.selection import select_source_checkpoint


def finalize_training(run_ref, output):
    import torch
    run_dir = Path(run_ref)
    run = read_json(run_dir / "run.json")
    if run.get("status") != "TRAINED" or run.get("phase") != "full":
        raise ContractError("only a completed full training run can be source-val selected")
    if run.get("task_weight_origin") != "trained_in_project":
        raise ContractError("external task weights cannot be finalized")
    metrics_path = run_dir / run.get("metrics_ref", "metrics.jsonl")
    selected = select_source_checkpoint(list(iter_jsonl(metrics_path)))
    checkpoint = run_dir / selected["checkpoint_ref"]
    if not checkpoint.is_file() or checkpoint.name in ("best.pt", "last.pt"):
        raise DataError("source_val selection must resolve to a concrete epoch checkpoint")
    value = torch.load(checkpoint, map_location="cpu")
    if (value.get("schema_version") != "0.3.0" or value.get("epoch") != selected["epoch"] or
            value.get("run_id") != run["training_run_id"] or
            value.get("task_weight_origin") != "trained_in_project" or
            not isinstance(value.get("model_state"), dict)):
        raise ContractError("selected epoch checkpoint disagrees with its training run")
    result = {"schema_version": "0.3.0", "status": "SELECTED", "training_phase": "full",
              "source_run_id": run["training_run_id"], "model_id": run["model_id"],
              "selected_epoch": selected["epoch"], "source_val_eer": selected["source_val_eer"],
              "selection_rule": "minimum_source_val_eer_then_earliest_epoch",
              "selected_checkpoint_ref": str(checkpoint.resolve()),
              "recipe_ref": run["recipe_ref"], "fit": run["fit"], "source_val": run["source_val"],
              "architecture": run["architecture"], "class_index_map": run["class_index_map"],
              "model_patch": run["model_patch"], "embedding_dim": 160,
              "initialization": run.get("initialization"),
              "task_weight_origin": "trained_in_project", "metrics_ref": str(metrics_path.resolve())}
    write_json_new(output, result)
    return result


def require_exportable(selected_ref):
    value = read_json(selected_ref)
    if (value.get("status") != "SELECTED" or value.get("training_phase") != "full" or
            value.get("task_weight_origin") != "trained_in_project"):
        raise ContractError("frozen export requires a full source-val-selected training run")
    checkpoint = Path(value.get("selected_checkpoint_ref", ""))
    if not checkpoint.is_file() or checkpoint.name in ("best.pt", "last.pt"):
        raise DataError("selected concrete epoch checkpoint is missing")
    return value

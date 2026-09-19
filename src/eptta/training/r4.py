"""Small frozen export with strict state loading and numerical parity."""
from pathlib import Path
import json
import os
import subprocess
import sys

from eptta.data.io import iter_jsonl, read_json, write_json_new
from eptta.errors import ContractError, DataError, ResourceError
from eptta.training.artifacts import require_exportable


def _fixture(selected):
    checkpoint = Path(selected["selected_checkpoint_ref"])
    run_dir = checkpoint.parent.parent
    job = read_json(run_dir / "source_train_job.json")
    fit_ref = job["source_job"]["fit"]["manifest_ref"]
    rows = list(iter_jsonl(fit_ref))
    if not rows:
        raise DataError("fit manifest is empty")
    row = min(rows, key=lambda item: int(item.get("sample_index", 0)))
    root = job["execution"]["data_roots"].get(row["root_key"])
    if not root:
        raise DataError("fit fixture root is unbound")
    audio = (Path(root) / row["audio_relpath"]).resolve()
    try:
        audio.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise DataError("fit fixture escapes its data root") from exc
    if not audio.is_file():
        raise DataError("fit fixture audio is missing")
    return job, audio


def launch_r4_export(selected_ref, output, gpu_id=0):
    selected = require_exportable(selected_ref)
    job, fixture_audio = _fixture(selected)
    destination = Path(output)
    if destination.exists():
        raise ContractError("frozen output exists; overwrite is forbidden")
    worker = Path(__file__).parents[3] / "workers/baseline_bridge.py"
    if not worker.is_file():
        raise ResourceError("frozen export worker is missing")
    worker_job = {"schema_version": "0.3.0", "job_type": "frozen_export",
                  "model_id": selected["model_id"], "architecture": selected["architecture"],
                  "initialization": selected.get("initialization"),
                  "checkpoint_ref": selected["selected_checkpoint_ref"],
                  "fixture_audio_ref": str(fixture_audio), "output_dir": str(destination.resolve()),
                  "bundle_fields": {"source_run_id": selected["source_run_id"],
                                    "epoch": selected["selected_epoch"],
                                    "class_index_map": selected["class_index_map"],
                                    "embedding_dim": selected["embedding_dim"],
                                    "training_phase": selected["training_phase"],
                                    "source_val_selection_ref": str(Path(selected_ref).resolve()),
                                    "preprocess": job["execution"]["preprocess"]}}
    job_path = destination.parent / (destination.name + ".export-job.json")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(job_path, worker_job)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run([sys.executable, str(worker), "export", "--job", str(job_path)],
                               check=False, env=environment)
    if completed.returncode:
        raise ResourceError("frozen export/parity worker failed with exit code %d" % completed.returncode)
    from eptta.models.frozen import verify_frozen_export
    bundle, _manifest, parity, _selection = verify_frozen_export(destination / "bundle.json")
    return {"schema_version": "0.3.0", "status": "PASS", "bundle_id": bundle["baseline_id"],
            "bundle_ref": str((destination / "bundle.json").resolve()),
            "parity": parity}

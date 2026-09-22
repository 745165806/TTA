#!/usr/bin/env python
"""Four-GPU confirmatory queue with durable, resumable job status."""
import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))

from baselines.locked_config import load_locked_document
from baselines.provenance import git_commit, require_clean_tree

DATASETS = ("itw_target90", "asv2021_la", "asv2021_df", "control_test")
METHODS = ("norm_only_audio", "tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full")
FINAL_STATUSES = {"SUCCESS", "FAILED", "RESOURCE_LIMIT", "CONFIG_MISMATCH"}


def job_key(dataset, method):
    return "%s/%s" % (dataset, method)


def load_run_config(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def config_matches(run_config, current_commit, locked_config_content, method, dataset):
    return (isinstance(run_config, dict)
            and run_config.get("git_commit") == current_commit
            and run_config.get("locked_config_content") == locked_config_content
            and run_config.get("method") == method
            and run_config.get("split") == dataset)


def can_skip_success(record, scores_path, run_config_path, current_commit,
                     locked_config_content, method, dataset):
    run_config = load_run_config(run_config_path)
    return (isinstance(record, dict) and record.get("status") == "SUCCESS"
            and Path(scores_path).is_file()
            and record.get("git_commit") == current_commit
            and record.get("locked_config_content") == locked_config_content
            and record.get("method") == method
            and record.get("dataset") == dataset
            and config_matches(run_config, current_commit, locked_config_content,
                               method, dataset))


def classify_failure(returncode, log_text):
    resource_markers = ("out of memory", "resource exhausted", "resource limit",
                        "cuda error: out of memory")
    if returncode in (137, -9) or any(marker in log_text.lower()
                                      for marker in resource_markers):
        return "RESOURCE_LIMIT"
    return "FAILED"


def final_exit_code(records):
    statuses = [record.get("status") for record in records.values()]
    if "FAILED" in statuses or "CONFIG_MISMATCH" in statuses:
        return 1
    if any(status != "SUCCESS" for status in statuses):
        return 2
    return 0


class StatusStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.json_path = self.directory / "job_status.json"
        self.csv_path = self.directory / "job_status.csv"
        self.lock = threading.Lock()
        if self.json_path.is_file():
            doc = json.loads(self.json_path.read_text(encoding="utf-8"))
            self.records = doc.get("jobs", {})
        else:
            self.records = {}

    def get(self, key):
        with self.lock:
            return self.records.get(key)

    def put(self, key, record):
        if record.get("status") not in FINAL_STATUSES:
            raise ValueError("invalid final job status")
        with self.lock:
            self.records[key] = record
            self._write()

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "0.2.0", "jobs": self.records}
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(self.json_path)
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["dataset", "method", "status", "returncode", "gpu",
                      "resumed_skipped", "output", "git_commit", "run_config_ref"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in sorted(self.records):
                writer.writerow({field: self.records[key].get(field) for field in fields})


def _record(dataset, method, status, gpu, output, current_commit,
            locked_config_content, returncode=None, resumed_skipped=False):
    return {"dataset": dataset, "method": method, "status": status,
            "returncode": returncode, "gpu": gpu,
            "resumed_skipped": resumed_skipped, "output": str(output),
            "git_commit": current_commit,
            "locked_config_content": locked_config_content,
            "run_config_ref": str(output / "run_config.json")}


def run_job(dataset, method, gpu, run_dir, locked_config, store,
            current_commit, locked_config_content):
    output = Path(run_dir) / dataset / method
    scores = output / "scores.jsonl"
    run_config_path = output / "run_config.json"
    key = job_key(dataset, method)
    previous = store.get(key)
    if can_skip_success(previous, scores, run_config_path, current_commit,
                        locked_config_content, method, dataset):
        record = dict(previous)
        record["resumed_skipped"] = True
        store.put(key, record)
        return
    if scores.exists():
        store.put(key, _record(dataset, method, "CONFIG_MISMATCH", gpu, output,
                               current_commit, locked_config_content))
        return
    existing_run_config = load_run_config(run_config_path)
    if run_config_path.exists() and not config_matches(
            existing_run_config, current_commit, locked_config_content, method, dataset):
        store.put(key, _record(dataset, method, "CONFIG_MISMATCH", gpu, output,
                               current_commit, locked_config_content))
        return

    log_path = Path(run_dir) / "logs" / ("%s_%s.log" % (dataset, method))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(EXP_DIR / "baselines/run_port.py"), "--method", method,
               "--split", dataset, "--output", str(output), "--locked-config",
               str(locked_config)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=EXP_DIR, check=False)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    status = "SUCCESS" if completed.returncode == 0 else classify_failure(
        completed.returncode, log_text)
    if status == "SUCCESS" and not scores.is_file():
        status = "FAILED"
    store.put(key, _record(dataset, method, status, gpu, output, current_commit,
                           locked_config_content, completed.returncode))


def guarded_run_job(dataset, method, gpu, run_dir, locked_config, store,
                    current_commit, locked_config_content):
    """Ensure even an unexpected launcher error produces a durable FAILED record."""
    try:
        run_job(dataset, method, gpu, run_dir, locked_config, store,
                current_commit, locked_config_content)
    except Exception as exc:  # the queue must never lose a claimed job silently
        output = Path(run_dir) / dataset / method
        record = _record(dataset, method, "FAILED", gpu, output, current_commit,
                         locked_config_content)
        record["launcher_error"] = str(exc)
        store.put(job_key(dataset, method), record)


def write_confirmatory_provenance(run_dir, validation_run, locked_config_content,
                                  current_commit, gpus):
    run_dir = Path(run_dir)
    summary = json.loads((Path(validation_run) / "validation_summary.json").read_text(
        encoding="utf-8"))
    ports = json.loads((Path(validation_run) / "pilot/port_validation.json").read_text(
        encoding="utf-8"))
    if (summary.get("validation_git_commit") != current_commit
            or summary.get("validation_status") != "PASS"
            or summary.get("published_ports_valid") is not True
            or ports.get("validation_git_commit") != current_commit
            or ports.get("locked_config_content") != locked_config_content):
        raise ValueError("validation run does not match confirmatory provenance")
    provenance = {
        "schema_version": "0.2.0", "git_commit": current_commit,
        "validation_run_ref": str(Path(validation_run).resolve()),
        "validation_git_commit": summary.get("validation_git_commit"),
        "locked_config_content": locked_config_content,
        "gpu_list": list(gpus), "datasets": list(DATASETS), "methods": list(METHODS),
        "start_configuration": {"worker_count": 4, "resume_policy": "exact_match"},
    }
    path = run_dir / "confirmatory_provenance.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != provenance:
            raise ValueError("CONFIG_MISMATCH: confirmatory provenance differs")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4:
        parser.error("exactly four GPUs are required")
    try:
        require_clean_tree()
        current_commit = git_commit()
    except ValueError as exc:
        parser.error(str(exc))
    locked_config_content = load_locked_document(args.locked_config)
    try:
        write_confirmatory_provenance(
            args.run_dir, args.validation_run, locked_config_content, current_commit, gpus)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    store = StatusStore(Path(args.run_dir))
    pending = queue.Queue()
    for dataset in DATASETS:
        for method in METHODS:
            pending.put((dataset, method))

    def gpu_worker(gpu):
        while True:
            try:
                dataset, method = pending.get_nowait()
            except queue.Empty:
                return
            try:
                guarded_run_job(dataset, method, gpu, args.run_dir,
                                args.locked_config, store, current_commit,
                                locked_config_content)
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(gpu_worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    raise SystemExit(final_exit_code(store.records))


if __name__ == "__main__":
    main()

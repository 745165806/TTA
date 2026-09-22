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
DATASETS = ("itw_target90", "asv2021_la", "asv2021_df", "control_test")
METHODS = ("norm_only_audio", "tent_audio_ep", "sar_audio_ep", "memo_audio_ep_full")
FINAL_STATUSES = {"SUCCESS", "FAILED", "RESOURCE_LIMIT"}


def job_key(dataset, method):
    return "%s/%s" % (dataset, method)


def can_skip_success(record, scores_path):
    return (isinstance(record, dict) and record.get("status") == "SUCCESS"
            and Path(scores_path).is_file())


def classify_failure(returncode, log_text):
    resource_markers = ("out of memory", "resource exhausted", "resource limit",
                        "cuda error: out of memory")
    if returncode in (137, -9) or any(marker in log_text.lower()
                                      for marker in resource_markers):
        return "RESOURCE_LIMIT"
    return "FAILED"


def final_exit_code(records):
    statuses = [record.get("status") for record in records.values()]
    if "FAILED" in statuses:
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
        document = {"schema_version": "0.1.0", "jobs": self.records}
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        tmp.replace(self.json_path)
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            fields = ["dataset", "method", "status", "returncode", "gpu",
                      "resumed_skipped", "output"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key in sorted(self.records):
                writer.writerow({field: self.records[key].get(field) for field in fields})


def run_job(dataset, method, gpu, run_dir, locked_config, store):
    output = Path(run_dir) / "confirmatory" / dataset / method
    scores = output / "scores.jsonl"
    key = job_key(dataset, method)
    previous = store.get(key)
    if can_skip_success(previous, scores):
        record = dict(previous)
        record["resumed_skipped"] = True
        store.put(key, record)
        return
    if scores.exists():
        store.put(key, {"dataset": dataset, "method": method, "status": "FAILED",
                        "returncode": None, "gpu": gpu, "resumed_skipped": False,
                        "output": str(output)})
        return

    log_path = Path(run_dir) / "confirmatory/logs" / ("%s_%s.log" % (dataset, method))
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
    store.put(key, {"dataset": dataset, "method": method, "status": status,
                    "returncode": completed.returncode, "gpu": gpu,
                    "resumed_skipped": False, "output": str(output)})


def guarded_run_job(dataset, method, gpu, run_dir, locked_config, store):
    """Ensure even an unexpected launcher error produces a durable FAILED record."""
    try:
        run_job(dataset, method, gpu, run_dir, locked_config, store)
    except Exception as exc:  # the queue must never lose a claimed job silently
        output = Path(run_dir) / "confirmatory" / dataset / method
        store.put(job_key(dataset, method), {
            "dataset": dataset, "method": method, "status": "FAILED",
            "returncode": None, "gpu": gpu, "resumed_skipped": False,
            "output": str(output), "launcher_error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--gpus", required=True)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4:
        parser.error("exactly four GPUs are required")
    store = StatusStore(Path(args.run_dir) / "confirmatory")
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
                                args.locked_config, store)
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(gpu_worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    raise SystemExit(final_exit_code(store.records))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate arbitrary compatible AASIST/SSL-AASIST parameter files."""
from __future__ import absolute_import, print_function

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "compat"))
from author_training import build_author_model, load_audio, sha256_file


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def parse_protocol(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            fields = line.strip().split()
            if len(fields) != 5 or fields[4] not in ("bonafide", "spoof"):
                raise ValueError("invalid ASVspoof protocol row %d" % number)
            speaker_id, sample_id, _unused, attack_id, key = fields
            if sample_id in seen:
                raise ValueError("duplicate protocol sample ID: %s" % sample_id)
            seen.add(sample_id)
            rows.append({"speaker_id": speaker_id, "sample_id": sample_id,
                         "attack_id": attack_id, "key": key,
                         "canonical_label": 0 if key == "bonafide" else 1})
    if not rows:
        raise ValueError("evaluation protocol is empty")
    return rows


class EvaluationDataset(object):
    def __init__(self, rows, audio_dir):
        self.rows = rows
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        sample_id = self.rows[index]["sample_id"]
        path = os.path.join(self.audio_dir, sample_id + ".flac")
        if not os.path.isfile(path):
            raise ValueError("evaluation audio is missing: %s" % path)
        return torch.from_numpy(load_audio(path)), sample_id


def checkpoint_state(value):
    if not isinstance(value, dict):
        raise ValueError("checkpoint must contain a state-dict mapping")
    if isinstance(value.get("model_state"), dict):
        state, checkpoint_format = value["model_state"], "eptta_model_state"
    elif isinstance(value.get("state_dict"), dict):
        state, checkpoint_format = value["state_dict"], "state_dict_field"
    elif value and all(hasattr(item, "shape") for item in value.values()):
        state, checkpoint_format = value, "raw_state_dict"
    else:
        raise ValueError("checkpoint has no recognized model parameter mapping")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: item for key, item in state.items()}
        checkpoint_format += "_module_prefix_removed"
    return state, checkpoint_format


def equal_error_rate(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if len(scores) != len(labels) or not scores or not positives or not negatives:
        raise ValueError("EER requires aligned scores containing both classes")
    pairs = sorted(zip((float(score) for score in scores), labels), reverse=True)
    false_accepts = 0
    true_accepts = 0
    points = [(0.0, 1.0)]
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                true_accepts += 1
            else:
                false_accepts += 1
            index += 1
        points.append((false_accepts / negatives, 1.0 - true_accepts / positives))
    for (far0, frr0), (far1, frr1) in zip(points, points[1:]):
        d0, d1 = far0 - frr0, far1 - frr1
        if d0 == 0:
            return far0
        if d0 * d1 <= 0:
            fraction = abs(d0) / (abs(d0) + abs(d1)) if d0 != d1 else 0.0
            return ((far0 + fraction * (far1 - far0)) +
                    (frr0 + fraction * (frr1 - frr0))) / 2.0
    far, frr = min(points, key=lambda point: abs(point[0] - point[1]))
    return (far + frr) / 2.0


def auroc(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank_sum += (((index + 1) + end) / 2.0) * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(scores, labels, threshold):
    prediction = [int(score > threshold) for score in scores]
    tp = sum(predicted == 1 and label == 1 for predicted, label in zip(prediction, labels))
    fp = sum(predicted == 1 and label == 0 for predicted, label in zip(prediction, labels))
    tn = sum(predicted == 0 and label == 0 for predicted, label in zip(prediction, labels))
    fn = sum(predicted == 0 and label == 1 for predicted, label in zip(prediction, labels))
    return {"count": len(scores), "threshold": float(threshold),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tp / (tp + fn), "fpr": fp / (fp + tn), "fnr": fn / (tp + fn),
            "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
            "auroc": auroc(scores, labels), "eer": equal_error_rate(scores, labels)}


def official_tdcf(job, rows, score_by_id, temporary):
    if job["asv_scores"] is None:
        return None
    import numpy
    for binding in (job["asv_scores"], job["tdcf_implementation"]):
        if sha256_file(binding["artifact_ref"]) != binding["sha256"]:
            raise ValueError("official metric input changed")
    # The pinned 2019 reference uses the removed numpy.float alias.
    if not hasattr(numpy, "float"):
        numpy.float = float
    path = job["tdcf_implementation"]["artifact_ref"]
    spec = importlib.util.spec_from_file_location("eptta_pinned_asvspoof2019_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cm_path = os.path.join(temporary, "official_cm_scores.txt")
    with open(cm_path, "x", encoding="utf-8") as stream:
        for row in rows:
            # The official implementation expects larger values to mean bona fide.
            stream.write("%s %s %s %.17g\n" % (row["sample_id"], row["attack_id"], row["key"],
                                                -score_by_id[row["sample_id"]]))
    report_path = os.path.join(temporary, "official_tdcf.txt")
    eer_percent, minimum = module.calculate_tDCF_EER(
        cm_scores_file=cm_path, asv_score_file=job["asv_scores"]["artifact_ref"],
        output_file=report_path, printout=True)
    return {"eer_percent": float(eer_percent), "min_tdcf": float(minimum),
            "score_direction": "larger_is_bonafide", "cm_scores_ref": "official_cm_scores.txt",
            "cm_scores_sha256": sha256_file(cm_path), "report_ref": "official_tdcf.txt",
            "report_sha256": sha256_file(report_path),
            "asv_scores_sha256": job["asv_scores"]["sha256"],
            "implementation_sha256": job["tdcf_implementation"]["sha256"]}


def evaluate(job):
    import torch

    required = {"schema_version", "job_type", "model_id", "architecture", "initialization",
                "checkpoint_ref", "checkpoint_sha256", "protocol_ref", "protocol_sha256",
                "expected_sample_count", "audio_dir", "output_dir", "batch_size", "num_workers",
                "threshold", "physical_gpu_id", "evaluation_tag", "worker_ref", "worker_sha256"}
    required.update({"asv_scores", "tdcf_implementation"})
    if set(job) != required or job["schema_version"] != "0.1.0" or job["job_type"] != "checkpoint_evaluation":
        raise ValueError("invalid checkpoint evaluation job")
    for path_key, hash_key in (("checkpoint_ref", "checkpoint_sha256"),
                               ("protocol_ref", "protocol_sha256"),
                               ("worker_ref", "worker_sha256")):
        if sha256_file(job[path_key]) != job[hash_key]:
            raise ValueError("hash-bound input changed: %s" % path_key)
    rows = parse_protocol(job["protocol_ref"])
    if len(rows) != job["expected_sample_count"]:
        raise ValueError("protocol sample count changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; checkpoint evaluation does not fall back to CPU")
    device = torch.device("cuda", 0)
    construction = {"source_job": {"model_id": job["model_id"],
                                     "initialization": job["initialization"]},
                    "execution": {"architecture": job["architecture"]}}
    adapter, construction_patch = build_author_model(construction, device)
    try:
        loaded = torch.load(job["checkpoint_ref"], map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(job["checkpoint_ref"], map_location="cpu")
    state, checkpoint_format = checkpoint_state(loaded)
    adapter.model.load_state_dict(state, strict=True)
    del loaded, state
    adapter.model.eval()
    dataset = EvaluationDataset(rows, job["audio_dir"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=job["batch_size"], shuffle=False,
                                         drop_last=False, num_workers=job["num_workers"], pin_memory=True)
    output = os.path.abspath(job["output_dir"])
    parent = os.path.dirname(output)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(output):
        raise ValueError("evaluation output exists; overwrite is forbidden")
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(output) + ".", dir=parent)
    started = time.time()
    score_by_id = {}
    progress_path = os.path.join(temporary, "progress.jsonl")
    try:
        with torch.inference_mode():
            for batch_index, (waveform, sample_ids) in enumerate(loader):
                waveform = waveform.to(device, non_blocking=True)
                _embedding, logits = adapter.forward(waveform, freq_aug=False)
                mapping = job["architecture"]["class_index_map"]
                scores = (logits[:, mapping["spoof"]] - logits[:, mapping["bonafide"]]).detach().cpu().tolist()
                for sample_id, score in zip(sample_ids, scores):
                    if not math.isfinite(score) or sample_id in score_by_id:
                        raise ValueError("duplicate or non-finite evaluation score")
                    score_by_id[sample_id] = float(score)
                if batch_index % 100 == 0 or len(score_by_id) == len(rows):
                    progress = {"scored": len(score_by_id), "expected": len(rows),
                                "elapsed_seconds": time.time() - started}
                    with open(progress_path, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n")
                    print("scored %d/%d" % (len(score_by_id), len(rows)), flush=True)
        if set(score_by_id) != {row["sample_id"] for row in rows}:
            raise ValueError("evaluation score coverage is incomplete")
        scores_path = os.path.join(temporary, "scores.jsonl")
        with open(scores_path, "x", encoding="utf-8") as stream:
            for row in rows:
                value = {"schema_version": "0.1.0", "sample_id": row["sample_id"],
                         "score": score_by_id[row["sample_id"]]}
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        scores = [score_by_id[row["sample_id"]] for row in rows]
        labels = [row["canonical_label"] for row in rows]
        aggregate = binary_metrics(scores, labels, job["threshold"])
        attacks = {}
        bona_scores = [score for score, label in zip(scores, labels) if label == 0]
        for attack_id in sorted({row["attack_id"] for row in rows if row["canonical_label"] == 1}):
            attack_scores = [score_by_id[row["sample_id"]] for row in rows if row["attack_id"] == attack_id]
            attacks[attack_id] = {"count": len(attack_scores),
                                  "eer": equal_error_rate(bona_scores + attack_scores,
                                                          [0] * len(bona_scores) + [1] * len(attack_scores))}
        elapsed = time.time() - started
        tdcf = official_tdcf(job, rows, score_by_id, temporary)
        metrics = {"schema_version": "0.1.0", "status": "EVALUATED",
                   "score_direction": "larger_is_spoof", "metrics": aggregate,
                   "attack_breakdown": attacks}
        if tdcf is not None:
            metrics["official_asvspoof2019_la"] = tdcf
        write_json(os.path.join(temporary, "metrics.json"), metrics)
        run = {"schema_version": "0.1.0", "status": "EVALUATED",
               "evaluation_tag": job["evaluation_tag"], "model_id": job["model_id"],
               "checkpoint_ref": job["checkpoint_ref"],
               "checkpoint_sha256": job["checkpoint_sha256"],
               "checkpoint_format": checkpoint_format,
               "training_completion_required": False,
               "protocol_ref": job["protocol_ref"], "protocol_sha256": job["protocol_sha256"],
               "audio_dir": job["audio_dir"], "sample_count": len(rows),
               "scores_ref": "scores.jsonl", "scores_sha256": sha256_file(scores_path),
               "metrics_ref": "metrics.json", "metrics_sha256": sha256_file(os.path.join(temporary, "metrics.json")),
               "progress_ref": "progress.jsonl", "progress_sha256": sha256_file(progress_path),
               "architecture": job["architecture"], "initialization": job["initialization"],
               "model_construction_patch": construction_patch,
               "worker_sha256": job["worker_sha256"], "batch_size": job["batch_size"],
               "num_workers": job["num_workers"], "physical_gpu_id": job["physical_gpu_id"],
               "torch_version": torch.__version__, "python_version": platform.python_version(),
               "cuda_device_name": torch.cuda.get_device_name(device),
               "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
               "elapsed_seconds": elapsed}
        write_json(os.path.join(temporary, "run.json"), run)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job, encoding="utf-8") as stream:
        job = json.load(stream)
    evaluate(job)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED"}), file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__,
                          "message": str(exc)}), file=sys.stderr)
        sys.exit(2)

#!/usr/bin/env python3
"""R5 stage-1 cache-parity bridge: independently re-encode deterministic views,
read the on-disk feature cache back, and compare embeddings / head logits /
score per UID and view index, while recording per-stage timing and peak memory.
"""
from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "compat"))
sys.path.insert(0, HERE)
import baseline_bridge  # noqa: E402  (reuse _views/_safe_audio_path/sha256_file/write_json)

_views = baseline_bridge._views
_safe_audio_path = baseline_bridge._safe_audio_path
sha256_file = baseline_bridge.sha256_file
write_json = baseline_bridge.write_json
load_audio = baseline_bridge.load_audio
_apply_condition = baseline_bridge._apply_condition


def _stats(values):
    import numpy
    array = numpy.asarray(values, dtype="float64")
    if array.size == 0:
        return {"count": 0, "max_abs": 0.0, "p50_abs": 0.0, "p95_abs": 0.0, "p99_abs": 0.0}
    return {"count": int(array.size), "max_abs": float(array.max()),
            "p50_abs": float(numpy.quantile(array, .50)), "p95_abs": float(numpy.quantile(array, .95)),
            "p99_abs": float(numpy.quantile(array, .99))}


def _joint_exceedances(left, right, atol, rtol):
    import numpy
    a = numpy.asarray(left, dtype="float64")
    b = numpy.asarray(right, dtype="float64")
    delta = numpy.abs(a - b)
    over_atol = int((delta > atol).sum())
    over_joint = int((delta > atol + rtol * numpy.abs(b)).sum())
    non_finite = int((not numpy.isfinite(a).all()) or (not numpy.isfinite(b).all()))
    return {"max_abs": float(delta.max()) if delta.size else 0.0,
            "over_atol": over_atol, "over_joint": over_joint, "non_finite": non_finite}


def validate_job(job):
    allowed = {"schema_version", "job_type", "role", "bundle_ref", "cache_ref", "manifest_ref",
               "manifest_sha256", "data_roots", "probe", "numerical_mode", "expected_ids",
               "batch_sizes", "atol", "rtol", "worker_sha256", "output_dir"}
    if set(job) != allowed or job["schema_version"] != "0.1.0" or job["job_type"] != "r5_parity":
        raise ValueError("invalid r5 parity job")
    if sha256_file(os.path.abspath(__file__)) != job["worker_sha256"]:
        raise ValueError("r5 parity worker hash mismatch")
    if job["probe"].get("num_views") != 3 or set(job["probe"]) != {
            "num_views", "seed", "noise_snr_db", "fir_side_gain"}:
        raise ValueError("unsupported probe contract")
    if job["numerical_mode"].get("dtype") != "float32":
        raise ValueError("r5 parity requires float32")
    if not (0 < float(job["atol"]) < 1 and 0 < float(job["rtol"]) < 1):
        raise ValueError("invalid parity tolerance")
    if not isinstance(job["expected_ids"], list) or not job["expected_ids"]:
        raise ValueError("expected IDs must be a nonempty list")


def _load_bundle(job):
    import torch
    bundle_root = os.path.dirname(job["bundle_ref"])
    with open(job["bundle_ref"], encoding="utf-8") as stream:
        bundle = json.load(stream)
    with open(os.path.join(bundle_root, "export_manifest.json"), encoding="utf-8") as stream:
        export_manifest = json.load(stream)
    for name, digest in export_manifest["files"].items():
        if sha256_file(os.path.join(bundle_root, name)) != digest:
            raise ValueError("frozen export file changed: %s" % name)
    if (bundle.get("training_status") != "FINALIZED" or bundle.get("training_phase") != "full" or
            bundle.get("task_weight_origin") != "trained_in_project"):
        raise ValueError("parity requires a finalized in-project full source model")
    with open(os.path.join(bundle_root, bundle["parity_report_ref"]), encoding="utf-8") as stream:
        parity = json.load(stream)
    if (parity.get("status") != "PASS" or parity.get("module_modes_stable") is not True or
            parity.get("buffers_stable") is not True):
        raise ValueError("parity requires frozen wrapper/head PASS")
    state = torch.load(os.path.join(bundle_root, "detector_state.pt"), map_location="cpu")
    head = torch.load(os.path.join(bundle_root, bundle["head_ref"]), map_location="cpu", weights_only=True)
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle["init_provenance"] if
                                    bundle["model_id"] == "ssl_aasist_source" else None},
                    "execution": {"architecture": state["architecture"]}}
    device = torch.device("cuda")
    adapter, _patch = baseline_bridge.build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    weight = head["w"].to(device)
    bias = float(head["b"])
    return bundle, bundle_root, adapter, weight, bias


def _read_cache(cache_ref, expected_ids):
    import numpy
    root = os.path.abspath(cache_ref)
    index_path = os.path.join(root, "index.json") if os.path.isdir(root) else root
    with open(index_path, encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("status") != "LOCKED" or index.get("format") != "sharded_npy_v1" or index.get("allow_pickle") is not False:
        raise ValueError("feature cache is not locked pickle-free")
    base = os.path.dirname(index_path)
    cached, seen, read_seconds = {}, set(), 0.0
    for item in index["chunks"]:
        array_path = os.path.join(base, item["array_ref"])
        ids_path = os.path.join(base, item["ids_ref"])
        if sha256_file(array_path) != item["array_sha256"] or sha256_file(ids_path) != item["ids_sha256"]:
            raise ValueError("feature cache chunk changed")
        with open(ids_path, encoding="utf-8") as stream:
            ids = json.load(stream)
        started = time.monotonic()
        with open(array_path, "rb") as stream:
            array = numpy.load(stream, allow_pickle=False)
        read_seconds += time.monotonic() - started
        if array.shape != tuple(item["shape"]) or array.dtype != numpy.dtype(item["dtype"]):
            raise ValueError("feature cache chunk metadata mismatch")
        if set(ids).intersection(seen) or len(set(ids)) != len(ids):
            raise ValueError("feature cache duplicate IDs")
        seen.update(ids)
        for offset, sample_id in enumerate(ids):
            cached[sample_id] = array[offset]
    if len(seen) != index["sample_count"] or set(cached) != set(expected_ids):
        raise ValueError("feature cache coverage mismatch")
    return cached, index, read_seconds


def parity(job):
    import numpy
    import torch
    validate_job(job)
    if not torch.cuda.is_available():
        raise RuntimeError("r5 parity requires an available CUDA device")
    numerical_execution = baseline_bridge._apply_numerical_mode(job["numerical_mode"])
    tf32_matmul = numerical_execution["tf32_matmul"]
    tf32_cudnn = numerical_execution["tf32_cudnn"]
    started = time.monotonic()
    bundle, bundle_root, adapter, weight, bias = _load_bundle(job)
    device = weight.device
    with open(job["manifest_ref"], encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    row_by_id = {}
    for row in rows:
        allowed = {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}
        if set(row) != allowed:
            raise ValueError("parity worker received annotations")
        row_by_id[row["sample_id"]] = row
    expected_ids = job["expected_ids"]
    if set(row_by_id) != set(expected_ids):
        raise ValueError("manifest / expected ID mismatch")
    cached, cache_index, read_seconds = _read_cache(job["cache_ref"], expected_ids)
    if cache_index.get("numerical_execution") != numerical_execution:
        raise ValueError("cache was not produced under the locked numerical execution mode")
    identity = cache_index.get("identity", {})
    if (identity.get("baseline_id") != bundle["baseline_id"] or
            identity.get("selected_checkpoint_sha256") != bundle["selected_checkpoint_sha256"] or
            identity.get("numerical_mode") != job["numerical_mode"]):
        raise ValueError("cache identity differs from the parity bundle/numerical mode")
    atol, rtol = float(job["atol"]), float(job["rtol"])
    probe = job["probe"]

    decode_seconds = view_seconds = encode_seconds = write_seconds = 0.0
    per_sample = []
    error_series = {"embedding": [], "head_logits": [], "score": []}
    head_weight = adapter.model.out_layer.weight.detach()
    head_bias = adapter.model.out_layer.bias.detach()
    with torch.no_grad():
        for sample_id in expected_ids:
            row = row_by_id[sample_id]
            audio_path = _safe_audio_path(job["data_roots"][row["root_key"]], row["audio_relpath"])
            if row["input_sha256"] is not None and sha256_file(audio_path) != row["input_sha256"]:
                raise ValueError("audio content hash mismatch: %s" % sample_id)
            t0 = time.monotonic()
            waveform = torch.from_numpy(load_audio(audio_path))
            decode_seconds += time.monotonic() - t0
            waveform = _apply_condition(waveform, sample_id, job["numerical_mode"].get("input_condition"))
            t0 = time.monotonic()
            views = _views(waveform, sample_id, probe).to(device)
            view_seconds += time.monotonic() - t0
            t0 = time.monotonic()
            embedding, _logits = adapter.forward(views, freq_aug=False)
            encode_seconds += time.monotonic() - t0
            online = embedding.detach().cpu().numpy()
            cached_array = cached[sample_id]
            online_logits = (torch.from_numpy(online).to(device) @ head_weight.T + head_bias).detach().cpu().numpy()
            cached_logits = (torch.from_numpy(cached_array).to(device) @ head_weight.T + head_bias).detach().cpu().numpy()
            online_score = float(torch.from_numpy(online[0]).to(device) @ weight + bias)
            cached_score = float(torch.from_numpy(cached_array[0]).to(device) @ weight + bias)
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as handle:
                handle.close()
                t0 = time.monotonic()
                with open(handle.name, "wb") as out:
                    numpy.save(out, online, allow_pickle=False)
                write_seconds += time.monotonic() - t0
                with open(handle.name, "rb") as inp:
                    roundtrip = numpy.load(inp, allow_pickle=False)
                os.unlink(handle.name)
            record = {"sample_id": sample_id,
                      "embedding": _joint_exceedances(online, cached_array, atol, rtol),
                      "z0": _joint_exceedances(online[0], cached_array[0], atol, rtol),
                      "head_logits": _joint_exceedances(online_logits, cached_logits, atol, rtol),
                      "score_online": online_score, "score_cached": cached_score,
                      "score_abs_error": abs(online_score - cached_score),
                      "roundtrip_bit_exact": bool((roundtrip == online).all())}
            error_series["embedding"].append(record["embedding"]["max_abs"])
            error_series["head_logits"].append(record["head_logits"]["max_abs"])
            error_series["score"].append(record["score_abs_error"])
            per_sample.append(record)

    # Read-back consistency: same cached z0 scored per-sample in different orders.
    z0_map = {sample_id: cached[sample_id][0] for sample_id in expected_ids}
    score_reference = {}
    for sample_id in expected_ids:
        score_reference[sample_id] = float(torch.from_numpy(z0_map[sample_id]).to(device) @ weight + bias)
    order_checks = {}
    for name, order in (("reversed", list(reversed(expected_ids))), ("even", expected_ids[::2]),
                        ("odd", expected_ids[1::2]), ("repeat_A", expected_ids[:7]),
                        ("repeat_B", expected_ids[7:14]), ("repeat_A_again", expected_ids[:7])):
        deltas = []
        for sample_id in order:
            score = float(torch.from_numpy(z0_map[sample_id]).to(device) @ weight + bias)
            deltas.append(abs(score - score_reference[sample_id]))
        order_checks[name] = {"max_abs": float(max(deltas)), "samples": len(order)}
    batch_checks = {}
    for size in job["batch_sizes"]:
        deltas, over_joint = [], 0
        for start in range(0, len(expected_ids), size):
            block_ids = expected_ids[start:start + size]
            z0s = numpy.stack([z0_map[sample_id] for sample_id in block_ids])
            tensor = torch.from_numpy(z0s).to(device)
            scores = (tensor @ weight + bias).detach().cpu().numpy()
            for sample_id, value in zip(block_ids, scores):
                delta = abs(float(value) - score_reference[sample_id])
                deltas.append(delta)
                if delta > atol + rtol * abs(score_reference[sample_id]):
                    over_joint += 1
        batch_checks[str(size)] = {"max_abs": float(max(deltas)), "samples": len(deltas),
                                   "over_joint": over_joint}

    elapsed = time.monotonic() - started
    peak_mem = int(torch.cuda.max_memory_allocated(device))
    joint_over = sum(1 for record in per_sample
                     if record["embedding"]["over_joint"] or record["head_logits"]["over_joint"] or
                     record["z0"]["over_joint"])
    non_finite = sum(1 for record in per_sample
                     if record["embedding"]["non_finite"] or record["head_logits"]["non_finite"])
    batch_over_joint = sum(value["over_joint"] for value in batch_checks.values())
    order_max = max((value["max_abs"] for value in order_checks.values()), default=0.0)
    status = "PASS" if (joint_over == 0 and non_finite == 0 and
                        all(record["roundtrip_bit_exact"] for record in per_sample) and
                        batch_over_joint == 0 and order_max <= 1e-6) else "FAIL"
    report = {"schema_version": "0.1.0", "status": status, "role": job["role"],
              "bundle_id": bundle["baseline_id"],
              "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
              "cache_key": cache_index["cache_key"], "atol": atol, "rtol": rtol,
              "sample_count": len(per_sample), "expected_sample_count": len(expected_ids),
              "coverage_complete": set(cached) == set(expected_ids),
              "duplicate_or_missing": int(not (set(cached) == set(expected_ids))),
              "non_finite_samples": non_finite,
              "joint_tolerance_exceeded_samples": joint_over,
              "error_statistics": {field: _stats(values) for field, values in error_series.items()},
              "batch_score_consistency": batch_checks, "order_score_consistency": order_checks,
              "roundtrip_bit_exact_all": all(record["roundtrip_bit_exact"] for record in per_sample),
              "timing": {"decode_seconds": decode_seconds, "view_seconds": view_seconds,
                         "encode_seconds": encode_seconds, "write_seconds": write_seconds,
                         "read_seconds": read_seconds, "elapsed_seconds": elapsed,
                         "samples": len(per_sample),
                         "decode_seconds_per_sample": decode_seconds / len(per_sample),
                         "view_seconds_per_sample": view_seconds / len(per_sample),
                         "encode_seconds_per_sample": encode_seconds / len(per_sample),
                         "write_seconds_per_sample": write_seconds / len(per_sample),
                         "read_seconds_per_sample": read_seconds / len(per_sample)},
              "peak_cuda_allocated_bytes": peak_mem,
              "requested_numerical_mode": job["numerical_mode"],
              "tf32_matmul": tf32_matmul, "tf32_cudnn": tf32_cudnn, "device": str(device)}
    destination = os.path.abspath(job["output_dir"])
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)
    try:
        with open(os.path.join(temporary, "per_sample.jsonl"), "x", encoding="utf-8") as stream:
            for record in per_sample:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        report["per_sample_ref"] = "per_sample.jsonl"
        write_json(os.path.join(temporary, "report.json"), report)
        os.replace(temporary, destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary)
        raise
    if status != "PASS":
        raise ValueError("r5 parity failed; diagnostic report retained")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("parity")
    command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job, encoding="utf-8") as stream:
        job = json.load(stream)
    if args.command == "parity":
        parity(job)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"schema_version": "0.1.0", "status": "FAILED",
                          "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        sys.exit(2)

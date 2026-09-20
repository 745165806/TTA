#!/usr/bin/env python3
"""Single-environment frozen export and label-free feature extraction worker."""
from __future__ import absolute_import, print_function

import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "compat"))
from author_training import build_author_model, independent_seed, load_audio


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _safe_audio_path(root, relative):
    candidate = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([os.path.realpath(root), candidate]) != os.path.realpath(root):
        raise ValueError("audio path escapes its data root")
    if not os.path.isfile(candidate):
        raise ValueError("audio file is missing: %s" % relative)
    return candidate


def _atomic_destination(destination):
    destination = os.path.abspath(destination)
    if os.path.exists(destination):
        raise ValueError("output exists; overwrite is forbidden: %s" % destination)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    return destination, tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)


def export(job):
    import torch
    import torch.nn.functional as functional
    required = {"schema_version", "job_type", "model_id", "architecture", "initialization",
                "checkpoint_ref", "fixture_audio_ref", "output_dir", "bundle_fields"}
    if set(job) != required or job["schema_version"] != "0.3.0" or job["job_type"] != "frozen_export":
        raise ValueError("invalid frozen export job")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    construction = {"source_job": {"model_id": job["model_id"], "initialization": job["initialization"]},
                    "execution": {"architecture": job["architecture"]}}
    adapter, patch = build_author_model(construction, device)
    checkpoint = torch.load(job["checkpoint_ref"], map_location=device)
    model_state = checkpoint.get("model_state")
    fields = job["bundle_fields"]
    if (checkpoint.get("schema_version") != "0.3.0" or not isinstance(model_state, dict) or
            checkpoint.get("task_weight_origin") != "trained_in_project"):
        raise ValueError("checkpoint is not a complete in-project epoch checkpoint")
    if (checkpoint.get("run_id") != fields["source_run_id"] or
            checkpoint.get("class_index_map") != fields["class_index_map"] or
            checkpoint.get("architecture") != job["architecture"] or
            checkpoint.get("epoch") != fields["epoch"]):
        raise ValueError("checkpoint structure/run/epoch differs from the source selection")
    adapter.model.load_state_dict(model_state, strict=True)
    adapter.model.eval()
    waveform = torch.from_numpy(load_audio(job["fixture_audio_ref"])).unsqueeze(0).to(device)
    before_modes = {name: module.training for name, module in adapter.model.named_modules()}
    before_buffers = {name: value.detach().cpu().clone() for name, value in adapter.model.named_buffers()}
    with torch.no_grad():
        embedding, author_logits = adapter.forward(waveform, freq_aug=False)
        head_logits = functional.linear(embedding, adapter.model.out_layer.weight, adapter.model.out_layer.bias)
        mapping = job["bundle_fields"]["class_index_map"]
        weight = adapter.model.out_layer.weight[mapping["spoof"]] - adapter.model.out_layer.weight[mapping["bonafide"]]
        bias = adapter.model.out_layer.bias[mapping["spoof"]] - adapter.model.out_layer.bias[mapping["bonafide"]]
        exported_score = embedding.matmul(weight) + bias
        author_score = author_logits[:, mapping["spoof"]] - author_logits[:, mapping["bonafide"]]
    after_buffers = dict(adapter.model.named_buffers())
    atol = rtol = 1e-5
    if not torch.allclose(author_logits, head_logits, atol=atol, rtol=rtol):
        raise ValueError("author forward and head(embedding) parity failed")
    if not torch.allclose(author_score, exported_score, atol=atol, rtol=rtol):
        raise ValueError("exported score parity failed")
    if before_modes != {name: module.training for name, module in adapter.model.named_modules()} or any(
            not torch.equal(value, after_buffers[name].detach().cpu()) for name, value in before_buffers.items()):
        raise ValueError("frozen forward changed module mode or persistent buffers")
    destination, temporary = _atomic_destination(job["output_dir"])
    try:
        torch.save({"schema_version": "0.3.0", "model_id": job["model_id"],
                    "model_state": adapter.model.state_dict(), "architecture": job["architecture"],
                    "model_construction_patch": patch, "class_index_map": mapping},
                   os.path.join(temporary, "detector_state.pt"))
        torch.save({"schema_version": "0.3.0", "w": weight.detach().cpu(), "b": bias.detach().cpu(),
                    "score_direction": "larger_is_spoof"}, os.path.join(temporary, "linear_head.pt"))
        write_json(os.path.join(temporary, "parity.json"),
                   {"schema_version": "0.3.0", "status": "PASS",
                    "fixture_ref": job["fixture_audio_ref"], "atol": atol, "rtol": rtol,
                    "author_head_max_abs": float((author_logits - head_logits).abs().max().cpu()),
                    "head_score_max_abs": float((author_score - exported_score).abs().max().cpu()),
                    "module_modes_stable": True, "buffers_stable": True})
        epoch = checkpoint.get("epoch", fields.get("epoch"))
        bundle = {"schema_version": "0.3.0", "model_id": job["model_id"],
                  "baseline_id": "%s:epoch_%04d" % (fields["source_run_id"], int(epoch)),
                  "source_run_id": fields["source_run_id"],
                  "checkpoint_ref": os.path.abspath(job["checkpoint_ref"]), "epoch": int(epoch),
                  "class_index_map": mapping, "head_ref": "linear_head.pt",
                  "detector_state_ref": "detector_state.pt", "embedding_dim": fields["embedding_dim"],
                  "task_weight_origin": "trained_in_project", "training_phase": fields["training_phase"],
                  "source_val_selection_ref": fields["source_val_selection_ref"],
                  "preprocess": fields["preprocess"], "initialization": job["initialization"],
                  "parity_report_ref": "parity.json"}
        write_json(os.path.join(temporary, "bundle.json"), bundle)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def _views(waveform, sample_index, probe):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(
        independent_seed(int(probe["seed"]), int(sample_index), view_index=1, namespace=2))
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_rms = waveform.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noisy = waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-float(probe["noise_snr_db"]) / 20.0))
    gain = float(probe["fir_side_gain"])
    padded = torch.nn.functional.pad(waveform[None, None], (1, 1), mode="reflect")
    kernel = torch.tensor([gain, 1.0, -gain], dtype=waveform.dtype).view(1, 1, 3)
    filtered = torch.nn.functional.conv1d(padded, kernel).view(-1)
    return torch.stack([waveform, noisy, filtered])


def _apply_numerical_mode(mode):
    import torch
    allowed = {"dtype", "tf32_matmul", "tf32_cudnn", "block_units", "input_condition"}
    if (not isinstance(mode, dict) or set(mode) - allowed or
            not {"dtype", "tf32_matmul", "tf32_cudnn", "block_units"}.issubset(mode) or
            mode["dtype"] not in ("float16", "float32", "float64") or
            type(mode["block_units"]) is not int or mode["block_units"] < 1 or
            any(type(mode[name]) is not bool for name in ("tf32_matmul", "tf32_cudnn"))):
        raise ValueError("invalid extraction numerical mode")
    torch.backends.cuda.matmul.allow_tf32 = mode["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = mode["tf32_cudnn"]
    return {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


def _apply_condition(waveform, sample_index, condition):
    import torch
    if not condition:
        return waveform
    if (condition.get("kind") != "awgn" or type(condition.get("snr_db")) not in (int, float) or
            type(condition.get("seed")) is not int):
        raise ValueError("input condition must be AWGN with numeric snr_db/integer seed")
    generator = torch.Generator(device="cpu").manual_seed(
        independent_seed(condition["seed"], int(sample_index), view_index=0, namespace=3))
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_rms = waveform.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    return waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-float(condition["snr_db"]) / 20.0))


def extract(job):
    import numpy
    import torch
    required = {"schema_version", "job_type", "purpose", "input_role", "bundle_ref", "manifest_ref",
                "data_roots", "probe", "numerical_mode", "worker_slot", "worker_count",
                "expected_ids", "cache_identity", "output_dir"}
    if set(job) != required or job["schema_version"] != "0.3.0" or job["job_type"] != "inference":
        raise ValueError("invalid inference job")
    role_policy = {"source_prepare": {"fit", "cal0"}, "select": {"select"},
                   "audit": {"audit", "cal1"},
                   "confirmatory": {"control_test", "target_test"}}
    if job["purpose"] not in role_policy or job["input_role"] not in role_policy[job["purpose"]]:
        raise ValueError("inference purpose/input role violates data-role policy")
    if job["probe"].get("num_views") != 3 or set(job["probe"]) != {
            "num_views", "seed", "noise_snr_db", "fir_side_gain"}:
        raise ValueError("unsupported view configuration")
    numerical_execution = _apply_numerical_mode(job["numerical_mode"])
    bundle_root = os.path.dirname(job["bundle_ref"])
    with open(job["bundle_ref"], encoding="utf-8") as stream:
        bundle = json.load(stream)
    with open(os.path.join(bundle_root, bundle["parity_report_ref"]), encoding="utf-8") as stream:
        parity = json.load(stream)
    if (parity.get("status") != "PASS" or parity.get("module_modes_stable") is not True or
            parity.get("buffers_stable") is not True):
        raise ValueError("frozen parity check is missing or failed")
    state = torch.load(os.path.join(bundle_root, bundle["detector_state_ref"]), map_location="cpu")
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle.get("initialization")},
                    "execution": {"architecture": state["architecture"]}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    expected = set(job["expected_ids"])
    rows = []
    with open(job["manifest_ref"], encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            allowed = {"schema_version", "sample_id", "sample_index", "root_key", "audio_relpath", "split_role"}
            if set(row) != allowed:
                raise ValueError("inference worker received labels, forbidden metadata, or no sample_index")
            if row["split_role"] != job["input_role"] or type(row["sample_index"]) is not int:
                raise ValueError("inference row violates role/index contract")
            if row["sample_id"] in expected:
                rows.append(row)
    if {row["sample_id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("inference shard ID coverage mismatch")
    destination, temporary = _atomic_destination(job["output_dir"])
    try:
        os.mkdir(os.path.join(temporary, "chunks"))
        chunks, block, block_ids = [], [], []
        block_size = int(job["numerical_mode"]["block_units"])
        def flush():
            if not block:
                return
            index = len(chunks)
            array = numpy.stack(block).astype(job["numerical_mode"]["dtype"], copy=False)
            array_name, ids_name = "chunk-%06d.npy" % index, "chunk-%06d.ids.json" % index
            with open(os.path.join(temporary, "chunks", array_name), "xb") as output:
                numpy.save(output, array, allow_pickle=False)
            write_json(os.path.join(temporary, "chunks", ids_name), list(block_ids))
            chunks.append({"index": index, "array_ref": "chunks/" + array_name,
                           "ids_ref": "chunks/" + ids_name, "count": len(block_ids),
                           "shape": list(array.shape), "dtype": str(array.dtype)})
            del block[:]
            del block_ids[:]
        with torch.no_grad():
            for row in rows:
                audio_path = _safe_audio_path(job["data_roots"][row["root_key"]], row["audio_relpath"])
                waveform = torch.from_numpy(load_audio(audio_path))
                waveform = _apply_condition(waveform, row["sample_index"],
                                            job["numerical_mode"].get("input_condition"))
                views = _views(waveform, row["sample_index"], job["probe"]).to(device)
                embedding, _logits = adapter.forward(views, freq_aug=False)
                block.append(embedding.detach().cpu().numpy())
                block_ids.append(row["sample_id"])
                if len(block) >= block_size:
                    flush()
        flush()
        metadata = {"schema_version": "0.2.0", "status": "READY", "format": "sharded_npy_v2",
                    "allow_pickle": False, "identity": job["cache_identity"],
                    "numerical_execution": numerical_execution, "num_views": 3,
                    "feature_dim": int(bundle["embedding_dim"]), "sample_count": len(rows),
                    "expected_ids": list(job["expected_ids"]), "chunks": chunks, "immutable": True}
        write_json(os.path.join(temporary, "index.json"), metadata)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "extract"):
        command = sub.add_parser(name)
        command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job, encoding="utf-8") as stream:
        job = json.load(stream)
    (export if args.command == "export" else extract)(job)
    return 0


if __name__ == "__main__":
    sys.exit(main())

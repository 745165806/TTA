#!/usr/bin/env python3
"""Build and parity-check a frozen detector from an in-project checkpoint."""
from __future__ import absolute_import, print_function

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "compat"))
from author_training import build_author_model, load_audio, sha256_file


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def validate_job(job):
    allowed = {"schema_version", "job_type", "model_id", "architecture", "initialization",
               "selected_checkpoint_ref", "selected_checkpoint_sha256", "fixture_audio_ref",
               "training_patch", "output_dir", "bundle_fields"}
    if set(job) != allowed or job["schema_version"] != "0.1.0" or job["job_type"] != "frozen_export":
        raise ValueError("invalid frozen export job")
    serialized = json.dumps(job, sort_keys=True).lower()
    for forbidden in ("canonical_label", "target_test", "control_test", '"select"', '"audit"', '"cal0"', '"cal1"'):
        if forbidden in serialized:
            raise ValueError("label/evaluation data leaked into frozen worker")
    if sha256_file(job["selected_checkpoint_ref"]) != job["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")


def export(job):
    import torch
    import torch.nn.functional as functional
    validate_job(job)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    construction = {"source_job": {"model_id": job["model_id"], "initialization": job["initialization"]},
                    "execution": {"architecture": job["architecture"]}}
    adapter, patch = build_author_model(construction, device)
    checkpoint = torch.load(job["selected_checkpoint_ref"], map_location=device)
    if checkpoint.get("task_weight_origin") != "trained_in_project":
        raise ValueError("checkpoint is not an in-project task training artifact")
    if checkpoint.get("patch") != job["training_patch"]:
        raise ValueError("checkpoint training patch provenance mismatch")
    adapter.model.load_state_dict(checkpoint["model_state"], strict=True)
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
    after_modes = {name: module.training for name, module in adapter.model.named_modules()}
    after_buffers = dict(adapter.model.named_buffers())
    atol = rtol = 1e-5
    if not torch.allclose(author_logits, head_logits, atol=atol, rtol=rtol):
        raise ValueError("author forward and head(embedding) parity failed")
    if not torch.allclose(author_score, exported_score, atol=atol, rtol=rtol):
        raise ValueError("exported linear score parity failed")
    if before_modes != after_modes or any(not torch.equal(value, after_buffers[name].detach().cpu())
                                          for name, value in before_buffers.items()):
        raise ValueError("frozen forward changed module mode or buffers")
    destination = os.path.abspath(job["output_dir"])
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)
    try:
        state_path = os.path.join(temporary, "detector_state.pt")
        head_path = os.path.join(temporary, "linear_head.pt")
        torch.save({"schema_version": "0.1.0", "model_id": job["model_id"],
                    "model_state": adapter.model.state_dict(), "architecture": job["architecture"],
                    "model_construction_patch": patch, "training_patch": job["training_patch"],
                    "class_index_map": mapping}, state_path)
        torch.save({"schema_version": "0.1.0", "w": weight.detach().cpu(), "b": bias.detach().cpu(),
                    "score_direction": "larger_is_spoof"}, head_path)
        parity = {"schema_version": "0.1.0", "status": "PASS", "fixture_ref": job["fixture_audio_ref"],
                  "atol": atol, "rtol": rtol,
                  "author_head_max_abs": float((author_logits - head_logits).abs().max().cpu()),
                  "head_score_max_abs": float((author_score - exported_score).abs().max().cpu()),
                  "module_modes_stable": True, "buffers_stable": True}
        write_json(os.path.join(temporary, "parity.json"), parity)
        fields = job["bundle_fields"]
        baseline_basis = job["selected_checkpoint_sha256"] + sha256_file(head_path) + "author_wrapper_v1"
        bundle = {"schema_version": "0.1.0", "model_id": job["model_id"],
                  "baseline_id": "baseline-" + hashlib.sha256(baseline_basis.encode()).hexdigest()[:20],
                  "selected_checkpoint_sha256": job["selected_checkpoint_sha256"],
                  "training_run_id": fields["training_run_id"],
                  "fit_snapshot_hash": fields["fit_snapshot_hash"],
                  "source_val_snapshot_hash": fields["source_val_snapshot_hash"],
                  "recipe_hash": fields["recipe_hash"], "init_provenance": fields["init_provenance"],
                  "task_training_provenance": fields["task_training_provenance"],
                  "eval_preprocess_hash": fields["eval_preprocess_hash"],
                  "class_index_map": mapping, "head_ref": "linear_head.pt",
                  "embedding_dim": fields["embedding_dim"], "training_status": "FINALIZED",
                  "training_phase": "full", "task_weight_origin": "trained_in_project",
                  "source_val_selection_ref": fields["source_val_selection_ref"],
                  "parity_report_ref": "parity.json"}
        write_json(os.path.join(temporary, "bundle.json"), bundle)
        export_manifest = {"schema_version": "0.1.0", "status": "LOCKED", "immutable": True,
                           "files": {name: sha256_file(os.path.join(temporary, name)) for name in
                                     ("detector_state.pt", "linear_head.pt", "parity.json", "bundle.json")}}
        write_json(os.path.join(temporary, "export_manifest.json"), export_manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def _safe_audio_path(root, relative):
    candidate = os.path.realpath(os.path.join(root, relative))
    if os.path.commonpath([os.path.realpath(root), candidate]) != os.path.realpath(root):
        raise ValueError("audio path escapes its bound root")
    return candidate


def _views(waveform, sample_id, probe):
    import torch
    seed_text = "%s\0%s" % (probe["seed"], sample_id)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_rms = waveform.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    snr = float(probe["noise_snr_db"])
    noisy = waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-snr / 20.0))
    gain = float(probe["fir_side_gain"])
    padded = torch.nn.functional.pad(waveform[None, None], (1, 1), mode="reflect")
    kernel = torch.tensor([gain, 1.0, -gain], dtype=waveform.dtype).view(1, 1, 3)
    filtered = torch.nn.functional.conv1d(padded, kernel).view(-1)
    return torch.stack([waveform, noisy, filtered])


def extract(job):
    import numpy
    import torch
    allowed = {"schema_version", "job_type", "purpose", "input_role", "bundle_ref", "manifest_ref", "manifest_sha256",
               "data_roots", "probe", "numerical_mode", "worker_slot", "worker_count",
               "expected_ids", "worker_sha256", "cache_identity", "cache_key", "output_dir"}
    if set(job) != allowed or job["job_type"] != "inference" or job["schema_version"] != "0.1.0":
        raise ValueError("invalid inference job")
    if sha256_file(os.path.abspath(__file__)) != job["worker_sha256"]:
        raise ValueError("extraction worker hash mismatch")
    role_policy = {"source_prepare": {"fit", "cal0"}, "select": {"select"},
                   "confirmatory": {"control_test", "target_test"}}
    if job["purpose"] not in role_policy or job["input_role"] not in role_policy[job["purpose"]]:
        raise ValueError("inference purpose/input role violates the stage permission policy")
    if sha256_file(job["manifest_ref"]) != job["manifest_sha256"]:
        raise ValueError("inference manifest hash mismatch")
    if job["probe"].get("num_views") != 3 or set(job["probe"]) != {
            "num_views", "seed", "noise_snr_db", "fir_side_gain"}:
        raise ValueError("unsupported probe contract")
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
        raise ValueError("inference requires a finalized in-project full source model")
    with open(os.path.join(bundle_root, bundle["parity_report_ref"]), encoding="utf-8") as stream:
        parity = json.load(stream)
    if (parity.get("status") != "PASS" or parity.get("module_modes_stable") is not True or
            parity.get("buffers_stable") is not True):
        raise ValueError("inference requires frozen parity PASS")
    selection_ref = bundle["source_val_selection_ref"]
    if sha256_file(selection_ref) != bundle["task_training_provenance"].get(
            "source_val_selection_sha256"):
        raise ValueError("source_val selection changed after export")
    with open(selection_ref, encoding="utf-8") as stream:
        selection = json.load(stream)
    expected_selection = {"status": "FINALIZED", "training_phase": "full",
                          "task_weight_origin": "trained_in_project",
                          "selected_checkpoint_sha256": bundle["selected_checkpoint_sha256"],
                          "training_run_id": bundle["training_run_id"],
                          "fit_snapshot_hash": bundle["fit_snapshot_hash"],
                          "source_val_snapshot_hash": bundle["source_val_snapshot_hash"],
                          "recipe_hash": bundle["recipe_hash"]}
    if any(selection.get(key) != value for key, value in expected_selection.items()):
        raise ValueError("frozen bundle and source_val selection disagree")
    state = torch.load(os.path.join(bundle_root, "detector_state.pt"), map_location="cpu")
    construction = {"source_job": {"model_id": bundle["model_id"],
                                    "initialization": bundle["init_provenance"] if
                                    bundle["model_id"] == "ssl_aasist_source" else None},
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
            if set(row) != {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}:
                raise ValueError("inference worker received annotations")
            if row["split_role"] != job["input_role"]:
                raise ValueError("inference row violates locked input role")
            if row["sample_id"] in expected:
                rows.append(row)
    if {row["sample_id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("inference shard ID coverage mismatch")
    destination = os.path.abspath(job["output_dir"])
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)
    try:
        os.mkdir(os.path.join(temporary, "chunks"))
        chunks = []
        block = []
        block_ids = []
        block_size = int(job["numerical_mode"].get("block_units", 256))
        def flush():
            if not block:
                return
            index = len(chunks)
            array = numpy.stack(block).astype(job["numerical_mode"]["dtype"], copy=False)
            array_name = "chunk-%06d.npy" % index
            ids_name = "chunk-%06d.ids.json" % index
            array_path = os.path.join(temporary, "chunks", array_name)
            ids_path = os.path.join(temporary, "chunks", ids_name)
            with open(array_path, "xb") as output:
                numpy.save(output, array, allow_pickle=False)
            write_json(ids_path, list(block_ids))
            chunks.append({"index": index, "array_ref": "chunks/" + array_name,
                           "array_sha256": sha256_file(array_path), "ids_ref": "chunks/" + ids_name,
                           "ids_sha256": sha256_file(ids_path), "count": len(block_ids),
                           "shape": list(array.shape), "dtype": str(array.dtype)})
            del block[:]
            del block_ids[:]
        with torch.no_grad():
            for row in rows:
                audio_path = _safe_audio_path(job["data_roots"][row["root_key"]], row["audio_relpath"])
                if sha256_file(audio_path) != row["input_sha256"]:
                    raise ValueError("audio content changed after manifest: %s" % row["sample_id"])
                waveform = torch.from_numpy(load_audio(audio_path))
                views = _views(waveform, row["sample_id"], job["probe"]).to(device)
                embedding, _logits = adapter.forward(views, freq_aug=False)
                block.append(embedding.detach().cpu().numpy())
                block_ids.append(row["sample_id"])
                if len(block) >= block_size:
                    flush()
        flush()
        metadata = {"schema_version": "0.1.0", "status": "LOCKED", "format": "sharded_npy_v1",
                    "allow_pickle": False, "cache_key": job["cache_key"], "identity": job["cache_identity"],
                    "num_views": 3, "feature_dim": int(bundle["embedding_dim"]),
                    "sample_count": len(rows),
                    "expected_ids_sha256": hashlib.sha256(json.dumps(job["expected_ids"], sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                    "chunks": chunks, "immutable": True}
        write_json(os.path.join(temporary, "index.json"), metadata)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("export")
    command.add_argument("--job", required=True)
    extract_command = sub.add_parser("extract")
    extract_command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job, encoding="utf-8") as stream:
        job = json.load(stream)
    if args.command == "export":
        export(job)
    else:
        extract(job)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"schema_version": "0.1.0", "status": "FAILED",
                          "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        sys.exit(2)

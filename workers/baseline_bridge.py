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
import time

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


def _validate_r4_job(job, kind):
    required = {"schema_version", "job_type", "model_id", "architecture", "initialization",
                "selected_checkpoint_ref", "selected_checkpoint_sha256", "training_patch",
                "candidate_dir", "validation_dir", "data_roots", "fit_manifest_ref",
                "fit_manifest_sha256", "source_val_manifest_ref", "source_val_manifest_sha256",
                "fit_ids", "source_val_ids", "class_index_map", "embedding_dim", "batch_sizes",
                "atol", "rtol", "worker_sha256", "gpu_hour_cap"}
    if set(job) != required or job.get("schema_version") != "0.1.0" or job.get("job_type") != kind:
        raise ValueError("invalid R4 job")
    if sha256_file(os.path.abspath(__file__)) != job["worker_sha256"]:
        raise ValueError("R4 worker hash mismatch")
    if sha256_file(job["selected_checkpoint_ref"]) != job["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")
    if job["model_id"] != "aasist_source" or job["embedding_dim"] != 160:
        raise ValueError("R4 job is not the audited AASIST/160-d contract")
    if job["class_index_map"] != {"spoof": 0, "bonafide": 1}:
        raise ValueError("unexpected native class mapping")
    if not (0 < float(job["gpu_hour_cap"]) <= 0.5):
        raise ValueError("R4 GPU-hour cap exceeds authorization")
    # Historical training provenance legitimately lists forbidden roles.  Leak
    # scanning is therefore restricted to the actual inference data binding.
    serialized = json.dumps({key: job[key] for key in
                             ("fit_manifest_ref", "source_val_manifest_ref", "fit_ids",
                              "source_val_ids", "data_roots")}, sort_keys=True).lower()
    for forbidden in ("canonical_label", "target_test", "control_test", '"select"', '"audit"', '"cal0"'):
        if forbidden in serialized:
            raise ValueError("annotations or forbidden roles leaked into R4 worker")


def _load_checkpoint(path, device):
    import torch
    # This is a controlled in-project checkpoint whose SHA-256 is checked before deserialization.
    return torch.load(path, map_location=device)


def r4_export_candidate(job):
    """Export weights only.  This command never writes an R5-eligible bundle."""
    import torch
    _validate_r4_job(job, "r4_export_candidate")
    device = torch.device("cpu")
    construction = {"source_job": {"model_id": job["model_id"], "initialization": job["initialization"]},
                    "execution": {"architecture": job["architecture"]}}
    adapter, construction_patch = build_author_model(construction, device)
    checkpoint = _load_checkpoint(job["selected_checkpoint_ref"], device)
    if checkpoint.get("task_weight_origin") != "trained_in_project":
        raise ValueError("checkpoint is not an in-project task artifact")
    if checkpoint.get("patch") != job["training_patch"]:
        raise ValueError("checkpoint training patch provenance mismatch")
    adapter.model.load_state_dict(checkpoint["model_state"], strict=True)
    adapter.model.eval()
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    mapping = job["class_index_map"]
    weight = (adapter.model.out_layer.weight[mapping["spoof"]] -
              adapter.model.out_layer.weight[mapping["bonafide"]]).detach().cpu()
    bias = (adapter.model.out_layer.bias[mapping["spoof"]] -
            adapter.model.out_layer.bias[mapping["bonafide"]]).detach().cpu()
    destination = os.path.abspath(job["candidate_dir"])
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)
    try:
        torch.save({"schema_version": "0.1.0", "model_id": job["model_id"],
                    "model_state": adapter.model.state_dict(), "architecture": job["architecture"],
                    "model_construction_patch": construction_patch,
                    "training_patch": job["training_patch"], "class_index_map": mapping},
                   os.path.join(temporary, "detector_state.pt"))
        torch.save({"schema_version": "0.1.0", "w": weight, "b": bias,
                    "score_formula": "native_logits[spoof]-native_logits[bonafide]",
                    "score_direction": "larger_is_spoof", "output_type": "logit_difference",
                    "unit": "dimensionless"}, os.path.join(temporary, "linear_head.pt"))
        shutil.copy2(os.path.abspath(__file__), os.path.join(temporary, "baseline_bridge.py"))
        shutil.copy2(os.path.join(HERE, "compat", "author_training.py"),
                     os.path.join(temporary, "author_training.py"))
        files = {name: sha256_file(os.path.join(temporary, name)) for name in
                 ("detector_state.pt", "linear_head.pt", "baseline_bridge.py", "author_training.py")}
        write_json(os.path.join(temporary, "candidate_manifest.json"),
                   {"schema_version": "0.1.0", "status": "CANDIDATE_NOT_R5_ELIGIBLE",
                    "selected_checkpoint_sha256": job["selected_checkpoint_sha256"], "files": files})
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def _read_unlabeled_manifest(path, expected_hash, expected_ids, role, data_roots):
    if sha256_file(path) != expected_hash:
        raise ValueError("%s R4 manifest hash mismatch" % role)
    expected = set(expected_ids)
    rows, seen = [], set()
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            allowed = {"schema_version", "sample_id", "root_key", "audio_relpath", "input_sha256", "split_role"}
            if set(row) != allowed or row["split_role"] != role:
                raise ValueError("R4 worker received annotated or mismatched row %d" % number)
            if row["sample_id"] in seen:
                raise ValueError("duplicate R4 UID: %s" % row["sample_id"])
            seen.add(row["sample_id"])
            if row["sample_id"] in expected:
                if row["root_key"] not in data_roots:
                    raise ValueError("unbound R4 data root")
                rows.append(row)
    if len(rows) != len(expected_ids) or {r["sample_id"] for r in rows} != expected:
        raise ValueError("R4 UID coverage mismatch")
    order = {sample_id: index for index, sample_id in enumerate(expected_ids)}
    return sorted(rows, key=lambda row: order[row["sample_id"]])


def _tensor_attr_inventory(model):
    import torch
    registered = {id(value) for value in list(model.parameters()) + list(model.buffers())}
    result = []
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor) and id(value) not in registered:
                result.append({"module": module_name, "name": name, "shape": list(value.shape),
                               "dtype": str(value.dtype), "device": str(value.device),
                               "derived_cache": name == "filters"})
    return result


def _error_stats(values, atol, rtol):
    import numpy
    array = numpy.asarray(values, dtype="float64")
    if array.size == 0:
        return {"count": 0, "max_abs": 0.0, "p50_abs": 0.0, "p95_abs": 0.0,
                "p99_abs": 0.0, "over_atol_count": 0}
    return {"count": int(array.size), "max_abs": float(array.max()),
            "p50_abs": float(numpy.quantile(array, .50)), "p95_abs": float(numpy.quantile(array, .95)),
            "p99_abs": float(numpy.quantile(array, .99)),
            "over_atol_count": int((array > float(atol)).sum()), "rtol": float(rtol)}


def r4_verify(job):
    """Reload candidate in a new process and compare independent reference/export instances."""
    import numpy
    import torch
    import torch.nn.functional as functional
    _validate_r4_job(job, "r4_verify")
    if not torch.cuda.is_available():
        raise RuntimeError("R4 real validation requires an available CUDA device")
    started = time.monotonic()
    device = torch.device("cuda")
    construction = {"source_job": {"model_id": job["model_id"], "initialization": job["initialization"]},
                    "execution": {"architecture": job["architecture"]}}
    reference, _ = build_author_model(construction, device)
    exported, _ = build_author_model(construction, device)
    checkpoint = _load_checkpoint(job["selected_checkpoint_ref"], device)
    candidate_state = _load_checkpoint(os.path.join(job["candidate_dir"], "detector_state.pt"), device)
    head = _load_checkpoint(os.path.join(job["candidate_dir"], "linear_head.pt"), device)
    reference.model.load_state_dict(checkpoint["model_state"], strict=True)
    exported.model.load_state_dict(candidate_state["model_state"], strict=True)
    for model in (reference.model, exported.model):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    ref_state, exp_state = reference.model.state_dict(), exported.model.state_dict()
    if set(ref_state) != set(exp_state) or any(not torch.equal(ref_state[k], exp_state[k]) for k in ref_state):
        raise ValueError("exported parameters/persistent buffers differ from reference")
    before_modes = [{n: m.training for n, m in model.named_modules()} for model in
                    (reference.model, exported.model)]
    before_buffers = [{n: v.detach().cpu().clone() for n, v in model.named_buffers()} for model in
                      (reference.model, exported.model)]
    before_attrs = [_tensor_attr_inventory(model) for model in (reference.model, exported.model)]
    fit_rows = _read_unlabeled_manifest(job["fit_manifest_ref"], job["fit_manifest_sha256"],
                                        job["fit_ids"], "fit", job["data_roots"])
    val_rows = _read_unlabeled_manifest(job["source_val_manifest_ref"], job["source_val_manifest_sha256"],
                                        job["source_val_ids"], "source_val", job["data_roots"])
    mapping, atol, rtol = job["class_index_map"], float(job["atol"]), float(job["rtol"])
    weight, bias = head["w"].to(device), head["b"].to(device)
    per_sample, errors = [], {name: [] for name in ("embedding", "native_logits", "head_logits", "score")}

    def run_rows(rows, batch_size, phase, sequence):
        for start in range(0, len(rows), batch_size):
            if time.monotonic() - started >= float(job["gpu_hour_cap"]) * 3600.0:
                raise RuntimeError("R4 GPU-hour hard cap reached before completion")
            batch_rows = rows[start:start + batch_size]
            waveforms = []
            for row in batch_rows:
                audio_path = _safe_audio_path(job["data_roots"][row["root_key"]], row["audio_relpath"])
                if row["input_sha256"] is not None and sha256_file(audio_path) != row["input_sha256"]:
                    raise ValueError("audio content hash mismatch: %s" % row["sample_id"])
                waveforms.append(load_audio(audio_path))
            waveform = torch.from_numpy(numpy.stack(waveforms)).to(device)
            with torch.no_grad():
                ref_embedding, ref_logits = reference.forward(waveform, freq_aug=False)
                exp_embedding, exp_logits = exported.forward(waveform, freq_aug=False)
                head_logits = functional.linear(exp_embedding, exported.model.out_layer.weight,
                                                exported.model.out_layer.bias)
                export_scores = exp_embedding.matmul(weight) + bias
                ref_scores = ref_logits[:, mapping["spoof"]] - ref_logits[:, mapping["bonafide"]]
            if list(ref_embedding.shape) != [len(batch_rows), 160] or list(ref_logits.shape) != [len(batch_rows), 2]:
                raise ValueError("unexpected AASIST embedding/logit shape")
            tensors = {"embedding": (ref_embedding, exp_embedding),
                       "native_logits": (ref_logits, exp_logits),
                       "head_logits": (exp_logits, head_logits), "score": (ref_scores, export_scores)}
            for index, row in enumerate(batch_rows):
                record = {"sample_id": row["sample_id"], "phase": phase, "sequence": sequence,
                          "batch_size": batch_size, "reference_score": float(ref_scores[index].cpu()),
                          "export_score": float(export_scores[index].cpu())}
                passed = True
                for name, (left, right) in tensors.items():
                    delta = (left[index] - right[index]).abs()
                    maximum = float(delta.max().cpu())
                    denom = right[index].abs().clamp_min(1e-12)
                    relative = float((delta / denom).max().cpu())
                    within = bool(torch.allclose(left[index], right[index], atol=atol, rtol=rtol))
                    record[name + "_max_abs"] = maximum
                    record[name + "_max_relative_stable"] = relative
                    errors[name].append(maximum)
                    passed = passed and within
                record["within_tolerance"] = passed
                per_sample.append(record)
                if not passed:
                    raise ValueError("R4 parity tolerance exceeded for %s" % row["sample_id"])

    # Fixed predeclared batch coverage, including tail, reversed order and two shards.
    for size in job["batch_sizes"]:
        run_rows(fit_rows, int(size), "fit_128", "forward")
    run_rows(list(reversed(fit_rows)), 7, "fit_128", "reversed")
    run_rows(fit_rows[::2], 48, "fit_128", "shard_even")
    run_rows(fit_rows[1::2], 48, "fit_128", "shard_odd")
    run_rows(fit_rows[:7], 7, "fit_128", "A_first")
    run_rows(fit_rows[7:14], 7, "fit_128", "B")
    run_rows(fit_rows[:7], 7, "fit_128", "A_repeat")
    run_rows(val_rows, 48, "source_val_full", "forward")

    # Gradient reaches a disposable embedding input while model/head parameters remain frozen.
    waveform = torch.from_numpy(numpy.stack([load_audio(_safe_audio_path(
        job["data_roots"][fit_rows[0]["root_key"]], fit_rows[0]["audio_relpath"]))])).to(device)
    with torch.no_grad():
        probe_embedding, _ = exported.forward(waveform, freq_aug=False)
    probe = probe_embedding.detach().clone().requires_grad_(True)
    probe_score = probe.matmul(weight.detach()) + bias.detach()
    probe_score.sum().backward()
    gradient_pass = (probe.grad is not None and bool(torch.isfinite(probe.grad).all()) and
                     float(probe.grad.abs().sum().cpu()) > 0 and
                     all(parameter.grad is None for parameter in exported.model.parameters()))
    after_modes = [{n: m.training for n, m in model.named_modules()} for model in
                   (reference.model, exported.model)]
    after_buffers = [{n: v.detach().cpu() for n, v in model.named_buffers()} for model in
                     (reference.model, exported.model)]
    modes_stable = before_modes == after_modes and all(not any(m.values()) for m in after_modes)
    buffers_stable = all(all(torch.equal(value, after_buffers[i][name]) for name, value in before_buffers[i].items())
                         for i in range(2))
    after_attrs = [_tensor_attr_inventory(model) for model in (reference.model, exported.model)]
    derived_filters = all(any(item["name"] == "filters" and item["derived_cache"] for item in attrs)
                          for attrs in after_attrs)
    elapsed = time.monotonic() - started
    gpu_hours = elapsed / 3600.0
    status = "PASS" if modes_stable and buffers_stable and gradient_pass and derived_filters and gpu_hours <= float(job["gpu_hour_cap"]) else "FAIL"
    destination = os.path.abspath(job["validation_dir"])
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix="." + os.path.basename(destination) + ".", dir=parent)
    try:
        with open(os.path.join(temporary, "per_sample.jsonl"), "x", encoding="utf-8") as stream:
            for row in per_sample:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        report = {"schema_version": "0.1.0", "status": status,
                  "independent_instances": True, "candidate_loaded_in_new_process": True,
                  "device": str(device), "dtype": "float32", "amp": False,
                  "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                  "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                  "freq_aug": False, "atol": atol, "rtol": rtol,
                  "embedding_shape": ["B", 160], "native_logits_shape": ["B", 2],
                  "score_formula": "native_logits[spoof]-native_logits[bonafide]",
                  "score_direction": "larger_is_spoof", "module_modes_stable": modes_stable,
                  "buffers_stable": buffers_stable, "parameters_frozen": True,
                  "embedding_input_gradient_pass": gradient_pass,
                  "unregistered_tensor_attrs_before": before_attrs,
                  "unregistered_tensor_attrs_after": after_attrs,
                  "deterministic_filters_rebuilt": derived_filters,
                  "fit_unique_count": len(fit_rows), "source_val_unique_count": len(val_rows),
                  "per_sample_record_count": len(per_sample),
                  "error_statistics": {name: _error_stats(values, atol, rtol) for name, values in errors.items()},
                  "elapsed_seconds": elapsed, "gpu_hours": gpu_hours,
                  "gpu_hour_cap": float(job["gpu_hour_cap"]),
                  "per_sample_ref": "per_sample.jsonl"}
        write_json(os.path.join(temporary, "parity.json"), report)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    if status != "PASS":
        raise ValueError("R4 validation failed; diagnostic report retained")


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


def _apply_numerical_mode(mode):
    """Apply the hash-bound backend mode instead of relying on process defaults."""
    import torch
    if not isinstance(mode, dict) or mode.get("dtype") not in ("float16", "float32", "float64"):
        raise ValueError("invalid extraction numerical mode")
    if type(mode.get("block_units")) is not int or mode["block_units"] < 1:
        raise ValueError("numerical block_units must be a positive integer")
    if "tf32" in mode:
        if set(mode) != {"dtype", "tf32", "block_units"} or type(mode["tf32"]) is not bool:
            raise ValueError("legacy numerical mode must contain a boolean tf32 flag")
        matmul = cudnn = mode["tf32"]
    else:
        allowed = {"dtype", "tf32_matmul", "tf32_cudnn", "block_units", "input_condition"}
        if not set(mode) <= allowed or not {"dtype", "tf32_matmul", "tf32_cudnn", "block_units"} <= set(mode) or any(
                type(mode[name]) is not bool for name in ("tf32_matmul", "tf32_cudnn")):
            raise ValueError("numerical mode must bind boolean TF32 matmul/cuDNN flags")
        matmul, cudnn = mode["tf32_matmul"], mode["tf32_cudnn"]
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn
    actual = {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
              "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if actual != {"tf32_matmul": matmul, "tf32_cudnn": cudnn}:
        raise RuntimeError("PyTorch did not apply the locked TF32 numerical mode")
    return actual


def _apply_condition(waveform, sample_id, condition):
    """Apply a deterministic input condition (AWGN) with an independent RNG namespace."""
    import torch
    if not condition:
        return waveform
    if condition.get("kind") != "awgn":
        raise ValueError("unsupported input condition kind")
    if type(condition.get("snr_db")) not in (int, float) or type(condition.get("seed")) is not int:
        raise ValueError("input condition must bind snr_db and integer seed")
    seed_text = "%s\0%s\0%s\0%s\0%s" % (condition["seed"], condition.get("namespace", ""),
                                        condition.get("purpose", "input_condition"), sample_id, "input_condition")
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
    signal_rms = waveform.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    noise_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(waveform.dtype).tiny)
    return waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-float(condition["snr_db"]) / 20.0))


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
    if job["cache_identity"].get("numerical_mode") != job["numerical_mode"]:
        raise ValueError("cache identity does not bind the requested numerical mode")
    numerical_execution = _apply_numerical_mode(job["numerical_mode"])
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
                if row["input_sha256"] is not None and sha256_file(audio_path) != row["input_sha256"]:
                    raise ValueError("audio content changed after manifest: %s" % row["sample_id"])
                waveform = torch.from_numpy(load_audio(audio_path))
                waveform = _apply_condition(waveform, row["sample_id"],
                                            job["numerical_mode"].get("input_condition"))
                views = _views(waveform, row["sample_id"], job["probe"]).to(device)
                embedding, _logits = adapter.forward(views, freq_aug=False)
                block.append(embedding.detach().cpu().numpy())
                block_ids.append(row["sample_id"])
                if len(block) >= block_size:
                    flush()
        flush()
        metadata = {"schema_version": "0.1.0", "status": "LOCKED", "format": "sharded_npy_v1",
                    "allow_pickle": False, "cache_key": job["cache_key"], "identity": job["cache_identity"],
                    "numerical_execution": numerical_execution,
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
    candidate_command = sub.add_parser("r4-export-candidate")
    candidate_command.add_argument("--job", required=True)
    verify_command = sub.add_parser("r4-verify")
    verify_command.add_argument("--job", required=True)
    extract_command = sub.add_parser("extract")
    extract_command.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    with open(args.job, encoding="utf-8") as stream:
        job = json.load(stream)
    if args.command == "export":
        export(job)
    elif args.command == "r4-export-candidate":
        r4_export_candidate(job)
    elif args.command == "r4-verify":
        r4_verify(job)
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

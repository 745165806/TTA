"""Bounded real SSL-AASIST source-audio smoke; never opens target manifests."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch
from torch.func import functional_call
from torch.nn import functional as F

from author_training import build_author_model, load_audio
from baselines.probe import three_view_probe
from eptta.baselines.ports.audio_native import SCOPE_A, preregistered_parameter_names
from core import BYOLSystem, configure_meta, episodic_scores, meta_objective, score


def _read_fit_pair(manifest, audio_root):
    rows = []
    groups = set()
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split_role") != "fit" or row.get("canonical_label") not in (0, 1):
                raise ValueError("smoke accepts fit rows with explicit labels only")
            if not rows or row["sample_id"] != rows[0]["sample_id"]:
                rows.append(row)
            if len(rows) == 2:
                break
    if len(rows) != 2:
        raise ValueError("fit has fewer than two independent IDs")
    for row in rows:
        rel = Path(row["audio_relpath"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe source audio path")
        path = (audio_root / rel).resolve()
        if not path.is_relative_to(audio_root.resolve()):
            raise ValueError("source audio path escapes root")
        row["waveform"] = torch.from_numpy(load_audio(path))
    return rows


def run(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite smoke report")
    torch.manual_seed(13)
    if args.device.startswith("cuda:"):
        torch.cuda.manual_seed_all(13)
    bundle_path = args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("task_weight_origin") != "trained_in_project":
        raise ValueError("smoke requires project-trained detector")
    state = torch.load(bundle_path.parent / bundle["detector_state_ref"], map_location="cpu",
                       weights_only=True)
    device = torch.device(args.device)
    construction = {"source_job": {"model_id": bundle["model_id"],
                                   "initialization": bundle["initialization"]},
                    "execution": {"architecture": state["architecture"]}}
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    adapter.model.eval()
    target_adapter, _target_patch = build_author_model(construction, device)
    rows = _read_fit_pair(args.fit_manifest, args.audio_root)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    system = BYOLSystem(adapter.model, target_adapter.model, bundle["class_index_map"]).to(device)
    chosen = configure_meta(system)
    backbone = sorted(name.removeprefix("online_model.") for name in chosen
                      if name.startswith("online_model."))
    inactive = {f"encoder.{i}.0.bn1.{part}" for i in range(1, 6)
                for part in ("weight", "bias")}
    expected = sorted(set(preregistered_parameter_names(system.online_model, SCOPE_A)) - inactive)
    if backbone != expected:
        raise AssertionError("real backend BN affine whitelist differs from audited scope")
    before_buffers = {name: value.detach().clone() for name, value in system.named_buffers()}
    before_parameters = {name: value.detach().clone() for name, value in system.named_parameters()}
    before_target = {name: value.detach().clone() for name, value in
                     system.target_model.state_dict().items()}

    def inputs(row):
        views = three_view_probe(row["waveform"], row["sample_index"])[[0, 2]].to(device)
        return views[:1], views

    original_a, views_a = inputs(rows[0])
    original_b, views_b = inputs(rows[1])
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    direct_k0 = score(system, original_a).detach()
    k0_a, k1_a, byol_loss, inner_grads = episodic_scores(
        system, original_a, views_a, args.inner_lr)
    _k0_b, _k1_b, _loss_b, _grads_b = episodic_scores(
        system, original_b, views_b, args.inner_lr)
    k0_again, k1_again, _loss_again, _grads_again = episodic_scores(
        system, original_a, views_a, args.inner_lr)
    if abs(float(k0_a - direct_k0)) > 1e-5:
        raise AssertionError("K=0 native path mismatch")
    if abs(float(k0_again - k0_a)) > 1e-6 or abs(float(k1_again - k1_a)) > 1e-6:
        raise AssertionError("A-B-A episode reset mismatch")
    backbone_grad = sum(float(inner_grads[name].detach().abs().sum()) for name in chosen
                        if name.startswith("online_model."))
    if backbone_grad <= 0 or abs(float(k1_a - k0_a)) <= 1e-8:
        raise AssertionError("real backbone BN update did not affect spoof score")
    outer, meta = meta_objective(system, views_a, views_b,
                                 rows[1]["canonical_label"], args.inner_lr)
    params = dict(system.named_parameters())
    outer_grads = torch.autograd.grad(outer, [params[name] for name in chosen],
                                      allow_unused=False)
    # Compare with the same fast update after detaching its inner gradients.
    # A nonzero difference directly tests the Hessian contribution on the real model.
    first_order_fast = dict(params)
    for name in chosen:
        first_order_fast[name] = params[name] - args.inner_lr * meta["inner_gradients"][name].detach()
    first_logits = functional_call(system, first_order_fast, (views_b[:1],),
                                   {"mode": "logits"}, strict=False)
    first_ce = F.cross_entropy(first_logits, torch.tensor(
        [1 - rows[1]["canonical_label"]], device=device))
    first_ssl = functional_call(system, first_order_fast, (views_b,),
                                {"mode": "byol"}, strict=False)
    first_grads = torch.autograd.grad(first_ce + 0.1 * first_ssl,
                                      [params[name] for name in chosen], allow_unused=False)
    hessian_contribution = sum(float((second - first).detach().abs().sum())
                               for second, first in zip(outer_grads, first_grads))
    if hessian_contribution <= 1e-7:
        raise AssertionError("real-model second-order contribution is absent")
    same_outer, _same_meta = meta_objective(system, views_a, views_a,
                                            rows[0]["canonical_label"], args.inner_lr)
    same_grads = torch.autograd.grad(same_outer, [params[name] for name in chosen],
                                     allow_unused=False)
    if not any(g.requires_grad for g in meta["inner_gradients"].values()):
        raise AssertionError("inner gradient lost second-order graph")
    if not all(torch.isfinite(g).all() for g in (*outer_grads, *same_grads)):
        raise FloatingPointError("non-finite second-order meta gradient")
    if not any(float(g.detach().abs().sum()) > 0 for g in outer_grads):
        raise AssertionError("second-order meta gradient is zero")
    for name, value in before_buffers.items():
        if not torch.equal(value, dict(system.named_buffers())[name]):
            raise AssertionError("BN or target buffer changed: " + name)
    for name, value in before_parameters.items():
        if not torch.equal(value, dict(system.named_parameters())[name]):
            raise AssertionError("episode changed initial parameter: " + name)
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise AssertionError("episode changed CPU random state")
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError("episode changed CUDA random state")
    for name, value in before_target.items():
        if not torch.equal(value, system.target_model.state_dict()[name]):
            raise AssertionError("EMA target changed within episode: " + name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"status": "PASS", "model": bundle["model_id"], "baseline_id": bundle["baseline_id"],
            "source_role": "fit", "support_sample_id": rows[0]["sample_id"],
            "query_sample_id": rows[1]["sample_id"], "backbone_bn_tensors": len(backbone),
            "backbone_bn_scalars": sum(params["online_model." + n].numel() for n in backbone),
            "inactive_author_bn_tensors": sorted(inactive),
            "total_fast_tensors": len(chosen), "k0": float(k0_a), "k1": float(k1_a),
            "score_change": float(k1_a - k0_a), "backbone_grad_l1": backbone_grad,
            "byol_loss": float(byol_loss), "cross_sample_meta_loss": float(outer.detach()),
            "same_sample_meta_loss": float(same_outer.detach()),
            "second_order_gradient_l1": sum(float(g.detach().abs().sum()) for g in outer_grads),
            "hessian_contribution_l1": hessian_contribution,
            "ema_unchanged": True, "buffers_unchanged": True,
            "parameters_unchanged": True, "torch_rng_unchanged": True,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inner-lr", type=float, default=3e-4)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

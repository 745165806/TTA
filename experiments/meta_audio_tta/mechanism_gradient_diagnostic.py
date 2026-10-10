"""Tiny, source-only real-model gradient alignment diagnostic (no training)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch
from torch.nn import functional as F
from torch.func import functional_call

from baselines.probe import three_view_probe
from core import BYOLSystem, configure_meta, fast_update, score, selected_bn_affine_names
from stage1_source import _dataset
from experiments.meta_audio_tta.meta_helpers import load_checkpoint as _load_checkpoint


def _write_jsonl(path: Path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _source_bn_names(model):
    from torch import nn
    names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        if module_name.startswith("ssl_model."):
            continue
        if module_name in {f"encoder.{i}.0.bn1" for i in range(1, 6)}:
            continue
        for local in ("weight", "bias"):
            if getattr(module, local) is not None:
                names.append(module_name + "." + local)
    if not names:
        raise RuntimeError("no live detector BN affine parameters")
    return tuple(sorted(names))


def _margin(model, waveform, params=None, system=None):
    if system is None:
        logits = model(waveform) if params is None else functional_call(
            model, params, (waveform,), strict=False)
    else:
        logits = system(waveform, mode="logits") if params is None else functional_call(
            system, params, (waveform,), {"mode": "logits"}, strict=False)
    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite or malformed anti-spoof logits")
    return logits[0, 0] - logits[0, 1]


def _ce_loss(margin, canonical_label):
    # Native output 0=spoof, 1=bonafide; canonical 0=bonafide, 1=spoof.
    native_label = 1 - canonical_label
    return F.binary_cross_entropy_with_logits(margin.reshape(1),
                                               torch.tensor([native_label], device=margin.device,
                                                            dtype=margin.dtype))


def run(args):
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(13008)
    config = json.loads(args.source_config.read_text())
    prior_path = args.prior_run / "joint" / "source_val.scores.jsonl"
    prior_rows = [json.loads(line) for line in prior_path.read_text().splitlines() if line]
    # Pre-registered subset: two smallest- and two largest-margin items per class,
    # ranked on the already fixed v1 CE Frozen source_val diagnostic.
    ce_path = args.prior_ce / "source_val.scores.jsonl"
    ce_rows = [json.loads(line) for line in ce_path.read_text().splitlines() if line]
    if {row["sample_id"] for row in prior_rows} != {row["sample_id"] for row in ce_rows}:
        raise ValueError("source-only cached CE/BYOL ID sets do not match")
    ce_by_id = {row["sample_id"]: row for row in ce_rows}
    fixed_rows = []
    for label in (0, 1):
        candidates = [row for row in ce_rows if row["canonical_label"] == label]
        candidates.sort(key=lambda row: (abs(row["k0"]), row["sample_id"]))
        if len(candidates) != 32:
            raise ValueError("expected fixed 32-per-class source_val subset")
        fixed_rows.extend(("hard", row) for row in candidates[:2])
        fixed_rows.extend(("easy", row) for row in candidates[-2:])
    if len({r[1]["sample_id"] for r in fixed_rows}) != 8:
        raise ValueError("fixed diagnostic IDs are not unique")
    source_val = _dataset(config, "source_val")
    index_by_id = {row["sample_id"]: i for i, row in enumerate(source_val.rows)}
    for _, row in fixed_rows:
        if row["sample_id"] not in index_by_id:
            raise ValueError("fixed source ID absent from existing source_val assignment")
    checkpoints = {
        "ce": args.stage1_root / "ce/checkpoints/epoch_0006.pt",
        "joint": args.stage1_root / "joint/checkpoints/epoch_0002.pt",
        "cross": args.stage2_root / "cross/checkpoints/epoch_0001.pt",
        "same": args.stage2_root / "same/checkpoints/epoch_0001.pt",
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cpu")
    manifest = {"status": "IN_PROGRESS", "device": "cpu", "torch_cuda_available": torch.cuda.is_available(),
                "source_role": "source_val", "source_val_n": 8,
                "class_difficulty_design": "CE Frozen margin; per class two smallest and two largest absolute margins among existing 32/class diagnostic subset",
                "acoustic_conditions": ["original", "deterministic_fir"],
                "learning_rate": 0.0003, "oracle": "single supervised CE gradient step on same sample; mechanism diagnostic only",
                "checkpoint_paths": {k: str(v) for k, v in checkpoints.items()},
                "source_config": str(args.source_config), "prior_source_run": str(args.prior_run),
                "prior_ce_source_run": str(args.prior_ce),
                "command": sys.argv, "threads": args.threads}
    (args.output / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_rows = []
    for variant, checkpoint in checkpoints.items():
        detector, system = _load_checkpoint(config, checkpoint, device)
        detector.eval()
        if system is not None:
            configure_meta(system)
        model = system if system is not None else detector
        names = (tuple(n for n in selected_bn_affine_names(system) if n.startswith("online_model."))
                 if system is not None else _source_bn_names(detector))
        params = OrderedDict(model.named_parameters())
        for difficulty, cached in fixed_rows:
            index = index_by_id[cached["sample_id"]]
            waveform, label, sample_id = source_val[index]
            sample_index = source_val.rows[index]["sample_index"]
            waveform = waveform.reshape(1, -1).to(device)
            probe = three_view_probe(waveform[0], sample_index)
            for condition, base_wave in (("original", waveform[0]),
                                         ("deterministic_fir", probe[2])):
                views = three_view_probe(base_wave, sample_index)[[0, 2]].to(device)
                original = views[:1]
                margin0 = _margin(detector, original, system=system)
                ce0 = _ce_loss(margin0, int(label))
                ce_grads = torch.autograd.grad(ce0, [params[n] for n in names],
                                               retain_graph=False, allow_unused=True)
                if any(g is None for g in ce_grads):
                    raise RuntimeError("CE gradient missing for selected shared BN affine")
                ce_vec = torch.cat([g.detach().reshape(-1).float() for g in ce_grads])
                if system is not None:
                    fast_byol, byol_loss, byol_grad_map = fast_update(
                        system, views, 0.0003, create_graph=False)
                    byol_grads = [byol_grad_map[n] for n in names]
                    byol_vec = torch.cat([g.detach().reshape(-1).float() for g in byol_grads])
                    byol_norm, ce_norm = byol_vec.norm(), ce_vec.norm()
                    dot = torch.dot(byol_vec, ce_vec)
                    cosine = dot / (byol_norm * ce_norm).clamp_min(1e-30)
                    fast_oracle = OrderedDict(params)
                    for name, grad in zip(names, ce_grads):
                        fast_oracle[name] = params[name] - 0.0003 * grad
                    margin_byol = _margin(detector, original, fast_byol, system)
                    margin_oracle = _margin(detector, original, fast_oracle, system)
                    ce_byol = _ce_loss(margin_byol, int(label))
                    ce_oracle = _ce_loss(margin_oracle, int(label))
                    matched_lr = 0.0003 * float(byol_norm / ce_norm.clamp_min(1e-30))
                    fast_oracle_matched = OrderedDict(params)
                    for name, grad in zip(names, ce_grads):
                        fast_oracle_matched[name] = params[name] - matched_lr * grad
                    margin_oracle_matched = _margin(detector, original, fast_oracle_matched, system)
                    ce_oracle_matched = _ce_loss(margin_oracle_matched, int(label))
                    aux = {"byol_loss": float(byol_loss.detach()),
                           "byol_gradient_l2": float(byol_norm),
                           "byol_update_l2": float((byol_vec * 0.0003).norm()),
                           "ce_gradient_l2": float(ce_norm), "gradient_dot": float(dot),
                           "gradient_cosine": float(cosine),
                           "byol_margin_after": float(margin_byol.detach()),
                           "byol_ce_loss_after": float(ce_byol.detach()),
                           "oracle_margin_after": float(margin_oracle.detach()),
                           "oracle_ce_loss_after": float(ce_oracle.detach()),
                           "oracle_matched_lr": matched_lr,
                           "oracle_matched_update_l2": float((ce_vec * matched_lr).norm()),
                           "oracle_matched_margin_after": float(margin_oracle_matched.detach()),
                           "oracle_matched_ce_loss_after": float(ce_oracle_matched.detach()),
                           "byol_margin_delta": float((margin_byol-margin0).detach()),
                           "oracle_margin_delta": float((margin_oracle-margin0).detach())}
                else:
                    oracle = OrderedDict(params)
                    for name, grad in zip(names, ce_grads):
                        oracle[name] = params[name] - 0.0003 * grad
                    margin_oracle = _margin(detector, original, oracle)
                    aux = {"byol_loss": None, "byol_gradient_l2": None,
                           "ce_gradient_l2": float(ce_vec.norm()), "gradient_dot": None,
                           "gradient_cosine": None, "byol_margin_after": None,
                           "byol_ce_loss_after": None,
                           "oracle_margin_after": float(margin_oracle.detach()),
                           "oracle_ce_loss_after": float(_ce_loss(margin_oracle, int(label)).detach()),
                           "oracle_matched_lr": None, "oracle_matched_update_l2": None,
                           "oracle_matched_margin_after": None, "oracle_matched_ce_loss_after": None,
                           "byol_margin_delta": None,
                           "oracle_margin_delta": float((margin_oracle-margin0).detach())}
                row = {"variant": variant, "sample_id": sample_id,
                       "canonical_label": int(label), "difficulty": difficulty,
                       "condition": condition, "k0_margin": float(margin0.detach()),
                       "k0_ce_loss": float(ce0.detach()), "gradient_parameter_count": ce_vec.numel(), **aux}
                if not all(v is None or math.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
                    raise FloatingPointError("nonfinite mechanism diagnostic value")
                all_rows.append(row)
                with (args.output / "per_sample.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        del system, detector, model, params
    (args.output / "run_config.json").write_text(json.dumps({**manifest, "status": "COMPLETE",
        "record_count": len(all_rows), "fixed_ids": [r[1]["sample_id"] for r in fixed_rows]}, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "records": len(all_rows), "output": str(args.output),
                      "device": str(device)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--prior-ce", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

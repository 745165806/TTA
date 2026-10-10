"""One bounded real-model optimizer step for CE or CE+BYOL resource sizing."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "workers/compat"),
                str(ROOT / "experiments/p2_calibration_baselines")]

import torch
from torch.nn import functional as F

from author_training import build_author_model
from baselines.probe import three_view_probe
from core import BYOLSystem
from smoke_real import _read_fit_pair


def run(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite resource report")
    bundle = json.loads((args.asset_root / "outputs_v2/ssl_aasist/frozen/bundle.json").read_text())
    state = torch.load(args.asset_root / "outputs_v2/ssl_aasist/frozen" /
                       bundle["detector_state_ref"], map_location="cpu", weights_only=True)
    construction = {"source_job": {"model_id": bundle["model_id"],
                                   "initialization": bundle["initialization"]},
                    "execution": {"architecture": state["architecture"]}}
    device = torch.device(args.device)
    adapter, _patch = build_author_model(construction, device)
    adapter.model.load_state_dict(state["model_state"], strict=True)
    detector = adapter.model
    rows = _read_fit_pair(args.fit_manifest, args.audio_root)
    views = [three_view_probe(row["waveform"], row["sample_index"])[[0, 2]]
             for row in rows]
    views = torch.stack([views[i % 2] for i in range(args.batch_size)]).to(device)
    canonical = torch.tensor([rows[i % 2]["canonical_label"] for i in range(args.batch_size)],
                             device=device)
    native = 1 - canonical
    weights = torch.tensor([0.1, 0.9], device=device)
    if args.mode == "joint":
        target, _ = build_author_model(construction, device)
        system = BYOLSystem(detector, target.model, bundle["class_index_map"]).to(device)
        system.source_train_mode()
        trainable = [p for p in system.parameters() if p.requires_grad]
    else:
        system = None
        detector.train()
        trainable = list(detector.parameters())
    optimizer = torch.optim.Adam(trainable, lr=1e-6, weight_decay=1e-4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    flat = views.reshape(args.batch_size * 2, -1)
    if system is not None:
        logits, online = system._online(flat)
    else:
        logits = detector(flat)
    supervised = F.cross_entropy(logits, native.repeat_interleave(2),
                                 weight=weights)
    if system is not None:
        # Each original audio has exactly two views. Source BYOL may use a
        # batch of pairs; this measurement also exercises the EMA branch.
        predictions = F.normalize(system.predictor(system.projector(online)), dim=-1)
        targets = F.normalize(system._target(flat), dim=-1).detach()
        a, b = predictions.reshape(args.batch_size, 2, -1).unbind(1)
        ta, tb = targets.reshape(args.batch_size, 2, -1).unbind(1)
        byol = (2 - 2 * (a * tb).sum(-1) + 2 - 2 * (b * ta).sum(-1)).mean() / 2
        loss = supervised + 0.1 * byol
    else:
        byol = None
        loss = supervised
    loss.backward()
    optimizer.step()
    if system is not None:
        system.update_ema()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {"status": "PASS", "mode": args.mode, "batch_original_audios": args.batch_size,
            "views_per_audio": 2, "source_role": "fit", "single_optimizer_step_seconds": elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
            "supervised_loss": float(supervised.detach()),
            "byol_loss": None if byol is None else float(byol.detach()),
            "loss": float(loss.detach()), "device": str(device),
            "task_checkpoint_used_for_engineering_only": bundle["baseline_id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("ce", "joint"), required=True)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("measurement batch requires at least two source audios")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

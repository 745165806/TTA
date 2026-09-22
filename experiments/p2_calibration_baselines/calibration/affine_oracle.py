#!/usr/bin/env python
"""P2-A: positive affine oracle calibration (POST_HOC_DEVELOPMENT_ONLY).

Fits score_cal = a * score + b with a > 0 via logistic BCE on target10 labels.
A positive affine transform is monotone, so EER/AUC/rank order must be unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from calibration.common import POST_HOC_MARKER, load_target10

torch.set_num_threads(1)


def nll_brier_ece(logits, labels, n_bins=15):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    prob = torch.sigmoid(logits)
    nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    brier = ((prob - labels) ** 2).mean().item()
    # 15-bin ECE on predicted spoof probability.
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    ece = 0.0
    for i in range(n_bins):
        mask = (prob > edges[i]) & (prob <= edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean().item()
        bin_conf = prob[mask].mean().item()
        ece += (mask.sum().item() / len(labels)) * abs(bin_conf - bin_acc)
    return nll, brier, ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.run_dir) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, labels, _ids, _res = load_target10()
    s = torch.tensor(scores, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)

    nll_before, brier_before, ece_before = nll_brier_ece(s, y)

    alpha = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([alpha, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        a = torch.nn.functional.softplus(alpha) + 1e-3
        logits = a * s + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    a_final = float(torch.nn.functional.softplus(alpha.detach()) + 1e-3)
    b_final = float(b.detach())
    cal = a_final * s + b_final
    nll_after, brier_after, ece_after = nll_brier_ece(cal.tolist(), labels)

    # Monotone invariance check.
    from calibration.common import binary_metrics as _bm
    before_m = _bm(list(scores), list(labels), 0.0)
    after_m = _bm(list(cal.tolist()), list(labels), 0.0)
    tol = 1e-9

    result = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "a": a_final,
        "b": b_final,
        "nll_before": nll_before, "nll_after": nll_after,
        "brier_before": brier_before, "brier_after": brier_after,
        "ece_before": ece_before, "ece_after": ece_after,
        "invariant_check": {
            "a_positive": a_final > 0,
            "EER_unchanged": abs(before_m["eer"] - after_m["eer"]) <= tol,
            "AUC_unchanged": abs(before_m["auroc"] - after_m["auroc"]) <= tol,
        },
    }
    if not (result["invariant_check"]["a_positive"] and
            result["invariant_check"]["EER_unchanged"] and
            result["invariant_check"]["AUC_unchanged"]):
        print("ERROR: affine calibration changed ranking", file=sys.stderr)
        sys.exit(2)

    (out_dir / "affine_oracle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

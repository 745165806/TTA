"""Source-only threshold and finite-grid selection helpers."""
import torch


def empirical_real_quantile(scores, labels, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    real = scores[labels == 0]
    if real.numel() < 2 or not bool(torch.isfinite(real).all()):
        raise ValueError("calibration requires finite bonafide source scores")
    # 'higher' makes the observed false-positive rate no larger than alpha.
    return float(torch.quantile(real, 1 - alpha, interpolation="higher"))


def select_earliest_minimum(candidates):
    if not candidates:
        raise ValueError("empty source selection candidates")
    return min(candidates, key=lambda item: (item["objective"], item["candidate_index"]))

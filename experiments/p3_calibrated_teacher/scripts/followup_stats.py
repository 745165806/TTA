"""Deterministic paired-bootstrap statistics shared by P3 follow-ups."""
import random

from eptta.evaluation.metrics import binary_metrics


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def interval(values):
    return [percentile(values, 0.025), percentile(values, 0.975)]


def paired_mean_bootstrap(differences, *, seed=2026, n_boot=2000):
    if not differences:
        raise ValueError("paired bootstrap requires nonempty differences")
    rng = random.Random(seed)
    n = len(differences)
    draws = [sum(differences[rng.randrange(n)] for _ in range(n)) / n
             for _ in range(n_boot)]
    return {
        "estimate": sum(differences) / n,
        "ci95": interval(draws), "seed": seed, "n": n_boot,
    }


def _ranking(scores, labels):
    metric = binary_metrics(scores, labels, 0.0)
    return metric["eer"], metric["auroc"]


def paired_ranking_bootstrap(candidate_scores, reference_scores, labels, *,
                             seed=2026, n_boot=2000):
    """Paired sample bootstrap for candidate-reference EER/AUC deltas."""
    if not (len(candidate_scores) == len(reference_scores) == len(labels)) or not labels:
        raise ValueError("paired ranking bootstrap inputs must be nonempty and aligned")
    candidate_scores = list(candidate_scores)
    reference_scores = list(reference_scores)
    labels = list(labels)
    candidate_eer, candidate_auc = _ranking(candidate_scores, labels)
    reference_eer, reference_auc = _ranking(reference_scores, labels)
    rng = random.Random(seed)
    n = len(labels)
    eer_draws, auc_draws = [], []
    while len(eer_draws) < n_boot:
        indices = [rng.randrange(n) for _ in range(n)]
        ys = [labels[i] for i in indices]
        if not any(ys) or all(ys):
            continue
        candidate = [candidate_scores[i] for i in indices]
        reference = [reference_scores[i] for i in indices]
        cand_eer, cand_auc = _ranking(candidate, ys)
        ref_eer, ref_auc = _ranking(reference, ys)
        eer_draws.append(cand_eer - ref_eer)
        auc_draws.append(cand_auc - ref_auc)
    return {
        "delta_EER": candidate_eer - reference_eer,
        "delta_EER_ci95": interval(eer_draws),
        "delta_AUC": candidate_auc - reference_auc,
        "delta_AUC_ci95": interval(auc_draws),
        "seed": seed, "n": n_boot,
    }


def task_gain_supported(comparison):
    return bool(comparison["delta_EER_ci95"][1] < 0 or
                comparison["delta_AUC_ci95"][0] > 0)

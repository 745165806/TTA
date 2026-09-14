"""Dependency-light exact binary score metrics (larger means spoof)."""
from eptta.errors import DataError
from eptta.training.selection import equal_error_rate


def _auroc(scores, labels):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise DataError("AUROC is undefined when one class is absent")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_metrics(scores, labels, threshold, frozen_scores=None):
    if len(scores) != len(labels) or not scores:
        raise DataError("metrics require nonempty aligned scores/labels")
    if any(type(label) is not int or label not in (0, 1) for label in labels) or not sum(labels) or sum(labels) == len(labels):
        raise DataError("metrics require both canonical binary classes")
    prediction = [int(score > threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(prediction, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(prediction, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(prediction, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(prediction, labels))
    result = {"count": len(scores), "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
              "tpr": tp / (tp + fn), "fpr": fp / (fp + tn), "fnr": fn / (tp + fn),
              "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
              "auroc": _auroc(scores, labels), "eer": equal_error_rate(scores, labels)}
    if frozen_scores is not None:
        if len(frozen_scores) != len(scores):
            raise DataError("frozen/adapted score coverage mismatch")
        before = [int(score > threshold) for score in frozen_scores]
        result["helpful_flips"] = sum(a != b and a == y for a, b, y in zip(prediction, before, labels))
        result["harmful_flips"] = sum(a != b and b == y for a, b, y in zip(prediction, before, labels))
    return result

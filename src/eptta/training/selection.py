"""Source-validation metrics and deterministic checkpoint selection."""
import math

from eptta.errors import ContractError, DataError


def equal_error_rate(scores, labels):
    """Compute EER by linear interpolation at the FAR/FRR crossing.

    Scores must increase toward spoof and canonical labels are bonafide=0,
    spoof=1. Every sample is consumed exactly once by the caller.
    """
    if len(scores) != len(labels) or not scores:
        raise DataError("EER requires equally sized nonempty scores and labels")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise DataError("EER labels must be canonical binary integers")
    if any(not isinstance(s, (int, float)) or not math.isfinite(s) for s in scores):
        raise DataError("EER scores must be finite")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise DataError("source_val EER is undefined when one class is absent")
    pairs = sorted(zip((float(s) for s in scores), labels), reverse=True)
    false_accepts = 0
    true_accepts = 0
    points = [(0.0, 1.0)]  # FAR, FRR with threshold above every score
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1] == 1:
                true_accepts += 1
            else:
                false_accepts += 1
            index += 1
        points.append((false_accepts / negatives, 1.0 - true_accepts / positives))
    for (far0, frr0), (far1, frr1) in zip(points, points[1:]):
        d0, d1 = far0 - frr0, far1 - frr1
        if d0 == 0:
            return far0
        if d0 * d1 <= 0:
            fraction = abs(d0) / (abs(d0) + abs(d1)) if d0 != d1 else 0.0
            far = far0 + fraction * (far1 - far0)
            frr = frr0 + fraction * (frr1 - frr0)
            return (far + frr) / 2.0
    far, frr = min(points, key=lambda p: abs(p[0] - p[1]))
    return (far + frr) / 2.0


def select_source_checkpoint(records):
    """Minimum finite source_val EER; exact ties select the earliest epoch."""
    candidates = []
    seen_epochs = set()
    for record in records:
        required = {"epoch", "source_val_eer", "checkpoint_ref"}
        if not required.issubset(record):
            raise ContractError("selection record is missing checkpoint evidence")
        epoch = record["epoch"]
        eer = record["source_val_eer"]
        if type(epoch) is not int or epoch < 0 or epoch in seen_epochs:
            raise ContractError("selection epochs must be unique nonnegative integers")
        if type(eer) not in (int, float) or not math.isfinite(eer) or not 0 <= eer <= 1:
            raise ContractError("source_val EER must be finite and in [0,1]")
        seen_epochs.add(epoch)
        candidates.append(record)
    if not candidates:
        raise ContractError("no source_val checkpoint candidates")
    return min(candidates, key=lambda item: (float(item["source_val_eer"]), item["epoch"]))

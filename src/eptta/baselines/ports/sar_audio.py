"""SAR audio port: reliable entropy filtering + two-phase SAM over backend normalization affine.

Audio mapping: lr=1.5625e-5 (official ResNet50-BN bs=1 branch), SGD momentum=0.9,
SAM rho=0.05, reliable margin = 0.4*log(2) for binary C=2.
"""
import math
from functools import partial

import torch

from eptta.baselines.ports.common import prediction_entropy
from eptta.baselines.ports.sam import SAM
from eptta.baselines.ports.tent_audio import logits_to_score, score_current


def sar_adapt(model, adapter, waveform, class_index_map, lr=1.5625e-5, momentum=0.9,
              rho=0.05, steps=1):
    """Returns (score_after, applied, reason, diagnostics)."""
    margin = 0.4 * math.log(2)

    _emb, logits0 = adapter.forward(waveform)
    entropy_first = float(prediction_entropy(logits0).mean().item())
    reliable_first = entropy_first < margin
    if not reliable_first:
        return score_current(model, adapter, waveform, class_index_map), False, \
            "unreliable_entropy", {"entropy_first": entropy_first, "entropy_second": None,
                                   "reliable_first": False, "reliable_second": False,
                                   "sam_grad_norm_first": None, "sam_perturb_norm": None,
                                   "sam_grad_norm_second": None}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SAM(params, partial(torch.optim.SGD, lr=lr, momentum=momentum), rho=rho)

    for _step in range(steps):
        _emb, logits = adapter.forward(waveform)
        loss = prediction_entropy(logits).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step()

        _emb, logits2 = adapter.forward(waveform)
        entropy_second = float(prediction_entropy(logits2).mean().item())
        reliable_second = entropy_second < margin
        if not reliable_second:
            # SAM-perturbed second pass is unreliable: restore perturbation, no step.
            with torch.no_grad():
                for p in params:
                    e_w = optimizer.state.pop(p, None)
                    if e_w is not None:
                        p.sub_(e_w)
            return score_current(model, adapter, waveform, class_index_map), False, \
                "unreliable_second_filter", {
                    "entropy_first": entropy_first, "entropy_second": entropy_second,
                    "reliable_first": True, "reliable_second": False,
                    "sam_grad_norm_first": optimizer.sam_grad_norm_first,
                    "sam_perturb_norm": optimizer.sam_perturb_norm,
                    "sam_grad_norm_second": None}

        loss2 = prediction_entropy(logits2).mean()
        optimizer.zero_grad()
        loss2.backward()
        optimizer.second_step()

    return score_current(model, adapter, waveform, class_index_map), True, None, {
        "entropy_first": entropy_first, "entropy_second": entropy_second,
        "reliable_first": True, "reliable_second": True,
        "sam_grad_norm_first": optimizer.sam_grad_norm_first,
        "sam_perturb_norm": optimizer.sam_perturb_norm,
        "sam_grad_norm_second": optimizer.sam_grad_norm_second}

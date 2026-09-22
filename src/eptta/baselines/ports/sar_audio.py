"""SAR audio port: reliable entropy filtering + two-phase SAM over backend normalization affine."""
import math

import torch

from eptta.baselines.ports.common import prediction_entropy
from eptta.baselines.ports.sam import SAM
from eptta.baselines.ports.tent_audio import logits_to_score


def sar_episodic(model, adapter, waveform, class_index_map, lr=0.001, momentum=0.9,
                 rho=0.05, steps=1):
    """Adapt one sample; returns (score_before, score_after, applied, reason)."""
    _emb, logits0 = adapter.forward(waveform)
    score_before = float(logits_to_score(logits0, class_index_map)[0].item())

    # Reliable sample selection: binary C=2 -> margin = 0.4 * log(2).
    margin = 0.4 * math.log(2)
    ent = float(prediction_entropy(logits0).mean().item())
    if ent >= margin:
        return score_before, score_before, False, "unreliable_entropy"

    params = [p for p in model.parameters() if p.requires_grad]
    from functools import partial
    base = partial(torch.optim.SGD, lr=lr, momentum=momentum)
    optimizer = SAM(params, base, rho=rho)

    for _step in range(steps):
        _emb, logits = adapter.forward(waveform)
        loss = prediction_entropy(logits).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step()
        _emb, logits2 = adapter.forward(waveform)
        loss2 = prediction_entropy(logits2).mean()
        optimizer.zero_grad()
        loss2.backward()
        optimizer.second_step()

    _emb, logits1 = adapter.forward(waveform)
    score_after = float(logits_to_score(logits1, class_index_map)[0].item())
    return score_before, score_after, True, None

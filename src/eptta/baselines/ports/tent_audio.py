"""TENT audio port: episodic entropy minimization over backend normalization affine.

Audio mapping (audited official cfgs/tent.yaml): Adam, lr=1e-3, steps=1, wd=0.
"""
import torch

from eptta.baselines.ports.common import prediction_entropy


def logits_to_score(logits, class_index_map):
    """score = logit_spoof - logit_bonafide (native class order from the map)."""
    return logits[:, class_index_map["spoof"]] - logits[:, class_index_map["bonafide"]]


def score_current(model, adapter, waveform, class_index_map):
    _emb, logits = adapter.forward(waveform)
    return float(logits_to_score(logits, class_index_map)[0].item())


def tent_adapt(model, adapter, waveform, class_index_map, lr=1e-3, steps=1):
    """Adam entropy-minimization update; returns the same-sample post-update score."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    for _step in range(steps):
        _emb, logits = adapter.forward(waveform)
        loss = prediction_entropy(logits).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return score_current(model, adapter, waveform, class_index_map)

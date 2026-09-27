"""TENT audio port: episodic entropy minimization over backend normalization affine."""
import torch

from eptta.baselines.ports.common import prediction_entropy


def logits_to_score(logits, class_index_map):
    """score = logit_spoof - logit_bonafide (native class order from the map)."""
    return logits[:, class_index_map["spoof"]] - logits[:, class_index_map["bonafide"]]


def tent_episodic(model, adapter, waveform, class_index_map, lr=0.001, momentum=0.9, steps=1):
    """Adapt one sample; returns (score_before, score_after)."""
    _emb, logits0 = adapter.forward(waveform)
    score_before = float(logits_to_score(logits0, class_index_map)[0].item())

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    for _step in range(steps):
        _emb, logits = adapter.forward(waveform)
        loss = prediction_entropy(logits).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _emb, logits1 = adapter.forward(waveform)
    score_after = float(logits_to_score(logits1, class_index_map)[0].item())
    return score_before, score_after

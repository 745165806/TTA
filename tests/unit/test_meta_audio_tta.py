"""Functional fast-weight and state contract; real-model gate is smoke_real.py."""
import torch
from torch import nn

from experiments.meta_audio_tta.core import (
    BYOLSystem, configure_meta, episodic_scores, meta_objective, score,
)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 160)
        self.bn = nn.BatchNorm1d(160)
        self.out_layer = nn.Linear(160, 2)

    def forward(self, waveform):
        return self.out_layer(self.bn(self.encoder(waveform)))


def _system():
    torch.manual_seed(7)
    system = BYOLSystem(Detector(), Detector(), {"spoof": 0, "bonafide": 1})
    configure_meta(system)
    return system


def test_second_order_cross_and_same_sample_have_source_label_gradient():
    system = _system()
    support = torch.tensor([[0.2, -0.5, 0.1, 0.6], [0.25, -0.45, 0.11, 0.55]])
    query = torch.tensor([[-0.2, 0.1, 0.7, -0.3], [-0.15, 0.12, 0.72, -0.28]])
    selected = [p for p in system.parameters() if p.requires_grad]
    cross, info = meta_objective(system, support, query, 1, 3e-4)
    cross_grads = torch.autograd.grad(cross, selected, retain_graph=True)
    assert any(g.requires_grad for g in info["inner_gradients"].values())
    assert all(torch.isfinite(g).all() for g in cross_grads)
    same, _ = meta_objective(system, support, support, 0, 3e-4)
    same_grads = torch.autograd.grad(same, selected)
    assert all(torch.isfinite(g).all() for g in same_grads)
    assert not torch.allclose(cross.detach(), same.detach())


def test_episode_does_not_change_starting_state_or_ema():
    system = _system()
    a = torch.tensor([[0.2, -0.5, 0.1, 0.6], [0.25, -0.45, 0.11, 0.55]])
    b = torch.tensor([[-0.2, 0.1, 0.7, -0.3], [-0.15, 0.12, 0.72, -0.28]])
    state = {key: value.detach().clone() for key, value in system.state_dict().items()}
    frozen = score(system, a[:1]).detach()
    before, after, _loss, _grads = episodic_scores(system, a[:1], a, 3e-4)
    episodic_scores(system, b[:1], b, 3e-4)
    again, after_again, _loss, _grads = episodic_scores(system, a[:1], a, 3e-4)
    assert torch.allclose(frozen, before, atol=1e-6)
    assert torch.allclose(before, again, atol=1e-6)
    assert torch.allclose(after, after_again, atol=1e-6)
    assert any(name.startswith("online_model.bn") and p.requires_grad
               for name, p in system.named_parameters())
    assert all(torch.equal(value, system.state_dict()[key]) for key, value in state.items())

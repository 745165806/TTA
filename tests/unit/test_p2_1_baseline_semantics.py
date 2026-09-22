"""Unit tests for the P2.1 baseline validation fixes."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.baselines.ports.common import (configure_full_model,
                                          configure_normalization_adaptation)
from eptta.baselines.ports.sam import SAM

ROOT = Path(__file__).resolve().parents[2]


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        self.bn = torch.nn.BatchNorm1d(4)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.bn(self.ssl_model(x)))


class _Adapter:
    def __init__(self, model):
        self.model = model

    def forward(self, waveform):
        return None, self.model(waveform)


def test_tent_uses_adam_lr_1e3(monkeypatch):
    from eptta.baselines.ports import tent_audio
    model = _MLP()
    configure_normalization_adaptation(model)
    adapter = _Adapter(model)
    x = torch.randn(4, 4)

    calls = []
    real_adam = torch.optim.Adam
    def spy(params, lr, **kw):
        calls.append({"lr": lr, "weight_decay": kw.get("weight_decay", 0.0)})
        return real_adam(params, lr=lr, **kw)
    monkeypatch.setattr(torch.optim, "Adam", spy)
    tent_audio.tent_adapt(model, adapter, x, {"spoof": 0, "bonafide": 1})
    assert calls and calls[0]["lr"] == 1e-3 and calls[0]["weight_decay"] == 0.0


def test_score_current_does_not_change_parameters():
    from eptta.baselines.ports import tent_audio
    model = _MLP()
    configure_normalization_adaptation(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    adapter = _Adapter(model)
    tent_audio.score_current(model, adapter, torch.randn(4, 4), {"spoof": 0, "bonafide": 1})
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], atol=0, rtol=0)


def test_sar_second_filter_branch(monkeypatch):
    from eptta.baselines.ports import sar_audio
    model = _MLP()
    configure_normalization_adaptation(model)
    adapter = _Adapter(model)
    x = torch.randn(4, 4)

    seq = [0.01, 0.01, 1.0]  # entropy_first, loss, entropy_second
    def fake_entropy(logits):
        v = seq.pop(0) if seq else 1.0
        return (logits * 0).mean() + v
    monkeypatch.setattr(sar_audio, "prediction_entropy", fake_entropy)

    sa, applied, reason, diag = sar_audio.sar_adapt(
        model, adapter, x, {"spoof": 0, "bonafide": 1})
    assert applied is False
    assert reason == "unreliable_second_filter"
    assert diag["reliable_first"] is True and diag["reliable_second"] is False


def test_sam_parameter_state_binding():
    model = torch.nn.Linear(4, 2)
    opt = SAM(model.parameters(), lambda ps: torch.optim.SGD(ps, lr=0.1), rho=0.05)
    x = torch.randn(3, 4)
    w0 = model.weight.detach().clone()

    # First backward: only weight gets a gradient (bias grad is zeroed).
    model(x).sum().backward()
    model.bias.grad = None
    opt.first_step()
    assert model.weight in opt.state, "perturbation must be stored per parameter"
    w_perturbed = model.weight.detach().clone()
    assert not torch.allclose(w_perturbed, w0)

    model.zero_grad()
    # Second backward: only bias gets a gradient (weight grad is zeroed).
    model(x).sum().backward()
    model.weight.grad = None
    opt.second_step()

    # Weight perturbation must still be restored even though weight had no
    # second-pass gradient (parameter-bound state, not pop(0) list order).
    assert torch.allclose(model.weight.detach(), w0, atol=1e-6, rtol=1e-6)


def test_memo_mapping_sgd_lr_2_5e_4():
    from eptta.baselines.ports import memo_audio
    import inspect
    src = inspect.getsource(memo_audio.memo_adapt)
    assert "torch.optim.SGD" in src
    assert "lr=2.5e-4" in src


def test_unknown_split_fails():
    spec_path = ROOT / "experiments/p2_calibration_baselines/baselines/splits.py"
    spec = importlib.util.spec_from_file_location("p2_splits", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.get_split("banana")

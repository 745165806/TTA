"""Unit tests for the audited audio TTA ports (no full model required)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.baselines.ports.common import (backend_batch_norm_modules,
                                          configure_normalization_adaptation,
                                          marginal_entropy, prediction_entropy,
                                          updated_parameter_names)
from eptta.baselines.ports.sam import SAM
from eptta.baselines.ports import tent_audio

ROOT = Path(__file__).resolve().parents[2]


class SyntheticBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        self.bn1 = torch.nn.BatchNorm1d(4)
        self.bn2 = torch.nn.BatchNorm1d(4)
        self.bn2d = torch.nn.BatchNorm2d(4)  # detected for scope test, not in forward path
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        h = self.ssl_model(x)
        h = self.bn1(h)
        h = self.bn2(h)
        return self.head(h)


class FakeAdapter:
    def __init__(self, model):
        self.model = model

    def forward(self, waveform):
        return None, self.model(waveform)


def test_label_free_manifest_rejects_label_fields(tmp_path):
    spec_path = ROOT / "experiments/p2_calibration_baselines/baselines/target_waveform.py"
    spec = importlib.util.spec_from_file_location("p2_target_waveform", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data_root = tmp_path / "audio"
    data_root.mkdir()
    (data_root / "a.wav").write_bytes(b"x")  # not read, only path validation

    good = tmp_path / "good.jsonl"
    good.write_text(json.dumps({"schema_version": "0.1.0", "sample_id": "a.wav",
                                "root_key": "root", "audio_relpath": "a.wav",
                                "split_role": "target_test", "sample_index": 0}) + "\n")
    ds = mod.TargetWaveformDataset(str(good), {"root": str(data_root)}, role="target_test")
    assert len(ds) == 1

    for key in ("label", "canonical_label", "attack", "class", "target", "y"):
        bad = tmp_path / ("bad_%s.jsonl" % key)
        bad.write_text(json.dumps({"schema_version": "0.1.0", "sample_id": "a.wav",
                                   "root_key": "root", "audio_relpath": "a.wav",
                                   "split_role": "target_test", key: 1}) + "\n")
        with pytest.raises(ValueError):
            mod.TargetWaveformDataset(str(bad), {"root": str(data_root)}, role="target_test")


def test_tent_configuration_scope_and_update():
    model = SyntheticBackend()
    configure_normalization_adaptation(model)
    names = updated_parameter_names(model)
    assert "bn1.weight" in names and "bn1.bias" in names
    assert "bn2.weight" in names and "bn2.bias" in names
    assert "head.weight" not in names
    assert "ssl_model.0.weight" not in names
    # Every non-selected parameter is frozen.
    for name, p in model.named_parameters():
        if name not in names:
            assert p.requires_grad is False

    adapter = FakeAdapter(model)
    x = torch.randn(4, 4)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n in names}
    score0 = tent_audio.score_current(model, adapter, x, {"spoof": 0, "bonafide": 1})
    score1 = tent_audio.tent_adapt(model, adapter, x, {"spoof": 0, "bonafide": 1}, steps=2)
    assert isinstance(score0, float) and isinstance(score1, float)
    changed = [n for n in before if not torch.allclose(before[n], dict(model.named_parameters())[n])]
    assert changed, "entropy update must change selected normalization params"


def test_model_reset_restores_exact_state():
    model = SyntheticBackend()
    configure_normalization_adaptation(model)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapter = FakeAdapter(model)
    x = torch.randn(4, 4)
    tent_audio.tent_adapt(model, adapter, x, {"spoof": 0, "bonafide": 1}, steps=1)
    model.load_state_dict(frozen)
    for k, v in frozen.items():
        torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)


def test_per_sample_reset_A_B_A_repeatable():
    model = SyntheticBackend()
    configure_normalization_adaptation(model)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapter = FakeAdapter(model)
    xa = torch.randn(4, 4)

    def run_a():
        model.load_state_dict(frozen)
        return tent_audio.tent_adapt(model, adapter, xa, {"spoof": 0, "bonafide": 1}, steps=1)

    first = run_a()
    # interleave B
    model.load_state_dict(frozen)
    tent_audio.tent_adapt(model, adapter, torch.randn(4, 4), {"spoof": 0, "bonafide": 1}, steps=1)
    again = run_a()
    assert first == again


def test_sam_two_phase_updates_params():
    model = torch.nn.Linear(4, 2)
    optimizer = SAM(model.parameters(), lambda ps: torch.optim.SGD(ps, lr=0.1), rho=0.05)
    x = torch.randn(3, 4)
    loss = model(x).mean()
    loss.backward()
    p_before = model.weight.detach().clone()
    optimizer.first_step()
    assert not torch.allclose(model.weight.detach(), p_before)
    p_perturbed = model.weight.detach().clone()
    model.zero_grad()
    model(x).mean().backward()
    optimizer.second_step()
    assert not torch.allclose(model.weight.detach(), p_perturbed)


def test_marginal_entropy_is_not_mean_entropy():
    logits = torch.tensor([[-1.0, 1.0], [2.0, -2.0], [0.0, 0.0]])
    marginal = marginal_entropy(logits)
    per_view_mean = prediction_entropy(logits).mean()
    assert not torch.allclose(marginal, per_view_mean)
    # entropy of averaged probability must be used, not the average entropy.
    p = torch.softmax(logits, dim=-1)
    expected = -(p.mean(0) * (p.mean(0).clamp_min(1e-12)).log()).sum()
    torch.testing.assert_close(marginal, expected, atol=1e-6, rtol=1e-6)


def test_sar_reliable_filter_and_abstain():
    from eptta.baselines.ports import sar_audio
    model = SyntheticBackend()
    configure_normalization_adaptation(model)
    adapter = FakeAdapter(model)
    x = torch.randn(4, 4)
    # A high-entropy first prediction (near-uniform logits) must abstain.
    with torch.no_grad():
        # force near-uniform logits by zeroing the head output
        model.head.weight.data.zero_()
        model.head.bias.data.zero_()
    sa, applied, reason, diag = sar_audio.sar_adapt(
        model, adapter, x, {"spoof": 0, "bonafide": 1}, steps=1)
    assert applied is False
    assert reason == "unreliable_entropy"
    assert diag["reliable_first"] is False

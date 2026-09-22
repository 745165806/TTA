"""Contract tests for the audio-native standard-TTA mapping."""
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from eptta.baselines.ports import memo_audio, sar_audio, tent_audio
from eptta.baselines.ports.audio_native import (
    SCOPE_A, SCOPE_B, assert_bn_buffers_unchanged, configure_audio_native,
    preregistered_parameter_names, reset_episode_state, snapshot_episode_state)
from eptta.baselines.ports.common import marginal_entropy, prediction_entropy

ROOT = Path(__file__).resolve().parents[2]


class AudioNativeToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        self.bn = torch.nn.BatchNorm1d(4)
        self.ln = torch.nn.LayerNorm(4)
        self.gn = torch.nn.GroupNorm(2, 4)
        self.att_weight11 = torch.nn.Parameter(torch.ones(4, 1))
        self.pos_S = torch.nn.Parameter(torch.ones(1, 2, 4))
        self.proj = torch.nn.Linear(4, 4)
        self.out_layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.ln(x)
        x = self.gn(x.unsqueeze(-1)).squeeze(-1)
        return self.out_layer(x + self.att_weight11.squeeze(-1))


class Adapter:
    def __init__(self, model): self.model = model
    def forward(self, waveform): return None, self.model(waveform)


def test_scope_excludes_generic_xlsr_and_head_and_is_preregistered():
    model = AudioNativeToy()
    scope_a = preregistered_parameter_names(model, SCOPE_A)
    assert scope_a
    assert all(not name.startswith("ssl_model.") for name in scope_a)
    assert all(not name.startswith("out_layer.") for name in scope_a)
    assert set(scope_a) == {"bn.weight", "bn.bias", "ln.weight", "ln.bias",
                            "gn.weight", "gn.bias"}
    scope_b = preregistered_parameter_names(model, SCOPE_B)
    assert set(scope_b) == set(scope_a) | {"att_weight11", "pos_S"}
    assert "proj.weight" not in scope_b


def test_bn_source_statistics_unchanged_while_affine_can_update():
    model = AudioNativeToy()
    configure_audio_native(model, SCOPE_A)
    assert model.training is False and model.bn.training is False
    state = snapshot_episode_state(model)
    adapter = Adapter(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    tent_audio.tent_adapt(model, adapter, torch.randn(3, 4), {"spoof": 0, "bonafide": 1})
    assert_bn_buffers_unchanged(model, state)
    changed = {name for name, p in model.named_parameters()
               if name in before and not torch.equal(p, before[name])}
    assert changed
    assert changed <= set(before)
    assert any(name.startswith(("bn.", "ln.", "gn.")) for name in changed)


def test_tent_calls_prediction_entropy(monkeypatch):
    model = AudioNativeToy(); configure_audio_native(model, SCOPE_A)
    adapter = Adapter(model); called = []
    original = tent_audio.prediction_entropy
    def spy(logits):
        called.append(logits.shape)
        return original(logits)
    monkeypatch.setattr(tent_audio, "prediction_entropy", spy)
    tent_audio.tent_adapt(model, adapter, torch.randn(3, 4), {"spoof": 0, "bonafide": 1})
    assert called


def test_sar_uses_both_filters_and_sam(monkeypatch):
    model = AudioNativeToy(); configure_audio_native(model, SCOPE_A)
    with torch.no_grad():
        model.out_layer.weight.fill_(4.0)
        model.out_layer.weight[1].fill_(-4.0)
        model.out_layer.bias[:] = torch.tensor([4.0, -4.0])
    calls = []
    real_sam = sar_audio.SAM
    class SpySAM(real_sam):
        def first_step(self): calls.append("first"); return super().first_step()
        def second_step(self): calls.append("second"); return super().second_step()
    monkeypatch.setattr(sar_audio, "SAM", SpySAM)
    _score, applied, reason, diag = sar_audio.sar_adapt(
        model, Adapter(model), torch.randn(3, 4), {"spoof": 0, "bonafide": 1},
        entropy_margin=1.0)
    assert applied and reason is None
    assert diag["reliable_first"] and diag["reliable_second"]
    assert calls == ["first", "second"]


def test_memo_uses_marginal_entropy_and_respects_scope(monkeypatch):
    model = AudioNativeToy(); configure_audio_native(model, SCOPE_A)
    selected = {name for name, p in model.named_parameters() if p.requires_grad}
    called = []
    original = memo_audio.marginal_entropy
    def spy(logits): called.append(True); return original(logits)
    monkeypatch.setattr(memo_audio, "marginal_entropy", spy)
    memo_audio.memo_adapt(model, Adapter(model), torch.randn(3, 4),
                          {"spoof": 0, "bonafide": 1}, respect_configured_scope=True)
    assert called
    assert {name for name, p in model.named_parameters() if p.requires_grad} == selected
    logits = torch.tensor([[-2.0, 2.0], [2.0, -2.0]])
    assert not torch.allclose(marginal_entropy(logits), prediction_entropy(logits).mean())


def test_source_augmentation_audit_has_no_target_label_input():
    path = ROOT / "experiments/audio_native_tta/audit_augmentations.py"
    text = path.read_text(encoding="utf-8")
    assert "read_source_labels" in text
    assert "target10" not in text.lower()
    config = json.loads((ROOT / "experiments/audio_native_tta/config.json").read_text())
    assert config["augmentation_audit"]["source_role"] == "select"


def test_actual_target_worker_loader_rejects_labels(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "audio_native_target_worker", ROOT / "experiments/audio_native_tta/target_worker.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    audio = tmp_path / "audio"; audio.mkdir(); (audio / "x.wav").write_bytes(b"unused")
    manifest = tmp_path / "unsafe.jsonl"
    manifest.write_text(json.dumps({
        "schema_version": "0.1.0", "sample_id": "x", "sample_index": 0,
        "root_key": "r", "audio_relpath": "x.wav", "split_role": "select",
        "label": 1}) + "\n")
    with pytest.raises(ValueError, match="unknown fields|label field"):
        module.TargetWaveformDataset(manifest, {"r": str(audio)}, role="select")


def test_per_sample_reset_is_exact():
    model = AudioNativeToy(); configure_audio_native(model, SCOPE_A)
    state = snapshot_episode_state(model); x = torch.randn(3, 4); adapter = Adapter(model)
    def episode():
        reset_episode_state(model, state)
        return tent_audio.tent_adapt(model, adapter, x, {"spoof": 0, "bonafide": 1})
    first = episode()
    reset_episode_state(model, state)
    tent_audio.tent_adapt(model, adapter, torch.randn(3, 4), {"spoof": 0, "bonafide": 1})
    second = episode()
    assert first == second
    reset_episode_state(model, state)
    for name, value in state.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

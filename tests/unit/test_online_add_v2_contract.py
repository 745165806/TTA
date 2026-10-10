"""Small checks for the separate online ADD development baseline protocol."""

import torch

from experiments.online_add_v2 import methods


class FixedNativeModel(torch.nn.Module):
    def forward(self, sequence):
        return sequence[:, :2]


def test_frozen_score_uses_native_spoof_minus_bonafide_order(monkeypatch):
    monkeypatch.setattr(methods, "load_frozen_backend", lambda device: (
        FixedNativeModel(), {"class_index_map": {"spoof": 0, "bonafide": 1}}))
    stream = methods.OnlineMethod("frozen", torch.device("cpu"), {})
    scores, logits = stream.predict(torch.tensor([[3.0, 1.0], [0.0, 2.0]]))
    assert scores.tolist() == [2.0, -2.0]
    assert logits.shape == (2, 2)
    stream.adapt(torch.zeros(2, 2), logits)
    assert stream.stats() == {"updates": 0, "adapted_samples": 0,
                              "seen": 2, "memory_size": 0, "trainable_names": []}


def test_new_stream_has_independent_counters(monkeypatch):
    monkeypatch.setattr(methods, "load_frozen_backend", lambda device: (
        FixedNativeModel(), {"class_index_map": {"spoof": 0, "bonafide": 1}}))
    first = methods.OnlineMethod("frozen", torch.device("cpu"), {})
    first.adapt(torch.zeros(3, 2), torch.zeros(3, 2))
    second = methods.OnlineMethod("frozen", torch.device("cpu"), {})
    assert first.stats()["seen"] == 3
    assert second.stats()["seen"] == 0

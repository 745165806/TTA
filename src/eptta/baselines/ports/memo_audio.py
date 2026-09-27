"""MEMO audio port placeholder.

The official optimizer/learning rate could not be confirmed (PORT_BLOCKED_AUDIT);
this module exposes the audited marginal-entropy objective only and must NOT be
used to run an invented optimizer.
"""
from eptta.baselines.ports.common import marginal_entropy
from eptta.errors import NotImplementedStage


def memo_episodic(model, adapter, waveform, class_index_map, **kwargs):
    raise NotImplementedStage(
        "memo_audio_ep_full is PORT_BLOCKED_AUDIT: official optimizer/lr unverified")


__all__ = ["marginal_entropy", "memo_episodic"]

"""Formal waveform-port entry (never the feature-cache ``run_method`` path)."""
from eptta.baselines.ports import tent_audio, sar_audio, memo_audio
from eptta.errors import NotImplementedStage


def run_waveform_method(method_id, model, adapter, waveform, class_index_map, **kwargs):
    """Episodic waveform adaptation for one sample (caller owns per-sample reset)."""
    if method_id == "tent_audio_ep":
        return tent_audio.tent_adapt(model, adapter, waveform, class_index_map, **kwargs)
    if method_id == "sar_audio_ep":
        return sar_audio.sar_adapt(model, adapter, waveform, class_index_map, **kwargs)
    if method_id == "memo_audio_ep_full":
        views = kwargs.pop("views")
        return memo_audio.memo_adapt(model, adapter, views, class_index_map, **kwargs)
    raise NotImplementedStage("unregistered waveform method: %s" % method_id)

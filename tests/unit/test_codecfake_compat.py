"""Codecfake compatibility is deterministic and leaves source audio intact."""
import numpy as np
import pytest
import soundfile as sf

from experiments.codecfake_compat.compat import materialize_float_wav, resample_to_16k
from workers.compat.author_training import load_audio


def test_non_native_resampling_is_deterministic_and_source_unchanged(tmp_path):
    rate = 24000
    time = np.arange(rate, dtype=np.float32) / rate
    original = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    source = tmp_path / "raw.wav"
    sf.write(source, original, rate, subtype="FLOAT")
    before, before_rate = sf.read(source, dtype="float32")
    first = resample_to_16k(source)
    second = resample_to_16k(source)
    assert first.dtype == np.float32 and first.shape == (16000,)
    np.testing.assert_array_equal(first, second)
    output = tmp_path / "derived.wav"
    details = materialize_float_wav(source, output)
    assert details["resampler"] == "scipy.signal.resample_poly"
    assert details["frames_after"] == 16000
    derived, derived_rate = sf.read(output, dtype="float32")
    assert derived_rate == 16000
    np.testing.assert_array_equal(derived, first)
    after, after_rate = sf.read(source, dtype="float32")
    assert before_rate == after_rate == rate
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(load_audio(output)).all()
    with pytest.raises(FileExistsError):
        materialize_float_wav(source, output)


def test_native_16k_is_forbidden_from_resampler(tmp_path):
    source = tmp_path / "native.wav"
    sf.write(source, np.ones(16000, dtype=np.float32), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="bypass"):
        resample_to_16k(source)


def test_fixed_selection_contains_no_label_fields():
    from experiments.codecfake_compat.run_compat import audit_fixed_selection
    rows, audited, counts = audit_fixed_selection()
    assert len(rows) == len(audited) == 512
    assert dict(counts) == {16000: 222, 24000: 209, 44100: 52, 48000: 29}
    assert all("label" not in row for row in rows)

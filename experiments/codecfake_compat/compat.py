"""Explicit Codecfake input compatibility without changing production load_audio."""
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly
import soundfile as sf


TARGET_RATE = 16000
WINDOW = ("kaiser", 5.0)


def inspect_audio(path):
    info = sf.info(path)
    if info.samplerate not in (16000, 24000, 44100, 48000) or info.frames < 1 or info.channels < 1:
        raise ValueError("unsupported or empty selected Codecfake waveform: %s" % path)
    return info


def resample_to_16k(path):
    """Return full-length float32 16-kHz audio for a nonnative source only."""
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate == TARGET_RATE:
        raise ValueError("native 16-kHz audio must bypass the resampler")
    if rate not in (24000, 44100, 48000):
        raise ValueError("unsupported selected sample rate")
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or audio.size < 1 or not np.isfinite(audio).all():
        raise ValueError("invalid input waveform")
    common = math.gcd(TARGET_RATE, rate)
    result = resample_poly(audio, up=TARGET_RATE // common, down=rate // common,
                           window=WINDOW, padtype="constant").astype(np.float32, copy=False)
    if result.size < 1 or not np.isfinite(result).all():
        raise ValueError("nonfinite or empty resampled waveform")
    return result


def materialize_float_wav(source, destination):
    """Create one derived audio artifact exclusively; never overwrite source."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError("derived audio destination already exists or aliases source")
    samples = resample_to_16k(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        sf.write(stream, samples, TARGET_RATE, format="WAV", subtype="FLOAT")
    info = sf.info(destination)
    if info.samplerate != TARGET_RATE or info.channels != 1 or info.frames != len(samples):
        raise ValueError("derived waveform failed write validation")
    return {"frames_before": inspect_audio(source).frames, "frames_after": info.frames,
            "dtype": "float32", "format": "WAV", "subtype": "FLOAT",
            "resampler": "scipy.signal.resample_poly", "resampler_version": scipy.__version__,
            "window": [WINDOW[0], WINDOW[1]], "padtype": "constant"}

"""Preprocess identity plus explicit, offline derived-audio preparation."""
from pathlib import Path

from eptta.config.validate import check_contract, content_hash
from eptta.data.io import sha256_file, write_json_new
from eptta.errors import ContractError, DataError


def preprocess_identity(contract):
    issues = check_contract(contract, "preprocess", "preprocess")
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    return contract["approval"]["content_sha256"]


def cache_identity(input_sha256, frozen_bundle_sha256, preprocess_contract, probe_id, dtype, numeric_mode):
    if not input_sha256 or not frozen_bundle_sha256:
        raise ContractError("cache identity requires content and frozen bundle hashes")
    return content_hash({"input_sha256": input_sha256, "frozen_bundle_sha256": frozen_bundle_sha256,
                         "preprocess_sha256": preprocess_identity(preprocess_contract), "probe_id": probe_id,
                         "dtype": dtype, "numeric_mode": numeric_mode})


def require_cache_preprocess(cache_metadata, preprocess_contract):
    expected = preprocess_identity(preprocess_contract)
    if cache_metadata.get("preprocess_sha256") != expected:
        raise DataError("cache preprocess identity mismatch; rebuild required")
    return True


def derive_16k_wav(source_ref, output_ref, parent_sample_id, log_ref):
    """Decode, mono-average and genuinely resample one file in a reviewed prep step.

    The source is never modified.  The derived WAV and its lineage log must both
    be new paths; callers perform this before staging/extraction, never in
    ``Dataset.__getitem__``.
    """
    import numpy as np
    import scipy
    import scipy.signal
    import soundfile

    source, output, log = Path(source_ref), Path(output_ref), Path(log_ref)
    if not source.is_file():
        raise DataError("BLOCKED_RESOURCE: resampling source is missing")
    if output.exists() or log.exists():
        raise ContractError("derived audio/log output exists; overwrite is forbidden")
    audio, sample_rate = soundfile.read(source, dtype="float32", always_2d=True)
    if audio.size == 0 or sample_rate < 1 or not np.isfinite(audio).all():
        raise DataError("decoded source audio is empty or non-finite")
    mono = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if sample_rate == 16000:
        derived = mono
        ratio = [1, 1]
    else:
        divisor = __import__("math").gcd(int(sample_rate), 16000)
        up, down = 16000 // divisor, int(sample_rate) // divisor
        derived = scipy.signal.resample_poly(mono, up, down, window=("kaiser", 5.0)).astype(np.float32)
        ratio = [up, down]
    output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(output, derived, 16000, subtype="FLOAT", format="WAV")
    lineage = {"schema_version": "0.2.0", "status": "DERIVED", "parent_sample_id": parent_sample_id,
               "source_ref": str(source.resolve()), "source_sha256": sha256_file(source),
               "derived_ref": str(output.resolve()), "derived_sha256": sha256_file(output),
               "source_sample_rate_hz": int(sample_rate), "derived_sample_rate_hz": 16000,
               "channels_rule": "soundfile_decode_then_mono_mean_all_classes",
               "resampler": "scipy.signal.resample_poly", "scipy_version": scipy.__version__,
               "ratio_up_down": ratio, "window": ["kaiser", 5.0],
               "command": ["derive_16k_wav", str(source.resolve()), str(output.resolve()),
                           parent_sample_id, str(log.resolve())]}
    write_json_new(log, lineage)
    return lineage

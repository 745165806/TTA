import io
import json

import numpy as np
import pytest
import soundfile as sf

from eptta.data import container_import
from eptta.errors import DataError


def _config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "part.parquet").write_bytes(b"fixture")
    return {"schema_version": "0.1.0", "dataset_id": "fixture", "release": "release-1",
            "subset": "explicit-test", "format": "parquet", "source_root": str(source),
            "sources": [{"pattern": "part.parquet", "expected_count": 1}],
            "audio_column": "audio", "label_column": "class", "group": {"column": "parent"},
            "label_map": {"real": 0, "fake": 1},
            "preprocess": {"input_sample_rate": 16000, "output_sample_rate": 16000,
                           "channels": "mono_mean", "resampler": "identity"},
            "output": str(tmp_path / "imported")}


def _audio(rate=16000):
    stream = io.BytesIO()
    sf.write(stream, np.zeros(rate // 10, dtype=np.float32), rate, format="WAV")
    return stream.getvalue()


def test_materialize_explicit_rows_and_refuse_overwrite(tmp_path, monkeypatch):
    config = _config(tmp_path)
    rows = [{"audio": {"bytes": _audio()}, "class": "real", "parent": "g1"},
            {"audio": {"bytes": _audio()}, "class": "fake", "parent": "g1"}]
    monkeypatch.setattr(container_import, "_rows", lambda *_: iter(rows))
    result = container_import.materialize(config)
    assert result["sample_count"] == 2
    content = (tmp_path / "imported" / "input.csv").read_text()
    assert "target_test" in content and "g1" in content
    assert len(list((tmp_path / "imported" / "audio").glob("*.wav"))) == 2
    assert json.loads((tmp_path / "imported" / "summary.json").read_text())["sample_count"] == 2
    with pytest.raises(Exception, match="overwrite is forbidden"):
        container_import.materialize(config)


def test_materialize_rejects_unmapped_label_without_publishing(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(container_import, "_rows", lambda *_: iter([
        {"audio": {"bytes": _audio()}, "class": "unknown", "parent": "g1"}]))
    with pytest.raises(DataError, match="unmapped label"):
        container_import.materialize(config)
    assert not (tmp_path / "imported").exists()


def test_materialize_resamples_explicit_22050_input(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["preprocess"] = {"input_sample_rate": 22050, "output_sample_rate": 16000,
                            "channels": "mono_mean", "resampler": "scipy_polyphase_v1"}
    config["select_groups"] = ["g1"]
    config["expected_sample_count"] = 1
    monkeypatch.setattr(container_import, "_rows", lambda *_: iter([
        {"audio": {"bytes": _audio(22050)}, "class": "real", "parent": "g1"},
        {"audio": {"bytes": _audio(22050)}, "class": "fake", "parent": "g2"}]))
    container_import.materialize(config)
    output = next((tmp_path / "imported" / "audio").glob("*.wav"))
    assert sf.info(output).samplerate == 16000
    assert sf.info(output).frames == 1600

import importlib.util
from pathlib import Path

import pytest

from eptta.training.selection import equal_error_rate, select_source_checkpoint


ROOT = Path(__file__).resolve().parents[2]


def _worker():
    path = ROOT / "workers/source_train_bridge.py"
    spec = importlib.util.spec_from_file_location("eptta_source_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _author():
    path = ROOT / "workers/compat/author_training.py"
    spec = importlib.util.spec_from_file_location("eptta_author_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_integer_random_rule_is_repeatable_and_epoch_sample_specific():
    module = _author()
    first = module._crop_start(211007, 64600, 13, 7, 0)
    assert first == module._crop_start(211007, 64600, 13, 7, 0)
    assert 0 <= first <= 211007 - 64600
    assert len({module._crop_start(211007, 64600, 13, index, epoch)
                for index, epoch in ((7, 0), (7, 1), (8, 0))}) > 1
    assert module._crop_start(211007, 64600, None, 7, 0) == 0


def test_ssl_legacy_audio_task_compat_removes_only_unsupported_task_fields():
    module = _author()
    omegaconf = pytest.importorskip("omegaconf")

    class AudioPretrainingConfig:
        __dataclass_fields__ = {"data": object()}

    cfg = omegaconf.OmegaConf.create(
        {"data": "/unused", "multiple_train_files": True, "eval_wer": False})
    removed = []

    def original(_dc, value, remove_missing=False):
        assert remove_missing is False
        return value

    result = module._merge_legacy_audio_task_config(
        original, AudioPretrainingConfig(), cfg, False, removed)
    assert dict(result) == {"data": "/unused"}
    assert removed == ["task.eval_wer", "task.multiple_train_files"]


def test_epoch_paths_and_aliases_keep_all_epochs(tmp_path):
    worker = _worker()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    first = Path(worker.epoch_checkpoint_path(str(checkpoints), 0))
    second = Path(worker.epoch_checkpoint_path(str(checkpoints), 1))
    first.write_bytes(b"epoch-zero")
    second.write_bytes(b"epoch-one")
    worker.publish_checkpoint_alias(str(first), str(checkpoints / "best.pt"))
    worker.publish_checkpoint_alias(str(first), str(checkpoints / "last.pt"))
    worker.publish_checkpoint_alias(str(second), str(checkpoints / "last.pt"))
    assert first.name == "epoch_0000.pt" and second.name == "epoch_0001.pt"
    assert first.read_bytes() == b"epoch-zero" and second.read_bytes() == b"epoch-one"
    assert (checkpoints / "best.pt").read_bytes() == b"epoch-zero"
    assert (checkpoints / "last.pt").read_bytes() == b"epoch-one"


def test_epoch_checkpoint_contains_full_reload_and_resume_state(tmp_path):
    torch = pytest.importorskip("torch")
    worker = _worker()
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    sampler = type("Sampler", (), {"epoch": 0, "policy": "shuffle_drop_global_tail"})()
    job = {"source_job": {"run_id": "source-run", "recipe_ref": "/recipe.yaml",
                           "fit": {"manifest_ref": "/fit.jsonl", "dataset_id": "d", "role": "fit"},
                           "source_val": {"manifest_ref": "/val.jsonl", "dataset_id": "d",
                                          "role": "source_val"},
                           "initialization": None, "training_seed": 13},
           "execution": {"architecture": {"model_id": "fixture"},
                         "class_index_map": {"spoof": 0, "bonafide": 1},
                         "random_rule": "explicit_sample_index_prng_v1"}}
    path = tmp_path / "epoch_0000.pt"
    worker.save_checkpoint(str(path), model, optimizer, scheduler, scaler, 0, 5, job,
                           {"kind": "none"}, 0, 1, sampler,
                           {"weighted_loss_numerator": 1.0, "sample_count": 2},
                           {"source_val_loss": 0.5, "source_val_eer": 0.25,
                            "source_val_count": 2, "source_val_ids": ["a", "b"]}, 0.25, 0)
    value = torch.load(path, map_location="cpu")
    assert set(value["model_state"]) == set(model.state_dict())
    assert {"optimizer_state", "scheduler_state", "scaler_state", "rng_states",
            "sampler_state", "best_metric", "best_epoch"}.issubset(value)
    clone = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
    clone.load_state_dict(value["model_state"], strict=True)
    for left, right in zip(model.state_dict().values(), clone.state_dict().values()):
        assert torch.equal(left, right)


def test_source_val_selection_uses_eer_then_earliest_epoch():
    assert equal_error_rate([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]) == 0.0
    selected = select_source_checkpoint([
        {"epoch": 2, "source_val_eer": 0.1, "checkpoint_ref": "epoch_0002.pt"},
        {"epoch": 1, "source_val_eer": 0.1, "checkpoint_ref": "epoch_0001.pt"}])
    assert selected["epoch"] == 1

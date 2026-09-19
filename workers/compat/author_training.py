"""Minimal adapters around author implementations in the active tta environment."""
from __future__ import absolute_import

import importlib.util
import json
import os
import random
import sys


AUTHOR = {
    "aasist_source": {
        "entrypoint": "models/AASIST.py",
    },
    "ssl_aasist_source": {
        "entrypoint": "model.py",
    },
}


def _merge_legacy_audio_task_config(original, dc, cfg, remove_missing, removed):
    """Discard unsupported legacy XLS-R task fields before task setup.

    Old checkpoints carry pretraining data/evaluation options that are absent
    from fairseq 0.12.2's ``AudioPretrainingConfig``.  They do not define model
    layers or weights.  Restrict cleanup to that task dataclass and record every
    removed field; model configuration and state loading remain untouched.
    """
    fields = getattr(dc, "__dataclass_fields__", {})
    if type(dc).__name__ == "AudioPretrainingConfig":
        unsupported = sorted(set(cfg.keys()) - set(fields))
        from omegaconf import open_dict
        with open_dict(cfg):
            for key in unsupported:
                del cfg[key]
        removed.extend("task." + key for key in unsupported)
    return original(dc, cfg, remove_missing=remove_missing)


def verify_author(execution):
    architecture = execution["architecture"]
    model_id = architecture["model_id"]
    expected = AUTHOR.get(model_id)
    if expected is None:
        raise ValueError("unregistered model_id")
    root = architecture["repository_ref"]
    entrypoint = os.path.join(root, expected["entrypoint"])
    if not os.path.isfile(entrypoint):
        raise ValueError("author model entrypoint is missing: %s" % entrypoint)
    return root, entrypoint


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuthorModelAdapter(object):
    def __init__(self, model, model_id):
        self.model = model
        self.model_id = model_id
        self.last_embedding = None
        self._hook = None
        if model_id == "ssl_aasist_source":
            def capture(_module, inputs):
                self.last_embedding = inputs[0]
            self._hook = model.out_layer.register_forward_pre_hook(capture)

    def forward(self, waveform, freq_aug=False):
        if self.model_id == "aasist_source":
            embedding, logits = self.model(waveform, Freq_aug=freq_aug)
            self.last_embedding = embedding
            return embedding, logits
        logits = self.model(waveform)
        if self.last_embedding is None:
            raise RuntimeError("SSL-AASIST head hook did not capture an embedding")
        return self.last_embedding, logits


def build_author_model(job, device):
    import torch
    execution = job["execution"]
    source_job = job["source_job"]
    model_id = source_job["model_id"]
    root, entrypoint = verify_author(execution)
    if root not in sys.path:
        sys.path.insert(0, root)
    if model_id == "aasist_source":
        module = _load_module(entrypoint, "eptta_pinned_aasist")
        with open(os.path.join(root, "config", "AASIST.conf"), encoding="utf-8") as stream:
            config = json.load(stream)["model_config"]
        model = module.Model(config)
        patch = {"kind": "none"}
    else:
        initialization = source_job.get("initialization")
        if not initialization or initialization.get("scope") != "generic_ssl_frontend_only":
            raise ValueError("SSL-AASIST requires generic SSL frontend initialization")
        init_path = initialization["artifact_ref"]
        if not os.path.isfile(init_path):
            raise ValueError("generic SSL initialization is missing")
        try:
            import fairseq
        except ImportError as exc:
            raise RuntimeError("the active tta environment must provide fairseq") from exc
        if not hasattr(fairseq, "checkpoint_utils"):
            raise RuntimeError("installed fairseq lacks checkpoint loading support")
        with open(entrypoint, encoding="utf-8") as stream:
            source = stream.read()
        old = "cp_path = 'xlsr2_300m.pt'"
        new = "cp_path = %r" % os.path.abspath(init_path)
        if source.count(old) != 1:
            raise ValueError("audited SSL initialization patch context mismatch")
        patched = source.replace(old, new)
        removed_legacy_fields = []
        original_task_merge = fairseq.tasks.merge_with_parent
        def compatible_task_merge(dc, cfg, remove_missing=False):
            return _merge_legacy_audio_task_config(
                original_task_merge, dc, cfg, remove_missing, removed_legacy_fields)
        fairseq.tasks.merge_with_parent = compatible_task_merge
        module = type(sys)("eptta_pinned_ssl_aasist")
        module.__file__ = entrypoint
        try:
            exec(compile(patched, entrypoint, "exec"), module.__dict__)
            model = module.Model(None, device)
        finally:
            fairseq.tasks.merge_with_parent = original_task_merge
        patch = {"kind": "explicit_generic_ssl_path_only", "artifact_ref": os.path.abspath(init_path),
                 "fairseq_version": getattr(fairseq, "__version__", "unknown"),
                 "removed_legacy_config_fields": removed_legacy_fields}
    model.to(device)
    return AuthorModelAdapter(model, model_id), patch


def independent_seed(seed, sample_index, epoch=0, view_index=0, namespace=0):
    """Mix explicit integers without Python hash or global RNG consumption order."""
    values = (seed, sample_index, epoch, view_index, namespace)
    if any(type(value) is not int for value in values):
        raise ValueError("random seed components must be integers")
    mask = (1 << 64) - 1
    value = 0x9E3779B97F4A7C15
    for item in values:
        value = (value + (item & mask) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return value & ((1 << 63) - 1)


def _crop_start(audio_length, length, seed, sample_index, epoch):
    if seed is None or audio_length <= length:
        return 0
    rng = random.Random(independent_seed(seed, sample_index, epoch, namespace=1))
    return rng.randrange(audio_length - length + 1)


def load_audio(path, expected_sample_rate=16000, length=64600, crop_seed=None,
               sample_index=0, epoch=0):
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError("soundfile is required in the source worker environment") from exc
    import numpy
    audio, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    if sample_rate != expected_sample_rate:
        raise ValueError("sample-rate mismatch for %s: %s" % (path, sample_rate))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0 or not numpy.isfinite(audio).all():
        raise ValueError("invalid audio tensor: %s" % path)
    if audio.shape[0] >= length:
        start = _crop_start(audio.shape[0], length, crop_seed, sample_index, epoch)
        return audio[start:start + length].copy()
    repeats = int(length / audio.shape[0]) + 1
    return numpy.tile(audio, repeats)[:length].copy()


class ManifestDataset(object):
    def __init__(self, manifest_ref, data_roots, role, training_seed=0):
        import torch
        self._dataset_base = torch.utils.data.Dataset
        self.rows = []
        self.role = role
        self.training_seed = int(training_seed)
        self.epoch = 0
        seen = set()
        with open(manifest_ref, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {"schema_version", "sample_id", "root_key", "audio_relpath",
                            "split_role", "canonical_label"}
                allowed = required | {"sample_index"}
                if not required.issubset(row) or set(row) - allowed or row["split_role"] != role:
                    raise ValueError("unsafe/mismatched %s manifest row %d" % (role, number))
                if row["sample_id"] in seen or row["canonical_label"] not in (0, 1):
                    raise ValueError("duplicate ID or invalid label in %s" % role)
                if row["root_key"] not in data_roots:
                    raise ValueError("unbound data root: %s" % row["root_key"])
                root = os.path.realpath(data_roots[row["root_key"]])
                audio_path = os.path.realpath(os.path.join(root, row["audio_relpath"]))
                if os.path.commonpath((root, audio_path)) != root:
                    raise ValueError("audio_relpath escapes its approved data root")
                seen.add(row["sample_id"])
                row["audio_path"] = audio_path
                self.rows.append(row)
        if not self.rows:
            raise ValueError("empty %s manifest" % role)
        if {row["canonical_label"] for row in self.rows} != {0, 1}:
            raise ValueError("%s manifest must contain both canonical classes" % role)
        provided = [row.get("sample_index") for row in self.rows]
        if all(value is None for value in provided):
            # Read compatibility only.  New prepare-data always persists explicit
            # indices; this migration changes old digest-derived random trajectories.
            index_by_id = {sample_id: index for index, sample_id in
                           enumerate(sorted(row["sample_id"] for row in self.rows))}
            for row in self.rows:
                row["sample_index"] = index_by_id[row["sample_id"]]
            self.random_rule = "legacy_sorted_sample_id_index_prng_v1"
        elif (any(type(value) is not int or value < 0 for value in provided) or
              len(set(provided)) != len(provided)):
            raise ValueError("sample_index must be unique non-negative integers")
        else:
            self.random_rule = "explicit_sample_index_prng_v1"

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        crop_seed = self.training_seed if self.role == "fit" else None
        waveform = torch.from_numpy(load_audio(row["audio_path"], crop_seed=crop_seed,
                                                sample_index=row["sample_index"], epoch=self.epoch))
        return waveform, int(row["canonical_label"]), row["sample_id"]


class EpochShardSampler(object):
    """No implicit DistributedSampler padding; tail policy is explicit."""
    def __init__(self, size, seed, rank, world_size, batch_size, policy):
        self.size = size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.policy = policy
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def indices(self):
        if self.size <= 0:
            raise ValueError("empty training set")
        order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(order)
        unit = self.world_size * self.batch_size
        if self.policy == "shuffle_drop_global_tail":
            order = order[:(len(order) // unit) * unit]
            if not order:
                raise ValueError("drop-tail training policy leaves no complete global batch")
        elif self.policy == "shuffle_repeat_to_even":
            needed = (-len(order)) % unit
            if needed:
                repeats = (needed + len(order) - 1) // len(order)
                order += (order * repeats)[:needed]
        elif self.policy != "shuffle_keep_tail" or self.world_size != 1:
            raise ValueError("unsupported sampler policy for this world size")
        return order[self.rank::self.world_size]

    def disclosure(self):
        """Report global repeats/drops without pretending to be a unique full pass."""
        unit = self.world_size * self.batch_size
        if self.policy == "shuffle_repeat_to_even":
            repeated = (-self.size) % unit
            dropped = 0
        elif self.policy == "shuffle_drop_global_tail":
            repeated = 0
            dropped = self.size % unit
        else:
            repeated = dropped = 0
        return {"source_sample_count": self.size, "repeated_sample_count": repeated,
                "dropped_sample_count": dropped,
                "unique_complete_pass": repeated == 0 and dropped == 0}


def canonical_to_native_tensor(labels, class_index_map):
    import torch
    mapping = torch.tensor([class_index_map["bonafide"], class_index_map["spoof"]],
                           dtype=torch.long, device=labels.device)
    return mapping[labels]

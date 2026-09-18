"""Python 3.7-compatible adapters around the pinned author implementations."""
from __future__ import absolute_import

import hashlib
import importlib.util
import json
import os
import random
import sys


AUTHOR = {
    "aasist_source": {
        "commit": "a04c9863f63d44471dde8a6abcb3b082b07cd1d1",
        "entrypoint": "models/AASIST.py",
        "sha256": "9e0d3e80937dd0577beea7883098465a479da23a198ebc0d712abcc59b0bec50",
    },
    "ssl_aasist_source": {
        "commit": "4acaa61dcef5f7610f43aa4d0b29c4559b970cd2",
        "entrypoint": "model.py",
        "sha256": "08b2b99b9cc0e90732746471325185f2eb144795ee35338e0a02951015a856c6",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_author(execution):
    architecture = execution["architecture"]
    model_id = architecture["model_id"]
    expected = AUTHOR.get(model_id)
    if expected is None:
        raise ValueError("unregistered model_id")
    root = architecture["repository_ref"]
    entrypoint = os.path.join(root, expected["entrypoint"])
    if architecture.get("repo_commit") != expected["commit"]:
        raise ValueError("author commit is not pinned to the audited revision")
    if sha256_file(entrypoint) != expected["sha256"]:
        raise ValueError("author entrypoint hash mismatch")
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
        patch = {"kind": "none", "sha256": hashlib.sha256(b"").hexdigest()}
    else:
        initialization = source_job.get("initialization")
        if not initialization or initialization.get("scope") != "generic_ssl_frontend_only":
            raise ValueError("SSL-AASIST requires generic SSL frontend initialization")
        init_path = initialization["artifact_ref"]
        if sha256_file(init_path) != initialization["sha256"]:
            raise ValueError("generic SSL initialization hash mismatch")
        vendored = os.path.join(root, "fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1")
        if vendored not in sys.path:
            sys.path.insert(0, vendored)
        with open(entrypoint, encoding="utf-8") as stream:
            source = stream.read()
        old = "cp_path = 'xlsr2_300m.pt'"
        new = "cp_path = %r" % os.path.abspath(init_path)
        if source.count(old) != 1:
            raise ValueError("audited SSL initialization patch context mismatch")
        patched = source.replace(old, new)
        patch_text = ("explicit_generic_ssl_path\n- %s\n+ %s\n" % (old, new)).encode("utf-8")
        patch = {"kind": "explicit_generic_ssl_path_only",
                 "sha256": hashlib.sha256(patch_text).hexdigest()}
        module = type(sys)("eptta_pinned_ssl_aasist")
        module.__file__ = entrypoint
        exec(compile(patched, entrypoint, "exec"), module.__dict__)
        model = module.Model(None, device)
    model.to(device)
    return AuthorModelAdapter(model, model_id), patch


def _crop_start(audio_length, length, crop_identity):
    if crop_identity is None or audio_length <= length:
        return 0
    digest = hashlib.sha256(crop_identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (audio_length - length + 1)


def load_audio(path, expected_sample_rate=16000, length=64600, crop_identity=None):
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
        start = _crop_start(audio.shape[0], length, crop_identity)
        return audio[start:start + length].copy()
    repeats = int(length / audio.shape[0]) + 1
    return numpy.tile(audio, repeats)[:length].copy()


class ManifestDataset(object):
    def __init__(self, manifest_ref, expected_hash, data_roots, role, training_seed=0):
        import torch
        self._dataset_base = torch.utils.data.Dataset
        if sha256_file(manifest_ref) != expected_hash:
            raise ValueError("%s manifest hash mismatch" % role)
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
                allowed = {"schema_version", "sample_id", "root_key", "audio_relpath",
                           "input_sha256", "split_role", "canonical_label"}
                if set(row) != allowed or row["split_role"] != role:
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

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        crop_identity = None
        if self.role == "fit":
            crop_identity = "%d\0%d\0%s" % (self.training_seed, self.epoch, row["sample_id"])
        waveform = torch.from_numpy(load_audio(row["audio_path"], crop_identity=crop_identity))
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

"""Factories for the pinned author networks; heavy imports stay worker-side."""
from pathlib import Path
import importlib.util

from eptta.errors import ContractError
from eptta.models.author import get_author_architecture, inspect_author_repository
from eptta.training.contracts import InitializationRef


class PinnedAuthorFactory:
    def __init__(self, model_id, repository_ref):
        self.model_id = model_id
        self.repository_ref = str(repository_ref)

    def class_index_map(self):
        return dict(get_author_architecture(self.model_id).class_index_map)

    def inspect(self):
        return inspect_author_repository(self.model_id, self.repository_ref)

    def build(self, architecture_lock, initialization=None, device="cpu"):
        verified = self.inspect()
        if architecture_lock.get("repo_commit") != verified["repo_commit"] or architecture_lock.get(
                "entrypoint_sha256") != verified["entrypoint_sha256"]:
            raise ContractError("architecture lock does not match pinned author source")
        if initialization is not None and type(initialization) is not InitializationRef:
            raise ContractError("initialization must use the typed generic-only contract")
        if self.model_id == "aasist_source" and initialization is not None:
            raise ContractError("AASIST cannot load initialization weights")
        if self.model_id == "ssl_aasist_source" and initialization is None:
            raise ContractError("SSL-AASIST requires generic SSL initialization")
        compat = Path(__file__).parents[3] / "workers/compat/author_training.py"
        spec = importlib.util.spec_from_file_location("eptta_author_training", compat)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        job = {"source_job": {"model_id": self.model_id,
                              "initialization": None if initialization is None else {
                                  "artifact_ref": initialization.artifact_ref, "sha256": initialization.sha256,
                                  "scope": initialization.scope,
                                  "pretraining_provenance": initialization.pretraining_provenance}},
               "execution": {"architecture": verified}}
        return module.build_author_model(job, device)


def get_model_factory(model_id, repository_ref):
    get_author_architecture(model_id)
    return PinnedAuthorFactory(model_id, repository_ref)

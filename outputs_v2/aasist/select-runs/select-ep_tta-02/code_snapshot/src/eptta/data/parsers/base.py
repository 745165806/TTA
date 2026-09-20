from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidatePlugin:
    dataset_id: str
    adapter_ids: tuple[str, ...]
    notes: tuple[str, ...]
    contract_required: bool = True

    def describe(self, inventory):
        return {"dataset_id": self.dataset_id, "adapter_candidates": self.adapter_ids,
                "notes": self.notes, "observations": inventory,
                "requires_explicit_dataset_config": self.contract_required}

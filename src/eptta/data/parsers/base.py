from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidatePlugin:
    dataset_id: str
    adapter_ids: tuple[str, ...]
    notes: tuple[str, ...]
    contract_required: bool = True

    def propose_contract(self, inventory):
        return {"dataset_id": self.dataset_id, "status": "PROPOSED", "adapter_candidates": self.adapter_ids,
                "notes": self.notes, "observations": inventory, "auto_approved": False}

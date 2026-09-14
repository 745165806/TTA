"""Proposal and human-review lock publication."""
from copy import deepcopy

from eptta.config.schema import check
from eptta.config.validate import check_contract, content_hash
from eptta.errors import ContractError


def empty_contract(kind):
    payloads = {
        "raw": {"format": None, "dataset_release": None, "encoding": None, "delimiter": None, "header": None, "columns": None,
                "json_paths": None, "record_id": None, "audio_path_rule": None, "label_field": None,
                "allowed_values": None, "missing_policy": None, "protocol_globs": None,
                "protocol_contexts": None},
        "label": {"policy_id": None, "raw_to_canonical": None, "unknown_policy": "quarantine"},
        "group": {"resolver": None, "source_mapping_ref": None, "source_mapping_sha256": None,
                  "group_quality": None},
    }
    return {"schema_version": "0.1.0", "status": "PROPOSED", "approval": None, "payload": payloads[kind]}


def propose_from_inventory(inventory):
    result = {"schema_version": "0.1.0", "status": "PROPOSED", "contracts": {}, "observations": {}}
    for dataset_id, observed in inventory["datasets"].items():
        result["contracts"][dataset_id] = {kind: empty_contract(kind) for kind in ("raw", "label", "group")}
        result["observations"][dataset_id] = {
            "protocol_files": [{key: item[key] for key in ("path", "size_bytes", "sha256", "sample_lines")}
                               for item in observed["protocol_files"]],
            "notice": "observations are candidates only; fill payloads and review before approval",
        }
    return result


def _extract(proposal, kind, dataset_id=None):
    if "contracts" in proposal:
        if not dataset_id:
            if len(proposal["contracts"]) != 1:
                raise ContractError("--dataset-id is required for a multi-dataset proposal")
            dataset_id = next(iter(proposal["contracts"]))
        try:
            return proposal["contracts"][dataset_id][kind]
        except KeyError as exc:
            raise ContractError(f"proposal has no {dataset_id}/{kind} contract") from exc
    if kind in proposal:
        return proposal[kind]
    return proposal


def approve(proposal, review, kind, dataset_id=None):
    contract = deepcopy(_extract(proposal, kind, dataset_id))
    check(contract, kind)
    if contract["status"] != "PROPOSED":
        raise ContractError("only a PROPOSED contract can be reviewed")
    required = {"accepted", "reviewer", "approved_at", "report_ref", "sample_evidence_ref"}
    if set(review) != required or review["accepted"] is not True:
        raise ContractError("review must explicitly accept and contain reviewer/time/report/sample evidence")
    contract["status"] = "LOCKED"
    contract["approval"] = {key: review[key] for key in required - {"accepted"}}
    contract["approval"]["content_sha256"] = content_hash(contract["payload"])
    issues = check_contract(contract, kind, kind)
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    return contract

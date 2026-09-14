"""Raw-to-canonical normalization at the trusted data-building boundary."""
import hashlib
from dataclasses import asdict

from eptta.data.contracts import CanonicalRecord, ContractIssue
from eptta.data.permissions import ExplicitLabelMapper, safe_relative
from eptta.errors import ContractError


def opaque_sample_id(dataset_id, dataset_release, record_id):
    identity = f"{dataset_id}\0{dataset_release or ''}\0{record_id}".encode("utf-8")
    return "s-" + hashlib.sha256(identity).hexdigest()


def _value(fields, name, record, issues):
    if name not in fields or fields[name] in (None, ""):
        issues.append(ContractIssue("MISSING_FIELD", name, record.record_ref,
                                    "required reviewed field is missing", True))
        return None
    return fields[name]


def normalize_record(record, dataset_id, dataset_release, staging_id, raw_contract, label_policy, group_policy,
                     group_mapping=None):
    raw = raw_contract["payload"]
    group = group_policy["payload"]
    issues = []
    record_id = _value(record.fields, raw["record_id"], record, issues)
    original_label = _value(record.fields, raw["label_field"], record, issues)
    mapping = ExplicitLabelMapper().map_label(original_label, label_policy)
    if original_label is not None and str(original_label) not in {str(v) for v in raw["allowed_values"]}:
        issues.append(ContractIssue("RAW_LABEL_OUTSIDE_VOCABULARY", raw["label_field"], record.record_ref,
                                    f"value not in approved vocabulary: {original_label!r}", True))
    if mapping.canonical_label is None:
        issues.append(ContractIssue("UNMAPPED_LABEL", raw["label_field"], record.record_ref,
                                    "label retained as null and record quarantined", raw["missing_policy"] == "error"))
    context = record.protocol_context or {}
    template = context.get("audio_path_template")
    audio_relpath = None
    if raw["audio_path_rule"] == "protocol_context_template" and template:
        try:
            audio_relpath = template.format_map({**record.fields, "record_id": record_id})
            safe_relative(audio_relpath)
        except (KeyError, ValueError, TypeError) as exc:
            issues.append(ContractIssue("AUDIO_PATH_TEMPLATE_ERROR", "audio_path_rule", record.record_ref,
                                        str(exc), True))
    elif isinstance(raw["audio_path_rule"], str) and raw["audio_path_rule"].startswith("field:"):
        key = raw["audio_path_rule"].split(":", 1)[1]
        audio_relpath = _value(record.fields, key, record, issues)
        if audio_relpath is not None:
            try:
                safe_relative(str(audio_relpath))
            except Exception as exc:
                issues.append(ContractIssue("UNSAFE_AUDIO_PATH", key, record.record_ref, str(exc), True))
    else:
        raise ContractError("unsupported reviewed audio_path_rule")
    resolver = group["resolver"]
    if resolver == "record_id":
        source_group_id = str(record_id) if record_id is not None else None
    elif isinstance(resolver, str) and resolver.startswith("field:"):
        source_group_id = _value(record.fields, resolver.split(":", 1)[1], record, issues)
        source_group_id = str(source_group_id) if source_group_id is not None else None
    elif isinstance(resolver, str) and resolver.startswith("mapping:"):
        key = resolver.split(":", 1)[1]
        raw_group = _value(record.fields, key, record, issues)
        source_group_id = None if raw_group is None else (group_mapping or {}).get(str(raw_group))
        if source_group_id is None:
            issues.append(ContractIssue("UNMAPPED_SOURCE_GROUP", key, record.record_ref,
                                        "source group mapping has no reviewed entry", True))
        else:
            source_group_id = str(source_group_id)
    else:
        raise ContractError("unsupported reviewed group resolver; use record_id or field:<logical-name>")
    quarantined = mapping.canonical_label is None or any(issue.blocking for issue in issues)
    if record_id is None or audio_relpath is None:
        return None, issues
    canonical = CanonicalRecord(
        "0.1.0", opaque_sample_id(dataset_id, dataset_release, str(record_id)), dataset_id, dataset_release,
        staging_id, context["root_key"], str(audio_relpath),
        {"file_key": record.source_file.rsplit("/", 1)[-1], "row": record.source_row},
        original_label, mapping.canonical_label, mapping.policy_id, mapping.policy_hash,
        mapping.mapping_reason, context.get("official_split"), "quarantine" if quarantined else "unassigned",
        source_group_id, group["group_quality"],
        speaker_id=str(record.fields["speaker_id"]) if record.fields.get("speaker_id") not in (None, "") else None,
        generator_id=str(record.fields["generator_id"]) if record.fields.get("generator_id") not in (None, "") else None,
        generator_family=str(record.fields["generator_family"]) if record.fields.get("generator_family") not in (None, "") else None,
        parent_id=str(record.fields["parent_id"]) if record.fields.get("parent_id") not in (None, "") else None,
        codec_id=str(record.fields["codec_id"]) if record.fields.get("codec_id") not in (None, "") else None,
        status="quarantined" if quarantined else "staged", arrival_batch_id=staging_id,
        source_release=dataset_release, source_file_sha256=record.source_file_sha256,
        raw_contract_hash=raw_contract["approval"]["content_sha256"])
    return asdict(canonical), issues

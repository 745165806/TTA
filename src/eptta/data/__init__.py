"""Reviewed data ingestion and immutable-manifest contracts."""

from eptta.data.contracts import CanonicalRecord, ContractIssue, LabelMapping, RawRecord
from eptta.data.source_manifests import (publish_label_free_manifest, seal_source_manifests,
                                         validate_source_snapshot)

__all__ = ["CanonicalRecord", "ContractIssue", "LabelMapping", "RawRecord",
           "publish_label_free_manifest", "seal_source_manifests", "validate_source_snapshot"]
from eptta.data.unified_labels import iter_unified_manifest, prepare_label_pack, validate_label_pack

__all__ = [
    "CanonicalRecord",
    "ContractIssue",
    "LabelMapping",
    "RawRecord",
    "iter_unified_manifest",
    "prepare_label_pack",
    "validate_label_pack",
]

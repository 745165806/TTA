"""Preprocess identity and cache compatibility without decoding or overwriting audio."""
from eptta.config.validate import check_contract, content_hash
from eptta.errors import ContractError, DataError


def preprocess_identity(contract):
    issues = check_contract(contract, "preprocess", "preprocess")
    if issues:
        raise ContractError("; ".join(issue.message for issue in issues))
    return contract["approval"]["content_sha256"]


def cache_identity(input_sha256, frozen_bundle_sha256, preprocess_contract, probe_id, dtype, numeric_mode):
    if not input_sha256 or not frozen_bundle_sha256:
        raise ContractError("cache identity requires content and frozen bundle hashes")
    return content_hash({"input_sha256": input_sha256, "frozen_bundle_sha256": frozen_bundle_sha256,
                         "preprocess_sha256": preprocess_identity(preprocess_contract), "probe_id": probe_id,
                         "dtype": dtype, "numeric_mode": numeric_mode})


def require_cache_preprocess(cache_metadata, preprocess_contract):
    expected = preprocess_identity(preprocess_contract)
    if cache_metadata.get("preprocess_sha256") != expected:
        raise DataError("cache preprocess identity mismatch; rebuild required")
    return True

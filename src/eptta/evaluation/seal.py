"""Scores are immutable before an evaluation sidecar may be opened."""
import math
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.evaluation.metrics import binary_metrics


def seal_scores(run_ref):
    run = Path(run_ref)
    score_path = run / "scores.jsonl"
    seen = set()
    count = 0
    for row in iter_jsonl(score_path):
        if set(row) - {"schema_version", "sample_id", "score", "score_before", "status", "method_id"}:
            raise DataError("score rows contain forbidden/unrecognized fields")
        if "canonical_label" in row or "attack" in row:
            raise DataError("unsealed run illegally contains target annotations")
        if row["sample_id"] in seen or not math.isfinite(row["score"]):
            raise DataError("duplicate ID or non-finite score")
        seen.add(row["sample_id"])
        count += 1
    if not count:
        raise DataError("cannot seal an empty score run")
    seal = {"schema_version": "0.1.0", "status": "SEALED", "score_ref": "scores.jsonl",
            "score_sha256": sha256_file(score_path), "sample_count": count,
            "sample_ids_sha256": content_hash(sorted(seen)), "labels_read": False, "immutable": True}
    write_json_new(run / "score_seal.json", seal)
    return seal


def evaluate_sealed(run_ref, labels_ref, output, threshold):
    run = Path(run_ref)
    seal_path = run / "score_seal.json"
    if not seal_path.is_file():
        raise ContractError("evaluation requires sealed scores before labels are opened")
    seal = read_json(seal_path)
    scores_path = run / seal["score_ref"]
    if seal.get("status") != "SEALED" or sha256_file(scores_path) != seal.get("score_sha256"):
        raise ContractError("evaluation requires an unchanged sealed score file")
    scores = {row["sample_id"]: row for row in iter_jsonl(scores_path)}
    labels = {}
    for row in iter_jsonl(labels_ref):
        if set(row) != {"schema_version", "sample_id", "canonical_label"}:
            raise DataError("evaluation label sidecar has an invalid field set")
        if row["sample_id"] in labels or row["canonical_label"] not in (0, 1):
            raise DataError("duplicate/invalid evaluation label")
        labels[row["sample_id"]] = row["canonical_label"]
    if set(scores) != set(labels) or len(scores) != seal["sample_count"]:
        raise DataError("sealed score and label ID coverage differ")
    order = sorted(scores)
    adapted = [scores[key]["score"] for key in order]
    frozen = [scores[key].get("score_before") for key in order]
    if any(value is None for value in frozen):
        frozen = None
    result = {"schema_version": "0.1.0", "status": "EVALUATED",
              "score_sha256": seal["score_sha256"], "labels_sha256": sha256_file(labels_ref),
              "metrics": binary_metrics(adapted, [labels[key] for key in order], threshold, frozen)}
    write_json_new(output, result)
    return result

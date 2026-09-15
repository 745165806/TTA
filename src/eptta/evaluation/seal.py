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
    status_counts = {}
    for row in iter_jsonl(score_path):
        if set(row) - {"schema_version", "sample_id", "score", "score_before", "status", "method_id",
                       "steps_completed", "objective_evaluations", "r_fro", "runtime_ref"}:
            raise DataError("score rows contain forbidden/unrecognized fields")
        if "canonical_label" in row or "attack" in row:
            raise DataError("unsealed run illegally contains target annotations")
        if row["sample_id"] in seen or not math.isfinite(row["score"]):
            raise DataError("duplicate ID or non-finite score")
        seen.add(row["sample_id"])
        status_counts[row.get("status", "unknown")] = status_counts.get(row.get("status", "unknown"), 0) + 1
        count += 1
    if not count:
        raise DataError("cannot seal an empty score run")
    run_metadata_path = run / "run.json"
    expected = count
    run_metadata_sha256 = None
    runtime_ref = None
    runtime_sha256 = None
    if run_metadata_path.is_file():
        run_metadata = read_json(run_metadata_path)
        expected = run_metadata.get("expected_sample_count", run_metadata.get("sample_count", count))
        if run_metadata.get("scores_sha256") != sha256_file(score_path):
            raise DataError("run metadata and scores differ")
        run_metadata_sha256 = sha256_file(run_metadata_path)
        cost = run_metadata.get("cost", {})
        runtime_ref = cost.get("runtime_ref")
        runtime_sha256 = cost.get("runtime_sha256")
        if runtime_ref:
            runtime_path = run / runtime_ref
            if not runtime_path.is_file() or sha256_file(runtime_path) != runtime_sha256:
                raise DataError("runtime cost report is missing or changed")
    coverage = {"schema_version": "0.1.0", "status": "COMPLETE" if expected == count else "INCOMPLETE",
                "expected_sample_count": expected, "scored_sample_count": count,
                "coverage_fraction": count / expected if expected else 0.0,
                "status_counts": status_counts, "sample_ids_sha256": content_hash(sorted(seen))}
    coverage_path = run / "coverage.json"
    write_json_new(coverage_path, coverage)
    if expected != count:
        raise DataError("score coverage is incomplete")
    seal = {"schema_version": "0.1.0", "status": "SEALED", "score_ref": "scores.jsonl",
            "score_sha256": sha256_file(score_path), "sample_count": count,
            "sample_ids_sha256": content_hash(sorted(seen)), "coverage_report_ref": "coverage.json",
            "coverage_report_sha256": sha256_file(coverage_path),
            "run_metadata_sha256": run_metadata_sha256, "runtime_report_ref": runtime_ref,
            "runtime_report_sha256": runtime_sha256,
            "labels_read": False, "immutable": True}
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
    coverage_path = run / seal.get("coverage_report_ref", "")
    if (not coverage_path.is_file() or sha256_file(coverage_path) !=
            seal.get("coverage_report_sha256")):
        raise ContractError("evaluation requires an unchanged complete coverage report")
    coverage = read_json(coverage_path)
    if coverage.get("status") != "COMPLETE" or coverage.get("coverage_fraction") != 1.0:
        raise ContractError("evaluation refuses incomplete score coverage")
    run_metadata_path = run / "run.json"
    if seal.get("run_metadata_sha256") is not None and (not run_metadata_path.is_file() or
            sha256_file(run_metadata_path) != seal["run_metadata_sha256"]):
        raise ContractError("evaluation requires unchanged run metadata")
    if seal.get("runtime_report_ref") is not None:
        runtime_path = run / seal["runtime_report_ref"]
        if not runtime_path.is_file() or sha256_file(runtime_path) != seal.get("runtime_report_sha256"):
            raise ContractError("evaluation requires the sealed runtime cost report")
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
              "coverage_report_sha256": seal["coverage_report_sha256"],
              "labels_read": True, "evaluator_role": "independent_post_seal",
              "metrics": binary_metrics(adapted, [labels[key] for key in order], threshold, frozen)}
    write_json_new(output, result)
    return result

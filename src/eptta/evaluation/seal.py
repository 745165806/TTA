"""Scores are immutable before an evaluation sidecar may be opened."""
import math
import csv
import json
from pathlib import Path

from eptta.config.validate import content_hash
from eptta.data.io import AtomicDirectory, iter_jsonl, read_json, sha256_file, write_json_new
from eptta.errors import ContractError, DataError
from eptta.evaluation.metrics import binary_metrics


def seal_scores(run_ref):
    run = Path(run_ref)
    score_path = run / "scores.jsonl"
    run_metadata_path = run / "run.json"
    if not run_metadata_path.is_file():
        raise DataError("score sealing requires run.json with preregistered expected IDs")
    run_metadata = read_json(run_metadata_path)
    expected_ref = run_metadata.get("expected_ids_ref")
    expected_sha256 = run_metadata.get("expected_ids_sha256")
    expected_path = run / expected_ref if isinstance(expected_ref, str) else None
    if (expected_path is None or not expected_path.is_file() or
            sha256_file(expected_path) != expected_sha256):
        raise DataError("run metadata lacks unchanged preregistered expected IDs")
    expected_ids = read_json(expected_path)
    if (type(expected_ids) is not list or not expected_ids or
            any(type(value) is not str or not value for value in expected_ids) or
            len(expected_ids) != len(set(expected_ids))):
        raise DataError("preregistered expected IDs are invalid")
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
    expected = len(expected_ids)
    run_metadata_sha256 = sha256_file(run_metadata_path)
    runtime_ref = None
    runtime_sha256 = None
    if run_metadata.get("expected_sample_count") != expected or run_metadata.get(
            "scores_sha256") != sha256_file(score_path):
        raise DataError("run metadata and scores/expected IDs differ")
    cost = run_metadata.get("cost", {})
    runtime_ref = cost.get("runtime_ref")
    runtime_sha256 = cost.get("runtime_sha256")
    if runtime_ref:
        runtime_path = run / runtime_ref
        if not runtime_path.is_file() or sha256_file(runtime_path) != runtime_sha256:
            raise DataError("runtime cost report is missing or changed")
    exact_ids = seen == set(expected_ids)
    coverage = {"schema_version": "0.2.0", "status": "COMPLETE" if expected == count and exact_ids else "INCOMPLETE",
                "expected_sample_count": expected, "scored_sample_count": count,
                "coverage_fraction": count / expected if expected else 0.0,
                "status_counts": status_counts, "sample_ids_sha256": content_hash(sorted(seen)),
                "expected_sample_ids_sha256": content_hash(sorted(expected_ids)),
                "exact_id_coverage": exact_ids}
    coverage_path = run / "coverage.json"
    write_json_new(coverage_path, coverage)
    if expected != count or not exact_ids:
        raise DataError("score ID coverage differs from the preregistered set")
    fallback_count = status_counts.get("fallback_numeric", 0)
    fallback_rate = fallback_count / expected
    limit = run_metadata.get("fallback_rate_max", 0.0)
    identity_ok = run_metadata.get("feature_cache_key") is not None and run_metadata.get(
        "artifact_bundle_id") is not None
    valid = identity_ok and fallback_rate <= limit
    seal = {"schema_version": "0.2.0", "status": "SEALED", "score_ref": "scores.jsonl",
            "score_sha256": sha256_file(score_path), "sample_count": count,
            "sample_ids_sha256": content_hash(sorted(seen)), "coverage_report_ref": "coverage.json",
            "coverage_report_sha256": sha256_file(coverage_path),
            "run_metadata_sha256": run_metadata_sha256, "runtime_report_ref": runtime_ref,
            "runtime_report_sha256": runtime_sha256,
            "fallback_count": fallback_count, "fallback_rate": fallback_rate,
            "fallback_rate_max": limit, "identity_valid": identity_ok,
            "valid_for_comparison": valid, "labels_read": False, "immutable": True}
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


def build_report(run_root, output):
    """Render registered evaluations only; never rescore, pool targets, or tune."""
    root = Path(run_root).resolve()
    plan_path = root / "report-plan.json"
    if not plan_path.is_file():
        raise ContractError("report requires ROOT/report-plan.json with explicit registered rows")
    plan = read_json(plan_path)
    if plan.get("status") != "LOCKED" or type(plan.get("rows")) is not list:
        raise ContractError("report plan must be LOCKED and list rows")
    fields = ["source", "model", "seed", "target", "release", "subset", "method", "config",
              "N", "EER_pct", "AUROC", "FPR_pct", "FNR_pct", "helpful_flips_by_class",
              "harmful_flips_by_class", "fallback", "cache_only_seconds", "encoding_seconds",
              "status", "reason"]
    rows = []
    for registered in plan["rows"]:
        base = {key: registered.get(key) for key in
                ("source", "model", "seed", "target", "release", "subset", "method", "config")}
        evaluation_ref = registered.get("evaluation_ref")
        run_ref = registered.get("run_ref")
        evaluation_path = (root / evaluation_ref).resolve() if evaluation_ref else None
        method_run_path = (root / run_ref).resolve() if run_ref else None
        if not evaluation_path or not evaluation_path.is_file():
            rows.append({**base, "N": None, "EER_pct": None, "AUROC": None, "FPR_pct": None,
                         "FNR_pct": None, "helpful_flips_by_class": None,
                         "harmful_flips_by_class": None, "fallback": None,
                         "cache_only_seconds": None, "encoding_seconds": None,
                         "status": registered.get("status", "NOT_RUN"),
                         "reason": registered.get("reason", "evaluation_missing")})
            continue
        evaluation = read_json(evaluation_path)
        metrics = evaluation.get("metrics", {})
        method_run = read_json(method_run_path) if method_run_path and method_run_path.is_file() else {}
        cost = method_run.get("cost", {})
        eer, fpr, fnr = metrics.get("eer"), metrics.get("fpr"), metrics.get("fnr")
        missing = []
        for name, value in (("EER", eer), ("AUROC", metrics.get("auroc")),
                            ("FPR", fpr), ("FNR", fnr),
                            ("cache_only_seconds", cost.get("elapsed_seconds")),
                            ("encoding_seconds", registered.get("encoding_seconds"))):
            if value is None:
                missing.append("missing_" + name)
        rows.append({**base, "N": method_run.get("sample_count", metrics.get("count")),
                     "EER_pct": None if eer is None else 100.0 * eer,
                     "AUROC": metrics.get("auroc"), "FPR_pct": None if fpr is None else 100.0 * fpr,
                     "FNR_pct": None if fnr is None else 100.0 * fnr,
                     "helpful_flips_by_class": metrics.get("helpful_flips_by_class"),
                     "harmful_flips_by_class": metrics.get("harmful_flips_by_class"),
                     "fallback": method_run.get("fallback_count"),
                     "cache_only_seconds": cost.get("elapsed_seconds"),
                     "encoding_seconds": registered.get("encoding_seconds"),
                     "status": evaluation.get("status", "EVALUATED"),
                     "reason": None if not missing else ";".join(missing)})
    destination = Path(output)
    with AtomicDirectory(destination) as temporary:
        with (temporary / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(row[key], sort_keys=True) if isinstance(row[key], (dict, list))
                                 else row[key] for key in fields})
        write_json_new(temporary / "summary.json", {"schema_version": "0.2.0", "rows": rows,
                                                     "pooled_across_datasets": False})
        lines = ["# Experiment report", "", "Metrics are reported per registered dataset scope; no pooled EER is computed.", "",
                 "| target | release/subset | method | N | EER (%) | AUROC | status | reason |",
                 "|---|---|---|---:|---:|---:|---|---|"]
        for row in rows:
            lines.append("| %s | %s/%s | %s | %s | %s | %s | %s | %s |" % (
                row["target"], row["release"], row["subset"], row["method"], row["N"],
                row["EER_pct"], row["AUROC"], row["status"], row["reason"] or ""))
        (temporary / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"schema_version": "0.2.0", "status": "REPORTED", "row_count": len(rows),
            "output": str(destination.resolve())}

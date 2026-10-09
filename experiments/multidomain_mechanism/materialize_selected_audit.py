"""Create selected-only audit labels from a completed post-score mechanism run.

Never opens an official evaluation-pool label file. The source is the existing
post-score sample_mechanisms.csv, whose score provenance must already be final.
"""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "audit_labels"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def materialize(source_run):
    source_run = source_run.resolve()
    if source_run.parent != (HERE / "results").resolve():
        raise ValueError("source must be a mechanism run")
    summary = read_json(source_run / "analysis/summary.json")
    if summary["status"] != "PARTIAL_THREE_DOMAIN_MECHANISM_DEV":
        raise ValueError("completed post-score analysis required")
    rows = list(csv.DictReader((source_run / "analysis/sample_mechanisms.csv").open(encoding="utf-8")))
    observed = {}
    for row in rows:
        domain, sid, value = row["domain"], row["sample_id"], int(row["label"])
        if value not in (0, 1):
            raise ValueError("noncanonical selected label")
        key = (domain, sid)
        if key in observed and observed[key] != value:
            raise ValueError("inconsistent repeated selected label")
        observed[key] = value
    OUTPUT.mkdir(exist_ok=False)
    for domain in summary["audited_domains"]:
        audit = read_json(HERE / "manifests" / f"{domain}_mechanism_audit.json")
        ids = {row["sample_id"] for row in audit["records"]}
        labels = {sid: y for (d, sid), y in observed.items() if d == domain}
        if ids != set(labels) or len(ids) != audit["count"]:
            raise ValueError("selected-only audit coverage mismatch")
        artifact = {"schema_version": "0.1.0", "role": "selected_only_post_score_audit",
                    "dataset_id": domain, "source_run": source_run.name, "count": len(ids),
                    "records": [{"sample_id": sid, "label": labels[sid]} for sid in sorted(ids)]}
        with (OUTPUT / f"{domain}.json").open("x", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source_run)


if __name__ == "__main__":
    main()

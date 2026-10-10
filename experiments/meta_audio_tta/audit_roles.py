"""Read-only source assignment audit; no assignment regeneration or content hashes."""
import argparse
import json
from pathlib import Path


def run(config):
    summary = {}
    all_ids = set()
    for role in ("fit", "source_val", "select", "cal0"):
        manifest = Path(config["roles"][role]["manifest"])
        with manifest.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row["sample_id"] for row in rows]
        if not rows or len(ids) != len(set(ids)) or set(ids) & all_ids:
            raise ValueError(role + " has empty, duplicate or overlapping source IDs")
        if any(row.get("split_role") != role or
               type(row.get("sample_index")) is not int or row["sample_index"] < 0
               for row in rows):
            raise ValueError(role + " has an invalid explicit role/index")
        all_ids.update(ids)
        if role in ("fit", "source_val"):
            labels = {row["sample_id"]: row["canonical_label"] for row in rows}
        else:
            if any(row.get("canonical_label") is not None for row in rows):
                raise ValueError(role + " master manifest unexpectedly exposes labels")
            sidecar = manifest.parent.parent / "labels" / f"{role}.jsonl"
            with sidecar.open(encoding="utf-8") as stream:
                label_rows = [json.loads(line) for line in stream if line.strip()]
            labels = {row["sample_id"]: row["canonical_label"] for row in label_rows}
            if len(labels) != len(label_rows) or set(labels) != set(ids):
                raise ValueError(role + " label sidecar coverage/uniqueness mismatch")
        if any(type(label) is not int or label not in (0, 1)
               for label in labels.values()):
            raise ValueError(role + " invalid canonical labels")
        summary[role] = {"manifest": str(manifest), "count": len(rows),
                         "bonafide": sum(label == 0 for label in labels.values()),
                         "spoof": sum(label == 1 for label in labels.values())}
    return {"status": "PASS", "assignments_regenerated": False,
            "role_ids_disjoint": True, "roles": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite role audit")
    result = run(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

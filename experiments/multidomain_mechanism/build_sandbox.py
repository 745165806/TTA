"""One-shot, label-free mechanism assignment from explicit local sources.

This script never opens target90, ASV label files, or Codecfake label columns.
It uses exclusive writes; a completed assignment cannot be silently redrawn.
"""
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"
SEED = 2026
TARGET = 512
ASV = {
    "asv2021_la": ("asv2021_la_eval", "asvspoof2021_la",
                   "/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_LA_eval"),
    "asv2021_df": ("asv2021_df_eval", "asvspoof2021_df",
                   "/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_DF_eval"),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rows_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def choose_ids(records, count=TARGET):
    ordered = sorted(records, key=lambda row: row["sample_id"])
    if len({r["sample_id"] for r in ordered}) != len(ordered):
        raise ValueError("duplicate sample IDs")
    rng = random.Random(SEED)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    return sorted((ordered[i] for i in indices[:min(count, len(indices))]),
                  key=lambda row: row["sample_id"])


def make_record(row, dataset, original_split, root):
    expected = {"sample_id", "root_key", "audio_relpath"}
    if not expected <= row.keys() or not row["sample_id"] or not row["audio_relpath"]:
        raise ValueError("incomplete inference row")
    rel = Path(row["audio_relpath"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unsafe audio path")
    return {
        "sample_id": row["sample_id"], "dataset_id": dataset,
        "original_split": original_split, "root_key": row["root_key"],
        "root_ref": root, "audio_relpath": row["audio_relpath"],
        "selection_seed": SEED,
    }


def emit(domain, records, *, policy, source_ref, label_ref=None, holdout_rule=None):
    for row in records:
        row["selection_policy"] = policy
    shared = {"schema_version": "0.1.0", "dataset_id": domain,
              "selection_seed": SEED, "selection_policy": policy,
              "source_ref": source_ref, "count": len(records), "records": records}
    select = dict(shared, role="mechanism_select")
    audit = dict(shared, role="mechanism_audit", label_status="DEFERRED_UNTIL_SCORES_COMPLETE",
                 label_source_ref=label_ref)
    if holdout_rule is not None:
        select["final_holdout_rule"] = holdout_rule
        audit["final_holdout_rule"] = holdout_rule
    write_once(MANIFESTS / f"{domain}_mechanism_select.json", select)
    write_once(MANIFESTS / f"{domain}_mechanism_audit.json", audit)
    return {"status": "READY" if records else "BLOCKED", "mechanism_count": len(records),
            "selection_policy": policy, "source_ref": source_ref, "label_source_ref": label_ref,
            "final_holdout_rule": holdout_rule}


def in_the_wild():
    source = ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json"
    doc = read_json(source)
    if doc.get("role") != "select" or doc.get("dataset_id") != "in_the_wild" or doc.get("count") != 3178:
        raise ValueError("target10 select contract changed")
    if "label" in json.dumps(doc):
        raise ValueError("target10 select contains label")
    root = "/media/dell/data/fakedata/release_in_the_wild"
    records = [make_record(row, "in_the_wild", "existing_target10", root)
               for row in choose_ids(doc["records"])]
    return emit("in_the_wild", records, policy="fixed_target10_id_sample_without_repartition",
                source_ref=str(source.relative_to(ROOT)),
                label_ref="experiments/target10_selection/manifests/inwild_target10.json")


def codecfake():
    protocol = Path("/media/dell/data/fakedata/Codecfake_Xie/extracted/label/label/dev.txt")
    root = Path("/media/dell/data/fakedata/Codecfake_Xie/extracted")
    records = []
    with protocol.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 3:
                raise ValueError("Codecfake dev format differs from explicit 3-column contract")
            filename = fields[0]  # label and attack columns never enter select
            records.append({"sample_id": "dev/" + filename, "root_key": "codecfake_xie",
                            "audio_relpath": "dev/" + filename})
    chosen = [make_record(row, "codecfake", "official_dev", str(root))
              for row in choose_ids(records)]
    if not all((root / row["audio_relpath"]).is_file() for row in chosen):
        raise ValueError("selected Codecfake dev audio missing")
    return emit("codecfake", chosen, policy="official_dev_sorted_ids_random2026",
                source_ref=str(root / "dev"), label_ref=str(protocol))


def asv(domain):
    folder, root_key, root = ASV[domain]
    base = ROOT / "data/manifests_v2" / folder
    groups_ref = base / "groups/target_test.jsonl"
    inference_ref = base / "inference/target_test.jsonl"
    groups = defaultdict(list)
    for row in rows_jsonl(groups_ref):
        if set(row) != {"sample_id", "schema_version", "source_group_id"}:
            raise ValueError("ASV group schema changed")
        groups[row["source_group_id"]].append(row["sample_id"])
    rng = random.Random(SEED)
    names = sorted(groups)
    rng.shuffle(names)
    selected_groups = []
    count = 0
    for name in names:
        size = len(groups[name])
        if count + size <= TARGET:
            selected_groups.append(name)
            count += size
        if count >= 256:
            break
    if count < 256:
        raise ValueError("cannot form a group-disjoint 256-512 sample ASV sandbox")
    wanted = {sid for name in selected_groups for sid in groups[name]}
    records = []
    for row in rows_jsonl(inference_ref):
        if row["sample_id"] in wanted:
            if row["root_key"] != root_key or row["split_role"] != "target_test":
                raise ValueError("ASV inference provenance changed")
            records.append(make_record(row, domain, "official_eval", root))
    if len(records) != count or len({r["sample_id"] for r in records}) != count:
        raise ValueError("ASV group/inference coverage mismatch")
    records.sort(key=lambda row: row["sample_id"])
    if not all((Path(root) / row["audio_relpath"]).is_file() for row in records):
        raise ValueError("selected ASV audio missing")
    rule = {"source_ref": str(inference_ref.relative_to(ROOT)),
            "mechanism_group_ids": sorted(selected_groups),
            "final_holdout": "all other explicit source_group_id values in this official_eval pool",
            "group_ref": str(groups_ref.relative_to(ROOT))}
    return emit(domain, records, policy="group_disjoint_official_eval_seed2026",
                source_ref=str(inference_ref.relative_to(ROOT)),
                label_ref=str((base / "labels/target_test.jsonl").relative_to(ROOT)),
                holdout_rule=rule)


def wavefake():
    source = Path("/media/dell/data/fakedata/WaveFake")
    if not source.is_dir():
        status = "MISSING"
    elif list((source / "data").glob("*.parquet")):
        status = "BLOCKED_PARQUET_READER_AND_FEATURE_CACHE"
    else:
        status = "MISSING_AUDIO_PARTITIONS"
    info = emit("wavefake", [], policy="pending_explicit_parquet_reader_and_feature_extraction",
                source_ref=str(source), label_ref=None)
    info["status"] = status
    return info


def main():
    if MANIFESTS.exists() and any(MANIFESTS.iterdir()):
        raise FileExistsError("sandbox assignment already exists; never redraw")
    inventory = {"in_the_wild": in_the_wild(), "codecfake": codecfake(),
                 "asv2021_la": asv("asv2021_la"), "asv2021_df": asv("asv2021_df"),
                 "wavefake": wavefake()}
    write_once(HERE / "sandbox_inventory.json", inventory)


if __name__ == "__main__":
    main()

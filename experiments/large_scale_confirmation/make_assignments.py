"""One-time label-free development assignments. No protocol/label file is opened."""
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ITW_ROOT = Path("/media/dell/data/fakedata/release_in_the_wild")
CODEC_ROOT = Path("/media/dell/data/fakedata/Codecfake_Xie/extracted")
LA_ROOT = Path("/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2019_LA_dev")
FORBIDDEN = {"label", "raw_label", "canonical_label", "attack_id", "correct_before",
             "correct_after", "helpful_update", "harmful_update"}


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def document(domain, root, policy, records):
    ids = [r["sample_id"] for r in records]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or any(FORBIDDEN.intersection(r) for r in records):
        raise ValueError("selection is unsorted, duplicate, or contains forbidden fields")
    return {"schema_version": "0.1.0", "dataset_id": domain, "role": "confirmation_select",
            "selection_seed": 2026, "selection_policy": policy, "root_ref": str(root),
            "count": len(records), "records": records}


def generate():
    old_itw = json.loads((ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json")
                         .read_text(encoding="utf-8"))
    if old_itw.get("role") != "select" or old_itw.get("count") != 3178:
        raise ValueError("existing target10 selection contract changed")
    itw_records = [{"sample_id": r["sample_id"], "sample_index": r["sample_index"],
                    "dataset_id": "in_the_wild", "original_split": "existing_target10",
                    "root_ref": str(ITW_ROOT), "audio_relpath": r["audio_relpath"],
                    "selection_seed": 2026} for r in old_itw["records"]]
    if any(FORBIDDEN.intersection(r) for r in old_itw["records"]):
        raise ValueError("target10 select contains forbidden field")
    itw_records.sort(key=lambda r: r["sample_id"])
    if len(itw_records) != 3178 or len({r["sample_index"] for r in itw_records}) != 3178:
        raise ValueError("target10 IDs or sample indices incomplete")

    fixed = json.loads((ROOT / "experiments/multidomain_mechanism/manifests/"
                        "codecfake_mechanism_select.json").read_text(encoding="utf-8"))
    if fixed.get("count") != 512 or fixed.get("role") != "mechanism_select":
        raise ValueError("old Codecfake fixed selection changed")
    fixed_ids = sorted(r["sample_id"] for r in fixed["records"])
    if len(set(fixed_ids)) != 512 or any(FORBIDDEN.intersection(r) for r in fixed["records"]):
        raise ValueError("invalid old Codecfake selected IDs")
    available = sorted("dev/" + p.name for p in (CODEC_ROOT / "dev").glob("*.wav") if p.is_file())
    remaining = sorted(set(available) - set(fixed_ids))
    if len(available) != 92596 or len(remaining) < 4488 or not set(fixed_ids) <= set(available):
        raise ValueError("Codecfake official-dev availability changed")
    added = sorted(random.Random(2026).sample(remaining, 4488))
    sample_indices = {sid: index for index, sid in enumerate(fixed_ids)}
    sample_indices.update({sid: 512 + index for index, sid in enumerate(added)})
    codecfake_records = [{"sample_id": sid, "sample_index": sample_indices[sid],
                          "dataset_id": "codecfake", "original_split": "official_dev",
                          "root_ref": str(CODEC_ROOT), "audio_relpath": sid,
                          "selection_seed": 2026, "old_fixed512_member": sid in set(fixed_ids)}
                         for sid in sorted(sample_indices)]

    la_available = sorted(p.stem for p in (LA_ROOT / "flac").glob("*.flac") if p.is_file())
    if len(la_available) != 24986 or len(set(la_available)) != len(la_available):
        raise ValueError("ASVspoof2019 LA official-dev FLAC availability changed")
    la_ids = sorted(random.Random(2026).sample(la_available, 5000))
    la_records = [{"sample_id": sid, "sample_index": index,
                   "dataset_id": "asv2019_la_dev", "original_split": "official_dev",
                   "root_ref": str(LA_ROOT), "audio_relpath": "flac/" + sid + ".flac",
                   "selection_seed": 2026} for index, sid in enumerate(la_ids)]

    folder = HERE / "manifests"
    folder.mkdir(exist_ok=False)
    choices = {
        "in_the_wild": document("in_the_wild", ITW_ROOT,
                                "reuse_existing_target10_all_3178", itw_records),
        "codecfake": document("codecfake", CODEC_ROOT,
                              "fixed512_plus_random2026_4488_official_dev", codecfake_records),
        "asv2019_la_dev": document("asv2019_la_dev", LA_ROOT,
                                   "random2026_5000_available_official_dev_ids", la_records),
    }
    for domain, value in choices.items():
        write_once(folder / (domain + "_confirmation_select.json"), value)
    write_once(HERE / "assignment_summary.json", {
        "selection_seed": 2026, "label_files_opened": False,
        "official_available_counts": {"codecfake": len(available),
                                      "asv2019_la_dev": len(la_available)},
        "selected_counts": {domain: value["count"] for domain, value in choices.items()},
        "codecfake_old_fixed512_included": sum(r["old_fixed512_member"] for r in codecfake_records),
        "selection_policy": {domain: value["selection_policy"] for domain, value in choices.items()},
    })
    print("assignments fixed", {domain: value["count"] for domain, value in choices.items()})


if __name__ == "__main__":
    generate()

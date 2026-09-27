"""Fixed label-free large development assignment invariants."""
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
HERE = ROOT / "experiments/large_scale_confirmation"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_fixed_counts_order_and_label_free_records():
    wanted = {"in_the_wild": 3178, "codecfake": 5000, "asv2019_la_dev": 5000}
    forbidden = {"label", "raw_label", "canonical_label", "attack_id"}
    for domain, count in wanted.items():
        doc = read(HERE / "manifests" / (domain + "_confirmation_select.json"))
        rows = doc["records"]
        ids = [row["sample_id"] for row in rows]
        assert doc["role"] == "confirmation_select"
        assert len(rows) == doc["count"] == count
        assert ids == sorted(ids) and len(set(ids)) == count
        assert len({row["sample_index"] for row in rows}) == count
        assert not forbidden.intersection(doc)
        assert all(not forbidden.intersection(row) for row in rows)


def test_old_itw_and_codecfake_assignments_are_preserved():
    old_itw = read(ROOT / "experiments/target10_selection/manifests/inwild_target10_select.json")
    new_itw = read(HERE / "manifests/in_the_wild_confirmation_select.json")
    assert {r["sample_id"]: r["sample_index"] for r in old_itw["records"]} == {
        r["sample_id"]: r["sample_index"] for r in new_itw["records"]}
    fixed = read(ROOT / "experiments/multidomain_mechanism/manifests/"
                 "codecfake_mechanism_select.json")
    old_ids = sorted(row["sample_id"] for row in fixed["records"])
    new = {r["sample_id"]: r for r in read(HERE / "manifests/codecfake_confirmation_select.json")["records"]}
    assert len(old_ids) == 512
    assert all(new[sid]["old_fixed512_member"] and new[sid]["sample_index"] == index
               for index, sid in enumerate(old_ids))


def test_audio_header_audit_is_label_free():
    report = read(HERE / "resource_audit.json")
    assert report["target_labels_read"] is False
    assert sum(report["selected_audio_sample_rate_counts"]["codecfake"].values()) == 5000
    assert report["selected_audio_sample_rate_counts"]["asv2019_la_dev"] == {"16000": 5000}

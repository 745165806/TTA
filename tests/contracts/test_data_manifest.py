import csv

from eptta.data.splits import stable_group_roles
from eptta.research import prepare_data


def test_new_group_assignment_is_order_independent_and_group_safe():
    first = stable_group_roles(13, ["g3", "g1", "g2"], {"fit": 0.7, "source_val": 0.3})
    second = stable_group_roles(13, ["g2", "g3", "g1"], {"fit": 0.7, "source_val": 0.3})
    assert first == second and set(first) == {"g1", "g2", "g3"}


def test_prepare_data_preserves_prior_role_and_index_then_appends(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,path,label,group\na,a.wav,bonafide,g1\nb,b.wav,spoof,g2\n")
    assignments = tmp_path / "assignments.csv"
    assignments.write_text("sample_id,sample_index,group_id,split_role\na,7,g1,source_val\n")
    output = tmp_path / "prepared"
    prepare_data({"schema_version": "0.1.0", "command": "prepare-data",
                  "dataset": {"dataset_id": "fixture", "release": "r1", "subset": "train",
                              "manifest": str(manifest)},
                  "columns": {"sample_id": "id", "audio_relpath": "path", "label": "label",
                              "group_id": "group"},
                  "label_map": {"bonafide": 0, "spoof": 1},
                  "split": {"reuse_existing": True, "seed": 13,
                            "assignments": str(assignments), "ratios": {"fit": 1.0}},
                  "output": str(output)})
    with (output / "assignments.csv").open(newline="") as stream:
        rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    assert rows["a"]["split_role"] == "source_val" and rows["a"]["sample_index"] == "7"
    assert rows["b"]["split_role"] == "fit" and rows["b"]["sample_index"] == "8"

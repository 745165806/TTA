"""Order control, method isolation and label-free score worker."""
from pathlib import Path

from experiments.large_scale_confirmation.run_scores import ARMS, ORDERS, ordered_ids


def test_six_orders_replay_and_exact_coverage():
    ids = ["id-%03d" % value for value in range(101)]
    assert ORDERS == ("manifest_order", "seed2026", "seed2027", "seed2028",
                      "seed2029", "seed2030")
    assert ordered_ids(ids, "manifest_order") == ids
    orders = [ordered_ids(ids, key) for key in ORDERS]
    assert all(set(order) == set(ids) and len(order) == len(ids) for order in orders)
    assert all(ordered_ids(ids, key) == order for key, order in zip(ORDERS, orders))
    assert len({tuple(order) for order in orders}) == 6


def test_only_preregistered_arms_and_no_label_reader_in_worker():
    assert ARMS == ("Frozen", "Per-sample Base", "Local-Base B32", "Local-O1 B32")
    source = (Path(__file__).parents[2] /
              "experiments/large_scale_confirmation/run_scores.py").read_text(encoding="utf-8")
    assert "selected_itw_labels" not in source
    assert "selected_codecfake_labels" not in source
    assert "label_source_ref" not in source
    assert "CODEC_PROTOCOL" not in source
    assert "target_labels_read\": False" in source

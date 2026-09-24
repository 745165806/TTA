"""Pure Python protocol/aggregation tests; no torch, audio or checkpoint imports."""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/protocol_tta"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "src"))
import protocol

spec = importlib.util.spec_from_file_location("protocol_aggregate_test", EXP / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def synthetic_rows(name, count):
    rows = []
    for i in range(count):
        reset, episode, since = protocol.reset_info(name, i)
        before_drift = 0.0 if reset else rows[-1]["parameter_distance_from_source"]
        rows.append({
            "sample_id": "sample%d" % i, "sample_index": i, "protocol": name,
            "reset_applied": reset, "episode_index": episode, "samples_since_reset": since,
            "source_frozen_score": float(i % 2), "score_before_update": float(i % 2),
            "current_pre_adapt_score": float(i % 2), "score_after": float(i % 2) + 0.1,
            "entropy_before": 0.6, "entropy_after": 0.5, "grad_norm": 0.1,
            "parameter_delta_norm": 0.01, "parameter_distance_before": before_drift,
            "parameter_distance_from_source": before_drift + 0.01,
            "runtime": 0.1, "source_reference_runtime": 0.03,
            "numeric_failure": False, "resource_failure": False,
            "bn_running_stats_unchanged": True, "adaptation_applied": True,
        })
    return rows


class ProtocolTests(unittest.TestCase):
    def test_schedules_and_boundaries(self):
        expected = {"episodic": list(range(300)), "continual": [0],
                    "reset32": list(range(0, 300, 32)), "reset128": [0, 128, 256]}
        for name, indices in expected.items():
            self.assertEqual([i for i in range(300) if protocol.reset_info(name, i)[0]], indices)
        self.assertEqual(protocol.reset_info("reset32", 31), (False, 0, 32))
        self.assertEqual(protocol.reset_info("reset32", 32), (True, 1, 1))
        self.assertEqual(protocol.reset_info("reset128", 127), (False, 0, 128))
        self.assertEqual(protocol.reset_info("reset128", 128), (True, 1, 1))

    def test_invalid_protocol(self):
        for name, index in [("unknown", 0), ("continual", -1), ("episodic", True)]:
            with self.assertRaises(ValueError):
                protocol.reset_info(name, index)

    def test_config_fixed(self):
        self.assertEqual(protocol.load_config(), protocol.FIXED)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            changed = dict(protocol.FIXED, lr=0.1)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol.load_config(path)

    def validate_rows(self, rows, name="continual", expected_count=32):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            return protocol.read_scores(path, name, ["sample%d" % i for i in range(expected_count)])

    def test_valid_sequences_beyond_windows(self):
        for name in protocol.PROTOCOLS:
            self.assertEqual(len(self.validate_rows(synthetic_rows(name, 260), name, 260)), 260)

    def test_duplicate_missing_and_reordered_rejected(self):
        rows = synthetic_rows("continual", 32)
        for invalid in (rows + rows[:1], rows[:-1], list(reversed(rows))):
            with self.assertRaises(ValueError):
                self.validate_rows(invalid)

    def test_nonfinite_diagnostics_rejected(self):
        for key in protocol.NUMERIC_FIELDS:
            for value in (math.nan, math.inf, None):
                rows = synthetic_rows("continual", 32)
                rows[5][key] = value
                with self.assertRaises(ValueError):
                    self.validate_rows(rows)

    def test_implicit_reset_and_false_schedule_rejected(self):
        for key, value in (("reset_applied", True), ("parameter_distance_before", 0.0),
                           ("bn_running_stats_unchanged", False), ("numeric_failure", True),
                           ("resource_failure", True), ("adaptation_applied", False)):
            rows = synthetic_rows("continual", 32)
            rows[5][key] = value
            with self.assertRaises(ValueError):
                self.validate_rows(rows)

    def test_smoke_checks_reference_config_and_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            ids = ["sample%d" % i for i in range(32)]
            manifest.write_text(json.dumps({"records": [
                {"schema_version": "0.1.0", "sample_id": sid, "root_key": "in_the_wild",
                 "audio_relpath": sid, "split_role": "select", "sample_index": i}
                for i, sid in enumerate(ids)]}), encoding="utf-8")
            for name in protocol.PROTOCOLS:
                out = root / name
                out.mkdir()
                config = {"parameters": protocol.FIXED, "protocol": name,
                          "sample_order_policy": "manifest_order", "sample_count": 32,
                          "bundle_path": "same", "checkpoint_ref": "same", "detector_state_ref": "same",
                          "baseline_id": "same", "source_run_id": "same",
                          "updated_parameter_names": ["bn.weight"], "tau0": 0.5,
                          "manifest_path": str(manifest), "asset_root": "same"}
                (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
                (out / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in synthetic_rows(name, 32)), encoding="utf-8")
            with patch.object(protocol, "MANIFEST", manifest):
                records, _ = protocol.validate_run(root, ids)
                self.assertEqual(set(records), set(protocol.PROTOCOLS))
                label_path = root / "experiments/target10_selection/manifests/inwild_target10.json"
                label_path.parent.mkdir(parents=True)
                label_path.write_text(json.dumps({"records": [
                    {"sample_id": sid, "label": i % 2} for i, sid in enumerate(ids)]}), encoding="utf-8")
                with patch.object(aggregate, "ROOT", root), patch.object(sys, "argv", [
                        "aggregate.py", "--run-dir", str(root)]), patch("builtins.print"):
                    aggregate.main()
                    with self.assertRaises(FileExistsError):
                        aggregate.main()
                summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(len(summary["rows"]), 5)
                self.assertEqual(summary["rows"][0]["protocol"], "Frozen-Waveform")
                self.assertTrue((root / "metrics.csv").is_file())
                self.assertTrue((root / "report.md").is_file())
                score_path = root / "continual/scores.jsonl"
                saved = score_path.read_text(encoding="utf-8")
                bad_rows = synthetic_rows("continual", 32)
                bad_rows[1]["source_frozen_score"] += 0.1
                score_path.write_text("\n".join(json.dumps(r) for r in bad_rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol.validate_run(root, ids)
                for row in bad_rows:
                    row["source_frozen_score"] = float(row["sample_index"] % 2)
                    row["parameter_delta_norm"] = 0.0
                    row["adaptation_applied"] = False
                score_path.write_text("\n".join(json.dumps(r) for r in bad_rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol.validate_run(root, ids)
                score_path.write_text(saved, encoding="utf-8")
                path = root / "continual/run_config.json"
                bad = json.loads(path.read_text())
                bad["checkpoint_ref"] = "different"
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    protocol.validate_run(root, ids)

    def test_aggregation_uses_common_source_not_current_before(self):
        rows = synthetic_rows("continual", 4)
        for r in rows:
            r["score_before_update"] = r["current_pre_adapt_score"] = 0.9
            r["score_after"] = 0.8
        labels = [0, 1, 0, 1]
        reference = [0.1, 0.9, 0.1, 0.9]
        result = aggregate.summarize("continual", rows, labels, 0.5, reference)
        self.assertEqual(result["harmful_flips"], 2)
        self.assertEqual(result["helpful_flips"], 0)
        self.assertAlmostEqual(result["mean_abs_instant_update"], 0.1)
        self.assertAlmostEqual(result["mean_abs_drift_before"], 0.4)
        self.assertAlmostEqual(result["mean_abs_total_delta"], 0.4)
        self.assertAlmostEqual(result["bonafide_mean_score_delta"], 0.7)

    def test_spearman_ties_and_constants(self):
        self.assertEqual(aggregate.spearman([1, 1, 3], [4, 4, 8]), 1.0)
        self.assertEqual(aggregate.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(aggregate.spearman([1, 1], [2, 3]))

    def test_label_coverage_duplicate_and_invalid_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.json"
            for rows in ([{"sample_id": "a", "label": 0}] * 2,
                         [{"sample_id": "a", "label": "0"}],
                         [{"sample_id": "b", "label": 0}]):
                path.write_text(json.dumps({"records": rows}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    aggregate.read_labels(path, ["a"])


if __name__ == "__main__":
    unittest.main()

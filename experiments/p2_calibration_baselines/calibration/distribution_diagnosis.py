#!/usr/bin/env python
"""P2-A: source vs target frozen-score distribution decomposition (CPU).

POST_HOC_DEVELOPMENT_ONLY: reads target10 labels only for diagnosis.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from calibration.common import POST_HOC_MARKER, load_cal0, load_target10, quantiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.run_dir) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    sc, yc, _ic, _res = load_cal0()
    st, yt, _it, _res2 = load_target10()

    groups = {
        "source_bonafide": [s for s, y in zip(sc, yc) if y == 0],
        "source_spoof": [s for s, y in zip(sc, yc) if y == 1],
        "target_bonafide": [s for s, y in zip(st, yt) if y == 0],
        "target_spoof": [s for s, y in zip(st, yt) if y == 1],
    }

    rows = []
    for name, values in groups.items():
        q = quantiles(values)
        rows.append({"group": name, **{k: q[k] for k in q}})

    with (out_dir / "score_distribution.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["group", "count", "mean", "std", "median",
                      "q01", "q05", "q25", "q50", "q75", "q95", "q99"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sb, ss = groups["source_bonafide"], groups["source_spoof"]
    tb, ts = groups["target_bonafide"], groups["target_spoof"]
    import statistics
    shift = {
        "POST_HOC_DEVELOPMENT_ONLY": True,
        "bonafide_mean_shift": statistics.fmean(tb) - statistics.fmean(sb),
        "spoof_mean_shift": statistics.fmean(ts) - statistics.fmean(ss),
        "bonafide_median_shift": statistics.median(tb) - statistics.median(sb),
        "spoof_median_shift": statistics.median(ts) - statistics.median(ss),
        "midpoint_shift": ((statistics.fmean(ts) + statistics.fmean(tb)) / 2
                           - (statistics.fmean(ss) + statistics.fmean(sb)) / 2),
        "class_separation_source": statistics.fmean(ss) - statistics.fmean(sb),
        "class_separation_target": statistics.fmean(ts) - statistics.fmean(tb),
    }
    (out_dir / "class_shift_summary.json").write_text(
        json.dumps(shift, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(shift, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

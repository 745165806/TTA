"""Two-direction supervised linear-probe transfer between development domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments.capacity_audit.run_ladder import (
    HERE, fit_arm, load_itw, load_wavefake, metric, write_new,
)


def run(args):
    if not str(sys.executable).endswith("/envs/tta/bin/python"):
        raise EnvironmentError("tta conda environment required")
    out = HERE / "results" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "predictions", "analysis"):
        (out / part).mkdir()
    try:
        itw_ids, itw_x, itw_y, itw_resources, itw_provenance, _ = load_itw()
        wave_ids, wave_x, wave_y, wave_resources, wave_provenance, wave_groups = load_wavefake(
            args.cache, args.assignment, args.labels)
        write_new(out / "run_config.json", {"role": "SUPERVISED_CROSS_DOMAIN_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL",
            "command": sys.argv, "model": "C2_linear_fixed_training_schedule",
            "itw_count": len(itw_ids), "wavefake_count": len(wave_ids),
            "target90_accessed": False, "final_holdout_accessed": False})
        write_new(out / "provenance.json", {"itw": itw_provenance, "wavefake": wave_provenance})
        rows = []
        for source, destination in (("itw", "wavefake"), ("wavefake", "itw")):
            if source == "itw":
                train_x, train_y, train_resources = itw_x, itw_y, itw_resources
                test_x, test_y, test_resources = wave_x, wave_y, wave_resources
            else:
                train_x, train_y, train_resources = wave_x, wave_y, wave_resources
                test_x, test_y, test_resources = itw_x, itw_y, itw_resources
            both_x = np.concatenate((train_x, test_x))
            both_y = np.concatenate((train_y, test_y))
            train = np.arange(len(train_x))
            heldout = np.arange(len(train_x), len(both_x))
            train_groups = None if source == "itw" else wave_groups
            all_groups = None if train_groups is None else train_groups + [f"itw:{i}" for i in range(len(test_x))]
            prediction, detail = fit_arm("C2_linear", both_x, both_y, train, heldout,
                                         train_resources, 0, all_groups)
            frozen = (test_x @ test_resources.w.numpy() + test_resources.b).astype(np.float64)
            frozen_metric = metric(frozen, test_y)
            transfer_metric = metric(prediction, test_y)
            rows.append({"train_domain": source, "test_domain": destination,
                "train_count": len(train_x), "test_count": len(test_x),
                "frozen_auc": frozen_metric["auroc"], "frozen_eer": frozen_metric["eer"],
                "probe_auc": transfer_metric["auroc"], "probe_eer": transfer_metric["eer"],
                "delta_auc": transfer_metric["auroc"]-frozen_metric["auroc"],
                "eer_improvement": frozen_metric["eer"]-transfer_metric["eer"],
                "training_detail": detail})
            print(source, "->", destination, rows[-1], flush=True)
        write_new(out / "analysis/summary.json", {"status": "PASS", "role":
            "SUPERVISED_CROSS_DOMAIN_DEVELOPMENT_DIAGNOSTIC_NOT_FINAL", "rows": rows,
            "target90_accessed": False, "final_holdout_accessed": False})
        (out / "analysis/report.md").write_text("# Cross-domain supervised development probe\n\n"
            "C2 linear probe trained on one development resource and tested on the other. "
            "This is not final evaluation.\n\n"
            + "| Train → Test | Frozen AUC/EER | Probe AUC/EER | ΔAUC | EER improvement |\n"
            + "|---|---:|---:|---:|---:|\n"
            + "\n".join(f"| {r['train_domain']} → {r['test_domain']} | "
               f"{r['frozen_auc']:.6f}/{r['frozen_eer']:.6f} | "
               f"{r['probe_auc']:.6f}/{r['probe_eer']:.6f} | "
               f"{r['delta_auc']:+.6f} | {r['eer_improvement']:+.6f} |" for r in rows)+"\n")
    except BaseException:
        write_new(out / "failure.json", {"status": "FAIL", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    run(parser.parse_args())

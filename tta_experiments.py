#!/usr/bin/env python3
"""Generate ordinary EP-TTA experiment configs; never train, score, or read test labels.

Run from the TTA repository root in the existing `tta` environment.
Compatible with the repository interfaces inspected on 2026-09-20.
New configuration directories are exclusive; existing results are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml


def read(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a mapping: {path}")
    return obj


def write(path: Path, obj: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Unsafe directory/family name: {value!r}")
    return value


def source_config(path: str, paths: str | None = None) -> dict[str, Any]:
    # Use the project's own path expansion, not a second configuration language.
    from eptta.research import load_config
    return load_config(path, "prepare-source", paths)


def source_base(cfg: dict[str, Any]) -> Path:
    return Path(cfg["resources_output"]).parent


def bundle_ref(cfg: dict[str, Any]) -> str:
    if cfg.get("frozen_bundle"):
        return str(cfg["frozen_bundle"])
    return str(Path(cfg["frozen_output"]) / "bundle.json")


def selected_families(selection_path: str | Path) -> dict[str, Any]:
    doc = read(selection_path)
    if doc.get("selected_on_role") != "select" or not doc.get("families"):
        raise ValueError("Expected a source-only selection with per-family metrics references")
    for family, row in doc["families"].items():
        safe_name(family)
        if row["method"] not in doc.get("methods", []):
            raise ValueError(f"Family method absent from selection.methods: {family}")
        run_dir = Path(row["metrics_ref"]).parent
        run_cfg, run = read(run_dir / "config.yaml"), read(run_dir / "run.json")
        if run_cfg.get("role") != "select" or run_cfg.get("method") != row["method"]:
            raise ValueError(f"Not the declared source-select candidate: {run_dir}")
        check_run_metadata(run, run_dir)
    return doc["families"]


def check_run_metadata(run: dict[str, Any], path: Path) -> None:
    if (run.get("status") != "SCORED" or
            run.get("coverage_fraction") != 1.0 or
            run.get("valid_for_comparison") is not True or
            run.get("target_labels_read") is not False):
        raise ValueError(f"Incomplete or invalid run; do not use for comparison: {path}")
    rate, limit = run.get("fallback_rate"), run.get("fallback_rate_max")
    if not isinstance(rate, (int, float)) or not isinstance(limit, (int, float)):
        raise ValueError(f"Missing fallback diagnostics: {path}")
    if not math.isfinite(rate) or not 0 <= rate <= limit <= 1:
        raise ValueError(f"Fallback rate exceeds the declared bound: {path}")


def paper_selection(args: argparse.Namespace) -> None:
    cfg = source_config(args.source_config, args.paths)
    base = source_base(cfg)
    initial = Path(args.selection) if args.selection else base / "selection.json"
    families = selected_families(initial)
    ep = copy.deepcopy(families["ep_tta"]["method"])
    original_plan = read(cfg["selection"])
    grid = [copy.deepcopy(c["method"]["config"])
            for c in original_plan["candidates"] if c["family"] == "ep_tta"]
    if not grid:
        raise ValueError("Original source selection has no EP hyperparameter grid")
    rank = int(read(cfg["artifact_plan"])["rank"])
    if rank < 1:
        raise ValueError("Source subspace rank must be positive")
    candidates: list[dict[str, Any]] = []
    # Reuse the already-selected reference runs without rescoring them.
    for family in ("frozen", "multiview_mean", "ep_tta"):
        row = families[family]
        candidates.append({"family": family, "method": copy.deepcopy(row["method"]),
                           "rank_order": [0], "run": str(Path(row["metrics_ref"]).parent)})

    def add(family: str, method_id: str, config: dict[str, Any],
            params: dict[str, Any] | None = None, order: list[Any] | None = None) -> None:
        candidates.append({"family": family,
                           "method": {"method_id": method_id, "config": copy.deepcopy(config),
                                      "params": copy.deepcopy(params or {})},
                           "rank_order": order or [0]})

    zero = copy.deepcopy(ep["config"])
    zero["steps"] = 0
    add("ep_k0", "ep_tta", zero)
    add("fixed_source_adapter", "fixed_source_adapter", ep["config"])
    # Controlled ablations: keep the source-selected EP hyperparameters fixed.
    for name in ("ep_no_keep", "ep_feature_pca_U", "ep_no_projection",
                 "ep_keep_l2", "ep_keep_logit", "ep_keep_fisher",
                 "ep_diagonal_R", "ep_scalar_adaptive", "source_ce_only"):
        add(name, name, ep["config"])
    # Do not select the most favorable random seed; retain all three controls.
    for index in range(3):
        add(f"ep_random_U_{index}", "ep_random_U", ep["config"], {"random_index": index})
    # Equal search budget to EP for alternative adaptation objectives.
    for name in ("entropy_same_adapter_no_keep", "entropy_same_adapter",
                 "memo_same_adapter_no_keep", "memo_same_adapter_keep"):
        for params in grid:
            add(name, name, params, order=[params["steps"], params["lr"]])
    # Static shrinkage uses the same Frobenius radius and source-only selection.
    for index in range(17):
        amount = ep["config"]["rho"] / math.sqrt(rank) * index / 16
        add("static_subspace", "static_subspace", ep["config"],
            {"amount": amount}, [index])

    out = Path(args.out_dir)
    if out.exists() or (base / "selection-paper.json").exists():
        raise ValueError("Paper configs/selection already exist; do not overwrite or silently rerun")
    if (base / "select-paper-runs").exists() or (base / "prepare-paper-selection").exists():
        raise ValueError("Paper run outputs already exist; inspect them before creating another plan")
    out.mkdir(parents=True)
    select = copy.deepcopy(original_plan["source_select"])
    select["output_root"] = str(base / "select-paper-runs")
    write(out / "selection-plan.yaml", {"schema_version": "0.1.0",
          "selected_on_role": "select", "source_select": select,
          "candidates": candidates, "output": str(base / "selection-paper.json")})
    write(out / "prepare-selection.yaml", {"schema_version": "0.1.0", "command": "prepare-source",
          "frozen_bundle": bundle_ref(cfg), "resources_output": cfg["resources_output"],
          "selection": str(out / "selection-plan.yaml"),
          "output": str(base / "prepare-paper-selection")})
    print(f"Created {len(candidates)} source-only candidates in {out}")
    print("Controlled ablations use EP's selected parameters; alternative objectives use its original grid.")
    print("Run: python -m eptta.cli prepare-source --config", out / "prepare-selection.yaml")


def make_runs(args: argparse.Namespace) -> None:
    cfg = source_config(args.source_config, args.paths)
    families = selected_families(args.selection)
    base, data = source_base(cfg), Path(args.data_dir)
    summary = read(data / "summary.json")
    manifest = data / "inference" / f"{args.role}.jsonl"
    extraction = copy.deepcopy(cfg["roles"]["fit"])
    dataset_id = args.dataset_id or summary["dataset_id"]
    roots = extraction["data_roots"]
    if args.audio_root or args.root_key:
        if not args.audio_root or not args.root_key:
            raise ValueError("For another dataset, supply both --audio-root and --root-key")
        roots = {args.root_key: args.audio_root}
    if dataset_id != summary["dataset_id"]:
        raise ValueError("--dataset-id differs from prepared data summary")
    required = {"schema_version", "sample_id", "root_key", "audio_relpath", "split_role"}
    ids: set[str] = set()
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not required.issubset(row) or set(row) - required - {"sample_index"}:
                raise ValueError(f"Forbidden/missing fields in inference manifest line {number}")
            if row["split_role"] != args.role or row["root_key"] not in roots:
                raise ValueError(f"Role/root mismatch in inference manifest line {number}")
            if not isinstance(row["sample_id"], str) or not row["sample_id"] or row["sample_id"] in ids:
                raise ValueError(f"Invalid or duplicate ID on line {number}")
            ids.add(row["sample_id"])
    if not ids:
        raise ValueError("Inference manifest is empty")
    safe_name(args.tag)
    safe_name(args.cache_name)
    out, run_root = Path(args.out_dir), base / "experiments" / args.tag
    if out.exists() or run_root.exists():
        raise ValueError("Config/run directory already exists; choose a new tag and config directory")
    out.mkdir(parents=True)
    scope = args.scope or "/".join(str(summary[key]) for key in ("dataset_id", "release", "subset")) + "/" + args.role
    for family, row in families.items():
        doc = {"schema_version": "0.1.0", "command": "run-tta", "run_name": family,
               "output_root": str(run_root), "role": args.role, "scope_id": scope,
               "input_manifest": str(manifest), "feature_cache": str(base / ("cache-" + args.cache_name)),
               "extraction": {"dataset_id": dataset_id, "data_roots": roots,
                              "probe": extraction["probe"], "numerical_mode": extraction["numerical_mode"],
                              "workers": 1},
               "resources": cfg["resources_output"], "frozen_bundle": bundle_ref(cfg),
               "method": copy.deepcopy(row["method"]), "threshold": None, "threshold_source": None,
               "fallback_rate_max": 0.01, "selection_source": args.selection,
               "seed": extraction["probe"]["seed"]}
        write(out / f"{family}.yaml", doc)
    print(f"Created {len(families)} configs for {len(ids)} unlabeled samples in {out}")
    print("Shared feature cache:", base / ("cache-" + args.cache_name))
    print("Result root:", run_root)


def check(args: argparse.Namespace) -> None:
    if args.selection:
        families = selected_families(args.selection)
        for name, row in sorted(families.items()):
            print(name, "source_select_eer=", row.get("eer"), "config=", row["method"])
        print("All selected source candidates have valid run metadata.")
        return
    root = Path(args.runs)
    paths = sorted(root.glob("*/run.json"))
    if not paths:
        raise ValueError(f"No direct child run.json found under {root}")
    scores_by_family: dict[str, dict[str, float]] = {}
    import csv
    for path in paths:
        run = read(path)
        check_run_metadata(run, path.parent)
        print(path.parent.name, "count=", run["sample_count"], "fallback_rate=", run["fallback_rate"])
        score_path = path.parent / "scores.csv"
        if score_path.exists() and path.parent.name in ("frozen", "ep_k0"):
            with score_path.open(encoding="utf-8") as stream:
                scores_by_family[path.parent.name] = {r["sample_id"]: float(r["score"])
                                                     for r in csv.DictReader(stream)}
    if set(scores_by_family) == {"frozen", "ep_k0"}:
        a, b = scores_by_family["frozen"], scores_by_family["ep_k0"]
        if a.keys() != b.keys():
            raise ValueError("Frozen and EP K=0 have different IDs")
        delta = max(abs(a[key] - b[key]) for key in a)
        print("Frozen vs EP K=0 max absolute score difference:", delta)
        if delta != 0.0:
            raise ValueError("K=0 does not exactly reproduce frozen on this identical cache")
    print("Run metadata checks passed; no target labels were read by this helper.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    paper = subs.add_parser("paper-selection", help="Generate source-only baseline/ablation selection configs")
    runs = subs.add_parser("runs", help="Generate runs from exactly recorded source-selected methods")
    for command in (paper, runs):
        command.add_argument("--source-config", required=True)
        command.add_argument("--paths")
        command.add_argument("--out-dir", required=True)
    paper.add_argument("--selection", help="Initial selection.json; defaults to the source output directory")
    runs.add_argument("--selection", required=True)
    runs.add_argument("--data-dir", required=True)
    runs.add_argument("--role", choices=("audit", "cal1", "control_test", "target_test"), required=True)
    runs.add_argument("--tag", required=True)
    runs.add_argument("--cache-name", required=True)
    runs.add_argument("--dataset-id")
    runs.add_argument("--audio-root")
    runs.add_argument("--root-key")
    runs.add_argument("--scope")
    check_parser = subs.add_parser("check", help="Check selected/source or scored-run validity; never read labels")
    group = check_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection")
    group.add_argument("--runs")
    args = parser.parse_args()
    try:
        {"paper-selection": paper_selection, "runs": make_runs, "check": check}[args.command](args)
        return 0
    except (OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

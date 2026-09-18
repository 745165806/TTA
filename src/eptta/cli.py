"""Six-command EP-TTA research CLI."""
import argparse
import json
import sys

from eptta import __version__
from eptta.errors import EPTTAError
from eptta.research import (COMMANDS, evaluate, load_config, prepare_data, prepare_source,
                            report, run_tta, train_source)


def parser():
    root = argparse.ArgumentParser(
        description="EP-TTA: prepare data, train/reuse source models, run unlabeled TTA, and evaluate")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare-data", "parse an explicit manifest, preserve/generate group-safe splits, and write role views"),
        ("train-source", "train or exactly resume a source detector using fit/source_val only"),
        ("prepare-source", "validate/reuse a frozen detector and build only requested source caches/resources"),
        ("run-tta", "score one label-free role with one source-selected method"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="experiment YAML/JSON")
        command.add_argument("--paths", help="optional local paths YAML/JSON (not committed)")
        if name == "train-source":
            command.add_argument("--resume", help="complete checkpoint to resume in the configured run directory")
    evaluate_parser = sub.add_parser(
        "evaluate", help="align complete scores with labels and compute metrics independently")
    evaluate_parser.add_argument("--run", required=True)
    evaluate_parser.add_argument("--labels", required=True)
    report_parser = sub.add_parser("report", help="summarize existing metrics without rescoring or selection")
    report_parser.add_argument("--runs", required=True)
    report_parser.add_argument("--out", required=True)
    return root


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate(args.run, args.labels)
        elif args.command == "report":
            result = report(args.runs, args.out)
        else:
            config = load_config(args.config, args.command, args.paths)
            if args.command == "train-source" and args.resume:
                config["resume"] = args.resume
            function = {"prepare-data": prepare_data, "train-source": train_source,
                        "prepare-source": prepare_source, "run-tta": run_tta}[args.command]
            result = function(config)
        emit(result)
        return 0
    except (EPTTAError, OSError, ValueError) as exc:
        emit({"schema_version": "0.1.0", "status": "ERROR", "code": type(exc).__name__,
              "message": str(exc), "command": args.command})
        return 2


if __name__ == "__main__":
    sys.exit(main())

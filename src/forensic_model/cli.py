"""Thin command-line adapters over the reusable library API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from forensic_model.experiment import run_smoke_evaluation
from forensic_model.manifest import load_manifest


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forensic-model")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-manifest", help="validate data licensing and split integrity")
    validate.add_argument("manifest", type=Path)

    smoke = commands.add_parser("smoke-evaluate", help="run the procedural end-to-end evaluation")
    smoke.add_argument("--output", type=Path)

    options = parser.parse_args(arguments)
    if options.command == "validate-manifest":
        samples = load_manifest(options.manifest)
        print(json.dumps({"samples": len(samples), "status": "valid"}, sort_keys=True))
        return 0

    report = run_smoke_evaluation()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

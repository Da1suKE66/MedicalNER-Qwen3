#!/usr/bin/env python3
"""Audit final train-ready Schema v2 records and their frozen split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    sha256_file,
    write_json,
)
from kg_lora.training_data_audit import audit_training_data  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "data/schema_v2/cleaned/pro_cot_schema_v2_train_ready.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/schema_v2/splits/split_manifest.json",
    )
    parser.add_argument(
        "--expected-split",
        default="all",
        help="Manifest split to match; use 'all' for the complete train-ready corpus.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/schema_v2_training_data_audit.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any gate fails; the JSON report is still written.",
    )
    return parser.parse_args(argv)


def _path_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _fatal_report(
    args: argparse.Namespace, exc: Exception
) -> dict[str, Any]:
    message = str(exc)
    failed_gate = (
        "strict_json"
        if isinstance(exc, json.JSONDecodeError)
        or "non-finite JSON number" in message
        else "input_load"
    )
    return {
        "passed": False,
        "failed_gates": [failed_gate],
        "gates": {
            failed_gate: {
                "passed": False,
                "finding_count": 1,
                "findings": [
                    {
                        "code": "audit_input_load_failed",
                        "error_type": type(exc).__name__,
                        "error": message,
                    }
                ],
            }
        },
        "input": _path_metadata(args.input),
        "split_manifest": _path_metadata(args.split_manifest),
        "expected_split": args.expected_split,
        "strict": bool(args.strict),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report.resolve() in {
        args.input.resolve(),
        args.split_manifest.resolve(),
    }:
        print("refusing to overwrite audit input with --report", file=sys.stderr)
        return 2

    try:
        records = record_list(load_json(args.input))
        split_manifest = load_json(args.split_manifest)
        if not isinstance(split_manifest, dict):
            raise TypeError("split manifest must be a JSON object")
        report = audit_training_data(
            records,
            split_manifest,
            load_schema(),
            expected_split=args.expected_split,
        )
        report.update(
            {
                "input": _path_metadata(args.input),
                "split_manifest": _path_metadata(args.split_manifest),
                "strict": bool(args.strict),
            }
        )
    except (OSError, TypeError, ValueError) as exc:
        report = _fatal_report(args, exc)

    write_json(args.report, report)
    failed = report.get("failed_gates", [])
    print(
        f"passed={report['passed']} records={report.get('record_count', 0)} "
        f"failed_gates={failed}"
    )
    print(f"wrote {args.report}")
    if args.strict and not report["passed"]:
        print(
            "strict training-data audit failed: " + ", ".join(map(str, failed)),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

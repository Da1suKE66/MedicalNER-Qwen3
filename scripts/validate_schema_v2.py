#!/usr/bin/env python3
"""Validate migrated KG records against the current Schema v2 draft."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    validate_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/schema_v2/migrated/pro_cot_schema_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/schema_v2_validation_report.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the report contains errors or review warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(record_list(load_json(args.input)), load_schema())
    write_json(args.output, report)
    print(
        f"records={report['record_count']} findings={report['records_with_findings']} "
        f"errors={report['error_counts']} warnings={report['warning_counts']}"
    )
    print(f"wrote {args.output}")
    if args.strict and (report["error_counts"] or report["warning_counts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Remove ungrounded/review-only graph items with a complete training audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import load_json, load_schema, record_list, write_json  # noqa: E402
from kg_lora.training_sanitizer import sanitize_dataset_for_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_icd_validated.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_train_ready.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/schema_v2_training_sanitization_report.json",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned, report = sanitize_dataset_for_training(
        record_list(load_json(args.input)), load_schema()
    )
    report.update({"input": str(args.input), "output": str(args.output)})
    write_json(args.output, cleaned)
    write_json(args.report, report)
    print(
        f"records={report['record_count']} statuses={report['status_counts']} "
        f"not_repaired={report['records_not_repaired']}"
    )
    print(f"totals={report['totals']}")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    if args.strict and report["records_not_repaired"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

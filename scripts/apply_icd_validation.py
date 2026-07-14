#!/usr/bin/env python3
"""Apply a complete WHO ICD API report to the evidence-cleaned Schema v2 data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.icd_validation import apply_icd_validation  # noqa: E402
from kg_lora.schema_v2 import load_json, load_schema, record_list, write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_evidence.json",
    )
    parser.add_argument(
        "--who-report",
        type=Path,
        default=ROOT / "reports/icd_api_validation_full_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_icd_validated.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/icd_validation_application_report.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Write the audit but do not write the WHO-validated intermediate "
            "unless every canonical code is verified and no errors or unresolved "
            "codes remain. Schema warnings are retained for the next sanitizer."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cleaned, audit = apply_icd_validation(
        record_list(load_json(args.input)),
        load_json(args.who_report),
        load_schema(),
    )
    audit.update(
        {
            "input": str(args.input),
            "who_report": str(args.who_report),
            "output": str(args.output),
            "output_written": False,
        }
    )
    strict_failure = args.strict and not audit["strict_ready"]
    if not strict_failure:
        write_json(args.output, cleaned)
        audit["output_written"] = True
    write_json(args.report, audit)
    print(
        f"records={audit['record_count']} statuses={audit['status_counts']} "
        f"icd_findings={audit['records_with_icd_findings']}"
    )
    print(f"totals={audit['totals']}")
    if audit["output_written"]:
        print(f"wrote {args.output}")
    else:
        print(
            "strict WHO validation failed; training input was not written",
            file=sys.stderr,
        )
    print(f"wrote {args.report}")
    return 1 if strict_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

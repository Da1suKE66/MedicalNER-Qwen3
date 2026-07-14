#!/usr/bin/env python3
"""Deterministically migrate generated KG records to the Schema v2 draft."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    MIGRATION_IMPLEMENTATION_VERSION,
    build_raw_indexes,
    load_json,
    load_schema,
    migrate_record,
    migration_config_fingerprint,
    record_list,
    validate_dataset,
    write_json,
)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/generated/gemini_split/pro_cot_001_858_complete_schema0413.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw/mental_disorders_20251125_165535.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/schema_v2/migrated/pro_cot_schema_v2.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/schema_v2_migration_report.json",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=ROOT / "data/schema_v2/manual_review.jsonl",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply-high-confidence-collapses", action="store_true")
    parser.add_argument(
        "--force-renormalize",
        action="store_true",
        help="rerun normalization even when implementation/config markers match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema()
    raw_indexes = build_raw_indexes(load_json(args.raw))
    records = record_list(load_json(args.input))
    if args.limit is not None:
        records = records[: args.limit]

    migrated_records = []
    migration_results = []
    statuses: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for record in records:
        migrated, result = migrate_record(
            record,
            raw_indexes,
            schema,
            apply_high_confidence_collapses=args.apply_high_confidence_collapses,
            force_renormalize=args.force_renormalize,
        )
        migrated_records.append(migrated)
        migration_results.append(result)
        statuses[result.get("status", "unknown")] += 1
        for change in result.get("changes", []):
            operation_counts[change.get("op", "unknown")] += 1

    second_pass = [
        migrate_record(
            record,
            raw_indexes,
            schema,
            apply_high_confidence_collapses=args.apply_high_confidence_collapses,
        )[0]
        for record in migrated_records
    ]
    idempotent = second_pass == migrated_records
    if not idempotent:
        raise AssertionError("Schema v2 migration is not idempotent")

    validation = validate_dataset(migrated_records, schema)
    report = {
        "schema_version": schema["schema_version"],
        "migration_implementation_version": MIGRATION_IMPLEMENTATION_VERSION,
        "migration_config_fingerprint": migration_config_fingerprint(
            schema,
            apply_high_confidence_collapses=args.apply_high_confidence_collapses,
        ),
        "force_renormalize": args.force_renormalize,
        "input": display_path(args.input),
        "input_record_count": len(records),
        "output_record_count": len(migrated_records),
        "no_silent_record_loss": len(records) == len(migrated_records),
        "idempotent": idempotent,
        "status_counts": dict(statuses),
        "operation_counts": dict(operation_counts),
        "validation_summary": {
            key: validation[key]
            for key in (
                "record_count",
                "records_with_findings",
                "error_counts",
                "warning_counts",
                "metrics",
            )
        },
    }
    write_json(args.output, migrated_records)
    write_json(args.report, report)

    review_rows = []
    for index, (record, result) in enumerate(zip(migrated_records, migration_results)):
        if result.get("status") not in {"manual_review", "invalid"}:
            continue
        review_rows.append(
            {
                "record_index": index,
                "source_record_id": record.get("source_record_id") or record.get("source_id"),
                "source_code": record.get("source_code") or record.get("code"),
                "source_title": record.get("source_title") or record.get("title"),
                "status": result.get("status"),
                "warnings": result.get("warnings", []),
                "errors": result.get("errors", []),
                "unverified_codes": result.get("unverified_codes", []),
            }
        )
    args.manual_review.parent.mkdir(parents=True, exist_ok=True)
    args.manual_review.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    print(
        f"migrated={len(migrated_records)} statuses={dict(statuses)} "
        f"idempotent={idempotent} review={len(review_rows)}"
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(f"wrote {args.manual_review}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restore exact relation evidence spans and emit an auditable cleaned dataset."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.evidence_repair import repair_record_evidence  # noqa: E402
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
        default=ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_evidence.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/evidence_span_repair_report.json",
    )
    parser.add_argument(
        "--drop-unresolved",
        action="store_true",
        help="Remove relations that cannot be grounded; every removal remains in the audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = record_list(load_json(args.input))
    output = []
    totals: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    findings = []
    for index, record in enumerate(records):
        repaired, audit = repair_record_evidence(
            record, drop_unresolved=args.drop_unresolved
        )
        output.append(repaired)
        for key in (
            "relation_count",
            "already_verified",
            "repaired",
            "unresolved",
            "dropped",
        ):
            totals[key] += int(audit[key])
        methods.update(audit["methods"])
        if audit["unresolved_relations"]:
            findings.append(
                {
                    "record_index": index,
                    "source_record_id": record.get("source_record_id"),
                    "source_code": record.get("source_code"),
                    "unresolved_relations": audit["unresolved_relations"],
                }
            )

    validation = validate_dataset(output, load_schema())
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "drop_unresolved": args.drop_unresolved,
        "record_count": len(output),
        "totals": dict(totals),
        "methods": dict(sorted(methods.items())),
        "records_with_unresolved_evidence": len(findings),
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
        "findings": findings,
    }
    write_json(args.output, output)
    write_json(args.report, report)
    print(
        f"records={len(output)} relations={totals['relation_count']} "
        f"already_verified={totals['already_verified']} repaired={totals['repaired']} "
        f"unresolved={totals['unresolved']} dropped={totals['dropped']}"
    )
    print(f"methods={dict(sorted(methods.items()))}")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

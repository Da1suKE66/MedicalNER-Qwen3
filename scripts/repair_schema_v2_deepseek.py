#!/usr/bin/env python3
"""Propose and safely apply constrained DeepSeek repairs to Schema v2 records."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.deepseek_repair import (  # noqa: E402
    DeepSeekAPIError,
    DeepSeekConfig,
    DeepSeekRepairClient,
    PatchValidationError,
    RepairPatch,
    apply_repair_patch,
    build_repair_task,
    cached_patch_path,
    save_safe_api_result,
    validate_patch_context,
)
from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    validate_dataset,
    validate_record,
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
        default=ROOT / "data/schema_v2/repaired/pro_cot_schema_v2_deepseek.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/deepseek_repair_report.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache/deepseek-repair",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["invalid"],
        choices=["invalid", "manual_review", "repaired"],
    )
    parser.add_argument("--source-record-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    parser.add_argument("--max-api-attempts", type=int, default=3)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        help="Override thinking mode for this run without changing .env.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply locally valid patches. Without this flag only proposals are audited.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file (never the input file).",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _load_cached(
    path: Path, expected_model: str
) -> tuple[RepairPatch, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("patch"), dict):
        raise PatchValidationError("cached API result has an invalid shape")
    if payload.get("model") != expected_model:
        raise PatchValidationError("cached API result model mismatch")
    return RepairPatch.model_validate(payload["patch"]), payload


def _safe_error(record_id: str, exc: Exception) -> str:
    return f"record={record_id} {type(exc).__name__}: {exc}"


def _load_env_file(path: Path) -> None:
    """Load the simple KEY=VALUE form used by this project without logging values."""

    if not path.exists():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise SystemExit(f"invalid env assignment at {path}:{line_number}")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(f"invalid env key at {path}:{line_number}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_confidence <= 1.0:
        raise SystemExit("--minimum-confidence must be in [0,1]")
    if args.max_api_attempts < 1:
        raise SystemExit("--max-api-attempts must be positive")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.report.resolve() == args.input.resolve():
        raise SystemExit("refusing to overwrite input with --report")
    if args.apply:
        if args.output.resolve() == args.input.resolve():
            raise SystemExit("refusing to overwrite input; choose a separate --output")
        if args.output.resolve() == args.report.resolve():
            raise SystemExit("--output and --report must be different files")
        if args.output.exists() and not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite existing output without --overwrite: {args.output}"
            )

    _load_env_file(args.env_file)
    requested_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    if not requested_model:
        raise SystemExit("DEEPSEEK_MODEL must not be empty")
    requested_thinking = (
        args.thinking
        or os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
    )
    if requested_thinking not in {"enabled", "disabled"}:
        raise SystemExit("DEEPSEEK_THINKING must be enabled or disabled")
    schema = load_schema()
    records = record_list(load_json(args.input))
    selected_ids = set(args.source_record_id)
    candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, record in enumerate(records):
        status = str((record.get("migration") or {}).get("status") or "")
        if status not in args.statuses:
            continue
        record_id = str(record.get("source_record_id") or "")
        if selected_ids and record_id not in selected_ids:
            continue
        candidates.append((index, record, validate_record(record, schema)))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    config: DeepSeekConfig | None = None
    client: DeepSeekRepairClient | None = None
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_records = list(records)
    statuses: Counter[str] = Counter()
    results: list[dict[str, Any]] = []

    for position, (record_index, record, findings) in enumerate(candidates, start=1):
        record_id = str(record.get("source_record_id") or "")
        row: dict[str, Any] = {
            "record_index": record_index,
            "source_record_id": record_id,
            "source_code": record.get("source_code"),
            "status_before": (record.get("migration") or {}).get("status"),
        }
        try:
            task = build_repair_task(
                record,
                schema,
                findings,
                request_thinking_mode=requested_thinking,
            )
            cache_path = cached_patch_path(args.cache_dir, task, requested_model)
            if cache_path.exists():
                patch, api_meta = _load_cached(cache_path, requested_model)
                row["cache"] = "hit"
            else:
                if config is None:
                    config = DeepSeekConfig.from_env()
                    config = replace(config, thinking=requested_thinking)
                    if config.model != requested_model:
                        raise DeepSeekAPIError(
                            "DeepSeek model changed while the repair process was running"
                        )
                    client = DeepSeekRepairClient(config)
                assert client is not None
                last_error: Exception | None = None
                for attempt in range(1, args.max_api_attempts + 1):
                    try:
                        patch, api_meta = client.propose_patch(task)
                        break
                    except DeepSeekAPIError as exc:
                        last_error = exc
                        print(
                            f"DeepSeek API attempt {attempt}/{args.max_api_attempts} "
                            + _safe_error(record_id, exc),
                            file=sys.stderr,
                        )
                        if attempt < args.max_api_attempts:
                            time.sleep(min(2 ** (attempt - 1), 8))
                else:
                    assert last_error is not None
                    raise last_error
                save_safe_api_result(
                    cache_path,
                    {
                        **api_meta,
                        "source_record_id": record_id,
                        "record_sha256": task["record_sha256"],
                    },
                )
                row["cache"] = "miss"

            context_errors = validate_patch_context(
                patch,
                record,
                schema,
                minimum_confidence=args.minimum_confidence,
            )
            row.update(
                {
                    "model": api_meta.get("model"),
                    "usage": api_meta.get("usage"),
                    "operation_count": len(patch.operations),
                    "operations": [item.op for item in patch.operations],
                    "needs_human_review": patch.needs_human_review,
                    "review_reason": patch.review_reason,
                    "context_errors": context_errors,
                }
            )
            if context_errors:
                row["result"] = "rejected_context"
            elif patch.needs_human_review:
                row["result"] = "manual_review"
            elif not args.apply:
                row["result"] = "valid_proposal"
            else:
                repaired, audit = apply_repair_patch(
                    record,
                    patch,
                    schema,
                    minimum_confidence=args.minimum_confidence,
                )
                output_records[record_index] = repaired
                row["result"] = "applied"
                row["validation_before_errors"] = len(
                    audit["validation_before"]["errors"]
                )
                row["validation_after_errors"] = len(
                    audit["validation_after"]["errors"]
                )
            statuses[row["result"]] += 1
        except (DeepSeekAPIError, PatchValidationError, ValueError) as exc:
            message = _safe_error(record_id, exc)
            print(message, file=sys.stderr)
            row.update({"result": "error", "error": message})
            statuses["error"] += 1
            if args.fail_fast:
                results.append(row)
                break
        results.append(row)
        print(f"[{position}/{len(candidates)}] record={record_id} result={row['result']}")

    validation = validate_dataset(output_records, schema)
    report = {
        "schema_version": schema["schema_version"],
        "model": requested_model,
        "thinking": requested_thinking,
        "input": str(args.input),
        "candidate_count": len(candidates),
        "apply": args.apply,
        "result_counts": dict(statuses),
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
        "results": results,
    }
    write_json(args.report, report)
    has_failures = bool(statuses["error"] or statuses["rejected_context"])
    if args.apply and not has_failures:
        write_json(args.output, output_records)
    print(f"results={dict(statuses)}")
    print(f"wrote {args.report}")
    if args.apply and not has_failures:
        print(f"wrote {args.output}")
    elif args.apply:
        print("did not write partial output because repair failures occurred", file=sys.stderr)
    if has_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

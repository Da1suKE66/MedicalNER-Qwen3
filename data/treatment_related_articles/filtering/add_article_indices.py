#!/usr/bin/env python3
"""Add stable zero-based ``index`` fields to articles.json records.

The source file stores each record as one JSON object on one physical line. This
script copies the file byte-for-byte except for inserting ``"index": N`` at the
start of each record object. The completed temporary file replaces the source
atomically, so a failed or interrupted run leaves the original file untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO


SCRIPT_DIR = Path(__file__).parent
DEFAULT_ARTICLES_PATH = SCRIPT_DIR.parent / "articles.json"

RECORDS_START_RE = re.compile(rb'^\s*"records"\s*:\s*\[\s*$')
RECORDS_END_RE = re.compile(rb"^\s*\]\s*,?\s*$")


def _json_record(line: bytes, line_number: int) -> dict[str, object]:
    stripped = line.strip()
    if stripped.endswith(b","):
        stripped = stripped[:-1].rstrip()
    try:
        record = json.loads(stripped)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Record line {line_number} is not a complete JSON object"
        ) from exc
    if not isinstance(record, dict):
        raise ValueError(f"Record line {line_number} is not a JSON object")
    return record


def _copy_with_indices(source: BinaryIO, target: BinaryIO) -> tuple[int, int, int]:
    in_records = False
    found_records = False
    finished_records = False
    record_count = 0
    missing_count = 0
    existing_count = 0

    for line_number, line in enumerate(source, start=1):
        logical_line = line.rstrip(b"\r\n")
        if not in_records:
            target.write(line)
            if not found_records and RECORDS_START_RE.fullmatch(logical_line):
                found_records = True
                in_records = True
            continue

        if RECORDS_END_RE.fullmatch(logical_line):
            target.write(line)
            in_records = False
            finished_records = True
            continue

        if not logical_line.strip():
            target.write(line)
            continue

        record = _json_record(line, line_number)
        current_index = record.get("index")
        if "index" in record:
            if type(current_index) is not int or current_index != record_count:
                raise ValueError(
                    f"Record {record_count} has conflicting index {current_index!r}"
                )
            existing_count += 1
            target.write(line)
        else:
            opening_brace = line.find(b"{")
            if opening_brace < 0:
                raise ValueError(f"Record line {line_number} has no opening brace")
            addition = f'"index": {record_count}, '.encode("ascii")
            target.write(line[: opening_brace + 1])
            target.write(addition)
            target.write(line[opening_brace + 1 :])
            missing_count += 1
        record_count += 1

    if not found_records:
        raise ValueError('The input JSON has no top-level "records" array')
    if not finished_records or in_records:
        raise ValueError('The top-level "records" array is not terminated')
    if record_count == 0:
        raise ValueError('The top-level "records" array is empty')
    if existing_count and missing_count:
        raise ValueError(
            "The records array mixes indexed and unindexed records; refusing a "
            "partial rewrite"
        )
    return record_count, missing_count, existing_count


def add_indices(path: Path, *, dry_run: bool = False) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Articles file does not exist: {path}")

    temporary_name: str | None = None
    try:
        with path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as target:
            temporary_name = target.name
            record_count, missing_count, existing_count = _copy_with_indices(
                source, target
            )
            target.flush()
            os.fsync(target.fileno())

        result: dict[str, object] = {
            "articles_file": str(path),
            "records": record_count,
            "indices_added": missing_count,
            "indices_already_present": existing_count,
            "changed": bool(missing_count and not dry_run),
            "dry_run": dry_run,
        }
        if dry_run or missing_count == 0:
            return result

        temporary_path = Path(temporary_name)
        shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
        temporary_name = None

        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return result
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically add zero-based index fields to every record in articles.json"
        )
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=DEFAULT_ARTICLES_PATH,
        help=f"Articles JSON to update (default: {DEFAULT_ARTICLES_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the proposed change without replacing the file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = add_indices(args.articles, dry_run=args.dry_run)
    except Exception as exc:
        raise SystemExit(f"Indexing failed: {type(exc).__name__}: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

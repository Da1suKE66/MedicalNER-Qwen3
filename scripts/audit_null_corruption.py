#!/usr/bin/env python3
"""Reproduce the historical nan-to-null corruption statistics."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    TOKEN_RE,
    load_json,
    load_schema,
    record_list,
    sha256_file,
    write_json,
)


def walk(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")
    else:
        yield path, value


def generalized_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def audit_payload(payload: Any, schema: dict[str, Any]) -> dict[str, Any]:
    corrections = schema["text_corrections"]
    damaged: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    unexpected_null_tokens: Counter[str] = Counter()
    null_paths: Counter[str] = Counter()

    for path, value in walk(payload):
        if value is None:
            null_paths[generalized_path(path)] += 1
            continue
        if not isinstance(value, str):
            continue
        for match in TOKEN_RE.finditer(value):
            token = match.group(0)
            lowered = token.lower()
            if lowered in corrections:
                damaged[lowered] += 1
                if len(examples[lowered]) < 3:
                    examples[lowered].append(
                        {
                            "path": path,
                            "damaged": token,
                            "restored": corrections[lowered],
                        }
                    )
            elif "null" in lowered:
                unexpected_null_tokens[lowered] += 1

    return {
        "schema_version": schema["schema_version"],
        "record_count": len(record_list(payload)),
        "damaged_token_types": len(damaged),
        "damaged_occurrences": sum(damaged.values()),
        "damaged_tokens": {
            token: {
                "count": damaged[token],
                "restored": corrections[token],
                "examples": examples[token],
            }
            for token in sorted(damaged)
        },
        "unexpected_tokens_containing_null": dict(sorted(unexpected_null_tokens.items())),
        "json_null_count": sum(null_paths.values()),
        "null_paths": dict(sorted(null_paths.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/mental_disorders_20251125_165535.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/null_corruption_audit.json",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    payload = load_json(args.input)
    report = audit_payload(payload, load_schema())
    report["input"] = display_path(args.input)
    report["input_sha256"] = sha256_file(args.input)
    write_json(args.output, report)
    print(
        f"records={report['record_count']} damaged_types={report['damaged_token_types']} "
        f"damaged_occurrences={report['damaged_occurrences']} "
        f"json_nulls={report['json_null_count']}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

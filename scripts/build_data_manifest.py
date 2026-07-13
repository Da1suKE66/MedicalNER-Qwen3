#!/usr/bin/env python3
"""Build a reproducible manifest for Schema v2 source artifacts."""

from __future__ import annotations

import argparse
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


DEFAULT_FILES = [
    "data/raw/mental_disorders_20251125_165535.json",
    "data/generated/gemini_split/pro_cot_001_858_complete_schema0413.json",
    "data/llamafactory/pro_cot_001_858_complete_llamafactory.json",
    "data/schema_regression/schema_regression_20.json",
]


def describe(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    result = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        result["record_count"] = len(record_list(payload))
    except TypeError:
        result["record_count"] = len(payload) if isinstance(payload, list) else None
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        result["source_metadata"] = payload["metadata"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/schema_v2/manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema()
    files = [ROOT / item if not Path(item).is_absolute() else Path(item) for item in args.files]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"manifest inputs missing: {missing}")
    manifest = {
        "schema_version": schema["schema_version"],
        "source_release": schema["source_release"],
        "files": [describe(path) for path in files],
    }
    write_json(args.output, manifest)
    print(f"wrote {len(files)} entries to {args.output}")


if __name__ == "__main__":
    main()

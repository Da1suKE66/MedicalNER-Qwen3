"""
Combine COT batch JSON files produced by run_cot_gemini_generation.py.

Each input batch is a top-level list. Sidecar metadata files with suffix
`.metadata.json` or `.json.metadata.json` are optional and summarized when
present.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_metadata_sidecar(input_path: Path) -> Path | None:
    candidates = [
        input_path.with_suffix(".metadata.json"),
        input_path.with_suffix(input_path.suffix + ".metadata.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def metadata_failure_count(sidecar_data: dict[str, Any]) -> int | None:
    failures = sidecar_data.get("failures")
    if failures is None:
        failures = sidecar_data.get("sidecar_failures")
    return int(failures) if failures is not None else None


def collect_input_paths(
    explicit_inputs: list[Path],
    input_dirs: list[Path],
    patterns: list[str],
    excludes: list[str],
    output_paths: set[Path],
) -> list[Path]:
    paths = list(explicit_inputs)
    patterns = patterns or ["pro_cot_*_schema0413.json"]

    for input_dir in input_dirs:
        for pattern in patterns:
            paths.extend(sorted(input_dir.glob(pattern)))

    unique_paths = []
    seen_paths = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths or resolved in output_paths:
            continue
        if path.name.endswith(".metadata.json"):
            continue
        if any(fnmatch.fnmatch(path.name, exclude) for exclude in excludes):
            continue
        unique_paths.append(path)
        seen_paths.add(resolved)

    return unique_paths


def dedupe_records(records: list[Any], key_name: str, keep: str) -> tuple[list[Any], list[Any]]:
    if not key_name:
        return records, []

    index_by_key = {}
    deduped = []
    duplicate_keys = []
    for record in records:
        key = record.get(key_name) if isinstance(record, dict) else None
        if key is None:
            deduped.append(record)
            continue

        if key in index_by_key:
            duplicate_keys.append(key)
            if keep == "last":
                deduped[index_by_key[key]] = record
            continue

        index_by_key[key] = len(deduped)
        deduped.append(record)

    return deduped, duplicate_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine COT batch JSON files.")
    parser.add_argument("--inputs", nargs="+", default=[], type=Path)
    parser.add_argument("--input-dir", action="append", default=[], type=Path)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata_output", type=Path)
    parser.add_argument("--dedupe-key", default="")
    parser.add_argument("--dedupe-keep", choices=["first", "last"], default="last")
    args = parser.parse_args()

    output_paths = {args.output.resolve()}
    if args.metadata_output:
        output_paths.add(args.metadata_output.resolve())
    input_paths = collect_input_paths(
        explicit_inputs=args.inputs,
        input_dirs=args.input_dir,
        patterns=args.pattern,
        excludes=args.exclude,
        output_paths=output_paths,
    )
    if not input_paths:
        raise SystemExit("No input files matched.")

    combined = []
    metadata = {
        "batches": [],
        "records": 0,
        "records_before_dedupe": 0,
        "dedupe_key": args.dedupe_key or None,
        "dedupe_keep": args.dedupe_keep if args.dedupe_key else None,
        "sidecar_failures": 0,
        "duplicate_global_idx": [],
        "deduped_keys": [],
        "missing_global_idx": [],
    }
    seen_global_idx = set()

    for input_path in input_paths:
        batch = load_json(input_path)
        if not isinstance(batch, list):
            raise ValueError(f"{input_path} is not a top-level list")
        combined.extend(batch)

        for item in batch:
            global_idx = item.get("global_idx") if isinstance(item, dict) else None
            if global_idx in seen_global_idx:
                metadata["duplicate_global_idx"].append(global_idx)
            seen_global_idx.add(global_idx)

        sidecar = find_metadata_sidecar(input_path)
        sidecar_data = load_json(sidecar) if sidecar else {}
        sidecar_failures = metadata_failure_count(sidecar_data)
        metadata["batches"].append(
            {
                "path": str(input_path),
                "records": len(batch),
                "metadata_records": sidecar_data.get("records"),
                "metadata_failures": sidecar_failures,
                "finish_reasons": [
                    item.get("finish_reason")
                    for item in sidecar_data.get("generation_diagnostics", [])
                ],
            }
        )
        metadata["sidecar_failures"] += sidecar_failures or 0

    metadata["records_before_dedupe"] = len(combined)
    combined, deduped_keys = dedupe_records(combined, args.dedupe_key, args.dedupe_keep)
    combined.sort(key=lambda item: item.get("global_idx", 10**12))
    metadata["records"] = len(combined)
    metadata["deduped_keys"] = deduped_keys
    metadata["global_idx_range"] = [
        combined[0].get("global_idx") if combined else None,
        combined[-1].get("global_idx") if combined else None,
    ]
    if combined:
        actual = {item.get("global_idx") for item in combined}
        start, end = metadata["global_idx_range"]
        metadata["missing_global_idx"] = [
            idx for idx in range(start, end + 1) if idx not in actual
        ]

    write_json(args.output, combined)
    if args.metadata_output:
        write_json(args.metadata_output, metadata)
    print(f"Wrote {len(combined)} records to {args.output}")
    if metadata["sidecar_failures"]:
        print(f"Failures reported by sidecars: {metadata['sidecar_failures']}")
    if metadata["missing_global_idx"]:
        print(f"Missing global_idx values: {metadata['missing_global_idx']}")
    if metadata["duplicate_global_idx"]:
        print(f"Duplicate global_idx values: {metadata['duplicate_global_idx']}")
    if metadata["deduped_keys"]:
        print(f"Deduped {len(metadata['deduped_keys'])} duplicate keys by {args.dedupe_key}")


if __name__ == "__main__":
    main()

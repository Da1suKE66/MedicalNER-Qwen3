"""
Combine COT batch JSON files produced by run_cot_gemini_generation.py.

Each input batch is a top-level list. Sidecar metadata files with suffix
`.metadata.json` are optional and summarized when present.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine COT batch JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata_output", type=Path)
    args = parser.parse_args()

    combined = []
    metadata = {
        "batches": [],
        "records": 0,
        "sidecar_failures": 0,
        "duplicate_global_idx": [],
        "missing_global_idx": [],
    }
    seen_global_idx = set()

    for input_path in args.inputs:
        batch = load_json(input_path)
        if not isinstance(batch, list):
            raise ValueError(f"{input_path} is not a top-level list")
        combined.extend(batch)

        for item in batch:
            global_idx = item.get("global_idx") if isinstance(item, dict) else None
            if global_idx in seen_global_idx:
                metadata["duplicate_global_idx"].append(global_idx)
            seen_global_idx.add(global_idx)

        sidecar = input_path.with_suffix(input_path.suffix + ".metadata.json")
        sidecar_data = load_json(sidecar) if sidecar.exists() else {}
        metadata["batches"].append(
            {
                "path": str(input_path),
                "records": len(batch),
                "metadata_records": sidecar_data.get("records"),
                "metadata_failures": sidecar_data.get("failures"),
                "finish_reasons": [
                    item.get("finish_reason")
                    for item in sidecar_data.get("generation_diagnostics", [])
                ],
            }
        )
        metadata["sidecar_failures"] += int(sidecar_data.get("failures") or 0)

    combined.sort(key=lambda item: item.get("global_idx", 10**12))
    metadata["records"] = len(combined)
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract selected LLaMA-Factory records for the schema regression set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TEST_SET = Path("data/schema_regression/schema_regression_20.json")
DEFAULT_SOURCE = Path("data/llamafactory/pro_cot_001_858_complete_llamafactory.json")
DEFAULT_OUTPUT = Path("data/llamafactory/schema_regression_20_llamafactory.json")
DEFAULT_DATASET_INFO = Path("data/llamafactory/dataset_info.json")
DEFAULT_DATASET_NAME = "schema_regression_20"
MEDICAL_TEXT_MARKER = "Medical text:\n"


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def extract_medical_text(record: dict[str, Any]) -> str:
    messages = record.get("messages", [])
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if MEDICAL_TEXT_MARKER in content:
            return content.split(MEDICAL_TEXT_MARKER, 1)[1].strip()
    raise ValueError("source record does not contain a user message with medical text")


def find_source_record(
    item: dict[str, Any], source_records: list[dict[str, Any]], used_indices: set[int]
) -> tuple[int, dict[str, Any]]:
    expected_input = item["input"].strip()
    global_idx = item.get("global_idx")

    if isinstance(global_idx, int) and 0 <= global_idx < len(source_records):
        source_text = extract_medical_text(source_records[global_idx])
        if source_text == expected_input:
            return global_idx, source_records[global_idx]
        raise ValueError(
            f"{item.get('test_id', '<unknown>')} global_idx={global_idx} does not match input"
        )

    matches = [
        idx
        for idx, record in enumerate(source_records)
        if idx not in used_indices and extract_medical_text(record) == expected_input
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{item.get('test_id', '<unknown>')} matched {len(matches)} source records by input"
        )
    return matches[0], source_records[matches[0]]


def update_dataset_info(path: Path, dataset_name: str, output_path: Path) -> None:
    dataset_info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    dataset_info[dataset_name] = {
        "file_name": output_path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    path.write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LLaMA-Factory eval dataset from schema regression metadata."
    )
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--no-update-dataset-info",
        action="store_true",
        help="Only write the extracted JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_items = load_json_or_jsonl(args.test_set)
    source_records = load_json_or_jsonl(args.source)

    if not isinstance(test_items, list):
        raise TypeError(f"{args.test_set} must contain a JSON array or JSONL records")
    if not isinstance(source_records, list):
        raise TypeError(f"{args.source} must contain a JSON array")

    extracted_records: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    selected_indices: list[int] = []

    for item in test_items:
        idx, record = find_source_record(item, source_records, used_indices)
        if idx in used_indices:
            raise ValueError(f"duplicate source index selected: {idx}")
        used_indices.add(idx)
        selected_indices.append(idx)
        extracted_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(extracted_records, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if not args.no_update_dataset_info:
        update_dataset_info(args.dataset_info, args.dataset_name, args.output)

    print(f"wrote {len(extracted_records)} records to {args.output}")
    print(f"selected source indices: {selected_indices}")
    if not args.no_update_dataset_info:
        print(f"registered dataset '{args.dataset_name}' in {args.dataset_info}")


if __name__ == "__main__":
    main()

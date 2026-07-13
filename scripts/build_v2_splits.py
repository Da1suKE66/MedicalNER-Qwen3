#!/usr/bin/env python3
"""Freeze hierarchy-grouped Schema v2 train/validation/held-out splits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    write_json,
)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def hierarchy_groups(raw_records: list[dict[str, Any]]) -> tuple[list[list[str]], int]:
    source_ids = [str(record["id"]) for record in raw_records]
    known = set(source_ids)
    union_find = UnionFind(source_ids)
    edge_count = 0
    for record in raw_records:
        source_id = str(record["id"])
        for related in list(record.get("parent") or []) + list(record.get("child") or []):
            related_id = str(related)
            if related_id in known:
                union_find.union(source_id, related_id)
                edge_count += 1
    groups: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        groups[union_find.find(source_id)].append(source_id)
    return list(groups.values()), edge_count


def deterministic_order(groups: list[list[str]], seed: str) -> list[list[str]]:
    def key(group: list[str]) -> str:
        identity = min(group)
        return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()

    return sorted(groups, key=key)


def take_groups(
    groups: list[list[str]], target_count: int
) -> tuple[list[list[str]], list[list[str]]]:
    selected: list[list[str]] = []
    selected_count = 0
    remaining = list(groups)
    while remaining and selected_count < target_count:
        group = remaining.pop(0)
        selected.append(group)
        selected_count += len(group)
    return selected, remaining


def flatten(groups: list[list[str]]) -> set[str]:
    return {source_id for group in groups for source_id in group}


def source_id_for_record(record: dict[str, Any]) -> str:
    value = record.get("source_record_id") or record.get("source_id")
    if not value:
        raise ValueError("record has no source_record_id/source_id")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data/raw/mental_disorders_20251125_165535.json",
    )
    parser.add_argument(
        "--schema-regression",
        type=Path,
        default=ROOT / "data/schema_regression/schema_regression_20.json",
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="Optional migrated records to materialize into the frozen split files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/schema_v2/splits"
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--heldout-ratio", type=float, default=0.10)
    parser.add_argument("--seed", default="medicalner-schema-v2-2025-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_ratio <= 0 or args.heldout_ratio <= 0:
        raise ValueError("validation and held-out ratios must be positive")
    if args.validation_ratio + args.heldout_ratio >= 1:
        raise ValueError("validation and held-out ratios must sum to less than 1")

    raw_records = record_list(load_json(args.raw))
    regression_records = record_list(load_json(args.schema_regression))
    raw_ids = [str(record["id"]) for record in raw_records]
    raw_id_set = set(raw_ids)
    regression_ids = {
        str(record.get("source_id") or record.get("source_record_id"))
        for record in regression_records
    }
    unknown_regression = regression_ids - raw_id_set
    if unknown_regression:
        raise ValueError(f"schema regression source IDs missing from raw data: {unknown_regression}")

    groups, edge_count = hierarchy_groups(raw_records)
    candidate_ids = raw_id_set - regression_ids
    candidate_groups = [
        [source_id for source_id in group if source_id in candidate_ids]
        for group in groups
    ]
    candidate_groups = [group for group in candidate_groups if group]
    ordered_groups = deterministic_order(candidate_groups, args.seed)

    candidate_count = len(candidate_ids)
    heldout_target = round(candidate_count * args.heldout_ratio)
    validation_target = round(candidate_count * args.validation_ratio)
    heldout_groups, remaining = take_groups(ordered_groups, heldout_target)
    validation_groups, train_groups = take_groups(remaining, validation_target)

    split_sets = {
        "train": flatten(train_groups),
        "validation": flatten(validation_groups),
        "test_v2_heldout": flatten(heldout_groups),
        "schema_regression_20": regression_ids,
    }
    primary_sets = [split_sets["train"], split_sets["validation"], split_sets["test_v2_heldout"]]
    if any(left & right for i, left in enumerate(primary_sets) for right in primary_sets[i + 1 :]):
        raise AssertionError("primary splits overlap")
    if set().union(*primary_sets) != candidate_ids:
        raise AssertionError("primary splits do not cover every non-regression source record")

    assignment = {
        source_id: split
        for split in ("train", "validation", "test_v2_heldout")
        for source_id in split_sets[split]
    }
    cross_split_hierarchy_edges = []
    for record in raw_records:
        source_id = str(record["id"])
        if source_id not in assignment:
            continue
        for related in list(record.get("parent") or []) + list(record.get("child") or []):
            related_id = str(related)
            if related_id in assignment and assignment[source_id] != assignment[related_id]:
                cross_split_hierarchy_edges.append(
                    {
                        "source": source_id,
                        "source_split": assignment[source_id],
                        "target": related_id,
                        "target_split": assignment[related_id],
                    }
                )
    if cross_split_hierarchy_edges:
        raise AssertionError("hierarchy edges cross primary splits")

    manifest = {
        "schema_version": load_schema()["schema_version"],
        "source_release": "2025-01",
        "seed": args.seed,
        "strategy": "connected components over in-dataset parent/child ICD-11 MMS links",
        "raw_record_count": len(raw_records),
        "hierarchy_component_count": len(groups),
        "internal_hierarchy_edge_references": edge_count,
        "cross_split_hierarchy_edges": cross_split_hierarchy_edges,
        "counts": {name: len(values) for name, values in split_sets.items()},
        "source_ids": {
            name: [source_id for source_id in raw_ids if source_id in values]
            for name, values in split_sets.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "split_manifest.json", manifest)

    if args.records:
        migrated = record_list(load_json(args.records))
        by_id = {source_id_for_record(record): record for record in migrated}
        missing = candidate_ids - set(by_id)
        if missing:
            raise ValueError(f"migrated records missing {len(missing)} split source IDs")
        for split, filename in (
            ("train", "train_v2.json"),
            ("validation", "validation_v2.json"),
            ("test_v2_heldout", "test_v2_heldout.json"),
        ):
            records = [by_id[source_id] for source_id in raw_ids if source_id in split_sets[split]]
            write_json(args.output_dir / filename, records)

    print(f"counts={manifest['counts']}")
    print(f"hierarchy_components={len(groups)} cross_split_edges=0")
    print(f"wrote {args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()

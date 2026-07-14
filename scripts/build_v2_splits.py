#!/usr/bin/env python3
"""Freeze hierarchy-grouped Schema v2 train/validation/held-out splits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    write_json,
)


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value

    def find(self, value: str) -> str:
        self.add(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


EXTERNAL_PARENT_PREFIX = "__schema_v2_external_parent__::"


def external_parent_node(parent_id: str) -> str:
    """Return a collision-resistant virtual Union-Find node for a missing parent."""

    return f"{EXTERNAL_PARENT_PREFIX}{parent_id}"


def build_hierarchy_groups(
    raw_records: list[dict[str, Any]],
) -> tuple[list[list[str]], dict[str, Any]]:
    """Group source records without splitting hierarchy siblings.

    Parent identifiers outside the local extract still carry hierarchy information.
    They are represented as virtual Union-Find nodes so that two local records that
    share the same external parent remain in the same connected component.
    """

    source_ids = [str(record["id"]) for record in raw_records]
    known = set(source_ids)
    colliding_ids = [value for value in source_ids if value.startswith(EXTERNAL_PARENT_PREFIX)]
    if colliding_ids:
        raise ValueError(
            "source IDs collide with the reserved external-parent prefix: "
            f"{colliding_ids[:3]}"
        )
    union_find = UnionFind(source_ids)
    internal_edge_references = 0
    external_parent_members: dict[str, set[str]] = defaultdict(set)
    for record in raw_records:
        source_id = str(record["id"])
        for parent in record.get("parent") or []:
            parent_id = str(parent)
            if parent_id in known:
                union_find.union(source_id, parent_id)
                internal_edge_references += 1
            else:
                union_find.union(source_id, external_parent_node(parent_id))
                external_parent_members[parent_id].add(source_id)
        for child in record.get("child") or []:
            child_id = str(child)
            if child_id in known:
                union_find.union(source_id, child_id)
                internal_edge_references += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        groups[union_find.find(source_id)].append(source_id)
    group_values = list(groups.values())
    metadata = {
        "internal_hierarchy_edge_references": internal_edge_references,
        "external_parent_node_count": len(external_parent_members),
        "external_parent_reference_count": sum(
            len(members) for members in external_parent_members.values()
        ),
        "shared_external_parent_node_count": sum(
            len(members) > 1 for members in external_parent_members.values()
        ),
        "component_count": len(group_values),
        "component_sizes_desc": sorted(
            (len(group) for group in group_values), reverse=True
        ),
    }
    return group_values, metadata


def hierarchy_groups(raw_records: list[dict[str, Any]]) -> tuple[list[list[str]], int]:
    """Backward-compatible public helper returning groups and internal edge count."""

    groups, metadata = build_hierarchy_groups(raw_records)
    return groups, int(metadata["internal_hierarchy_edge_references"])


def deterministic_order(groups: list[list[str]], seed: str) -> list[list[str]]:
    def key(group: list[str]) -> str:
        identity = min(group)
        return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()

    return sorted(groups, key=key)


def take_groups(
    groups: list[list[str]], target_count: int
) -> tuple[list[list[str]], list[list[str]]]:
    """Select whole components with a deterministic closest-size subset sum."""

    if target_count <= 0 or not groups:
        return [], list(groups)

    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(groups):
        size = len(group)
        for count, indexes in list(reachable.items()):
            reachable.setdefault(count + size, indexes + (index,))

    positive_counts = [count for count in reachable if count > 0]
    best_count = min(
        positive_counts,
        key=lambda count: (
            abs(count - target_count),
            count < target_count,
            len(reachable[count]),
            reachable[count],
        ),
    )
    selected_indexes = set(reachable[best_count])
    selected = [group for index, group in enumerate(groups) if index in selected_indexes]
    remaining = [group for index, group in enumerate(groups) if index not in selected_indexes]
    return selected, remaining


def flatten(groups: list[list[str]]) -> set[str]:
    return {source_id for group in groups for source_id in group}


def source_id_for_record(record: dict[str, Any]) -> str:
    value = record.get("source_record_id") or record.get("source_id")
    if not value:
        raise ValueError("record has no source_record_id/source_id")
    return str(value)


def hierarchy_leakage_report(
    raw_records: list[dict[str, Any]], assignment: dict[str, str]
) -> dict[str, Any]:
    """Audit direct hierarchy edges and shared external parents across primary splits."""

    known = {str(record["id"]) for record in raw_records}
    direct_cross_split_edges: list[dict[str, str]] = []
    external_parent_assignments: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in raw_records:
        source_id = str(record["id"])
        source_split = assignment.get(source_id)
        if source_split is None:
            continue
        for related in list(record.get("parent") or []) + list(record.get("child") or []):
            related_id = str(related)
            target_split = assignment.get(related_id)
            if target_split is not None and target_split != source_split:
                direct_cross_split_edges.append(
                    {
                        "source": source_id,
                        "source_split": source_split,
                        "target": related_id,
                        "target_split": target_split,
                    }
                )
        for parent in record.get("parent") or []:
            parent_id = str(parent)
            if parent_id not in known:
                external_parent_assignments[parent_id][source_split].append(source_id)

    shared_external_parent_cross_splits = []
    for parent_id, split_members in sorted(external_parent_assignments.items()):
        if len(split_members) <= 1:
            continue
        shared_external_parent_cross_splits.append(
            {
                "external_parent": parent_id,
                "members_by_split": {
                    split: sorted(members)
                    for split, members in sorted(split_members.items())
                },
            }
        )

    passed = not direct_cross_split_edges and not shared_external_parent_cross_splits
    return {
        "scope": "primary splits only; schema_regression_20 is a reserved structural check",
        "passed": passed,
        "direct_cross_split_edge_count": len(direct_cross_split_edges),
        "direct_cross_split_edges": direct_cross_split_edges,
        "shared_external_parent_cross_split_count": len(
            shared_external_parent_cross_splits
        ),
        "shared_external_parent_cross_splits": shared_external_parent_cross_splits,
    }


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

    groups, hierarchy_metadata = build_hierarchy_groups(raw_records)
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
    leakage = hierarchy_leakage_report(raw_records, assignment)
    if not leakage["passed"]:
        raise AssertionError("hierarchy or shared-parent siblings cross primary splits")

    manifest = {
        "schema_version": load_schema()["schema_version"],
        "source_release": "2025-01",
        "seed": args.seed,
        "strategy": (
            "connected components over in-dataset parent/child ICD-11 MMS links "
            "plus virtual Union-Find nodes for shared external parents"
        ),
        "raw_record_count": len(raw_records),
        "hierarchy_component_count": len(groups),
        "internal_hierarchy_edge_references": hierarchy_metadata[
            "internal_hierarchy_edge_references"
        ],
        "external_parent_node_count": hierarchy_metadata["external_parent_node_count"],
        "external_parent_reference_count": hierarchy_metadata[
            "external_parent_reference_count"
        ],
        "shared_external_parent_node_count": hierarchy_metadata[
            "shared_external_parent_node_count"
        ],
        "hierarchy_component_sizes_desc": hierarchy_metadata["component_sizes_desc"],
        # Retained for compatibility with the existing schema-v2 integration test.
        "cross_split_hierarchy_edges": leakage["direct_cross_split_edges"],
        "hierarchy_leakage_check": leakage,
        "split_eligibility": {
            "eligible": {
                "reason": "non_regression_source_record",
                "count": candidate_count,
            },
            "excluded": [
                {
                    "reason": "reserved_schema_regression_20",
                    "count": len(regression_ids),
                    "note": "reported separately and not treated as held-out generalization",
                }
            ],
        },
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
    print(
        f"hierarchy_components={len(groups)} direct_cross_split_edges=0 "
        "shared_external_parent_cross_splits=0"
    )
    print(f"wrote {args.output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()

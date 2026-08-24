#!/usr/bin/env python3
"""Build a frozen output-only SFT dataset from the revised relations CSV.

The previous dataset mixed an older relation vocabulary with the revised
workbook.  This script makes the protocol explicit before training:

* keep only rows marked as directly admissible or source-correctable;
* apply the CSV's relation-renaming suggestions;
* remove relations that the CSV turns into properties or deletes;
* validate both source and target labels against the CSV target contract;
* replace the prompt relation table with source/target constrained rows; and
* compact JSON targets so a long graph spends fewer tokens on whitespace.

This is deliberately a data builder, not a medical-semantic adjudicator.  A
review queue remains in the report and is not silently promoted to training.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ACTIVE_PREFIXES = ("可加入", "需修正 Source 后加入", "需修正 Source 并复核")
MODULE_TO_LABEL = {
    "TreatmentPlan": "Treatment Plan",
    "Communication": "Communication Strategy",
}

# Names present in the older teacher targets.  A None value means the revised
# CSV explicitly turns the old edge into a node property or removes it.
LEGACY_RELATION_MAP: dict[str, str | None] = {
    "has_manifestation": "has_subsymptom",
    "supports_Diagnostic Criterion_of": "supports_diagnostic_criterion",
    "required_for": "required_for_diagnosis_of",
    "increases_risk_of": None,  # revised CSV: increased_b node property
    "worsens": None,  # revised CSV: aggravated_by node property
    "has_specifier": None,  # revised CSV: core Disease property
}


def normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def source_label(value: str) -> str:
    value = value.strip()
    return MODULE_TO_LABEL.get(value, value)


def parse_target_labels(value: str) -> list[str]:
    value = value.strip()
    if not value or value.startswith("直接删除") or value.startswith("改为"):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [source_label(str(item)) for item in parsed if str(item).strip()]


def is_active(status: str) -> bool:
    return any(status.strip().startswith(prefix) for prefix in ACTIVE_PREFIXES)


def canonical_relation(row: dict[str, str]) -> str:
    return (row.get("Relation重命名建议") or row.get("Relation") or "").strip()


def load_policy(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    all_rows: list[dict[str, Any]] = []
    active_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = {
            "module": row.get("模块", "").strip(),
            "source_label": source_label(row.get("Source修正建议") or row.get("模块", "")),
            "relation": row.get("Relation", "").strip(),
            "canonical_relation": canonical_relation(row),
            "chinese": row.get("中文", "").strip(),
            "target_labels": parse_target_labels(row.get("Target（最终推断）", "")),
            "status": row.get("建议状态", "").strip(),
            "confidence": row.get("置信度", "").strip(),
            "source_correction": row.get("Source修正建议", "").strip(),
            "rename_suggestion": row.get("Relation重命名建议", "").strip(),
            "notes": row.get("推断/Schema备注", "").strip(),
        }
        all_rows.append(item)
        if is_active(item["status"]):
            active_by_relation[item["canonical_relation"]].append(item)
    return all_rows, active_by_relation


def relation_policy_for(
    relation_name: str,
    source_entity_label: str,
    active_by_relation: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, dict[str, Any] | None, str]:
    canonical = LEGACY_RELATION_MAP.get(relation_name, relation_name)
    if canonical is None:
        return None, None, "revised_property_or_deleted"
    candidates = active_by_relation.get(canonical, [])
    if not candidates:
        return None, None, "not_active_in_relations_csv"
    source_norm = normalize(source_entity_label)
    exact = [item for item in candidates if normalize(item["source_label"]) == source_norm]
    if exact:
        return canonical, exact[0], "active"
    # If a relation is intentionally shared by several source modules, a
    # unique relation row is still usable.  Do not accept a source mismatch
    # when the CSV has an explicit, distinct source contract.
    if len(candidates) == 1 and not candidates[0]["source_correction"]:
        return canonical, candidates[0], "active_shared_source"
    return None, None, "source_label_not_allowed"


def parse_output(text: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"<output>(.*?)</output>", text, flags=re.S | re.I)
    if match:
        inner = match.group(1).strip()
    else:
        inner = text.strip()
    graph = json.loads(inner)
    if not isinstance(graph, dict) or not isinstance(graph.get("entities"), list):
        raise ValueError("assistant target is not a graph object")
    if not isinstance(graph.get("relations"), list):
        raise ValueError("assistant target has no relations list")
    return inner, graph


def compact_assistant(text: str, graph: dict[str, Any]) -> str:
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    if re.search(r"<output>", text, flags=re.I):
        return f"<output>\n{payload}\n</output>"
    return payload


def replace_relation_table(system: str, table: str) -> str:
    start = system.find("# Relation definitions")
    end = system.find("# Extraction contract")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("system prompt does not contain the expected relation section")
    return system[:start] + table.rstrip() + "\n\n" + system[end:]


def relation_table(policy: list[dict[str, Any]]) -> str:
    lines = [
        "# Relation definitions (revised relations.csv; frozen active subset)",
        "| Source label | Relation | Allowed target labels | 中文 |",
        "|---|---|---|---|",
    ]
    for item in policy:
        if not item["target_labels"]:
            continue
        targets = ", ".join(item["target_labels"])
        chinese = item.get("chinese", "")
        lines.append(
            f"| {item['source_label']} | {item['canonical_relation']} | {targets} | {chinese} |"
        )
    return "\n".join(lines)


def build_policy_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in all_rows:
        if not is_active(item["status"]) or not item["target_labels"]:
            continue
        key = (
            item["source_label"],
            item["canonical_relation"],
            tuple(item["target_labels"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def transform_dataset(
    source: Path,
    destination: Path,
    policy_rows: list[dict[str, Any]],
    active_by_relation: dict[str, list[dict[str, Any]]],
    oversample_relation: int,
) -> dict[str, Any]:
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a list dataset: {source}")
    table = relation_table(policy_rows)
    out: list[dict[str, Any]] = []
    stats = Counter()
    relation_before = Counter()
    relation_after = Counter()
    dropped_by_reason = Counter()
    max_entities = max_relations = 0

    for index, row in enumerate(rows):
        item = deepcopy(row)
        messages = item.get("messages") or item.get("conversations")
        if not isinstance(messages, list):
            raise ValueError(f"row {index} has no messages/conversations")
        for message in messages:
            if message.get("role") == "system":
                message["content"] = replace_relation_table(message["content"], table)
                message["content"] = message["content"].replace(
                    "5. The workbook does not define target-label constraints. Do not invent target types or additional direction rules.",
                    "5. The revised relation table is a closed source/target contract. Emit a relation only when both endpoint labels are allowed for that row; never substitute Symptom for a long Diagnostic Criterion.",
                )
        assistant = next((m for m in messages if m.get("role") == "assistant"), None)
        if assistant is None:
            raise ValueError(f"row {index} has no assistant message")
        _, graph = parse_output(assistant["content"])
        entities = {
            str(entity.get("id") or ""): entity
            for entity in graph.get("entities", [])
            if isinstance(entity, dict)
        }
        for relation in graph.get("relations", []):
            if isinstance(relation, dict):
                relation_before[str(relation.get("relation") or "")] += 1

        new_relations = []
        for relation in graph.get("relations", []):
            if not isinstance(relation, dict):
                stats["invalid_relation"] += 1
                continue
            source_id = str(relation.get("source") or "")
            target_id = str(relation.get("target") or "")
            source_entity = entities.get(source_id)
            target_entity = entities.get(target_id)
            if not source_entity or not target_entity:
                dropped_by_reason["missing_endpoint"] += 1
                continue
            canonical, spec, reason = relation_policy_for(
                str(relation.get("relation") or ""),
                str(source_entity.get("label") or ""),
                active_by_relation,
            )
            if spec is None or canonical is None:
                dropped_by_reason[reason] += 1
                continue
            target_label = source_label(str(target_entity.get("label") or ""))
            if normalize(target_label) not in {normalize(v) for v in spec["target_labels"]}:
                dropped_by_reason["target_label_not_allowed"] += 1
                continue
            updated = dict(relation)
            updated["relation"] = canonical
            new_relations.append(updated)
            relation_after[canonical] += 1
        graph["relations"] = new_relations
        max_entities = max(max_entities, len(graph.get("entities", [])))
        max_relations = max(max_relations, len(new_relations))
        assistant["content"] = compact_assistant(assistant["content"], graph)
        # Keep the same row once, then optionally repeat relation-bearing rows.
        copies = 1 + (oversample_relation if new_relations else 0)
        out.extend(deepcopy(item) for _ in range(copies))
        stats["rows"] += 1
        stats["relation_bearing_rows"] += bool(new_relations)
        stats["output_rows"] += copies

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "source": str(source),
        "destination": str(destination),
        "input_rows": len(rows),
        "output_rows": len(out),
        "oversample_relation": oversample_relation,
        "rows_with_relations": stats["relation_bearing_rows"],
        "relation_count_before": sum(relation_before.values()),
        "relation_count_after": sum(relation_after.values()),
        "relation_types_before": dict(relation_before),
        "relation_types_after": dict(relation_after),
        "dropped_relations_by_reason": dict(dropped_by_reason),
        "max_entities": max_entities,
        "max_relations": max_relations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relations-csv", type=Path, required=True)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--dev", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--oversample-relation",
        type=int,
        default=0,
        help="extra copies for rows retaining at least one revised relation",
    )
    args = parser.parse_args()
    if args.oversample_relation < 0:
        raise SystemExit("--oversample-relation must be non-negative")

    all_rows, active_by_relation = load_policy(args.relations_csv)
    policy_rows = build_policy_rows(all_rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "relations_v2_active_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "source_csv": str(args.relations_csv),
                "active_prefixes": list(ACTIVE_PREFIXES),
                "legacy_relation_map": LEGACY_RELATION_MAP,
                "all_rows": all_rows,
                "active_rows": policy_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = []
    dataset_info: dict[str, Any] = {}
    for split, paths in (("train", args.train), ("dev", args.dev)):
        for path in paths:
            destination = output_dir / f"{path.stem}_relations_v2.json"
            summary = transform_dataset(
                path,
                destination,
                policy_rows,
                active_by_relation,
                args.oversample_relation if split == "train" else 0,
            )
            summary["split"] = split
            summaries.append(summary)
            dataset_name = f"medicalner_relations_v2_{split}"
            dataset_info[dataset_name] = {
                "file_name": destination.name,
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
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "relations_csv": str(args.relations_csv),
        "active_policy_rows": len(policy_rows),
        "review_rows": len(all_rows) - len(policy_rows),
        "summaries": summaries,
    }
    report_path = output_dir / "relations_v2_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

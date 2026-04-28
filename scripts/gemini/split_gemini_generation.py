"""
Split ICD disorder records into separate Gemini inputs and merge generated KG output.

The intended workflow is:
1. Split the source ICD JSON into:
   - structural input: identifiers, hierarchy, synonyms, inclusions/exclusions, etc.
   - long-text input: diagnosticCriteria and narrowerTerms, with minimal context.
2. Run Gemini separately on each input with prompts tailored to the field group.
3. Merge the two generated outputs by source_id.

Expected generated output shape for merge:
[
  {
    "source_id": "http://id.who.int/...",
    "code": "6A00",
    "title": "Disorders of intellectual development",
    "entities": [...],
    "relations": [...]
  }
]

The merge command also accepts {"records": [...]} or a single record object.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LONG_TEXT_FIELDS = {"definition", "longDefinition", "diagnosticCriteria", "narrowerTerms"}

IDENTITY_FIELDS = ["id", "code", "title"]

STRUCTURAL_EXCLUDE_FIELDS = LONG_TEXT_FIELDS

TEXT_CONTEXT_FIELDS = ["id", "code", "title", "definition"]


def reference_keys(reference: str) -> list[str]:
    if not reference:
        return []
    keys = [reference]
    tail = reference.rstrip("/").split("/")[-1]
    if tail:
        keys.append(tail)
    return keys


def build_entity_lookup(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for entity in entities:
        source_id = entity.get("id", "")
        item = {
            "id": source_id,
            "code": entity.get("code", ""),
            "title": entity.get("title", ""),
            "classKind": entity.get("classKind", ""),
        }
        for key in reference_keys(source_id):
            lookup[key] = item
        if "/release/" in source_id and "/mms/" in source_id:
            foundation_id = source_id.split("/mms/", 1)[-1]
            lookup[f"http://id.who.int/icd/entity/{foundation_id}"] = item
            lookup[foundation_id] = item
    return lookup


def resolve_references(
    references: list[str], lookup: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    resolved = []
    for reference in references or []:
        match = None
        for key in reference_keys(reference):
            if key in lookup:
                match = lookup[key]
                break
        if match:
            resolved.append({"reference": reference, **match})
        else:
            resolved.append({"reference": reference, "id": reference, "code": "", "title": ""})
    return resolved


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_source(input_path: Path, output_dir: Path, prefix: str) -> None:
    data = load_json(input_path)
    entities = data.get("entities", [])
    entity_lookup = build_entity_lookup(entities)

    structural_entities = []
    text_entities = []
    manifest_records = []

    for entity in entities:
        source_id = entity.get("id")
        structural_entity = {
            key: value
            for key, value in entity.items()
            if key not in STRUCTURAL_EXCLUDE_FIELDS
        }
        structural_entity["source_id"] = source_id
        for field in ["parent", "child", "ancestor", "descendant"]:
            structural_entity[f"{field}Resolved"] = resolve_references(
                entity.get(field, []), entity_lookup
            )
        structural_entities.append(structural_entity)

        text_entity = {
            key: entity.get(key, "" if key != "narrowerTerms" else [])
            for key in TEXT_CONTEXT_FIELDS
        }
        text_entity["source_id"] = source_id
        text_entity["diagnosticCriteria"] = entity.get("diagnosticCriteria", "")
        text_entity["narrowerTerms"] = entity.get("narrowerTerms", [])
        text_entities.append(text_entity)

        manifest_records.append(
            {
                "source_id": source_id,
                "code": entity.get("code", ""),
                "title": entity.get("title", ""),
                "has_diagnosticCriteria": bool(entity.get("diagnosticCriteria")),
                "narrowerTerms_count": len(entity.get("narrowerTerms") or []),
            }
        )

    metadata = dict(data.get("metadata", {}))
    metadata["split_from"] = str(input_path)
    metadata["split_note"] = (
        "Generated for two-pass Gemini extraction: structural fields and long text fields."
    )

    write_json(
        output_dir / f"{prefix}.structure.json",
        {"metadata": metadata | {"split": "structure"}, "entities": structural_entities},
    )
    write_json(
        output_dir / f"{prefix}.long_text.json",
        {"metadata": metadata | {"split": "long_text"}, "entities": text_entities},
    )
    write_json(
        output_dir / f"{prefix}.manifest.json",
        {"metadata": metadata, "records": manifest_records},
    )

    print(f"Wrote {len(structural_entities)} structural records")
    print(f"Wrote {len(text_entities)} long-text records")
    print(f"Output directory: {output_dir}")


def normalize_records(data: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    elif isinstance(data, dict) and (
        "entities" in data or "relationships" in data or "relations" in data
    ):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"{label} must be a list, a single record, or an object with records")

    records = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{label}[{idx}] is not an object")
        if not record.get("source_id"):
            raise ValueError(f"{label}[{idx}] missing source_id")
        records.append(record)
    return records


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_unique(left: list[Any], right: list[Any]) -> list[Any]:
    merged = []
    seen = set()
    for item in [*left, *right]:
        marker = stable_json(item)
        if marker in seen:
            continue
        seen.add(marker)
        merged.append(item)
    return merged


def get_relations(record: dict[str, Any]) -> list[Any]:
    relations = record.get("relations")
    if relations is not None:
        return relations
    return record.get("relationships", [])


def entity_key(entity: dict[str, Any]) -> tuple[str, str] | None:
    name = str(entity.get("name", "")).strip().lower()
    label = str(entity.get("label", "")).strip().lower()
    if not name or not label:
        return None
    return label, name


def scoped_id(scope: str, entity_id: str) -> str:
    if not entity_id:
        return ""
    return f"{scope}_{entity_id}"


def merge_property_value(current: Any, incoming: Any) -> Any:
    if current in ("", None, [], {}):
        return incoming
    if incoming in ("", None, [], {}):
        return current
    if isinstance(current, dict) and isinstance(incoming, dict):
        return merge_properties(current, incoming)
    if current == incoming:
        return current
    if isinstance(current, str) and isinstance(incoming, str):
        parts = [part.strip() for part in current.split(";") if part.strip()]
        if incoming.strip() not in parts:
            parts.append(incoming.strip())
        return "; ".join(parts)
    return current


def merge_properties(
    current: dict[str, Any] | None, incoming: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(current or {})
    for key, value in (incoming or {}).items():
        merged[key] = merge_property_value(merged.get(key), value)
    return merged


def add_scoped_record(
    merged: dict[str, Any],
    entity_index: dict[tuple[str, str], str],
    record: dict[str, Any],
    scope: str,
) -> None:
    local_to_merged_id = {}

    for entity in record.get("entities", []):
        if not isinstance(entity, dict):
            continue
        original_id = str(entity.get("id", "")).strip()
        scoped_entity = dict(entity)
        scoped_entity["id"] = scoped_id(scope, original_id) if original_id else ""

        key = entity_key(scoped_entity)
        if key and key in entity_index:
            canonical_id = entity_index[key]
            local_to_merged_id[original_id] = canonical_id
            for existing in merged["entities"]:
                if existing.get("id") == canonical_id:
                    existing["properties"] = merge_properties(
                        existing.get("properties", {}), scoped_entity.get("properties", {})
                    )
                    break
            continue

        if not scoped_entity["id"]:
            scoped_entity["id"] = f"{scope}_E{len(merged['entities']) + 1}"
        local_to_merged_id[original_id] = scoped_entity["id"]
        if key:
            entity_index[key] = scoped_entity["id"]
        merged["entities"].append(scoped_entity)

    scoped_relations = []
    for relation in get_relations(record):
        if not isinstance(relation, dict):
            continue
        scoped_relation = dict(relation)
        for endpoint in ["source", "target"]:
            original = str(scoped_relation.get(endpoint, "")).strip()
            if original in local_to_merged_id:
                scoped_relation[endpoint] = local_to_merged_id[original]
        scoped_relations.append(scoped_relation)

    merged["relations"] = merge_unique(merged.get("relations", []), scoped_relations)


def merge_outputs(
    structure_output: Path,
    long_text_output: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> None:
    structure_records = normalize_records(load_json(structure_output), "structure_output")
    text_records = normalize_records(load_json(long_text_output), "long_text_output")

    merged_by_id: dict[str, dict[str, Any]] = {}

    if manifest_path:
        manifest = load_json(manifest_path)
        for item in manifest.get("records", []):
            source_id = item["source_id"]
            merged_by_id[source_id] = {
                "source_id": source_id,
                "code": item.get("code", ""),
                "title": item.get("title", ""),
                "entities": [],
                "relations": [],
                "provenance": {"structure": False, "long_text": False},
            }

    entity_indexes: dict[str, dict[tuple[str, str], str]] = {}

    for label, records in [("structure", structure_records), ("long_text", text_records)]:
        for record in records:
            source_id = record["source_id"]
            merged = merged_by_id.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "code": record.get("code", ""),
                    "title": record.get("title", ""),
                    "entities": [],
                    "relations": [],
                    "provenance": {"structure": False, "long_text": False},
                },
            )
            if not merged.get("code") and record.get("code"):
                merged["code"] = record["code"]
            if not merged.get("title") and record.get("title"):
                merged["title"] = record["title"]

            entity_index = entity_indexes.setdefault(
                source_id,
                {
                    key: str(entity.get("id", ""))
                    for entity in merged.get("entities", [])
                    if (key := entity_key(entity))
                },
            )
            add_scoped_record(merged, entity_index, record, label)
            merged["provenance"][label] = True

    output = {
        "metadata": {
            "structure_output": str(structure_output),
            "long_text_output": str(long_text_output),
            "record_count": len(merged_by_id),
        },
        "records": list(merged_by_id.values()),
    }
    write_json(output_path, output)
    print(f"Wrote merged output: {output_path}")
    print(f"Merged records: {len(merged_by_id)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split ICD JSON for two-pass Gemini generation and merge outputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Create structure/long-text inputs")
    split_parser.add_argument("--input", required=True, type=Path)
    split_parser.add_argument("--output_dir", required=True, type=Path)
    split_parser.add_argument("--prefix", default="mental_disorders")

    merge_parser = subparsers.add_parser("merge", help="Merge two generated KG outputs")
    merge_parser.add_argument("--structure_output", required=True, type=Path)
    merge_parser.add_argument("--long_text_output", required=True, type=Path)
    merge_parser.add_argument("--output", required=True, type=Path)
    merge_parser.add_argument("--manifest", type=Path)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "split":
        split_source(args.input, args.output_dir, args.prefix)
    elif args.command == "merge":
        merge_outputs(
            args.structure_output,
            args.long_text_output,
            args.output,
            args.manifest,
        )


if __name__ == "__main__":
    main()

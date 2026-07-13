"""Versioned Schema v2 migration and validation helpers.

The module intentionally uses only the Python standard library so that audits can run
before a CUDA/NPU training environment is created.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_POINTER = REPO_ROOT / "schemas/current.json"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema(pointer_path: Path = DEFAULT_SCHEMA_POINTER) -> dict[str, Any]:
    pointer = load_json(pointer_path)
    schema_path = pointer_path.parent / pointer["schema_path"]
    schema = load_json(schema_path)
    if schema.get("schema_version") != pointer.get("schema_version"):
        raise ValueError("schemas/current.json version does not match the target schema")
    schema["_schema_path"] = str(schema_path)
    return schema


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper() and original[1:].islower():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def correct_text(text: str, corrections: dict[str, str]) -> tuple[str, Counter[str]]:
    applied: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        replacement = corrections.get(token.lower())
        if replacement is None:
            return token
        applied[token.lower()] += 1
        return _preserve_case(token, replacement)

    return TOKEN_RE.sub(replace, text), applied


def correct_text_tree(value: Any, corrections: dict[str, str]) -> tuple[Any, Counter[str]]:
    total: Counter[str] = Counter()

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            repaired, applied = correct_text(item, corrections)
            total.update(applied)
            return repaired
        return item

    return visit(value), total


def record_list(raw_payload: Any) -> list[dict[str, Any]]:
    if isinstance(raw_payload, list):
        return raw_payload
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("entities"), list):
        return raw_payload["entities"]
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("records"), list):
        return raw_payload["records"]
    raise TypeError("expected a JSON list or an object containing entities/records")


def build_raw_indexes(raw_payload: Any) -> dict[str, Any]:
    records = record_list(raw_payload)
    by_id = {str(record.get("id")): record for record in records if record.get("id")}
    by_title: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_title.setdefault(normalize_text(record.get("title")), []).append(record)
    return {"records": records, "by_id": by_id, "by_title": by_title}


def resolve_source_record(
    record: dict[str, Any], raw_indexes: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    by_id = raw_indexes["by_id"]
    source_id = record.get("source_record_id") or record.get("source_id")
    if source_id and str(source_id) in by_id:
        return by_id[str(source_id)], "source_record_id"

    global_idx = record.get("global_idx")
    if isinstance(global_idx, int) and 0 <= global_idx < len(raw_indexes["records"]):
        candidate = raw_indexes["records"][global_idx]
        expected_title = normalize_text(record.get("source_title") or record.get("title"))
        if expected_title and normalize_text(candidate.get("title")) == expected_title:
            return candidate, "global_idx+title"

    title = normalize_text(record.get("source_title") or record.get("title"))
    matches = raw_indexes["by_title"].get(title, [])
    if len(matches) == 1:
        return matches[0], "canonical_title"
    return None, "unresolved"


def graph_key(record: dict[str, Any]) -> str | None:
    for key in ("output", "gold_output"):
        if isinstance(record.get(key), dict):
            return key
    if isinstance(record.get("entities"), list):
        return None
    return None


def graph_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    key = graph_key(record)
    if key:
        return record[key]
    if isinstance(record.get("entities"), list):
        return record
    return None


def _main_disease_id(graph: dict[str, Any], source_title: str) -> str | None:
    matches = [
        str(entity.get("id"))
        for entity in graph.get("entities", [])
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == normalize_text(source_title)
    ]
    return matches[0] if len(matches) == 1 else None


def _merge_property(existing: Any, incoming: Any) -> Any:
    if existing in (None, "", []):
        return incoming
    if incoming in (None, "", []):
        return existing
    if existing == incoming:
        return existing
    if isinstance(existing, list):
        result = list(existing)
        for value in incoming if isinstance(incoming, list) else [incoming]:
            if value not in result:
                result.append(value)
        return result
    return [existing, incoming]


def _canonicalize_properties(
    graph: dict[str, Any],
    main_disease_id: str | None,
    source_record: dict[str, Any],
    schema: dict[str, Any],
    changes: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    unverified_codes: list[dict[str, Any]],
) -> None:
    aliases = schema["property_aliases"]
    code_aliases = {key for key, value in aliases.items() if value == "icdcode"}

    for entity in graph.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("id", ""))
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            properties = {}

        canonical: dict[str, Any] = {}
        for key, value in properties.items():
            new_key = aliases.get(key, key)
            if key in code_aliases or new_key == "icdcode":
                if entity_id == main_disease_id:
                    continue
                if value not in (None, ""):
                    unverified_codes.append(
                        {"entity_id": entity_id, "property": key, "value": value}
                    )
                    warnings.append(
                        {
                            "code": "non_main_code_requires_entity_linking",
                            "entity_id": entity_id,
                            "value": value,
                        }
                    )
                continue
            canonical[new_key] = _merge_property(canonical.get(new_key), value)
            if new_key != key:
                changes.append(
                    {
                        "op": "rename_property",
                        "entity_id": entity_id,
                        "from": key,
                        "to": new_key,
                    }
                )

        if entity_id == main_disease_id:
            canonical.update(
                {
                    "icdcode": source_record.get("code", ""),
                    "coding_system": schema["coding"]["coding_system"],
                    "icd_release": schema["source_release"],
                    "icd_uri": source_record.get("id", ""),
                }
            )
            changes.append(
                {
                    "op": "inject_main_disease_coding",
                    "entity_id": entity_id,
                    "icdcode": source_record.get("code", ""),
                }
            )
        entity["properties"] = canonical


def _labels_by_id(graph: dict[str, Any]) -> dict[str, str]:
    return {
        str(entity.get("id")): str(entity.get("label"))
        for entity in graph.get("entities", [])
        if isinstance(entity, dict) and entity.get("id")
    }


def _matches_direction(spec: dict[str, Any], source_label: str, target_label: str) -> bool:
    return source_label in spec.get("source", []) and target_label in spec.get("target", [])


def _normalize_relations(
    graph: dict[str, Any],
    source_text: str,
    schema: dict[str, Any],
    changes: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    labels = _labels_by_id(graph)
    relation_specs = schema["relation_types"]
    lowered_text = source_text.lower()

    for index, relation in enumerate(graph.get("relations", [])):
        if not isinstance(relation, dict):
            errors.append({"code": "relation_not_object", "relation_index": index})
            continue
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        if source not in labels or target not in labels:
            errors.append(
                {
                    "code": "ghost_relation_endpoint",
                    "relation_index": index,
                    "source": source,
                    "target": target,
                }
            )
            continue

        relation_name = str(relation.get("relation", ""))
        spec = relation_specs.get(relation_name)
        if spec is None:
            errors.append(
                {
                    "code": "undefined_relation",
                    "relation_index": index,
                    "relation": relation_name,
                }
            )
            continue
        if spec.get("status") == "needs_medical_review":
            warnings.append(
                {
                    "code": "relation_semantics_need_medical_review",
                    "relation_index": index,
                    "relation": relation_name,
                }
            )
        else:
            actual = (labels[source], labels[target])
            if not _matches_direction(spec, *actual):
                reversed_pair = (labels[target], labels[source])
                if _matches_direction(spec, *reversed_pair):
                    relation["source"], relation["target"] = target, source
                    changes.append(
                        {
                            "op": "swap_relation_endpoints",
                            "relation_index": index,
                            "relation": relation_name,
                            "from": [source, target],
                            "to": [target, source],
                        }
                    )
                else:
                    errors.append(
                        {
                            "code": "invalid_relation_domain_range",
                            "relation_index": index,
                            "relation": relation_name,
                            "actual": list(actual),
                            "expected_source": spec.get("source", []),
                            "expected_target": spec.get("target", []),
                        }
                    )

        evidence = relation.get("evidence")
        if isinstance(evidence, str) and evidence.strip() and "evidence_span" not in relation:
            needle = evidence.strip().lower()
            start = lowered_text.find(needle)
            if start >= 0 and lowered_text.find(needle, start + 1) < 0:
                relation["evidence_span"] = {
                    "basis": "record.input",
                    "text": source_text[start : start + len(needle)],
                    "start": start,
                    "end": start + len(needle),
                }
                changes.append(
                    {
                        "op": "add_relation_evidence_span",
                        "relation_index": index,
                    }
                )


def _definition_contains_child(source_text: str, parent_name: str, child_name: str) -> int:
    lowered = source_text.lower()
    parent = re.escape(parent_name.lower())
    cue = re.search(
        parent + r"\s+(?:refers to|includes|is characterized by|is characterised by)",
        lowered,
    )
    if cue is None:
        return -1
    sentence_end = len(lowered)
    for marker in (". ", ".\n", "\n\n"):
        found = lowered.find(marker, cue.end())
        if found >= 0:
            sentence_end = min(sentence_end, found)
    child_start = lowered.find(child_name.lower(), cue.end(), sentence_end)
    return child_start


def _collapse_description_children(
    graph: dict[str, Any], source_text: str, changes: list[dict[str, Any]]
) -> None:
    entities = [entity for entity in graph.get("entities", []) if isinstance(entity, dict)]
    relations = [relation for relation in graph.get("relations", []) if isinstance(relation, dict)]
    removed_ids: set[str] = set()

    for parent in entities:
        if parent.get("label") != "Symptom":
            continue
        parent_id = str(parent.get("id", ""))
        description = str((parent.get("properties") or {}).get("description", ""))
        if not parent_id or not description:
            continue
        parent_relations = [
            relation
            for relation in relations
            if str(relation.get("source")) == parent_id
            and relation.get("relation")
            in {"is_core_symptom_of", "is_associated_symptom_of"}
        ]
        for child in entities:
            child_id = str(child.get("id", ""))
            child_name = str(child.get("name", "")).strip()
            if (
                child_id == parent_id
                or child_id in removed_ids
                or child.get("label") != "Symptom"
                or not child_name
                or child.get("properties") not in ({}, None)
                or child_name.lower() not in description.lower()
            ):
                continue
            touching = [
                relation
                for relation in relations
                if str(relation.get("source")) == child_id
                or str(relation.get("target")) == child_id
            ]
            if len(touching) != 1 or str(touching[0].get("source")) != child_id:
                continue
            child_relation = touching[0]
            same_parent_relation = any(
                relation.get("relation") == child_relation.get("relation")
                and str(relation.get("target")) == str(child_relation.get("target"))
                for relation in parent_relations
            )
            if not same_parent_relation:
                continue
            start = _definition_contains_child(
                source_text, str(parent.get("name", "")), child_name
            )
            if start < 0:
                continue

            manifestations = (parent.get("properties") or {}).setdefault("manifestations", [])
            if child_name not in manifestations:
                manifestations.append(child_name)
            removed_ids.add(child_id)
            changes.append(
                {
                    "op": "move_entity_to_manifestation",
                    "entity_id": child_id,
                    "parent_entity_id": parent_id,
                    "evidence": {
                        "basis": "record.input",
                        "text": source_text[start : start + len(child_name)],
                        "start": start,
                        "end": start + len(child_name),
                    },
                }
            )

    if removed_ids:
        graph["entities"] = [
            entity for entity in entities if str(entity.get("id")) not in removed_ids
        ]
        graph["relations"] = [
            relation
            for relation in relations
            if str(relation.get("source")) not in removed_ids
            and str(relation.get("target")) not in removed_ids
        ]


def migrate_record(
    record: dict[str, Any],
    raw_indexes: dict[str, Any],
    schema: dict[str, Any],
    *,
    apply_high_confidence_collapses: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    version = schema["schema_version"]
    if record.get("schema_version") == version:
        existing = copy.deepcopy(record)
        return existing, copy.deepcopy(existing.get("migration", {}))

    migrated, text_fixes = correct_text_tree(record, schema["text_corrections"])
    source_record, source_match = resolve_source_record(migrated, raw_indexes)
    changes: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    unverified_codes: list[dict[str, Any]] = []

    if text_fixes:
        changes.append({"op": "repair_text_tokens", "tokens": dict(text_fixes)})
    if source_record is None:
        errors.append({"code": "source_record_unresolved"})
    else:
        repaired_source, _ = correct_text_tree(source_record, schema["text_corrections"])
        migrated.update(
            {
                "schema_version": version,
                "source_record_id": repaired_source.get("id"),
                "source_code": repaired_source.get("code"),
                "source_title": repaired_source.get("title"),
                "source_release": schema["source_release"],
            }
        )
        changes.append({"op": "attach_source_identity", "method": source_match})
        source_record = repaired_source

    graph = graph_from_record(migrated)
    if graph is None:
        errors.append({"code": "graph_missing"})
    elif source_record is not None:
        main_id = _main_disease_id(graph, str(source_record.get("title", "")))
        if main_id is None:
            errors.append({"code": "main_disease_unresolved"})
        _canonicalize_properties(
            graph,
            main_id,
            source_record,
            schema,
            changes,
            warnings,
            unverified_codes,
        )
        if apply_high_confidence_collapses:
            _collapse_description_children(graph, str(migrated.get("input", "")), changes)
        _normalize_relations(
            graph,
            str(migrated.get("input", "")),
            schema,
            changes,
            warnings,
            errors,
        )

    status = "invalid" if errors else "manual_review" if warnings else "repaired"
    migration = {
        "target_schema_version": version,
        "status": status,
        "source_match": source_match,
        "changes": changes,
        "warnings": warnings,
        "errors": errors,
        "unverified_codes": unverified_codes,
    }
    migrated["migration"] = migration
    return migrated, migration


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics = {"entities": 0, "relations": 0, "relations_with_evidence": 0}

    for field in (
        "schema_version",
        "source_record_id",
        "source_code",
        "source_title",
        "source_release",
    ):
        if record.get(field) in (None, ""):
            errors.append({"code": "missing_record_field", "field": field})
    if record.get("schema_version") != schema["schema_version"]:
        errors.append(
            {
                "code": "schema_version_mismatch",
                "actual": record.get("schema_version"),
                "expected": schema["schema_version"],
            }
        )

    graph = graph_from_record(record)
    if graph is None:
        errors.append({"code": "graph_missing"})
        return {"errors": errors, "warnings": warnings, "metrics": metrics}

    entities = [entity for entity in graph.get("entities", []) if isinstance(entity, dict)]
    relations = [relation for relation in graph.get("relations", []) if isinstance(relation, dict)]
    metrics["entities"] = len(entities)
    metrics["relations"] = len(relations)

    labels: dict[str, str] = {}
    for entity in entities:
        entity_id = str(entity.get("id", ""))
        label = str(entity.get("label", ""))
        if not entity_id:
            errors.append({"code": "entity_id_missing"})
            continue
        if entity_id in labels:
            errors.append({"code": "duplicate_entity_id", "entity_id": entity_id})
        labels[entity_id] = label
        spec = schema["entity_types"].get(label)
        if spec is None:
            errors.append(
                {"code": "undefined_entity_label", "entity_id": entity_id, "label": label}
            )
            continue
        allowed = set(spec["properties"])
        for key in (entity.get("properties") or {}):
            if key not in allowed:
                errors.append(
                    {
                        "code": "undefined_property",
                        "entity_id": entity_id,
                        "label": label,
                        "property": key,
                    }
                )

    main_id = _main_disease_id(graph, str(record.get("source_title", "")))
    if main_id is None:
        errors.append({"code": "main_disease_unresolved"})
    else:
        main = next(entity for entity in entities if str(entity.get("id")) == main_id)
        props = main.get("properties") or {}
        expected = {
            "icdcode": record.get("source_code"),
            "coding_system": schema["coding"]["coding_system"],
            "icd_release": record.get("source_release"),
            "icd_uri": record.get("source_record_id"),
        }
        for key, value in expected.items():
            if props.get(key) != value:
                errors.append(
                    {
                        "code": "main_disease_coding_mismatch",
                        "entity_id": main_id,
                        "property": key,
                        "actual": props.get(key),
                        "expected": value,
                    }
                )

    relation_specs = schema["relation_types"]
    for index, relation in enumerate(relations):
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        relation_name = str(relation.get("relation", ""))
        if source not in labels or target not in labels:
            errors.append(
                {
                    "code": "ghost_relation_endpoint",
                    "relation_index": index,
                    "source": source,
                    "target": target,
                }
            )
            continue
        spec = relation_specs.get(relation_name)
        if spec is None:
            errors.append(
                {
                    "code": "undefined_relation",
                    "relation_index": index,
                    "relation": relation_name,
                }
            )
        elif spec.get("status") == "needs_medical_review":
            warnings.append(
                {
                    "code": "relation_semantics_need_medical_review",
                    "relation_index": index,
                    "relation": relation_name,
                }
            )
        elif not _matches_direction(spec, labels[source], labels[target]):
            errors.append(
                {
                    "code": "invalid_relation_domain_range",
                    "relation_index": index,
                    "relation": relation_name,
                    "actual": [labels[source], labels[target]],
                }
            )
        if relation.get("evidence") or relation.get("evidence_span"):
            metrics["relations_with_evidence"] += 1

    return {"errors": errors, "warnings": warnings, "metrics": metrics}


def validate_dataset(records: Iterable[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
    results = []
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    totals = Counter()
    for index, record in enumerate(records):
        result = validate_record(record, schema)
        for error in result["errors"]:
            error_counts[error["code"]] += 1
        for warning in result["warnings"]:
            warning_counts[warning["code"]] += 1
        totals.update(result["metrics"])
        if result["errors"] or result["warnings"]:
            results.append(
                {
                    "record_index": index,
                    "source_record_id": record.get("source_record_id")
                    or record.get("source_id"),
                    **result,
                }
            )
    record_count = index + 1 if "index" in locals() else 0
    return {
        "schema_version": schema["schema_version"],
        "record_count": record_count,
        "records_with_findings": len(results),
        "error_counts": dict(error_counts),
        "warning_counts": dict(warning_counts),
        "metrics": dict(totals),
        "findings": results,
    }

"""Versioned Schema v2 migration and validation helpers.

The module intentionally uses only the Python standard library so that audits can run
before a CUDA/NPU training environment is created.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_POINTER = REPO_ROOT / "schemas/current.json"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
MIGRATION_IMPLEMENTATION_VERSION = "schema-v2-core.2"
GRAPH_SOURCE_FIELDS = {
    "source_id": "id",
    "code": "code",
    "title": "title",
}


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def load_json(path: Path) -> Any:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_constant
    )

    def reject_nonfinite_values(value: Any, location: str = "$") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite JSON number is not allowed at {location}")
        if isinstance(value, dict):
            for key, child in value.items():
                reject_nonfinite_values(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_nonfinite_values(child, f"{location}[{index}]")

    reject_nonfinite_values(payload)
    return payload


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


def migration_config_fingerprint(
    schema: dict[str, Any], *, apply_high_confidence_collapses: bool
) -> str:
    """Return a stable fingerprint for every option that changes migration output."""

    public_schema = {
        key: value for key, value in schema.items() if not str(key).startswith("_")
    }
    payload = {
        "schema": public_schema,
        "apply_high_confidence_collapses": apply_high_confidence_collapses,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    raw_entities = graph.get("entities", [])
    if not isinstance(raw_entities, list):
        return None
    matches = [
        str(entity.get("id"))
        for entity in raw_entities
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == normalize_text(source_title)
    ]
    return matches[0] if len(matches) == 1 else None


def _ensure_canonical_main_disease(
    graph: dict[str, Any],
    source_record: dict[str, Any],
    changes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> str | None:
    """Inject a provenance-only main Disease when no exact canonical node exists.

    Existing near matches are deliberately left untouched: choosing or renaming one
    would require semantic inference. Multiple exact matches are also rejected rather
    than silently selecting one.
    """

    entities = graph.get("entities")
    if not isinstance(entities, list):
        return None
    canonical_title = str(source_record.get("title", ""))
    exact_matches = [
        entity
        for entity in entities
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == normalize_text(canonical_title)
    ]
    if len(exact_matches) == 1:
        return str(exact_matches[0].get("id"))
    if len(exact_matches) > 1:
        errors.append(
            {
                "code": "main_disease_ambiguous",
                "canonical_title": canonical_title,
                "entity_ids": [str(entity.get("id", "")) for entity in exact_matches],
            }
        )
        return None
    if not canonical_title:
        errors.append({"code": "canonical_source_title_missing"})
        return None

    existing_ids = {
        str(entity.get("id"))
        for entity in entities
        if isinstance(entity, dict) and entity.get("id")
    }
    base_id = "D_CANONICAL_SOURCE"
    entity_id = base_id
    suffix = 2
    while entity_id in existing_ids:
        entity_id = f"{base_id}_{suffix}"
        suffix += 1
    entities.append(
        {
            "id": entity_id,
            "label": "Disease",
            "name": canonical_title,
            "properties": {},
        }
    )
    changes.append(
        {
            "op": "inject_canonical_main_disease",
            "entity_id": entity_id,
            "source_record_id": source_record.get("id"),
            "source_code": source_record.get("code"),
            "source_title": canonical_title,
            "basis": "canonical_source_metadata",
        }
    )
    return entity_id


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

    raw_entities = graph.get("entities", [])
    if not isinstance(raw_entities, list):
        return
    for index, entity in enumerate(raw_entities):
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


def _canonicalize_graph_source_identity(
    graph: dict[str, Any],
    source_record: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    before = {field: graph.get(field) for field in GRAPH_SOURCE_FIELDS}
    expected = {
        field: source_record.get(source_field)
        for field, source_field in GRAPH_SOURCE_FIELDS.items()
    }
    for field, value in expected.items():
        graph[field] = value
    if before != expected:
        changes.append(
            {
                "op": "canonicalize_graph_source_identity",
                "from": before,
                "to": expected,
            }
        )


def _graph_structure_errors(graph: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    entities = graph.get("entities")
    if not isinstance(entities, list):
        errors.append(
            {
                "code": "entities_not_list",
                "actual_type": type(entities).__name__,
            }
        )
    else:
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(
                    {
                        "code": "entity_not_object",
                        "entity_index": index,
                        "actual_type": type(entity).__name__,
                    }
                )
                continue
            for field in ("id", "label", "name"):
                if entity.get(field) in (None, ""):
                    errors.append(
                        {
                            "code": "missing_entity_field",
                            "entity_index": index,
                            "field": field,
                        }
                    )
            properties = entity.get("properties")
            if not isinstance(properties, dict):
                errors.append(
                    {
                        "code": "entity_properties_not_object",
                        "entity_index": index,
                        "entity_id": str(entity.get("id", "")),
                        "actual_type": type(properties).__name__,
                    }
                )

    relations = graph.get("relations")
    if not isinstance(relations, list):
        errors.append(
            {
                "code": "relations_not_list",
                "actual_type": type(relations).__name__,
            }
        )
    else:
        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                errors.append(
                    {
                        "code": "relation_not_object",
                        "relation_index": index,
                        "actual_type": type(relation).__name__,
                    }
                )
                continue
            for field in ("source", "target", "relation"):
                if relation.get(field) in (None, ""):
                    errors.append(
                        {
                            "code": "missing_relation_field",
                            "relation_index": index,
                            "field": field,
                        }
                    )
    return errors


def _labels_by_id(graph: dict[str, Any]) -> dict[str, str]:
    raw_entities = graph.get("entities", [])
    if not isinstance(raw_entities, list):
        return {}
    return {
        str(entity.get("id")): str(entity.get("label"))
        for entity in raw_entities
        if isinstance(entity, dict) and entity.get("id")
    }


def _matches_direction(spec: dict[str, Any], source_label: str, target_label: str) -> bool:
    allowed_pairs = spec.get("allowed_pairs")
    if isinstance(allowed_pairs, list):
        return [source_label, target_label] in allowed_pairs
    return source_label in spec.get("source", []) and target_label in spec.get("target", [])


def _rewrite_relation(
    relation_name: str,
    source_label: str,
    target_label: str,
    schema: dict[str, Any],
) -> str:
    """Apply only schema-declared, endpoint-specific legacy relation rewrites."""

    for rewrite in schema.get("relation_rewrites", []):
        if not isinstance(rewrite, dict) or rewrite.get("from") != relation_name:
            continue
        if source_label not in rewrite.get("source", []):
            continue
        if target_label not in rewrite.get("target", []):
            continue
        replacement = rewrite.get("to")
        if isinstance(replacement, str) and replacement:
            return replacement
    return relation_name


def _repair_known_relation_shapes(
    graph: dict[str, Any],
    main_disease_id: str | None,
    changes: list[dict[str, Any]],
) -> None:
    """Repair endpoint patterns whose predicate semantics determine one outcome."""

    if not main_disease_id:
        return
    labels = _labels_by_id(graph)
    relations = graph.get("relations")
    if not isinstance(relations, list):
        return
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            continue
        source = str(relation.get("source", ""))
        target = str(relation.get("target", ""))
        relation_name = str(relation.get("relation", ""))
        before = {
            "source": source,
            "target": target,
            "relation": relation_name,
        }
        reason = ""
        if (
            relation_name == "required_for_diagnosis_of"
            and labels.get(source) == "Diagnostic Criteria"
            and labels.get(target) == "Symptom"
        ):
            relation["target"] = main_disease_id
            reason = "diagnosis predicate must target the canonical main Disease"
        elif (
            relation_name == "co_occurs_with_frequency"
            and source == main_disease_id
            and labels.get(source) == "Disease"
            and labels.get(target) == "Symptom"
        ):
            relation["source"] = target
            relation["target"] = main_disease_id
            relation["relation"] = "is_associated_symptom_of"
            reason = "symptom co-occurrence is represented as Symptom-to-Disease association"
        if reason:
            changes.append(
                {
                    "op": "repair_known_relation_shape",
                    "relation_index": index,
                    "reason": reason,
                    "from": before,
                    "to": {
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "relation": relation.get("relation"),
                    },
                }
            )


def _iter_grounding_candidates(entity: dict[str, Any]) -> Iterable[str]:
    name = entity.get("name")
    if isinstance(name, str) and name.strip():
        yield name.strip()

    def visit(value: Any) -> Iterable[str]:
        if isinstance(value, str) and value.strip():
            yield value.strip()
        elif isinstance(value, list):
            for item in value:
                yield from visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from visit(item)

    yield from visit(entity.get("properties") or {})


def _add_grounded_patient_diagnosis_relations(
    graph: dict[str, Any],
    source_text: str,
    main_disease_id: str | None,
    changes: list[dict[str, Any]],
) -> None:
    if not main_disease_id:
        return
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return
    existing = {
        (str(relation.get("source")), str(relation.get("target")), relation.get("relation"))
        for relation in relations
        if isinstance(relation, dict)
    }
    lowered = source_text.lower()
    for entity in entities:
        if not isinstance(entity, dict) or entity.get("label") != "Patient Information":
            continue
        entity_id = str(entity.get("id") or "")
        relation_key = (entity_id, main_disease_id, "affects_diagnosis_of")
        if not entity_id or relation_key in existing:
            continue
        located: tuple[int, str] | None = None
        for candidate in _iter_grounding_candidates(entity):
            if len(candidate) < 4:
                continue
            start = lowered.find(candidate.lower())
            if start >= 0:
                located = (start, source_text[start : start + len(candidate)])
                break
        if located is None:
            continue
        start, evidence_text = located
        relation = {
            "source": entity_id,
            "target": main_disease_id,
            "relation": "affects_diagnosis_of",
            "evidence": evidence_text,
            "evidence_span": {
                "basis": "record.input",
                "text": evidence_text,
                "start": start,
                "end": start + len(evidence_text),
            },
        }
        relations.append(relation)
        existing.add(relation_key)
        changes.append(
            {
                "op": "add_grounded_patient_diagnosis_relation",
                "entity_id": entity_id,
                "main_disease_id": main_disease_id,
                "evidence": relation["evidence_span"],
            }
        )


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

    raw_relations = graph.get("relations", [])
    if not isinstance(raw_relations, list):
        return
    for index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
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
        rewritten_name = _rewrite_relation(
            relation_name, labels[source], labels[target], schema
        )
        if rewritten_name != relation_name:
            relation["relation"] = rewritten_name
            changes.append(
                {
                    "op": "rewrite_relation",
                    "relation_index": index,
                    "from": relation_name,
                    "to": rewritten_name,
                    "source_label": labels[source],
                    "target_label": labels[target],
                }
            )
            relation_name = rewritten_name
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
    raw_entities = graph.get("entities", [])
    raw_relations = graph.get("relations", [])
    if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
        return
    entities = [entity for entity in raw_entities if isinstance(entity, dict)]
    relations = [relation for relation in raw_relations if isinstance(relation, dict)]
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
            entity
            for entity in raw_entities
            if not isinstance(entity, dict)
            or str(entity.get("id")) not in removed_ids
        ]
        graph["relations"] = [
            relation
            for relation in raw_relations
            if not isinstance(relation, dict)
            or (
                str(relation.get("source")) not in removed_ids
                and str(relation.get("target")) not in removed_ids
            )
        ]


def migrate_record(
    record: dict[str, Any],
    raw_indexes: dict[str, Any],
    schema: dict[str, Any],
    *,
    apply_high_confidence_collapses: bool = False,
    force_renormalize: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    version = schema["schema_version"]
    config_fingerprint = migration_config_fingerprint(
        schema,
        apply_high_confidence_collapses=apply_high_confidence_collapses,
    )
    existing_migration = record.get("migration")
    migration_is_current = (
        isinstance(existing_migration, dict)
        and existing_migration.get("target_schema_version") == version
        and existing_migration.get("implementation_version")
        == MIGRATION_IMPLEMENTATION_VERSION
        and existing_migration.get("config_fingerprint") == config_fingerprint
    )
    if (
        not force_renormalize
        and record.get("schema_version") == version
        and migration_is_current
    ):
        existing = copy.deepcopy(record)
        return existing, copy.deepcopy(existing.get("migration", {}))

    migrated, text_fixes = correct_text_tree(record, schema["text_corrections"])
    migrated["schema_version"] = version
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
    else:
        errors.extend(_graph_structure_errors(graph))
    if graph is not None and source_record is not None:
        _canonicalize_graph_source_identity(graph, source_record, changes)
        main_id = _ensure_canonical_main_disease(
            graph, source_record, changes, errors
        )
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
        _repair_known_relation_shapes(graph, main_id, changes)
        _add_grounded_patient_diagnosis_relations(
            graph, str(migrated.get("input", "")), main_id, changes
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
        raw_relations = graph.get("relations", [])
        if isinstance(raw_relations, list):
            for index, relation in enumerate(raw_relations):
                if not isinstance(relation, dict):
                    continue
                evidence_errors, _, _ = _validate_relation_evidence(
                    relation, index, str(migrated.get("input", ""))
                )
                errors.extend(evidence_errors)

    status = "invalid" if errors else "manual_review" if warnings else "repaired"
    migration = {
        "target_schema_version": version,
        "implementation_version": MIGRATION_IMPLEMENTATION_VERSION,
        "config_fingerprint": config_fingerprint,
        "status": status,
        "source_match": source_match,
        "changes": changes,
        "warnings": warnings,
        "errors": errors,
        "unverified_codes": unverified_codes,
    }
    migrated["migration"] = migration
    return migrated, migration


def _validate_relation_evidence(
    relation: dict[str, Any], relation_index: int, source_text: str
) -> tuple[list[dict[str, Any]], bool, bool]:
    errors: list[dict[str, Any]] = []
    evidence = relation.get("evidence")
    has_retained_text = False
    if "evidence" in relation:
        if not isinstance(evidence, str):
            errors.append(
                {
                    "code": "relation_evidence_not_string",
                    "relation_index": relation_index,
                    "actual_type": type(evidence).__name__,
                }
            )
        else:
            has_retained_text = bool(evidence.strip())

    if "evidence_span" not in relation:
        return errors, has_retained_text, False

    span_errors: list[dict[str, Any]] = []
    span = relation.get("evidence_span")
    if not isinstance(span, dict):
        span_errors.append(
            {
                "code": "evidence_span_not_object",
                "relation_index": relation_index,
                "actual_type": type(span).__name__,
            }
        )
        errors.extend(span_errors)
        return errors, has_retained_text, False

    for field in ("basis", "text", "start", "end"):
        if field not in span:
            span_errors.append(
                {
                    "code": "evidence_span_missing_field",
                    "relation_index": relation_index,
                    "field": field,
                }
            )
    if span.get("basis") != "record.input":
        span_errors.append(
            {
                "code": "evidence_span_invalid_basis",
                "relation_index": relation_index,
                "actual": span.get("basis"),
                "expected": "record.input",
            }
        )

    span_text = span.get("text")
    if not isinstance(span_text, str) or not span_text:
        span_errors.append(
            {
                "code": "evidence_span_invalid_text",
                "relation_index": relation_index,
            }
        )

    start = span.get("start")
    end = span.get("end")
    valid_integer_bounds = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    )
    if not valid_integer_bounds:
        span_errors.append(
            {
                "code": "evidence_span_invalid_bound_type",
                "relation_index": relation_index,
            }
        )
    elif not (0 <= start < end <= len(source_text)):
        span_errors.append(
            {
                "code": "evidence_span_out_of_bounds",
                "relation_index": relation_index,
                "start": start,
                "end": end,
                "input_length": len(source_text),
            }
        )
    elif isinstance(span_text, str) and source_text[start:end] != span_text:
        span_errors.append(
            {
                "code": "evidence_span_text_mismatch",
                "relation_index": relation_index,
                "actual": span_text,
                "expected": source_text[start:end],
            }
        )

    if (
        has_retained_text
        and isinstance(span_text, str)
        and evidence.strip() != span_text
    ):
        span_errors.append(
            {
                "code": "evidence_span_retained_text_mismatch",
                "relation_index": relation_index,
            }
        )
    errors.extend(span_errors)
    return errors, has_retained_text, not span_errors


def validate_record(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    metrics = {
        "entities": 0,
        "relations": 0,
        "relations_with_retained_evidence_text": 0,
        "relations_with_verified_evidence_span": 0,
    }

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

    errors.extend(_graph_structure_errors(graph))
    raw_entities = graph.get("entities", [])
    raw_relations = graph.get("relations", [])
    entities = (
        [entity for entity in raw_entities if isinstance(entity, dict)]
        if isinstance(raw_entities, list)
        else []
    )
    relations = (
        [relation for relation in raw_relations if isinstance(relation, dict)]
        if isinstance(raw_relations, list)
        else []
    )
    metrics["entities"] = len(entities)
    metrics["relations"] = len(relations)

    expected_graph_identity = {
        "source_id": record.get("source_record_id"),
        "code": record.get("source_code"),
        "title": record.get("source_title"),
    }
    for field, expected in expected_graph_identity.items():
        if graph.get(field) != expected:
            errors.append(
                {
                    "code": "graph_source_identity_mismatch",
                    "field": field,
                    "actual": graph.get(field),
                    "expected": expected,
                }
            )

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
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            continue
        allowed = set(spec["properties"])
        for key in properties:
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
        props = main.get("properties")
        if not isinstance(props, dict):
            props = {}
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
        evidence_errors, retained_text, verified_span = _validate_relation_evidence(
            relation, index, str(record.get("input", ""))
        )
        errors.extend(evidence_errors)
        if retained_text:
            metrics["relations_with_retained_evidence_text"] += 1
        if verified_span:
            metrics["relations_with_verified_evidence_span"] += 1
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

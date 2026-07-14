"""Build a grounded training graph while retaining a complete removal audit."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Iterable

from .evidence_repair import locate_evidence_span, span_is_exact
from .schema_v2 import graph_from_record, normalize_text, validate_record


LEGACY_GENERATION_FIELDS = frozenset(
    {
        "code",
        "cot",
        "cot_style",
        "input_chars",
        "input_used",
        "original_input_chars",
        "response_had_output_tag",
        "response_had_think_tag",
        "response_had_thought_parts",
        "source_id",
        "success",
        "title",
    }
)


def sanitize_record_for_training(
    original: dict[str, Any], schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = copy.deepcopy(original)
    removed_legacy_fields = sorted(
        key for key in LEGACY_GENERATION_FIELDS if key in record
    )
    for key in removed_legacy_fields:
        record.pop(key, None)
    graph = graph_from_record(record)
    if graph is None:
        return record, {
            "dropped_relations": [],
            "dropped_entities": [],
            "removed_legacy_fields": removed_legacy_fields,
            "retained_relation_count": 0,
            "retained_entity_count": 0,
        }
    source_text = str(record.get("input") or "")
    relation_specs = schema["relation_types"]
    retained_relations: list[dict[str, Any]] = []
    dropped_relations: list[dict[str, Any]] = []
    raw_relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
    for relation_index, relation in enumerate(raw_relations):
        reason = ""
        if not isinstance(relation, dict):
            reason = "relation_not_object"
        else:
            spec = relation_specs.get(relation.get("relation"))
            if spec is None:
                reason = "undefined_relation"
            elif spec.get("status") != "active":
                reason = "relation_requires_medical_review"
            elif not span_is_exact(
                relation.get("evidence_span"), source_text, relation.get("evidence")
            ):
                reason = "relation_has_no_verified_evidence_span"
        if reason:
            dropped_relations.append(
                {
                    "relation_index": relation_index,
                    "reason": reason,
                    "relation": copy.deepcopy(relation),
                }
            )
        else:
            retained_relations.append(relation)
    graph["relations"] = retained_relations

    relation_entity_ids = {
        str(relation.get(endpoint))
        for relation in retained_relations
        for endpoint in ("source", "target")
    }
    retained_entities: list[dict[str, Any]] = []
    dropped_entities: list[dict[str, Any]] = []
    raw_entities = graph.get("entities") if isinstance(graph.get("entities"), list) else []
    for entity_index, entity in enumerate(raw_entities):
        if not isinstance(entity, dict):
            dropped_entities.append(
                {
                    "entity_index": entity_index,
                    "reason": "entity_not_object",
                    "entity": copy.deepcopy(entity),
                }
            )
            continue
        entity_id = str(entity.get("id") or "")
        is_main = (
            entity.get("label") == "Disease"
            and normalize_text(entity.get("name"))
            == normalize_text(record.get("source_title"))
        )
        entity_name = str(entity.get("name") or "")
        name_span = locate_evidence_span(source_text, entity_name)
        # ``locate_evidence_span`` also supports punctuation-normalized evidence
        # repair.  That is useful for relation review, but an orphan entity must
        # be stricter: its emitted name itself must occur in the source (apart
        # from case/whitespace normalization), otherwise it is not a safe target.
        name_is_grounded = bool(
            name_span
            and normalize_text(name_span.text) == normalize_text(entity_name)
        )
        if is_main or name_is_grounded or entity_id in relation_entity_ids:
            retained_entities.append(entity)
        else:
            dropped_entities.append(
                {
                    "entity_index": entity_index,
                    "entity_id": entity_id,
                    "label": entity.get("label"),
                    "name": entity.get("name"),
                    "reason": "not_canonical_not_text_grounded_and_no_grounded_relation",
                }
            )
    graph["entities"] = retained_entities

    validation = validate_record(record, schema)
    errors = list(validation["errors"])
    icd_validation = record.get("icd_validation")
    if not isinstance(icd_validation, dict) or icd_validation.get("status") != "verified":
        errors.append(
            {
                "code": "canonical_icd_not_verified",
                "details": (
                    icd_validation.get("errors")
                    if isinstance(icd_validation, dict)
                    else "icd_validation is missing"
                ),
            }
        )
    migration = record.get("migration")
    if not isinstance(migration, dict):
        migration = {}
        record["migration"] = migration
    unresolved_codes = migration.get("unverified_codes") or []
    migration["errors"] = errors
    migration["warnings"] = list(validation["warnings"])
    migration["status"] = (
        "invalid"
        if errors
        else "manual_review"
        if validation["warnings"] or unresolved_codes
        else "repaired"
    )
    audit = {
        "dropped_relations": dropped_relations,
        "dropped_entities": dropped_entities,
        "removed_legacy_fields": removed_legacy_fields,
        "retained_relation_count": len(retained_relations),
        "retained_entity_count": len(retained_entities),
        "validation": validation,
    }
    record["training_sanitization"] = audit
    return record, audit


def sanitize_dataset_for_training(
    records: Iterable[dict[str, Any]], schema: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    totals: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    findings = []
    for record_index, record in enumerate(records):
        cleaned, audit = sanitize_record_for_training(record, schema)
        output.append(cleaned)
        totals["retained_relations"] += audit["retained_relation_count"]
        totals["retained_entities"] += audit["retained_entity_count"]
        totals["dropped_relations"] += len(audit["dropped_relations"])
        totals["dropped_entities"] += len(audit["dropped_entities"])
        totals["removed_legacy_fields"] += len(audit["removed_legacy_fields"])
        reason_counts = Counter(
            item["reason"] for item in audit["dropped_relations"]
        )
        for reason, count in reason_counts.items():
            totals[f"dropped_relation_reason:{reason}"] += count
        status = str((cleaned.get("migration") or {}).get("status") or "missing")
        statuses[status] += 1
        if status != "repaired":
            findings.append(
                {
                    "record_index": record_index,
                    "source_record_id": cleaned.get("source_record_id"),
                    "source_code": cleaned.get("source_code"),
                    "status": status,
                    "errors": (cleaned.get("migration") or {}).get("errors", []),
                    "warnings": (cleaned.get("migration") or {}).get("warnings", []),
                    "unverified_codes": (cleaned.get("migration") or {}).get(
                        "unverified_codes", []
                    ),
                }
            )
    report = {
        "schema_version": schema.get("schema_version"),
        "record_count": len(output),
        "status_counts": dict(sorted(statuses.items())),
        "totals": dict(sorted(totals.items())),
        "records_not_repaired": len(findings),
        "findings": findings,
    }
    return output, report

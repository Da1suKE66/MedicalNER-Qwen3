"""Compile two independent semantic audits into an atomic Schema v2 patch."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from .schema_v2 import _matches_direction, graph_from_record, normalize_text, validate_record
from .semantic_audit import (
    SemanticAuditResponse,
    SemanticOperation,
    build_semantic_audit_task,
    canonical_operation_signature,
    require_valid_semantic_audit_response,
    response_semantic_signature,
    response_sha256,
    sha256_value,
)


SEMANTIC_PATCH_PROTOCOL_VERSION = "semantic-consensus-patch-v1"


class SemanticPatchError(ValueError):
    """Raised when consensus or deterministic patch invariants are not satisfied."""


_ENTITY_ID_PREFIX = {
    "Disease": "D",
    "Symptom": "S",
    "Diagnostic Criteria": "DC",
    "Interview Tool": "IT",
    "Patient Information": "PI",
    "Medication": "M",
    "Communication Method": "CM",
    "Risk Information": "R",
}


def _operation_payload(operation: SemanticOperation) -> dict[str, Any]:
    payload = operation.model_dump(mode="json")
    payload["canonical_signature"] = canonical_operation_signature(operation)
    return payload


def compile_consensus_patch(
    *,
    record: dict[str, Any],
    schema: dict[str, Any],
    dataset_sha256: str,
    primary: SemanticAuditResponse,
    review: SemanticAuditResponse,
    minimum_confidence: float = 0.95,
) -> dict[str, Any]:
    """Compile only exact primary/review consensus into a hash-bound patch."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise SemanticPatchError("minimum_confidence must be in [0,1]")
    primary_task = build_semantic_audit_task(
        record, schema, dataset_sha256=dataset_sha256, phase="primary"
    )
    review_task = build_semantic_audit_task(
        record, schema, dataset_sha256=dataset_sha256, phase="blind_review"
    )
    require_valid_semantic_audit_response(primary, primary_task)
    require_valid_semantic_audit_response(review, review_task)
    if primary.unresolved or review.unresolved:
        raise SemanticPatchError("an audit contains unresolved findings")
    if response_semantic_signature(primary) != response_semantic_signature(review):
        raise SemanticPatchError("primary and blind-review semantic decisions conflict")
    if len(primary.proposed_operations) != len(review.proposed_operations):
        raise SemanticPatchError("primary and review operation counts conflict")
    review_by_signature = {
        canonical_operation_signature(operation): operation
        for operation in review.proposed_operations
    }
    if len(review_by_signature) != len(review.proposed_operations):
        raise SemanticPatchError("review contains duplicate canonical operations")
    operation_rows: list[dict[str, Any]] = []
    for operation in primary.proposed_operations:
        signature = canonical_operation_signature(operation)
        reviewed = review_by_signature.get(signature)
        if reviewed is None:
            raise SemanticPatchError("primary operation lacks exact blind-review consensus")
        if operation.confidence < minimum_confidence or reviewed.confidence < minimum_confidence:
            raise SemanticPatchError(
                f"operation {operation.operation_ref} is below the consensus confidence gate"
            )
        row = _operation_payload(operation)
        row["primary_confidence"] = operation.confidence
        row["review_confidence"] = reviewed.confidence
        row["review_operation_ref"] = reviewed.operation_ref
        operation_rows.append(row)
    operation_rows.sort(key=lambda item: (item["canonical_signature"], item["operation_ref"]))
    patch = {
        "patch_protocol_version": SEMANTIC_PATCH_PROTOCOL_VERSION,
        "source_record_id": primary_task["source_record_id"],
        "dataset_sha256": dataset_sha256,
        "record_sha256": primary_task["record_sha256"],
        "schema_sha256": primary_task["schema_sha256"],
        "prompt_sha256": primary_task["prompt_sha256"],
        "primary_task_sha256": primary_task["task_sha256"],
        "review_task_sha256": review_task["task_sha256"],
        "primary_response_sha256": response_sha256(primary),
        "review_response_sha256": response_sha256(review),
        "main_disease_entity_key": primary_task["main_disease_entity_key"],
        "entity_preconditions": {
            item["entity_key"]: item["entity_sha256"]
            for item in primary_task["entity_inventory"]
        },
        "relation_preconditions": {
            item["relation_key"]: item["relation_sha256"]
            for item in primary_task["relation_inventory"]
        },
        "operations": operation_rows,
    }
    patch["patch_sha256"] = sha256_value(patch)
    return patch


def _assert_patch_identity(
    patch: dict[str, Any],
    record: dict[str, Any],
    schema: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    if patch.get("patch_protocol_version") != SEMANTIC_PATCH_PROTOCOL_VERSION:
        raise SemanticPatchError("unsupported semantic patch protocol")
    primary_task = build_semantic_audit_task(
        record, schema, dataset_sha256=dataset_sha256, phase="primary"
    )
    expected = {
        "source_record_id": primary_task["source_record_id"],
        "dataset_sha256": dataset_sha256,
        "record_sha256": primary_task["record_sha256"],
        "schema_sha256": primary_task["schema_sha256"],
        "prompt_sha256": primary_task["prompt_sha256"],
        "primary_task_sha256": primary_task["task_sha256"],
        "main_disease_entity_key": primary_task["main_disease_entity_key"],
    }
    review_task = build_semantic_audit_task(
        record, schema, dataset_sha256=dataset_sha256, phase="blind_review"
    )
    expected["review_task_sha256"] = review_task["task_sha256"]
    for key, value in expected.items():
        if patch.get(key) != value:
            raise SemanticPatchError(f"patch {key} precondition mismatch")
    payload = {key: value for key, value in patch.items() if key != "patch_sha256"}
    if patch.get("patch_sha256") != sha256_value(payload):
        raise SemanticPatchError("patch_sha256 mismatch")
    if patch.get("entity_preconditions") != {
        item["entity_key"]: item["entity_sha256"]
        for item in primary_task["entity_inventory"]
    }:
        raise SemanticPatchError("entity snapshot preconditions mismatch")
    if patch.get("relation_preconditions") != {
        item["relation_key"]: item["relation_sha256"]
        for item in primary_task["relation_inventory"]
    }:
        raise SemanticPatchError("relation snapshot preconditions mismatch")
    entity_mutations: Counter[str] = Counter()
    relation_mutations: Counter[str] = Counter()
    for operation in patch.get("operations") or []:
        if not isinstance(operation, dict):
            raise SemanticPatchError("patch operation is not an object")
        op = operation.get("op")
        entity_key = operation.get("entity_key")
        relation_key = operation.get("relation_key")
        if op in {
            "update_entity",
            "remove_entity",
            "collapse_symptom_into_manifestations",
        } and entity_key:
            entity_mutations[str(entity_key)] += 1
        if op in {"replace_relation", "remove_relation", "set_relation_evidence"} and relation_key:
            relation_mutations[str(relation_key)] += 1
        if (
            op == "collapse_symptom_into_manifestations"
            and entity_key == operation.get("parent_entity_key")
        ):
            raise SemanticPatchError("patch cannot collapse an entity into itself")
    duplicated_entities = sorted(
        key for key, count in entity_mutations.items() if count > 1
    )
    duplicated_relations = sorted(
        key for key, count in relation_mutations.items() if count > 1
    )
    if duplicated_entities:
        raise SemanticPatchError(
            f"patch contains conflicting entity mutations: {duplicated_entities}"
        )
    if duplicated_relations:
        raise SemanticPatchError(
            f"patch contains conflicting relation mutations: {duplicated_relations}"
        )
    return primary_task


def _new_entity_id(
    operation: dict[str, Any], label: str, existing_ids: set[str]
) -> str:
    prefix = _ENTITY_ID_PREFIX[label]
    digest = str(operation["canonical_signature"])[:10].upper()
    candidate = f"{prefix}_AUD_{digest}"
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{prefix}_AUD_{digest}_{suffix}"
        suffix += 1
    existing_ids.add(candidate)
    return candidate


def _exact_span(operation: dict[str, Any], source_text: str) -> dict[str, Any]:
    span = operation.get("evidence")
    if not isinstance(span, dict):
        raise SemanticPatchError(f"{operation.get('operation_ref')}: evidence is missing")
    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(text, str)
        or start < 0
        or end <= start
        or source_text[start:end] != text
    ):
        raise SemanticPatchError(
            f"{operation.get('operation_ref')}: evidence is not an exact record.input span"
        )
    return {
        "basis": "record.input",
        "text": text,
        "start": start,
        "end": end,
    }


def _relation_from_operation(
    operation: dict[str, Any],
    ref_to_id: dict[str, str],
    source_text: str,
) -> dict[str, Any]:
    source_ref = operation.get("source_ref")
    target_ref = operation.get("target_ref")
    if source_ref not in ref_to_id or target_ref not in ref_to_id:
        raise SemanticPatchError(
            f"{operation.get('operation_ref')}: relation endpoint ref is unavailable"
        )
    span = _exact_span(operation, source_text)
    return {
        "source": ref_to_id[source_ref],
        "target": ref_to_id[target_ref],
        "relation": operation["replacement_relation"],
        "evidence": span["text"],
        "evidence_span": span,
    }


def _check_properties_for_label(
    entity: dict[str, Any], schema: dict[str, Any]
) -> None:
    label = str(entity.get("label") or "")
    spec = schema.get("entity_types", {}).get(label)
    if not isinstance(spec, dict):
        raise SemanticPatchError(f"entity label is outside schema: {label}")
    properties = entity.get("properties")
    if not isinstance(properties, dict):
        raise SemanticPatchError(f"entity {entity.get('id')} properties is not an object")
    invalid = set(properties) - set(spec.get("properties", []))
    if invalid:
        raise SemanticPatchError(
            f"entity {entity.get('id')} has properties invalid for {label}: {sorted(invalid)}"
        )


def strict_semantic_record_errors(
    record: dict[str, Any], schema: dict[str, Any], original: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    for key in (
        "source_record_id",
        "source_code",
        "source_title",
        "source_release",
        "schema_version",
    ):
        if record.get(key) != original.get(key):
            errors.append(f"protected record field changed: {key}")
    graph = graph_from_record(record)
    original_graph = graph_from_record(original)
    if graph is None or original_graph is None:
        return errors + ["record graph is missing"]
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return errors + ["entities/relations must be lists"]
    entity_ids = [
        str(entity.get("id") or "") for entity in entities if isinstance(entity, dict)
    ]
    if len(entity_ids) != len(entities) or not all(entity_ids):
        errors.append("every entity must be an object with a non-empty id")
    if len(entity_ids) != len(set(entity_ids)):
        errors.append("duplicate entity IDs remain")
    by_id = {
        str(entity.get("id")): entity
        for entity in entities
        if isinstance(entity, dict) and entity.get("id")
    }
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        try:
            _check_properties_for_label(entity, schema)
        except SemanticPatchError as exc:
            errors.append(str(exc))
    source_text = str(record.get("input") or "")
    duplicate_relations: Counter[str] = Counter()
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"relation[{index}] is not an object")
            continue
        if relation.get("source") not in by_id or relation.get("target") not in by_id:
            errors.append(f"relation[{index}] has a ghost endpoint")
        name = str(relation.get("relation") or "")
        spec = schema.get("relation_types", {}).get(name)
        if not isinstance(spec, dict) or spec.get("status") != "active":
            errors.append(f"relation[{index}] is not active: {name}")
        elif relation.get("source") in by_id and relation.get("target") in by_id:
            if not _matches_direction(
                spec,
                str(by_id[relation["source"]].get("label") or ""),
                str(by_id[relation["target"]].get("label") or ""),
            ):
                errors.append(f"relation[{index}] has invalid domain/range")
        if "relation_name" in relation or "relation_type" in relation:
            errors.append(f"relation[{index}] retains legacy relation fields")
        span = relation.get("evidence_span")
        if not isinstance(span, dict):
            errors.append(f"relation[{index}] lacks evidence_span")
        else:
            start = span.get("start")
            end = span.get("end")
            text = span.get("text")
            if (
                span.get("basis") != "record.input"
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not isinstance(text, str)
                or start < 0
                or end <= start
                or source_text[start:end] != text
                or relation.get("evidence") != text
            ):
                errors.append(f"relation[{index}] evidence is not exact")
        duplicate_relations[
            json.dumps(
                {
                    "source": relation.get("source"),
                    "target": relation.get("target"),
                    "relation": relation.get("relation"),
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        ] += 1
    if any(count > 1 for count in duplicate_relations.values()):
        errors.append("duplicate canonical relations remain")

    source_title = normalize_text(record.get("source_title"))
    source_id = str(record.get("source_record_id") or "")
    main = [
        entity
        for entity in entities
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == source_title
        and str((entity.get("properties") or {}).get("icd_uri") or "") == source_id
    ]
    original_main = [
        entity
        for entity in original_graph.get("entities", [])
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == source_title
        and str((entity.get("properties") or {}).get("icd_uri") or "") == source_id
    ]
    if len(main) != 1 or len(original_main) != 1 or main[0] != original_main[0]:
        errors.append("canonical main Disease or ICD provenance changed")
    validation = validate_record(record, schema)
    for error in validation.get("errors", []):
        errors.append(f"schema:{error.get('code')}:{error}")
    try:
        json.dumps(record, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"record is not strict JSON: {exc}")
    return errors


def apply_consensus_patch(
    *,
    record: dict[str, Any],
    schema: dict[str, Any],
    dataset_sha256: str,
    patch: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one compiled patch transactionally and enforce the strict final gate."""

    task = _assert_patch_identity(patch, record, schema, dataset_sha256)
    repaired = copy.deepcopy(record)
    graph = graph_from_record(repaired)
    if graph is None:
        raise SemanticPatchError("record graph is missing")
    entities = graph["entities"]
    relations = graph["relations"]
    entity_items = task["entity_inventory"]
    relation_items = task["relation_inventory"]
    entity_key_to_id = {item["entity_key"]: item["id"] for item in entity_items}
    relation_key_to_object = {
        item["relation_key"]: relations[index]
        for index, item in enumerate(relation_items)
    }
    relation_key_to_index = {
        item["relation_key"]: index for index, item in enumerate(relation_items)
    }
    by_id = {str(entity["id"]): entity for entity in entities}
    original_by_id = copy.deepcopy(by_id)
    existing_ids = set(by_id)
    ref_to_id = dict(entity_key_to_id)
    operation_rows = list(patch.get("operations") or [])
    new_entity_ids: dict[str, str] = {}

    for operation in operation_rows:
        if operation.get("op") != "add_entity":
            continue
        label = str(operation.get("replacement_label") or "")
        name = str(operation.get("replacement_name") or "").strip()
        new_ref = str(operation.get("new_entity_ref") or "")
        if label not in schema["entity_types"] or not name or not new_ref:
            raise SemanticPatchError("add_entity has invalid label/name/ref")
        entity_id = _new_entity_id(operation, label, existing_ids)
        entity = {"id": entity_id, "label": label, "name": name, "properties": {}}
        entities.append(entity)
        by_id[entity_id] = entity
        ref_to_id[new_ref] = entity_id
        new_entity_ids[new_ref] = entity_id

    for operation in operation_rows:
        if operation.get("op") != "update_entity":
            continue
        entity_id = entity_key_to_id[str(operation.get("entity_key"))]
        entity = by_id[entity_id]
        if operation.get("replacement_label"):
            entity["label"] = operation["replacement_label"]
        if operation.get("replacement_name"):
            entity["name"] = str(operation["replacement_name"]).strip()
        if operation.get("replacement_properties") is not None:
            replacement_properties = operation["replacement_properties"]
            if not isinstance(replacement_properties, dict):
                raise SemanticPatchError(
                    f"update_entity replacement_properties is not an object: {entity_id}"
                )
            entity["properties"] = copy.deepcopy(replacement_properties)
        _check_properties_for_label(entity, schema)

    remove_relation_keys = {
        str(operation.get("relation_key"))
        for operation in operation_rows
        if operation.get("op") in {"remove_relation", "replace_relation"}
    }
    remove_entity_ids: set[str] = set()
    collapsed: list[dict[str, Any]] = []
    for operation in operation_rows:
        if operation.get("op") != "collapse_symptom_into_manifestations":
            continue
        child_key = str(operation.get("entity_key"))
        parent_key = str(operation.get("parent_entity_key"))
        child_id = entity_key_to_id[child_key]
        parent_id = entity_key_to_id[parent_key]
        child = by_id[child_id]
        parent = by_id[parent_id]
        if child.get("label") != "Symptom" or parent.get("label") != "Symptom":
            raise SemanticPatchError("collapse endpoints must remain Symptom")
        child_properties = child.get("properties") or {}
        if any(value not in (None, "", []) for value in child_properties.values()):
            raise SemanticPatchError(
                f"collapse would lose non-empty properties from {child_id}"
            )
        manifestation = str(operation.get("manifestation_text") or "").strip()
        if normalize_text(manifestation) != normalize_text(child.get("name")):
            raise SemanticPatchError("collapse manifestation must equal the child name")
        equivalent_relation_keys: set[str] = set()
        for relation_key, relation in relation_key_to_object.items():
            if child_id not in {relation.get("source"), relation.get("target")}:
                continue
            if relation_key in remove_relation_keys:
                continue
            replaced_source = parent_id if relation.get("source") == child_id else relation.get("source")
            replaced_target = parent_id if relation.get("target") == child_id else relation.get("target")
            if any(
                other is not relation
                and other.get("source") == replaced_source
                and other.get("target") == replaced_target
                and other.get("relation") == relation.get("relation")
                for other in relations
            ):
                equivalent_relation_keys.add(relation_key)
            else:
                raise SemanticPatchError(
                    f"collapse leaves an unhandled incident relation: {relation_key}"
                )
        remove_relation_keys.update(equivalent_relation_keys)
        properties = parent.setdefault("properties", {})
        manifestations = properties.setdefault("manifestations", [])
        if not isinstance(manifestations, list):
            raise SemanticPatchError("parent manifestations is not a list")
        if manifestation not in manifestations:
            manifestations.append(manifestation)
        remove_entity_ids.add(child_id)
        collapsed.append(
            {
                "child_id": child_id,
                "parent_id": parent_id,
                "manifestation": manifestation,
            }
        )

    for operation in operation_rows:
        if operation.get("op") != "remove_entity":
            continue
        entity_id = entity_key_to_id[str(operation.get("entity_key"))]
        for relation_key, relation in relation_key_to_object.items():
            if entity_id in {relation.get("source"), relation.get("target")} and relation_key not in remove_relation_keys:
                raise SemanticPatchError(
                    f"remove_entity leaves an unhandled incident relation: {relation_key}"
                )
        remove_entity_ids.add(entity_id)

    replaced_by_key: dict[str, dict[str, Any]] = {}
    added_relations: list[dict[str, Any]] = []
    for operation in operation_rows:
        op = operation.get("op")
        if op == "replace_relation":
            replaced_by_key[str(operation["relation_key"])] = _relation_from_operation(
                operation, ref_to_id, str(repaired.get("input") or "")
            )
        elif op == "set_relation_evidence":
            relation_key = str(operation["relation_key"])
            updated = {
                key: value
                for key, value in relation_key_to_object[relation_key].items()
                if key
                not in {
                    "relation_name",
                    "relation_type",
                    "evidence_original",
                    "evidence_repair",
                }
            }
            span = _exact_span(operation, str(repaired.get("input") or ""))
            updated["evidence"] = span["text"]
            updated["evidence_span"] = span
            replaced_by_key[relation_key] = updated
        elif op == "add_relation":
            added_relations.append(
                _relation_from_operation(
                    operation, ref_to_id, str(repaired.get("input") or "")
                )
            )

    final_relations: list[dict[str, Any]] = []
    for relation_key, index in sorted(relation_key_to_index.items(), key=lambda item: item[1]):
        if relation_key in remove_relation_keys and relation_key not in replaced_by_key:
            continue
        relation = replaced_by_key.get(relation_key, relations[index])
        cleaned = {
            key: value
            for key, value in relation.items()
            if key not in {"relation_name", "relation_type"}
        }
        final_relations.append(cleaned)
    final_relations.extend(added_relations)
    graph["relations"] = final_relations
    graph["entities"] = [
        entity for entity in entities if str(entity.get("id")) not in remove_entity_ids
    ]

    errors = strict_semantic_record_errors(repaired, schema, record)
    if errors:
        raise SemanticPatchError("; ".join(errors))
    audit = {
        "patch_sha256": patch["patch_sha256"],
        "operations_applied": len(operation_rows),
        "new_entity_ids": new_entity_ids,
        "removed_entity_ids": sorted(remove_entity_ids),
        "collapsed": collapsed,
        "before_record_sha256": sha256_value(record),
        "after_record_sha256": sha256_value(repaired),
    }
    repaired.setdefault("semantic_audit", {})
    repaired["semantic_audit"] = {
        "protocol_version": SEMANTIC_PATCH_PROTOCOL_VERSION,
        "patch_sha256": patch["patch_sha256"],
        "primary_response_sha256": patch["primary_response_sha256"],
        "review_response_sha256": patch["review_response_sha256"],
    }
    # Metadata was added after graph validation; strict JSON is checked once more.
    json.dumps(repaired, ensure_ascii=False, allow_nan=False)
    audit["after_record_sha256"] = sha256_value(repaired)
    del original_by_id
    return repaired, audit

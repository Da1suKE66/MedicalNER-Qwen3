"""Full-coverage, hash-bound semantic auditing for Schema v2 graphs.

The audit protocol is deliberately separate from :mod:`deepseek_repair`.  A
repair-only response cannot prove that every entity, relation, and high-risk
semantic cue was inspected.  This module builds immutable per-record tasks,
requires exact-set coverage in the response, and describes mutations with
stable object fingerprints instead of array indexes.

No function in this module writes a repaired dataset.  Model proposals must be
independently reviewed and compiled into a deterministic patch before they can
be applied.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema_v2 import _matches_direction, graph_from_record, normalize_text


SEMANTIC_AUDIT_PROTOCOL_VERSION = "deepseek-semantic-audit-v1"

REQUIRED_DIMENSIONS = (
    "disease_classification",
    "symptom_classification",
    "description_fragmentation",
    "entity_grounding_and_boundary",
    "relation_type",
    "relation_direction_and_endpoints",
    "patient_information_links",
    "exclusion_and_differential_relations",
    "somatic_causality",
    "missing_entities_and_relations",
    "data_quality",
)


class SemanticAuditValidationError(ValueError):
    """Raised when an audit response is incomplete or violates the frozen task."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def public_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in schema.items() if not str(key).startswith("_")}


def entity_inventory(graph: dict[str, Any]) -> list[dict[str, Any]]:
    raw = graph.get("entities")
    if not isinstance(raw, list):
        raise SemanticAuditValidationError("record entities is not a list")
    inventory: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entity in enumerate(raw):
        if not isinstance(entity, dict):
            raise SemanticAuditValidationError(f"entity[{index}] is not an object")
        entity_id = str(entity.get("id") or "")
        if not entity_id or entity_id in seen_ids:
            raise SemanticAuditValidationError("entity IDs must be non-empty and unique")
        seen_ids.add(entity_id)
        digest = sha256_value(entity)
        inventory.append(
            {
                "entity_key": f"E:{entity_id}:{digest[:16]}",
                "entity_sha256": digest,
                "id": entity_id,
                "label": str(entity.get("label") or ""),
                "name": str(entity.get("name") or ""),
                "properties": entity.get("properties")
                if isinstance(entity.get("properties"), dict)
                else {},
            }
        )
    return inventory


def relation_inventory(
    graph: dict[str, Any], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw = graph.get("relations")
    if not isinstance(raw, list):
        raise SemanticAuditValidationError("record relations is not a list")
    by_id = {item["id"]: item["entity_key"] for item in entities}
    duplicate_counts: Counter[str] = Counter()
    inventory: list[dict[str, Any]] = []
    for index, relation in enumerate(raw):
        if not isinstance(relation, dict):
            raise SemanticAuditValidationError(f"relation[{index}] is not an object")
        source_id = str(relation.get("source") or "")
        target_id = str(relation.get("target") or "")
        if source_id not in by_id or target_id not in by_id:
            raise SemanticAuditValidationError(
                f"relation[{index}] has an endpoint outside the entity inventory"
            )
        digest = sha256_value(relation)
        ordinal = duplicate_counts[digest]
        duplicate_counts[digest] += 1
        inventory.append(
            {
                "relation_key": f"R:{digest[:20]}:{ordinal}",
                "relation_sha256": digest,
                "source_entity_key": by_id[source_id],
                "target_entity_key": by_id[target_id],
                "source_id": source_id,
                "target_id": target_id,
                "relation": str(relation.get("relation") or ""),
                "evidence": relation.get("evidence"),
                "evidence_span": relation.get("evidence_span"),
                "legacy_relation_name": relation.get("relation_name"),
                "legacy_relation_type": relation.get("relation_type"),
            }
        )
    return inventory


_TEXT_CUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "exclusion": re.compile(
        r"\b(?:exclude(?:d|s|ing)?|rule(?:d)?\s+out|not\s+better\s+accounted\s+for|"
        r"not\s+attributable\s+to|absence\s+of)\b",
        re.IGNORECASE,
    ),
    "differential_diagnosis": re.compile(
        r"\b(?:differential\s+diagnosis|boundary\s+with|distinguish(?:ed)?\s+from|"
        r"differentiat(?:e|ed|ing)\s+from)\b",
        re.IGNORECASE,
    ),
    "somatic_cause": re.compile(
        r"\b(?:due\s+to|caus(?:e|ed|es|ing)|attributable\s+to|aetiolog(?:y|ical)|"
        r"etiolog(?:y|ical)|medical\s+condition|neurological\s+(?:condition|disease)|"
        r"brain\s+injury|infection|endocrine\s+(?:condition|disease))\b",
        re.IGNORECASE,
    ),
    "patient_information": re.compile(
        r"\b(?:male|males|female|females|men|women|child|children|adolescen\w*|"
        r"adult|adults|older\s+adult|pregnan\w*|age|sex|gender|patient\s+history)\b",
        re.IGNORECASE,
    ),
    "source_token_corruption": re.compile(
        r"\b(?:predominulltly|pregnullcy|maintenullce|malignullt|finullcial|"
        r"anullkastic|dominullt|consonullt|determinullt)\w*\b",
        re.IGNORECASE,
    ),
}

_LEGACY_RELATION_NAME_EQUIVALENTS: dict[str, set[str]] = {
    "subsumes": {"subsumes"},
    "differentiates_from": {"differentiates from"},
    "co_occurs_with_frequency": {"co occurrence frequency"},
    "associated_with_poor_prognosis_in": {"associated with poor prognosis"},
    "is_core_symptom_of": {"core symptom of", "is core symptom of"},
    "is_associated_symptom_of": {"associated symptom of", "is associated symptom of"},
    "required_for_diagnosis_of": {"required for diagnosis of"},
    "supports_subtyping_of": {"supports subtyping of"},
    "first_line_for": {"first line for"},
    "informed_by_patient_demographics": {"informed by patient demographics"},
    "affects_diagnosis_of": {"affects diagnosis of"},
    "must_be_ruled_out_for": {"must be ruled out for"},
    "excludes_diagnosis_of": {"excludes diagnosis of"},
    "somatic_cause_of": {"somatic cause of"},
    "precedes": {"precedes"},
    "follows": {"follows"},
    "modulated_by": {"modulated by"},
    "excludes_if_present": {"excludes if present"},
    "assesses_for": {"assesses for"},
    "recommended_for": {"recommended for"},
    "triggers_alert_when": {"triggers alert when"},
    "mediated_by": {"mediated by"},
}


def _normalized_legacy_name(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def _legacy_relation_name_conflicts(relation: dict[str, Any]) -> bool:
    legacy_name = relation.get("legacy_relation_name")
    if not legacy_name:
        return False
    allowed = _LEGACY_RELATION_NAME_EQUIVALENTS.get(str(relation.get("relation") or ""))
    return bool(allowed) and _normalized_legacy_name(legacy_name) not in allowed


def _cue_key(category: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"C:{category}:{start}:{end}:{digest}"


def _text_cues(source_text: str) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for category, pattern in _TEXT_CUE_PATTERNS.items():
        prior_windows: list[tuple[int, int]] = []
        for match in pattern.finditer(source_text):
            start = max(0, match.start() - 120)
            end = min(len(source_text), match.end() + 220)
            if any(start < previous_end and previous_start < end for previous_start, previous_end in prior_windows):
                continue
            text = source_text[start:end]
            prior_windows.append((start, end))
            cues.append(
                {
                    "cue_key": _cue_key(category, start, end, text),
                    "category": category,
                    "text": text,
                    "start": start,
                    "end": end,
                    "entity_keys": [],
                    "relation_keys": [],
                }
            )
            # Cues are a forced checklist, not a replacement for reading the full text.
            if len(prior_windows) >= 12:
                break
    return cues


def cue_inventory(
    source_text: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cues = _text_cues(source_text)
    cue_keys = {cue["cue_key"] for cue in cues}

    def add_structural(
        category: str,
        *,
        entity_keys: list[str] | None = None,
        relation_keys: list[str] | None = None,
        detail: str,
    ) -> None:
        signature = "|".join(
            [category, *(entity_keys or []), *(relation_keys or []), detail]
        )
        key = f"C:{category}:{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]}"
        if key in cue_keys:
            return
        cue_keys.add(key)
        cues.append(
            {
                "cue_key": key,
                "category": category,
                "text": detail,
                "start": None,
                "end": None,
                "entity_keys": entity_keys or [],
                "relation_keys": relation_keys or [],
            }
        )

    main_candidates = [
        item
        for item in entities
        if item["label"] == "Disease"
        and (item.get("properties") or {}).get("icd_uri")
    ]
    main_key = main_candidates[0]["entity_key"] if len(main_candidates) == 1 else None
    for entity in entities:
        label = entity["label"]
        name = entity["name"]
        if label == "Disease" and entity["entity_key"] != main_key:
            add_structural(
                "disease_candidate",
                entity_keys=[entity["entity_key"]],
                detail=f"Decide whether this is an independently diagnosable ICD-11-style Disease: {name}",
            )
        if label == "Symptom" and len(name) >= 80:
            add_structural(
                "long_symptom",
                entity_keys=[entity["entity_key"]],
                detail=f"Check whether this long phrase is a standalone symptom or descriptive text: {name}",
            )
        if label == "Patient Information":
            related = [
                relation["relation_key"]
                for relation in relations
                if entity["entity_key"]
                in {relation["source_entity_key"], relation["target_entity_key"]}
            ]
            add_structural(
                "patient_information",
                entity_keys=[entity["entity_key"]],
                relation_keys=related,
                detail=f"Verify that patient information has only semantically entailed links: {name}",
            )

    symptoms = [item for item in entities if item["label"] == "Symptom"]
    for parent in symptoms:
        description = str((parent.get("properties") or {}).get("description") or "")
        normalized_description = normalize_text(description)
        if not normalized_description:
            continue
        for child in symptoms:
            if child["entity_key"] == parent["entity_key"]:
                continue
            child_name = normalize_text(child["name"])
            if len(child_name) < 5 or child_name not in normalized_description:
                continue
            add_structural(
                "description_fragment_candidate",
                entity_keys=[parent["entity_key"], child["entity_key"]],
                detail=(
                    f"Check whether child '{child['name']}' is only a manifestation "
                    f"inside parent '{parent['name']}' rather than an independent node."
                ),
            )

    for relation in relations:
        relation_name = relation["relation"]
        if not isinstance(relation.get("evidence_span"), dict):
            add_structural(
                "missing_relation_evidence",
                relation_keys=[relation["relation_key"]],
                detail=f"This {relation_name} relation lacks an exact evidence span.",
            )
        if relation_name in {"excludes_if_present", "excludes_diagnosis_of", "must_be_ruled_out_for"}:
            add_structural(
                "exclusion",
                relation_keys=[relation["relation_key"]],
                detail=f"Verify exclusion semantics, target diagnosis, and direction for {relation_name}.",
            )
        if relation_name == "subsumes":
            source = next(item for item in entities if item["entity_key"] == relation["source_entity_key"])
            target = next(item for item in entities if item["entity_key"] == relation["target_entity_key"])
            if source["label"] == target["label"] == "Symptom":
                add_structural(
                    "symptom_hierarchy_direction",
                    entity_keys=[source["entity_key"], target["entity_key"]],
                    relation_keys=[relation["relation_key"]],
                    detail="Verify broader-to-narrower direction for this Symptom subsumes relation.",
                )
        if _legacy_relation_name_conflicts(relation):
            add_structural(
                "legacy_relation_conflict",
                relation_keys=[relation["relation_key"]],
                detail=(
                    "Legacy relation_name contradicts canonical relation; review the canonical "
                    "semantics. All legacy fields are removed deterministically after audit."
                ),
            )
    relation_triples: dict[tuple[str, str, str], list[str]] = {}
    for relation in relations:
        triple = (
            relation["source_entity_key"],
            relation["target_entity_key"],
            relation["relation"],
        )
        relation_triples.setdefault(triple, []).append(relation["relation_key"])
    for triple, keys in relation_triples.items():
        if len(keys) < 2:
            continue
        add_structural(
            "duplicate_relation_candidate",
            relation_keys=keys,
            detail=(
                "Multiple relations share the same source, target, and predicate; "
                f"retain at most one evidence-grounded edge: {triple[2]}."
            ),
        )
    return sorted(cues, key=lambda item: item["cue_key"])


def evidence_span_inventory(
    source_text: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    target_chunk_chars: int = 420,
) -> list[dict[str, Any]]:
    """Build exact, pre-indexed spans so the model never calculates offsets."""

    raw_boundaries = [0]
    for match in re.finditer(r"(?<=[.!?])(?:[ \t]+|\n+)|\n+", source_text):
        raw_boundaries.append(match.end())
    raw_boundaries.append(len(source_text))
    spans: set[tuple[int, int]] = set()

    def add_span(start: int, end: int) -> None:
        while start < end and source_text[start].isspace():
            start += 1
        while end > start and source_text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.add((start, end))

    for boundary_index in range(len(raw_boundaries) - 1):
        start = raw_boundaries[boundary_index]
        end = raw_boundaries[boundary_index + 1]
        while end - start > target_chunk_chars:
            preferred_end = min(end, start + target_chunk_chars)
            split = source_text.rfind(" ", start + target_chunk_chars // 2, preferred_end)
            if split <= start:
                split = preferred_end
            add_span(start, split)
            start = split
        add_span(start, end)

    # Entity names are the most useful minimal evidence spans for boundary and
    # label decisions.  Pre-index every literal occurrence (case-insensitive,
    # while preserving the original source slice) so the model never has to
    # count Unicode offsets itself.
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        pattern = re.escape(name)
        if name[0].isalnum() or name[0] == "_":
            pattern = rf"(?<!\w){pattern}"
        if name[-1].isalnum() or name[-1] == "_":
            pattern = rf"{pattern}(?!\w)"
        for occurrence_index, match in enumerate(
            re.finditer(pattern, source_text, re.IGNORECASE)
        ):
            add_span(match.start(), match.end())
            if occurrence_index >= 7:
                break

    for relation in relations:
        span = relation.get("evidence_span")
        if isinstance(span, dict) and isinstance(span.get("start"), int) and isinstance(span.get("end"), int):
            start = span["start"]
            end = span["end"]
            if 0 <= start < end <= len(source_text) and source_text[start:end] == span.get("text"):
                spans.add((start, end))
    for cue in cues:
        start = cue.get("start")
        end = cue.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(source_text):
            spans.add((start, end))

    inventory = []
    for start, end in sorted(spans):
        text = source_text[start:end]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        inventory.append(
            {
                "span_key": f"SPAN:{start}:{end}:{digest}",
                "basis": "record.input",
                "text": text,
                "start": start,
                "end": end,
            }
        )
    return inventory


def _main_disease_key(
    record: dict[str, Any], entities: list[dict[str, Any]]
) -> str:
    source_id = str(record.get("source_record_id") or "")
    source_title = normalize_text(record.get("source_title"))
    matches = [
        item
        for item in entities
        if item["label"] == "Disease"
        and normalize_text(item["name"]) == source_title
        and str((item.get("properties") or {}).get("icd_uri") or "") == source_id
    ]
    if len(matches) != 1:
        raise SemanticAuditValidationError(
            "record must have exactly one canonical, WHO-linked main Disease"
        )
    return str(matches[0]["entity_key"])


def classify_risk(
    record: dict[str, Any],
    schema: dict[str, Any],
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    cues: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: set[str] = set()
    migration = record.get("migration") if isinstance(record.get("migration"), dict) else {}
    if migration.get("status") == "manual_review" or migration.get("errors"):
        reasons.add("migration_manual_review")
    if migration.get("discarded_non_main_codes"):
        for discarded in migration["discarded_non_main_codes"]:
            if isinstance(discarded, dict) and discarded.get("action") not in {
                "not_attached_main_disease_only_schema"
            }:
                reasons.add("non_main_disease_icd_mismatch")
                break
    for relation in relations:
        spec = schema.get("relation_types", {}).get(relation["relation"], {})
        if spec.get("status") != "active":
            reasons.add("review_status_relation")
        if not isinstance(relation.get("evidence_span"), dict):
            reasons.add("missing_relation_evidence")
    categories = {cue["category"] for cue in cues}
    if categories & {
        "description_fragment_candidate",
        "duplicate_relation_candidate",
        "long_symptom",
        "symptom_hierarchy_direction",
    }:
        reasons.add("entity_boundary_candidate")
    if len(entities) > 1 and not relations:
        reasons.add("nontrivial_zero_relation_graph")
    if reasons:
        return "high", sorted(reasons)
    non_main_diseases = [
        item
        for item in entities
        if item["label"] == "Disease"
        and not (item.get("properties") or {}).get("icd_uri")
    ]
    if non_main_diseases:
        return "medium", ["unlinked_non_main_disease"]
    return "low", []


def build_semantic_audit_task(
    record: dict[str, Any],
    schema: dict[str, Any],
    *,
    dataset_sha256: str,
    phase: Literal["primary", "blind_review"],
) -> dict[str, Any]:
    graph = graph_from_record(record)
    if graph is None:
        raise SemanticAuditValidationError("record has no graph")
    source_record_id = str(record.get("source_record_id") or "")
    if not source_record_id:
        raise SemanticAuditValidationError("record has no source_record_id")
    source_text = record.get("input")
    if not isinstance(source_text, str):
        raise SemanticAuditValidationError("record input is not text")
    entities = entity_inventory(graph)
    relations = relation_inventory(graph, entities)
    cues = cue_inventory(source_text, entities, relations)
    evidence_spans = evidence_span_inventory(source_text, entities, relations, cues)
    schema_payload = public_schema(schema)
    risk_tier, risk_reasons = classify_risk(
        record, schema_payload, entities, relations, cues
    )
    task: dict[str, Any] = {
        "protocol_version": SEMANTIC_AUDIT_PROTOCOL_VERSION,
        "phase": phase,
        "dataset_sha256": dataset_sha256,
        "schema_sha256": sha256_value(schema_payload),
        "prompt_sha256": hashlib.sha256(
            SEMANTIC_AUDIT_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "source_record_id": source_record_id,
        "record_sha256": sha256_value(record),
        "schema_version": str(record.get("schema_version") or ""),
        "main_disease_entity_key": _main_disease_key(record, entities),
        "original_text": source_text,
        "entity_inventory": entities,
        "relation_inventory": relations,
        "cue_inventory": cues,
        "evidence_span_inventory": evidence_spans,
        "required_dimensions": list(REQUIRED_DIMENSIONS),
        "risk_tier": risk_tier,
        "risk_reasons": risk_reasons,
        "allowed_entity_types": schema_payload["entity_types"],
        "allowed_relation_types": schema_payload["relation_types"],
        "semantic_contract": {
            "disease": (
                "Use Disease only for an independently diagnosable disorder/category, "
                "not a descriptive phrase, mechanism, demographic group, or generic condition."
            ),
            "symptom": (
                "A Symptom node must be an independently meaningful clinical concept. "
                "Definition fragments or examples belong in description/manifestations."
            ),
            "subsumes": "broader concept -> narrower concept",
            "must_be_ruled_out_for": "alternative Disease -> diagnosis being considered",
            "excludes_diagnosis_of": (
                "exclusionary Diagnostic Criteria or Symptom -> diagnosis excluded"
            ),
            "somatic_cause_of": "somatic Disease -> psychiatric/behavioural target Disease",
            "affects_diagnosis_of": (
                "Patient Information -> Disease only when the text entails diagnostic impact; "
                "a demographic word alone is insufficient evidence"
            ),
            "who_boundary": (
                "Never invent or modify an ICD code. Mark needs_who_lookup; WHO API owns linking."
            ),
        },
    }
    task["task_sha256"] = sha256_value(task)
    return task


class ExactSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: Literal["record.input"] = "record.input"
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_bounds(self) -> "ExactSpan":
        if self.end <= self.start:
            raise ValueError("span end must be greater than start")
        return self


EntityVerdict = Literal[
    "correct",
    "disease_not_diagnosable",
    "disease_should_be_symptom",
    "symptom_should_be_disease",
    "wrong_label",
    "description_fragment",
    "wrong_boundary",
    "duplicate",
    "unsupported",
    "uncertain",
]


class EntityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_key: str
    verdict: EntityVerdict
    recommended_label: str | None = None
    parent_entity_key: str | None = None
    duplicate_of_entity_key: str | None = None
    evidence: list[ExactSpan] = Field(default_factory=list, max_length=8)
    needs_who_lookup: bool = False
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class RelationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_key: str
    semantic_verdict: Literal[
        "correct",
        "wrong_type",
        "wrong_direction",
        "wrong_endpoint",
        "unsupported",
        "duplicate",
        "uncertain",
    ]
    evidence_verdict: Literal["exact", "repairable", "unsupported", "uncertain"]
    proposed_source_ref: str | None = None
    proposed_target_ref: str | None = None
    proposed_relation: str | None = None
    replacement_evidence: ExactSpan | None = None
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class CueAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cue_key: str
    verdict: Literal[
        "represented_correctly", "missing_graph_fact", "not_graph_fact", "uncertain"
    ]
    represented_by_relation_keys: list[str] = Field(default_factory=list)
    proposed_operation_refs: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)


class DimensionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Literal[
        "disease_classification",
        "symptom_classification",
        "description_fragmentation",
        "entity_grounding_and_boundary",
        "relation_type",
        "relation_direction_and_endpoints",
        "patient_information_links",
        "exclusion_and_differential_relations",
        "somatic_causality",
        "missing_entities_and_relations",
        "data_quality",
    ]
    status: Literal["pass", "issues_found", "not_applicable", "uncertain"]
    linked_entity_keys: list[str] = Field(default_factory=list)
    linked_relation_keys: list[str] = Field(default_factory=list)
    linked_cue_keys: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=500)


SemanticOperationName = Literal[
    "add_entity",
    "update_entity",
    "remove_entity",
    "collapse_symptom_into_manifestations",
    "add_relation",
    "replace_relation",
    "remove_relation",
    "set_relation_evidence",
]


class SemanticOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_ref: str = Field(pattern=r"^op:[A-Za-z0-9_.-]+$")
    op: SemanticOperationName
    entity_key: str | None = None
    new_entity_ref: str | None = Field(default=None, pattern=r"^new:[A-Za-z0-9_.-]+$")
    replacement_label: str | None = None
    replacement_name: str | None = None
    replacement_properties: dict[str, Any] | None = None
    parent_entity_key: str | None = None
    manifestation_text: str | None = None
    relation_key: str | None = None
    source_ref: str | None = None
    target_ref: str | None = None
    replacement_relation: str | None = None
    evidence: ExactSpan | None = None
    reason: str = Field(min_length=1, max_length=600)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_operation_shape(self) -> "SemanticOperation":
        if self.evidence is None:
            raise ValueError(f"{self.op} requires exact record.input evidence")
        operation_fields = {
            "entity_key",
            "new_entity_ref",
            "replacement_label",
            "replacement_name",
            "replacement_properties",
            "parent_entity_key",
            "manifestation_text",
            "relation_key",
            "source_ref",
            "target_ref",
            "replacement_relation",
        }
        allowed_fields = {
            "add_entity": {"new_entity_ref", "replacement_label", "replacement_name"},
            "update_entity": {
                "entity_key",
                "replacement_label",
                "replacement_name",
                "replacement_properties",
            },
            "remove_entity": {"entity_key"},
            "collapse_symptom_into_manifestations": {
                "entity_key",
                "parent_entity_key",
                "manifestation_text",
            },
            "add_relation": {"source_ref", "target_ref", "replacement_relation"},
            "replace_relation": {
                "relation_key",
                "source_ref",
                "target_ref",
                "replacement_relation",
            },
            "remove_relation": {"relation_key"},
            "set_relation_evidence": {"relation_key"},
        }[self.op]
        unexpected = sorted(
            field_name
            for field_name in operation_fields - allowed_fields
            if getattr(self, field_name) is not None
        )
        if unexpected:
            raise ValueError(f"{self.op} has non-null unused fields: {unexpected}")
        if self.op == "add_entity":
            if not (self.new_entity_ref and self.replacement_label and self.replacement_name):
                raise ValueError("add_entity requires new_entity_ref, label, and name")
        elif self.op == "update_entity":
            if not self.entity_key or not (
                self.replacement_label
                or self.replacement_name
                or self.replacement_properties is not None
            ):
                raise ValueError("update_entity requires entity_key and a replacement")
        elif self.op == "remove_entity":
            if not self.entity_key:
                raise ValueError("remove_entity requires entity_key")
        elif self.op == "collapse_symptom_into_manifestations":
            if not (self.entity_key and self.parent_entity_key and self.manifestation_text):
                raise ValueError(
                    "collapse_symptom_into_manifestations requires child, parent, and text"
                )
        elif self.op == "add_relation":
            if not (self.source_ref and self.target_ref and self.replacement_relation):
                raise ValueError("add_relation requires source, target, and relation")
        elif self.op == "replace_relation":
            if not (
                self.relation_key
                and self.source_ref
                and self.target_ref
                and self.replacement_relation
            ):
                raise ValueError(
                    "replace_relation requires relation_key, endpoints, and relation"
                )
        elif self.op in {"remove_relation", "set_relation_evidence"}:
            if not self.relation_key:
                raise ValueError(f"{self.op} requires relation_key")
        return self


class UnresolvedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_ref: str
    category: Literal[
        "entity", "relation", "cue", "missing_fact", "schema_gap", "medical_review"
    ]
    object_keys: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=800)


class SemanticAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal["deepseek-semantic-audit-v1"]
    phase: Literal["primary", "blind_review"]
    dataset_sha256: str
    schema_sha256: str
    prompt_sha256: str
    task_sha256: str
    source_record_id: str
    record_sha256: str
    entity_assessments: list[EntityAssessment]
    relation_assessments: list[RelationAssessment]
    cue_assessments: list[CueAssessment]
    dimension_assessments: list[DimensionAssessment]
    proposed_operations: list[SemanticOperation] = Field(default_factory=list, max_length=100)
    unresolved: list[UnresolvedFinding] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)


def _exact_key_set(
    *, expected: set[str], actual: list[str], label: str
) -> list[str]:
    errors: list[str] = []
    counts = Counter(actual)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        errors.append(f"duplicate {label} keys: {duplicates}")
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unknown = sorted(actual_set - expected)
    if missing:
        errors.append(f"missing {label} keys: {missing}")
    if unknown:
        errors.append(f"unknown {label} keys: {unknown}")
    return errors


def _check_span(span: ExactSpan, source_text: str, context: str) -> str | None:
    if span.end > len(source_text):
        return f"{context}: evidence end is outside record.input"
    if source_text[span.start : span.end] != span.text:
        return f"{context}: evidence offsets do not match exact text"
    return None


def _span_signature(span: ExactSpan | dict[str, Any]) -> tuple[str, str, int, int]:
    if isinstance(span, ExactSpan):
        return (span.basis, span.text, span.start, span.end)
    return (
        str(span.get("basis") or ""),
        str(span.get("text") or ""),
        int(span.get("start", -1)),
        int(span.get("end", -1)),
    )


def _property_leaf_counter(value: Any) -> Counter[str]:
    """Compare property migrations without allowing content loss or invention."""

    leaves: Counter[str] = Counter()
    if isinstance(value, dict):
        for nested in value.values():
            leaves.update(_property_leaf_counter(nested))
    elif isinstance(value, list):
        for nested in value:
            leaves.update(_property_leaf_counter(nested))
    elif value not in (None, ""):
        leaves[canonical_json(value)] += 1
    return leaves


def validate_semantic_audit_response(
    response: SemanticAuditResponse,
    task: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for field in (
        "protocol_version",
        "phase",
        "dataset_sha256",
        "schema_sha256",
        "prompt_sha256",
        "task_sha256",
        "source_record_id",
        "record_sha256",
    ):
        if getattr(response, field) != task[field]:
            errors.append(f"{field} mismatch")

    entity_items = task["entity_inventory"]
    relation_items = task["relation_inventory"]
    cue_items = task["cue_inventory"]
    entity_keys = {item["entity_key"] for item in entity_items}
    relation_keys = {item["relation_key"] for item in relation_items}
    cue_keys = {item["cue_key"] for item in cue_items}
    errors.extend(
        _exact_key_set(
            expected=entity_keys,
            actual=[item.entity_key for item in response.entity_assessments],
            label="entity assessment",
        )
    )
    errors.extend(
        _exact_key_set(
            expected=relation_keys,
            actual=[item.relation_key for item in response.relation_assessments],
            label="relation assessment",
        )
    )
    errors.extend(
        _exact_key_set(
            expected=cue_keys,
            actual=[item.cue_key for item in response.cue_assessments],
            label="cue assessment",
        )
    )
    errors.extend(
        _exact_key_set(
            expected=set(REQUIRED_DIMENSIONS),
            actual=[item.dimension for item in response.dimension_assessments],
            label="dimension assessment",
        )
    )

    source_text = str(task["original_text"])
    allowed_labels = set(task["allowed_entity_types"])
    allowed_relations = task["allowed_relation_types"]
    allowed_evidence_spans = {
        _span_signature(item) for item in task["evidence_span_inventory"]
    }
    entity_by_key = {item["entity_key"]: item for item in entity_items}
    operation_refs = [operation.operation_ref for operation in response.proposed_operations]
    if len(operation_refs) != len(set(operation_refs)):
        errors.append("proposed operation_ref values must be unique")
    new_refs = [
        operation.new_entity_ref
        for operation in response.proposed_operations
        if operation.new_entity_ref
    ]
    if len(new_refs) != len(set(new_refs)):
        errors.append("new_entity_ref values must be unique")
    valid_refs = entity_keys | set(new_refs)
    label_by_ref = {item["entity_key"]: item["label"] for item in entity_items}
    for operation in response.proposed_operations:
        if operation.op == "add_entity" and operation.new_entity_ref:
            label_by_ref[operation.new_entity_ref] = str(operation.replacement_label or "")
        elif (
            operation.op == "update_entity"
            and operation.entity_key
            and operation.replacement_label
        ):
            label_by_ref[operation.entity_key] = operation.replacement_label
    unresolved_object_keys = {
        key for finding in response.unresolved for key in finding.object_keys
    }
    allowed_unresolved_keys = (
        entity_keys | relation_keys | cue_keys | set(REQUIRED_DIMENSIONS)
    )
    finding_refs = [finding.finding_ref for finding in response.unresolved]
    if len(finding_refs) != len(set(finding_refs)):
        errors.append("unresolved finding_ref values must be unique")
    for finding in response.unresolved:
        if not finding.object_keys:
            errors.append(f"unresolved {finding.finding_ref}: object_keys cannot be empty")
        unknown = set(finding.object_keys) - allowed_unresolved_keys
        if unknown:
            errors.append(
                f"unresolved {finding.finding_ref}: unknown object keys {sorted(unknown)}"
            )
    entity_operations: dict[str, list[SemanticOperation]] = {}
    relation_operations: dict[str, list[SemanticOperation]] = {}
    for operation in response.proposed_operations:
        if operation.entity_key:
            entity_operations.setdefault(operation.entity_key, []).append(operation)
        if operation.relation_key:
            relation_operations.setdefault(operation.relation_key, []).append(operation)

    for entity_key, operations in entity_operations.items():
        direct_mutations = [
            operation
            for operation in operations
            if operation.op
            in {
                "update_entity",
                "remove_entity",
                "collapse_symptom_into_manifestations",
            }
        ]
        if len(direct_mutations) > 1:
            errors.append(
                f"entity {entity_key}: multiple conflicting direct mutation operations"
            )
    for relation_key, operations in relation_operations.items():
        direct_mutations = [
            operation
            for operation in operations
            if operation.op
            in {
                "replace_relation",
                "remove_relation",
                "set_relation_evidence",
            }
        ]
        if len(direct_mutations) > 1:
            errors.append(
                f"relation {relation_key}: multiple conflicting direct mutation operations"
            )

    removed_entity_keys = {
        str(operation.entity_key)
        for operation in response.proposed_operations
        if operation.op in {"remove_entity", "collapse_symptom_into_manifestations"}
        and operation.entity_key
    }
    for relation in relation_items:
        relation_key = relation["relation_key"]
        direct_operations = relation_operations.get(relation_key, [])
        terminal_operations = [
            operation
            for operation in direct_operations
            if operation.op in {"remove_relation", "replace_relation"}
        ]
        incident_removed = removed_entity_keys & {
            relation["source_entity_key"],
            relation["target_entity_key"],
        }
        if (
            incident_removed
            and not terminal_operations
            and relation_key not in unresolved_object_keys
        ):
            errors.append(
                f"relation {relation_key}: removed/collapsed endpoint requires remove_relation, replace_relation, or unresolved"
            )
        for operation in terminal_operations:
            if operation.op == "replace_relation" and removed_entity_keys & {
                str(operation.source_ref),
                str(operation.target_ref),
            }:
                errors.append(
                    f"relation {relation_key}: replacement cannot retain a removed/collapsed endpoint"
                )

        source_key = relation["source_entity_key"]
        target_key = relation["target_entity_key"]
        source_label = label_by_ref[source_key]
        target_label = label_by_ref[target_key]
        endpoint_label_changed = (
            source_label != entity_by_key[source_key]["label"]
            or target_label != entity_by_key[target_key]["label"]
        )
        if (
            endpoint_label_changed
            and not terminal_operations
            and relation_key not in unresolved_object_keys
        ):
            spec = allowed_relations.get(relation["relation"], {})
            if not isinstance(spec, dict) or not _matches_direction(
                spec, source_label, target_label
            ):
                errors.append(
                    f"relation {relation_key}: entity label change requires relation replacement/removal or unresolved"
                )

    directly_mutated_entities = {
        entity_key
        for entity_key, operations in entity_operations.items()
        if any(
            operation.op
            in {
                "update_entity",
                "remove_entity",
                "collapse_symptom_into_manifestations",
            }
            for operation in operations
        )
    }
    for operation in response.proposed_operations:
        if (
            operation.op == "collapse_symptom_into_manifestations"
            and operation.parent_entity_key in directly_mutated_entities
        ):
            errors.append(
                f"{operation.operation_ref}: collapse parent cannot also receive a direct mutation"
            )

    for assessment in response.entity_assessments:
        for index, span in enumerate(assessment.evidence):
            error = _check_span(span, source_text, f"entity {assessment.entity_key}[{index}]")
            if error:
                errors.append(error)
            if _span_signature(span) not in allowed_evidence_spans:
                errors.append(
                    f"entity {assessment.entity_key}[{index}]: evidence is not copied from evidence_span_inventory"
                )
        if assessment.recommended_label and assessment.recommended_label not in allowed_labels:
            errors.append(
                f"entity {assessment.entity_key}: recommended label is outside schema"
            )
        for key in (assessment.parent_entity_key, assessment.duplicate_of_entity_key):
            if key and key not in entity_keys:
                errors.append(f"entity {assessment.entity_key}: unknown referenced entity {key}")
        if assessment.entity_key == task["main_disease_entity_key"] and assessment.verdict != "correct":
            errors.append("canonical main Disease cannot receive a mutation verdict")
        current = entity_by_key.get(assessment.entity_key, {})
        final_label = assessment.recommended_label or current.get("label")
        retained_as_disease = final_label == "Disease" and assessment.verdict not in {
            "disease_not_diagnosable",
            "disease_should_be_symptom",
            "description_fragment",
            "duplicate",
            "unsupported",
        }
        if assessment.entity_key == task["main_disease_entity_key"]:
            if assessment.needs_who_lookup:
                errors.append("canonical main Disease is already WHO-linked")
        elif retained_as_disease and not assessment.needs_who_lookup:
            errors.append(
                f"entity {assessment.entity_key}: retained non-main Disease requires WHO lookup"
            )
        direct_operations = [
            operation
            for operation in entity_operations.get(assessment.entity_key, [])
            if operation.op
            in {
                "update_entity",
                "remove_entity",
                "collapse_symptom_into_manifestations",
            }
        ]
        is_unresolved = assessment.entity_key in unresolved_object_keys
        if assessment.verdict == "correct":
            if any(
                value is not None
                for value in (
                    assessment.recommended_label,
                    assessment.parent_entity_key,
                    assessment.duplicate_of_entity_key,
                )
            ):
                errors.append(
                    f"entity {assessment.entity_key}: correct verdict requires null recommendation fields"
                )
            if direct_operations:
                errors.append(
                    f"entity {assessment.entity_key}: correct verdict cannot have a direct mutation"
                )
        elif assessment.verdict == "uncertain":
            if not is_unresolved:
                errors.append(
                    f"entity {assessment.entity_key}: uncertain verdict requires unresolved finding"
                )
            if direct_operations:
                errors.append(
                    f"entity {assessment.entity_key}: uncertain verdict cannot have a direct mutation"
                )
        elif not direct_operations and not is_unresolved:
            errors.append(
                f"entity {assessment.entity_key}: non-correct verdict lacks operation or unresolved finding"
            )

        compatible_ops: dict[str, set[str]] = {
            "disease_not_diagnosable": {"update_entity", "remove_entity"},
            "disease_should_be_symptom": {"update_entity"},
            "symptom_should_be_disease": {"update_entity"},
            "wrong_label": {"update_entity"},
            "description_fragment": {"collapse_symptom_into_manifestations"},
            "wrong_boundary": {"update_entity"},
            "duplicate": {"remove_entity"},
            "unsupported": {"remove_entity"},
        }
        expected_ops = compatible_ops.get(assessment.verdict, set())
        for operation in direct_operations:
            if operation.op not in expected_ops:
                errors.append(
                    f"entity {assessment.entity_key}: {operation.op} does not close verdict {assessment.verdict}"
                )
            if (
                operation.op == "update_entity"
                and operation.replacement_label is not None
                and operation.replacement_label != assessment.recommended_label
            ):
                errors.append(
                    f"entity {assessment.entity_key}: recommended_label disagrees with operation"
                )
            if (
                assessment.parent_entity_key is not None
                and operation.op == "collapse_symptom_into_manifestations"
                and operation.parent_entity_key != assessment.parent_entity_key
            ):
                errors.append(
                    f"entity {assessment.entity_key}: parent_entity_key disagrees with operation"
                )
            if operation.op == "update_entity":
                target_label = operation.replacement_label or str(
                    current.get("label") or ""
                )
                label_changes = target_label != current.get("label")
                replacement_properties = operation.replacement_properties
                if label_changes and replacement_properties is None:
                    errors.append(
                        f"entity {assessment.entity_key}: label change requires replacement_properties"
                    )
                if not label_changes and replacement_properties is not None:
                    errors.append(
                        f"entity {assessment.entity_key}: replacement_properties are allowed only for a label change"
                    )
                if replacement_properties is not None:
                    target_spec = task["allowed_entity_types"].get(
                        target_label, {}
                    )
                    allowed_property_names = set(target_spec.get("properties", []))
                    invalid_property_names = (
                        set(replacement_properties) - allowed_property_names
                    )
                    if invalid_property_names:
                        errors.append(
                            f"entity {assessment.entity_key}: replacement_properties are outside target label: {sorted(invalid_property_names)}"
                        )
                    if _property_leaf_counter(current.get("properties") or {}) != (
                        _property_leaf_counter(replacement_properties)
                    ):
                        errors.append(
                            f"entity {assessment.entity_key}: property migration must preserve every non-empty value exactly"
                        )
        if assessment.verdict == "disease_should_be_symptom" and direct_operations:
            if (
                current.get("label") != "Disease"
                or direct_operations[0].replacement_label != "Symptom"
            ):
                errors.append(
                    f"entity {assessment.entity_key}: disease_should_be_symptom must update label to Symptom"
                )
        if assessment.verdict == "disease_not_diagnosable" and direct_operations:
            if current.get("label") != "Disease":
                errors.append(
                    f"entity {assessment.entity_key}: disease_not_diagnosable requires a current Disease"
                )
            if (
                direct_operations[0].op == "update_entity"
                and (
                    not direct_operations[0].replacement_label
                    or direct_operations[0].replacement_label == "Disease"
                )
            ):
                errors.append(
                    f"entity {assessment.entity_key}: disease_not_diagnosable must change to a non-Disease label or remove the entity"
                )
        if assessment.verdict == "symptom_should_be_disease" and direct_operations:
            if (
                current.get("label") != "Symptom"
                or direct_operations[0].replacement_label != "Disease"
            ):
                errors.append(
                    f"entity {assessment.entity_key}: symptom_should_be_disease must update label to Disease"
                )
        if assessment.verdict == "wrong_label" and direct_operations:
            replacement_label = direct_operations[0].replacement_label
            if not replacement_label or replacement_label == current.get("label"):
                errors.append(
                    f"entity {assessment.entity_key}: wrong_label must change replacement_label"
                )
        if assessment.verdict == "wrong_boundary" and direct_operations:
            if not direct_operations[0].replacement_name:
                errors.append(
                    f"entity {assessment.entity_key}: wrong_boundary must update replacement_name"
                )
            if direct_operations[0].replacement_label:
                errors.append(
                    f"entity {assessment.entity_key}: wrong_boundary cannot change the entity label"
                )
            if normalize_text(direct_operations[0].replacement_name) == normalize_text(
                current.get("name")
            ):
                errors.append(
                    f"entity {assessment.entity_key}: wrong_boundary replacement_name must change the normalized name"
                )

    relation_by_key = {item["relation_key"]: item for item in relation_items}
    for assessment in response.relation_assessments:
        if assessment.replacement_evidence:
            error = _check_span(
                assessment.replacement_evidence,
                source_text,
                f"relation {assessment.relation_key}",
            )
            if error:
                errors.append(error)
            if _span_signature(assessment.replacement_evidence) not in allowed_evidence_spans:
                errors.append(
                    f"relation {assessment.relation_key}: replacement_evidence is not copied from evidence_span_inventory"
                )
        for ref in (assessment.proposed_source_ref, assessment.proposed_target_ref):
            if ref and ref not in valid_refs:
                errors.append(f"relation {assessment.relation_key}: unknown endpoint ref {ref}")
        if assessment.proposed_relation:
            spec = allowed_relations.get(assessment.proposed_relation)
            if not isinstance(spec, dict) or spec.get("status") != "active":
                errors.append(
                    f"relation {assessment.relation_key}: proposed relation is not active"
                )
        current_relation = relation_by_key[assessment.relation_key]
        current_spec = allowed_relations.get(current_relation["relation"], {})
        current_active = (
            isinstance(current_spec, dict) and current_spec.get("status") == "active"
        )
        span = current_relation.get("evidence_span")
        current_evidence_exact = (
            isinstance(span, dict)
            and span.get("basis") == "record.input"
            and isinstance(span.get("start"), int)
            and isinstance(span.get("end"), int)
            and isinstance(span.get("text"), str)
            and 0 <= span["start"] < span["end"] <= len(source_text)
            and source_text[span["start"] : span["end"]] == span["text"]
            and current_relation.get("evidence") == span["text"]
        )
        direct_operations = relation_operations.get(assessment.relation_key, [])
        is_unresolved = assessment.relation_key in unresolved_object_keys
        if not current_active and assessment.semantic_verdict == "correct":
            errors.append(
                f"relation {assessment.relation_key}: review-status relation cannot be semantically correct"
            )
        if not current_evidence_exact and assessment.evidence_verdict == "exact":
            errors.append(
                f"relation {assessment.relation_key}: missing/invalid baseline evidence cannot be exact"
            )
        needs_semantic_action = assessment.semantic_verdict != "correct" or not current_active
        needs_evidence_action = assessment.evidence_verdict != "exact" or not current_evidence_exact
        if (needs_semantic_action or needs_evidence_action) and not direct_operations and not is_unresolved:
            errors.append(
                f"relation {assessment.relation_key}: finding lacks operation or unresolved finding"
            )
        if assessment.semantic_verdict == "uncertain" or assessment.evidence_verdict == "uncertain":
            if not is_unresolved:
                errors.append(
                    f"relation {assessment.relation_key}: uncertain verdict requires unresolved finding"
                )
            if direct_operations:
                errors.append(
                    f"relation {assessment.relation_key}: uncertain verdict cannot have a direct mutation"
                )

        semantic_allowed: set[str]
        if assessment.semantic_verdict in {
            "wrong_type",
            "wrong_direction",
            "wrong_endpoint",
        } or not current_active:
            semantic_allowed = {"replace_relation", "remove_relation"}
        elif assessment.semantic_verdict in {"unsupported", "duplicate"}:
            semantic_allowed = {"remove_relation"}
        elif assessment.semantic_verdict == "correct":
            semantic_allowed = {
                "remove_relation",
                "set_relation_evidence",
            }
        else:
            semantic_allowed = set()

        evidence_allowed: set[str]
        if assessment.evidence_verdict == "repairable":
            evidence_allowed = {"replace_relation", "set_relation_evidence"}
        elif assessment.evidence_verdict == "unsupported":
            evidence_allowed = {"remove_relation"}
        elif assessment.evidence_verdict == "exact":
            evidence_allowed = {
                "replace_relation",
                "remove_relation",
                "set_relation_evidence",
            }
        else:
            evidence_allowed = set()

        allowed_direct_ops = semantic_allowed & evidence_allowed
        for operation in direct_operations:
            if operation.op not in allowed_direct_ops:
                errors.append(
                    f"relation {assessment.relation_key}: {operation.op} does not close semantic/evidence verdicts"
                )
            if operation.op == "replace_relation":
                if assessment.semantic_verdict != "correct" or not current_active:
                    if not (
                        assessment.proposed_source_ref
                        and assessment.proposed_target_ref
                        and assessment.proposed_relation
                    ):
                        errors.append(
                            f"relation {assessment.relation_key}: semantic replacement fields are required"
                        )
                for field_name, operation_value in (
                    ("proposed_source_ref", operation.source_ref),
                    ("proposed_target_ref", operation.target_ref),
                    ("proposed_relation", operation.replacement_relation),
                ):
                    assessment_value = getattr(assessment, field_name)
                    if assessment_value is not None and assessment_value != operation_value:
                        errors.append(
                            f"relation {assessment.relation_key}: {field_name} disagrees with operation"
                        )
            elif any(
                value is not None
                for value in (
                    assessment.proposed_source_ref,
                    assessment.proposed_target_ref,
                    assessment.proposed_relation,
                )
            ):
                errors.append(
                    f"relation {assessment.relation_key}: proposed replacement fields require replace_relation"
                )
            if assessment.replacement_evidence is not None and operation.op in {
                "replace_relation",
                "set_relation_evidence",
            }:
                if _span_signature(assessment.replacement_evidence) != _span_signature(
                    operation.evidence  # type: ignore[arg-type]
                ):
                    errors.append(
                        f"relation {assessment.relation_key}: replacement_evidence disagrees with operation"
                    )
            if operation.op == "replace_relation":
                current_source = current_relation["source_entity_key"]
                current_target = current_relation["target_entity_key"]
                current_predicate = current_relation["relation"]
                if (
                    assessment.semantic_verdict == "wrong_type"
                    and operation.replacement_relation == current_predicate
                ):
                    errors.append(
                        f"relation {assessment.relation_key}: wrong_type replacement must change the predicate"
                    )
                if assessment.semantic_verdict == "wrong_direction" and not (
                    operation.source_ref == current_target
                    and operation.target_ref == current_source
                ):
                    errors.append(
                        f"relation {assessment.relation_key}: wrong_direction replacement must reverse the endpoints"
                    )
                if assessment.semantic_verdict == "wrong_endpoint" and (
                    operation.source_ref == current_source
                    and operation.target_ref == current_target
                ):
                    errors.append(
                        f"relation {assessment.relation_key}: wrong_endpoint replacement must change an endpoint"
                    )
        if assessment.evidence_verdict == "repairable" and direct_operations:
            if assessment.replacement_evidence is None:
                errors.append(
                    f"relation {assessment.relation_key}: repairable evidence requires replacement_evidence"
                )
            elif current_evidence_exact and _span_signature(
                assessment.replacement_evidence
            ) == _span_signature(span):
                errors.append(
                    f"relation {assessment.relation_key}: repairable evidence must change the exact span"
                )
        if not direct_operations and any(
            value is not None
            for value in (
                assessment.proposed_source_ref,
                assessment.proposed_target_ref,
                assessment.proposed_relation,
                assessment.replacement_evidence,
            )
        ):
            errors.append(
                f"relation {assessment.relation_key}: proposal fields require a direct relation operation"
            )
        if (
            not needs_semantic_action
            and not needs_evidence_action
            and direct_operations
        ):
            errors.append(
                f"relation {assessment.relation_key}: correct exact relation cannot have a direct mutation"
            )

    for assessment in response.cue_assessments:
        unknown_relations = set(assessment.represented_by_relation_keys) - relation_keys
        if unknown_relations:
            errors.append(
                f"cue {assessment.cue_key}: unknown represented relation keys {sorted(unknown_relations)}"
            )
        unknown_operations = set(assessment.proposed_operation_refs) - set(operation_refs)
        if unknown_operations:
            errors.append(
                f"cue {assessment.cue_key}: unknown operation refs {sorted(unknown_operations)}"
            )
        if assessment.verdict == "represented_correctly":
            if not assessment.represented_by_relation_keys:
                errors.append(
                    f"cue {assessment.cue_key}: represented_correctly requires at least one relation key"
                )
            if assessment.proposed_operation_refs:
                errors.append(
                    f"cue {assessment.cue_key}: represented_correctly cannot propose operations"
                )
        if assessment.verdict == "not_graph_fact" and (
            assessment.represented_by_relation_keys
            or assessment.proposed_operation_refs
        ):
            errors.append(
                f"cue {assessment.cue_key}: not_graph_fact requires empty relation/operation refs"
            )
        if assessment.verdict == "uncertain":
            if assessment.cue_key not in unresolved_object_keys:
                errors.append(
                    f"cue {assessment.cue_key}: uncertain verdict requires unresolved finding"
                )
            if assessment.proposed_operation_refs:
                errors.append(
                    f"cue {assessment.cue_key}: uncertain verdict cannot propose operations"
                )
        if (
            assessment.verdict == "missing_graph_fact"
            and not assessment.proposed_operation_refs
            and assessment.cue_key not in unresolved_object_keys
        ):
            errors.append(
                f"cue {assessment.cue_key}: finding lacks operation or unresolved finding"
            )

    missing_fact_cues_by_operation: dict[str, set[str]] = {}
    for assessment in response.cue_assessments:
        if assessment.verdict != "missing_graph_fact":
            continue
        for operation_ref in assessment.proposed_operation_refs:
            missing_fact_cues_by_operation.setdefault(operation_ref, set()).add(
                assessment.cue_key
            )

    closed_cue_keys = {
        assessment.cue_key
        for assessment in response.cue_assessments
        if assessment.proposed_operation_refs
        or assessment.cue_key in unresolved_object_keys
    }

    issue_linked_cue_keys: set[str] = set()
    for dimension in response.dimension_assessments:
        if set(dimension.linked_entity_keys) - entity_keys:
            errors.append(f"dimension {dimension.dimension}: unknown entity key")
        if set(dimension.linked_relation_keys) - relation_keys:
            errors.append(f"dimension {dimension.dimension}: unknown relation key")
        if set(dimension.linked_cue_keys) - cue_keys:
            errors.append(f"dimension {dimension.dimension}: unknown cue key")
        if dimension.status in {"issues_found", "uncertain"}:
            issue_linked_cue_keys.update(dimension.linked_cue_keys)
            linked_keys = (
                set(dimension.linked_entity_keys)
                | set(dimension.linked_relation_keys)
                | set(dimension.linked_cue_keys)
            )
            operation_keys = (
                set(entity_operations) | set(relation_operations) | closed_cue_keys
            )
            linked_to_closure = bool(
                linked_keys & (operation_keys | unresolved_object_keys)
            )
            if not linked_to_closure:
                errors.append(
                    f"dimension {dimension.dimension}: issue lacks linked operation or unresolved finding"
                )

    for operation in response.proposed_operations:
        error = _check_span(operation.evidence, source_text, operation.operation_ref)  # type: ignore[arg-type]
        if error:
            errors.append(error)
        if _span_signature(operation.evidence) not in allowed_evidence_spans:  # type: ignore[arg-type]
            errors.append(
                f"{operation.operation_ref}: evidence is not copied from evidence_span_inventory"
            )
        for key in (operation.entity_key, operation.parent_entity_key):
            if key and key not in entity_keys:
                errors.append(f"{operation.operation_ref}: unknown entity key {key}")
        if operation.entity_key == task["main_disease_entity_key"]:
            errors.append(f"{operation.operation_ref}: canonical main Disease is protected")
        if operation.relation_key and operation.relation_key not in relation_keys:
            errors.append(
                f"{operation.operation_ref}: unknown relation key {operation.relation_key}"
            )
        if operation.replacement_label and operation.replacement_label not in allowed_labels:
            errors.append(f"{operation.operation_ref}: replacement label is outside schema")
        if operation.replacement_relation:
            spec = allowed_relations.get(operation.replacement_relation)
            if not isinstance(spec, dict) or spec.get("status") != "active":
                errors.append(f"{operation.operation_ref}: replacement relation is not active")
        for ref in (operation.source_ref, operation.target_ref):
            if ref and ref not in valid_refs:
                errors.append(f"{operation.operation_ref}: unknown endpoint ref {ref}")
            if ref and ref in removed_entity_keys:
                errors.append(
                    f"{operation.operation_ref}: relation cannot use a removed/collapsed endpoint"
                )
        if operation.op in {"add_entity", "add_relation"}:
            linked_cues = missing_fact_cues_by_operation.get(
                operation.operation_ref, set()
            )
            if not linked_cues:
                errors.append(
                    f"{operation.operation_ref}: add operation must be linked from a missing_graph_fact cue"
                )
            elif not linked_cues & issue_linked_cue_keys:
                errors.append(
                    f"{operation.operation_ref}: add operation cue must be linked from an issue dimension"
                )
        if operation.op == "collapse_symptom_into_manifestations":
            child = entity_by_key.get(str(operation.entity_key))
            parent = entity_by_key.get(str(operation.parent_entity_key))
            if not child or not parent or child["label"] != "Symptom" or parent["label"] != "Symptom":
                errors.append(
                    f"{operation.operation_ref}: collapse endpoints must both be Symptom"
                )
            if operation.entity_key == operation.parent_entity_key:
                errors.append(f"{operation.operation_ref}: cannot collapse an entity into itself")
            if child and _property_leaf_counter(child.get("properties") or {}):
                errors.append(
                    f"{operation.operation_ref}: non-empty child properties cannot be collapsed losslessly; use unresolved"
                )
            if child and normalize_text(operation.manifestation_text) != normalize_text(
                child.get("name")
            ):
                errors.append(
                    f"{operation.operation_ref}: manifestation_text must equal the child entity name"
                )
        if operation.op in {"add_relation", "replace_relation"}:
            spec = allowed_relations.get(str(operation.replacement_relation), {})
            source_label = label_by_ref.get(str(operation.source_ref))
            target_label = label_by_ref.get(str(operation.target_ref))
            if operation.source_ref == operation.target_ref:
                errors.append(f"{operation.operation_ref}: relation cannot be self-referential")
            if source_label and target_label and spec and not _matches_direction(
                spec, source_label, target_label
            ):
                errors.append(f"{operation.operation_ref}: relation domain/range mismatch")
        if operation.op == "add_entity" and operation.replacement_label == "Disease":
            matching = [
                assessment
                for assessment in response.entity_assessments
                if assessment.needs_who_lookup
            ]
            # Existing assessments cannot represent a new entity; the operation reason
            # must explicitly retain the WHO boundary instead of supplying code fields.
            if "who" not in operation.reason.lower() and "icd" not in operation.reason.lower():
                errors.append(
                    f"{operation.operation_ref}: new Disease must request WHO/ICD lookup"
                )
            del matching

    update_by_entity = {
        str(operation.entity_key): operation
        for operation in response.proposed_operations
        if operation.op == "update_entity" and operation.entity_key
    }
    final_concepts: Counter[tuple[str, str]] = Counter()
    for item in entity_items:
        if item["entity_key"] in removed_entity_keys:
            continue
        update = update_by_entity.get(item["entity_key"])
        final_label = update.replacement_label if update and update.replacement_label else item["label"]
        final_name = update.replacement_name if update and update.replacement_name else item["name"]
        final_concepts[(normalize_text(final_label), normalize_text(final_name))] += 1
    for operation in response.proposed_operations:
        if operation.op == "add_entity":
            final_concepts[
                (
                    normalize_text(operation.replacement_label),
                    normalize_text(operation.replacement_name),
                )
            ] += 1
    duplicate_concepts = sorted(
        concept for concept, count in final_concepts.items() if count > 1
    )
    if duplicate_concepts:
        errors.append(
            f"final entity graph contains duplicate label/name concepts: {duplicate_concepts}"
        )

    final_relation_triples: Counter[tuple[str, str, str]] = Counter()
    for relation in relation_items:
        direct = relation_operations.get(relation["relation_key"], [])
        if any(operation.op == "remove_relation" for operation in direct):
            continue
        replacement = next(
            (operation for operation in direct if operation.op == "replace_relation"),
            None,
        )
        if replacement:
            triple = (
                str(replacement.source_ref),
                str(replacement.target_ref),
                str(replacement.replacement_relation),
            )
        else:
            triple = (
                relation["source_entity_key"],
                relation["target_entity_key"],
                relation["relation"],
            )
        final_relation_triples[triple] += 1
    for operation in response.proposed_operations:
        if operation.op == "add_relation":
            final_relation_triples[
                (
                    str(operation.source_ref),
                    str(operation.target_ref),
                    str(operation.replacement_relation),
                )
            ] += 1
    duplicate_relations = sorted(
        triple for triple, count in final_relation_triples.items() if count > 1
    )
    if duplicate_relations:
        errors.append(
            f"final graph contains duplicate source/target/relation triples: {duplicate_relations}"
        )

    return errors


def require_valid_semantic_audit_response(
    response: SemanticAuditResponse, task: dict[str, Any]
) -> None:
    errors = validate_semantic_audit_response(response, task)
    if errors:
        raise SemanticAuditValidationError("; ".join(errors))


def response_sha256(response: SemanticAuditResponse) -> str:
    return sha256_value(response.model_dump(mode="json"))


def canonical_operation_signature(operation: SemanticOperation) -> str:
    payload = operation.model_dump(mode="json")
    for key in ("operation_ref", "reason", "confidence"):
        payload.pop(key, None)
    return sha256_value(payload)


def response_semantic_signature(response: SemanticAuditResponse) -> dict[str, Any]:
    """Return the reason/confidence-independent decision signature for consensus."""

    operation_signatures = {
        operation.operation_ref: canonical_operation_signature(operation)
        for operation in response.proposed_operations
    }
    return {
        "entities": {
            item.entity_key: (
                item.verdict,
                item.recommended_label,
                item.parent_entity_key,
                item.duplicate_of_entity_key,
                item.needs_who_lookup,
            )
            for item in response.entity_assessments
        },
        "relations": {
            item.relation_key: (
                item.semantic_verdict,
                item.evidence_verdict,
                item.proposed_source_ref,
                item.proposed_target_ref,
                item.proposed_relation,
                item.replacement_evidence.model_dump(mode="json")
                if item.replacement_evidence
                else None,
            )
            for item in response.relation_assessments
        },
        "cues": {
            item.cue_key: (
                item.verdict,
                tuple(sorted(item.represented_by_relation_keys)),
                tuple(
                    sorted(
                        operation_signatures[operation_ref]
                        for operation_ref in item.proposed_operation_refs
                    )
                ),
            )
            for item in response.cue_assessments
        },
        "dimensions": {
            item.dimension: (
                item.status,
                tuple(sorted(item.linked_entity_keys)),
                tuple(sorted(item.linked_relation_keys)),
                tuple(sorted(item.linked_cue_keys)),
            )
            for item in response.dimension_assessments
        },
        "operations": sorted(
            canonical_operation_signature(operation)
            for operation in response.proposed_operations
        ),
        "unresolved": sorted(
            (item.category, tuple(sorted(item.object_keys)))
            for item in response.unresolved
        ),
    }


def response_requires_review(response: SemanticAuditResponse) -> bool:
    if response.proposed_operations or response.unresolved:
        return True
    if response.overall_confidence < 0.95:
        return True
    for assessment in response.entity_assessments:
        if assessment.verdict != "correct" or assessment.confidence < 0.95:
            return True
    for assessment in response.relation_assessments:
        if (
            assessment.semantic_verdict != "correct"
            or assessment.evidence_verdict != "exact"
            or assessment.confidence < 0.95
        ):
            return True
    for assessment in response.cue_assessments:
        if assessment.verdict == "uncertain" or assessment.confidence < 0.95:
            return True
    return False


def response_has_unresolved_semantics(
    response: SemanticAuditResponse, *, minimum_confidence: float = 0.95
) -> bool:
    """Return whether a response is unsafe to finalize without adjudication."""

    if response.unresolved or response.overall_confidence < minimum_confidence:
        return True
    if any(
        item.verdict == "uncertain" or item.confidence < minimum_confidence
        for item in response.entity_assessments
    ):
        return True
    if any(
        item.semantic_verdict == "uncertain"
        or item.evidence_verdict == "uncertain"
        or item.confidence < minimum_confidence
        for item in response.relation_assessments
    ):
        return True
    if any(
        item.verdict == "uncertain" or item.confidence < minimum_confidence
        for item in response.cue_assessments
    ):
        return True
    return any(
        item.status == "uncertain" for item in response.dimension_assessments
    )


SEMANTIC_AUDIT_SYSTEM_PROMPT = r"""You are a meticulous medical knowledge-graph auditor.
Audit exactly one frozen ICD-11-oriented Schema v2 record. Return one JSON object and
nothing else. You must inspect the complete original_text and every supplied entity,
relation, cue, and required dimension. Coverage is checked locally by exact key-set
equality; omitting or duplicating a key invalidates the response.

Medical rules:
- Disease means an independently diagnosable disorder/category. Descriptive phrases,
  symptoms, mechanisms, demographics, generic "medical condition", medication effects,
  and section headings are not Disease nodes merely because they sound clinical.
- A Symptom node must be independently meaningful. A phrase that only elaborates a
  parent symptom belongs in its manifestations. Never collapse if that loses distinct
  properties or relations.
- subsumes is broader -> narrower. must_be_ruled_out_for is alternative Disease ->
  diagnosis under consideration. excludes_diagnosis_of is exclusionary Criteria or
  Symptom -> diagnosis excluded. somatic_cause_of is somatic Disease -> psychiatric or
  behavioural target Disease. Patient Information affects_diagnosis_of only when the
  text entails diagnostic impact; a word such as "males" alone is not sufficient.
- WHO ICD classification exclusions are not automatically clinical rule-out relations.
- Never invent, copy, or modify ICD codes. Set needs_who_lookup=true for every retained
  non-main Disease (including a Symptom reclassified as Disease), and false for the
  already linked canonical main Disease. WHO API validation is a separate stage.
- The canonical main Disease is protected and must receive verdict=correct.
- Never calculate character offsets. For every assessment evidence,
  relation.replacement_evidence, and proposed-operation evidence, copy an exact object
  from evidence_span_inventory, using only its basis/text/start/end fields and omitting
  span_key. If no supplied span supports a safe decision, create an unresolved finding.
- Use only active schema relations for add/replace operations. Prefer unresolved over
  guessing. Natural-language reasons are concise.

Local closure contract (violations reject the response before it is stored):
- Every non-correct entity verdict must have exactly one compatible direct mutation
  targeting that entity, or an unresolved finding whose object_keys contains its
  entity_key. uncertain must use unresolved and cannot carry a mutation. Use
  update_entity for wrong labels/boundaries, remove_entity for unsupported/duplicates,
  and only collapse_symptom_into_manifestations for description fragments. The
  manifestation_text must equal the child name. If the child has any non-empty
  properties, this operation cannot migrate them losslessly: use unresolved, never
  remove the description fragment and lose its clinical detail.
  disease_should_be_symptom must update the label to Symptom;
  symptom_should_be_disease must update it to Disease. For verdict=correct, all
  recommendation fields must be null. A collapse parent cannot itself be updated,
  removed, or collapsed in the same response. Every label-changing update_entity must
  provide replacement_properties using only keys allowed by the target label and must
  preserve every non-empty original property value exactly (move values; never drop,
  rewrite, or invent them). If no semantically safe lossless mapping exists, unresolved.
- Every wrong/non-exact relation assessment, every relation whose current schema status
  is not active, and every relation with absent/invalid baseline evidence must have one
  compatible replace/remove/set-evidence operation or an unresolved finding containing
  its relation_key. A non-active relation can never be semantic_verdict=correct, and a
  missing/invalid baseline span can never be evidence_verdict=exact.
- wrong_type/wrong_direction/wrong_endpoint require replace_relation or
  remove_relation; unsupported/duplicate require remove_relation. When semantics are
  correct, pure repairable evidence uses only set_relation_evidence; replace_relation
  is reserved for a semantic change. proposed_source_ref,
  proposed_target_ref, proposed_relation, and replacement_evidence must exactly agree
  with the replace/set operation payload; all four must be null when there is no direct
  relation operation. uncertain must use unresolved.
- Every operation must make the state change named by its verdict. wrong_type changes
  the predicate; wrong_direction reverses the current endpoints; wrong_endpoint changes
  at least one endpoint. If a currently exact span is called repairable, its replacement
  must be a different pre-indexed span. Never use a no-op to close a finding.
- Every relation incident to an entity being removed or collapsed must itself be
  removed/replaced, or made unresolved. A replacement/add relation cannot use an entity
  being removed. If an entity label change makes a retained relation violate its
  domain/range, replace/remove that relation or make it unresolved.
- represented_correctly cues require at least one represented relation and no operation;
  not_graph_fact requires both reference lists empty. missing_graph_fact must list the
  operation_ref values that close it or have an unresolved finding containing its
  cue_key. uncertain cues require unresolved and cannot propose an operation. For a new
  entity or relation, link the add operation from the missing_graph_fact cue; then link
  the affected dimension to that cue_key.
- Every issues_found or uncertain dimension must link an entity/relation/cue whose
  operation or unresolved finding closes the issue. Only one direct mutation may target
  the same existing entity or relation. Never add a duplicate label/name concept, never
  create a self-referential relation, and never collapse an entity into itself. The
  final graph may contain at most one relation per source/target/predicate triple;
  different evidence spans do not justify duplicate edges.
- represented_by_relation_keys, cue operation links, and every dimension linked-key
  list are part of blind-review consensus. Include only links you actually verified.
- Unresolved findings have exactly this shape:
  {"finding_ref":"finding:1",
   "category":"entity|relation|cue|missing_fact|schema_gap|medical_review",
   "object_keys":["an exact entity_key, relation_key, cue_key, or dimension name"],
   "reason":"why no safe deterministic operation is possible"}.

Return all fields required by this shape:
{
  "protocol_version":"deepseek-semantic-audit-v1",
  "phase":"primary|blind_review",
  "dataset_sha256":"copy", "schema_sha256":"copy", "prompt_sha256":"copy",
  "task_sha256":"copy", "source_record_id":"copy", "record_sha256":"copy",
  "entity_assessments":[{
    "entity_key":"copy each exactly once",
    "verdict":"correct|disease_not_diagnosable|disease_should_be_symptom|symptom_should_be_disease|wrong_label|description_fragment|wrong_boundary|duplicate|unsupported|uncertain",
    "recommended_label":null, "parent_entity_key":null,
    "duplicate_of_entity_key":null, "evidence":[], "needs_who_lookup":false,
    "reason":"...", "confidence":0.0
  }],
  "relation_assessments":[{
    "relation_key":"copy each exactly once",
    "semantic_verdict":"correct|wrong_type|wrong_direction|wrong_endpoint|unsupported|duplicate|uncertain",
    "evidence_verdict":"exact|repairable|unsupported|uncertain",
    "proposed_source_ref":null, "proposed_target_ref":null,
    "proposed_relation":null, "replacement_evidence":null,
    "reason":"...", "confidence":0.0
  }],
  "cue_assessments":[{
    "cue_key":"copy each exactly once",
    "verdict":"represented_correctly|missing_graph_fact|not_graph_fact|uncertain",
    "represented_by_relation_keys":[], "proposed_operation_refs":[],
    "reason":"...", "confidence":0.0
  }],
  "dimension_assessments":[{
    "dimension":"copy each required dimension exactly once",
    "status":"pass|issues_found|not_applicable|uncertain",
    "linked_entity_keys":[], "linked_relation_keys":[], "linked_cue_keys":[],
    "reason":"..."
  }],
  "proposed_operations":[{
    "operation_ref":"op:1",
    "op":"add_entity|update_entity|remove_entity|collapse_symptom_into_manifestations|add_relation|replace_relation|remove_relation|set_relation_evidence",
    "entity_key":null, "new_entity_ref":null, "replacement_label":null,
    "replacement_name":null, "replacement_properties":null,
    "parent_entity_key":null,
    "manifestation_text":null, "relation_key":null, "source_ref":null,
    "target_ref":null, "replacement_relation":null,
    "evidence":{"basis":"record.input","text":"exact quote","start":0,"end":1},
    "reason":"...", "confidence":0.0
  }],
  "unresolved":[{
    "finding_ref":"finding:1",
    "category":"entity|relation|cue|missing_fact|schema_gap|medical_review",
    "object_keys":["copy one or more exact keys"], "reason":"..."
  }],
  "overall_confidence":0.0
}

Use null for every unused optional field. If no cues or no relations are supplied,
return an empty list for that section. Blind review is independent: no first response
is available, so audit the frozen record from scratch.
"""

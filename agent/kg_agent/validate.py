"""Local validator and coverage diagnostics for the assembled graph."""

from __future__ import annotations

from collections import Counter

from .contracts import EntityCandidate, Graph, SourceDocument, ValidationReport, allowed_relations, normalize_label
from .source import resolve_span_ref


def validate_candidates(
    candidates: list[EntityCandidate], document: SourceDocument
) -> tuple[list[EntityCandidate], ValidationReport]:
    report = ValidationReport(valid=True)
    valid: list[EntityCandidate] = []
    spans = document.span_map()
    for candidate in candidates:
        label = normalize_label(candidate.label)
        if resolve_span_ref(document, candidate.span_id) is None:
            report.add("ungrounded_entity", f"span does not exist: {candidate.span_id}", {"span_id": candidate.span_id})
            continue
        if label is None:
            report.add("unknown_label", f"unknown entity label: {candidate.label}", {"span_id": candidate.span_id})
            continue
        valid.append(
            EntityCandidate(
                span_id=candidate.span_id,
                label=label,
                importance=candidate.importance,
                name=candidate.name,
            )
        )
    return valid, report


def validate_graph(graph: Graph, document: SourceDocument) -> ValidationReport:
    report = ValidationReport(valid=True)
    spans = document.span_map()
    entity_ids = [entity.id for entity in graph.entities]
    if len(entity_ids) != len(set(entity_ids)):
        report.add("duplicate_entity_id", "entity IDs must be unique")

    id_to_entity = {entity.id: entity for entity in graph.entities}
    for entity in graph.entities:
        if not entity.name.strip():
            report.add("empty_entity_name", f"empty name for {entity.id}", {"id": entity.id})
        if normalize_label(entity.label) != entity.label:
            report.add("noncanonical_label", f"noncanonical label for {entity.id}", {"id": entity.id})
        if document.source_type != "structured_icd":
            for span_id in entity.span_ids:
                if resolve_span_ref(document, span_id) is None:
                    report.add(
                        "ungrounded_entity",
                        f"entity {entity.id} references missing span {span_id}",
                        {"id": entity.id, "span_id": span_id},
                    )

    seen_relations: Counter[tuple[str, str, str]] = Counter()
    for relation in graph.relations:
        key = (relation.source, relation.target, relation.relation)
        seen_relations[key] += 1
        if relation.source not in id_to_entity or relation.target not in id_to_entity:
            report.add("missing_endpoint", f"relation endpoint does not exist: {key}", {"relation": relation.__dict__})
            continue
        source_label = id_to_entity[relation.source].label
        target_label = id_to_entity[relation.target].label
        if relation.relation not in allowed_relations(source_label, target_label):
            report.add(
                "invalid_signature",
                f"{source_label} -{relation.relation}-> {target_label} is not allowed",
                {"relation": relation.__dict__},
            )
        if relation.source == relation.target:
            report.add("self_relation", f"self relation is not allowed: {key}", {"relation": relation.__dict__})
        if document.source_type == "structured_icd":
            if not relation.evidence_ids:
                report.add("missing_evidence", f"structured relation has no field evidence: {key}")
        else:
            if not relation.evidence_ids:
                report.add("missing_evidence", f"relation has no sentence evidence: {key}")
            for evidence_id in relation.evidence_ids:
                if resolve_span_ref(document, evidence_id) is None:
                    report.add(
                        "invalid_evidence",
                        f"evidence span does not exist: {evidence_id}",
                        {"relation": relation.__dict__},
                    )

    for key, count in seen_relations.items():
        if count > 1:
            report.add("duplicate_relation", f"duplicate relation: {key}", {"count": count})
    return report


def coverage_audit(graph: Graph, document: SourceDocument) -> dict[str, int | float]:
    """Return cheap structural coverage numbers for experiment tracking."""

    grounded = 0
    mentions = 0
    for entity in graph.entities:
        mentions += len(entity.span_ids)
        grounded += sum(resolve_span_ref(document, span_id) is not None for span_id in entity.span_ids)
    return {
        "entities": len(graph.entities),
        "relations": len(graph.relations),
        "entity_mentions": mentions,
        "grounded_entity_mentions": grounded,
        "grounding_rate": (grounded / mentions) if mentions else 1.0,
    }

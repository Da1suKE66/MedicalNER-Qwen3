"""Deterministic grounding, deduplication, and local ID assignment."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Iterable

from .contracts import (
    Entity,
    EntityCandidate,
    IMPORTANCE_ORDER,
    LABEL_PREFIXES,
    SourceDocument,
    Span,
    natural_id_key,
    normalize_importance,
    normalize_label,
)
from .source import resolve_span_ref


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


def _find_mention(source: str, mention: str) -> tuple[int, int] | None:
    """Find a copied mention while preserving source character offsets."""

    mention = str(mention or "").strip()
    if not mention:
        return None
    direct = source.find(mention)
    if direct >= 0:
        return direct, direct + len(mention)
    insensitive = re.search(re.escape(mention), source, flags=re.IGNORECASE)
    if insensitive is not None:
        return insensitive.start(), insensitive.end()

    # Permit harmless whitespace differences while still requiring the same
    # token sequence. This is useful when a model copies a line break as a
    # normal space, but it does not permit a generated paraphrase.
    tokens = [token for token in re.split(r"\s+", mention) if token]
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def _ground_copied_mention(
    candidate_span: Span,
    candidate_span_id: str,
    mention: str,
    document: SourceDocument,
) -> tuple[str, str] | None:
    """Return canonical source text and a stable local span reference."""

    base_id = candidate_span_id.split(":", 1)[0]
    base_span = document.span_map().get(base_id)
    if base_span is None:
        return None
    region_start = max(0, candidate_span.start - base_span.start)
    region_end = min(len(base_span.text), candidate_span.end - base_span.start)
    if region_end <= region_start:
        return None
    match = _find_mention(base_span.text[region_start:region_end], mention)
    if match is None:
        return None
    start = region_start + match[0]
    end = region_start + match[1]
    source_text = base_span.text[start:end]
    reference = base_span.id if start == 0 and end == len(base_span.text) else f"{base_span.id}:{start}-{end}"
    return source_text, reference


def normalize_candidates(
    candidates: Iterable[EntityCandidate],
    document: SourceDocument,
    fixed_entities: Iterable[Entity] = (),
) -> tuple[list[Entity], dict[str, str], list[dict[str, str]]]:
    """Ground candidates to source spans and assign stable IDs.

    Returns ``(entities, span_to_entity_id, dropped_candidates)``.  Invalid
    span IDs and unknown labels are dropped here, before relation classification
    can see them.  Exact same-label/same-text mentions are merged; span IDs for
    all mentions remain attached to the resulting entity for provenance.
    """

    dropped: list[dict[str, str]] = []
    fixed = list(fixed_entities)

    # Group exact concepts first.  The source text, not the model's free-form
    # name, is authoritative for normal candidates.
    groups: dict[tuple[str, str], dict[str, object]] = {}
    span_to_group: dict[str, tuple[str, str]] = {}
    for candidate in candidates:
        label = normalize_label(candidate.label)
        importance = normalize_importance(candidate.importance)
        if label not in LABEL_PREFIXES:
            dropped.append({"span_id": candidate.span_id, "reason": "unknown_label"})
            continue
        resolved_span = resolve_span_ref(document, candidate.span_id)
        if resolved_span is None:
            dropped.append({"span_id": candidate.span_id, "reason": "span_not_found"})
            continue
        if candidate.name:
            grounded = _ground_copied_mention(resolved_span, candidate.span_id, candidate.name, document)
            if grounded is None:
                dropped.append({"span_id": candidate.span_id, "reason": "mention_not_in_span"})
                continue
            text, grounded_reference = grounded
            grounded_span = resolve_span_ref(document, grounded_reference)
            if grounded_span is None:
                dropped.append({"span_id": candidate.span_id, "reason": "mention_span_not_found"})
                continue
        else:
            text = resolved_span.text
            grounded_reference = candidate.span_id
            grounded_span = resolved_span
        name = text.strip()
        key = (label, normalize_text(name))
        if not key[1]:
            dropped.append({"span_id": candidate.span_id, "reason": "empty_name"})
            continue
        if key not in groups:
            groups[key] = {
                "label": label,
                "name": text.strip(),
                "importance": importance,
                "span_ids": [],
                "first_offset": grounded_span.start,
            }
        group = groups[key]
        assert isinstance(group["span_ids"], list)
        group["span_ids"].append(grounded_reference)
        if IMPORTANCE_ORDER.get(importance, 99) < IMPORTANCE_ORDER.get(str(group["importance"]), 99):
            group["importance"] = importance
        group["first_offset"] = min(int(group["first_offset"]), grounded_span.start)
        span_to_group[candidate.span_id] = key
        span_to_group[grounded_reference] = key

    # Fixed nodes are inserted before text candidates and can be connected to
    # every text chunk without requiring a fake source span.
    entities: list[Entity] = []
    entity_key_to_id: dict[tuple[str, str], str] = {}
    used_ids: set[str] = set()
    counters: defaultdict[str, int] = defaultdict(int)

    def reserve_id(entity_id: str) -> None:
        used_ids.add(entity_id)
        prefix = "".join(ch for ch in entity_id if not ch.isdigit())
        digits = "".join(ch for ch in entity_id if ch.isdigit())
        counters[prefix] = max(counters[prefix], int(digits or 0))

    def next_id(label: str) -> str:
        prefix = LABEL_PREFIXES[label]
        counters[prefix] += 1
        value = f"{prefix}{counters[prefix]}"
        while value in used_ids:
            counters[prefix] += 1
            value = f"{prefix}{counters[prefix]}"
        return value

    for fixed_entity in fixed:
        label = normalize_label(fixed_entity.label)
        if label not in LABEL_PREFIXES:
            continue
        entity_id = fixed_entity.id or next_id(label)
        if label == "Disease" and not any(item.id == "D1" for item in entities):
            entity_id = "D1"
        reserve_id(entity_id)
        entity = Entity(
            id=entity_id,
            label=label,
            name=fixed_entity.name.strip(),
            properties={"Name": fixed_entity.name.strip(), **fixed_entity.properties},
            span_ids=list(fixed_entity.span_ids),
            importance=fixed_entity.importance,
        )
        entities.append(entity)
        entity_key_to_id[(label, normalize_text(entity.name))] = entity.id

    def sort_key(item: tuple[tuple[str, str], dict[str, object]]) -> tuple[int, int, str, str]:
        key, group = item
        return (
            int(group["first_offset"]),
            IMPORTANCE_ORDER.get(str(group["importance"]), 99),
            key[0],
            key[1],
        )

    # Put Disease first when a free-text document contains an explicit disease
    # mention.  This preserves the historical D1 convention.
    ordered_groups = sorted(groups.items(), key=sort_key)
    ordered_groups.sort(key=lambda item: (0 if item[1]["label"] == "Disease" else 1, sort_key(item)))
    for key, group in ordered_groups:
        label, name_key = key
        if key in entity_key_to_id:
            entity_id = entity_key_to_id[key]
            existing = next(entity for entity in entities if entity.id == entity_id)
            existing.span_ids = list(dict.fromkeys([*existing.span_ids, *group["span_ids"]]))
        else:
            entity_id = next_id(label)
            if label == "Disease" and not any(item.id == "D1" for item in entities):
                entity_id = "D1"
                reserve_id(entity_id)
            properties = {"Name": str(group["name"])}
            entity = Entity(
                id=entity_id,
                label=label,
                name=str(group["name"]),
                properties=properties,
                span_ids=list(group["span_ids"]),
                importance=str(group["importance"]),
            )
            entities.append(entity)
            entity_key_to_id[key] = entity_id

        for span_id in group["span_ids"]:  # type: ignore[union-attr]
            span_to_group[span_id] = key

    span_to_entity_id: dict[str, str] = {}
    for span_id, group_key in span_to_group.items():
        if group_key in entity_key_to_id:
            span_to_entity_id[span_id] = entity_key_to_id[group_key]

    entities.sort(key=lambda item: natural_id_key(item.id))
    return entities, span_to_entity_id, dropped

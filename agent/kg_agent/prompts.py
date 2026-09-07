"""Small JSON-only prompts and tolerant parsers for the two LLM stages."""

from __future__ import annotations

import json
import re
from typing import Any

from .contracts import EntityCandidate, SCHEMA_LABELS, normalize_importance, normalize_label


def build_entity_prompt(
    spans: list[dict[str, str]],
    *,
    target_name: str | None = None,
    max_candidates: int = 64,
    mention_copy: bool = False,
) -> str:
    span_block = "\n".join(
        f"{item['span_id']} [{item.get('section', 'body')}] (local range 0:{len(item['text'])}): {item['text']}"
        for item in spans
    )
    target_line = f"The document's fixed target disease is: {target_name}." if target_name else "No fixed target disease is provided."
    if mention_copy:
        selection_rules = """For each selected concept, copy its exact surface form into `mention`.
Do not generate character offsets. The program will ground `mention` inside `span_id`.
The mention must be a contiguous substring of the cited source span, including its original spelling."""
        output_shape = '{"entity_candidates":[{"span_id":"SENT_001","mention":"exact source phrase","label":"Symptom","importance":"core"}]}'
    else:
        selection_rules = """For a phrase inside a sentence, return a sentence-local reference such as SENT_003:14-62; use the full SENT_003 only when the whole sentence is the concept."""
        output_shape = '{"entity_candidates":[{"span_id":"SENT_001","label":"Symptom","importance":"core"}]}'
    return f"""You are the entity-span selector in a medical knowledge-graph pipeline.
{target_line}
Select only distinct, explicitly stated clinical concepts from the numbered source spans.
{selection_rules}
Return at most {max_candidates} candidates. Do not invent text, IDs, names, codes, or evidence.
Use exactly one label from: {', '.join(SCHEMA_LABELS)}.
Use importance from: required, core, differential, treatment, risk, associated, incidental, alias.
Aliases and ordinary background words should be omitted unless they are clinically meaningful.
Output JSON only with this exact shape:
{output_shape}

SOURCE SPANS
{span_block}
""".strip()


def build_relation_prompt(
    entity_rows: list[dict[str, str]],
    pairs: list[dict[str, Any]],
    spans: list[dict[str, str]],
    *,
    emit_none: bool = False,
) -> str:
    def short_name(value: str) -> str:
        value = str(value)
        return value if len(value) <= 500 else value[:497] + "..."

    entities = "\n".join(
        f"{row['id']} ({row['label']}; importance={row.get('importance', 'associated')}): {short_name(row['name'])}"
        for row in entity_rows
    )
    pair_text = "\n".join(
        f"{pair['pair_id']}: {pair['source']} -> {pair['target']} | allowed={','.join(pair['allowed_relations'])}"
        for pair in pairs
    )
    span_text = "\n".join(
        f"{span['span_id']}: {span['text']}" for span in spans
    )
    if emit_none:
        decision_rules = """For every pair, return exactly one decision. Use relation `NONE` when the source spans do not explicitly support any allowed relation; use an empty evidence_span_id for NONE.
Do not omit pairs. The order and pair IDs must be preserved."""
    else:
        decision_rules = """For each pair, choose one relation from its allowed list or NONE. Do not invent endpoints or relation names.
Evidence must be one source span ID that supports the chosen relation. Omit NONE pairs from the output."""
    return f"""You are the relation classifier in a medical knowledge-graph pipeline.
All entity IDs and pair IDs below are fixed. Classify only explicitly supported relations.
{decision_rules}
Output JSON only with this exact shape:
{{"relations":[{{"pair_id":"P0001","relation":"is_core_symptom_of","evidence_span_id":"SENT_001"}}]}}

ENTITIES
{entities}

CANDIDATE PAIRS
{pair_text}

SOURCE SPANS
{span_text}
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model output."""

    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).replace("```", "").strip()
    if "<output>" in cleaned.lower():
        cleaned = re.split(r"<output>", cleaned, maxsplit=1, flags=re.IGNORECASE)[1]
    cleaned = re.sub(r"<\|[^>]+\|>", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(cleaned[start : index + 1])
                if isinstance(value, dict):
                    return value
                break
    raise ValueError("model output contains no complete JSON object")


def parse_entity_candidates(text: str) -> list[EntityCandidate]:
    value = extract_json_object(text)
    raw_items = value.get("entity_candidates", [])
    if not isinstance(raw_items, list):
        return []
    candidates: list[EntityCandidate] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        label = normalize_label(item.get("label"))
        span_id = str(item.get("span_id") or "").strip()
        if not label or not span_id:
            continue
        candidates.append(
            EntityCandidate(
                span_id=span_id,
                label=label,
                importance=normalize_importance(item.get("importance")),
                name=str(item.get("mention") or item.get("name") or "").strip() or None,
            )
        )
    return candidates


def parse_relation_decisions(text: str) -> list[dict[str, str]]:
    value = extract_json_object(text)
    raw_items = value.get("relations", [])
    if not isinstance(raw_items, list):
        return []
    decisions: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id") or "").strip()
        relation = str(item.get("relation") or "").strip()
        evidence_id = str(item.get("evidence_span_id") or "").strip()
        if pair_id and relation and relation.casefold() != "none":
            decisions.append(
                {"pair_id": pair_id, "relation": relation, "evidence_span_id": evidence_id}
            )
    return decisions

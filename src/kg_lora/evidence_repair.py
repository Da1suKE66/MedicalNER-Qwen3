"""Deterministically restore exact evidence spans from generated evidence text."""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schema_v2 import graph_from_record


ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…|\[\s*(?:\.{3,}|…)\s*\])")
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class LocatedEvidence:
    start: int
    end: int
    text: str
    method: str
    candidate_count: int

    def as_span(self) -> dict[str, Any]:
        return {
            "basis": "record.input",
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "verification_method": self.method,
            "candidate_count": self.candidate_count,
        }


def _all_matches(pattern: re.Pattern[str], text: str) -> list[re.Match[str]]:
    return list(pattern.finditer(text))


def _whitespace_flexible_pattern(value: str) -> str:
    pieces = re.split(r"\s+", value.strip())
    return r"\s+".join(re.escape(piece) for piece in pieces if piece)


def _choose(
    matches: list[re.Match[str]], text: str, method: str
) -> LocatedEvidence | None:
    if not matches:
        return None
    # The shortest matching source span is the least expansive interpretation of
    # an abbreviated evidence string. Ties are resolved by the earliest offset.
    selected = min(matches, key=lambda match: (match.end() - match.start(), match.start()))
    return LocatedEvidence(
        start=selected.start(),
        end=selected.end(),
        text=text[selected.start() : selected.end()],
        method=method if len(matches) == 1 else f"{method}_first_of_{len(matches)}",
        candidate_count=len(matches),
    )


def locate_evidence_span(source_text: str, evidence: str) -> LocatedEvidence | None:
    """Locate generated evidence without fuzzy semantic guessing.

    Supported transformations are deliberately narrow: case-insensitive exact
    matching, whitespace folding, punctuation-only differences, and explicit
    ellipsis gaps. Every accepted result is still an exact slice of ``source_text``.
    """

    if not isinstance(source_text, str) or not isinstance(evidence, str):
        return None
    needle = evidence.strip()
    if not needle:
        return None

    exact = _all_matches(re.compile(re.escape(needle), re.IGNORECASE), source_text)
    located = _choose(exact, source_text, "case_insensitive_exact")
    if located:
        return located

    whitespace_pattern = _whitespace_flexible_pattern(needle)
    if whitespace_pattern:
        whitespace_matches = _all_matches(
            re.compile(whitespace_pattern, re.IGNORECASE), source_text
        )
        located = _choose(whitespace_matches, source_text, "whitespace_normalized")
        if located:
            return located

    ellipsis_parts = [
        part.strip()
        for part in ELLIPSIS_RE.split(needle)
        if part and part.strip()
    ]
    if len(ellipsis_parts) >= 2 and sum(len(WORD_RE.findall(p)) for p in ellipsis_parts) >= 5:
        part_patterns = [_whitespace_flexible_pattern(part) for part in ellipsis_parts]
        if all(part_patterns):
            # A bounded wildcard handles omitted list members or intervening clauses
            # while preventing a match from spanning an entire long source record.
            pattern = r"[\s\S]{0,2500}?".join(part_patterns)
            ellipsis_matches = _all_matches(re.compile(pattern, re.IGNORECASE), source_text)
            located = _choose(ellipsis_matches, source_text, "explicit_ellipsis_anchors")
            if located:
                return located

    tokens = WORD_RE.findall(needle)
    if len(tokens) >= 5:
        punctuation_pattern = r"\W+".join(re.escape(token) for token in tokens)
        punctuation_matches = _all_matches(
            re.compile(punctuation_pattern, re.IGNORECASE), source_text
        )
        bounded = [
            match
            for match in punctuation_matches
            if match.end() - match.start() <= max(len(needle) * 2, len(needle) + 80)
        ]
        located = _choose(bounded, source_text, "punctuation_normalized")
        if located:
            return located
    return None


def span_is_exact(span: Any, source_text: str, retained_evidence: Any) -> bool:
    if not isinstance(span, dict):
        return False
    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not (0 <= start < end <= len(source_text))
        or not isinstance(text, str)
    ):
        return False
    if source_text[start:end] != text:
        return False
    return not isinstance(retained_evidence, str) or retained_evidence.strip() == text


def repair_record_evidence(
    record: dict[str, Any], *, drop_unresolved: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = copy.deepcopy(record)
    graph = graph_from_record(repaired)
    if graph is None or not isinstance(graph.get("relations"), list):
        return repaired, {
            "relation_count": 0,
            "already_verified": 0,
            "repaired": 0,
            "unresolved": 0,
            "dropped": 0,
            "methods": {},
            "unresolved_relations": [],
        }

    source_text = str(repaired.get("input") or "")
    output_relations: list[Any] = []
    methods: Counter[str] = Counter()
    unresolved_rows: list[dict[str, Any]] = []
    already_verified = 0
    repaired_count = 0
    dropped = 0
    raw_relations = graph["relations"]

    for relation_index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            output_relations.append(relation)
            continue
        if span_is_exact(
            relation.get("evidence_span"), source_text, relation.get("evidence")
        ):
            already_verified += 1
            output_relations.append(relation)
            continue
        evidence = relation.get("evidence")
        located = locate_evidence_span(source_text, evidence) if isinstance(evidence, str) else None
        if located is not None:
            original = evidence
            if original.strip() != located.text:
                relation["evidence_original"] = original
            relation["evidence"] = located.text
            relation["evidence_span"] = located.as_span()
            relation["evidence_repair"] = {
                "method": located.method,
                "candidate_count": located.candidate_count,
            }
            methods[located.method] += 1
            repaired_count += 1
            output_relations.append(relation)
            continue

        unresolved_rows.append(
            {
                "relation_index": relation_index,
                "source": relation.get("source"),
                "target": relation.get("target"),
                "relation": relation.get("relation"),
                "evidence": evidence,
                "action": "dropped" if drop_unresolved else "retained_for_review",
            }
        )
        if drop_unresolved:
            dropped += 1
        else:
            output_relations.append(relation)

    if drop_unresolved:
        graph["relations"] = output_relations
    audit = {
        "relation_count": len(raw_relations),
        "already_verified": already_verified,
        "repaired": repaired_count,
        "unresolved": len(unresolved_rows),
        "dropped": dropped,
        "methods": dict(sorted(methods.items())),
        "unresolved_relations": unresolved_rows,
    }
    repaired["evidence_repair"] = audit
    return repaired, audit

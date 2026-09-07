"""The same input-only named-pair/context construction for training and inference."""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict

from .contracts import Entity, EntityCandidate, allowed_relations, normalize_label
from .normalize import normalize_candidates, normalize_text
from .source import SourceDocument


def mentions(name, document):
    # Regex IGNORECASE preserves offsets, unlike indexing into casefolded text.
    words = str(name).strip().split()
    if not words:
        return []
    pattern = re.compile(r"\s+".join(map(re.escape, words)), re.I)
    result = []
    for span in document.spans:
        for m in pattern.finditer(span.text):
            # A concept must not be an accidental sub-word match.
            if (
                m.start()
                and span.text[m.start() - 1].isalnum()
                and m.group()[0].isalnum()
            ):
                continue
            if (
                m.end() < len(span.text)
                and span.text[m.end()].isalnum()
                and m.group()[-1].isalnum()
            ):
                continue
            result.append((span, f"{span.id}:{m.start()}-{m.end()}", m.group()))
    return result


def ground_nodes(nodes, document):
    candidates = []
    original = {}
    missing = []
    for node in nodes:
        name = node.get("name") or node.get("properties", {}).get("Name", "")
        label = normalize_label(node.get("label"))
        matches = mentions(name, document)
        if not matches or not label:
            missing.append(str(node.get("id")))
            continue
        for span, ref, text in matches:
            candidates.append(
                EntityCandidate(
                    ref,
                    label,
                    (
                        "required"
                        if label == "Diagnostic Criterion"
                        else "core" if label == "Disease" else "associated"
                    ),
                    text,
                )
            )
        original[str(node.get("id"))] = (label, normalize_text(matches[0][2]))
    entities, _, _ = normalize_candidates(candidates, document)
    by_key = {(e.label, normalize_text(e.name)): e.id for e in entities}
    mapping = {old: by_key[key] for old, key in original.items() if key in by_key}
    return entities, mapping, missing


@dataclass
class PairInput:
    source: str
    target: str
    source_name: str
    target_name: str
    source_label: str
    target_label: str
    allowed: list[str]
    span_ids: list[str]
    context: str
    distance: int

    def prompt(self):
        # No pair ID, ordinal or gold-dependent endpoint list is exposed.
        return (
            f"SOURCE ({self.source_label}): {self.source_name}\n"
            f"TARGET ({self.target_label}): {self.target_name}\n"
            f"SOURCE TEXT:\n{self.context}\n"
            "Classify only relations explicitly supported for this ordered pair.\n"
        )


def pair_inputs(entities, document, max_distance=2, max_context_chars=2200):
    """Bound local evidence, never truncate an ordered global prefix of pairs.

    A candidate may remain absent when there is no bounded explicit context;
    callers must measure this recall ceiling rather than inject gold endpoints.
    """
    positions = {
        e.id: sorted(
            {
                i
                for i, s in enumerate(document.spans)
                if any(ref.split(":")[0] == s.id for ref in e.span_ids)
            }
        )
        for e in entities
    }
    results = []
    for source in entities:
        for target in entities:
            if source.id == target.id:
                continue
            allowed = list(allowed_relations(source.label, target.label))
            if not allowed:
                continue
            windows = set()
            for i in positions[source.id]:
                for j in positions[target.id]:
                    if abs(i - j) <= max_distance:
                        windows.add((min(i, j), max(i, j)))
            if not windows:
                continue
            # Multiple occurrences: expose at most two nearest short windows.
            selected = []
            used = set()
            chars = 0
            for a, b in sorted(windows, key=lambda w: (w[1] - w[0], w[0])):
                spans = document.spans[a : b + 1]
                extra = [s for s in spans if s.id not in used]
                if chars + sum(len(s.text) for s in extra) > max_context_chars:
                    continue
                selected.extend(extra)
                used.update(s.id for s in extra)
                chars += sum(len(s.text) for s in extra)
                if len(used) >= 3 or len(selected) > 0 and len(windows) == 1:
                    break
                if len(selected) >= 2:
                    break
            if not selected:
                continue
            selected.sort(key=lambda s: s.start)
            context = "\n".join(s.text for s in selected)
            if not mentions_in_text(source.name, context) or not mentions_in_text(
                target.name, context
            ):
                continue
            results.append(
                PairInput(
                    source.id,
                    target.id,
                    source.name,
                    target.name,
                    source.label,
                    target.label,
                    allowed,
                    [s.id for s in selected],
                    context,
                    min(b - a for a, b in windows),
                )
            )
    return sorted(
        results,
        key=lambda p: (
            normalize_text(p.source_name),
            p.source_label,
            normalize_text(p.target_name),
            p.target_label,
        ),
    )


def mentions_in_text(name, text):
    return normalize_text(name) in normalize_text(text)


def evidence_spans(evidence, document):
    """Return exact span overlap of the supplied evidence; no fabricated fallback."""
    if isinstance(evidence, list):
        return set().union(*(evidence_spans(e, document) for e in evidence))
    evidence = str(evidence or "").strip()
    if not evidence:
        return set()
    if evidence in document.span_map():
        return {evidence}
    pattern = re.compile(r"\s+".join(map(re.escape, evidence.split())), re.I)
    hit = pattern.search(document.raw)
    if not hit:
        return set()
    return {s.id for s in document.spans if s.start < hit.end() and s.end > hit.start()}


def relation_rejection(relation, nodes):
    """Narrow, auditable rules; a missing literal alone does not reject coreference."""
    by_id = {str(n["id"]): n for n in nodes}
    source = by_id.get(str(relation.get("source")))
    target = by_id.get(str(relation.get("target")))
    if not source or not target:
        return "missing_endpoint"
    # Explicit, optional semantic guard, not a target constraint claimed to be
    # written in the source-only ontology. All 2,689 corresponding train edges
    # use Symptom -> Disease. It also catches a disease surface mistyped Symptom.
    if relation.get("relation") in {
        "is_core_symptom_of",
        "is_associated_symptom_of",
    } and (source["label"] != "Symptom" or target["label"] != "Disease"):
        return "symptom_disease_type_mismatch"
    ev = relation.get("evidence", "")
    if isinstance(ev, list):
        ev = "\n".join(ev)
    ev = normalize_text(ev)
    if (
        relation.get("relation") == "is_associated_symptom_of"
        and target["label"] == "Disease"
    ):
        if not mentions_in_text(target["name"], ev):
            other = [
                n
                for n in nodes
                if n["label"] == "Disease"
                and n["id"] != target["id"]
                and mentions_in_text(n["name"], ev)
            ]
            if other and mentions_in_text(source["name"], ev):
                return "different_disease_in_evidence"
    if (
        relation.get("relation") == "has_manifestation"
        and source["label"] == target["label"] == "Symptom"
    ):
        # Reject only two comma-separated siblings inside the same 'such as'
        # enumeration; parent->child manifestations remain possible.
        match = re.search(r"\bsuch as\s+(.+?)(?:\s+are\b|\s+is\b|[.;]|$)", ev)
        if match:
            items = [
                normalize_text(s.strip(" ,"))
                for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", match.group(1))
            ]
            if (
                normalize_text(source["name"]) in items
                and normalize_text(target["name"]) in items
            ):
                return "enumerated_siblings"
    return None

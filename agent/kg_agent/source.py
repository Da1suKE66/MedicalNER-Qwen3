"""Source routing, deterministic ICD parsing, and stable text spans."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import Entity, Graph, Relation, SourceDocument, Span


ICD_KEYS = {
    "uri",
    "icd_uri",
    "code",
    "icd_code",
    "title",
    "canonical_title",
    "parent",
    "ancestor",
    "ancestors",
    "descendant",
    "descendants",
    "exclusion",
    "exclusions",
    "inclusion",
    "inclusions",
    "synonym",
    "synonyms",
    "indexterm",
    "index_terms",
}


def _key(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    normalized = {_key(key): value for key, value in mapping.items()}
    for name in names:
        value = normalized.get(_key(name))
        if value not in (None, "", []):
            return value
    return None


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        # A concept object is one item; a map of concepts is a collection.
        concept_keys = {"name", "title", "label", "term", "code", "uri", "id"}
        if any(_key(key) in concept_keys for key in value):
            return [value]
        return list(value.values())
    return [value]


def display_name(value: Any) -> str:
    if isinstance(value, Mapping):
        found = _first(value, "title", "name", "label", "term", "display", "text")
        if found is not None:
            return str(found).strip()
        code = _first(value, "code", "id", "uri")
        return str(code).strip() if code is not None else ""
    return str(value).strip()


def normalize_source_text(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n")


_ICD_RELEASE_URI_RE = re.compile(r"^https?://id\.who\.int/icd/release/11/[^/]+/(?:mms|entity)/")
_ICD_CODE_RE = re.compile(r"^[0-9A-Z]{2,}(?:\.[0-9A-Z]+)*$", re.IGNORECASE)


def extract_source_payload(raw: str) -> Any:
    """Remove the legacy chat wrapper and decode its common ICD text form.

    The held-out corpus stores both free text and ICD records inside a
    ``Medical text:`` prompt.  Treating that wrapper as clinical source text
    makes the span selector spend capacity on instructions and bypasses the
    deterministic ICD branch.  This helper keeps the public router tolerant
    of both the old wrapper and native JSON/dict inputs.
    """

    text = normalize_source_text(raw)
    marker = re.search(r"(?im)^\s*Medical text:\s*$", text)
    payload = text[marker.end() :].strip() if marker else text.strip()

    if payload.startswith("{") or payload.startswith("["):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass

    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(lines) < 3 or not _ICD_RELEASE_URI_RE.match(lines[0]) or not _ICD_CODE_RE.match(lines[1]):
        return payload if marker else raw

    data: dict[str, Any] = {"uri": lines[0], "code": lines[1], "title": lines[2]}
    labelled_lists: list[tuple[int, list[dict[str, Any]]]] = []
    for index, line in enumerate(lines[3:], start=3):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
            continue
        if any("label" in item for item in value):
            labelled_lists.append((index, value))

    # In the corpus export, a fully referenced labelled list is the explicit
    # exclusion list.  Other labelled lists are often child/ancestor lookup
    # payloads whose first items have empty foundationReference values; those
    # cannot be safely mapped to graph concepts from this text alone.  Do not
    # turn them into descendants, because that creates many false subtype or
    # exclusion edges.  Deprecated-label lists are aliases.
    fully_referenced_lists = [
        (index, values)
        for index, values in labelled_lists
        if all(
            str(item.get("foundationReference") or item.get("foundation_reference") or "").strip()
            and str(item.get("label") or "").strip()
            for item in values
        )
    ]
    if fully_referenced_lists:
        _, exclusion_values = fully_referenced_lists[0]
        data["exclusions"] = [
            {"title": str(item.get("label") or "").strip(), "uri": str(item.get("foundationReference") or item.get("foundation_reference") or "").strip()}
            for item in exclusion_values
            if str(item.get("label") or "").strip()
        ]

    aliases: list[str] = []
    for _, values in labelled_lists:
        if values and all("deprecated" in item for item in values):
            aliases.extend(str(item.get("label") or "").strip() for item in values if str(item.get("label") or "").strip())
    if aliases:
        data["synonyms"] = aliases
    return data


class SentenceSpanSplitter:
    """Split text without losing character offsets.

    This is deliberately conservative: section headings and punctuation are
    recognized by rules, while the LLM decides which spans are clinical
    concepts.  Every span is assigned a deterministic SENT_NNN ID.
    """

    heading_re = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?P<header>[\w][^\n]{0,100}?)\s*:?\s*$",
        re.UNICODE,
    )
    sentence_re = re.compile(r"[^\n。！？!?\.]+(?:[。！？!?。]|\.(?=\s|$))?|[^\n]+$", re.UNICODE)
    max_span_chars = 1800

    def split(self, text: str) -> list[Span]:
        text = normalize_source_text(text)
        if not text.strip():
            return []

        sections: list[tuple[int, int, str]] = []
        for line in re.finditer(r"[^\n]*(?:\n|$)", text):
            raw_line = line.group(0).rstrip("\n")
            stripped = raw_line.strip()
            if not stripped or len(stripped) > 100:
                continue
            heading = self.heading_re.match(raw_line)
            if not heading:
                continue
            header = heading.group("header").strip(" #:")
            if self._looks_like_heading(header):
                sections.append((line.start(), line.end(), header))

        def section_at(offset: int) -> str:
            current = "body"
            for start, _, name in sections:
                if start <= offset:
                    current = name
                else:
                    break
            return current

        raw_spans: list[Span] = []
        for sentence_start, sentence_end in self.sentence_ranges(text):
            raw = text[sentence_start:sentence_end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start = sentence_start + leading
            end = sentence_start + trailing
            if end <= start:
                continue
            value = text[start:end]
            if not value.strip():
                continue
            raw_spans.append(
                Span(
                    id=f"SENT_{len(raw_spans) + 1:03d}",
                    text=value,
                    start=start,
                    end=end,
                    section=section_at(start),
                )
            )

        # Medical JSON-like inputs and long diagnostic sections may contain a
        # paragraph with no sentence punctuation. Bound it before chunking so
        # one malformed/long field cannot recreate the old 100k-token prompt.
        spans: list[Span] = []
        for original in raw_spans:
            cursor = 0
            while cursor < len(original.text):
                proposed_end = min(cursor + self.max_span_chars, len(original.text))
                if proposed_end < len(original.text):
                    boundary = max(
                        original.text.rfind(" ", cursor + self.max_span_chars // 2, proposed_end),
                        original.text.rfind(";", cursor + self.max_span_chars // 2, proposed_end),
                        original.text.rfind("，", cursor + self.max_span_chars // 2, proposed_end),
                        original.text.rfind(",", cursor + self.max_span_chars // 2, proposed_end),
                    )
                    if boundary > cursor:
                        proposed_end = boundary
                piece = original.text[cursor:proposed_end]
                left_trim = len(piece) - len(piece.lstrip())
                right_trim = len(piece.rstrip())
                start = original.start + cursor + left_trim
                end = original.start + cursor + right_trim
                if end > start:
                    spans.append(
                        Span(
                            id=f"SENT_{len(spans) + 1:03d}",
                            text=text[start:end],
                            start=start,
                            end=end,
                            section=original.section,
                        )
                    )
                cursor = max(proposed_end, cursor + 1)
        return spans

    @staticmethod
    def sentence_ranges(text: str):
        """Keep decimals and common abbreviations intact; preserve every offset."""
        for line in re.finditer(r"[^\n]+", text):
            start = line.start()
            for mark in re.finditer(r"[.!?。！？]+", line.group()):
                end = line.start() + mark.end()
                if mark.group() == ".":
                    if end < line.end() and not text[end].isspace():
                        continue
                    prefix = text[start:end]
                    if re.search(r"(?:\b(?:e\.g|i\.e|Dr|Mr|Mrs|Ms|Prof|vs|etc)|\b[A-Z])\.$", prefix, re.I):
                        continue
                yield start, end
                start = end
            if start < line.end():
                yield start, line.end()

    @staticmethod
    def _looks_like_heading(value: str) -> bool:
        if not value or len(value) > 90:
            return False
        if value.endswith((".", "。", "！", "？", "!", "?")):
            return False
        words = value.split()
        if len(words) > 12:
            return False
        # A one-line sentence is not a heading merely because it has no period.
        return len(words) <= 8 or value.isupper() or value.endswith(":")


def chunk_spans(spans: Iterable[Span], max_spans: int = 16, max_chars: int = 6000) -> list[list[Span]]:
    chunks: list[list[Span]] = []
    current: list[Span] = []
    chars = 0
    for span in spans:
        if current and (len(current) >= max_spans or chars + len(span.text) > max_chars):
            chunks.append(current)
            current = []
            chars = 0
        current.append(span)
        chars += len(span.text)
    if current:
        chunks.append(current)
    return chunks


def resolve_span_ref(document: SourceDocument, reference: str) -> Span | None:
    """Resolve SENT_003 or a sentence-local SENT_003:start-end reference."""

    direct = document.span_map().get(reference)
    if direct is not None:
        return direct
    match = re.fullmatch(r"(?P<base>SENT_\d{3,}):(?P<start>\d+)-(?P<end>\d+)", reference.strip())
    if not match:
        return None
    base = document.span_map().get(match.group("base"))
    if base is None:
        return None
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start < 0 or end <= start or end > len(base.text):
        return None
    return Span(
        id=reference,
        text=base.text[start:end],
        start=base.start + start,
        end=base.start + end,
        section=base.section,
    )


class SourceRouter:
    """Route JSON-like ICD records away from the LLM."""

    def __init__(self, splitter: SentenceSpanSplitter | None = None):
        self.splitter = splitter or SentenceSpanSplitter()

    def route(self, raw: Any, metadata: Mapping[str, Any] | None = None) -> SourceDocument:
        parsed: Any = raw
        if isinstance(raw, str):
            parsed = extract_source_payload(raw)

        if isinstance(parsed, Mapping) and self._is_icd_record(parsed):
            return SourceDocument(
                source_type="structured_icd",
                raw=json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                metadata={**dict(metadata or {}), **dict(parsed)},
            )

        text = normalize_source_text(parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False))
        return SourceDocument(
            source_type="free_text",
            raw=text,
            metadata=dict(metadata or {}),
            spans=self.splitter.split(text),
        )

    @staticmethod
    def _is_icd_record(record: Mapping[str, Any]) -> bool:
        keys = {_key(key) for key in record}
        return len(keys & ICD_KEYS) >= 2 or bool(keys & {"icd_code", "icd_uri"})


def parse_icd_graph(record: Mapping[str, Any] | SourceDocument) -> Graph:
    """Parse common ICD fields into a deterministic graph.

    Synonyms and index terms are retained as aliases on the canonical disease;
    they are not promoted to graph nodes.  The policy can be changed in this
    function without changing either LLM stage.
    """

    if isinstance(record, SourceDocument):
        data = record.metadata
    else:
        data = record

    entities: list[Entity] = []
    relations: list[Relation] = []
    entity_by_key: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {"D": 1}

    title = display_name(_first(data, "title", "canonical_title", "name", "label"))
    title = title or "Unnamed disease"
    code = _first(data, "code", "icd_code")
    uri = _first(data, "uri", "icd_uri")

    def concept_key(name: str) -> tuple[str, str]:
        return "Disease", " ".join(name.casefold().split())

    def ensure_disease(value: Any, *, main: bool = False, path: str = "") -> str | None:
        name = display_name(value)
        if not name:
            return None
        key = concept_key(name)
        if key in entity_by_key:
            return entity_by_key[key]
        if main:
            node_id = "D1"
        else:
            counters["D"] = counters.get("D", 1) + 1
            node_id = f"D{counters['D']}"
        properties: dict[str, Any] = {"Name": name}
        if isinstance(value, Mapping):
            item_code = _first(value, "code", "icd_code")
            item_uri = _first(value, "uri", "icd_uri")
            if item_code not in (None, ""):
                properties["Code"] = item_code
            if item_uri not in (None, ""):
                properties["URI"] = item_uri
        if main:
            if code not in (None, ""):
                properties["Code"] = code
            if uri not in (None, ""):
                properties["URI"] = uri
        entities.append(Entity(node_id, "Disease", name, properties))
        entity_by_key[key] = node_id
        return node_id

    main_id = ensure_disease(title, main=True)
    assert main_id == "D1"

    aliases: list[str] = []
    for field_name in ("synonyms", "synonym", "indexTerms", "index_terms", "inclusions", "inclusion"):
        aliases.extend(name for item in _items(_first(data, field_name)) if (name := display_name(item)))
    if aliases:
        entities[0].properties["Aliases"] = sorted(dict.fromkeys(aliases), key=str.casefold)

    def add_relation(source: str | None, target: str | None, relation: str, path: str, evidence: Any) -> None:
        if not source or not target or source == target:
            return
        evidence_text = display_name(evidence)
        candidate = Relation(
            source=source,
            target=target,
            relation=relation,
            evidence_ids=[path],
            evidence_text=[evidence_text] if evidence_text else [],
        )
        same = next(
            (item for item in relations if item.source == source and item.target == target and item.relation == relation),
            None,
        )
        if same is None:
            relations.append(candidate)
        else:
            for evidence_id in candidate.evidence_ids:
                if evidence_id not in same.evidence_ids:
                    same.evidence_ids.append(evidence_id)
            for text in candidate.evidence_text:
                if text not in same.evidence_text:
                    same.evidence_text.append(text)

    for index, item in enumerate(_items(_first(data, "descendants", "descendant"))):
        child = ensure_disease(item, path=f"descendants[{index}]")
        add_relation(child, main_id, "subtype_of", f"descendants[{index}]", item)

    for field_name in ("parent", "ancestor", "ancestors"):
        for index, item in enumerate(_items(_first(data, field_name))):
            parent = ensure_disease(item, path=f"{field_name}[{index}]")
            add_relation(main_id, parent, "subtype_of", f"{field_name}[{index}]", item)

    for index, item in enumerate(_items(_first(data, "exclusions", "exclusion"))):
        excluded = ensure_disease(item, path=f"exclusions[{index}]")
        add_relation(main_id, excluded, "rules_out", f"exclusions[{index}]", item)

    return Graph(entities=entities, relations=relations)

"""Conservative text repair with offsets into the immutable input string.

All offsets are half-open Python Unicode-code-point indices, not UTF-8 bytes or
UTF-16 indices.  normalize_with_offsets preserves case and whitespace structure
(except CRLF/lone CR -> LF and NFKC's own whitespace conversions).  Matching
folds case and whitespace in a separate, symmetric comparison view.

An output character's offset pair covers all of its contributing input
characters.  Expansions share a range; compositions, reordered marks and repaired
words can have overlapping ranges.  map_range returns their enclosing interval,
not an invented one-to-one alignment.  Keep original_text alongside the mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, NamedTuple
import unicodedata


__all__ = [
    "normalize_with_offsets",
    "map_range",
    "find_mentions",
    "repair_graph_text",
]


class _Unit(NamedTuple):
    char: str
    start: int
    end: int


def _bounds(units: Sequence[_Unit]) -> tuple[int, int]:
    return min(u.start for u in units), max(u.end for u in units)


def _word_char(char: str) -> bool:
    # Combining marks are word continuations too: do not match an unaccented
    # fragment inside a marked word merely because Python's \w excludes marks.
    return char == "_" or char.isalnum() or unicodedata.category(char)[0] == "M"


@lru_cache(maxsize=1)
def _corrections() -> dict[str, str]:
    # Reuse the same audited dictionary as generative.protocol, without importing
    # the protocol/pipeline (which may themselves import this independent module).
    path = Path(__file__).resolve().parents[2] / "schemas/v2.0.0/schema.json"
    return json.loads(path.read_text(encoding="utf-8"))["text_corrections"]


@lru_cache(maxsize=1)
def _correction_pattern() -> re.Pattern[str]:
    keys = sorted(_corrections(), key=lambda key: (-len(key), key))
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, keys)) + r")(?!\w)", re.I)


def _known_matches(text: str):
    for match in _correction_pattern().finditer(text):
        a, b = match.span()
        if (a and _word_char(text[a - 1])) or (b < len(text) and _word_char(text[b])):
            continue
        yield match


def _nfkc_units(raw: str) -> list[_Unit]:
    """NFKD decomposition/order followed by NFC composition, retaining origins.

Per-character NFKC alone is wrong for combining marks and Hangul Jamo.  Canonical
ordering must cross input-character boundaries, and composition must respect
blocking by intervening combining classes.  Python's NFC supplies the Unicode
composition table, including composition exclusions and algorithmic Hangul.
    """
    ordered: list[_Unit] = []
    marks: list[_Unit] = []
    for index, char in enumerate(raw):
        for part in unicodedata.normalize("NFKD", char):
            unit = _Unit(part, index, index + 1)
            if unicodedata.combining(part):
                marks.append(unit)
            else:
                ordered.extend(sorted(marks, key=lambda u: unicodedata.combining(u.char)))
                marks.clear()
                ordered.append(unit)
    ordered.extend(sorted(marks, key=lambda u: unicodedata.combining(u.char)))

    result: list[_Unit] = []
    starter: int | None = None
    last_class = 0
    for unit in ordered:
        combining_class = unicodedata.combining(unit.char)
        if starter is not None and (last_class == 0 or last_class < combining_class):
            previous = result[starter]
            composed = unicodedata.normalize("NFC", previous.char + unit.char)
            if len(composed) == 1:
                result[starter] = _Unit(
                    composed, min(previous.start, unit.start), max(previous.end, unit.end)
                )
                continue
        if combining_class == 0:
            starter = len(result)
        result.append(unit)
        last_class = combining_class
    return result


def _line_endings(units: list[_Unit]) -> list[_Unit]:
    result: list[_Unit] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if unit.char == "\r":
            if index + 1 < len(units) and units[index + 1].char == "\n":
                following = units[index + 1]
                unit = _Unit("\n", min(unit.start, following.start), max(unit.end, following.end))
                index += 1
            else:
                unit = _Unit("\n", unit.start, unit.end)
        result.append(unit)
        index += 1
    return result


def _repair_units(units: list[_Unit]) -> list[_Unit]:
    text = "".join(u.char for u in units)
    result: list[_Unit] = []
    cursor = 0
    for match in _known_matches(text):
        a, b = match.span()
        before = match.group()
        after = _corrections()[before.lower()]
        if before.isupper():
            after = after.upper()
        elif before[0].isupper():
            after = after[0].upper() + after[1:]
        result.extend(units[cursor:a])
        raw_start, raw_end = _bounds(units[a:b])
        # A corrected word is an atomic edit: partial ranges conservatively map
        # to the whole original word, not a guessed character alignment.
        result.extend(_Unit(char, raw_start, raw_end) for char in after)
        cursor = b
    result.extend(units[cursor:])
    return result


def _changes(raw: str, units: list[_Unit]) -> list[dict[str, Any]]:
    """Describe disjoint final edits, combining overlapping origin intervals."""
    changes: list[dict[str, Any]] = []
    # A later canonically reordered mark can reach back past an immediately
    # adjacent unit.  Only close a group when *all* remaining origins lie after
    # it, otherwise recorded edits would overlap and could not be replayed.
    suffix_start = [len(raw)] * (len(units) + 1)
    for index in range(len(units) - 1, -1, -1):
        suffix_start[index] = min(units[index].start, suffix_start[index + 1])
    start = 0
    while start < len(units):
        raw_start, raw_end = units[start].start, units[start].end
        end = start + 1
        while end < len(units) and suffix_start[end] < raw_end:
            raw_start = min(raw_start, units[end].start)
            raw_end = max(raw_end, units[end].end)
            end += 1
        before = raw[raw_start:raw_end]
        after = "".join(u.char for u in units[start:end])
        if before != after:
            changes.append({
                "kind": "normalization",
                "applied": True,
                "start": start,
                "end": end,
                "original_start": raw_start,
                "original_end": raw_end,
                "before": before,
                "after": after,
            })
        start = end
    return changes


def normalize_with_offsets(raw: str, repair: bool = True) -> dict[str, Any]:
    """Return text, original_text, offsets, changes, and offset_unit.

offsets[i] is [raw_start, raw_end] for output character i.  Every change's
start/end indexes the final text; original_start/original_end indexes raw.
The input must be the actual immutable source, not a previously normalized copy.
repair=False disables only dictionary correction, not NFKC or line endings.
Case and non-line-ending whitespace are retained, and no words, punctuation,
negation or numeric separators are removed.  Applying this function again to
its text is text-idempotent, but naturally creates a *new* original_text basis.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    if not isinstance(repair, bool):
        raise TypeError("repair must be a bool")
    units = _line_endings(_nfkc_units(raw))
    if repair:
        units = _repair_units(units)
    return {
        "text": "".join(u.char for u in units),
        "original_text": raw,
        "offsets": [[u.start, u.end] for u in units],
        "changes": _changes(raw, units),
        "offset_unit": "unicode_codepoint",
    }


def map_range(mapping: Mapping[str, Any], start: int, end: int) -> tuple[int, int]:
    """Map a half-open final-text range to its covering original-text interval.

This also accepts a JSON-round-tripped normalize_with_offsets result.  At a
zero-width boundary inside an expansion/atomic edit, the answer is the covering
ambiguous interval, not an exact empty cursor.  Exact boundaries return (p, p).
Invalid/reversed/out-of-bounds ranges raise rather than silently clipping.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be a normalize_with_offsets result")
    if type(start) is not int or type(end) is not int:
        raise TypeError("range endpoints must be integers")
    text, raw, offsets = mapping["text"], mapping["original_text"], mapping["offsets"]
    if not isinstance(text, str) or not isinstance(raw, str):
        raise TypeError("mapping text and original_text must be strings")
    if len(offsets) != len(text):
        raise ValueError("offset count must equal normalized text length")
    if not 0 <= start <= end <= len(text):
        raise ValueError("range is outside normalized text")

    def bounds_at(index: int) -> tuple[int, int]:
        pair = offsets[index]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each offset must be a start/end pair")
        a, b = pair
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(raw):
            raise ValueError("invalid original offset")
        return a, b

    if start == end:
        if start == 0:
            return 0, 0
        if start == len(text):
            return len(raw), len(raw)
        left, right = bounds_at(start - 1), bounds_at(start)
        if left[1] <= right[0]:
            return left[1], right[0]
        return min(left[0], right[0]), max(left[1], right[1])
    pairs = [bounds_at(index) for index in range(start, end)]
    return min(a for a, _ in pairs), max(b for _, b in pairs)


@lru_cache(maxsize=8)
def _comparison_units(text: str) -> tuple[_Unit, ...]:
    # Many entity/evidence queries share one document.  Cache only eight
    # immutable comparison views; no source or mapping is persisted to disk.
    folded = [
        _Unit(char, unit.start, unit.end)
        for unit in _nfkc_units(text)
        for char in unit.char.casefold()
    ]
    result: list[_Unit] = []
    whitespace: list[_Unit] = []
    for unit in folded:
        if unit.char.isspace():
            whitespace.append(unit)
            continue
        if whitespace and result:
            a, b = _bounds(whitespace)
            result.append(_Unit(" ", a, b))
        whitespace.clear()
        result.append(unit)
    return tuple(result)  # Trim comparison whitespace only.


def find_mentions(needle: str, normalized_text: str) -> list[tuple[int, int]]:
    """Find all (including overlapping) matches, indexed into normalized_text.

Both arguments receive NFKC, Unicode casefold and whitespace folding; neither is
dictionary-repaired here.  Repair first, when desired, using the explicit repair
APIs.  A word-like query endpoint requires a word boundary only at that endpoint.
Regex punctuation is literal; punctuation-only endpoints do not get unconditional
lookarounds.  No fuzzy spelling, accent stripping, digit/negation/punctuation
deletion or semantic aliasing is performed.
    """
    if not isinstance(needle, str) or not isinstance(normalized_text, str):
        raise TypeError("needle and normalized_text must be strings")
    query = "".join(u.char for u in _comparison_units(needle))
    if not query:
        return []
    units = _comparison_units(normalized_text)
    text = "".join(u.char for u in units)
    hits = set()
    for match in re.finditer("(?=" + re.escape(query) + ")", text):
        start = match.start()
        end = start + len(query)
        if _word_char(query[0]) and start and _word_char(text[start - 1]):
            continue
        if _word_char(query[-1]) and end < len(text) and _word_char(text[end]):
            continue
        hits.add(_bounds(units[start:end]))
    return sorted(hits)


# Explicit prose fields, compared case/underscore-insensitively.  Unknown fields
# remain unchanged with applied=False diagnostics when normalization is pending.
_TEXT_FIELDS = frozenset({
    "name", "names", "title", "titles", "text", "texts", "description",
    "descriptions", "definition", "definitions", "evidence", "evidences",
    "evidencetext", "evidencesentence", "evidencesentences", "evidencefragment",
    "evidencefragments", "sentence", "sentences", "summary", "summaries",
    "comment", "comments", "note", "notes", "alias", "aliases", "synonym",
    "synonyms", "term", "terms", "caption", "abstract", "diagnosticcriteria",
    "diagnosticrequirements", "essentialrequiredfeatures", "additionalclinicalfeatures",
    "boundarywithnormality", "differentialdiagnosis", "developmentalpresentations",
    "culturerelatedfeatures", "sexandgenderrelatedfeatures", "inclusionnotes",
    "exclusionnotes", "inclusionterms", "exclusionterms",
})
_PROTECTED_FIELDS = frozenset({
    "id", "ids", "identifier", "identifiers", "code", "codes", "uri", "uris",
    "url", "urls", "iri", "iris", "label", "labels", "type", "types", "relation",
    "relationtype", "relationlabel", "relationname", "predicate", "source", "target",
    "head", "tail", "subject", "object", "schema", "version", "ontology",
    "foundationreference", "reference", "references", "ref", "refs", "start", "end",
    "offset", "offsets", "confidence", "score", "originaltext", "originalrecord",
    "originalspans", "originalsource", "originalpayload", "rawtext", "rawsource",
    "provenance",
})
_PROTECTED_SUFFIXES = frozenset({
    "id", "ids", "identifier", "identifiers", "code", "codes", "uri", "uris",
    "url", "urls", "iri", "iris", "uuid", "uuids", "version", "hash", "sha256",
    "offset", "offsets", "type", "types", "label", "labels", "schema",
})


def _field_kind(key: Any, inherited_text: bool) -> str:
    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(key))
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    words = re.findall(r"[a-z0-9]+", separated.lower())
    compact = "".join(words)
    if (compact in _PROTECTED_FIELDS or
            (words and (words[-1] in _PROTECTED_SUFFIXES or words[0] in {"schema", "ontology"}))):
        return "protected"
    if compact in _TEXT_FIELDS or inherited_text:
        return "text"
    return "unknown"


def repair_graph_text(graph: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Return a deep copy and local-field changes/diagnostics; never relabel.

Known descriptive fields (including Definition) are normalized/repaired through
nested lists/dicts.  That descriptive context is inherited, but protected IDs,
codes, URIs, types, relations, schemas and provenance subtrees always win.
Unknown string fields are *not* modified: pending edits are emitted with kind
unclassified_text_field and applied=False.  Filter on applied when counting edits.
Paths use JSON Pointer escaping.  Change offsets are local to that string field,
not source-document offsets.  Legal nulls, booleans and numbers remain unchanged;
the function does not serialize JSON or replace arbitrary 'null' substrings.
    """
    result = copy.deepcopy(graph)
    changes: list[dict[str, Any]] = []

    def visit(value: Any, path: str, text_field: bool = False) -> Any:
        if isinstance(value, dict):
            for key, item in value.items():
                kind = _field_kind(key, text_field)
                if kind == "protected":
                    continue
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                value[key] = visit(item, f"{path}/{escaped}", kind == "text")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = visit(item, f"{path}/{index}", text_field)
        elif isinstance(value, str):
            mapping = normalize_with_offsets(value)
            for change in mapping["changes"]:
                event = {"path": path, **change}
                if not text_field:
                    event.update(kind="unclassified_text_field", applied=False)
                changes.append(event)
            if text_field:
                return mapping["text"]
        return value

    return visit(result, ""), changes

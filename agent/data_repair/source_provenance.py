"""Lossless source views; no source repair, list-role inference, or graph creation.

``source_view`` accepts the ORIGINAL user string, not ``SourceDocument.raw``.
All public ``start``/``end`` offsets are half-open Python string character
offsets into that user string (not bytes, normalized text, or tokenizer IDs).
``payload_start``/``payload_end`` on nested fragments instead index raw_payload.

The separate ``routed`` dict is exactly ``asdict(SourceRouter().route(user))``.
Its unfortunately named ``raw`` field is a LEGACY DERIVED VIEW: it can be NFKC
normalized or reconstructed JSON, and its spans index only that derived text.
In particular, legacy metadata may infer exclusions from an unnamed ICD list.
Neither that inference nor URI syntax is validated semantic evidence here.
New release builders should retain the raw fragments and review labelled_lists,
not create relations from routed metadata. Source/sample indexing belongs to
the caller. All heuristics emit uncertain flags only, never replacement text.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
import json
import re
from typing import Any
from urllib.parse import urlsplit

from kg_agent.source import SourceRouter


VERSION = "source-provenance-v2"
_MARKER = re.compile(
    r"(?:\A|(?<=[\r\n]))[^\S\r\n]*Medical text:[^\S\r\n]*(?:\r\n|\r|\n|$)",
    re.IGNORECASE,
)
_JSON_SPACE = " \t\r\n"
_DECODER = json.JSONDecoder()
_REFERENCE_KEYS = {"foundationreference", "foundation_reference"}
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "vs", "etc", "fig", "no"}


def _fragment(text: str, base: int, start: int, end: int) -> dict[str, Any]:
    return {
        "raw": text[start:end],
        "start": base + start,
        "end": base + end,
        "payload_start": start,
        "payload_end": end,
    }


def _skip_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset] in _JSON_SPACE:
        offset += 1
    return offset


def _line_starts(text: str) -> list[int]:
    return [0] + [m.end() for m in re.finditer(r"\r\n|\r|\n", text)]


def _json_blocks(text: str, base: int) -> tuple[Any, list[dict[str, Any]]]:
    """Keep a whole JSON payload or independent JSON containers in legacy lines.

    A successfully decoded pretty-printed container consumes its interior lines
    once. Bad JSON is retained, not replaced with an empty dict/list. Decoded
    values are convenience views; exact raw fragments remain authoritative
    (including whitespace, escapes, duplicate JSON keys and numeric spelling).
    """
    starts = _line_starts(text)

    def block(start, end, value, error=None):
        return {
            **_fragment(text, base, start, end),
            "line_index": bisect_right(starts, start) - 1,
            "value": value,
            "decoded": error is None,
            "parse_error": error,
        }

    first = _skip_space(text, 0)
    if first < len(text) and text[first] in "[{":
        try:
            value, end = _DECODER.raw_decode(text, first)
            if not text[end:].strip(_JSON_SPACE):
                return value, [block(first, end, value)]
        except json.JSONDecodeError:
            pass

    blocks = []
    consumed = 0
    for line_index, line_start in enumerate(starts):
        if line_start < consumed:
            continue
        line_end = starts[line_index + 1] if line_index + 1 < len(starts) else len(text)
        start = _skip_space(text, line_start)
        if start >= line_end or text[start] not in "[{":
            continue
        try:
            value, end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            blocks.append(block(start, line_end, None, exc.msg))
            consumed = line_end
        else:
            blocks.append(block(start, end, value))
            consumed = end
    return None, blocks


def _json_children(text: str, start: int):
    """Yield exact value boundaries, not substring searches for reserialized JSON."""
    is_object = text[start] == "{"
    stop = "}" if is_object else "]"
    cursor = _skip_space(text, start + 1)
    index = 0
    while text[cursor] != stop:
        key = index
        if is_object:
            key, cursor = _DECODER.raw_decode(text, cursor)
            cursor = _skip_space(text, cursor)
            # The enclosing value was already successfully JSON-decoded.
            cursor = _skip_space(text, cursor + 1)  # colon
        value, end = _DECODER.raw_decode(text, cursor)
        yield key, value, cursor, end
        index += 1
        cursor = _skip_space(text, end)
        if text[cursor] == ",":
            cursor = _skip_space(text, cursor + 1)


def _valid_reference(value: Any) -> bool:
    """HTTP(S) URI syntax only: NOT network resolution or a clinical identity check."""
    if not isinstance(value, str) or not value.strip():
        return False
    value = value.strip()
    if any(c.isspace() for c in value):
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port  # Also reject malformed/out-of-range ports.
        return bool(
            parsed.scheme.lower() in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and (port is None or 0 < port <= 65535)
        )
    except ValueError:
        return False


def _list_item(text, base, index, value, start, end):
    references = []
    reasons = ["list_semantics_unverified"]
    if isinstance(value, dict):
        for key, reference in value.items():
            if key.casefold() in _REFERENCE_KEYS:
                references.append(
                    {
                        "field": key,
                        "value": reference,
                        "nonempty": isinstance(reference, str) and bool(reference.strip()),
                        "valid_uri_syntax": _valid_reference(reference),
                    }
                )
        if not isinstance(value.get("label"), str) or not value["label"].strip():
            reasons.append("missing_or_invalid_label")
    else:
        reasons.append("non_object_item")
    valid = any(r["valid_uri_syntax"] for r in references)
    if not valid:
        reasons.append("missing_or_invalid_reference")
    return {
        **_fragment(text, base, start, end),
        "index": index,
        "value": value,
        "references": references,
        "has_valid_reference": valid,
        "reference_validation": "http_uri_syntax_only_not_resolved",
        "semantic_status": "review",
        "needs_review": True,
        "review_reasons": reasons,
    }


def _labelled_lists(text: str, base: int, blocks: list[dict[str, Any]]):
    results = []
    starts = _line_starts(text)

    def visit(value, start, end, path, block_index, declared_field=None):
        if not isinstance(value, (dict, list)):
            return
        children = list(_json_children(text, start))
        if isinstance(value, list) and any(
            isinstance(v, dict) and "label" in v for _, v, _, _ in children
        ):
            results.append(
                {
                    **_fragment(text, base, start, end),
                    "json_block_index": block_index,
                    "json_pointer": path,
                    "line_index": bisect_right(starts, start) - 1,
                    # A literal key is provenance, not an inferred relation role.
                    "declared_field": declared_field,
                    "semantic_status": "review",
                    "needs_review": True,
                    "items": [
                        _list_item(text, base, i, v, a, b)
                        for i, v, a, b in children
                    ],
                }
            )
        for key, child, a, b in children:
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            visit(
                child, a, b, f"{path}/{escaped}", block_index,
                key if isinstance(value, dict) else None,
            )

    for i, block in enumerate(blocks):
        if block["decoded"]:
            visit(block["value"], block["payload_start"], block["payload_end"], "", i)
    return results


def _text_flags(text: str, base: int) -> list[dict[str, Any]]:
    flags = []

    def flag(code, start, end, reason, **details):
        flags.append(
            {
                **_fragment(text, base, start, end),
                "code": code,
                "uncertain": True,
                "needs_review": True,
                "reason": reason,
                **details,
            }
        )

    # Narrow observed suffix, not a general rule about lowercase sentence starts.
    # Deliberately do not guess the missing word/prefix (e.g. "impai").
    fragment = re.match(r"\s*(rment)\b", text, re.IGNORECASE)
    if fragment:
        flag(
            "suspected_start_truncation", *fragment.span(1),
            "Known isolated leading suffix; missing prefix/content is unknown.",
        )
    for match in re.finditer(r"\b[A-Za-z]{2,}[-\u00ad][ \t]*\r?\n[ \t]*[a-z]{2,}\b", text):
        flag(
            "suspected_word_break", *match.span(),
            "Hyphenated word across lines; may be intentional typesetting.",
        )
    fragment = re.search(r"\b[A-Za-z]{2,}[-\u00ad]\s*\Z", text)
    if fragment:
        flag(
            "suspected_end_truncation", *fragment.span(),
            "Source ends in a hyphenated word fragment; no completion inferred.",
        )

    headings = {}
    for match in re.finditer(r"[^\r\n]+", text):
        line = match.group().strip()
        markdown = re.match(r"#{1,6}[ \t]+(.+)", line)
        body = markdown.group(1) if markdown else line
        is_heading = markdown or (
            len(body) <= 90
            and len(body.split()) <= 12
            and (body.endswith(":") or (body.isupper() and len(body) >= 3))
        )
        if not is_heading:
            continue
        key = " ".join(body.strip(" #: ").split()).casefold()
        if not key:
            continue
        if key in headings:
            flag(
                "duplicate_header", *match.span(),
                "Repeated explicit heading; may indicate overlapping source blocks.",
                first_start=base + headings[key][0],
                first_end=base + headings[key][1],
            )
        else:
            headings[key] = match.span()

    boundary = re.compile(r"\b([A-Za-z]{2,})([.!?])([A-Z][a-z]+)")
    for match in boundary.finditer(text):
        if match.group(2) == "." and match.group(1).lower() in _ABBREVIATIONS:
            continue
        token_start = match.start()
        while token_start and not text[token_start - 1].isspace():
            token_start -= 1
        token_end = match.end()
        while token_end < len(text) and not text[token_end].isspace():
            token_end += 1
        token = text[token_start:token_end]
        if "://" in token or "@" in token:
            continue
        flag(
            "no_space_sentence_boundary", *match.span(),
            "Punctuation immediately followed by a capitalized word; review only.",
        )
    return sorted(flags, key=lambda f: (f["start"], f["end"], f["code"]))


def source_view(user_text: str) -> dict[str, Any]:
    """Return raw provenance plus a clearly separated, unchanged legacy route.

    Strip the preamble through the FIRST standalone ``Medical text:`` line,
    including that line's terminator. Keep every subsequent character, including
    leading blank lines, CRLF and trailing whitespace. Without a marker, keep
    the entire input. ``user_text[payload_start:payload_end] == raw_payload``.

    ``json_payload`` is decoded only for a complete JSON container payload;
    ``json_blocks`` also retains containers embedded in legacy ICD export lines.
    Every member of a labelled list survives independently, even unlabelled,
    non-object or invalid-reference members. List/item indices are zero-based;
    line_index is relative to raw_payload. No entity/relation is generated.
    ``issues`` (also exposed as ``flags``) contains source-risk dictionaries with
    code/needs_review; per-list semantic review stays inside labelled_lists.
    """
    if not isinstance(user_text, str):
        raise TypeError("source_view requires the original user string")
    markers = list(_MARKER.finditer(user_text))
    marker = markers[0] if markers else None
    start = marker.end() if marker else 0
    payload = user_text[start:]
    # Call the public router unchanged, with the original input, for compatibility.
    routed = asdict(SourceRouter().route(user_text))
    parsed, blocks = _json_blocks(payload, start)
    flags = _text_flags(payload, start) if routed["source_type"] == "free_text" else []
    for block in blocks:
        if not block["decoded"]:
            flags.append(
                {
                    **_fragment(payload, start, block["payload_start"], block["payload_end"]),
                    "code": "unparsed_json_block",
                    "uncertain": True,
                    "needs_review": True,
                    "reason": block["parse_error"],
                }
            )
    for extra in markers[1:]:
        flags.append(
            {
                **_fragment(payload, start, extra.start() - start, extra.end() - start),
                "code": "multiple_payload_markers",
                "uncertain": True,
                "needs_review": True,
                "reason": "Only the first standalone wrapper was removed; later text is retained.",
            }
        )
    issues = sorted(flags, key=lambda f: (f["start"], f["end"], f["code"]))
    return {
        "version": VERSION,
        "raw_user": user_text,
        "raw_payload": payload,
        "payload_start": start,
        "payload_end": len(user_text),
        "offset_unit": "python_unicode_character_half_open",
        "wrapper": (
            {"start": marker.start(), "end": marker.end(), "raw": marker.group()}
            if marker else None
        ),
        "source_type": routed["source_type"],
        "routed": routed,
        "routed_is_original": False,
        "routed_semantics_verified": False,
        "routed_span_offset_space": "routed.raw_only_not_original_user",
        "json_payload": parsed,
        "json_blocks": blocks,
        "labelled_lists": _labelled_lists(payload, start, blocks),
        "issues": issues,
        "flags": issues,
    }

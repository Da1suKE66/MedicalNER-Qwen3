"""Immutable 148-case historical regression, with independently proven titles.

This set has been inspected and used historically. It is not a blind test and
can never be admitted as training data. Input provenance is established solely
from original user messages and raw source fields, before reading graph labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import unicodedata

from build_stage_data import _message
from evaluate import _graph, _entity_set, _relation_set, _validate_gold_graph
from kg_agent.source import SourceRouter
from semantic.data import canonical, sha
from semantic.runtime import verify_title_context


VERSION = "semantic-historical-regression-v1"
EVALUATION_SET = "historical_regression"
FIELDS = ("definition", "longDefinition", "diagnosticCriteria")
AUDITED_INPUT_SHA256 = {
    "gold": "09893cf989348136ed6d49561571fb5ba9579d7b5c46240f3d9a0971d466b24f",
    "raw_sources": "a85657742af50bd24bac501ceeae04ddb1c576e033fa3501773a4e6b4327d086",
    "source_match": "8bd81b9e12b75c8d02eb6a184bb62d0f4e86508f30e6aa13c6cc9929b894f695",
}
ORIGINAL_FILES = {"gold": "original_historical_records.json", "raw_sources": "original_raw_sources.json",
                  "source_match": "input_source_match.json"}


def payload_span(user):
    """Unwrap legacy instructions without normalizing any source characters."""
    if not isinstance(user, str):
        raise TypeError("Original user input must be text")
    markers = list(re.finditer(r"(?im)^\s*Medical text:\s*$", user))
    if len(markers) > 1:
        raise ValueError("Ambiguous historical source wrapper")
    a, b = markers[0].end() if markers else 0, len(user)
    while a < b and user[a].isspace(): a += 1
    while b > a and user[b - 1].isspace(): b -= 1
    return a, b


def _all_occurrences(needle, text):
    result, cursor = [], 0
    while needle:
        start = text.find(needle, cursor)
        if start < 0:
            break
        result.append((start, start + len(needle)))
        cursor = start + 1
    return result


def _match_normalize(value):
    # Match the declared diagnostic audit normalization exactly; unlike typed
    # graph scoring, this input provenance diagnostic does not case-fold.
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _legacy_title_span(user, start, end, raw):
    """Locate the title field, even when its value also occurs in other fields.

    These audited legacy records have URI, code and title as the first three
    nonempty payload lines. No arbitrary substring occurrence is selected.
    """
    lines, offset = [], start
    for line in user[start:end].splitlines(keepends=True):
        value = line.strip()
        if value:
            a = offset + len(line) - len(line.lstrip())
            lines.append((value, a, a + len(value)))
        offset += len(line)
    if len(lines) < 3 or [line[0] for line in lines[:3]] != [raw["id"], raw["code"], raw["title"]]:
        raise ValueError("Historical title is not the verified original URI/code/title field")
    return lines[2][1:]


def _source_views(raw_sources):
    rows = raw_sources.get("entities") if isinstance(raw_sources, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Original raw source entity collection required")
    by_id = {}
    for row in rows:
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise ValueError("Original raw source IDs must be unique")
        if not isinstance(row.get("title"), str) or not row["title"]:
            raise ValueError("Original raw source title required")
        if any(row.get(field) is not None and not isinstance(row[field], str) for field in FIELDS):
            raise ValueError("Raw matching fields must contain original text")
        by_id[source_id] = row
    return by_id


def compile_records(originals, raw_sources, source_match, *, expected_samples=148):
    """Recompute every source match; supplied match JSON is a checked claim."""
    if not isinstance(originals, list) or len(originals) != expected_samples:
        raise ValueError("Historical evaluation requires every original case")
    by_source = _source_views(raw_sources)
    inputs, anchors, source_receipts = [], {}, {}
    for index, original in enumerate(originals):
        user = _message(original, "user")
        a, b = payload_span(user)
        text = user[a:b]
        if not text:
            raise ValueError("Empty original historical source")
        routed = SourceRouter().route(user)
        inputs.append({"index": index, "user": user, "text": text, "start": a, "end": b,
                       "source_type": routed.source_type})
        if routed.source_type != "structured_icd":
            continue
        source_id = routed.metadata.get("uri")
        raw = by_source.get(source_id)
        if (raw is None or raw["title"] != routed.metadata.get("title")
                or raw.get("code") != routed.metadata.get("code")):
            raise ValueError("Historical structured URI/title/code differs from original raw source")
        if source_id in anchors:
            raise ValueError("Ambiguous historical structured source anchor")
        title = raw["title"]
        title_start, title_end = _legacy_title_span(user, a, b, raw)
        group = "raw858:" + source_id
        record_sha = sha(original)
        title_context = {"text": title, "source_group_id": group, "source_record_sha256": record_sha,
                         "source_start": title_start, "source_end": title_end}
        anchors[source_id] = {"sample_index": index, "group": group, "title_context": title_context}
        source_receipts[record_sha] = {"source_group_id": group, "original_record": deepcopy(original)}
        inputs[-1].update(source_id=source_id, provenance_status="structured_source_anchor")

    claimed = source_match.get("matches") if isinstance(source_match, dict) else None
    if not isinstance(claimed, list):
        raise ValueError("Historical input matching audit is required")
    claimed_by_index = {}
    for row in claimed:
        index = row.get("sample_index")
        if type(index) is not int or index in claimed_by_index:
            raise ValueError("Duplicate or invalid source match sample identity")
        claimed_by_index[index] = row
    free_indices = {row["index"] for row in inputs if row["source_type"] == "free_text"}
    if set(claimed_by_index) != free_indices:
        raise ValueError("Source matching audit must account for every free-text case and no structured case")
    counts, provenance = Counter(), []
    normalized_fields = {source_id: {field: _match_normalize(raw.get(field) or "") for field in FIELDS}
                         for source_id, raw in by_source.items()}
    for item in inputs:
        index, text = item["index"], item["text"]
        if item["source_type"] != "free_text":
            continue
        matches = []
        normalized = set()
        normalized_text = _match_normalize(text)
        for source_id, raw in by_source.items():
            for field in FIELDS:
                value = raw.get(field) or ""
                for a, b in _all_occurrences(text, value):
                    matches.append({"source_id": source_id, "source_record_sha256": sha(raw), "field": field,
                                    "field_sha256": sha(value), "start": a, "end": b})
                if normalized_text and normalized_text in normalized_fields[source_id][field]:
                    normalized.add(source_id)
        exact = sorted({row["source_id"] for row in matches})
        claim = claimed_by_index[index]
        if claim.get("exact_source_ids") != exact or claim.get("nfkc_whitespace_source_ids") != sorted(normalized):
            raise ValueError(f"Historical input-only source mapping failed independent reconstruction at {index}")
        counts[f"exact:{len(exact)}"] += 1
        counts[f"normalized:{len(normalized)}"] += 1
        # A normalized-only match or ambiguous field occurrence never grants a
        # title. No disease names from graph labels are used as a fallback.
        unique = len(matches) == 1 and len(exact) == 1 and exact[0] in anchors
        anchor = anchors[exact[0]] if unique else None
        if claim.get("input_only_anchor_sample_index") != (anchor["sample_index"] if anchor else None):
            raise ValueError("Claimed historical title anchor is not proven by the source ID")
        if unique:
            item.update(source_id=exact[0], provenance_status="exact_raw_source_and_historical_anchor",
                        match_receipt=matches[0])
        else:
            item.update(source_id=None, provenance_status="unmatched_no_title")
        provenance.append({"sample_index": index, "original_user_sha256": sha(item["user"]),
            "source_text_sha256": sha(text), "status": item["provenance_status"],
            "exact_matches": matches, "normalized_source_ids_diagnostic_only": sorted(normalized),
            "anchor_sample_index": anchor["sample_index"] if anchor else None})
    if source_match.get("counts") != dict(counts):
        raise ValueError("Historical matching aggregate differs from independent reconstruction")
    title_matches = sum(row["status"] == "exact_raw_source_and_historical_anchor" for row in provenance)
    if source_match.get("historical_anchor_counts") != {"unique_exact_match_and_historical_anchor": title_matches}:
        raise ValueError("Historical anchor coverage claim changed")

    # Graph labels are first read here, after all input associations are fixed.
    cases, identities = [], []
    for item, original in zip(inputs, originals):
        index, source_id = item["index"], item.get("source_id")
        group = anchors[source_id]["group"] if source_id is not None else "unmatched_input_sha256:" + sha(item["text"])
        title_context = deepcopy(anchors[source_id]["title_context"]) if source_id is not None else None
        metadata = {"source_group_id": group, "source_record_sha256": sha(original), "source_index": index,
            "source_type": item["source_type"], "source_view_origin": "original_historical_user_payload",
            "source_payload_start": item["start"], "source_payload_end": item["end"],
            "effective_user_sha256": sha(item["user"]), "source_text_sha256": sha(item["text"]),
            "historical_provenance_status": item["provenance_status"], "training_allowed": False}
        if title_context:
            metadata["title_context"] = title_context
            verify_title_context(metadata, source_receipts, source_group_id=group)
        gold = _graph(_message(original, "assistant"))
        if gold is None:
            raise ValueError("Malformed historical original gold graph")
        _validate_gold_graph(gold)
        identity = {"version": VERSION + "-input-v1", "sha256": sha({"text": item["text"], "metadata": metadata}),
                    "source_text_sha256": sha(item["text"]), "source_view_origin": metadata["source_view_origin"]}
        case = {"id": f"historical:{index}:{sha(item['user'])[:16]}", "index": index, "split": EVALUATION_SET,
            "evaluation_set": EVALUATION_SET, "training_allowed": False, "blind_test": False,
            "source_group_id": group, "source_type": item["source_type"], "text": item["text"], "metadata": metadata,
            "gold": deepcopy(gold), "gold_sha256": sha(gold), "original_record": deepcopy(original),
            "original_record_sha256": sha(original), "input_identity": identity,
            "input_provenance": {"status": item["provenance_status"], "raw_source_id": source_id,
                                 "match_receipt": deepcopy(item.get("match_receipt"))}}
        cases.append(case)
        identities.append({"id": case["id"], "sample_index": index, "original_record_sha256": sha(original),
                           "original_user_sha256": sha(item["user"]), "gold_sha256": sha(gold), "input_sha256": identity["sha256"]})
    denominators = {}
    for kind in ("overall", "free_text", "structured_icd"):
        selected = [case for case in cases if kind == "overall" or case["source_type"] == kind]
        denominators[kind] = {"samples": len(selected), "gold_entities": sum(len(_entity_set(case["gold"])) for case in selected),
                              "gold_relations": sum(len(_relation_set(case["gold"])) for case in selected)}
    return cases, source_receipts, {"case_identities": identities, "case_identity_sha256": sha(identities),
        "source_provenance": provenance, "matching_counts": dict(counts), "denominators": denominators,
        "title_context_coverage": {"free_text_with_verified_title": title_matches,
            "free_text_without_title": len(free_indices) - title_matches, "structured_anchors": len(anchors)},
        "all_original_gold_retained": True, "matching_used_graph_labels": False}


def build_release(gold_path, raw_path, match_path, output_dir, *, expected_samples=148, expected_hashes=None, short_chars=2400):
    expected_hashes = AUDITED_INPUT_SHA256 if expected_hashes is None else expected_hashes
    paths = {"gold": Path(gold_path), "raw_sources": Path(raw_path), "source_match": Path(match_path)}
    inputs = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: sha(value) for key, value in inputs.items()}
    if hashes != expected_hashes:
        raise ValueError("Historical input bytes differ from the explicitly audited source hashes")
    cases, receipts, audit = compile_records(*(json.loads(inputs[key]) for key in ("gold", "raw_sources", "source_match")), expected_samples=expected_samples)
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Historical regression releases are immutable; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    for key, filename in ORIGINAL_FILES.items():
        (out / filename).write_bytes(inputs[key])
    (out / "evaluation").mkdir()
    (out / "evaluation/historical_records.jsonl").write_text("".join(canonical(case) + "\n" for case in cases))
    (out / "title_source_receipts.json").write_text(canonical(receipts) + "\n")
    (out / "provenance_audit.json").write_text(canonical(audit) + "\n")
    names = [*ORIGINAL_FILES.values(), "evaluation/historical_records.jsonl", "title_source_receipts.json", "provenance_audit.json"]
    for name in names: (out / name).chmod(0o600)
    manifest = {"data_version": VERSION, "evaluation_set": EVALUATION_SET, "blind_test": False,
        "training_allowed": False, "gold_unchanged": True, "samples": len(cases), "short_chars": short_chars,
        "source_artifacts": hashes, "output_sha256": {name: sha((out / name).read_bytes()) for name in names},
        "case_identity_sha256": audit["case_identity_sha256"], "denominators": audit["denominators"],
        "title_context_coverage": audit["title_context_coverage"],
        "record_counts": dict(Counter(EVALUATION_SET + ":" + case["source_type"] for case in cases)),
        "source_group_policy": "Verified raw858 source URI; unmatched input SHA is unlinked and grants no title. No positional row-index grouping.",
        "context_policy": "Exact unique original source-field containment plus same-ID historical structured anchor; unmatched cases retain original text without a guessed title"}
    (out / "manifest.json").write_text(canonical(manifest) + "\n")
    (out / "manifest.json").chmod(0o600)
    if any(path.read_bytes() != inputs[key] for key, path in paths.items()):
        raise ValueError("Historical input files changed during compilation")
    return manifest


def load_release(directory, *, expected_samples=148):
    """Rebuild source/title associations and each case identity independently."""
    root = Path(directory)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("data_version") != VERSION or manifest.get("evaluation_set") != EVALUATION_SET
            or manifest.get("training_allowed") is not False or manifest.get("blind_test") is not False
            or manifest.get("gold_unchanged") is not True):
        raise ValueError("Historical regression must remain nonblind and forbidden for training")
    required = {*ORIGINAL_FILES.values(), "evaluation/historical_records.jsonl", "title_source_receipts.json", "provenance_audit.json"}
    if set(manifest.get("output_sha256", {})) != required:
        raise ValueError("Historical artifact inventory differs from the fixed release contract")
    for name, digest in manifest["output_sha256"].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError("Historical artifact SHA mismatch: " + name)
    original_bytes = {key: (root / filename).read_bytes() for key, filename in ORIGINAL_FILES.items()}
    hashes = {key: sha(value) for key, value in original_bytes.items()}
    if hashes != manifest["source_artifacts"] or hashes != AUDITED_INPUT_SHA256:
        raise ValueError("Historical originals differ from the audited frozen population")
    expected, receipts, audit = compile_records(*(json.loads(original_bytes[key]) for key in ("gold", "raw_sources", "source_match")), expected_samples=expected_samples)
    cases = [json.loads(line) for line in (root / "evaluation/historical_records.jsonl").read_text().splitlines() if line.strip()]
    if canonical(cases) != canonical(expected):
        raise ValueError("Historical cases differ from exact original gold/input/provenance reconstruction")
    if (json.loads((root / "title_source_receipts.json").read_text()) != receipts
            or json.loads((root / "provenance_audit.json").read_text()) != audit):
        raise ValueError("Historical source or per-case identity receipt changed")
    for key in ("denominators", "title_context_coverage", "case_identity_sha256"):
        if manifest[key] != audit[key]:
            raise ValueError("Historical audit totals differ from reconstruction")
    if manifest["samples"] != len(cases) or manifest["record_counts"] != dict(Counter(EVALUATION_SET + ":" + case["source_type"] for case in cases)):
        raise ValueError("Historical sample/stratum counts changed")
    return cases, {"manifest_sha256": sha(manifest_bytes), "input_files_sha256": manifest["output_sha256"],
        "record_counts": manifest["record_counts"], "data_version": VERSION, "evaluation_set": EVALUATION_SET,
        "source_artifacts": hashes, "short_chars": manifest["short_chars"], "blind_test": False, "training_allowed": False,
        "denominators": audit["denominators"], "title_context_coverage": audit["title_context_coverage"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gold", "raw-sources", "source-match", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    report = build_release(args.gold, args.raw_sources, args.source_match, args.output_dir)
    print(canonical({key: report[key] for key in ("samples", "denominators", "title_context_coverage", "blind_test", "training_allowed")}))


if __name__ == "__main__":
    main()

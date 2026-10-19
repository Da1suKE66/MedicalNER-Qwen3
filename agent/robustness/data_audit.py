"""Read-only, offline DeepSeek/preprocessing audit; reports contain no source text.

Run from the repository with ``PYTHONPATH=agent python -B -m
robustness.data_audit --output-dir /absolute/path/to/a/new/report-directory``.
Only reports are written. No model, tokenizer, network, gold edits, or training
dependencies are required. Source indices are zero-based; JSONL lines are one-based.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

from build_pair_data import source_groups
from build_stage_data import _message
from evaluate import _graph
from generative.build_data import apply_rebalancing, convert
from generative.protocol import (
    ProtocolConfig, build_batches, dumps, exact_hits, original_offset,
    parse_decisions, repair_text, span_hits, supervision,
)
from kg_agent.contracts import ONTOLOGY, allowed_relations, normalize_label
from kg_agent.normalize import normalize_text
from kg_agent.source import SourceRouter


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAME = "deepseek_watermark_20260804_182312_output_only_full_groupdisjoint_train.json"
EXPECTED_SHA256 = "74a81d4bedc5d3f7252d11ef5d57287e1605d164cb55aaca61c9ed1bc81ae05b"
CODE_PATHS = [
    "agent/robustness/data_audit.py", "agent/build_pair_data.py",
    "agent/build_stage_data.py", "agent/evaluate.py", "agent/generative/build_data.py",
    "agent/generative/protocol.py", "agent/generative/chat.py", "agent/train_sft.py",
    "agent/kg_agent/source.py", "agent/kg_agent/normalize.py",
    "agent/kg_agent/contracts.py", "agent/kg_agent/ontology.json",
    "schemas/v2.0.0/schema.json", "agent/data_repair/pipeline.py", "agent/data_repair/verify.py",
]


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def file_sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    # Do not propagate JSONDecodeError: its message may contain private input.
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("Invalid JSON input (content redacted)") from None


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                yield line_number, None, "blank_line"
                continue
            try:
                value = json.loads(line)
            except ValueError:
                yield line_number, None, "invalid_json"
                continue
            yield line_number, value, None


class Findings:
    """Only trusted issue codes, numerical indices and hashes may be reported."""

    def __init__(self, limit=8):
        if limit < 0:
            raise ValueError("Example limit must be nonnegative")
        self.limit = limit
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.sources = defaultdict(set)

    def add(self, code, **location):
        allowed = {"source_index", "entity_index", "relation_index", "line", "batch_index", "sha256"}
        if set(location) - allowed:
            raise ValueError("Unsafe report location field")
        if any(not isinstance(v, int) or isinstance(v, bool) for k, v in location.items() if k != "sha256"):
            raise ValueError("Report indices must be integers")
        if "sha256" in location and not re.fullmatch(r"[0-9a-f]{64}", location["sha256"]):
            raise ValueError("Report hash must be SHA256")
        self.counts[code] += 1
        if "source_index" in location:
            self.sources[code].add(location["source_index"])
        if len(self.examples[code]) < self.limit:
            self.examples[code].append(location)

    def result(self):
        return {"counts": dict(self.counts), "source_record_counts": {k: len(v) for k, v in self.sources.items()}, "examples": dict(self.examples)}


def identifier(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() else None


def node_name(node):
    props = node.get("properties")
    return str(node.get("name") or (props.get("Name") if isinstance(props, dict) else "") or "")


def evidence_check(edge, doc, *, repair=False, visible=None):
    """Independent ALL-fragment check, unlike a union of successfully found hits.

    Literal span IDs remain supported, matching the current protocol. Repeated
    occurrences are counted, not adjudicated; exact matches are not entailment.
    """
    value = edge.get("evidence") or edge.get("evidence_sentence") or ""
    fragments = value if isinstance(value, list) else [value]
    shape_ok = bool(fragments) and all(isinstance(v, str) and v.strip() for v in fragments)
    span_map = doc.span_map()
    found, locations, hit_count = [], [], 0
    for fragment in fragments:
        if not isinstance(fragment, str) or not fragment.strip():
            found.append(False)
            locations.append(set())
            continue
        text = repair_text(fragment)[0] if repair else fragment
        hits = exact_hits(text, doc.raw)
        refs = {text} if text in span_map else span_hits(text, doc)
        found.append(bool(refs))
        locations.append(refs)
        hit_count += len(hits)
    union = set().union(*locations) if locations else set()
    visible = set(span_map) if visible is None else set(visible)
    return {
        "fragments": len(fragments), "shape_ok": bool(shape_ok),
        "all_found": bool(shape_ok and all(found)), "any_found": any(found),
        "partial_found": any(found) and not all(found),
        "all_visible": bool(shape_ok and all(found) and union <= visible),
        "matched_occurrences": hit_count,
    }


def audit_raw(records, *, example_limit=8):
    issues = Findings(example_limit)
    counts = Counter(assistant_strict_json=0, assistant_output_wrapper_json=0, assistant_fenced_json=0, assistant_graphs_valid=0)
    by_type, labels, relation_labels, systems = defaultdict(Counter), Counter(), Counter(), Counter()
    docs, graphs = [], []
    for index, record in enumerate(records):
        loc = {"source_index": index}
        counts["records"] += 1
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            raise ValueError("Expected teacher message records (content redacted)")
        messages = record["messages"]
        if not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Invalid teacher message shape (content redacted)")
        if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
            issues.add("message_role_order_or_count", **loc)
        system, answer = _message(record, "system"), _message(record, "assistant")
        systems[sha(system)] += 1
        counts["schema_version_present"] += ONTOLOGY["version"] in system
        if ONTOLOGY["version"] not in system:
            issues.add("teacher_schema_version_mismatch", **loc)
        counts["assistant_thinking_markers"] += bool(re.search(r"</?think\b", answer, re.I))
        try:
            strict = json.loads(answer)
            counts["assistant_strict_json"] += 1
        except ValueError:
            strict = None
            wrapper = re.fullmatch(r"\s*<output>\s*(.*?)\s*</output>\s*", answer, re.S | re.I)
            fence = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", answer, re.S | re.I)
            if wrapper or fence:
                try:
                    strict = json.loads((wrapper or fence).group(1))
                    counts["assistant_output_wrapper_json" if wrapper else "assistant_fenced_json"] += 1
                except ValueError:
                    pass
            if strict is None:
                issues.add("assistant_requires_tolerant_json_extraction", **loc)
        graph = _graph(answer)
        docs.append(SourceRouter().route(_message(record, "user")))
        graphs.append(graph)
        doc = docs[-1]
        c = by_type[doc.source_type]
        c["records"] += 1
        counts[doc.source_type] += 1
        if graph is None:
            issues.add("invalid_teacher_graph", **loc)
            continue
        counts["assistant_graphs_valid"] += 1
        if not isinstance(strict, dict) or not {"entities", "relations"} <= set(strict):
            issues.add("teacher_graph_requires_wrapper_or_tolerant_parser", **loc)
        # ICD graph references are checked separately; no neural grounding claim.
        c["entities"] += len(graph["entities"])
        c["relations"] += len(graph["relations"])
        by_id, seen, grounded = {}, set(), set()
        repaired, edits = repair_text(doc.raw)
        c["source_repair_occurrences"] += len(edits)
        c["source_repair_records"] += bool(edits)
        for ni, node in enumerate(graph["entities"]):
            nloc = {**loc, "entity_index": ni}
            if not isinstance(node, dict):
                issues.add("invalid_entity_shape", **nloc)
                continue
            key = identifier(node.get("id"))
            if not key:
                issues.add("invalid_entity_id", **nloc)
            elif key in seen:
                issues.add("duplicate_entity_id", **nloc)
            seen.add(key)
            by_id[key] = node
            label = normalize_label(node.get("label"))
            if not label:
                issues.add("unknown_entity_type", **nloc)
            elif doc.source_type == "free_text":
                labels[label] += 1
            if label and node.get("label") != label:
                issues.add("entity_type_needs_canonicalization", **nloc)
            name = node_name(node)
            props = node.get("properties") or {}
            if isinstance(props, dict) and node.get("name") and props.get("Name") and normalize_text(node["name"]) != normalize_text(props["Name"]):
                issues.add("conflicting_entity_name_fields", **nloc)
            if doc.source_type == "free_text":
                before = bool(exact_hits(name, doc.raw))
                after = bool(exact_hits(repair_text(name)[0], repaired))
                c["grounded_entities_before"] += before
                c["grounded_entities_after"] += after
                if before and label:
                    grounded.add(key)
                if not before:
                    issues.add("entity_not_exactly_grounded", **nloc)
        pairs, triples = defaultdict(list), Counter()
        repaired_doc = SourceRouter().route(repaired)
        for ri, edge in enumerate(graph["relations"]):
            rloc = {**loc, "relation_index": ri}
            if not isinstance(edge, dict):
                issues.add("invalid_relation_shape", **rloc)
                continue
            a, b = identifier(edge.get("source")), identifier(edge.get("target"))
            label = edge.get("relation")
            if a not in by_id or b not in by_id or a is None or b is None:
                issues.add("dangling_relation_reference", **rloc)
            if a == b:
                issues.add("self_relation", **rloc)
            valid_label = isinstance(label, str)
            if not valid_label:
                issues.add("invalid_relation_label", **rloc)
                continue
            pairs[a, b].append(label)
            triples[a, b, label] += 1
            if doc.source_type != "free_text":
                continue
            if a not in grounded or b not in grounded:
                c["relations_with_ungrounded_endpoint"] += 1
            source_label = normalize_label(by_id.get(a, {}).get("label"))
            target_label = normalize_label(by_id.get(b, {}).get("label"))
            if label not in allowed_relations(source_label, target_label):
                issues.add("relation_outside_source_module", **rloc)
            else:
                relation_labels[label] += 1
            for suffix, current_doc, do_repair in (("before", doc, False), ("after", repaired_doc, True)):
                ev = evidence_check(edge, current_doc, repair=do_repair)
                c[f"relations_all_evidence_found_{suffix}"] += ev["all_found"]
                c[f"relations_partial_evidence_found_{suffix}"] += ev["partial_found"]
                if suffix == "before" and not ev["all_found"]:
                    issues.add("evidence_fragment_missing_or_invalid", **rloc)
            c["evidence_list_relations"] += isinstance(edge.get("evidence"), list)
        c["annotated_pairs"] += len(pairs)
        c["unique_relation_labels"] += len(triples)
        c["duplicate_relation_annotations"] += sum(v - 1 for v in triples.values())
        c["repeated_endpoint_pairs"] += sum(len(v) > 1 for v in pairs.values())
        c["multi_relation_pairs"] += sum(len(set(v)) > 1 for v in pairs.values())
        if doc.source_type == "free_text":
            c["spans"] += len(doc.spans)
            c["single_character_spans"] += sum(len(s.text) == 1 for s in doc.spans)
    result = {"counts": dict(counts), "by_source_type": {k: dict(v) for k, v in by_type.items()},
              "free_text_entity_labels": dict(labels), "free_text_relation_labels": dict(relation_labels),
              "system_sha256_counts": dict(systems), "issues": issues.result()}
    return result, docs, graphs


def overlap_audit(docs, groups, splits, *, repair=False, normalize=True, example_limit=8):
    entries = defaultdict(list)
    for i, (doc, group, split) in enumerate(zip(docs, groups, splits)):
        text = repair_text(doc.raw)[0] if repair else doc.raw
        entries[doc.source_type, sha(normalize_text(text) if normalize else text)].append((i, group, split))
    counts = Counter({f"{kind}_cross_split_text_hashes": 0 for kind in {d.source_type for d in docs}})
    examples, duplicates = [], []
    for (kind, digest), items in sorted(entries.items()):
        if len(items) > 1:
            counts[f"{kind}_duplicate_text_hashes"] += 1
            counts[f"{kind}_duplicate_extra_records"] += len(items) - 1
            if len(duplicates) < example_limit:
                duplicates.append({"sha256": digest, "source_indices": [v[0] for v in items], "group_ids": sorted({v[1] for v in items})})
        if len({v[2] for v in items}) > 1:
            counts[f"{kind}_cross_split_text_hashes"] += 1
            if len(examples) < example_limit:
                examples.append({"sha256": digest, "source_indices": [v[0] for v in items], "group_ids": sorted({v[1] for v in items})})
    return {"counts": dict(counts), "cross_split_examples": examples, "duplicate_text_examples": duplicates,
            "normalization": "routed source, NFKC + whitespace collapse + casefold" if normalize else "exact routed source (router already applies NFKC)"}


def audit_splits(records, docs, frozen, upstream, source_sha, *, seed=42, example_limit=8):
    groups = source_groups(records)
    train, tune = set(frozen["train_groups"]), set(frozen["tune_groups"])
    if train & tune or train | tune != set(groups):
        raise ValueError("Frozen groups overlap or do not cover raw records")
    if frozen.get("source_train_sha256") != source_sha:
        raise ValueError("Frozen source SHA256 mismatch")
    splits = ["tune" if g in tune else "train" for g in groups]
    shuffled = sorted(set(groups))
    random.Random(seed).shuffle(shuffled)
    inferred_tune = set(shuffled[:round(len(shuffled) * .1)])
    checks = {"source_hash_matches": True, "group_intersection": len(train & tune),
              "seed_split_matches": tune == inferred_tune,
              "groups": len(set(groups)), "train_groups": len(train), "tune_groups": len(tune),
              "records_by_split": dict(Counter(splits)), "free_text_by_split": dict(Counter(s for d, s in zip(docs, splits) if d.source_type == "free_text")),
              "group_assignment_sha256": canonical_sha(groups),
              "raw_text_overlap": overlap_audit(docs, groups, splits, example_limit=example_limit),
              "repaired_text_overlap": overlap_audit(docs, groups, splits, repair=True, example_limit=example_limit),
              "provenance_limit": "ICD anchors reconstruct local groups; original chunks are not independently verified by this audit."}
    anchors = defaultdict(list)
    for i, (doc, group, split) in enumerate(zip(docs, groups, splits)):
        if doc.source_type == "structured_icd":
            identity = doc.metadata.get("uri") or doc.metadata.get("icd_uri") or doc.metadata.get("code") or doc.metadata.get("icd_code")
            if identity is not None:
                anchors[sha(normalize_text(str(identity)))].append((i, group, split))
    checks["anchor_identity"] = {
        "unique_uri_or_code_hashes": len(anchors),
        "duplicate_anchor_hashes": sum(len(v) > 1 for v in anchors.values()),
        "cross_split_anchor_hashes": sum(len({split for _, _, split in v}) > 1 for v in anchors.values()),
    }
    if upstream is not None:
        ti, di = upstream["train_indices"], upstream["dev_indices"]
        matching = [v for v in upstream.get("outputs", []) if v.get("train_sha256") == source_sha]
        checks["upstream"] = {
            "matching_output_entries": len(matching), "train_indices": len(ti), "dev_indices": len(di),
            "indices_disjoint": not set(ti) & set(di),
            "indices_unique": len(ti) == len(set(ti)) and len(di) == len(set(di)),
            "indices_cover_parent": set(ti) | set(di) == set(range(upstream["records"])),
            "train_count_matches": len(ti) == len(records),
            "group_count_matches": upstream["train_groups"] == len(set(groups)),
            "record_order_is_parent_subsequence": ti == sorted(ti),
            "manifest_provenance_sha256": upstream.get("provenance_sha256"),
        }
    return checks, groups, splits


def expected_original_spans(doc, repair):
    text, changes = repair_text(doc.raw) if repair else (doc.raw, [])
    routed = SourceRouter().route(text)
    return {s.id: doc.raw[original_offset(s.start, changes):original_offset(s.end, changes, end=True)] for s in routed.spans}


def audit_row(row, line, docs, graphs, groups, splits, config, issues, span_cache):
    """Validate stored payload, target, raw-source provenance and supervision."""
    c = Counter(rows=1)
    loc = {"line": line}
    required = {"sample_id", "group_id", "split", "prompt", "completion", "target", "pairs", "entities", "spans", "original_spans", "sample_weight", "annotation_source", "id"}
    if not isinstance(row, dict) or not required <= set(row):
        issues.add("row_shape_missing_fields", **loc)
        return c
    index = row["sample_id"]
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(docs):
        issues.add("invalid_source_index", **loc)
        return c
    loc["source_index"] = index
    if row["group_id"] != groups[index] or row["split"] != splits[index]:
        issues.add("row_source_group_or_split_mismatch", **loc)
    if docs[index].source_type != "free_text":
        issues.add("structured_icd_in_neural_supervision", **loc)
    if row["annotation_source"] != "teacher_silver_not_human_gold":
        issues.add("annotation_provenance_mismatch", **loc)
    if not isinstance(row["sample_weight"], (int, float)) or isinstance(row["sample_weight"], bool) or not math.isfinite(row["sample_weight"]) or row["sample_weight"] <= 0:
        issues.add("invalid_sample_weight", **loc)
    else:
        c["weighted_rows"] += row["sample_weight"]
    if not isinstance(row["prompt"], str) or not isinstance(row["completion"], str):
        issues.add("prompt_or_completion_not_text", **loc)
        return c
    c["prompt_characters"] += len(row["prompt"])
    c["completion_characters"] += len(row["completion"])
    c["prompt_thinking_markers"] += bool(re.search(r"</?think\b", row["prompt"], re.I))
    c["completion_thinking_markers"] += bool(re.search(r"</?think\b", row["completion"], re.I))
    c["already_chat_rendered_rows"] += row["prompt"].startswith("<|im_start|>")
    try:
        completion = json.loads(row["completion"])
    except ValueError:
        issues.add("completion_invalid_json", **loc)
        return c
    if completion != row["target"]:
        issues.add("completion_target_mismatch", **loc)
    if not all(isinstance(row[k], list) and all(isinstance(v, dict) for v in row[k]) for k in ("pairs", "entities", "spans")):
        issues.add("payload_table_shape", **loc)
        return c
    try:
        mapping = {e["id"]: e["entity_id"] for e in row["entities"]}
        if len(mapping) != len(row["entities"]) or len(set(mapping.values())) != len(mapping):
            issues.add("duplicate_entity_reference", **loc)
        pair_ids = [p["pair_id"] for p in row["pairs"]]
        if len(set(pair_ids)) != len(pair_ids):
            issues.add("duplicate_pair_reference", **loc)
        if len({s["span_id"] for s in row["spans"]}) != len(row["spans"]):
            issues.add("duplicate_span_reference", **loc)
        payload = json.loads(row["prompt"].strip().splitlines()[-1])
        expected = {
            "entities": [{k: v for k, v in e.items() if k != "id"} for e in row["entities"]],
            "pairs": [{"pair_id": p["pair_id"], "source": mapping[p["source"]], "target": mapping[p["target"]], "allowed_relations": p["allowed"]} for p in row["pairs"]],
            "spans": row["spans"],
        }
        if payload != expected:
            issues.add("prompt_payload_mismatch", **loc)
        if not isinstance(payload, dict) or set(payload) != {"entities", "pairs", "spans"}:
            issues.add("unexpected_prompt_payload_fields", **loc)
        # Entity tables stored outside the prompt have one extra raw graph ID.
        if any(set(e) != {"entity_id", "id", "name", "label"} for e in row["entities"]):
            issues.add("unexpected_entity_fields_in_prompt", **loc)
        entities = {e["id"]: e for e in row["entities"]}
        for pair in row["pairs"]:
            a, b = entities[pair["source"]], entities[pair["target"]]
            if a["id"] == b["id"] or pair["allowed"] != list(allowed_relations(a["label"], b["label"])):
                issues.add("pair_type_contract_mismatch", **loc)
        parsed, errors = parse_decisions(row["completion"], row)
    except (KeyError, ValueError, TypeError, AttributeError):
        issues.add("payload_reference_or_parser_error", **loc)
        return c
    for error in set(errors):
        # Parser errors are fixed program codes, never strings from the data.
        issues.add("target_" + error, **loc)
    if errors:
        return c
    c["schema_valid_rows"] += 1
    c["pairs"] += len(row["pairs"])
    c["positive_edges"] += len(parsed)
    decisions = completion["decisions"]
    positives = [bool(d["relations"]) for d in decisions]
    c["positive_pairs"] += sum(positives)
    c["none_pairs"] += len(positives) - sum(positives)
    c["multi_relation_pairs"] += sum(len(d["relations"]) > 1 for d in decisions)
    c["mixed_batches"] += any(positives) and not all(positives)
    c["positive_prefix_mixed_batches"] += bool(any(positives) and not all(positives) and positives == sorted(positives, reverse=True))
    if index not in span_cache:
        text = repair_text(docs[index].raw)[0] if config.repair else docs[index].raw
        span_cache[index] = (expected_original_spans(docs[index], config.repair), SourceRouter().route(text))
    if not isinstance(row["original_spans"], dict) or set(row["original_spans"]) != {s["span_id"] for s in row["spans"]}:
        issues.add("original_span_reference_mismatch", **loc)
    else:
        mismatches = sum(span_cache[index][0].get(k) != v for k, v in row["original_spans"].items())
        c["original_span_mismatches"] += mismatches
        if mismatches:
            issues.add("original_spans_not_original_source", **loc)
    graph = graphs[index]
    if graph is None:
        issues.add("row_has_invalid_teacher_graph", **loc)
        return c
    original_nodes = {str(e["id"]): e for e in graph["entities"] if isinstance(e, dict) and "id" in e}
    current_doc = span_cache[index][1]
    text = current_doc.raw
    for entity in row["entities"]:
        original = original_nodes.get(entity["id"])
        if original is None:
            issues.add("row_entity_not_in_teacher", **loc)
            continue
        name = repair_text(node_name(original))[0] if config.repair else node_name(original)
        if entity["name"] != name or entity["label"] != normalize_label(original["label"]):
            issues.add("row_entity_identity_mismatch", **loc)
        if not exact_hits(entity["name"], text):
            issues.add("row_entity_ungrounded", **loc)
    span_map = current_doc.span_map()
    if any(s.get("span_id") not in span_map or span_map[s["span_id"]].text != s.get("text") for s in row["spans"]):
        issues.add("row_span_not_in_routed_source", **loc)
    annotations = defaultdict(list)
    for ri, edge in enumerate(graph["relations"]):
        if isinstance(edge, dict):
            annotations[str(edge.get("source")), str(edge.get("target"))].append((ri, edge))
    for pair, decision in zip(row["pairs"], decisions):
        edges = annotations[pair["source"], pair["target"]]
        expected_labels = sorted({edge["relation"] for _, edge in edges})
        if not config.multilabel:
            expected_labels = expected_labels[:1]
        actual_labels = sorted(e["relation"] for e in decision["relations"])
        if actual_labels != expected_labels:
            issues.add("stored_labels_differ_from_annotated_pair", **loc)
        if not edges and not actual_labels:
            c["unannotated_silver_pairs_taught_none"] += 1
        for ri, edge in edges:
            check = evidence_check(edge, current_doc, repair=config.repair, visible={s["span_id"] for s in row["spans"]})
            if not check["all_found"]:
                issues.add("supervised_annotation_has_unresolved_fragment", **loc, relation_index=ri)
    return c


def audit_saved(path, docs, graphs, groups, splits, config, manifest, split, *, example_limit=8):
    counts, issues, lengths = Counter(), Findings(example_limit), []
    sources, group_ids, row_ids, label_counts, span_cache = set(), set(), set(), Counter(), {}
    hasher, raw_hash = hashlib.sha256(), file_sha(path)
    for line, row, error in jsonl(path):
        if error:
            issues.add(error, line=line)
            continue
        c = audit_row(row, line, docs, graphs, groups, splits, config, issues, span_cache)
        counts.update(c)
        if not isinstance(row, dict):
            continue
        hasher.update((dumps(row) + "\n").encode())
        if row.get("split") != split:
            issues.add("wrong_file_split", line=line)
        if isinstance(row.get("sample_id"), int):
            sources.add(row["sample_id"])
        if isinstance(row.get("group_id"), int):
            group_ids.add(row["group_id"])
        key = canonical_sha(row.get("id"))
        if key in row_ids:
            issues.add("duplicate_row_id", line=line)
        row_ids.add(key)
        if c.get("schema_valid_rows"):
            label_counts.update(e["relation"] for d in row["target"]["decisions"] for e in d["relations"])
            lengths.append(len(row["prompt"]) + len(row["completion"]))
    counts["source_records"] = len(sources)
    counts["source_groups"] = len(group_ids)
    expected = manifest.get("stats", {}).get(split, {})
    comparisons = {k: {"manifest": expected.get(k), "observed": counts.get(observed, 0), "matches": expected.get(k) == counts.get(observed, 0)}
                   for k, observed in {"rows": "rows", "retained_pairs": "pairs", "positive_edges": "positive_edges", "none_pairs": "none_pairs", "multi_relation_pairs": "multi_relation_pairs"}.items()}
    return {"sha256": raw_hash, "bytes": Path(path).stat().st_size,
            "manifest_hash_matches": raw_hash == manifest["dataset_sha256"][split],
            "canonical_jsonl_sha256": hasher.hexdigest(), "counts": dict(counts),
            "label_counts": dict(label_counts), "manifest_counts": comparisons,
            "characters_not_tokens": {"min": min(lengths, default=0), "max": max(lengths, default=0), "mean": sum(lengths) / max(1, len(lengths))},
            "missing_free_text_source_indices": [i for i, (d, s) in enumerate(zip(docs, splits)) if s == split and d.source_type == "free_text" and i not in sources],
            "issues": issues.result()}


def trace_record(record, index, config, *, sampled, seed=42):
    """Separate candidate, sampling, direct UNKNOWN and batch-collateral losses.

    Production supervision is invoked both per batch and per pair; this does not
    change training. 'Valid' here means converter-accepted, not medical truth.
    """
    original_doc = SourceRouter().route(_message(record, "user"))
    if original_doc.source_type != "free_text":
        return Counter(icd_records=1)
    graph = _graph(_message(record, "assistant"))
    if graph is None:
        return Counter(invalid_teacher_records=1)
    batches, doc, _ = build_batches(original_doc.raw, graph["entities"], config, sample_id=index)
    all_pairs = {(p["source"], p["target"]) for b in batches for p in b["pairs"]}
    gold = {(str(e["source"]), str(e["target"]), e["relation"]) for e in graph["relations"]}
    positives = {(a, b) for a, b, _ in gold}
    c = Counter(records=1, raw_relation_annotations=len(graph["relations"]), raw_unique_positive_edges=len(gold),
                all_candidate_pairs=len(all_pairs), candidate_annotated_pairs=len(all_pairs & positives),
                candidate_unannotated_pairs=len(all_pairs - positives),
                positive_edges_outside_candidates=sum((a, b) not in all_pairs for a, b, _ in gold))
    c["records_without_any_candidate_pairs"] = not all_pairs
    retained = all_pairs
    if sampled:
        negatives = sorted(all_pairs - positives)
        random.Random(seed + index).shuffle(negatives)
        retained = (all_pairs & positives) | set(negatives[:max(8, config.negative_ratio * len(positives & all_pairs))])
        # Always retain immutable source here: context/pair coverage is unchanged
        # by the old converter's separate original_spans provenance defect.
        batches, doc, _ = build_batches(original_doc.raw, graph["entities"], config, sample_id=index, selected_pairs=retained)
    c["sampled_unannotated_pairs"] = len(retained - positives)
    c["downsampled_unannotated_pairs"] = len(all_pairs - retained)
    c["selected_pairs"] = len(retained)
    for batch in batches:
        target, _ = supervision(batch, graph, doc, config)
        if target is None:
            c["unknown_batches"] += 1
            c["whole_batch_dropped_pairs"] += len(batch["pairs"])
        for pair in batch["pairs"]:
            one, _ = supervision({**batch, "pairs": [pair]}, graph, doc, config)
            pair_gold = {label for a, b, label in gold if (a, b) == (pair["source"], pair["target"])}
            if one is None:
                c["direct_unknown_pairs"] += 1
                c["direct_unknown_positive_edges"] += len(pair_gold)
            elif target is None:
                labels = one["decisions"][0]["relations"]
                c["collateral_converter_valid_pairs"] += 1
                c["collateral_positive_pairs"] += bool(labels)
                c["collateral_positive_edges"] += len(labels)
                c["collateral_none_pairs"] += not labels
                c["intentional_single_label_loss"] += max(0, len(pair_gold) - len(labels))
            else:
                labels = one["decisions"][0]["relations"]
                c["retained_pairs"] += 1
                c["retained_positive_edges"] += len(labels)
                c["retained_none_pairs"] += not labels
                c["retained_multi_relation_pairs"] += len(labels) > 1
                c["intentional_single_label_loss"] += max(0, len(pair_gold) - len(labels))
    c["positive_edge_conservation_gap"] = len(gold) - sum(c[k] for k in ("positive_edges_outside_candidates", "direct_unknown_positive_edges", "collateral_positive_edges", "retained_positive_edges", "intentional_single_label_loss"))
    c["pair_conservation_gap"] = len(all_pairs) - c["downsampled_unannotated_pairs"] - c["whole_batch_dropped_pairs"] - c["retained_pairs"]
    return c


def replay_variant(records, groups, splits, config, seed, *, example_limit=8):
    """Replay the real converter, then independently decompose its losses."""
    subsets = {"train": [], "tune": []}
    converter_counts, losses = defaultdict(Counter), defaultdict(Counter)
    loss_examples = Findings(example_limit)
    for i, (record, group, split) in enumerate(zip(records, groups, splits)):
        rows, stats = convert(record, i, group, config, sampled=split == "train", seed=seed)
        converter_counts[split].update(stats)
        for row in rows:
            row["split"] = split
        subsets[split].extend(rows)
        trace = trace_record(record, i, config, sampled=split == "train", seed=seed)
        losses[split].update(trace)
        if trace["direct_unknown_pairs"]:
            loss_examples.add("source_has_direct_unknown", source_index=i)
        if trace["collateral_positive_edges"]:
            loss_examples.add("source_loses_collateral_positive_supervision", source_index=i)
    if config.rebalance:
        apply_rebalancing(subsets["train"])
    outputs = {}
    for split, rows in subsets.items():
        h = hashlib.sha256()
        for row in rows:
            h.update((dumps(row) + "\n").encode())
        observed = losses[split]
        stats = converter_counts[split]
        outputs[split] = {
            "sha256": h.hexdigest(), "converter_counts": dict(stats),
            "loss_decomposition": dict(observed),
            "trace_matches_converter": all(observed.get(a, 0) == stats.get(b, 0) for a, b in (
                ("retained_pairs", "retained_pairs"), ("retained_positive_edges", "positive_edges"),
                ("retained_none_pairs", "none_pairs"), ("unknown_batches", "unknown_batches"),
                ("whole_batch_dropped_pairs", "unknown_pairs"))),
        }
    return {"splits": outputs, "loss_examples": loss_examples.result(),
            "negative_policy": "Unannotated silver pairs are sampled and taught empty relations; they are not verified medical negatives.",
            "unknown_policy": "Direct UNKNOWN is separate from converter-valid collateral; graph evaluation must retain all original annotations."}


def synthetic_record(text, graph):
    return {"messages": [{"role": "system", "content": ONTOLOGY["version"]},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": dumps(graph)}]}


def reproduce_bugs():
    text = "Fatigue accompanies Condition Alpha during pregnullcy."
    nodes = [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
             {"id": "D1", "label": "Disease", "name": "Condition Alpha"}]
    edge = {"source": "S1", "target": "D1", "relation": "is_associated_symptom_of", "evidence": text}
    config = ProtocolConfig(repair=True, strict_evidence=True)
    batches, doc, _ = build_batches(text, nodes, config)
    graph = {"entities": nodes, "relations": [{**edge, "evidence": [text, "SYNTHETIC_MISSING_FRAGMENT_57"]}]}
    target, _ = supervision(batches[0], graph, doc, config)
    independent = evidence_check(graph["relations"][0], doc, repair=True)
    record = synthetic_record(text, {"entities": nodes, "relations": [edge]})
    original = expected_original_spans(SourceRouter().route(text), True)
    provenance = {}
    for sampled in (False, True):
        rows, _ = convert(record, 1, 0, config, sampled=sampled)
        mismatches = sum(original.get(k) != v for row in rows for k, v in row["original_spans"].items())
        provenance["sampled" if sampled else "unsampled"] = {"rows": len(rows), "original_span_mismatches": mismatches}
    return {
        "partial_evidence_list": {"strict_evidence": True, "fragments": independent["fragments"],
                                  "all_fragments_found": independent["all_found"],
                                  "production_accepts": target is not None,
                                  "bug_observed": target is not None and not independent["all_found"]},
        "sampled_original_span_provenance": {**provenance,
                                             "bug_observed": provenance["sampled"]["original_span_mismatches"] > 0},
        "fixtures_are_synthetic": True,
    }


def synthetic_loss_mask_check():
    """Exercise actual renderer/dataset/collator with a clearly synthetic tokenizer.

    Only the unused torch import and tensor constructor are substituted. This
    checks mask boundaries, EOS and padding, NOT real Qwen token lengths/model.
    """
    from types import SimpleNamespace
    from unittest.mock import patch
    from generative.chat import render_prompt
    from train_sft import CausalCollator, JsonlSFTDataset

    class Tokenizer:
        eos_token_id, pad_token_id = 1, 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(ch) + 2 for ch in text]}

        def apply_chat_template(self, messages, **kwargs):
            if kwargs != {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}:
                raise ValueError("Non-thinking chat renderer contract changed")
            return "<|im_start|>user\n" + messages[0]["content"] + "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    tokenizer = Tokenizer()
    prompt = render_prompt(tokenizer, "SYNTHETIC_INPUT")
    rows = [{"prompt": prompt, "completion": '{"decisions":[]}'}, {"prompt": prompt, "completion": "{}"}]
    fake_torch = SimpleNamespace(long="long", tensor=lambda value, **kwargs: value)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        dataset = JsonlSFTDataset(rows, tokenizer, 4096)
        batch = CausalCollator(tokenizer)(dataset.items)
        checks = {"prompt_and_empty_thinking_fully_masked": all(all(x == -100 for x in item["labels"][:len(prompt)]) for item in dataset.items),
                  "completion_only_supervised": all(item["labels"][len(prompt):] == item["input_ids"][len(prompt):] for item in dataset.items),
                  "eos_supervised": all(item["labels"][-1] == 1 for item in dataset.items),
                  "padding_masked": all(label == -100 for labels, mask in zip(batch["labels"], batch["attention_mask"]) for label, active in zip(labels, mask) if not active)}
        try:
            JsonlSFTDataset(rows, tokenizer, 2)
            checks["overlength_rejected"] = False
        except ValueError:
            checks["overlength_rejected"] = True
    return {"checks": checks, "all_passed": all(checks.values()), "tokenizer": "synthetic_character_tokenizer",
            "real_qwen_tokenization_verified": False, "model_or_gpu_executed": False}


def audit_repair(directory, records, groups, splits, source_sha):
    from data_repair.verify import verify_release
    manifest = read_json(directory / "manifest.json")
    verification = verify_release(directory)
    checks = Counter()
    seen = set()
    for line, row, error in jsonl(directory / "frozen_split.jsonl"):
        if error or not isinstance(row, dict):
            checks["invalid_frozen_split_rows"] += 1
            continue
        i = row.get("source_index")
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(records):
            checks["invalid_source_indices"] += 1
            continue
        checks["duplicate_source_indices"] += i in seen
        seen.add(i)
        checks["group_or_split_mismatches"] += row.get("group_id") != groups[i] or row.get("split") != splits[i]
    checks["missing_source_indices"] = len(records) - len(seen)
    for _, row, error in jsonl(directory / "records.jsonl"):
        if error or not isinstance(row, dict):
            checks["invalid_repaired_rows"] += 1
            continue
        i = row.get("source_index")
        if not isinstance(i, int) or not 0 <= i < len(records):
            checks["invalid_repaired_source_indices"] += 1
            continue
        checks["original_record_content_mismatches"] += row.get("original_record_sha256") != canonical_sha(records[i])
    return {"verification": verification, "source_hash_matches": manifest["source_sha256"] == source_sha,
            "frozen_split_checks": dict(checks), "summary": manifest["summary"],
            "release_gates": manifest["release_gates"], "review_reason_counts": manifest["review_reason_counts"],
            "used_as_training_release": False,
            "note": "Candidate repair release is separate from the raw-input generative exports; no clinical relabeling is performed."}


def default_inputs(workspace):
    """Bounded, known locations only: never recursively crawl tooling/models."""
    work = Path(workspace) / "work"
    return {"raw": work / "audit_20260907/data" / SOURCE_NAME,
            "upstream": work / "audit_20260907/data/groupdisjoint_split_manifest.json",
            "data_dir": work / "generative_20260908/data",
            "repair_dir": work / "data_repair_20260908/release_v1"}


def audit_source_overlap(raw, dev, *, expected_sha256, expected_dev_sha256,
                         expected_records=1345, expected_dev_records=148, example_limit=8):
    """Bounded train/dev source-only audit: no graph conversion or variant replay."""
    inputs = {"train_source": Path(raw), "historical_dev": Path(dev)}
    hashes = {key: file_sha(path) for key, path in inputs.items()}
    if hashes["train_source"] != expected_sha256 or hashes["historical_dev"] != expected_dev_sha256:
        raise ValueError("Source overlap input SHA256 mismatch; incomplete transfer is not accepted")
    counts, documents = {}, []
    for key, path in inputs.items():
        records = read_json(path)
        expected = expected_records if key == "train_source" else expected_dev_records
        if not isinstance(records, list) or len(records) != expected:
            raise ValueError("Source overlap input record count mismatch")
        if any(not isinstance(r, dict) or not isinstance(r.get("messages"), list) or not all(isinstance(m, dict) for m in r["messages"]) or not _message(r, "user") for r in records):
            raise ValueError("Missing source message in overlap input")
        docs = [SourceRouter().route(_message(r, "user")) for r in records]
        counts[key] = {"records": len(docs), **dict(Counter(d.source_type for d in docs))}
        documents.extend(docs)
    # Ordinals below are only internal helper keys, not asserted source groups.
    splits = ["train_source"] * expected_records + ["historical_dev"] * expected_dev_records
    comparisons = {}
    for name, do_repair, do_normalize in (("exact_routed", False, False), ("normalized_routed", False, True), ("normalized_repaired", True, True)):
        result = overlap_audit(documents, list(range(len(documents))), splits, repair=do_repair, normalize=do_normalize, example_limit=example_limit)
        for field in ("cross_split_examples", "duplicate_text_examples"):
            for item in result[field]:
                item.pop("group_ids", None)
                item["locations"] = [{"dataset": splits[i], "source_index": i if i < expected_records else i - expected_records} for i in item.pop("source_indices")]
        comparisons[name] = result
    unchanged = all(file_sha(path) == hashes[key] for key, path in inputs.items())
    return {"audit_kind": "source_overlap", "status": "source_overlap_completed" if unchanged else "input_changed_rerun_required",
            "inputs": {key: {"path": str(path.resolve()), "sha256": hashes[key], **counts[key]} for key, path in inputs.items()},
            "comparisons": comparisons, "inputs_unchanged": unchanged,
            "scope": "Routed user source only; no graph labels, no variant replay, no remote access or medical text output.",
            "limits": ["Zero exact overlap does not prove absence of semantic leakage.", "Historical dev is a regression set, not a new blind test."]}


def run_audit(*, raw, data_dir, repair_dir=None, upstream=None, dev=None,
              variants=None, expected_sha256=EXPECTED_SHA256, expected_records=1345,
              example_limit=8, replay=True, progress=None):
    code_start = {p: file_sha(ROOT / p) for p in CODE_PATHS}
    raw, data_dir = Path(raw), Path(data_dir)
    source_sha = file_sha(raw)
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError("Raw input SHA256 differs from the registered version")
    records = read_json(raw)
    if not isinstance(records, list) or (expected_records is not None and len(records) != expected_records):
        raise ValueError("Raw record count or list shape mismatch")
    names = variants or sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("G") and (p / "manifest.json").is_file())
    if not names or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("No valid explicit variant directories")
    inputs = [raw, *[data_dir / n / filename for n in names for filename in ("manifest.json", "train.jsonl", "tune.jsonl")]]
    if upstream:
        inputs.append(Path(upstream))
    if repair_dir:
        repair_dir = Path(repair_dir)
        inputs.extend(repair_dir / n for n in ("manifest.json", "records.jsonl", "changes.jsonl", "pair_candidates.jsonl", "review_queue.jsonl", "frozen_split.jsonl"))
    tune_records = data_dir / "tune_records.json"
    if tune_records.is_file():
        inputs.append(tune_records)
    if dev:
        inputs.append(Path(dev))
    input_start = {str(p.resolve()): file_sha(p) for p in inputs}
    raw_report, docs, graphs = audit_raw(records, example_limit=example_limit)
    first = read_json(data_dir / names[0] / "manifest.json")
    split_report, groups, splits = audit_splits(records, docs, first, read_json(upstream) if upstream else None, source_sha, seed=first["seed"], example_limit=example_limit)
    if tune_records.is_file():
        exported = read_json(tune_records)
        expected = [r for r, split in zip(records, splits) if split == "tune"]
        split_report["tune_records_exact_content_and_order"] = exported == expected
    report = {"audit_version": 1, "privacy": "Aggregates and zero-based source/annotation indices or SHA256 only; no medical text.",
              "source": {"path": str(raw.resolve()), "sha256": source_sha, "bytes": raw.stat().st_size},
              "raw": raw_report, "splits": split_report, "variants": {},
              "synthetic_reproductions": reproduce_bugs(), "loss_mask": synthetic_loss_mask_check()}
    for name in names:
        if progress:
            progress(name)
        root = data_dir / name
        manifest = read_json(root / "manifest.json")
        config = ProtocolConfig(**manifest["config"])
        if manifest["source_train_sha256"] != source_sha or set(manifest["train_groups"]) != set(first["train_groups"]) or set(manifest["tune_groups"]) != set(first["tune_groups"]):
            raise ValueError("Variant source hash or split differs from frozen input")
        entry = {"config": asdict(config), "seed": manifest["seed"], "saved": {}}
        for split in ("train", "tune"):
            entry["saved"][split] = audit_saved(root / f"{split}.jsonl", docs, graphs, groups, splits, config, manifest, split, example_limit=example_limit)
        if replay:
            entry["replay"] = replay_variant(records, groups, splits, config, manifest["seed"], example_limit=example_limit)
            for split in ("train", "tune"):
                entry["replay"]["splits"][split]["matches_saved_bytes"] = entry["replay"]["splits"][split]["sha256"] == entry["saved"][split]["sha256"]
        report["variants"][name] = entry
    if repair_dir:
        report["repair_release"] = audit_repair(repair_dir, records, groups, splits, source_sha)
    if dev:
        dev_records = read_json(dev)
        dev_docs = [SourceRouter().route(_message(r, "user")) for r in dev_records]
        report["historical_dev"] = {"status": "local_exact_text_check_only", "records": len(dev_records), "sha256": file_sha(dev),
                                    "overlap": overlap_audit(docs + dev_docs, groups + list(range(len(dev_docs))), ["train_source"] * len(docs) + ["historical_dev"] * len(dev_docs), example_limit=example_limit)}
    else:
        report["historical_dev"] = {"status": "not_reverified_no_local_raw_dev_selected", "note": "Historical aggregate claims and upstream index disjointness do not independently verify dev text. No remote access."}
    report["code_sha256"] = code_start
    report["code_changed_during_audit"] = [p for p in code_start if file_sha(ROOT / p) != code_start[p]]
    report["input_sha256"] = input_start
    report["inputs_changed_during_audit"] = [p for p in input_start if file_sha(p) != input_start[p]]
    report["status"] = "audit_completed" if not report["code_changed_during_audit"] and not report["inputs_changed_during_audit"] else "concurrent_change_detected_rerun_required"
    report["caveats"] = ["Silver mismatch and exact-grounding failures are not adjudicated medical errors.",
                         "Source text equality is checked, not semantic similarity or independent original chunk provenance.",
                         "Stored NONE supervision is not a verified negative pool; repair review remains UNKNOWN.",
                         "Character lengths and synthetic masks are not real Qwen token or GPU validation.",
                         "Replay differences after a core fix are version differences, not permission to overwrite old exports."]
    return report


def markdown_summary(report):
    lines = ["# Local data/preprocessing audit", "", f"Status: `{report['status']}`.", "",
             f"Raw input SHA256: `{report['source']['sha256']}`.", "",
             "| Variant | Split | Rows | Pairs | Positive labels | NONE | Original-span mismatch rows | Replay bytes equal |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, entry in report["variants"].items():
        for split, data in entry["saved"].items():
            c = data["counts"]
            parity = entry.get("replay", {}).get("splits", {}).get(split, {}).get("matches_saved_bytes", "not run")
            lines.append(f"| {name} | {split} | {c.get('rows', 0)} | {c.get('pairs', 0)} | {c.get('positive_edges', 0)} | {c.get('none_pairs', 0)} | {data['issues']['counts'].get('original_spans_not_original_source', 0)} | {parity} |")
    lines += ["", "See `audit.json` for hashes, schema checks, source indices, independent evidence checks and loss decomposition.", "",
              "Synthetic reproductions:", "", "```json", json.dumps(report["synthetic_reproductions"], indent=2), "```", "",
              "Limits: no clinical adjudication, no remote/model calls, no true-tokenizer length validation; no raw dev text check unless --dev was supplied.", ""]
    return "\n".join(lines)


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("Report directory must be new; refusing to overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    filename = "overlap.json" if report.get("audit_kind") == "source_overlap" else "audit.json"
    if filename == "overlap.json":
        lines = ["# Bounded train/dev exact-source overlap audit", "", f"Status: `{report['status']}`.", ""]
        for key, value in report["inputs"].items():
            lines.extend([f"{key}: {value['records']} records; SHA256 `{value['sha256']}`.", ""])
        for key, value in report["comparisons"].items():
            lines.extend([f"{key}: `{json.dumps(value['counts'], sort_keys=True)}`", ""])
        lines.append("Source indices and duplicate hashes are in overlap.json. No graph conversion, medical adjudication or remote access was performed.")
        summary = "\n".join(lines) + "\n"
    else:
        summary = markdown_summary(report)
    for name, content in ((filename, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"), ("SUMMARY.md", summary)):
        path = output_dir / name
        with path.open("x", encoding="utf-8") as handle:
            path.chmod(0o600)
            handle.write(content)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=ROOT.parent)
    p.add_argument("--raw", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--repair-dir", type=Path)
    p.add_argument("--upstream", type=Path)
    p.add_argument("--dev", type=Path)
    p.add_argument("--overlap-only", action="store_true", help="Only compare raw train/dev source texts; no variant replay")
    p.add_argument("--expected-dev-sha256")
    p.add_argument("--expected-dev-records", type=int, default=148)
    p.add_argument("--variants", nargs="+")
    p.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    p.add_argument("--expected-records", type=int, default=1345)
    p.add_argument("--example-limit", type=int, default=8)
    p.add_argument("--skip-replay", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        p.error("Output directory already exists; choose a new report version")
    if args.overlap_only and (not args.dev or not args.expected_dev_sha256):
        p.error("--overlap-only requires --dev and --expected-dev-sha256")
    defaults = default_inputs(args.workspace)
    try:
        if args.dev and args.expected_dev_sha256 and file_sha(args.dev) != args.expected_dev_sha256:
            raise ValueError("Dev input SHA256 mismatch")
        if args.overlap_only:
            report = audit_source_overlap(args.raw or defaults["raw"], args.dev, expected_sha256=args.expected_sha256,
                                          expected_dev_sha256=args.expected_dev_sha256, expected_records=args.expected_records,
                                          expected_dev_records=args.expected_dev_records, example_limit=args.example_limit)
        else:
            report = run_audit(raw=args.raw or defaults["raw"], data_dir=args.data_dir or defaults["data_dir"],
                           repair_dir=args.repair_dir or defaults["repair_dir"], upstream=args.upstream or defaults["upstream"],
                           dev=args.dev, variants=args.variants, expected_sha256=args.expected_sha256,
                           expected_records=args.expected_records, example_limit=args.example_limit, replay=not args.skip_replay,
                           progress=lambda name: print(json.dumps({"auditing_variant": name}), file=sys.stderr, flush=True))
        write_report(report, args.output_dir)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Exceptions may originate in other readers; never echo their contents.
        print(json.dumps({"status": "audit_failed", "error_type": type(exc).__name__, "detail": "Input/contract validation failed; contents redacted."}), file=sys.stderr)
        return 2
    filename = "overlap.json" if args.overlap_only else "audit.json"
    print(json.dumps({"status": report["status"], "report": str(args.output_dir.resolve() / filename)}))
    return 0 if report["status"] in {"audit_completed", "source_overlap_completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

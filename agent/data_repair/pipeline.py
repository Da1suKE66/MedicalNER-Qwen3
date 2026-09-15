"""Repair mechanically verifiable defects; quarantine uncertainty, never invent NONE.

Writes a NEW private candidate release. Does not train, call an external LLM,
change the ontology, overwrite inputs, or certify clinical correctness.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from build_pair_data import source_groups
from build_stage_data import _message
from evaluate import _graph
from generative.protocol import exact_hits, original_offset, repair_text
from kg_agent.contracts import ONTOLOGY
from kg_agent.source import SourceRouter


VERSION = "teacher-silver-repair-v1"
ROOT = Path(__file__).resolve().parents[2]
NODE_LABELS = {x.casefold(): x for x in ONTOLOGY["node_labels"]}
SOURCE_RELATIONS = {
    (x["source_label"], x["relation"]) for x in ONTOLOGY["source_relations"]
}
TEXT_FIELDS = {"name", "Name", "text", "description", "Description", "evidence", "evidence_sentence"}
NEGATION = re.compile(r"\b(?:not|without|absence|absent|excluded|excluding)\b", re.I)
ENUMERATION = re.compile(r"\b(?:including|such as)\b|e\.g\.", re.I)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(value):
    data = value if isinstance(value, bytes) else str(value).encode()
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return " ".join(str(value).split()).casefold()


def identifier(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() else None


def text_repairs(value, *, path="", text_field=False):
    """Repair descriptive values only, not identifiers, relation labels, URLs or schema."""
    changes = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result[key], edits = text_repairs(item, path=f"{path}/{key}", text_field=key in TEXT_FIELDS)
            changes.extend(edits)
        return result, changes
    if isinstance(value, list):
        result = []
        for i, item in enumerate(value):
            fixed, edits = text_repairs(item, path=f"{path}/{i}", text_field=text_field)
            result.append(fixed)
            changes.extend(edits)
        return result, changes
    if isinstance(value, str) and text_field:
        fixed, edits = repair_text(value)
        return fixed, [{"path": path, **e} for e in edits]
    return value, []


def evidence_values(edge):
    value = edge.get("evidence") or edge.get("evidence_sentence") or ""
    values = value if isinstance(value, list) else [value]
    return values if all(isinstance(v, str) for v in values) else []


def locate(value, text, changes):
    return [
        {"start": a, "end": b, "text": text[a:b],
         "original_start": original_offset(a, changes),
         "original_end": original_offset(b, changes, end=True)}
        for a, b in exact_hits(value, text)
    ]


def repair_record(record, index, group, split):
    """Return all annotations, including review cases; never drop a source record."""
    record_id = f"source-{index:06d}"
    result = {"record_id": record_id, "source_index": index, "group_id": group,
              "split": split, "original_record_sha256": sha(dumps(record)),
              "original_record": copy.deepcopy(record), "changes": [], "issues": [],
              "entities": [], "annotations": [], "pairs": [],
              "unannotated_pair_policy": "UNKNOWN_NOT_VERIFIED_NONE",
              "semantic_adjudication": "not_performed"}
    doc = SourceRouter().route(_message(record, "user"))
    result["source_type"] = doc.source_type
    original = doc.raw
    text, source_changes = repair_text(original)
    result["changes"].extend({"path": "/source", **e} for e in source_changes)
    result["source"] = {"original_text": original, "text": text,
                        "original_sha256": sha(original), "repaired_sha256": sha(text),
                        "offset_basis": "NFKC-normalized routed source, not original JSON bytes"}
    if doc.source_type == "structured_icd":
        # ICD graph is not neural relation supervision. Preserve its complete input.
        result["status"] = "deterministic_icd_only"
        return result
    graph = _graph(_message(record, "assistant"))
    if graph is None:
        result["issues"].append("invalid_teacher_graph")
        result["status"] = "review"
        return result
    fixed, edits = text_repairs(graph, path="/graph")
    result["changes"].extend(edits)
    result["repaired_graph"] = fixed
    new_doc = SourceRouter().route(text)
    result["spans"] = [asdict(s) for s in new_doc.spans]
    assert all(text[s.start:s.end] == s.text for s in new_doc.spans)
    ids = [identifier(n.get("id")) for n in fixed["entities"] if isinstance(n, dict)]
    duplicates = {key for key, count in Counter(ids).items() if key and count > 1}
    if duplicates:
        result["issues"].append("duplicate_entity_ids")
    by_id = {}
    for i, node in enumerate(fixed["entities"]):
        issues = []
        if not isinstance(node, dict):
            result["entities"].append({"index": i, "issues": ["invalid_entity_shape"], "mentions": []})
            continue
        key = identifier(node.get("id"))
        props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        name = str(node.get("name") or props.get("Name") or "").strip()
        label = NODE_LABELS.get(str(node.get("label", "")).strip().casefold())
        if not key:
            issues.append("invalid_entity_id")
        if key in duplicates:
            issues.append("duplicate_entity_id")
        if not label:
            issues.append("unknown_entity_label")
        elif node.get("label") != label:
            result["changes"].append({"path": f"/graph/entities/{i}/label", "before": node.get("label"), "after": label, "op": "canonical_label_case"})
            node["label"] = label
        if node.get("name") and props.get("Name") and normalized(node["name"]) != normalized(props["Name"]):
            issues.append("conflicting_entity_name_fields")
        hits = locate(name, text, source_changes)
        if not hits:
            issues.append("entity_not_grounded")
        item = {"index": i, "id": key, "name": name, "label": label,
                "mentions": hits, "issues": issues, "concept_key": [label, normalized(name)]}
        result["entities"].append(item)
        if key and key not in duplicates:
            by_id[key] = item
    by_pair = defaultdict(list)
    for i, edge in enumerate(fixed["relations"]):
        annotation = {"index": i, "annotation": edge, "issues": [], "evidence": []}
        issues = annotation["issues"]
        if not isinstance(edge, dict):
            issues.append("invalid_relation_shape")
            result["annotations"].append(annotation)
            continue
        src, dst = identifier(edge.get("source")), identifier(edge.get("target"))
        label = edge.get("relation")
        if not isinstance(label, str):
            issues.append("invalid_relation_label")
            label = None
        source, target = by_id.get(src), by_id.get(dst)
        if not source or not target:
            issues.append("unknown_or_ambiguous_endpoint")
        elif source["issues"] or target["issues"]:
            issues.append("endpoint_requires_review")
        if src == dst:
            issues.append("self_relation_requires_review")
        if not source or (source["label"], label) not in SOURCE_RELATIONS:
            issues.append("relation_outside_source_module")
        values = evidence_values(edge)
        if not values or not any(v.strip() for v in values):
            issues.append("missing_evidence_text")
        for value in values:
            hits = locate(value, text, source_changes)
            if not hits:
                issues.append("evidence_not_found")
            elif len(hits) != 1:
                issues.append("evidence_occurrence_ambiguous")
            annotation["evidence"].extend(hits)
        ev_text = "\n".join(x["text"] for x in annotation["evidence"])
        if ev_text and source and target:
            # These are REVIEW HEURISTICS, not asserted semantic contradictions.
            if not exact_hits(source["name"], ev_text) or not exact_hits(target["name"], ev_text):
                issues.append("evidence_endpoint_binding_needs_review")
            if NEGATION.search(ev_text):
                issues.append("negation_scope_needs_review")
            if label in {"is_core_symptom_of", "is_associated_symptom_of"} and target["label"] != "Disease":
                issues.append("symptom_target_type_needs_review")
            if label in {"has_manifestation", "part_of", "subtype_of"} and ENUMERATION.search(ev_text):
                issues.append("enumeration_vs_hierarchy_needs_review")
        annotation["issues"] = sorted(set(issues))
        result["annotations"].append(annotation)
        by_pair[(src, dst)].append(annotation)
    for (src, dst), annotations in by_pair.items():
        valid = not result["issues"] and all(not a["issues"] for a in annotations)
        grouped = defaultdict(list)
        for a in annotations:
            label = a["annotation"].get("relation")
            grouped[label if isinstance(label, str) else None].append(a)
        decisions = []
        for label, values in grouped.items():
            evidence = {(x["start"], x["end"]): x for a in values for x in a["evidence"]}
            decisions.append({"relation": label, "evidence": [evidence[k] for k in sorted(evidence)],
                              "original_annotation_indices": [a["index"] for a in values]})
        result["pairs"].append({"source": src, "target": dst,
                                "status": "mechanically_supported_silver" if valid else "review",
                                "relations": sorted(decisions, key=lambda x: str(x["relation"])),
                                "annotation_indices": [a["index"] for a in annotations]})
    # Order depends on input mentions, not relation labels or positive status.
    def pair_order(p):
        def pos(key):
            mentions = by_id.get(key, {}).get("mentions", [])
            return mentions[0]["start"] if mentions else len(text)
        return pos(p["source"]), pos(p["target"]), str(p["source"]), str(p["target"])
    result["pairs"].sort(key=pair_order)
    result["status"] = "review" if (result["issues"] or any(e["issues"] for e in result["entities"]) or any(a["issues"] for a in result["annotations"])) else "mechanically_checked_silver"
    return result


def split_assignment(records, frozen, source_sha):
    if frozen["source_train_sha256"] != source_sha:
        raise ValueError("Frozen split belongs to a different source file")
    train, tune = set(frozen["train_groups"]), set(frozen["tune_groups"])
    groups = source_groups(records)
    if train & tune or train | tune != set(groups):
        raise ValueError("Frozen source-group split is incomplete or overlapping")
    return groups, ["train" if group in train else "tune" for group in groups]


def run_pipeline(input_path, split_manifest, output_dir, *, expected_sha256):
    input_path, split_manifest, output_dir = map(Path, (input_path, split_manifest, output_dir))
    if output_dir.exists():
        raise ValueError("Output must be a new versioned directory; refusing overwrite")
    raw = input_path.read_bytes()
    digest = sha(raw)
    if digest != expected_sha256:
        raise ValueError("Input SHA256 mismatch; refusing an unregistered data version")
    records = json.loads(raw)
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Expected a list of teacher chat records")
    schema = ONTOLOGY["version"]
    if not all(schema in _message(r, "system") for r in records):
        raise ValueError("Teacher schema provenance mismatch; explicit migration required")
    frozen = json.loads(split_manifest.read_text())
    groups, splits = split_assignment(records, frozen, digest)
    repaired = [repair_record(r, i, g, s) for i, (r, g, s) in enumerate(zip(records, groups, splits))]
    # A repair can make two previously different texts identical. Do not move records
    # between frozen splits or permit overlapping sources into the training candidate.
    hashes = defaultdict(list)
    for r in repaired:
        if r["source_type"] == "free_text":
            hashes[sha(normalized(r["source"]["text"]))].append(r)
    overlap_hashes = 0
    for items in hashes.values():
        if len({r["split"] for r in items}) > 1:
            overlap_hashes += 1
            for r in items:
                r["issues"].append("cross_split_source_overlap_after_repair")
                r["status"] = "review"
                for p in r["pairs"]:
                    p["status"] = "review"
    summary = Counter(records=len(records))
    by_split = defaultdict(Counter)
    issue_counts = Counter()
    review, pairs, changes = [], [], []
    for r in repaired:
        c = by_split[r["split"]]
        c["records"] += 1
        c[r["source_type"]] += 1
        c["records_with_repairs"] += bool(r["changes"])
        c["text_repair_occurrences_source"] += sum(x["path"] == "/source" for x in r["changes"])
        c["text_repair_occurrences_graph"] += sum(x["path"].startswith("/graph") and x.get("op") != "canonical_label_case" for x in r["changes"])
        c["entities"] += len(r["entities"])
        c["grounded_entities"] += sum(bool(e["mentions"]) for e in r["entities"])
        c["original_annotations"] += len(r["annotations"])
        c["multilabel_pairs_preserved"] += sum(len(p["relations"]) > 1 for p in r["pairs"])
        c["single_character_spans"] += sum(len(s["text"]) == 1 for s in r.get("spans", []))
        c["spans"] += len(r.get("spans", []))
        for change in r["changes"]:
            changes.append({"record_id": r["record_id"], "split": r["split"], **change})
        items = [("record", None, r["issues"])]
        items += [("entity", e["index"], e["issues"]) for e in r["entities"]]
        items += [("relation", a["index"], a["issues"]) for a in r["annotations"]]
        for kind, index, issues in items:
            if issues:
                issue_counts.update(issues)
                review.append({"record_id": r["record_id"], "split": r["split"], "kind": kind,
                               "index": index, "reasons": issues,
                               "decision": "PENDING", "not_a_negative": True})
        for p in r["pairs"]:
            c[f"pairs_{p['status']}"] += 1
            c[f"relation_labels_{p['status']}"] += len(p["relations"])
            if p["status"] == "mechanically_supported_silver":
                pairs.append({"record_id": r["record_id"], "group_id": r["group_id"], "split": r["split"], **p,
                              "annotation_source": "teacher_silver_not_human_gold",
                              "negative_policy": "unannotated_pairs_are_unknown",
                              "ready_for_automatic_training": False})
    for counts in by_split.values():
        summary.update({k: v for k, v in counts.items() if k != "records"})
    # Private output includes original/repaired text; stdout/report contains counts only.
    output_dir.mkdir(parents=True, mode=0o700)
    def write_jsonl(name, rows):
        p = output_dir / name
        with p.open("x") as handle:
            p.chmod(0o600)
            for row in rows:
                handle.write(dumps(row) + "\n")
    write_jsonl("records.jsonl", repaired)
    write_jsonl("changes.jsonl", changes)
    write_jsonl("review_queue.jsonl", review)
    write_jsonl("pair_candidates.jsonl", pairs)
    split_rows = [{"record_id": r["record_id"], "source_index": r["source_index"], "group_id": r["group_id"], "split": r["split"]} for r in repaired]
    write_jsonl("frozen_split.jsonl", split_rows)
    code_files = [Path(__file__), ROOT / "agent/generative/protocol.py", ROOT / "agent/kg_agent/source.py", ROOT / "agent/kg_agent/ontology.json", ROOT / "schemas/v2.0.0/schema.json"]
    manifest = {"version": VERSION, "status": "candidate_release_review_required",
                "source_path": str(input_path.resolve()), "source_sha256": digest,
                "schema": schema, "frozen_split_sha256": sha(split_manifest.read_bytes()),
                "train_groups": len(frozen["train_groups"]), "tune_groups": len(frozen["tune_groups"]),
                "summary": dict(summary), "by_split": {k: dict(v) for k, v in by_split.items()},
                "review_reason_counts": dict(issue_counts), "review_items": len(review),
                "cross_split_source_hashes": overlap_hashes,
                "raw_record_conservation": len(records) == len(repaired),
                "input_unchanged": sha(input_path.read_bytes()) == digest,
                "code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in code_files},
                "output_sha256": {p.name: sha(p.read_bytes()) for p in output_dir.glob("*.jsonl")},
                "release_gates": {"human_semantic_adjudication": False, "negative_pool_reviewed": False,
                                  "historical_dev_overlap_checked": False, "new_model_evaluation_complete": False},
                "policy": ["Never overwrite original inputs or frozen tests", "No first-sentence evidence fallback",
                           "No ontology migration or speculative entity merging", "Keep all labels for each ordered pair",
                           "One unresolved label quarantines the whole pair", "Unannotated does not mean NONE",
                           "Mechanical support is not semantic entailment", "No model training or external API called"]}
    if not manifest["input_unchanged"] or not manifest["raw_record_conservation"]:
        raise RuntimeError("Data conservation check failed")
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x") as handle:
        manifest_path.chmod(0o600)
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--frozen-split", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    result = run_pipeline(args.input, args.frozen_split, args.output_dir, expected_sha256=args.expected_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert full-graph teacher records into stage-specific SFT JSONL.

Input is the existing list of records with ``messages`` containing system,
user, and assistant messages.  The script intentionally drops samples whose
gold concepts cannot be grounded to source spans instead of teaching the
model to hallucinate them.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from kg_agent.contracts import Entity, EntityCandidate, Graph, SourceDocument, allowed_relations, normalize_label
from kg_agent.normalize import normalize_text
from kg_agent.prompts import build_entity_prompt, build_relation_prompt, extract_json_object
from kg_agent.source import SourceRouter, chunk_spans, extract_source_payload, resolve_span_ref


def _message(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def _graph_from_teacher(text: str) -> dict[str, Any]:
    value = extract_json_object(text)
    if isinstance(value.get("output"), str):
        try:
            value = extract_json_object(value["output"])
        except ValueError:
            pass
    if not isinstance(value.get("entities"), list):
        value["entities"] = []
    if not isinstance(value.get("relations"), list):
        value["relations"] = []
    return value


def _raw_user_payload(text: str) -> Any:
    return extract_source_payload(text)


def _node_name(node: dict[str, Any]) -> str:
    value = node.get("name") or node.get("text") or node.get("properties", {}).get("Name")
    return str(value or "").strip()


def _align(name: str, spans: list[Any]) -> str | None:
    needle = normalize_text(name)
    if not needle:
        return None
    exact = [span for span in spans if normalize_text(span.text) == needle]
    if exact:
        return min(exact, key=lambda span: (span.start, len(span.text))).id
    contained = [span for span in spans if needle in normalize_text(span.text)]
    if contained:
        span = min(contained, key=lambda span: (len(span.text), span.start))
        # Prefer a sentence-local phrase span so the model is not trained to
        # use the entire sentence as the entity name.
        start = span.text.casefold().find(name.casefold())
        if start >= 0:
            return f"{span.id}:{start}-{start + len(name)}"
        return span.id
    # Permit a sentence-level target name to align when punctuation differs.
    compact_needle = re.sub(r"[^\w\u4e00-\u9fff]+", "", needle)
    for span in spans:
        compact_span = re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_text(span.text))
        if compact_needle and compact_needle in compact_span:
            return span.id
    return None


def _importance(node: dict[str, Any]) -> str:
    label = (normalize_label(node.get("label")) or "").casefold()
    if label == "disease":
        return "core"
    if "criterion" in label:
        return "required"
    return "associated"


def _source_mention(span_id: str, name: str, spans: list[Any]) -> str:
    """Use the exact source substring for mention-copy targets.

    Gold names occasionally differ from the source only by punctuation or
    whitespace.  The v4 entity contract deliberately teaches the model to
    copy source text, so the target must be the text at the aligned offsets,
    not the teacher's normalized name.
    """

    document = SourceDocument(source_type="free_text", raw="", spans=spans)
    resolved = resolve_span_ref(document, span_id)
    return resolved.text if resolved is not None else name


def _canonical_gold_entities(nodes: list[dict[str, Any]], spans: list[Any]) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for node in nodes:
        node_id = str(node.get("id") or "").strip()
        label = normalize_label(node.get("label"))
        name = _node_name(node)
        span_id = _align(name, spans)
        if not node_id or not label or not name or not span_id:
            missing.append({"id": node_id, "name": name, "reason": "not_groundable"})
            continue
        rows.append(
            {
                "gold_id": node_id,
                "label": label,
                "name": name,
                "mention": _source_mention(span_id, name, spans),
                "span_id": span_id,
                "importance": _importance(node),
            }
        )

    # Match the runtime convention: Disease first, then source order, then
    # label/name. This keeps relation training IDs stable without asking the
    # LLM to generate IDs.
    span_offsets = {span.id: span.start for span in spans}
    rows.sort(key=lambda row: (0 if row["label"] == "Disease" else 1, span_offsets[row["span_id"].split(":", 1)[0]], row["label"], normalize_text(row["name"])))
    counters: dict[str, int] = {}
    prefixes = {
        "Disease": "D", "Symptom": "S", "Diagnostic Criterion": "DC", "Etiology": "E",
        "Risk Factor": "RF", "Treatment": "T", "Prognostic Factor": "PF", "Functional Impact": "FI",
        "Test": "X", "Medication": "M", "Procedure": "P", "Anatomy": "A", "Phenotype": "PH",
        "Risk": "R", "Assessment Scale": "AS", "Assessment": "AS", "Examination": "EX",
    }
    gold_to_stage: dict[str, str] = {}
    for row in rows:
        prefix = prefixes.get(row["label"], "O")
        counters[prefix] = counters.get(prefix, 0) + 1
        stage_id = f"{prefix}{counters[prefix]}"
        if row["label"] == "Disease" and not any(value == "D1" for value in gold_to_stage.values()):
            stage_id = "D1"
            counters["D"] = max(counters["D"], 1)
        row["stage_id"] = stage_id
        gold_to_stage[row["gold_id"]] = stage_id
    return rows, gold_to_stage, missing


def _evidence_span(value: Any, spans: list[Any], source_name: str, target_name: str) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    text = str(value or "").strip()
    span_ids = {span.id for span in spans}
    if text in span_ids:
        return text
    if text:
        match = [span for span in spans if normalize_text(text) in normalize_text(span.text)]
        if match:
            return min(match, key=lambda span: (len(span.text), span.start)).id
    both = [
        span for span in spans
        if normalize_text(source_name) in normalize_text(span.text)
        and normalize_text(target_name) in normalize_text(span.text)
    ]
    if both:
        return both[0].id
    return spans[0].id if spans else ""


def make_records(
    record: dict[str, Any],
    index: int,
    *,
    mention_copy: bool = False,
    relation_emit_none: bool = False,
    relation_batch_size: int = 32,
    relation_negative_ratio: int = 1,
    relation_negative_floor: int = 8,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    raw_user = _raw_user_payload(_message(record, "user"))
    document = SourceRouter().route(raw_user)
    teacher = _graph_from_teacher(_message(record, "assistant"))
    report: dict[str, Any] = {"index": index, "source_type": document.source_type, "missing_entities": []}
    if document.source_type == "structured_icd":
        report["skipped"] = "deterministic_source"
        return None, None, report

    nodes = [node for node in teacher["entities"] if isinstance(node, dict)]
    rows, gold_to_stage, missing = _canonical_gold_entities(nodes, document.spans)
    report["missing_entities"] = missing
    report["gold_entities"] = len(nodes)
    report["grounded_entities"] = len(rows)
    entity_records: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunk_spans(document.spans)):
        chunk_ids = {span.id for span in chunk}
        targets: list[dict[str, Any]] = []
        for row in rows:
            if row["span_id"].split(":", 1)[0] not in chunk_ids:
                continue
            target: dict[str, Any] = {
                "span_id": row["span_id"].split(":", 1)[0] if mention_copy else row["span_id"],
                "label": row["label"],
                "importance": row["importance"],
            }
            if mention_copy:
                target["mention"] = row["mention"]
            targets.append(target)
        prompt = build_entity_prompt(
            [{"span_id": span.id, "text": span.text, "section": span.section} for span in chunk],
            max_candidates=64,
            mention_copy=mention_copy,
        )
        entity_records.append(
            {
                "id": f"{index}:{chunk_index}",
                "sample_id": index,
                "chunk_index": chunk_index,
                "stage": "entities",
                "prompt": prompt,
                "completion": json.dumps({"entity_candidates": targets}, ensure_ascii=False, separators=(",", ":")),
            }
        )

    entity_rows = [
        {
            "id": row["stage_id"],
            "label": row["label"],
            "name": row["name"],
            "span_id": row["span_id"],
            "importance": row["importance"],
        }
        for row in rows
    ]
    stage_by_id = {row["stage_id"]: row for row in rows}
    relation_specs: list[dict[str, str]] = []
    for relation in teacher["relations"]:
        if not isinstance(relation, dict):
            continue
        source = gold_to_stage.get(str(relation.get("source") or ""))
        target = gold_to_stage.get(str(relation.get("target") or ""))
        relation_name = str(relation.get("relation") or relation.get("type") or relation.get("label") or "").strip()
        source_row = stage_by_id.get(source or "")
        target_row = stage_by_id.get(target or "")
        allowed = allowed_relations(source_row["label"], target_row["label"]) if source_row and target_row else ()
        if not source_row or not target_row or relation_name not in allowed:
            report.setdefault("skipped_relations", []).append({"source": source, "target": target, "relation": relation_name})
            continue
        evidence = relation.get("evidence_span_id") or relation.get("evidence_sentence") or relation.get("evidence")
        evidence_id = _evidence_span(evidence, document.spans, source_row["name"], target_row["name"])
        relation_specs.append({"source": source, "target": target, "relation": relation_name, "evidence_id": evidence_id})

    relation_records: list[dict[str, Any]] = []
    relation_candidate_count = 0
    positive_relation_count = 0
    # Relation prompts are chunk-local, just like inference. Endpoint rows for
    # a relation are also included in the chunk containing its evidence, even
    # when the two mentions occur in different text chunks.
    for chunk_index, chunk in enumerate(chunk_spans(document.spans)):
        chunk_ids = {span.id for span in chunk}
        chunk_specs = [spec for spec in relation_specs if spec["evidence_id"].split(":", 1)[0] in chunk_ids]
        required_ids = {spec["source"] for spec in chunk_specs} | {spec["target"] for spec in chunk_specs}
        active_rows = [
            row for row in entity_rows
            if row["span_id"].split(":", 1)[0] in chunk_ids or row["id"] in required_ids
        ]
        pairs: list[dict[str, Any]] = []
        for source in active_rows:
            for target in active_rows:
                if source["id"] == target["id"]:
                    continue
                allowed = allowed_relations(source["label"], target["label"])
                if allowed:
                    pairs.append({"pair_id": "", "source": source["id"], "target": target["id"], "allowed_relations": list(allowed)})
        positive_by_endpoint = {
            (spec["source"], spec["target"]): spec for spec in chunk_specs
        }
        positive_endpoint_keys = set(positive_by_endpoint)
        positives_first = [pair for pair in pairs if (pair["source"], pair["target"]) in positive_endpoint_keys]
        negatives = [pair for pair in pairs if (pair["source"], pair["target"]) not in positive_endpoint_keys]
        if relation_emit_none:
            if positives_first:
                negative_budget = max(
                    relation_negative_floor,
                    min(
                        max(0, relation_batch_size - len(positives_first)),
                        len(positives_first) * max(1, relation_negative_ratio),
                    ),
                )
            else:
                negative_budget = relation_negative_floor
            selected_keys = positive_endpoint_keys | {
                (pair["source"], pair["target"])
                for pair in negatives[:negative_budget]
            }
            # Keep the same deterministic order used by inference.  Putting
            # positives first makes the first positions a hidden label cue and
            # causes the model to over-predict relations at runtime.
            pairs = [
                pair
                for pair in pairs
                if (pair["source"], pair["target"]) in selected_keys
            ]
        else:
            pairs = positives_first + negatives[: max(0, 128 - len(positives_first))]
        for pair_index, pair in enumerate(pairs, start=1):
            pair["pair_id"] = f"P{pair_index:04d}"
        relation_candidate_count += len(pairs)
        if not pairs:
            continue
        pair_by_nodes = {(pair["source"], pair["target"]): pair for pair in pairs}
        positive_relation_count += sum(
            (spec["source"], spec["target"]) in pair_by_nodes for spec in chunk_specs
        )
        span_rows = [
            {"span_id": span.id, "text": span.text, "section": span.section}
            for span in chunk
        ]
        if relation_emit_none:
            for batch_index, batch_start in enumerate(range(0, len(pairs), max(1, relation_batch_size))):
                pair_batch = pairs[batch_start : batch_start + max(1, relation_batch_size)]
                decisions: list[dict[str, str]] = []
                for pair in pair_batch:
                    spec = positive_by_endpoint.get((pair["source"], pair["target"]))
                    if spec:
                        decisions.append(
                            {
                                "pair_id": pair["pair_id"],
                                "relation": spec["relation"],
                                "evidence_span_id": spec["evidence_id"],
                            }
                        )
                    else:
                        decisions.append(
                            {
                                "pair_id": pair["pair_id"],
                                "relation": "NONE",
                                "evidence_span_id": "",
                            }
                        )
                prompt = build_relation_prompt(
                    active_rows,
                    pair_batch,
                    span_rows,
                    emit_none=True,
                )
                relation_records.append(
                    {
                        "id": f"{index}:{chunk_index}:{batch_index}",
                        "sample_id": index,
                        "chunk_index": chunk_index,
                        "batch_index": batch_index,
                        "stage": "relations",
                        "prompt": prompt,
                        "completion": json.dumps(
                            {"relations": decisions},
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                )
        else:
            positive: dict[str, dict[str, str]] = {}
            for spec in chunk_specs:
                pair = pair_by_nodes.get((spec["source"], spec["target"]))
                if not pair:
                    continue
                positive[pair["pair_id"]] = {
                    "pair_id": pair["pair_id"],
                    "relation": spec["relation"],
                    "evidence_span_id": spec["evidence_id"],
                }
            prompt = build_relation_prompt(active_rows, pairs, span_rows)
            relation_records.append(
                {
                    "id": f"{index}:{chunk_index}",
                    "sample_id": index,
                    "chunk_index": chunk_index,
                    "stage": "relations",
                    "prompt": prompt,
                    "completion": json.dumps(
                        {"relations": list(positive.values())},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
    report["relation_candidates"] = relation_candidate_count
    report["positive_relations"] = positive_relation_count
    return {"records": entity_records}, {"records": relation_records}, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Existing list-of-records JSON")
    parser.add_argument("--entity-output", required=True)
    parser.add_argument("--relation-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mention-copy", action="store_true")
    parser.add_argument("--relation-emit-none", action="store_true")
    parser.add_argument("--relation-batch-size", type=int, default=32)
    parser.add_argument("--relation-negative-ratio", type=int, default=1)
    parser.add_argument("--relation-negative-floor", type=int, default=8)
    args = parser.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    entity_records: list[dict[str, Any]] = []
    relation_records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        entities, relations, report = make_records(
            record,
            index,
            mention_copy=args.mention_copy,
            relation_emit_none=args.relation_emit_none,
            relation_batch_size=args.relation_batch_size,
            relation_negative_ratio=args.relation_negative_ratio,
            relation_negative_floor=args.relation_negative_floor,
        )
        if entities:
            entity_records.extend(entities["records"])
        if relations:
            relation_records.extend(relations["records"])
        reports.append(report)

    for path, values in (
        (args.entity_output, entity_records),
        (args.relation_output, relation_records),
    ):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in values) + "\n", encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"entity_records": len(entity_records), "relation_records": len(relation_records), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

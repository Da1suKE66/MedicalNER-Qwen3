#!/usr/bin/env python3
"""Compute teacher-relative structural metrics for a comparison artifact.

These metrics are diagnostic only: the DeepSeek target is treated as a noisy
reference, not as a medical gold annotation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)
OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.I | re.S)


def normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def parse_json_block(item: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(item, dict):
        return None, "missing"
    text = str(item.get("output") or item.get("raw") or "").strip()
    match = OUTPUT_RE.search(text)
    if match:
        text = match.group(1).strip()
    # Batched generation can pad rows after EOS; those special tokens are not
    # part of the JSON and should be audited separately, not counted as a JSON
    # syntax failure.
    text = FENCE_RE.sub("", text)
    text = re.sub(r"<\|(?:im_end|endoftext|eot_id|end_of_text)\|>", "", text, flags=re.I)
    text = text.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        # If a backend appended harmless text after a complete object, retain
        # the first JSON object while leaving true mid-object truncation invalid.
        try:
            parsed, end = json.JSONDecoder().raw_decode(text)
            if text[end:].strip():
                return None, "invalid_json"
        except Exception:
            return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "non_object"
    return parsed, "ok"


def schema_violations(graph: dict[str, Any] | None) -> list[str]:
    if graph is None:
        return ["not_object"]
    errors: list[str] = []
    if set(graph) != {"entities", "relations"}:
        errors.append("top_level_keys")
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not isinstance(entities, list):
        errors.append("entities_not_list")
        entities = []
    if not isinstance(relations, list):
        errors.append("relations_not_list")
        relations = []
    ids: set[str] = set()
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(f"entity_{i}_not_object")
            continue
        if set(entity) != {"id", "label", "name", "properties"}:
            errors.append(f"entity_{i}_keys")
        entity_id = str(entity.get("id") or "")
        if not entity_id or entity_id in ids:
            errors.append(f"entity_{i}_id")
        ids.add(entity_id)
        if not isinstance(entity.get("properties"), dict):
            errors.append(f"entity_{i}_properties")
    for i, relation in enumerate(relations):
        if not isinstance(relation, dict):
            errors.append(f"relation_{i}_not_object")
            continue
        if set(relation) != {"source", "target", "relation", "evidence"}:
            errors.append(f"relation_{i}_keys")
        if relation.get("source") not in ids or relation.get("target") not in ids:
            errors.append(f"relation_{i}_endpoint")
    return errors


def entity_key(entity: dict[str, Any]) -> tuple[str, str, str]:
    properties = entity.get("properties") if isinstance(entity.get("properties"), dict) else {}
    code = properties.get("ICD-11 Code", properties.get("icdcode", ""))
    return normalize(entity.get("label")), normalize(entity.get("name")), normalize(code)


def graph_sets(graph: dict[str, Any] | None) -> tuple[set[tuple[str, str, str]], set[tuple[Any, ...]]]:
    if not isinstance(graph, dict):
        return set(), set()
    entities = [x for x in graph.get("entities", []) if isinstance(x, dict)]
    by_id = {str(x.get("id")): entity_key(x) for x in entities}
    entity_set = set(by_id.values())
    relations: set[tuple[Any, ...]] = set()
    for rel in graph.get("relations", []):
        if not isinstance(rel, dict):
            continue
        source = by_id.get(str(rel.get("source")), ("<missing>", str(rel.get("source")), ""))
        target = by_id.get(str(rel.get("target")), ("<missing>", str(rel.get("target")), ""))
        relations.add((source, target, normalize(rel.get("relation"))))
    return entity_set, relations


def graph_content_canonical(graph: dict[str, Any] | None) -> Any:
    """Canonicalize ordering while retaining all content and relation evidence."""
    if not isinstance(graph, dict):
        return None
    entities = []
    for entity in graph.get("entities", []):
        if isinstance(entity, dict):
            entities.append(
                (
                    entity.get("label"),
                    entity.get("name"),
                    json.dumps(entity.get("properties"), ensure_ascii=False, sort_keys=True),
                )
            )
    relations = []
    for relation in graph.get("relations", []):
        if isinstance(relation, dict):
            relations.append(tuple(sorted(relation.items())))
    return {"entities": sorted(entities), "relations": sorted(relations)}


def source_text_for_case(case: dict[str, Any], source_chunk: dict[str, Any] | None) -> str:
    """Return the actual chunk text used for grounding checks."""
    if isinstance(source_chunk, dict):
        value = source_chunk.get("input")
        if isinstance(value, str) and value.strip():
            return value
    user = str(case.get("user") or "")
    return user.split("Medical text:\n", 1)[-1]


def source_fields_for_chunk(source_chunk: dict[str, Any] | None) -> list[str]:
    if not isinstance(source_chunk, dict):
        return ["unknown"]
    fields = source_chunk.get("snapshot_provenance", {}).get("fields", [])
    names = sorted({str(item.get("source_field")) for item in fields if isinstance(item, dict) and item.get("source_field")})
    return names or ["unknown"]


def grounding_items(graph: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Collect strings that should be literally grounded in the source chunk."""
    if not isinstance(graph, dict):
        return []
    items: list[tuple[str, str]] = []
    for entity in graph.get("entities", []):
        if not isinstance(entity, dict):
            continue
        for kind, value in [("entity_name", entity.get("name"))]:
            if isinstance(value, str) and value.strip():
                items.append((kind, value))
        props = entity.get("properties")
        if isinstance(props, dict):
            for key, value in props.items():
                if isinstance(value, str) and value.strip():
                    items.append((f"property:{key}", value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            items.append((f"property:{key}", item))
    for relation in graph.get("relations", []):
        if isinstance(relation, dict) and isinstance(relation.get("evidence"), str) and relation["evidence"].strip():
            items.append(("relation_evidence", relation["evidence"]))
    return items


def grounding_counts(graph: dict[str, Any] | None, source: str) -> tuple[int, int, Counter]:
    source_norm = normalize(source)
    total = 0
    violations = 0
    by_kind: Counter = Counter()
    for kind, value in grounding_items(graph):
        total += 1
        if normalize(value) not in source_norm:
            violations += 1
            by_kind[kind] += 1
    return total, violations, by_kind


def generation_completion_status(item: dict[str, Any] | None) -> str:
    if not isinstance(item, dict):
        return "missing"
    raw = str(item.get("raw") or "")
    if "</output>" in raw or re.search(r"<\|(?:im_end|endoftext|eot_id|end_of_text)\|>", raw, flags=re.I):
        return "terminated_special"
    if raw.rstrip().endswith("}") or raw.rstrip().endswith("]"):
        return "terminated_json"
    return "possibly_truncated"


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument(
        "--source-chunks",
        type=Path,
        help="optional frozen chunks.json; adds source-field buckets and exact grounding checks",
    )
    args = ap.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    source_chunks: dict[int, dict[str, Any]] = {}
    if args.source_chunks:
        frozen = json.loads(args.source_chunks.read_text(encoding="utf-8"))
        source_chunks = {i: item for i, item in enumerate(frozen) if isinstance(item, dict)}
    model_names = ["base_qwen", "output_priority", "output_only"]
    report: dict[str, Any] = {"metadata": payload.get("metadata", {}), "cases": len(cases), "models": {}}
    for model in model_names:
        stats = Counter()
        relation_counts = Counter()
        entity_count = []
        relation_count = []
        field_stats: dict[str, Counter] = defaultdict(Counter)
        field_relation_counts: dict[str, Counter] = defaultdict(Counter)
        grounding_kind_counts = Counter()
        for case in cases:
            case_id = int(case.get("id", -1))
            source_chunk = source_chunks.get(case_id)
            source = source_text_for_case(case, source_chunk)
            fields = source_fields_for_chunk(source_chunk)
            target, target_status = parse_json_block(case.get("deepseek_original"))
            pred, pred_status = parse_json_block(case.get(model))
            stats[f"target_parse_{target_status}"] += 1
            stats[f"pred_parse_{pred_status}"] += 1
            stats[f"generation_{generation_completion_status(case.get(model))}"] += 1
            stats["content_exact"] += graph_content_canonical(target) == graph_content_canonical(pred)
            target_entities, target_relations = graph_sets(target)
            pred_entities, pred_relations = graph_sets(pred)
            tp_e = len(target_entities & pred_entities)
            fp_e = len(pred_entities - target_entities)
            fn_e = len(target_entities - pred_entities)
            tp_r = len(target_relations & pred_relations)
            fp_r = len(pred_relations - target_relations)
            fn_r = len(target_relations - pred_relations)
            stats["entity_tp"] += tp_e
            stats["entity_fp"] += fp_e
            stats["entity_fn"] += fn_e
            stats["relation_tp"] += tp_r
            stats["relation_fp"] += fp_r
            stats["relation_fn"] += fn_r
            ground_total, ground_violations, ground_kinds = grounding_counts(pred, source)
            stats["grounding_items"] += ground_total
            stats["grounding_violations"] += ground_violations
            grounding_kind_counts.update(ground_kinds)
            for field in fields:
                field_stats[field]["cases"] += 1
                field_stats[field]["entity_tp"] += tp_e
                field_stats[field]["entity_fp"] += fp_e
                field_stats[field]["entity_fn"] += fn_e
                field_stats[field]["relation_tp"] += tp_r
                field_stats[field]["relation_fp"] += fp_r
                field_stats[field]["relation_fn"] += fn_r
                field_stats[field]["schema_valid"] += bool(pred is not None and not schema_violations(pred))
                field_stats[field][f"pred_parse_{pred_status}"] += 1
                field_stats[field]["grounding_items"] += ground_total
                field_stats[field]["grounding_violations"] += ground_violations
            if pred is not None:
                errors = schema_violations(pred)
                stats["schema_valid"] += not errors
                stats["schema_violations"] += len(errors)
                entity_count.append(len(pred.get("entities") or []))
                relation_count.append(len(pred.get("relations") or []))
                relation_counts.update(normalize(x.get("relation")) for x in pred.get("relations", []) if isinstance(x, dict))
            else:
                stats["schema_valid"] += 0
        tp_e, fp_e, fn_e = stats["entity_tp"], stats["entity_fp"], stats["entity_fn"]
        tp_r, fp_r, fn_r = stats["relation_tp"], stats["relation_fp"], stats["relation_fn"]

        def metric_block(prefix: str, values: Counter) -> dict[str, Any]:
            tp, fp, fn = values[f"{prefix}_tp"], values[f"{prefix}_fp"], values[f"{prefix}_fn"]
            predicted = tp + fp
            target_total = tp + fn
            return {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "predicted_total": predicted,
                "target_total": target_total,
                "precision": tp / predicted if predicted else 0.0,
                "recall": tp / target_total if target_total else 0.0,
                "f1": f1(tp, fp, fn),
                "hallucination_rate": fp / predicted if predicted else 0.0,
                "under_extraction_rate": fn / target_total if target_total else 0.0,
            }

        by_field: dict[str, Any] = {}
        for field, values in sorted(field_stats.items()):
            by_field[field] = {
                "cases": values["cases"],
                "prediction_parse": {k: v for k, v in values.items() if k.startswith("pred_parse_")},
                "schema_valid_cases": values["schema_valid"],
                "entity": metric_block("entity", values),
                "relation": metric_block("relation", values),
                "grounding_items": values["grounding_items"],
                "grounding_violations": values["grounding_violations"],
                "grounding_violation_rate": values["grounding_violations"] / values["grounding_items"] if values["grounding_items"] else 0.0,
            }
        report["models"][model] = {
            "cases": len(cases),
            "target_relative_parse": {k: v for k, v in stats.items() if k.startswith("target_parse_")},
            "prediction_parse": {k: v for k, v in stats.items() if k.startswith("pred_parse_")},
            "prediction_schema_valid_cases": stats["schema_valid"],
            "prediction_schema_violation_count": stats["schema_violations"],
            "content_exact_cases": stats["content_exact"],
            "generation_completion": {k.removeprefix("generation_"): v for k, v in stats.items() if k.startswith("generation_")},
            "entity": metric_block("entity", stats),
            "relation": metric_block("relation", stats),
            "grounding": {"items": stats["grounding_items"], "violations": stats["grounding_violations"], "violation_rate": stats["grounding_violations"] / stats["grounding_items"] if stats["grounding_items"] else 0.0, "violations_by_kind": dict(grounding_kind_counts)},
            "mean_predicted_entities": sum(entity_count) / len(entity_count) if entity_count else 0.0,
            "mean_predicted_relations": sum(relation_count) / len(relation_count) if relation_count else 0.0,
            "predicted_relation_types": dict(relation_counts),
            "by_source_field": by_field,
        }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

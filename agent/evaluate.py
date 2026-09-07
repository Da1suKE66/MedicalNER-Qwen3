#!/usr/bin/env python3
"""Evaluate graph predictions without rewarding malformed or ungrounded JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kg_agent.normalize import normalize_text
from kg_agent.prompts import extract_json_object
from kg_agent.source import SourceRouter


def _graph(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and isinstance(value.get("output"), dict):
        value = value["output"]
    if isinstance(value, dict) and isinstance(value.get("output_only"), dict):
        value = value["output_only"]
    if isinstance(value, dict) and isinstance(value.get("output"), str):
        try:
            value = extract_json_object(value["output"])
        except ValueError:
            return None
    if isinstance(value, str):
        try:
            value = extract_json_object(value)
        except ValueError:
            return None
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list) or not isinstance(value.get("relations"), list):
        return None
    return value


def _node_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("properties", {}).get("Name") or "").strip()


def _entity_set(graph: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(node.get("label") or "").strip(), normalize_text(_node_name(node)))
        for node in graph["entities"]
        if isinstance(node, dict) and _node_name(node)
    }


def _relation_set(graph: dict[str, Any]) -> set[tuple[tuple[str, str], str, tuple[str, str]]]:
    by_id = {
        str(node.get("id")): (str(node.get("label") or ""), normalize_text(_node_name(node)))
        for node in graph["entities"]
        if isinstance(node, dict) and node.get("id") is not None
    }
    result: set[tuple[tuple[str, str], str, tuple[str, str]]] = set()
    for relation in graph["relations"]:
        if not isinstance(relation, dict):
            continue
        source = by_id.get(str(relation.get("source")))
        target = by_id.get(str(relation.get("target")))
        name = str(relation.get("relation") or relation.get("type") or "").strip()
        if source and target and name:
            result.add((source, name, target))
    return result


def _prf(predicted: set[Any], gold: set[Any]) -> dict[str, float | int]:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _records_from_predictions(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("cases"), list):
        cases = sorted(value["cases"], key=lambda item: int(item.get("id", 0)))
        return [case.get("output_only") or case.get("output") for case in cases]
    return [value]


def _load_predictions(path: str) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gold_records = json.loads(Path(args.gold_data).read_text(encoding="utf-8"))
    predictions = _records_from_predictions(_load_predictions(args.predictions))
    if not predictions:
        raise ValueError(f"predictions file is empty: {args.predictions}")
    if len(predictions) != len(gold_records):
        raise ValueError(
            f"prediction count mismatch: got {len(predictions)}, expected {len(gold_records)}"
        )
    router = SourceRouter()
    totals = {
        "samples": len(gold_records),
        "json_valid": 0,
        "entity": {"tp": 0, "fp": 0, "fn": 0},
        "relation": {"tp": 0, "fp": 0, "fn": 0},
        "relation_type": {"tp": 0, "fp": 0, "fn": 0},
        "source_endpoint": {"tp": 0, "fp": 0, "fn": 0},
        "target_endpoint": {"tp": 0, "fp": 0, "fn": 0},
        "grounded_entities": 0,
        "predicted_entities": 0,
    }
    for index, record in enumerate(gold_records):
        messages = record.get("messages", [])
        gold_text = next((message.get("content", "") for message in messages if message.get("role") == "assistant"), "")
        gold = _graph(gold_text) or {"entities": [], "relations": []}
        predicted = _graph(predictions[index]) if index < len(predictions) else None
        if predicted is None:
            continue
        totals["json_valid"] += 1
        entity_metrics = _prf(_entity_set(predicted), _entity_set(gold))
        relation_metrics = _prf(_relation_set(predicted), _relation_set(gold))
        for key in ("tp", "fp", "fn"):
            totals["entity"][key] += entity_metrics[key]
            totals["relation"][key] += relation_metrics[key]
        predicted_relations = _relation_set(predicted)
        gold_relations = _relation_set(gold)
        predicted_types = {item[1] for item in predicted_relations}
        gold_types = {item[1] for item in gold_relations}
        predicted_sources = {item[0] for item in predicted_relations}
        gold_sources = {item[0] for item in gold_relations}
        predicted_targets = {item[2] for item in predicted_relations}
        gold_targets = {item[2] for item in gold_relations}
        for metric, predicted_set, gold_set in (
            ("relation_type", predicted_types, gold_types),
            ("source_endpoint", predicted_sources, gold_sources),
            ("target_endpoint", predicted_targets, gold_targets),
        ):
            values = _prf(predicted_set, gold_set)
            for key in ("tp", "fp", "fn"):
                totals[metric][key] += values[key]

        raw_user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        document = router.route(raw_user)
        source_text = normalize_text(document.raw)
        for node in predicted["entities"]:
            if not isinstance(node, dict):
                continue
            name = normalize_text(_node_name(node))
            totals["predicted_entities"] += 1
            if name and name in source_text:
                totals["grounded_entities"] += 1

    def summarize(counter: dict[str, int]) -> dict[str, float | int]:
        tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**counter, "precision": precision, "recall": recall, "f1": f1}

    result = {
        "samples": totals["samples"],
        "json_closure_rate": totals["json_valid"] / totals["samples"] if totals["samples"] else 0.0,
        "entity": summarize(totals["entity"]),
        "relation": summarize(totals["relation"]),
        "relation_type": summarize(totals["relation_type"]),
        "source_endpoint": summarize(totals["source_endpoint"]),
        "target_endpoint": summarize(totals["target_endpoint"]),
        "grounding_rate": totals["grounded_entities"] / totals["predicted_entities"] if totals["predicted_entities"] else 0.0,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

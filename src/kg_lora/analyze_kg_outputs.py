#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_relations(output: dict):
    relations = output.get("relations")
    if isinstance(relations, list):
        return relations
    relationships = output.get("relationships")
    if isinstance(relationships, list):
        return relationships
    return []


def analyze_records(records):
    entity_counts = []
    relation_counts = []
    input_lengths = []
    cot_lengths = []
    empty_property_values = 0
    total_property_values = 0
    chunk_counts = []
    successful_chunk_counts = []

    for item in records:
        output = item.get("output") or {}
        entities = output.get("entities") or []
        relations = get_relations(output)
        entity_counts.append(len(entities))
        relation_counts.append(len(relations))
        input_lengths.append(len(item.get("input_used") or item.get("input") or ""))
        cot_lengths.append(len((item.get("cot") or "").strip()))

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            for value in (entity.get("properties") or {}).values():
                total_property_values += 1
                if not str(value).strip():
                    empty_property_values += 1

        if "chunk_count" in item:
            chunk_counts.append(item.get("chunk_count", 0))
            successful_chunk_counts.append(item.get("successful_chunks", 0))

    def stats(values):
        if not values:
            return None
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
            "mean": round(statistics.mean(values), 2),
        }

    return {
        "count": len(records),
        "entity_count": stats(entity_counts),
        "relation_count": stats(relation_counts),
        "input_length": stats(input_lengths),
        "cot_length": stats(cot_lengths),
        "empty_property_ratio": round(empty_property_values / total_property_values, 4) if total_property_values else None,
        "chunk_count": stats(chunk_counts),
        "successful_chunk_count": stats(successful_chunk_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze KG extraction output quality.")
    parser.add_argument("--input", required=True, help="Path to cot_filtered/cot_raw json.")
    args = parser.parse_args()

    records = load_json(Path(args.input))
    if not isinstance(records, list):
        raise SystemExit("Input JSON must be a list.")

    summary = analyze_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

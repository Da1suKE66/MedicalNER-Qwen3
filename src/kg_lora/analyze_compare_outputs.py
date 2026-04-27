#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_json_maybe(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def get_relations(payload):
    if not isinstance(payload, dict):
        return []
    relations = payload.get("relations")
    if isinstance(relations, list):
        return relations
    relationships = payload.get("relationships")
    if isinstance(relationships, list):
        return relationships
    return []


def stats(values):
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize compare_qwen_outputs results.")
    parser.add_argument("--input", required=True, help="Path to comparison json.")
    args = parser.parse_args()

    records = load_json(Path(args.input))
    if not isinstance(records, list):
        raise SystemExit("Input JSON must be a list.")

    if records and "output" in records[0]:
        model_name = records[0].get("model") or "unknown_model"
        valid_cases = 0
        entity_counts = []
        relation_counts = []
        invalid_cases = []

        for item in records:
            payload = parse_json_maybe(item.get("output"))
            if payload is None:
                invalid_cases.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                    }
                )
                continue

            valid_cases += 1
            entities = payload.get("entities")
            entity_counts.append(len(entities) if isinstance(entities, list) else 0)
            relation_counts.append(len(get_relations(payload)))

        summary = {
            model_name: {
                "total_cases": len(records),
                "valid_json_cases": valid_cases,
                "valid_json_ratio": round(valid_cases / len(records), 4) if records else None,
                "entity_count": stats(entity_counts),
                "relation_count": stats(relation_counts),
                "invalid_cases": invalid_cases,
            }
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summary = {}
    model_names = set()
    for item in records:
        outputs = item.get("outputs") or {}
        model_names.update(outputs.keys())

    for model_name in sorted(model_names):
        valid_cases = 0
        entity_counts = []
        relation_counts = []
        invalid_cases = []

        for item in records:
            payload = parse_json_maybe((item.get("outputs") or {}).get(model_name))
            if payload is None:
                invalid_cases.append(
                    {
                        "id": item.get("id"),
                        "title": item.get("title"),
                    }
                )
                continue

            valid_cases += 1
            entities = payload.get("entities")
            entity_counts.append(len(entities) if isinstance(entities, list) else 0)
            relation_counts.append(len(get_relations(payload)))

        summary[model_name] = {
            "total_cases": len(records),
            "valid_json_cases": valid_cases,
            "valid_json_ratio": round(valid_cases / len(records), 4) if records else None,
            "entity_count": stats(entity_counts),
            "relation_count": stats(relation_counts),
            "invalid_cases": invalid_cases,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

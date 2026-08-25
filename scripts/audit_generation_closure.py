#!/usr/bin/env python3
"""Audit JSON closure and hard generation cutoffs in comparison artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.I | re.S)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.S)
SPECIAL_RE = re.compile(r"<\|(?:im_end|endoftext|eot_id|end_of_text)\|>", re.I)


def parse_output(value: Any) -> tuple[dict[str, Any] | None, str, dict[str, int]]:
    if not isinstance(value, dict):
        return None, "missing", {"open_braces": 0, "close_braces": 0}
    raw = str(value.get("raw") or "")
    text = str(value.get("output") or raw).strip()
    match = OUTPUT_RE.search(text)
    if match:
        text = match.group(1).strip()
    elif re.search(r"<output>", text, flags=re.I):
        # Qwen3 occasionally emits the opening wrapper but omits only the
        # closing marker.  Strip the wrapper so a valid JSON body is not
        # misclassified as invalid JSON; the raw wrapper defect remains in
        # the report through the original string and brace/token metadata.
        text = re.split(r"<output>", text, maxsplit=1, flags=re.I)[1].strip()
    text = SPECIAL_RE.sub("", FENCE_RE.sub("", text)).strip()
    braces = {"open_braces": text.count("{"), "close_braces": text.count("}")}
    try:
        graph = json.loads(text)
    except Exception:
        return None, "invalid_json", braces
    if not isinstance(graph, dict):
        return None, "non_object", braces
    if not isinstance(graph.get("entities"), list) or not isinstance(graph.get("relations"), list):
        return None, "invalid_graph_shape", braces
    return graph, "ok", braces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise SystemExit("artifact must contain a cases list")

    rows: list[dict[str, Any]] = []
    aggregate = Counter()
    for case in cases:
        if not isinstance(case, dict):
            continue
        model_fields = [
            key
            for key, value in case.items()
            if key not in {"id", "split", "system", "user", "deepseek_original", "generation_meta"}
            and isinstance(value, dict)
        ]
        for field in ["deepseek_original", *model_fields]:
            value = case.get(field)
            if not isinstance(value, dict):
                continue
            graph, status, braces = parse_output(value)
            meta = (case.get("generation_meta") or {}).get(field, {})
            hit_max = bool(meta.get("hit_max_new_tokens"))
            generated = meta.get("generated_tokens")
            row = {
                "id": case.get("id"),
                "split": case.get("split"),
                "model_field": field,
                "parse_status": status,
                "generated_tokens": generated,
                "max_new_tokens": meta.get("max_new_tokens"),
                "hit_max_new_tokens": hit_max,
                **braces,
                "entity_count": len(graph.get("entities", [])) if graph else 0,
                "relation_count": len(graph.get("relations", [])) if graph else 0,
            }
            rows.append(row)
            aggregate[(field, status)] += 1
            aggregate[(field, "hit_max_new_tokens")] += int(hit_max)
            aggregate[(field, "brace_mismatch")] += int(braces["open_braces"] != braces["close_braces"])

    by_model: dict[str, dict[str, Any]] = {}
    fields = sorted({str(row["model_field"]) for row in rows})
    for field in fields:
        subset = [row for row in rows if row["model_field"] == field]
        by_model[field] = {
            "count": len(subset),
            "status_counts": dict(Counter(row["parse_status"] for row in subset)),
            "hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in subset),
            "brace_mismatch": sum(row["open_braces"] != row["close_braces"] for row in subset),
            "max_generated_tokens": max(
                (row["generated_tokens"] or 0 for row in subset), default=0
            ),
            "invalid_or_truncated": [
                row for row in subset if row["parse_status"] != "ok" or row["hit_max_new_tokens"]
            ],
        }

    report = {
        "metadata": {"artifact": str(args.artifact), "case_count": len(cases)},
        "by_model": by_model,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

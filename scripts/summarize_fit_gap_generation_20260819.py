#!/usr/bin/env python3
"""Summarize matched train/heldout generation results by split.

This is a diagnostic report, not a medical gold-standard evaluation: the
DeepSeek output is used only as the frozen reference target for comparing
identical records under the same decoding setup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_comparison_20260811 import (
    f1,
    graph_content_canonical,
    graph_sets,
    grounding_counts,
    parse_json_block,
    schema_violations,
    source_fields_for_chunk,
    source_text_for_case,
)


def summarize(cases: list[dict[str, Any]], model: str, chunks: dict[int, dict[str, Any]]) -> dict[str, Any]:
    s = Counter()
    generated: list[int] = []
    prompt: list[int] = []
    per_case: list[dict[str, Any]] = []
    for case in cases:
        target, target_status = parse_json_block(case.get("deepseek_original"))
        pred, pred_status = parse_json_block(case.get(model))
        s[f"target_parse_{target_status}"] += 1
        s[f"pred_parse_{pred_status}"] += 1
        s["schema_valid"] += bool(pred is not None and not schema_violations(pred))
        s["content_exact"] += graph_content_canonical(target) == graph_content_canonical(pred)
        te, tr = graph_sets(target)
        pe, pr = graph_sets(pred)
        s["entity_tp"] += len(te & pe)
        s["entity_fp"] += len(pe - te)
        s["entity_fn"] += len(te - pe)
        s["relation_tp"] += len(tr & pr)
        s["relation_fp"] += len(pr - tr)
        s["relation_fn"] += len(tr - pr)
        source = source_text_for_case(case, chunks.get(int(case["id"])))
        gt, gv, _ = grounding_counts(pred, source)
        s["grounding_items"] += gt
        s["grounding_violations"] += gv
        meta = (case.get("generation_meta") or {}).get(model) or {}
        if isinstance(meta.get("generated_tokens"), int):
            generated.append(meta["generated_tokens"])
        if isinstance(meta.get("prompt_tokens"), int):
            prompt.append(meta["prompt_tokens"])
        if meta.get("hit_max_new_tokens"):
            s["hit_max_new_tokens"] += 1
        per_case.append(
            {
                "id": case.get("id"),
                "split": case.get("split"),
                "source_fields": source_fields_for_chunk(chunks.get(int(case["id"]))),
                "target_parse": target_status,
                "prediction_parse": pred_status,
                "schema_valid": bool(pred is not None and not schema_violations(pred)),
                "content_exact": graph_content_canonical(target) == graph_content_canonical(pred),
                "target_entities": len(te),
                "predicted_entities": len(pe),
                "target_relations": len(tr),
                "predicted_relations": len(pr),
                "generation_meta": meta,
            }
        )

    def block(prefix: str) -> dict[str, Any]:
        tp, fp, fn = (s[f"{prefix}_{x}"] for x in ("tp", "fp", "fn"))
        total_pred, total_target = tp + fp, tp + fn
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_total": total_pred,
            "target_total": total_target,
            "precision": tp / total_pred if total_pred else 0.0,
            "recall": tp / total_target if total_target else 0.0,
            "f1": f1(tp, fp, fn),
            "hallucination_rate": fp / total_pred if total_pred else 0.0,
            "under_extraction_rate": fn / total_target if total_target else 0.0,
        }

    return {
        "cases": len(cases),
        "ids": [c.get("id") for c in cases],
        "prediction_parse": {k: v for k, v in s.items() if k.startswith("pred_parse_")},
        "schema_valid_cases": s["schema_valid"],
        "content_exact_cases": s["content_exact"],
        "entity": block("entity"),
        "relation": block("relation"),
        "grounding": {
            "items": s["grounding_items"],
            "violations": s["grounding_violations"],
            "violation_rate": s["grounding_violations"] / s["grounding_items"] if s["grounding_items"] else 0.0,
        },
        "token_usage": {
            "prompt_min": min(prompt) if prompt else None,
            "prompt_max": max(prompt) if prompt else None,
            "generated_min": min(generated) if generated else None,
            "generated_max": max(generated) if generated else None,
            "generated_mean": sum(generated) / len(generated) if generated else None,
            "hit_max_new_tokens_cases": s["hit_max_new_tokens"],
        },
        "per_case": per_case,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--source-chunks", type=Path, required=True)
    ap.add_argument("--model", default="output_only")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    frozen = json.loads(args.source_chunks.read_text(encoding="utf-8"))
    chunks = {i: item for i, item in enumerate(frozen) if isinstance(item, dict)}
    cases = payload.get("cases") or []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_split.setdefault(str(case.get("split") or "unknown"), []).append(case)
    report = {
        "metadata": payload.get("metadata", {}),
        "model": args.model,
        "diagnostic_note": "DeepSeek target is a noisy reference; split gap is the primary fit diagnostic.",
        "by_split": {k: summarize(v, args.model, chunks) for k, v in sorted(by_split.items())},
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

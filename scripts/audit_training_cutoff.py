#!/usr/bin/env python3
"""Audit rendered Qwen3 chat-template lengths before a LLaMA-Factory run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def render(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> list[int]:
    kwargs = dict(tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking)
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--cutoff", type=int, default=16384)
    ap.add_argument("--near-cutoff-ratio", type=float, default=0.95)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    records: list[dict[str, Any]] = []
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in args.dataset:
        path = Path(dataset_name)
        rows = json.loads(path.read_text())
        lengths: list[int] = []
        over: list[dict[str, Any]] = []
        near: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            messages = row.get("conversations") or row.get("messages") or []
            ids = render(tokenizer, messages, args.enable_thinking)
            n = len(ids)
            item = {"index": idx, "id": row.get("id", idx), "tokens": n}
            lengths.append(n)
            records.append({"dataset": str(path), **item})
            if n > args.cutoff:
                over.append(item)
            if n >= int(args.cutoff * args.near_cutoff_ratio):
                near.append(item)
        by_dataset[str(path)] = {
            "count": len(lengths),
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": (sum(lengths) / len(lengths)) if lengths else 0,
            "p50": pct(lengths, 0.50),
            "p90": pct(lengths, 0.90),
            "p95": pct(lengths, 0.95),
            "p99": pct(lengths, 0.99),
            "over_cutoff": len(over),
            "near_cutoff": len(near),
            "over_examples": sorted(over, key=lambda x: x["tokens"], reverse=True)[:20],
        }
    lengths = [r["tokens"] for r in records]
    report = {
        "tokenizer": args.tokenizer,
        "cutoff": args.cutoff,
        "near_cutoff_ratio": args.near_cutoff_ratio,
        "enable_thinking": args.enable_thinking,
        "total": len(records),
        "over_cutoff": sum(r["tokens"] > args.cutoff for r in records),
        "near_cutoff": sum(r["tokens"] >= int(args.cutoff * args.near_cutoff_ratio) for r in records),
        "summary": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": (sum(lengths) / len(lengths)) if lengths else 0,
            "p50": pct(lengths, 0.50),
            "p90": pct(lengths, 0.90),
            "p95": pct(lengths, 0.95),
            "p99": pct(lengths, 0.99),
        },
        "by_dataset": by_dataset,
        "largest": sorted(records, key=lambda x: x["tokens"], reverse=True)[:50],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("total", "over_cutoff", "near_cutoff", "summary")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

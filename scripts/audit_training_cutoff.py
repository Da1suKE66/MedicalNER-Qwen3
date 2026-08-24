#!/usr/bin/env python3
"""Audit prompt/target token lengths before an SFT run.

This deliberately measures the exact ShareGPT messages that LLaMA-Factory will
read, including the compact output-only assistant target.  It does not change
the dataset.  The report makes truncation visible instead of relying on the
loss value alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    report: dict[str, Any] = {
        "tokenizer": args.tokenizer,
        "cutoff_len": args.cutoff_len,
        "splits": [],
    }
    for data_path in args.data:
        rows = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"expected list dataset: {data_path}")
        totals: list[int] = []
        prompts: list[int] = []
        targets: list[int] = []
        over: list[dict[str, Any]] = []
        invalid = 0
        for index, row in enumerate(rows):
            messages = row.get("messages") or row.get("conversations")
            if not isinstance(messages, list):
                invalid += 1
                continue
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            target_message = next((m for m in messages if m.get("role") == "assistant"), None)
            if target_message is None:
                invalid += 1
                continue
            # Chat-template tokens are the closest local reproduction of the
            # qwen3 template.  If a tokenizer lacks a chat template, retain a
            # conservative concatenated count rather than silently returning 0.
            try:
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except (TypeError, ValueError):
                prompt_text = "\n".join(str(m.get("content") or "") for m in prompt_messages)
            prompt_tokens = token_count(tokenizer, prompt_text)
            target_tokens = token_count(tokenizer, str(target_message.get("content") or ""))
            total = prompt_tokens + target_tokens
            prompts.append(prompt_tokens)
            targets.append(target_tokens)
            totals.append(total)
            if total > args.cutoff_len:
                over.append(
                    {
                        "index": index,
                        "prompt_tokens": prompt_tokens,
                        "target_tokens": target_tokens,
                        "total_tokens": total,
                        "overflow_tokens": total - args.cutoff_len,
                    }
                )
        split_report = {
            "data": str(data_path),
            "rows": len(rows),
            "invalid_rows": invalid,
            "prompt": {
                "min": min(prompts) if prompts else None,
                "median": statistics.median(prompts) if prompts else None,
                "p95": quantile(prompts, 0.95),
                "max": max(prompts) if prompts else None,
            },
            "target": {
                "min": min(targets) if targets else None,
                "median": statistics.median(targets) if targets else None,
                "p95": quantile(targets, 0.95),
                "max": max(targets) if targets else None,
            },
            "combined": {
                "min": min(totals) if totals else None,
                "median": statistics.median(totals) if totals else None,
                "p95": quantile(totals, 0.95),
                "max": max(totals) if totals else None,
            },
            "over_cutoff_count": len(over),
            "over_cutoff_fraction": (len(over) / len(totals)) if totals else 0.0,
            "longest_rows": sorted(over, key=lambda item: item["total_tokens"], reverse=True)[:50],
        }
        report["splits"].append(split_report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

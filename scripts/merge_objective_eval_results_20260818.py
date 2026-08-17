#!/usr/bin/env python3
"""Merge two objective-specific probe artifacts by deterministic record id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cases(path: Path) -> tuple[dict, dict[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = {int(case["id"]): case for case in payload.get("cases", [])}
    return payload, cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-only", type=Path, required=True)
    ap.add_argument("--priority", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    output_payload, output_cases = load_cases(args.output_only)
    priority_payload, priority_cases = load_cases(args.priority)
    base_payload, base_cases = (None, {})
    if args.base:
        base_payload, base_cases = load_cases(args.base)
    if set(output_cases) != set(priority_cases):
        raise SystemExit(
            "probe id mismatch: output-only=%s priority=%s"
            % (sorted(output_cases), sorted(priority_cases))
        )
    if args.base and set(base_cases) != set(output_cases):
        raise SystemExit(
            "probe id mismatch: output-only=%s base=%s"
            % (sorted(output_cases), sorted(base_cases))
        )

    merged_cases = []
    for case_id in sorted(output_cases):
        only = output_cases[case_id]
        priority = priority_cases[case_id]
        merged = {
            "id": case_id,
            "split": priority.get("split", only.get("split")),
            "system": priority.get("system", only.get("system")),
            "user": priority.get("user", only.get("user")),
            # The parsed JSON target is equivalent across the two converted
            # variants; keep the priority raw target so the original think
            # span remains available for audit.
            "deepseek_original": priority.get("deepseek_original", only.get("deepseek_original")),
            "base_qwen": (
                base_cases[case_id].get("base_qwen")
                if args.base
                else priority.get("base_qwen") or only.get("base_qwen")
            ),
            "output_priority": priority.get("output_priority"),
            "output_only": only.get("output_only"),
            "generation_meta": {
                **(only.get("generation_meta") or {}),
                **(priority.get("generation_meta") or {}),
            },
        }
        merged_cases.append(merged)

    metadata = {
        "merged_from": {
            "output_only": str(args.output_only),
            "priority": str(args.priority),
            "base": str(args.base) if args.base else None,
        },
        "selection": "same explicit probe ids from both objective evaluations",
        "generation": {
            "output_only": output_payload.get("metadata", {}).get("generation"),
            "priority": priority_payload.get("metadata", {}).get("generation"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "cases": merged_cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

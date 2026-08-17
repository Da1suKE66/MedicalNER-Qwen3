#!/usr/bin/env python3
"""Audit a frozen chunk snapshot and build record-disjoint SFT canary splits."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def register_dataset(dataset_info: dict, name: str, path: Path) -> None:
    dataset_info[name] = {
        "file_name": path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--converted", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--excluded-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--dataset-info", required=True, type=Path)
    parser.add_argument("--train-dataset-name", required=True)
    parser.add_argument("--eval-dataset-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    converted = json.loads(args.converted.read_text(encoding="utf-8"))
    if len(snapshot) != len(converted):
        raise SystemExit("snapshot and converted sample counts differ")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    token_lengths: list[int] = []
    eligible: list[tuple[dict, dict, int]] = []
    excluded: list[dict] = []
    label_counts: collections.Counter[str] = collections.Counter()
    relation_counts: collections.Counter[str] = collections.Counter()
    empty_relation_samples = 0

    for source, training_item in zip(snapshot, converted):
        messages = training_item["messages"]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        length = len(token_ids)
        token_lengths.append(length)
        output = source.get("output") or {}
        entities = output.get("entities") or []
        relations = output.get("relations") or []
        label_counts.update(entity.get("label") for entity in entities)
        relation_counts.update(relation.get("relation") for relation in relations)
        empty_relation_samples += int(not relations)
        if length > args.cutoff_len:
            excluded.append(
                {
                    "reason": "over_cutoff",
                    "token_length": length,
                    "provenance": source.get("snapshot_provenance"),
                }
            )
            continue
        eligible.append((source, training_item, length))

    groups: dict[int, list[tuple[dict, dict, int]]] = collections.defaultdict(list)
    for item in eligible:
        global_idx = int(item[0]["snapshot_provenance"]["global_idx"])
        groups[global_idx].append(item)
    group_ids = sorted(groups)
    random.Random(args.seed).shuffle(group_ids)

    eval_items: list[dict] = []
    eval_groups: set[int] = set()
    for group_id in group_ids:
        if len(eval_items) >= args.eval_samples:
            break
        eval_groups.add(group_id)
        remaining = args.eval_samples - len(eval_items)
        eval_items.extend(item[1] for item in groups[group_id][:remaining])

    train_items: list[dict] = []
    train_groups: set[int] = set()
    for group_id in group_ids:
        if group_id in eval_groups or len(train_items) >= args.train_samples:
            continue
        train_groups.add(group_id)
        remaining = args.train_samples - len(train_items)
        train_items.extend(item[1] for item in groups[group_id][:remaining])

    if not train_items or not eval_items:
        raise SystemExit("not enough eligible samples for record-disjoint splits")
    if train_groups & eval_groups:
        raise AssertionError("record leakage between train and evaluation splits")

    write_json(args.train_output, train_items)
    write_json(args.eval_output, eval_items)
    write_json(args.excluded_output, excluded)
    dataset_info = (
        json.loads(args.dataset_info.read_text(encoding="utf-8"))
        if args.dataset_info.exists()
        else {}
    )
    register_dataset(dataset_info, args.train_dataset_name, args.train_output)
    register_dataset(dataset_info, args.eval_dataset_name, args.eval_output)
    write_json(args.dataset_info, dataset_info)

    report = {
        "snapshot_samples": len(snapshot),
        "cutoff_len": args.cutoff_len,
        "eligible_samples": len(eligible),
        "excluded_over_cutoff": len(excluded),
        "token_lengths": {
            "min": min(token_lengths),
            "median": percentile(token_lengths, 0.5),
            "p90": percentile(token_lengths, 0.9),
            "p95": percentile(token_lengths, 0.95),
            "max": max(token_lengths),
        },
        "train_samples": len(train_items),
        "eval_samples": len(eval_items),
        "train_record_groups": len(train_groups),
        "eval_record_groups": len(eval_groups),
        "train_record_group_ids": sorted(train_groups),
        "eval_record_group_ids": sorted(eval_groups),
        "record_group_overlap": sorted(train_groups & eval_groups),
        "local_contract_repair_samples": sum(
            bool(item.get("local_contract_repair")) for item in snapshot
        ),
        "fallback_32768_samples": sum(
            item.get("max_output_tokens_used") == 32768 for item in snapshot
        ),
        "empty_relation_samples": empty_relation_samples,
        "entity_label_counts": dict(label_counts.most_common()),
        "relation_counts": dict(relation_counts.most_common()),
        "train_output": str(args.train_output),
        "eval_output": str(args.eval_output),
        "excluded_output": str(args.excluded_output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

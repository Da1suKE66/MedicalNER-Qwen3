#!/usr/bin/env python3
"""Matched train-vs-heldout fit-gap audit for the output-only objective.

The split is fixed by ``groupdisjoint_split_manifest.json``.  We first choose
same-field, similar-length train/heldout pairs, then optionally measure the
teacher-forced loss of several adapters on exactly those pairs.  A low train
loss with a much higher heldout loss is evidence of memorization/overfitting;
high loss on both sides points to capacity or supervision problems instead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


FIELD_ORDER = [
    "definition",
    "diagnosticCriteria",
    "exclusions",
    "synonyms",
    "ancestor",
    "descendant",
    "parent",
    "child",
    "codingNote",
    "title",
    "id",
]


def assistant_text(record: dict[str, Any]) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == "assistant")


def user_text(record: dict[str, Any]) -> str:
    return next(item["content"] for item in record["messages"] if item["role"] == "user")


def source_fields(chunk: dict[str, Any]) -> list[str]:
    values = chunk.get("snapshot_provenance", {}).get("fields", [])
    return sorted({str(item.get("source_field")) for item in values if item.get("source_field")})


def primary_field(chunk: dict[str, Any]) -> str:
    fields = set(source_fields(chunk))
    for name in FIELD_ORDER:
        if name in fields:
            return name
    return sorted(fields)[0] if fields else "unknown"


def match_score(train_record: dict[str, Any], dev_record: dict[str, Any]) -> float:
    train_target = len(assistant_text(train_record))
    dev_target = len(assistant_text(dev_record))
    train_input = len(user_text(train_record))
    dev_input = len(user_text(dev_record))
    # Log-length distance keeps one very long record from dominating; input
    # length is a small tie breaker for similar target lengths.
    return abs(math.log1p(train_target) - math.log1p(dev_target)) + 0.2 * abs(
        math.log1p(train_input) - math.log1p(dev_input)
    )


def choose_pairs(
    records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    train_indices: list[int],
    dev_indices: list[int],
    max_pairs: int,
) -> list[dict[str, Any]]:
    train_by_field: dict[str, list[int]] = {}
    dev_by_field: dict[str, list[int]] = {}
    for idx in train_indices:
        train_by_field.setdefault(primary_field(chunks[idx]), []).append(idx)
    for idx in dev_indices:
        dev_by_field.setdefault(primary_field(chunks[idx]), []).append(idx)

    candidates: list[tuple[float, str, int, int]] = []
    for field in sorted(set(train_by_field) & set(dev_by_field)):
        for dev_idx in dev_by_field[field]:
            for train_idx in train_by_field[field]:
                candidates.append((match_score(records[train_idx], records[dev_idx]), field, train_idx, dev_idx))
    candidates.sort()

    selected: list[dict[str, Any]] = []
    used_train: set[int] = set()
    used_dev: set[int] = set()
    used_fields: set[str] = set()

    # First pass: one pair per source field, which prevents the audit from
    # being dominated by the most common diagnosticCriteria chunks.
    for score, field, train_idx, dev_idx in candidates:
        if field in used_fields or train_idx in used_train or dev_idx in used_dev:
            continue
        selected.append({"field": field, "train_index": train_idx, "heldout_index": dev_idx, "match_score": score})
        used_train.add(train_idx)
        used_dev.add(dev_idx)
        used_fields.add(field)
        if len(selected) >= max_pairs:
            return selected

    # Fill remaining slots with the closest unused pairs.
    for score, field, train_idx, dev_idx in candidates:
        if train_idx in used_train or dev_idx in used_dev:
            continue
        selected.append({"field": field, "train_index": train_idx, "heldout_index": dev_idx, "match_score": score})
        used_train.add(train_idx)
        used_dev.add(dev_idx)
        if len(selected) >= max_pairs:
            break
    return selected


def make_texts(tokenizer, record: dict[str, Any]) -> tuple[str, str]:
    messages = record["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    return prompt, full


def load_model(base_model: str, adapter: str | None):
    kwargs = {"trust_remote_code": True, "device_map": "cuda:0", "torch_dtype": torch.bfloat16}
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, attn_implementation="flash_attention_2", **kwargs)
    except (ImportError, RuntimeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def case_loss(model, tokenizer, record: dict[str, Any], max_seq_len: int) -> dict[str, Any]:
    prompt, full = make_texts(tokenizer, record)
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if full_ids.shape[1] > max_seq_len:
        return {"loss": None, "prompt_tokens": int(prompt_ids.shape[1]), "target_tokens": int(full_ids.shape[1] - prompt_ids.shape[1]), "skipped": "over_max_seq_len"}
    input_ids = full_ids.to("cuda:0")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    prompt_len = min(int(prompt_ids.shape[1]), labels.shape[1])
    labels[:, :prompt_len] = -100
    with torch.inference_mode():
        result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return {
        "loss": float(result.loss.detach().cpu()),
        "prompt_tokens": prompt_len,
        "target_tokens": int((labels != -100).sum().item()),
        "skipped": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--output-only-adapter")
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    args = ap.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_indices = sorted(int(item) for item in manifest["train_indices"])
    dev_indices = sorted(int(item) for item in manifest["dev_indices"])
    pairs = choose_pairs(records, chunks, train_indices, dev_indices, args.max_pairs)
    selected = sorted({item["train_index"] for item in pairs} | {item["heldout_index"] for item in pairs})
    result: dict[str, Any] = {
        "metadata": {
            "data": args.data,
            "split_manifest": args.split_manifest,
            "train_records": len(train_indices),
            "heldout_records": len(dev_indices),
            "max_pairs": args.max_pairs,
            "selection_rule": "same primary source field and nearest input/target character length",
            "selected_indices": selected,
        },
        "pairs": pairs,
        "losses": {},
    }
    if args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        specs: list[tuple[str, str | None]] = [("base_qwen", None)]
        if args.output_only_adapter:
            specs.append(("output_only_best", args.output_only_adapter))
        for checkpoint in args.checkpoints:
            specs.append((Path(checkpoint).name, checkpoint))
        for name, adapter in specs:
            print(f"loading {name}", flush=True)
            model = load_model(args.base_model, adapter)
            entries = []
            for pair in pairs:
                for split, idx in (("train", pair["train_index"]), ("heldout", pair["heldout_index"])):
                    item = case_loss(model, tokenizer, records[idx], args.max_seq_len)
                    entries.append({"split": split, "index": idx, "field": pair["field"], **item})
                    print(f"{name} {split}:{idx} loss={item['loss']}", flush=True)
            train_losses = [item["loss"] for item in entries if item["split"] == "train" and item["loss"] is not None]
            heldout_losses = [item["loss"] for item in entries if item["split"] == "heldout" and item["loss"] is not None]
            result["losses"][name] = {
                "entries": entries,
                "train_mean": sum(train_losses) / len(train_losses) if train_losses else None,
                "heldout_mean": sum(heldout_losses) / len(heldout_losses) if heldout_losses else None,
                "gap_heldout_minus_train": (sum(heldout_losses) / len(heldout_losses) - sum(train_losses) / len(train_losses)) if train_losses and heldout_losses else None,
                "ratio_heldout_over_train": (sum(heldout_losses) / len(heldout_losses)) / (sum(train_losses) / len(train_losses)) if train_losses and heldout_losses and sum(train_losses) else None,
            }
            del model
            torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

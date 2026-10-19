#!/usr/bin/env python3
"""Replay legacy stage SFT; current relation data uses generative.train."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def validate_legacy_training_rows(rows: list[dict[str, Any]], stage: str) -> None:
    """Keep raw-prompt replay separate from the current relation contract.

    Historical stage exports have a stage tag but no split tag. Preserve those
    exports; an explicit non-training split is never admissible. New relation
    formats need their corpus supervision gate and shared chat renderer.
    """
    from generative.supervision_guard import is_partial_row

    if not rows:
        raise ValueError("Empty legacy training data")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Legacy training rows must be JSON objects")
        if is_partial_row(row) or any(key in row for key in ("pairs", "target", "output_contract")):
            raise ValueError(
                "Current/partial relation data must use generative.train, which "
                "enforces the supervision gate and shared non-thinking chat template"
            )
        if row.get("stage") != stage:
            raise ValueError(
                f"Require legacy rows tagged stage={stage!r}; current relation "
                "data must use generative.train"
            )
        if "split" in row and row["split"] != "train":
            raise ValueError("Only split='train' is allowed; held-out rows cannot enter SFT")
        for key in ("prompt", "completion"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing/non-string legacy {key}")
        if row["prompt"].startswith("<|im_start|>"):
            raise ValueError("Legacy replay requires raw prompts; use generative.train for chat-rendered training")


def validate_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output must be new/empty; never overwrite old checkpoints")


class JsonlSFTDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int):
        import torch

        self.torch = torch
        self.items: list[dict[str, Any]] = []
        for row in rows:
            prompt = str(row.get("prompt") or "")
            completion = str(row.get("completion") or "")
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            partial = (row.get("data_version", "").startswith("teacher-silver-repair-v2")
                       or any(key in row for key in ("loss_mask_contract", "loss_mask_spans", "pair_supervision")))
            if partial:
                from data_repair.supervision_v2 import masked_token_labels

                completion_ids, completion_labels = masked_token_labels(row, tokenizer)
            else:
                completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
                completion_labels = list(completion_ids)
            if tokenizer.eos_token_id is not None:
                completion_ids = completion_ids + [tokenizer.eos_token_id]
                completion_labels = completion_labels + [tokenizer.eos_token_id]
            if len(prompt_ids) + len(completion_ids) > max_seq_length:
                raise ValueError(f"Record {row.get('id')} exceeds max_seq_length: {len(prompt_ids)} input + {len(completion_ids)} target tokens. Rechunk the source; silently removing the entity table is forbidden.")
            input_ids = prompt_ids + completion_ids
            input_ids = input_ids[:max_seq_length]
            prompt_length = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_length + completion_labels
            labels = labels[:max_seq_length]
            item = {"input_ids": input_ids, "labels": labels}
            if partial:
                item["_supervision_counts"] = {
                    "prompt_tokens": prompt_length,
                    "completion_tokens": len(completion_ids),
                    "masked_completion_tokens": sum(label == -100 for label in completion_labels),
                    "supervised_pairs": sum(not status.startswith("UNKNOWN_") for status in row["pair_supervision"].values()),
                    "unknown_pairs": sum(status.startswith("UNKNOWN_") for status in row["pair_supervision"].values()),
                }
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(len(item["input_ids"]) for item in items)
        pad_id = self.tokenizer.pad_token_id
        input_ids = []
        attention_mask = []
        labels = []
        for item in items:
            pad = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("entities", "relations"), required=True)
    parser.add_argument("--data", required=True, help="Stage JSONL generated by build_stage_data.py")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fail on data/overwrite errors before importing the GPU runtime or loading
    # a base model. The generic dataset remains shared by generative.train.
    rows = _read_jsonl(args.data, args.limit)
    validate_legacy_training_rows(rows, args.stage)
    out = Path(args.output_dir)
    validate_output_directory(out)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = JsonlSFTDataset(rows, tokenizer, args.max_seq_length)
    validate_output_directory(out)
    out.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    manifest = {
        "stage": args.stage,
        "base_model": args.base_model,
        "records": len(dataset),
        "max_seq_length": args.max_seq_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "prompt_template": "legacy raw prompt replay, paired with kg_agent.pipeline.TransformersBackend",
        "split_validation": "explicit tags must be train; untagged historical stage exports retained, source disjointness not certified",
    }
    (out / "training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

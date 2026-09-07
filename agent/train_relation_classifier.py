"""Qwen3 LoRA multi-label sequence classifier. No autoregressive relation output."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

from pair_metrics import select_threshold


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def encode_rows(rows, tokenizer, labels, max_length):
    encoded = []
    excluded = []
    for i, row in enumerate(rows):
        ids = tokenizer(row["prompt"], add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
        if len(ids) > max_length:
            excluded.append(
                {
                    "index": i,
                    "id": row["id"],
                    "tokens": len(ids),
                    "positive": bool(row.get("labels")),
                }
            )
            continue
        encoded.append(
            {
                "index": i,
                "input_ids": ids,
                "labels": [float(l in row.get("labels", [])) for l in labels],
                "allowed_mask": [float(l in row["allowed"]) for l in labels],
            }
        )
    return encoded, excluded


def collate(rows, pad_id):
    import torch

    length = max(len(r["input_ids"]) for r in rows)
    return {
        "input_ids": torch.tensor(
            [r["input_ids"] + [pad_id] * (length - len(r["input_ids"])) for r in rows]
        ),
        "attention_mask": torch.tensor(
            [
                [1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"]))
                for r in rows
            ]
        ),
        "labels": torch.tensor([r["labels"] for r in rows], dtype=torch.float32),
        "allowed_mask": torch.tensor(
            [r["allowed_mask"] for r in rows], dtype=torch.float32
        ),
    }


def predict(model, tokenizer, rows, labels, max_length=768, batch_size=8):
    import torch

    encoded, excluded = encode_rows(rows, tokenizer, labels, max_length)
    probabilities = [[0.0] * len(labels) for _ in rows]
    model.eval()
    encoded.sort(key=lambda r: len(r["input_ids"]))
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            selected = encoded[start : start + batch_size]
            batch = collate(selected, tokenizer.pad_token_id)
            logits = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
            ).logits
            probs = torch.sigmoid(logits.float()).cpu().tolist()
            for r, ps in zip(selected, probs):
                probabilities[r["index"]] = ps
    return probabilities, excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--positive-weight", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=0)
    args = p.parse_args()
    import torch
    import transformers, peft
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = Path(args.data_dir)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory; existing runs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "manifest.json").read_text())
    labels = meta["labels"]
    train = load_rows(data / "train.jsonl")
    tune = load_rows(data / "tune.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded, excluded = encode_rows(train, tokenizer, labels, args.max_length)
    if not encoded:
        raise ValueError("No usable training pairs")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["score"],
        ),
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.to("cuda")
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    batches = math.ceil(len(encoded) / args.batch_size)
    updates_per_epoch = math.ceil(batches / args.gradient_accumulation)
    total = args.epochs * updates_per_epoch
    warmup = max(1, int(0.05 * total))
    step = 0
    best = None
    history = []
    started = time.time()
    manifest = {
        "args": vars(args),
        "labels": labels,
        "data_manifest": meta,
        "data_files_sha256": {
            f: hashlib.sha256((data / f).read_bytes()).hexdigest()
            for f in ["train.jsonl", "tune.jsonl", "manifest.json"]
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "excluded_train": excluded,
        "used_train_pairs": len(encoded),
        "total_steps": total,
        "loss": "per-pair mean binary cross entropy on allowed relation labels, positive weight recorded in args",
        "selection": "tuning groups only: maximize F0.5 among thresholds with precision >= .5; fallback maximize F0.5 if none qualify",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "start",
                "pairs": len(encoded),
                "excluded": len(excluded),
                "total_steps": total,
            }
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        seen = 0
        for bidx, start in enumerate(range(0, len(order), args.batch_size)):
            selected = [encoded[i] for i in order[start : start + args.batch_size]]
            batch = {
                k: v.to("cuda")
                for k, v in collate(selected, tokenizer.pad_token_id).items()
            }
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.float()
            losses = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                batch["labels"],
                pos_weight=torch.tensor(args.positive_weight, device="cuda"),
                reduction="none",
            )
            mask = batch["allowed_mask"]
            loss = ((losses * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
            # Scale the final partial accumulation by its actual number of batches.
            block_start = (
                bidx // args.gradient_accumulation
            ) * args.gradient_accumulation
            accumulation = min(args.gradient_accumulation, batches - block_start)
            (loss / accumulation).backward()
            epoch_loss += loss.item() * len(selected)
            seen += len(selected)
            if (bidx + 1) % args.gradient_accumulation == 0 or bidx + 1 == batches:
                step += 1
                factor = (
                    step / warmup
                    if step <= warmup
                    else 0.5
                    * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
                )
                for g in optimizer.param_groups:
                    g["lr"] = args.learning_rate * factor
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if step % 25 == 0 or step == 1:
                    log = {
                        "event": "train",
                        "epoch": epoch,
                        "step": step,
                        "loss": epoch_loss / seen,
                        "lr": optimizer.param_groups[0]["lr"],
                        "seconds": time.time() - started,
                    }
                    history.append(log)
                    print(json.dumps(log), flush=True)
                    (out / "training_history.json").write_text(
                        json.dumps(history, indent=2) + "\n"
                    )
            if args.max_steps and step >= args.max_steps:
                break
        probs, excluded_tune = predict(
            model, tokenizer, tune, labels, args.max_length, 8
        )
        selected, sweep = select_threshold(tune, probs, labels)
        log = {
            "event": "tune",
            "epoch": epoch,
            "step": step,
            "train_loss": epoch_loss / seen,
            "selected": selected,
            "sweep": sweep,
            "excluded_tune": excluded_tune,
            "seconds": time.time() - started,
        }
        history.append(log)
        print(
            json.dumps(
                {
                    "event": "tune",
                    "epoch": epoch,
                    **{k: v for k, v in selected.items() if k != "per_class"},
                }
            ),
            flush=True,
        )
        (out / f"tune_epoch{epoch}.json").write_text(json.dumps(log, indent=2) + "\n")
        value = (
            selected["precision"] >= 0.5 and selected["tp"] > 0,
            selected["f0_5"],
            selected["precision"],
        )
        if best is None or value > best:
            best = value
            checkpoint = out / f"epoch{epoch}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            selection = {
                "checkpoint": str(checkpoint),
                "epoch": epoch,
                "threshold": selected["threshold"],
                "labels": labels,
                "max_length": args.max_length,
                "metrics": selected,
                "selection_split": "train-group-held-out tuning set",
            }
            (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        if args.max_steps and step >= args.max_steps:
            break
    print(
        json.dumps(
            {
                "event": "complete",
                "seconds": time.time() - started,
                "selection": json.loads((out / "selection.json").read_text()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

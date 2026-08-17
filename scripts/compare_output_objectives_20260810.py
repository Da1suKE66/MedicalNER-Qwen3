#!/usr/bin/env python3
"""Compare DeepSeek targets, base Qwen3, and the two objective adapters."""

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_target(text: str) -> dict[str, str | None]:
    think = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    output = re.search(r"<output>(.*?)</output>", text, flags=re.S)
    if output is None and think is None:
        # With enable_thinking=false Qwen emits the final JSON directly.
        final_output = text.strip()
    else:
        final_output = output.group(1).strip() if output else None
    return {
        "raw": text,
        "think": think.group(1).strip() if think else None,
        "output": final_output,
    }


def pick_three(ids: list[int]) -> list[int]:
    if len(ids) < 3:
        raise ValueError("Need at least three ids in each split")
    return [ids[0], ids[len(ids) // 2], ids[-1]]


def make_prompt(tokenizer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_bundle(base_model: str, adapter: str | None):
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "cuda:0",
        "quantization_config": bnb,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, attn_implementation="flash_attention_2", **model_kwargs
        )
    except (ImportError, ValueError, RuntimeError):
        # Keep the comparison runnable on nodes without flash-attn; the remote
        # stable node has flash-attn 2.8.3 and takes this fast path.
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, system: str, user: str, max_new_tokens: int) -> str:
    prompt = make_prompt(tokenizer, system, user)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=False).strip()


def generate_batch(
    tokenizer,
    model,
    cases: list[dict],
    max_new_tokens: int,
) -> list[str]:
    """Generate a small left-padded batch without changing greedy semantics."""
    prompts = [make_prompt(tokenizer, case["system"], case["user"]) for case in cases]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda:0")
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = ids[:, input_width:]
    return [tokenizer.decode(row, skip_special_tokens=False).strip() for row in new_ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--priority-adapter", required=True)
    ap.add_argument("--output-only-adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="limit selected cases for a canary")
    ap.add_argument("--indices", default=None, help="comma-separated positional record indices to select")
    ap.add_argument(
        "--only-model",
        choices=("base_qwen", "output_priority", "output_only"),
        default=None,
        help="run only one model; useful for parallelizing the three model passes",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--selection",
        choices=("six", "all_heldout"),
        default="six",
        help="six auditable probes (default) or every record in the deterministic held-out split",
    )
    args = ap.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    split = Dataset.from_dict({"idx": list(range(len(records)))})
    split = split.train_test_split(test_size=0.1, seed=args.seed)
    train_ids = sorted(split["train"]["idx"])
    eval_ids = sorted(split["test"]["idx"])
    if args.selection == "six":
        selected = [("train", i) for i in pick_three(train_ids)] + [
            ("heldout_eval", i) for i in pick_three(eval_ids)
        ]
    else:
        selected = [("heldout_eval", i) for i in eval_ids]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
    if args.indices:
        wanted = {int(value.strip()) for value in args.indices.split(",") if value.strip()}
        selected = [item for item in selected if item[1] in wanted]

    cases = []
    for split_name, idx in selected:
        messages = records[idx]["messages"]
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        target = next(m["content"] for m in messages if m["role"] == "assistant")
        cases.append(
            {
                "id": idx,
                "split": split_name,
                "system": system,
                "user": user,
                "deepseek_original": parse_target(target),
                "base_qwen": None,
                "output_priority": None,
                "output_only": None,
            }
        )

    model_specs = [
        ("base_qwen", None),
        ("output_priority", args.priority_adapter),
        ("output_only", args.output_only_adapter),
    ]
    if args.only_model:
        model_specs = [item for item in model_specs if item[0] == args.only_model]
    for name, adapter in model_specs:
        print(f"Loading {name}...", flush=True)
        tokenizer, model = load_bundle(args.base_model, adapter)
        try:
            batch_size = max(1, args.batch_size)
            for start in range(0, len(cases), batch_size):
                batch = cases[start : start + batch_size]
                raws = (
                    generate_batch(tokenizer, model, batch, args.max_new_tokens)
                    if len(batch) > 1
                    else [generate(tokenizer, model, batch[0]["system"], batch[0]["user"], args.max_new_tokens)]
                )
                for case, raw in zip(batch, raws):
                    case[name] = parse_target(raw)
                    print(f"{name} case={case['split']}:{case['id']} chars={len(raw)}", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "metadata": {
            "data_records": len(records),
            "split_seed": args.seed,
            "train_records": len(train_ids),
            "heldout_eval_records": len(eval_ids),
            "selection": "three evenly spaced ids from each deterministic LLaMA-Factory split",
            "generation": "greedy, enable_thinking=false, max_new_tokens=%d, batch_size=%d"
            % (args.max_new_tokens, max(1, args.batch_size)),
            "model_only": args.only_model,
        },
        "cases": cases,
    }
    if args.selection == "all_heldout":
        payload["metadata"]["selection"] = "all records from the deterministic LLaMA-Factory held-out split"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()

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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StructuredCompletionStoppingCriteria(StoppingCriteria):
    """Stop once a complete output object or output wrapper is emitted.

    Some checkpoints continue producing repeated JSON-like text after the
    first complete object and never emit EOS.  This diagnostic stopper is
    opt-in so the ordinary comparison remains an untouched greedy baseline.
    It decodes only on likely closing-token steps, then uses the same target
    parser boundary as evaluation (``</output>`` or a complete JSON value).
    """

    def __init__(self, tokenizer, prompt_offsets, enable_thinking=False):
        self.tokenizer = tokenizer
        # ``input_ids`` is left-padded for batches, so the slice must start at
        # the common padded width rather than each row's unpadded length.
        self.prompt_offsets = list(prompt_offsets)
        self.enable_thinking = enable_thinking
        self.trigger_ids = set()
        for text in ("}", ">", "<", "]"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if ids:
                self.trigger_ids.add(ids[-1])

    def __call__(self, input_ids, scores, **kwargs):
        for row, prompt_offset in zip(input_ids, self.prompt_offsets):
            if row.shape[0] <= prompt_offset:
                continue
            if int(row[-1]) not in self.trigger_ids:
                continue
            generated = self.tokenizer.decode(
                row[prompt_offset:], skip_special_tokens=False
            )
            if "<output>" in generated:
                output_part = generated.split("<output>", 1)[1]
                if "</output>" in output_part:
                    return True
                candidate = output_part.lstrip()
            else:
                candidate = generated.lstrip()
            if not candidate.startswith("{"):
                continue
            try:
                _, end = json.JSONDecoder().raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if end > 0:
                return True
        return False


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


def make_prompt(tokenizer, system: str, user: str, enable_thinking: bool = False) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_bundle(base_model: str, adapter: str | None, quantize: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "cuda:0",
    }
    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        # Full bf16 inference fits comfortably on A100 80GB and avoids the
        # bitsandbytes torch.compile path that makes long generations spend
        # minutes in inductor compilation.
        model_kwargs["torch_dtype"] = torch.bfloat16
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


def generate(
    tokenizer,
    model,
    system: str,
    user: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    stop_on_structured_complete: bool = False,
) -> tuple[str, int, int]:
    prompt = make_prompt(tokenizer, system, user, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    stopping = None
    if stop_on_structured_complete:
        stopping = StoppingCriteriaList(
            [
                StructuredCompletionStoppingCriteria(
                    tokenizer,
                    [int(inputs["input_ids"].shape[1])],
                    enable_thinking=enable_thinking,
                )
            ]
        )
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            disable_compile=True,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
        )
    new_ids = ids[0, inputs["input_ids"].shape[1] :]
    return (
        tokenizer.decode(new_ids, skip_special_tokens=False).strip(),
        int(new_ids.shape[-1]),
        int(inputs["attention_mask"].sum().item()),
    )


def generate_batch(
    tokenizer,
    model,
    cases: list[dict],
    max_new_tokens: int,
    enable_thinking: bool = False,
    stop_on_structured_complete: bool = False,
) -> list[tuple[str, int, int]]:
    """Generate a small left-padded batch without changing greedy semantics."""
    prompts = [
        make_prompt(tokenizer, case["system"], case["user"], enable_thinking=enable_thinking)
        for case in cases
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to("cuda:0")
    input_width = inputs["input_ids"].shape[1]
    stopping = None
    if stop_on_structured_complete:
        stopping = StoppingCriteriaList(
            [
                StructuredCompletionStoppingCriteria(
                    tokenizer,
                    [int(inputs["input_ids"].shape[1])] * inputs["input_ids"].shape[0],
                    enable_thinking=enable_thinking,
                )
            ]
        )
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            disable_compile=True,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
        )
    new_ids = ids[:, input_width:]
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    generated_tokens = int(new_ids.shape[-1])
    return [
        (tokenizer.decode(row, skip_special_tokens=False).strip(), generated_tokens, int(prompt_tokens))
        for row, prompt_tokens in zip(new_ids, prompt_lengths)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--priority-adapter", required=True)
    ap.add_argument("--output-only-adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument(
        "--stop-on-structured-complete",
        action="store_true",
        help="diagnostic: stop after </output> or the first complete JSON object",
    )
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Qwen3 thinking mode in the generation prompt (use for the priority objective)",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--no-quantization",
        action="store_true",
        help="load the base in bf16 for faster long-generation diagnostics",
    )
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
        "--split-manifest",
        default=None,
        help="optional groupdisjoint_split_manifest.json; use its train_indices/dev_indices instead of a random split",
    )
    ap.add_argument(
        "--selection",
        choices=("six", "all_heldout"),
        default="six",
        help="six auditable probes (default) or every record in the deterministic held-out split",
    )
    args = ap.parse_args()

    records = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        train_ids = sorted(int(i) for i in manifest["train_indices"])
        eval_ids = sorted(int(i) for i in manifest["dev_indices"])
        if set(train_ids) | set(eval_ids) != set(range(len(records))):
            raise ValueError("split manifest does not cover every record exactly")
        if set(train_ids) & set(eval_ids):
            raise ValueError("split manifest train/dev indices overlap")
    else:
        split = Dataset.from_dict({"idx": list(range(len(records)))})
        split = split.train_test_split(test_size=0.1, seed=args.seed)
        train_ids = sorted(split["train"]["idx"])
        eval_ids = sorted(split["test"]["idx"])
    if args.indices:
        # Explicit indices are authoritative. The previous implementation
        # filtered the default six probes first, silently dropping requested
        # cases that were not among those six.
        wanted = {int(value.strip()) for value in args.indices.split(",") if value.strip()}
        selected = [
            ("train" if i in set(train_ids) else "heldout_eval", i)
            for i in sorted(wanted)
            if 0 <= i < len(records)
        ]
    elif args.selection == "six":
        selected = [("train", i) for i in pick_three(train_ids)] + [
            ("heldout_eval", i) for i in pick_three(eval_ids)
        ]
    else:
        selected = [("heldout_eval", i) for i in eval_ids]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save_current() -> None:
        selection_label = "three evenly spaced ids from each deterministic LLaMA-Factory split"
        if args.indices:
            selection_label = "explicit positional record indices"
        elif args.selection == "all_heldout":
            selection_label = "all records from the deterministic LLaMA-Factory held-out split"
        payload = {
            "metadata": {
                "data_records": len(records),
                "split_seed": args.seed,
                "train_records": len(train_ids),
                "heldout_eval_records": len(eval_ids),
                "split_manifest": args.split_manifest,
                "selection": selection_label,
                "generation": "greedy, enable_thinking=%s, max_new_tokens=%d, batch_size=%d, stop_on_structured_complete=%s"
                % (
                    args.enable_thinking,
                    args.max_new_tokens,
                    max(1, args.batch_size),
                    args.stop_on_structured_complete,
                ),
                "model_only": args.only_model,
            },
            "cases": cases,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, adapter in model_specs:
        print(f"Loading {name}...", flush=True)
        tokenizer, model = load_bundle(args.base_model, adapter, quantize=not args.no_quantization)
        try:
            batch_size = max(1, args.batch_size)
            for start in range(0, len(cases), batch_size):
                batch = cases[start : start + batch_size]
                generations = (
                    generate_batch(
                        tokenizer,
                        model,
                        batch,
                        args.max_new_tokens,
                        args.enable_thinking,
                        args.stop_on_structured_complete,
                    )
                    if len(batch) > 1
                    else [
                        generate(
                            tokenizer,
                            model,
                            batch[0]["system"],
                            batch[0]["user"],
                            args.max_new_tokens,
                            args.enable_thinking,
                            args.stop_on_structured_complete,
                        )
                    ]
                )
                for case, (raw, generated_tokens, prompt_tokens) in zip(batch, generations):
                    case[name] = parse_target(raw)
                    case.setdefault("generation_meta", {})[name] = {
                        "prompt_tokens": prompt_tokens,
                        "generated_tokens": generated_tokens,
                        "max_new_tokens": args.max_new_tokens,
                        "hit_max_new_tokens": generated_tokens >= args.max_new_tokens,
                    }
                    print(f"{name} case={case['split']}:{case['id']} chars={len(raw)}", flush=True)
                    # Keep a valid partial result so a dropped remote session
                    # does not erase already generated cases.
                    save_current()
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    save_current()
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()

"""SFT / DPO / online GRPO entry point, with the same generative relation head.

GPU execution must use an authorized idle/shared device. No job is auto-launched.
Checkpoint selection is a separate full-graph tuning operation, never train loss.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

from .protocol import dumps, reward_components
from .chat import render_prompt


def read_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty training data: {path}")
    return rows


def length_audit(rows, tokenizer, max_length, mode):
    """Fail instead of truncating prompt, named endpoints, or preference suffix."""
    lengths = []
    for row in rows:
        suffixes = (
            [row["chosen"], row["rejected"]] if mode == "dpo" else [row["completion"]]
        )
        for suffix in suffixes:
            n = (
                len(tokenizer.encode(row["prompt"], add_special_tokens=True))
                + len(tokenizer.encode(suffix, add_special_tokens=False))
                + 2
            )
            if n > max_length:
                raise ValueError(
                    f"Overlength {row.get('id')}: {n}>{max_length}; rechunk and record excluded coverage, never silently truncate"
                )
            lengths.append(n)
    return {
        "sequences": len(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sft", "dpo", "grpo"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument(
        "--sft-adapter", help="Mandatory SFT starting/reference checkpoint for DPO/GRPO"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--max-completion-length", type=int, default=1536)
    p.add_argument("--beta", type=float)
    p.add_argument("--evidence-weight", type=float, default=0.2)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values are smoke/budget-controlled runs, not full epochs",
    )
    p.add_argument(
        "--verify-save",
        action="store_true",
        help="Check saved adapter logits against the in-memory trained policy",
    )
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.mode != "sft" and not args.sft_adapter:
        p.error("Preference/RL must start at an explicitly selected SFT checkpoint")
    rows = read_rows(args.data)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError(
            "Require explicitly train-tagged rows; evaluation/untagged data may not enter training"
        )
    if args.mode == "dpo" and any(r.get("chosen") == r.get("rejected") for r in rows):
        raise ValueError(
            "Identical chosen/rejected pairs are not informative preferences"
        )
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new/empty; never overwrite old checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    rows = [{**row, "prompt": render_prompt(tokenizer, row["prompt"])} for row in rows]
    audit = length_audit(rows, tokenizer, args.max_length, args.mode)
    manifest = {
        "status": "preflight",
        "arguments": vars(args),
        "length_audit": audit,
        "records": len(rows),
        "prompt_template": "Qwen3 apply_chat_template enable_thinking=False shared by train/eval/RL",
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "groups": sorted(set(r["group_id"] for r in rows)),
        "versions": {
            m: importlib.metadata.version(m) for m in ["torch", "transformers", "peft"]
        },
        "caveat": "Teacher-silver proxy supervision/rewards, not human-validated medical correctness",
    }
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(dumps(manifest))
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires the authorized GPU; preflight-only is CPU-safe"
        )
    if args.mode != "sft" and importlib.metadata.version("trl") != "0.23.1":
        raise RuntimeError("Use the isolated, pinned TRL 0.23.1 environment")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05 if args.mode == "sft" else 0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    common = dict(
        output_dir=str(out),
        learning_rate=args.learning_rate
        or {"sft": 5e-5, "dpo": 5e-6, "grpo": 2e-6}[args.mode],
        num_train_epochs=args.epochs or (3 if args.mode == "sft" else 1),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=[],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    if args.mode == "sft":
        from train_sft import JsonlSFTDataset, CausalCollator

        model = get_peft_model(model, lora)
        model.enable_input_require_grads()
        model.config.use_cache = False
        tokenizer.padding_side = "right"
        dataset = JsonlSFTDataset(rows, tokenizer, args.max_length)
        weights = [r.get("sample_weight", 1.0) for r in rows]

        class WeightedTrainer(Trainer):
            def _get_train_sampler(self, train_dataset=None):
                if len(set(weights)) == 1:
                    return super()._get_train_sampler(train_dataset)
                return torch.utils.data.WeightedRandomSampler(
                    weights,
                    len(weights),
                    replacement=True,
                    generator=torch.Generator().manual_seed(args.seed),
                )

        trainer = WeightedTrainer(
            model=model,
            args=TrainingArguments(**common),
            train_dataset=dataset,
            data_collator=CausalCollator(tokenizer),
        )
    elif args.mode == "dpo":
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer

        # Both adapters start at SFT. Disabling the SFT adapter would incorrectly
        # use the original base model as reference, changing the DPO objective.
        model = PeftModel.from_pretrained(
            model, args.sft_adapter, adapter_name="policy", is_trainable=True
        )
        model.load_adapter(
            args.sft_adapter, adapter_name="reference", is_trainable=False
        )
        model.set_adapter("policy")
        trainer = DPOTrainer(
            model=model,
            args=DPOConfig(
                **common,
                beta=args.beta or 0.1,
                max_length=args.max_length,
                model_adapter_name="policy",
                ref_adapter_name="reference",
                loss_type="sigmoid",
                precompute_ref_log_probs=True,
            ),
            train_dataset=Dataset.from_list(
                [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]
            ),
            processing_class=tokenizer,
        )
        manifest["reference_policy"] = (
            "Frozen identical SFT adapter, not bare Qwen base"
        )
    else:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        # GRPO's PEFT reference is adapter-disabled. Merge SFT into the in-memory
        # base first, then train a fresh residual LoRA; reference is exactly SFT.
        # Save only RL LoRA + base/SFT provenance; evaluator must rebuild this chain.
        model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()
        model = get_peft_model(model, lora)
        tokenizer.padding_side = "left"
        rollout_path = out / "rollouts.jsonl"

        def relation_reward(
            completions, batch_json, target_json, trainer_state=None, **kwargs
        ):
            results = []
            with rollout_path.open("a") as handle:
                for completion, batch, target in zip(
                    completions, batch_json, target_json
                ):
                    if not isinstance(completion, str):
                        raise ValueError(
                            "This protocol uses plain string prompts/completions"
                        )
                    batch, target = json.loads(batch), json.loads(target)
                    score = reward_components(
                        completion, batch, target, evidence_weight=args.evidence_weight
                    )
                    handle.write(
                        dumps(
                            {
                                "step": getattr(trainer_state, "global_step", None),
                                "id": batch["id"],
                                "completion": completion,
                                **score,
                            }
                        )
                        + "\n"
                    )
                    results.append(score["reward"])
            return results

        # Real on-policy rollouts, not precomputed chosen/rejected preference rows.
        if (args.batch_size * args.gradient_accumulation) % args.num_generations:
            raise ValueError(
                "Effective generation batch must be divisible by num_generations"
            )
        dataset = Dataset.from_list(
            [
                {
                    "prompt": r["prompt"],
                    "batch_json": dumps(r),
                    "target_json": dumps(r["target"]),
                }
                for r in rows
            ]
        )
        trainer = GRPOTrainer(
            model=model,
            args=GRPOConfig(
                **common,
                beta=args.beta or 0.04,
                max_prompt_length=args.max_length,
                max_completion_length=args.max_completion_length,
                num_generations=args.num_generations,
                temperature=1.0,
                top_p=1.0,
                use_vllm=False,
                loss_type="dr_grpo",
                scale_rewards=False,
                mask_truncated_completions=False,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=relation_reward,
        )
        manifest["reference_policy"] = (
            "SFT merged in memory; fresh residual RL LoRA; reload base+SFT.merge()+RL"
        )
        manifest["max_completion_length_note"] = (
            "Training rollout cap, separate from 16000-token evaluation cap; truncated rollouts logged and penalized"
        )
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    manifest["total_parameters"] = sum(p.numel() for p in model.parameters())
    if any(
        p.requires_grad and "lora_" not in name for name, p in model.named_parameters()
    ):
        raise RuntimeError(
            "Unexpected trainable base parameter; this study is LoRA-only"
        )
    trainer.train()
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(out / "final")
    if args.verify_save:
        active = "policy" if args.mode == "dpo" else "default"
        saved = out / "final" / active if active != "default" else out / "final"
        model.eval()
        model.set_adapter(active)
        inputs = tokenizer(rows[0]["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            expected = model(**inputs).logits[:, -1, :].float().cpu()
        model.load_adapter(
            str(saved), adapter_name="save_verification", is_trainable=False
        )
        model.set_adapter("save_verification")
        with torch.inference_mode():
            actual = model(**inputs).logits[:, -1, :].float().cpu()
        delta = float((expected - actual).abs().max())
        manifest["save_reload_max_logit_difference"] = delta
        model.set_adapter(active)
        model.delete_adapter("save_verification")
        if delta > 0.02:
            raise RuntimeError(
                f"Saved adapter reload differs from trained policy: {delta}"
            )
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    manifest["status"] = "training_complete_evaluation_pending"
    manifest["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

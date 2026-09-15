"""SFT / DPO / online GRPO entry point, with the same generative relation head.

GPU execution must use an authorized idle/shared device. No job is auto-launched.
Checkpoint selection is a separate full-graph tuning operation, never train loss.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .protocol import dumps, parse_decisions, reward_components
from .chat import render_prompt


def read_rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty training data: {path}")
    return rows


def prompt_batch(row):
    """Recover only input-visible contract fields, never silver/provenance fields.

    The shared protocol appends its JSON payload on the last nonempty line.
    Validate before chat rendering. This is not proof of source-split provenance.
    """
    try:
        payload = json.loads(row["prompt"].strip().splitlines()[-1])
        if set(payload) != {"entities", "pairs", "spans"}:
            raise ValueError("unexpected payload fields")
        for key, fields in (
            ("entities", {"entity_id", "name", "label"}),
            ("pairs", {"pair_id", "source", "target", "allowed_relations"}),
            ("spans", {"span_id", "text"}),
        ):
            if not isinstance(payload[key], list) or not payload[key]:
                raise ValueError(f"empty or invalid {key}")
            if any(
                not isinstance(item, dict) or set(item) != fields
                for item in payload[key]
            ):
                raise ValueError(f"unexpected {key} fields (possible metadata leakage)")
        for key, id_key in (
            ("entities", "entity_id"),
            ("pairs", "pair_id"),
            ("spans", "span_id"),
        ):
            ids = [item[id_key] for item in payload[key]]
            if any(not isinstance(i, str) or not i for i in ids) or len(
                set(ids)
            ) != len(ids):
                raise ValueError(f"invalid or duplicate {id_key}")
        entities = {e["entity_id"] for e in payload["entities"]}
        for pair in payload["pairs"]:
            if pair["source"] not in entities or pair["target"] not in entities:
                raise ValueError("unknown endpoint")
            if (
                not isinstance(pair["allowed_relations"], list)
                or not pair["allowed_relations"]
            ):
                raise ValueError("invalid allowed relations")
        return {
            "id": row["id"],
            "pairs": [
                {"pair_id": p["pair_id"], "allowed": p["allowed_relations"]}
                for p in payload["pairs"]
            ],
            "spans": payload["spans"],
        }
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid input-only prompt payload for {row.get('id')}: {exc}"
        ) from exc


def validate_training_rows(rows, mode):
    if not rows:
        raise ValueError("Empty training data")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "train":
            raise ValueError(
                "Require explicitly train-tagged rows; no tune/regression data"
            )
        for key in ("id", "group_id"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (str, int))
                or not str(value).strip()
            ):
                raise ValueError(f"Missing/invalid {key}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate training id: {row['id']}")
        seen.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Empty/non-string prompt: {row['id']}")
        if row["prompt"].startswith("<|im_start|>"):
            raise ValueError("Prompt is already chat-rendered")
        if mode == "sft":
            continue
        batch = prompt_batch(row)
        fields = ("chosen", "rejected") if mode == "dpo" else ("completion",)
        parsed = []
        for field in fields:
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string {field}: {row['id']}")
            decisions, errors = parse_decisions(text, batch)
            if errors:
                raise ValueError(f"Invalid {field} for {row['id']}: {errors}")
            parsed.append(
                sorted(
                    (d["pair_id"], d["relation"], tuple(d["evidence_span_ids"]))
                    for d in decisions
                )
            )
        if mode == "dpo" and parsed[0] == parsed[1]:
            raise ValueError(
                "Identical chosen/rejected decisions are not informative preferences"
            )
        if mode == "grpo" and json.loads(row["completion"]) != row.get("target"):
            raise ValueError(f"Completion/target mismatch: {row['id']}")


def length_audit(rows, tokenizer, max_length, mode, max_completion_length=None):
    """Fail instead of truncating prompt, named endpoints, or preference suffix."""
    lengths, prompt_lengths, completion_lengths = [], [], []
    for row in rows:
        # TRL 0.23.1 tokenizes already-rendered strings without extra specials;
        # its DPO tokenize_row appends exactly one EOS to each completion.
        prompt_n = len(
            tokenizer.encode(row["prompt"], add_special_tokens=mode == "sft")
        )
        prompt_lengths.append(prompt_n)
        if mode == "grpo":
            if (
                max_completion_length is None
                or not 0 < max_completion_length < max_length
            ):
                raise ValueError("GRPO needs 0 < max_completion_length < max_length")
            if prompt_n + max_completion_length > max_length:
                raise ValueError(
                    f"Overlength rollout budget {row.get('id')}: prompt + rollout cap exceeds {max_length}"
                )
        suffixes = (
            [row["chosen"], row["rejected"]] if mode == "dpo" else [row["completion"]]
        )
        for suffix in suffixes:
            completion_n = len(
                tokenizer.encode(suffix, add_special_tokens=False)
            ) + int(tokenizer.eos_token_id is not None)
            completion_lengths.append(completion_n)
            n = prompt_n + completion_n
            if mode == "grpo" and completion_n > max_completion_length:
                raise ValueError(
                    f"Overlength GRPO target {row.get('id')}: {completion_n}>{max_completion_length}"
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
        "prompt_tokens_once": sum(prompt_lengths),
        "completion_tokens_once": sum(completion_lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "max_completion_tokens": max(completion_lengths),
        "count_scope": "one dataset pass, not observed training exposure; completions include EOS",
    }


def rl_config_kwargs(args):
    """Version-specific TRL knobs, also checked by dependency-free CPU tests."""
    if args.mode == "dpo":
        return dict(
            beta=args.beta,
            max_length=args.max_length,
            # DPOConfig defaults to 512 and otherwise silently slices the prompt.
            max_prompt_length=None,
            max_completion_length=None,
            model_adapter_name="policy",
            ref_adapter_name="reference",
            loss_type="sigmoid",
            precompute_ref_log_probs=True,
            reference_free=False,
            sync_ref_model=False,
            disable_dropout=True,
        )
    return dict(
        beta=args.beta,
        max_prompt_length=args.max_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        steps_per_generation=args.gradient_accumulation,
        num_iterations=1,
        temperature=1.0,
        top_p=1.0,
        use_vllm=False,
        loss_type="dr_grpo",
        scale_rewards=False,
        sync_ref_model=False,
        disable_dropout=True,
        mask_truncated_completions=False,
    )


def grpo_dataset_rows(rows):
    # The trainer sends ONLY prompt to the tokenizer/model and keeps the other
    # columns for reward_funcs. Do not forward the entire silver training row.
    return [
        {
            "prompt": r["prompt"],
            "batch_json": dumps(r["reward_batch"]),
            "target_json": dumps(r["target"]),
        }
        for r in rows
    ]


def make_relation_reward(path, evidence_weight, eos_token_id, counters):
    def relation_reward(
        completions,
        batch_json,
        target_json,
        completion_ids,
        trainer_state=None,
        **kwargs,
    ):
        if not (
            len(completions)
            == len(batch_json)
            == len(target_json)
            == len(completion_ids)
        ):
            raise ValueError("Rollout/reward column length mismatch")
        records = []
        for completion, batch, target, ids in zip(
            completions, batch_json, target_json, completion_ids
        ):
            if not isinstance(completion, str):
                raise ValueError("This protocol uses plain string prompts/completions")
            batch, target = json.loads(batch), json.loads(target)
            score = reward_components(
                completion, batch, target, evidence_weight=evidence_weight
            )
            if not math.isfinite(score["reward"]):
                raise ValueError("Non-finite rollout reward")
            records.append(
                {
                    "step": getattr(trainer_state, "global_step", None),
                    "id": batch["id"],
                    "completion": completion,
                    "completion_tokens": len(ids),
                    "eos_observed": eos_token_id in ids,
                    "truncated": eos_token_id not in ids,
                    **score,
                }
            )
        with Path(path).open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(dumps(record) + "\n")
                counters["online_rollouts"] += 1
                counters["rollout_completion_tokens"] += record["completion_tokens"]
                counters["truncated_rollouts"] += int(record["truncated"])
        return [r["reward"] for r in records]

    return relation_reward


def adapter_path(checkpoint, mode):
    return Path(checkpoint) / "policy" if mode == "dpo" else Path(checkpoint)


def checkpoint_kwargs(args):
    steps = args.save_steps or (min(50, args.max_steps) if args.max_steps > 0 else None)
    return dict(
        save_strategy="steps" if steps else "epoch",
        save_steps=steps or 500,
        save_total_limit=args.save_total_limit,
    )


def adapter_fingerprint(model, adapter):
    result, count = hashlib.sha256(), 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" in name and f".{adapter}." in name:
            result.update(name.replace(f".{adapter}.", ".ADAPTER.").encode())
            result.update(str(tuple(parameter.shape)).encode())
            result.update(
                parameter.detach().float().cpu().contiguous().numpy().tobytes()
            )
            count += 1
    if not count:
        raise RuntimeError(f"No LoRA parameters found for {adapter}")
    return result.hexdigest()


def initialize_dpo_reference(model, checkpoint):
    """Load the exact SFT reference, including PEFT 0.17.1's upcast edge case.

    Adding a second adapter can copy FP32 checkpoint values into temporary BF16
    tensors, then upcast those rounded values. A bounded second load into the
    now-upcast adapter recovers checkpoint precision. Never relax equality or
    copy a changed policy into the reference. Observed on the real pinned 8B.
    """
    policy_before = adapter_fingerprint(model, "policy")
    for attempt in (1, 2):
        model.load_adapter(checkpoint, adapter_name="reference", is_trainable=False, local_files_only=True)
        model.set_adapter("policy")
        if adapter_fingerprint(model, "policy") != policy_before:
            raise RuntimeError("Loading the reference unexpectedly changed the SFT policy")
        if adapter_fingerprint(model, "reference") == policy_before:
            return {"reference_load_attempts": attempt, "precision_reload_needed": attempt == 2,
                    "initial_adapters_bitwise_equal": True, "initial_sft_adapter_sha256": policy_before}
    raise RuntimeError("Exact SFT reference could not be restored; do not bypass DPO initialization equality")


def record_dpo_batch(counters, inputs):
    # Called only after policy compute_loss, never during reference precomputation.
    counters["dpo_pair_loss_exposures"] += len(inputs["prompt_input_ids"])
    counters["dpo_prompt_token_loss_exposures"] += 2 * int(
        inputs["prompt_attention_mask"].sum().item()
    )
    counters["dpo_completion_token_loss_exposures"] += sum(
        int(inputs[k].sum().item())
        for k in ("chosen_attention_mask", "rejected_attention_mask")
    )


class PairCountDataset:
    """Keep counts attached to sampled indices, including weighted resampling."""

    def __init__(self, dataset, rows):
        self.dataset = dataset
        self.counts = [len(r["pairs"]) for r in rows]
        if len(dataset) != len(self.counts):
            raise ValueError("SFT dataset/pair-count alignment mismatch")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "_pair_count": self.counts[index]}


class PairCountCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, items):
        counts = [item["_pair_count"] for item in items]
        batch = self.collator(
            [{k: v for k, v in item.items() if k != "_pair_count"} for item in items]
        )
        return {**batch, "_pair_counts": counts}


def record_sft_batch(counters, inputs, pair_counts):
    if len(inputs["input_ids"]) != len(pair_counts):
        raise ValueError("SFT batch/pair-count alignment mismatch")
    tokens = int(inputs["attention_mask"].sum().item())
    supervised = int((inputs["labels"] != -100).sum().item())
    counters["sft_sequence_loss_exposures"] += len(pair_counts)
    counters["sft_pair_loss_exposures"] += sum(pair_counts)
    counters["sft_input_token_loss_exposures"] += tokens
    counters["sft_completion_token_loss_exposures"] += supervised
    counters["sft_prompt_token_loss_exposures"] += tokens - supervised


def load_saved_policy(args, saved, model_loader, peft_loader, model_kwargs):
    """Independent reconstruction; reusing the trained merged base is insufficient."""
    fresh = model_loader.from_pretrained(args.base_model, **model_kwargs)
    if args.mode == "grpo":
        fresh = peft_loader.from_pretrained(
            fresh, args.sft_adapter, local_files_only=True
        ).merge_and_unload()
    return peft_loader.from_pretrained(
        fresh, str(saved), is_trainable=False, local_files_only=True
    )


def terminal_inference_model(trainer, model, active):
    """Match the plain evaluator, removing Trainer's autocast forward wrapper.

    Must only run after training/save: unwrap mutates the forward function. An
    outer autocast(enabled=False) alone does not remove the nested wrapper.
    See Accelerate1.10.1 unwrap_model(keep_fp32_wrapper=False).
    """
    model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    model.eval()
    model.set_adapter(active)
    model.config.use_cache = False
    return model


def require_saved_weight_equality(model, expected):
    observed = adapter_fingerprint(model, "default")
    if observed != expected:
        raise RuntimeError("Saved/reloaded adapter differs from the recorded trained weights")
    return observed


def parse_args(argv=None):
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
    p.add_argument(
        "--max-length",
        type=int,
        default=6144,
        help="Total prompt + completion token budget in every mode",
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=1536,
        help="GRPO rollout/target cap including EOS; no DPO completion truncation",
    )
    p.add_argument("--beta", type=float)
    p.add_argument("--evidence-weight", type=float, default=0.2)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-steps",
        type=int,
        help="Optimizer steps between checkpoints; finite runs default to at most 50",
    )
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values are smoke/budget-controlled runs, not full epochs",
    )
    p.add_argument(
        "--verify-save",
        action="store_true",
        help="Reconstruct a fresh base(+merged SFT)+saved policy and compare logits",
    )
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.mode != "sft" and not args.sft_adapter:
        p.error("Preference/RL must start at an explicitly selected SFT checkpoint")
    for name in (
        "batch_size",
        "gradient_accumulation",
        "max_length",
        "max_completion_length",
        "save_total_limit",
    ):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.max_steps != -1 and args.max_steps < 1:
        p.error("max_steps must be -1 (epoch budget) or positive")
    if args.save_steps is not None and args.save_steps < 1:
        p.error("save_steps must be positive")
    args.learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else {"sft": 5e-5, "dpo": 5e-6, "grpo": 2e-6}[args.mode]
    )
    args.epochs = (
        args.epochs if args.epochs is not None else (3 if args.mode == "sft" else 1)
    )
    args.beta = (
        args.beta if args.beta is not None else (0.04 if args.mode == "grpo" else 0.1)
    )
    for name in ("learning_rate", "epochs", "beta", "evidence_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if (
        args.learning_rate == 0
        or args.epochs == 0
        or (args.mode == "dpo" and args.beta == 0)
    ):
        p.error("learning_rate/epochs and DPO beta must be positive")
    # This entry point writes one manifest/rollout stream; DDP needs rank-safe IO
    # and exposure reduction before it can be claimed supported.
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error(
            "This audited entry point supports one process only; launch each run on one authorized GPU"
        )
    if args.mode == "grpo":
        if (
            args.num_generations < 2
            or (args.batch_size * args.gradient_accumulation) % args.num_generations
        ):
            p.error(
                "GRPO needs >=2 generations dividing batch_size * gradient_accumulation"
            )
        if args.max_completion_length >= args.max_length:
            p.error(
                "GRPO completion cap must leave positive prompt space in max_length"
            )
    return args


def main():
    args = parse_args()
    rows = read_rows(args.data)
    validate_training_rows(rows, args.mode)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be new/empty; never overwrite old checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "preflight_started", "arguments": vars(args)}
    try:
        execute(args, rows, out, manifest)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def execute(args, rows, out, manifest):
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
    versions = ["torch", "transformers", "peft"] + (
        ["trl", "datasets", "accelerate"] if args.mode != "sft" else []
    )
    manifest["versions"] = {m: importlib.metadata.version(m) for m in versions}
    if args.mode != "sft" and manifest["versions"]["trl"] != "0.23.1":
        raise RuntimeError("Use the isolated, pinned TRL 0.23.1 environment")
    if (
        args.mode == "grpo"
        and len(rows)
        < (args.batch_size * args.gradient_accumulation) // args.num_generations
    ):
        raise ValueError("Too few prompt rows for a complete GRPO generation batch")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The shared Qwen3 protocol requires an EOS token")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    if args.mode == "grpo":
        rows = [{**r, "reward_batch": prompt_batch(r)} for r in rows]
    rows = [{**row, "prompt": render_prompt(tokenizer, row["prompt"])} for row in rows]
    audit = length_audit(
        rows, tokenizer, args.max_length, args.mode, args.max_completion_length
    )
    manifest.update(
        {
            "status": "cpu_preflight_passed_no_model_loaded",
            "arguments": vars(args),
            "length_audit": audit,
            "records": len(rows),
            "prompt_template": "Qwen3 apply_chat_template enable_thinking=False shared by train/eval/RL",
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "groups": sorted(set(r["group_id"] for r in rows), key=str),
            "effective_batch_sequences": args.batch_size * args.gradient_accumulation,
            "grpo_unique_prompts_per_generation_batch": (
                (args.batch_size * args.gradient_accumulation) // args.num_generations
                if args.mode == "grpo"
                else None
            ),
            "caveat": "Teacher-silver proxy supervision/rewards, not human-validated medical correctness",
        }
    )
    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.preflight_only:
        print(dumps(manifest))
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires the authorized GPU; preflight-only is CPU-safe"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This pinned training recipe requires a BF16-capable authorized GPU"
        )
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    context_length = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_length, int) and args.max_length > context_length:
        raise ValueError(
            f"max_length exceeds model context: {args.max_length}>{context_length}"
        )
    counters = Counter()
    manifest["observed_exposure"] = counters
    manifest["exposure_scope"] = (
        "SFT/DPO successful policy loss forward calls (not optimizer-step estimates); non-padding tokens include EOS. GRPO counters are actual reward-scored sampled completions, not dataset rows."
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
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        # TRL0.23.1's generation context reinstates default reentrant
        # checkpointing. Match that runtime explicitly for GRPO and provide
        # gradient-bearing embedding outputs; base weights remain frozen.
        gradient_checkpointing_kwargs={"use_reentrant": args.mode == "grpo"},
        logging_steps=5,
        logging_first_step=True,
        **checkpoint_kwargs(args),
        report_to=[],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        # GRPO needs custom reward columns; DPO's collator needs only token IDs.
        remove_unused_columns=args.mode == "dpo",
        disable_tqdm=True,
    )
    manifest["effective_trainer_config"] = {
        **common,
        **(rl_config_kwargs(args) if args.mode != "sft" else {}),
    }
    if args.mode == "sft":
        from train_sft import JsonlSFTDataset, CausalCollator

        model = get_peft_model(model, lora)
        model.enable_input_require_grads()
        model.config.use_cache = False
        tokenizer.padding_side = "right"
        dataset = PairCountDataset(
            JsonlSFTDataset(rows, tokenizer, args.max_length), rows
        )
        weights = [r.get("sample_weight", 1.0) for r in rows]

        class WeightedTrainer(Trainer):
            def compute_loss(
                self, model, inputs, return_outputs=False, num_items_in_batch=None
            ):
                # Counts are never tokenized or forwarded to the language model.
                pair_counts = inputs.pop("_pair_counts")
                loss = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                if model.training:
                    record_sft_batch(counters, inputs, pair_counts)
                return loss

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
            data_collator=PairCountCollator(CausalCollator(tokenizer)),
        )
    elif args.mode == "dpo":
        from datasets import Dataset
        from trl import DPOConfig, DPOTrainer

        # Both adapters start at SFT. Disabling the SFT adapter would incorrectly
        # use the original base model as reference, changing the DPO objective.
        model = PeftModel.from_pretrained(
            model,
            args.sft_adapter,
            adapter_name="policy",
            is_trainable=True,
            local_files_only=True,
        )
        manifest["dpo_initialization"] = initialize_dpo_reference(model, args.sft_adapter)

        class AuditedDPOTrainer(DPOTrainer):
            def compute_loss(
                self, model, inputs, return_outputs=False, num_items_in_batch=None
            ):
                loss = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                if model.training:
                    record_dpo_batch(counters, inputs)
                return loss

        trainer = AuditedDPOTrainer(
            model=model,
            args=DPOConfig(**common, **rl_config_kwargs(args)),
            train_dataset=Dataset.from_list(
                [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows]
            ),
            processing_class=tokenizer,
        )
        manifest["reference_policy"] = (
            "Frozen identical SFT adapter, not bare Qwen base"
        )
        manifest["reference_adapter_sha256_before"] = adapter_fingerprint(
            model, "reference"
        )
        if (
            adapter_fingerprint(model, "policy")
            != manifest["reference_adapter_sha256_before"]
        ):
            raise RuntimeError("Initial DPO policy/reference adapter weights differ")
    else:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        # GRPO's PEFT reference is adapter-disabled. Merge SFT into the in-memory
        # base first, then train a fresh residual LoRA; reference is exactly SFT.
        # Save only RL LoRA + base/SFT provenance; evaluator must rebuild this chain.
        model = PeftModel.from_pretrained(
            model, args.sft_adapter, local_files_only=True
        ).merge_and_unload()
        model = get_peft_model(model, lora)
        model.enable_input_require_grads()
        manifest["gradient_checkpointing_compatibility"] = (
            "TRL0.23.1 re-enables reentrant checkpointing after generation; input activation gradients enabled, base parameters frozen"
        )
        tokenizer.padding_side = "left"
        rollout_path = out / "rollouts.jsonl"

        # Real on-policy rollouts, not precomputed chosen/rejected preference rows.
        dataset = Dataset.from_list(grpo_dataset_rows(rows))
        relation_reward = make_relation_reward(
            rollout_path, args.evidence_weight, tokenizer.eos_token_id, counters
        )
        trainer = GRPOTrainer(
            model=model,
            args=GRPOConfig(**common, **rl_config_kwargs(args)),
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=relation_reward,
        )
        manifest["reference_policy"] = (
            "SFT merged in memory; fresh residual RL LoRA; reload base+SFT.merge()+RL"
        )
        manifest["max_completion_length_note"] = (
            "Rollout cap reserved within max_length; EOS absence logged as truncation. Protocol penalizes malformed JSON, not truncation itself; no extra truncation penalty is claimed."
        )
        manifest["kl_enabled"] = args.beta > 0
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    manifest["total_parameters"] = sum(p.numel() for p in model.parameters())
    if not manifest["trainable_parameters"] or any(
        p.requires_grad
        and ("lora_" not in name or (args.mode == "dpo" and ".policy." not in name))
        for name, p in model.named_parameters()
    ):
        raise RuntimeError(
            "Unexpected trainable base parameter; this study is LoRA-only"
        )
    active = "policy" if args.mode == "dpo" else "default"
    manifest["policy_adapter_sha256_before"] = adapter_fingerprint(model, active)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    train_result = trainer.train()
    manifest["global_step"] = trainer.state.global_step
    manifest["train_metrics"] = train_result.metrics
    if not math.isfinite(train_result.metrics.get("train_loss", 0.0)):
        raise RuntimeError("Non-finite final training loss")
    manifest["policy_adapter_sha256_after"] = adapter_fingerprint(model, active)
    manifest["policy_weights_changed"] = (
        manifest["policy_adapter_sha256_before"]
        != manifest["policy_adapter_sha256_after"]
    )
    if args.mode == "dpo":
        manifest["reference_adapter_sha256_after"] = adapter_fingerprint(
            model, "reference"
        )
        if (
            manifest["reference_adapter_sha256_before"]
            != manifest["reference_adapter_sha256_after"]
        ):
            raise RuntimeError("Frozen SFT reference changed during DPO")
    if args.max_steps > 0 and trainer.state.global_step != args.max_steps:
        raise RuntimeError(
            "Training stopped before the requested optimizer-step budget"
        )
    trainer.save_model(str(out / "final"))
    tokenizer.save_pretrained(out / "final")
    saved = adapter_path(out / "final", args.mode)
    manifest["final_adapter"] = str(saved)
    manifest["checkpoint_adapters"] = [
        str(adapter_path(p, args.mode))
        for p in sorted(
            out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if p.is_dir()
    ]
    if args.verify_save:
        model = terminal_inference_model(trainer, model, active)
        device = model.device
        inputs = tokenizer(
            rows[0]["prompt"],
            add_special_tokens=args.mode == "sft",
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            expected = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        # Avoid holding two 8B bases on the same GPU. The trained model remains
        # on CPU after this terminal verification; no further training follows.
        model.to("cpu")
        torch.cuda.empty_cache()
        reloaded = (
            load_saved_policy(
                args, saved, AutoModelForCausalLM, PeftModel, model_kwargs
            )
            .to(device)
            .eval()
        )
        manifest["saved_adapter_sha256_verified"] = require_saved_weight_equality(
            reloaded, manifest["policy_adapter_sha256_after"]
        )
        with torch.inference_mode(), torch.autocast(device.type, enabled=False):
            actual = reloaded(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        delta = float((expected - actual).abs().max())
        manifest["save_reload_max_logit_difference"] = delta
        manifest["save_reload_scope"] = (
            "exact trained/saved LoRA weight fingerprint, fresh base(+SFT merge)+saved adapter; both plain inference without Trainer autocast wrapper; first training prompt final-token logits only"
        )
        if not math.isfinite(delta) or delta > 0.02:
            raise RuntimeError(
                f"Saved adapter reload differs from trained policy: {delta}"
            )
        del reloaded
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    manifest["status"] = "training_complete_evaluation_pending"
    manifest["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

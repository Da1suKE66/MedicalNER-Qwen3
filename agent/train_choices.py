"""Bounded SFT engineering verification of the non-JSON, two-stage LM contract.

Only recipe-verified complete artificial worlds are admitted in this first
release. Partial medical silver targets require a separate supervision migration;
they must not become empty/negative choices by removing their metadata.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path


def _exposure_delta(metadata):
    """Count one prepared microbatch without trusting model-facing metadata."""
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Choice training requires nonempty per-row exposure metadata")
    delta = Counter(batches=1)
    for item in metadata:
        if not isinstance(item, dict) or item.get("stage") not in {"entities", "relations"}:
            raise ValueError("Invalid choice exposure stage")
        delta["rows"] += 1
        delta[f"{item['stage']}_rows"] += 1
        for key in ("prompt_tokens", "completion_tokens", "action_count"):
            value = item.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid choice exposure count: {key}")
            delta[key] += value
    return delta


class ChoiceExposureMixin:
    """Record only microbatches whose HF training_step returned successfully.

    The collator can run ahead of training via dataloader prefetch. Its metadata
    travels with the batch, is removed before HF/model forwarding, and is only
    committed after the real training_step (including backward) returns. This
    counts consumed training tasks, not optimizer updates or merely fetched rows.
    """

    def __init__(self, *args, **kwargs):
        self.observed_exposure = Counter()
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, *args, **kwargs):
        if "_choice_counts_batch" not in inputs:
            raise ValueError("Choice training batch is missing exposure metadata")
        metadata = inputs.pop("_choice_counts_batch")
        if len(metadata) != len(inputs["input_ids"]):
            raise ValueError("Choice exposure metadata does not match batch size")
        delta = _exposure_delta(metadata)
        result = super().training_step(model, inputs, *args, **kwargs)
        self.observed_exposure.update(delta)
        return result


def main(argv=None, *, relation_format="choices"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Recipe-verified recovery preferences; rejected JSON is never a model target")
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--relation-format", choices=("choices", "labels"), default=relation_format)
    p.add_argument("--stages", nargs="+", choices=("entities", "relations"))
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.stages is None:
        args.stages = ["relations"] if args.relation_format == "labels" else ["entities", "relations"]
    if args.relation_format == "labels" and args.stages != ["relations"]:
        p.error("Literal-label training supports the relation stage only")
    if (any(getattr(args, k) < 1 for k in ("max_steps", "max_length", "batch_size", "gradient_accumulation"))
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or len(set(args.stages)) != len(args.stages)):
        p.error("Invalid training budget, learning rate, or duplicate stage")
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new output directory; previous runs are immutable")
    from kg_agent.choice_backend import adapter_files_digest, base_input_digest
    if args.relation_format == "labels":
        from generative.label_data import compile_recovery_row, encode_row
        from kg_agent.label_backend import CONTRACT, COMPLETION_ENCODING, protocol_implementation_digest
        receipt_filename = "label_contract.json"
    else:
        from generative.choice_data import compile_recovery_row, encode_row
        from kg_agent.choice_backend import CONTRACT, protocol_implementation_digest
        COMPLETION_ENCODING = "independent_integer_newline_tokens_then_eos"
        receipt_filename = "choice_contract.json"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    source = Path(args.data).read_bytes()
    rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty source or repeated source IDs")
    compiled = [compile_recovery_row(row, stage) for row in rows for stage in args.stages]
    encoded = [encode_row(row, tokenizer, args.max_length, require_train=True) for row in compiled]
    audit = {
        "status": "preflight_complete", "scope": "artificial_pipeline_contract_engineering_only",
        "output_contract": CONTRACT, "args": vars(args), "protocol_sha256": protocol_implementation_digest(),
        "base_input_sha256": base_input_digest(args.base_model),
        "data_sha256": hashlib.sha256(source).hexdigest(), "source_rows": len(rows),
        "training_rows": len(encoded), "stages": args.stages,
        "max_total_tokens": max(len(item["input_ids"]) for item in encoded),
        "completion_encoding": COMPLETION_ENCODING,
        "medical_training": False, "clinical_quality_claim": False,
        "unknown_policy": "Rejected at reconstruction; no unknown-to-empty conversion",
        "exposure_counting_version": "successful_training_step_v2",
        "exposure_definitions": {
            "observed_exposure": "Microbatches whose Trainer.training_step returned successfully, including backward",
            "collator_exposures": "Prepared microbatches, including prefetch that may never enter training_step",
            "realized_collator_exposures": "Backward-compatible alias of collator_exposures, not training exposure",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_manifest.json"
    def save():
        manifest.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    save()
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, PeftModel
    from train_sft import CausalCollator
    from generative.train import adapter_fingerprint, require_saved_weight_equality
    set_seed(args.seed)
    model_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.enable_input_require_grads()
    model.to("cuda")
    before = adapter_fingerprint(model, "default")
    collator_exposures = Counter()
    delegate = CausalCollator(tokenizer)
    def collate(items):
        metadata = [dict(item["_choice_counts"]) for item in items]
        delta = _exposure_delta(metadata)
        batch = delegate([{k: item[k] for k in ("input_ids", "labels")} for item in items])
        batch["_choice_counts_batch"] = metadata
        collator_exposures.update(delta)
        return batch
    training_args = TrainingArguments(
        output_dir=str(out), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, warmup_ratio=0.03, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, logging_first_step=True, save_strategy="no", report_to=[],
        logging_nan_inf_filter=False,
        remove_unused_columns=False, seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    class ChoiceTrainer(ChoiceExposureMixin, Trainer):
        pass
    trainer = ChoiceTrainer(model=model, args=training_args, train_dataset=encoded, data_collator=collate)
    audit.update(status="training", adapter_parameters_before=before, policy_adapter_sha256_before=before)
    save()
    try:
        result = trainer.train()
    finally:
        # Persist partial counts on failure too; the failing training_step is
        # excluded, even though its inputs and later prefetch may be prepared.
        audit.update(observed_exposure=dict(trainer.observed_exposure),
                     collator_exposures=dict(collator_exposures),
                     realized_collator_exposures=dict(collator_exposures))
        save()
    trainer.save_state()
    trained = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    trained.eval()
    trained.set_adapter("default")
    trained.config.use_cache = False
    after = adapter_fingerprint(trained, "default")
    if any(not torch.isfinite(parameter).all().item()
           for name, parameter in trained.named_parameters() if "lora_" in name):
        raise RuntimeError("Nonfinite choice adapter parameters")
    if before == after or trainer.state.global_step != args.max_steps:
        raise RuntimeError("Choice SFT failed to update the adapter for the requested budget")
    if not math.isfinite(float(result.metrics["train_loss"])):
        raise RuntimeError("Nonfinite choice training loss")
    final = out / "final"
    trained.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    audit.update(adapter_parameters_after=after, optimizer_updates=trainer.state.global_step,
                 train_metrics=result.metrics,
                 adapter_sha256=adapter_files_digest(final), status="verifying_saved_adapter")
    audit.update(policy_adapter_sha256_after=after, policy_weights_changed=True,
                 global_step=trainer.state.global_step)
    save()
    del trained, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs), str(final),
        local_files_only=True,
    )
    require_saved_weight_equality(reloaded, after)
    audit.update(status="training_complete_evaluation_pending", saved_reload_exact=True,
                 saved_adapter_sha256_verified=after, final_adapter=str(final))
    save()
    receipt = {k: audit[k] for k in ("output_contract", "stages", "completion_encoding", "protocol_sha256", "adapter_sha256", "base_input_sha256", "scope")}
    receipt.update(training_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   training_data_sha256=audit["data_sha256"], medical_quality_evaluated=False)
    (final / receipt_filename).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

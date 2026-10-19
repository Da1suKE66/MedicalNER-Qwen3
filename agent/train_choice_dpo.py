"""Finite-action DPO and matched continued-SFT; never train generated JSON.

Both modes consume the same certified preference rows and start from the same
bounded-choice adapter. Completion tokens are assembled by the choice compiler,
not by tokenizing a serialized action list. This is an artificial contract probe,
not a medical quality experiment. Run as ``python agent/train_choice_dpo.py``.
"""

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import traceback


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("dpo", "continued-sft"), default="dpo")
    p.add_argument("--data", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--parent-adapter", "--sft-adapter", dest="parent_adapter", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=6144)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if args.batch_size != 1:
        p.error("This bounded implementation supports serial batch-size 1 only")
    if any(getattr(args, k) < 1 for k in ("max_steps", "gradient_accumulation", "max_length")):
        p.error("Step, accumulation, and token budgets must be positive")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0
           for k in ("learning_rate", "beta", "max_grad_norm")):
        p.error("Learning rate, beta, and gradient norm must be finite and positive")
    return args


def row_schedule(size, count, seed):
    """A seeded permutation per pass, shared verbatim by the two objectives."""
    if size < 1 or count < 1:
        raise ValueError("Empty data or exposure budget")
    rng, result = random.Random(seed), []
    while len(result) < count:
        indices = list(range(size))
        rng.shuffle(indices)
        result.extend(indices)
    return result[:count]


def completion_window(prompt_length, completion_length):
    """Logits at p-1 predict first completion token; EOS is also supervised."""
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("Prompt and completion must both be nonempty")
    return slice(prompt_length - 1, prompt_length + completion_length - 1)


def completion_logps(model, prompt_ids, completion_ids, *, torch_module=None):
    """Sum causal log probabilities for completion only (no length averaging)."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    window = completion_window(len(prompt_ids), len(completion_ids))
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    # Slice before FP32 conversion: do not allocate FP32 prompt x full-vocab logits.
    logits = output.logits[:, window, :].float()
    targets = ids[:, len(prompt_ids):].unsqueeze(-1)
    values = torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)
    if values.shape[-1] != len(completion_ids):
        raise RuntimeError("Completion log-probability shift/length mismatch")
    return values.sum(dim=-1).squeeze(0)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, *, functional=None):
    if functional is None:
        from torch.nn import functional
    margin = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -functional.logsigmoid(beta * margin)


def activate(model, name, *, training):
    """PEFT set_adapter can re-enable gradients: always reapply an explicit mask."""
    model.set_adapter(name)
    selected = []
    for key, parameter in model.named_parameters():
        trainable = training and name == "policy" and "lora_" in key and ".policy." in key
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    model.train(training)
    if training and not selected:
        raise RuntimeError("No trainable policy LoRA weights after adapter switch")
    return selected


def parent_receipt(path, base_model):
    from kg_agent.choice_backend import check_adapter_contract
    parent = Path(path)
    # Retain both stage capabilities, and require evaluation of entity drift later.
    contract = check_adapter_contract(parent, "entities", base_model=base_model)
    check_adapter_contract(parent, "relations", base_model=base_model)
    payload = (parent.parent / "run_manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != contract.get("training_manifest_sha256"):
        raise ValueError("Parent training manifest does not match its adapter receipt")
    receipt = json.loads(payload)
    fingerprint = receipt.get("saved_adapter_sha256_verified")
    if (receipt.get("status") != "training_complete_evaluation_pending"
            or receipt.get("saved_reload_exact") is not True
            or fingerprint != receipt.get("policy_adapter_sha256_after")
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in fingerprint)):
        raise ValueError("Parent must have completed training with exact saved/reloaded weights")
    return contract, receipt, hashlib.sha256(payload).hexdigest()


def validate_encoded(item, eos_token_id):
    for key in ("prompt_ids", "chosen_ids", "rejected_ids"):
        ids = item.get(key)
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f"Invalid compiler token list: {key}")
        if key != "prompt_ids" and (ids[-1] != eos_token_id or ids.count(eos_token_id) != 1):
            raise ValueError("Each completion must contain exactly one terminal EOS")
    if item["chosen_ids"] == item["rejected_ids"]:
        raise ValueError("Preference alternatives must differ")
    if item.get("stage") != "relations":
        raise ValueError("This first preference release supports relations only")


def exposure_delta(item, mode):
    p, c, r = (len(item[k]) for k in ("prompt_ids", "chosen_ids", "rejected_ids"))
    return Counter(rows=1, relations_rows=1, successful_backward_microsteps=1,
                   chosen_completion_tokens=c, rejected_completion_tokens=r if mode == "dpo" else 0,
                   policy_completion_loss_tokens=c + (r if mode == "dpo" else 0),
                   policy_prompt_context_tokens=p * (2 if mode == "dpo" else 1),
                   prompt_loss_tokens=0)


def train_loop(model, encoded, schedule, args, audit, save):
    import torch
    from generative.train import adapter_fingerprint
    parameters = activate(model, "policy", training=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    observed, reference_exposure = Counter(), Counter()
    reference_cache, losses, updates = {}, [], 0
    try:
        for microstep, index in enumerate(schedule):
            item = encoded[index]
            if args.mode == "dpo" and index not in reference_cache:
                activate(model, "reference", training=False)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    rc = completion_logps(model, item["prompt_ids"], item["chosen_ids"])
                    rr = completion_logps(model, item["prompt_ids"], item["rejected_ids"])
                reference_cache[index] = (float(rc), float(rr))
                if not all(math.isfinite(v) for v in reference_cache[index]):
                    raise RuntimeError("Nonfinite frozen reference log probabilities")
                reference_exposure.update(rows=1, sequences=2,
                    prompt_context_tokens=2 * len(item["prompt_ids"]),
                    completion_scored_tokens=len(item["chosen_ids"]) + len(item["rejected_ids"]))
            activate(model, "policy", training=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pc = completion_logps(model, item["prompt_ids"], item["chosen_ids"])
                if args.mode == "dpo":
                    pr = completion_logps(model, item["prompt_ids"], item["rejected_ids"])
                    rc_value, rr_value = reference_cache[index]
                    if microstep == 0:
                        error = max(abs(float(pc.detach()) - rc_value), abs(float(pr.detach()) - rr_value))
                        audit["initial_policy_reference_logprob_max_abs_error"] = error
                        # Eval versus training checkpoint paths can differ by tiny reduction error.
                        if error > 0.05:
                            raise RuntimeError("Initial policy/reference scoring mismatch")
                    loss = dpo_loss(pc, pr, rc_value, rr_value, args.beta)
                else:
                    loss = -pc / len(item["chosen_ids"])
            scalar_loss = float(loss.detach())
            if not math.isfinite(scalar_loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / args.gradient_accumulation).backward()
            observed.update(exposure_delta(item, args.mode))
            losses.append(scalar_loss)
            if (microstep + 1) % args.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                audit.update(optimizer_updates=updates, global_step=updates,
                             observed_exposure=dict(observed), reference_exposure=dict(reference_exposure),
                             last_gradient_norm=float(norm), train_loss=sum(losses) / len(losses))
                save()
                print(json.dumps({"step": updates, "mode": args.mode,
                    "loss": sum(losses[-args.gradient_accumulation:]) / args.gradient_accumulation}), flush=True)
        if updates != args.max_steps:
            raise RuntimeError("Optimizer update budget was not completed")
        if args.mode == "dpo":
            final_reference = adapter_fingerprint(model, "reference")
            if final_reference != audit["policy_adapter_sha256_before"]:
                raise RuntimeError("Frozen reference changed during training")
            audit.update(reference_adapter_sha256_after=final_reference, reference_unchanged=True)
    finally:
        audit.update(observed_exposure=dict(observed), reference_exposure=dict(reference_exposure),
                     optimizer_updates=updates, global_step=updates)
        save()


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new output directory; prior runs are immutable")
    from generative.choice_preferences import encode_preference_row
    from kg_agent.choice_backend import (CONTRACT, adapter_files_digest,
        base_input_digest, protocol_implementation_digest)
    from transformers import AutoTokenizer
    contract, parent, parent_hash = parent_receipt(args.parent_adapter, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    payload = Path(args.data).read_bytes()
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty source or repeated preference IDs")
    encoded = [encode_preference_row(r, tokenizer, args.max_length, require_train=True) for r in rows]
    for item in encoded:
        validate_encoded(item, tokenizer.eos_token_id)
    schedule = row_schedule(len(rows), args.max_steps * args.gradient_accumulation, args.seed)
    ordered_ids = [rows[i]["id"] for i in schedule]
    audit = dict(status="preflight_complete", scope="artificial_contract_only_not_medical",
        output_contract=CONTRACT, args=vars(args), stages=contract["stages"], training_stages=["relations"],
        completion_encoding="independent_integer_newline_tokens_then_eos",
        protocol_sha256=protocol_implementation_digest(), base_input_sha256=base_input_digest(args.base_model),
        parent_manifest_sha256=parent_hash, parent_adapter_sha256=contract["adapter_sha256"],
        expected_parent_parameter_sha256=parent["saved_adapter_sha256_verified"],
        data_sha256=hashlib.sha256(payload).hexdigest(), source_rows=len(rows),
        scheduled_rows=len(schedule), scheduled_unique_rows=len(set(schedule)),
        row_schedule_sha256=hashlib.sha256(json.dumps(ordered_ids).encode()).hexdigest(),
        paired_chosen_encoding_sha256=hashlib.sha256(json.dumps([
            [rows[i]["id"], encoded[i]["prompt_ids"], encoded[i]["chosen_ids"]] for i in schedule]).encode()).hexdigest(),
        max_total_tokens=max(len(e["prompt_ids"]) + max(len(e["chosen_ids"]), len(e["rejected_ids"])) for e in encoded),
        objective="completion_sum_logprob_sigmoid_dpo" if args.mode == "dpo" else "chosen_completion_mean_nll",
        fairness="Same parent, chosen tokens, row order, optimizer updates/lr; DPO also scores rejected and frozen reference",
        optimizer="AdamW; weight_decay=0; constant learning rate; fresh optimizer; no dropout",
        exposure_counting_version="successful_backward_microsteps_v1", medical_quality_evaluated=False,
        entity_drift_evaluation_required=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_manifest.json"
    def save():
        manifest.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    save()
    if args.preflight_only:
        print(json.dumps(audit), flush=True)
        return
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from peft import PeftModel
    from generative.train import adapter_fingerprint, initialize_dpo_reference, require_saved_weight_equality
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This pinned probe requires a BF16 CUDA GPU")
        set_seed(args.seed)
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs), args.parent_adapter,
            adapter_name="policy", is_trainable=True, local_files_only=True)
        before = adapter_fingerprint(model, "policy")
        if before != parent["saved_adapter_sha256_verified"]:
            raise RuntimeError("Loaded policy does not equal the parent trained weights")
        if args.mode == "dpo":
            audit.update(initialize_dpo_reference(model, args.parent_adapter))
            reference_before = adapter_fingerprint(model, "reference")
            if reference_before != before:
                raise RuntimeError("Initial frozen reference differs from policy")
            audit["reference_adapter_sha256_before"] = reference_before
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        audit.update(status="training", policy_adapter_sha256_before=before,
                     parent_initialization_verified=True, optimizer_updates=0)
        save()
        train_loop(model, encoded, schedule, args, audit, save)
        activate(model, "policy", training=False)
        after = adapter_fingerprint(model, "policy")
        if after == before:
            raise RuntimeError("Training did not change policy adapter weights")
        if any(not torch.isfinite(p).all().item() for n, p in model.named_parameters() if ".policy." in n and "lora_" in n):
            raise RuntimeError("Nonfinite policy adapter parameters")
        export = out / "_adapter_export"
        model.save_pretrained(export, selected_adapters=["policy"], safe_serialization=True)
        final = out / "final"
        (export / "policy").replace(final)
        tokenizer.save_pretrained(final)
        audit.update(status="verifying_saved_adapter", policy_adapter_sha256_after=after,
                     policy_weights_changed=True, adapter_sha256=adapter_files_digest(final), final_adapter=str(final))
        save()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs),
                                             str(final), local_files_only=True)
        require_saved_weight_equality(reloaded, after)
        audit.update(status="training_complete_evaluation_pending", saved_reload_exact=True,
                     saved_adapter_sha256_verified=after)
        save()
        receipt = {key: audit[key] for key in ("output_contract", "stages", "completion_encoding",
            "protocol_sha256", "adapter_sha256", "base_input_sha256", "scope")}
        receipt.update(training_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                       training_data_sha256=audit["data_sha256"], training_stages=audit["training_stages"],
                       parent_adapter_sha256=audit["parent_adapter_sha256"], medical_quality_evaluated=False)
        (final / "choice_contract.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(audit), flush=True)
    except BaseException:
        audit.update(status="failed", error_traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()

"""Train targeted semantic preferences from the exact completed canonical SFT.

A small GPU smoke is explicitly non-evaluable. Formal training completes every
admitted preference for the declared epochs and checks frozen reference, masks,
EOS, saved weights and a fresh reload before publishing an evaluable adapter.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from generative.chat import render_prompt
from semantic import protocol as p
from semantic.data import canonical, encode_row, sha, validate_row
import build_pairs as pairs
VERSION = "semantic-targeted-dpo52-v2"
LIMITATIONS = [
    "52 preferences: 36 source-reviewed semantic contradictions and 16 outside-table references.",
    "Chosen retains complete frozen silver; no new clinical truth or exhaustive negatives are claimed.",
    "Original source188 is excluded from this preference release; the unchanged SFT parent may contain original silver defects.",
    "Data composition, update count and learning rate all differ from DPO208; no single-factor attribution.",
    "Mechanical holdout is from original TRAIN, not an unseen clinical test set.",
    "Full matched pipeline evaluation is required; training loss is not extraction accuracy.",
]
LOCKS = dict(pairs.LOCKS,
    parent_manifest="dbb8342e60fbd192f570578c458a825a2c8e6808ea78e094c6963e09ee5f3551",
    parent_parameters="2dc005293f7ef2ffe7cb3de9253117001417c3066115a0624c162f3b7311e9ab",
    parent_data="4d51e1c99473c5fb392e5dd78b01e7a0ff3024332a68f3b9974fbcbdbd0c1d13")
encode_pair = pairs.encode_pair


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locked_bytes(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, "Frozen SHA mismatch: " + str(path))
    return raw


def rows_from(raw):
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    require(bool(rows), "Empty row file")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate row identity")
    return rows


def atomic_json(path, value, *, before_replace=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(canonical(value) + "\n")
        temporary.chmod(0o600)
        # Final publication checks the deadline AFTER serialization/disk write,
        # immediately before the atomic rename. Ordinary progress saves omit it.
        if before_replace is not None:
            before_replace()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def finalize_receipt(out, receipt, *, base_model, check_deadline,
                     verify_final_inputs, started, clock=time.monotonic,
                     wall_clock=time.time):
    """Validate an isolated candidate, then publish completion exactly once.

    The official path stays incomplete through the unmodified runtime checker,
    temporary-directory cleanup, and the final source/adapter checks. A SIGKILL
    during those steps leaves that incomplete receipt; handled failures record
    failed=True. This helper performs no Torch work and is fault-testable on CPU.
    """
    from semantic.runtime import check_adapter_contract

    out = Path(out)
    official = out / "training_manifest.json"
    require(receipt.get("completed") is False and receipt.get("method") == "dpo",
            "Finalization requires an incomplete honest DPO receipt")
    require(json.loads(official.read_text()) == receipt,
            "Official receipt changed before finalization")
    try:
        receipt.update(status="verifying_runtime_contract",
                       current_runtime_contract_check_passed=False)
        atomic_json(official, receipt)
        check_deadline()
        # A sibling directory cannot be mistaken for the official output. Hard
        # links preserve the exact saved adapter bytes without duplicating them.
        # The checker is read-only; final input verification rehashes the official
        # adapter after the candidate and all temporary files have been removed.
        with tempfile.TemporaryDirectory(prefix=".dpo-contract-candidate-", dir=out.parent) as candidate_dir:
            candidate_dir = Path(candidate_dir)
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                os.link(out / name, candidate_dir / name)
            candidate = deepcopy(receipt)
            candidate.update(completed=True, failed=False,
                             status="isolated_contract_validation_candidate")
            atomic_json(candidate_dir / "training_manifest.json", candidate)
            checked = check_adapter_contract(candidate_dir, "relations", base_model=base_model)
            require(checked == candidate, "Runtime contract checker returned a different receipt")
        verify_final_inputs()
        require(json.loads(official.read_text()) == receipt,
                "Official receipt changed during finalization")
        check_deadline()
        completed = deepcopy(receipt)
        completed.update(completed=True, failed=False,
                         status="complete_preference_epochs_evaluation_pending",
                         current_runtime_contract_check_passed=True,
                         finished_unix=wall_clock(), elapsed_seconds=clock() - started)
        # No validation or success-status write follows this atomic publication.
        # A write that crosses the deadline is discarded before replacement.
        atomic_json(official, completed, before_replace=check_deadline)
        receipt.clear()
        receipt.update(completed)
    except BaseException as exc:
        receipt.update(completed=False, failed=True, status="failed",
                       current_runtime_contract_check_passed=False,
                       error_type=type(exc).__name__, error=str(exc))
        atomic_json(official, receipt)
        raise
    return receipt


def dependency_locks():
    import generative.chat as chat
    import generative.train as train_helpers
    import semantic.data as data
    import semantic.runtime as runtime
    return {str(Path(module.__file__).resolve()): sha(Path(module.__file__).read_bytes())
            for module in (chat, train_helpers, data, runtime, p, pairs)}


def validate_parent_receipt(receipt):
    require(receipt.get("contract") == p.CONTRACT and receipt.get("stage") == "relations",
            "Parent is not a semantic relation adapter")
    require(receipt.get("variant") == "short_canonical", "Parent input/order differs from candidates")
    require(receipt.get("method", "sft") == "sft", "Reference parent must be the corresponding SFT")
    require(receipt.get("completed") is True and receipt.get("smoke_only") is False
            and receipt.get("failed") is not True, "Parent must be completed non-smoke SFT")
    require(receipt.get("saved_weights_bitwise_verified") is True, "Parent disk weights were not verified")
    require(receipt.get("adapter_final_sha256") == LOCKS["parent_parameters"], "Wrong parent parameter fingerprint")
    require(receipt.get("data_sha256") == LOCKS["parent_data"], "Wrong parent training release")
    require(receipt.get("protocol_sha256") == LOCKS["protocol"], "Parent protocol drift")
    require(receipt.get("observed_exposure", {}).get("all_rows_seen") is True,
            "Parent did not complete its released data")


def check_parent(path, base_model):
    # Production checker is read-only and rejects incomplete/smoke/wrong-stage
    # adapters. We do not change or monkeypatch it in the executable entry point.
    from semantic.runtime import check_adapter_contract
    raw = locked_bytes(Path(path) / "training_manifest.json", LOCKS["parent_manifest"])
    receipt = json.loads(raw)
    validate_parent_receipt(receipt)
    checked = check_adapter_contract(path, "relations", base_model=base_model)
    require(checked == receipt, "Parent changed during admission")
    return receipt


def epoch_schedule(size, epochs, seed, accumulation):
    require(type(epochs) is int and 1 <= epochs <= 3, "Supports 1 to 3 explicit complete preference epochs")
    require(type(accumulation) is int and accumulation >= 1 and size % accumulation == 0,
            "Accumulation must divide the admitted rows; no partial/omitted updates")
    rng, schedule = random.Random(seed), []
    for _ in range(epochs):
        order = list(range(size)); rng.shuffle(order); schedule.extend(order)
    return schedule


def completion_window(prompt_length, completion_length):
    require(prompt_length > 0 and completion_length > 0, "Nonempty prompt/completion required")
    return slice(prompt_length - 1, prompt_length + completion_length - 1)


def completion_logps(model, prompt_ids, completion_ids, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = output.logits[:, completion_window(len(prompt_ids), len(completion_ids)), :].float()
    labels = ids[:, len(prompt_ids):].unsqueeze(-1)
    values = torch.nn.functional.log_softmax(logits, dim=-1).gather(-1, labels).squeeze(-1)
    require(values.shape[-1] == len(completion_ids), "Causal completion shift mismatch")
    return values.sum(dim=-1).squeeze(0)


def dpo_loss(pc, pr, rc, rr, beta, *, functional=None):
    if functional is None:
        from torch.nn import functional
    return -functional.logsigmoid(beta * ((pc - pr) - (rc - rr)))


def activate(model, name, *, training):
    require(name in ("policy", "reference") and not (name == "reference" and training), "Reference is never trainable")
    model.set_adapter(name)
    selected = []
    for key, parameter in model.named_parameters():
        trainable = training and "lora_" in key and ".policy." in key
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    model.train(training)
    require(not training or bool(selected), "No policy LoRA parameters")
    return selected


def require_reference_unchanged(model, expected, *, fingerprint=None):
    if fingerprint is None:
        from generative.train import adapter_fingerprint as fingerprint
    actual = fingerprint(model, "reference")
    require(actual == expected, "Frozen SFT reference changed")
    return actual


def exposure(encoded, visits):
    counts = [visits.get(i, 0) for i in range(len(encoded))]
    return {"pairs": len(encoded), "unique_pairs_seen": sum(c > 0 for c in counts),
        "successful_backward_microsteps": sum(counts), "min_visits": min(counts), "max_visits": max(counts),
        "chosen_completion_loss_tokens": sum(c * len(e["chosen_ids"]) for e, c in zip(encoded, counts)),
        "rejected_completion_loss_tokens": sum(c * len(e["rejected_ids"]) for e, c in zip(encoded, counts)),
        "prompt_context_token_exposures": sum(2 * c * len(e["prompt_ids"]) for e, c in zip(encoded, counts)),
        "prompt_loss_tokens": 0, "eos_scored_per_preference_visit": 2,
        "scope": "Successful policy backward calls; reference-only forwards reported separately"}


def require_complete_exposure(encoded, visits, epochs, updates, accumulation):
    require(set(visits) <= set(range(len(encoded))), "Unknown training row was exposed")
    result = exposure(encoded, visits)
    require(result["min_visits"] == result["max_visits"] == epochs, "Incomplete full preference epochs")
    require(updates == len(encoded) * epochs // accumulation, "Incomplete optimizer budget")
    return result


def verify_saved_tensors(disk, live, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    require(disk.keys() == live.keys(), "Saved adapter differs bitwise from trained tensors")
    for name in disk:
        actual, expected = disk[name], live[name].detach().cpu()
        # torch.equal alone treats some equal-valued different dtypes (and
        # signed zero) as equal. Compare dtype/shape and the byte representation.
        require(actual.dtype == expected.dtype and actual.shape == expected.shape
                and torch_module.equal(actual.contiguous().view(torch_module.uint8),
                                       expected.contiguous().view(torch_module.uint8)),
                "Saved adapter differs bitwise from trained tensors")


def train(args, encoded, tokenizer, admission, parent):
    # This function is never invoked by audit-data or preflight.
    require(args.execute is True, "GPU training requires explicit --execute")
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from generative.train import adapter_fingerprint, initialize_dpo_reference
    from kg_agent.choice_backend import adapter_files_digest, base_input_digest
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "safetensors")}
    require(versions["torch"].startswith("2.8.0") and versions["transformers"] == "4.56.2"
            and versions["peft"] == "0.17.1", "Use the pinned medicalner runtime")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.device_count() == 1,
            "Expose exactly one authorized BF16 CUDA GPU via the existing queue lock")
    schedule = epoch_schedule(len(encoded), args.epochs, args.seed, args.gradient_accumulation)
    out = Path(args.output_dir)
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "Output must be new/empty")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    started, wall = time.monotonic(), time.time()
    deadline = started + args.max_seconds
    visits, reference_cache, updates = Counter(), {}, 0
    receipt = {"contract": p.CONTRACT, "stage": "relations", "variant": "short_canonical", "method": "dpo",
        "implementation_version": VERSION, "scope": "complete_epochs_over_admitted_targeted_preferences_not_full_corpus_sft",
        "completed": False, "failed": False, "smoke_only": args.smoke_only, "status": "initializing",
        "medical_quality_evaluated": False, "base_model": args.base_model,
        "base_input_sha256": base_input_digest(args.base_model), "base_input_hash_scope": "config_and_tokenizer_not_base_weights",
        "protocol_sha256": LOCKS["protocol"], "trainer_sha256": sha(Path(__file__).read_bytes()),
        "data_sha256": admission["train_data_sha256"], "admission": admission, "parameters": vars(args),
        "parent_adapter": args.parent_adapter, "parent_manifest_sha256": LOCKS["parent_manifest"],
        "parent_parameters_sha256": LOCKS["parent_parameters"], "parent_sft_data_sha256": parent["data_sha256"],
        "objective": "completion_sum_logprob_sigmoid_dpo_fixed_corresponding_sft_reference",
        "dropout": 0.0, "optimizer": "AdamW_policy_LoRA_only_constant_lr_weight_decay_0",
        "row_schedule_sha256": sha([encoded[i]["id"] for i in schedule]), "epochs": args.epochs,
        "planned_optimizer_updates": len(schedule) // args.gradient_accumulation,
        "environment": versions, "started_unix": wall, "limitations": LIMITATIONS}

    def save():
        receipt.update(observed_exposure=exposure(encoded, visits), global_step=updates,
                       elapsed_seconds=time.monotonic() - started)
        atomic_json(out / "training_manifest.json", receipt)

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("Bounded DPO deadline expired; incomplete run cannot be evaluated/ranked")

    def interrupted(signum, frame):
        raise InterruptedError("DPO interrupted by SIGTERM")

    old_term = signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        set_seed(args.seed)
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs),
            args.parent_adapter, adapter_name="policy", is_trainable=True, local_files_only=True)
        initial = adapter_fingerprint(model, "policy")
        if initial != LOCKS["parent_parameters"]:
            model.load_adapter(args.parent_adapter, adapter_name="policy", is_trainable=True, local_files_only=True)
            initial = adapter_fingerprint(model, "policy")
        require(initial == LOCKS["parent_parameters"], "Loaded policy differs from exact parent SFT")
        receipt["reference_initialization"] = initialize_dpo_reference(model, args.parent_adapter)
        receipt["reference_before_sha256"] = require_reference_unchanged(model, initial)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        del module
        require(args.max_length <= model.config.max_position_embeddings, "Token budget exceeds base context")
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to("cuda")
        require(adapter_fingerprint(model, "policy") == initial, "CUDA transfer changed policy weights")
        require_reference_unchanged(model, initial)
        receipt.update(status="reference_precompute", adapter_initial_sha256=initial)
        save()
        # Same whole literal completion and precision on both branches. Ref
        # cache cannot be reused across parent/data/tokenizer changes.
        activate(model, "reference", training=False)
        receipt["reference_exposure"] = {"pairs": 0, "sequences": 0, "completion_scored_tokens": 0,
            "prompt_context_tokens": 0, "scope": "Completed no-grad reference scoring forwards; zero optimizer exposure"}
        for index, item in enumerate(encoded):
            check_deadline()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                rc = float(completion_logps(model, item["prompt_ids"], item["chosen_ids"]))
                receipt["reference_exposure"]["sequences"] += 1
                receipt["reference_exposure"]["prompt_context_tokens"] += len(item["prompt_ids"])
                receipt["reference_exposure"]["completion_scored_tokens"] += len(item["chosen_ids"])
                rr = float(completion_logps(model, item["prompt_ids"], item["rejected_ids"]))
                receipt["reference_exposure"]["sequences"] += 1
                receipt["reference_exposure"]["prompt_context_tokens"] += len(item["prompt_ids"])
                receipt["reference_exposure"]["completion_scored_tokens"] += len(item["rejected_ids"])
            require(math.isfinite(rc) and math.isfinite(rr), "Non-finite reference scores")
            reference_cache[index] = (rc, rr)
            receipt["reference_exposure"]["pairs"] += 1
            if (index + 1) % 8 == 0:
                save()
        activate(model, "policy", training=False)
        zero_checks = []
        for index in sorted({schedule[0], max(range(len(encoded)), key=lambda i: len(encoded[i]["prompt_ids"]) + len(encoded[i]["chosen_ids"]))}):
            check_deadline()
            item = encoded[index]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pc = float(completion_logps(model, item["prompt_ids"], item["chosen_ids"]))
                pr = float(completion_logps(model, item["prompt_ids"], item["rejected_ids"]))
            rc, rr = reference_cache[index]
            error = max(abs(pc - rc), abs(pr - rr))
            require(math.isfinite(error) and error <= 0.05, "Zero-step policy/reference scoring mismatch")
            zero_loss = -math.log(1.0 / (1.0 + math.exp(-args.beta * ((pc - pr) - (rc - rr)))))
            require(abs(zero_loss - math.log(2)) < 0.01, "Initial DPO objective differs from log(2)")
            zero_checks.append({"pair_id": item["id"], "logprob_max_abs_error": error, "initial_dpo_loss": zero_loss})
        receipt["zero_step_checks"] = zero_checks
        receipt["zero_step_policy_forward_exposure"] = {
            "sequences": 2 * len(zero_checks),
            "scope": "Extra no-grad policy/reference equality checks; not training loss exposure"}
        parameters = activate(model, "policy", training=True)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        receipt["status"] = "training"
        save()
        losses = []
        for microstep, index in enumerate(schedule):
            check_deadline()
            item = encoded[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pc = completion_logps(model, item["prompt_ids"], item["chosen_ids"])
                pr = completion_logps(model, item["prompt_ids"], item["rejected_ids"])
                rc, rr = reference_cache[index]
                loss = dpo_loss(pc, pr, rc, rr, args.beta)
            scalar = float(loss.detach())
            require(math.isfinite(scalar), "Non-finite DPO loss")
            (loss / args.gradient_accumulation).backward()
            visits[index] += 1
            losses.append(scalar)
            if (microstep + 1) % args.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                updates += 1
                receipt.update(last_gradient_norm=float(norm), train_loss=sum(losses) / len(losses))
                save()
                print(canonical({"step": updates, "loss": sum(losses[-args.gradient_accumulation:]) / args.gradient_accumulation}), flush=True)
        require_complete_exposure(encoded, visits, args.epochs, updates, args.gradient_accumulation)
        receipt["reference_after_sha256"] = require_reference_unchanged(model, initial)
        require(all(not parameter.requires_grad and parameter.grad is None for name, parameter in model.named_parameters()
                    if ".reference." in name or "lora_" not in name), "Base/reference has gradients")
        final = adapter_fingerprint(model, "policy")
        require(final != initial, "Policy did not update")
        require(all(torch.isfinite(parameter).all().item() for name, parameter in model.named_parameters()
                    if "lora_" in name and ".policy." in name), "Non-finite final policy")
        check_deadline()
        activate(model, "policy", training=False)
        first = encoded[schedule[0]]["prompt_ids"]
        inputs = torch.tensor([first], dtype=torch.long, device=model.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            expected_logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[:, -1, :].float().cpu()
        model.save_pretrained(out / "_export", selected_adapters=["policy"], safe_serialization=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (out / "_export" / "policy" / name).replace(out / name)
        tokenizer.save_pretrained(out)
        disk = load_file(str(out / "adapter_model.safetensors"), device="cpu")
        live = get_peft_model_state_dict(model, adapter_name="policy")
        verify_saved_tensors(disk, live, torch_module=torch)
        receipt.update(status="verifying_fresh_reload", adapter_final_sha256=final,
            adapter_files_sha256=adapter_files_digest(out), saved_weights_bitwise_verified=True,
            policy_weights_changed=True, reference_unchanged=True)
        save()
        del live, disk, optimizer, parameters, model, pc, pr, loss
        gc.collect(); torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs),
            str(out), adapter_name="policy", is_trainable=False, local_files_only=True)
        if adapter_fingerprint(reloaded, "policy") != final:
            reloaded.load_adapter(str(out), adapter_name="policy", is_trainable=False, local_files_only=True)
        require(adapter_fingerprint(reloaded, "policy") == final, "Fresh reload differs from final weights")
        reloaded.to("cuda").eval()
        reloaded.set_adapter("policy")
        require(adapter_fingerprint(reloaded, "policy") == final, "Reload CUDA transfer changed weights")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            actual = reloaded(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits[:, -1, :].float().cpu()
        delta = float((actual - expected_logits).abs().max())
        require(math.isfinite(delta) and delta <= 0.02, "Fresh plain-inference logits differ")
        receipt.update(saved_reload_exact=True, saved_reload_parameter_sha256=final,
            save_reload_max_logit_difference=delta,
            save_reload_scope="Exact adapter fingerprint; first scheduled prompt last-token plain logits only",
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
        def verify_final_inputs():
            for path, expected in admission["input_file_locks"].items():
                locked_bytes(path, expected)
            for path, expected in admission["dependency_source_locks"].items():
                locked_bytes(path, expected)
            check_parent(args.parent_adapter, args.base_model)
            require(sha(Path(__file__).read_bytes()) == receipt["trainer_sha256"], "Training code changed during run")
            locked_bytes(Path(p.__file__), LOCKS["protocol"])
            require(adapter_files_digest(out) == receipt["adapter_files_sha256"],
                    "Official adapter changed during final verification")
        # Persist latest verification/exposure fields while still incomplete.
        save()
        if args.smoke_only:
            verify_final_inputs()
            check_deadline()
            receipt.update(completed=True, failed=False, status="gpu_smoke_complete_not_evaluable",
                current_runtime_contract_check_passed=False, finished_unix=time.time(),
                elapsed_seconds=time.monotonic() - started)
            atomic_json(out / "training_manifest.json", receipt, before_replace=check_deadline)
        else:
            finalize_receipt(out, receipt, base_model=args.base_model,
                             check_deadline=check_deadline,
                             verify_final_inputs=verify_final_inputs, started=started)
    except BaseException as exc:
        receipt.update(completed=False, failed=True, status="failed",
                       error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
    return receipt


def admit(candidates_root, base_release, overlap_release, tokenizer_root, *, max_length=4096):
    from transformers import AutoTokenizer
    root = Path(candidates_root)
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    require(manifest["data_version"] == pairs.VERSION and manifest["parent_view"] == "short_canonical",
            "Wrong candidate parent/view")
    require(manifest["protocol_sha256"] == LOCKS["protocol"], "Candidate protocol changed")
    sources, donors, full, file_locks = pairs.load_sources(base_release, overlap_release)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, trust_remote_code=False)
    encoded, groups, observed = [], {}, []
    # Manifest source paths can differ between local and resident reconstruction;
    # current base/overlap releases are always independently hashed above.
    for split, name in (("train", "candidate_train.jsonl"), ("mechanical_holdout", "mechanical_holdout.jsonl")):
        blob = locked_bytes(root / name, manifest["output_sha256"][name])
        rows = rows_from(blob)
        require(len(rows) == manifest["train_pairs" if split == "train" else "mechanical_holdout_pairs"],
                "Incomplete preference partition")
        groups[split] = set()
        for pair in rows:
            require(pair["split"] == split, "Partition file changed")
            pairs.verify_pair(pair, sources, donors, full, require_train=False)
            item = encode_pair(pair, tokenizer, max_length, require_train=False)
            groups[split].add(str(item["group_id"]))
            observed.append(pair["id"])
            if split == "train":
                encoded.append(item)
        file_locks[str((root / name).resolve())] = manifest["output_sha256"][name]
    require(len(observed) == len(set(observed)), "Duplicate preference identity across partitions")
    require(not groups["train"] & groups["mechanical_holdout"], "Train/holdout leakage")
    file_locks[str((root / "manifest.json").resolve())] = sha(raw)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = Path(tokenizer_root) / name
        file_locks[str(path.resolve())] = sha(path.read_bytes())
    report = {"status": "cpu_data_verified_gpu_pending", "method": "dpo",
        "candidate_manifest_sha256": sha(raw), "train_data_sha256": manifest["output_sha256"]["candidate_train.jsonl"],
        "train_pairs": len(encoded), "mechanical_holdout_pairs_excluded_from_training": manifest["mechanical_holdout_pairs"],
        "train_source_groups": len(groups["train"]), "mechanical_holdout_source_groups": len(groups["mechanical_holdout"]),
        "cross_role_group_overlap": 0, "no_truncation": True,
        "max_prompt_plus_completion_tokens": max(len(v["prompt_ids"]) + max(len(v["chosen_ids"]), len(v["rejected_ids"])) for v in encoded),
        "prompt_tokens_one_pass": sum(len(v["prompt_ids"]) for v in encoded),
        "chosen_tokens_one_pass_including_eos": sum(len(v["chosen_ids"]) for v in encoded),
        "rejected_tokens_one_pass_including_eos": sum(len(v["rejected_ids"]) for v in encoded),
        "prompt_loss_tokens": 0, "chosen_sft_token_mask_identity_checked": True,
        "mutation_counts": manifest["mutation_counts"], "input_file_locks": file_locks,
        "dependency_source_locks": dependency_locks(), "limitations": LIMITATIONS}
    return encoded, tokenizer, report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit-data", "preflight", "train"))
    for name in ("candidates-root", "base-release", "overlap-release", "tokenizer", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--parent-adapter")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 2 or args.gradient_accumulation < 1:
        parser.error("Positive token/accumulation limits required")
    if args.action != "audit-data":
        if not args.base_model or not args.parent_adapter:
            parser.error("Exact semantic parent and base required")
        if args.epochs not in (1, 2, 3) or args.learning_rate is None or args.beta is None:
            parser.error("Explicit epochs (1–3), learning rate and beta required")
        if not all(math.isfinite(v) and v > 0 for v in (args.learning_rate, args.beta, args.max_seconds)) or args.learning_rate >= 1:
            parser.error("Finite positive optimizer/deadline values required")
    if args.action == "train" and not args.execute:
        parser.error("GPU training requires --execute")
    if args.action != "train" and (args.execute or args.smoke_only):
        parser.error("Execution/smoke flags apply only to train")
    output = Path(args.output_dir).resolve()
    for protected in (args.candidates_root, args.base_release, args.overlap_release, args.base_model, args.parent_adapter):
        if protected:
            path = Path(protected).resolve()
            if output == path or output.is_relative_to(path):
                parser.error("Output cannot modify source, parent or base")
    return args


def smoke_indices(encoded, train_pairs, count=4):
    """Cover the real maximum and every present mutation/subkind without repeats."""
    require(len(encoded) == len(train_pairs) and len(encoded) >= count, "Insufficient smoke rows")
    require(len({v["id"] for v in encoded}) == len(encoded), "Duplicate preference identities")
    require(all(v["id"] == pair["id"] for v, pair in zip(encoded, train_pairs)), "Smoke alignment drift")
    def category(pair):
        mutation = pair["mutation"]
        kind = mutation["kind"]
        if kind == "reviewed_semantic_contradiction":
            require(mutation.get("semantic_change_kind") in ("explicit_negation_label_flip", "subtype_direction_reversal"),
                    "Unknown semantic smoke category")
            return kind + "/" + mutation["semantic_change_kind"]
        return kind
    kinds = sorted({category(pair) for pair in train_pairs})
    require(len(kinds) <= count, "Smoke count cannot cover all mutation kinds")
    length = lambda i: len(encoded[i]["prompt_ids"]) + max(len(encoded[i]["chosen_ids"]), len(encoded[i]["rejected_ids"]))
    ranked = sorted(range(len(encoded)), key=lambda i: (-length(i), encoded[i]["id"]))
    selected = {ranked[0]}
    for kind in kinds:
        selected.add(next(i for i in ranked if category(train_pairs[i]) == kind))
    for i in ranked:
        if len(selected) == count:
            break
        selected.add(i)
    require(len(selected) == count and ranked[0] in selected, "Smoke maximum coverage failed")
    require({category(train_pairs[i]) for i in selected} == set(kinds), "Smoke kind coverage failed")
    return sorted(selected)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())), "Output must be new/empty")
    encoded, tokenizer, admission = admit(args.candidates_root, args.base_release, args.overlap_release,
        args.tokenizer, max_length=args.max_length)
    parent = None
    if args.action != "audit-data":
        parent = check_parent(args.parent_adapter, args.base_model)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            require((Path(args.tokenizer) / name).read_bytes() == (Path(args.base_model) / name).read_bytes(),
                    "Training/base tokenizer mismatch")
        from transformers import AutoTokenizer
        actual_tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False)
        train_pairs = rows_from(locked_bytes(Path(args.candidates_root) / "candidate_train.jsonl", admission["train_data_sha256"]))
        require([encode_pair(row, actual_tokenizer, args.max_length) for row in train_pairs] == encoded,
                "Actual model tokenizer changed IDs/EOS")
        tokenizer = actual_tokenizer
        admission["actual_base_token_encoding_verified"] = True
        if args.smoke_only:
            selected = smoke_indices(encoded, train_pairs)
            require(args.epochs == 1 and args.gradient_accumulation in (1, 2, 4),
                    "Smoke is four explicit longest/category pairs for one epoch")
            encoded = [encoded[i] for i in selected]
            admission.update(smoke_subset_pair_ids=[v["id"] for v in encoded],
                smoke_subset_count=4, full_release_pairs_not_all_trained=True,
                smoke_selection="global longest, longest each present mutation/subkind, longest remaining without duplicates")
        epoch_schedule(len(encoded), args.epochs, args.seed, args.gradient_accumulation)
    if args.action == "train":
        return train(args, encoded, tokenizer, admission, parent)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    admission.update(parent_contract_checked=parent is not None, gpu_execution_performed=False,
        action=args.action, trainer_sha256=sha(Path(__file__).read_bytes()))
    if parent is not None:
        admission.update(parent_parameters_sha256=parent["adapter_final_sha256"],
            planned_optimizer_updates=len(encoded) * args.epochs // args.gradient_accumulation)
    atomic_json(out / "admission.json", admission)
    print(canonical({key: admission[key] for key in ("status", "train_pairs", "mechanical_holdout_pairs_excluded_from_training",
                                                   "max_prompt_plus_completion_tokens", "parent_contract_checked")}), flush=True)
    return admission


if __name__ == "__main__":
    main()

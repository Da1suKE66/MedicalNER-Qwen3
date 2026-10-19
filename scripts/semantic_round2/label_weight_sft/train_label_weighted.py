"""Isolated candidate: full-source relation SFT with label-token weighting.

Only the escaped semantic lines are learned. JSON is an artifact container,
never the model completion. This trainer does not certify silver completeness.
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
import re
import time

from semantic.data import canonical, encode_row, sha, validate_row
from semantic.protocol import CONTRACT
from label_weights import attach_weights, pad_weights, WeightedLossMixin

DATA_SHA='4d51e1c99473c5fb392e5dd78b01e7a0ff3024332a68f3b9974fbcbdbd0c1d13'
PARENT_MANIFEST_SHA='dbb8342e60fbd192f570578c458a825a2c8e6808ea78e094c6963e09ee5f3551'
PARENT_PARAMETERS_SHA='2dc005293f7ef2ffe7cb3de9253117001417c3066115a0624c162f3b7311e9ab'


def admit_parent(path,base_model):
    from semantic.runtime import check_adapter_contract
    manifest=Path(path)/'training_manifest.json'
    if sha(manifest.read_bytes())!=PARENT_MANIFEST_SHA:
        raise ValueError('Parent SFT manifest mismatch')
    parent=check_adapter_contract(path,'relations',base_model=base_model)
    if parent['data_sha256']!=DATA_SHA or parent['adapter_final_sha256']!=PARENT_PARAMETERS_SHA:
        raise ValueError('Parent SFT data or parameter receipt mismatch')
    return parent


def read_training_rows(path, stage):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty training release")
    seen, variants = set(), set()
    for row in rows:
        validate_row(row)
        if row["stage"] != stage or row["id"] in seen:
            raise ValueError("Mixed stages or duplicate training identity")
        seen.add(row["id"])
        variants.add(row["variant"])
    if len(variants) != 1:
        raise ValueError("Each experiment must declare one input/order variant")
    return rows


def validate_arguments(args):
    if args.stage!='relations' or not 1<=args.label_weight<=8:
        raise ValueError('Only relation label weighting1..8 is supported')
    if args.data_sha256!=DATA_SHA:
        raise ValueError('Candidate is locked to the full1000-row relation release')
    if args.parent_adapter and (args.lora_rank,args.lora_alpha,args.dropout)!=(16,32,.05):
        raise ValueError('Continuation must preserve parent adapter configuration')
    if args.epochs <= 0 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("Training counts must be positive")
    if args.max_steps != -1 and (not args.smoke_only or args.max_steps < 1):
        raise ValueError("Step-capped runs must be explicitly marked smoke-only")
    if not args.smoke_only and not float(args.epochs).is_integer():
        raise ValueError("Formal runs require a whole number of full dataset epochs")
    if not 0 < args.learning_rate < 1 or args.lora_rank < 1:
        raise ValueError("Invalid optimizer/adapter parameter")
    if args.max_length < 2 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid length or dropout")
    if not args.smoke_only and not re.fullmatch(r"[a-f0-9]{64}", getattr(args, "data_sha256", None) or ""):
        raise ValueError("Formal training requires the audited release data SHA256")


def exposure_summary(rows, counts):
    seen = [counts.get(i, 0) for i in range(len(rows))]
    return {"rows": len(rows), "unique_rows_seen": sum(n > 0 for n in seen),
            "row_visits": sum(seen), "min_visits": min(seen), "max_visits": max(seen),
            "unique_source_groups_seen": len({r["group_id"] for i, r in enumerate(rows) if seen[i]}),
            "all_rows_seen": all(n > 0 for n in seen),
            "scope": "successful training_step calls, including gradient accumulation"}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.replace(path)


def train(args):
    validate_arguments(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file
    from train_sft import CausalCollator
    from generative.train import adapter_fingerprint
    from kg_agent.choice_backend import base_input_digest, adapter_files_digest
    from semantic import protocol

    if importlib.metadata.version('transformers')!='4.56.2':
        raise ValueError('Accumulation handling was inspected for Transformers4.56.2 only')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('Only one GPU process is admitted')
    parent=admit_parent(args.parent_adapter,args.base_model) if args.parent_adapter else None

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing run")
    data_digest = sha(Path(args.data).read_bytes())
    if args.data_sha256 and data_digest != args.data_sha256:
        raise ValueError("Training file differs from the audited release")
    rows = read_training_rows(args.data, args.stage)
    if len(rows)!=1000:
        raise ValueError('Expected all1000 complete relation rows')
    if sha(Path(args.data).read_bytes()) != data_digest:
        raise ValueError("Training file changed during admission")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [attach_weights(encode_row(row,tokenizer,args.max_length),row['completion'],
                              tokenizer,args.label_weight) for row in rows]
    longest_row_index=max(range(len(encoded)),key=lambda i:len(encoded[i]['input_ids']))
    lengths = [len(item["input_ids"]) for item in encoded]
    target_counts = [item["_choice_counts"]["completion_tokens"] for item in encoded]
    label_counts=[item.pop('_label_tokens') for item in encoded]
    weighted_mass=sum(sum(item['loss_weights']) for item in encoded)
    for index, item in enumerate(encoded):
        item.pop("_choice_counts")
        item["_row_index"] = index

    class Dataset:
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    collator = CausalCollator(tokenizer)

    def tracked_collator(items):
        batch = collator(items)
        batch['loss_weights']=torch.tensor(pad_weights(items,batch['labels'].shape[1]),dtype=torch.float32)
        batch["_row_indices"] = [item["_row_index"] for item in items]
        return batch

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, device_map={"": 0})
    model.config.use_cache = False
    if args.parent_adapter:
        model=PeftModel.from_pretrained(model,args.parent_adapter,adapter_name='default',
                                      is_trainable=True,local_files_only=True)
        if adapter_fingerprint(model,'default')!=PARENT_PARAMETERS_SHA:
            model.load_adapter(args.parent_adapter,adapter_name='default',is_trainable=True,local_files_only=True)
        if adapter_fingerprint(model,'default')!=PARENT_PARAMETERS_SHA:
            raise ValueError('Actual loaded parent differs from frozen SFT parameters')
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    initial = adapter_fingerprint(model, "default")
    counts = Counter()
    observed = {"successful_microsteps": 0, "supervised_tokens": 0,'weighted_token_mass':0.,
                'relation_label_tokens':0,'longest_example_seen':False}
    started = time.time()
    receipt = {"contract": CONTRACT, "stage": args.stage, "variant": rows[0]["variant"],
               "base_model": args.base_model, "base_input_sha256": base_input_digest(args.base_model),
               "base_input_hash_scope": "config_and_tokenizer_not_base_weights",
               "data_sha256": data_digest,
               "audited_release_sha256_checked": bool(args.data_sha256),
               "protocol_sha256": sha(Path(protocol.__file__).read_bytes()),
               "trainer_sha256": sha(Path(__file__).read_bytes()),
               'weighting_helper_sha256':sha(Path(__file__).with_name('label_weights.py').read_bytes()),
               'initialization':'continued_original_SFT' if parent else 'fresh_base_LoRA',
               'parent_manifest_sha256':PARENT_MANIFEST_SHA if parent else None,
               'parent_parameters_sha256':PARENT_PARAMETERS_SHA if parent else None,
               'parent_adapter_runtime_contract_checked':bool(parent),
               'optimizer_state':'new; no resumed optimizer or scheduler',
               'loss':{'objective':'sum(weight*causal_CE)/sum(weight) over each gradient accumulation group',
                       'label_token_weight':args.label_weight,'all_other_target_tokens_weight':1.,
                       'prompt_padding_weight':0.,'EOS_weight':1.,
                       'normalization':'full_accumulation_group_weighted_token_mass'},
               "parameters": vars(args), "completed": False, "smoke_only": args.smoke_only,
               "started_unix": started, "adapter_initial_sha256": initial,
               "dataset": {"rows": len(rows), "source_groups": len({r["group_id"] for r in rows}),
                           "source_records": len({r["metadata"]["source_record_sha256"] for r in rows}),
                           "tokens_one_pass": sum(lengths), "max_tokens": max(lengths),
                           "target_tokens_one_pass": sum(target_counts), "max_target_tokens": max(target_counts),
                           'relation_label_tokens_one_pass':sum(label_counts),'weighted_token_mass_one_pass':weighted_mass,
                           "no_truncation": True, "prompt_loss_masked": True, "eos_supervised": True},
               "environment": {package: importlib.metadata.version(package) for package in ("torch", "transformers", "peft", "safetensors")},
               "selection_policy": "fixed epochs; compare checkpoints by complete frozen tune evaluation"}
    write_json(output / "training_manifest.json", receipt)

    class ObservedTrainer(WeightedLossMixin,Trainer):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.model_accepts_loss_kwargs=True

        def _get_train_sampler(self,train_dataset=None):
            if args.smoke_only:
                return torch.utils.data.SubsetRandomSampler([longest_row_index])
            return super()._get_train_sampler(train_dataset)

        def training_step(self, model, inputs, num_items_in_batch=None):
            indices = inputs.pop("_row_indices")
            supervised_tokens = int((inputs["labels"][:, 1:] != -100).sum().item())
            mass=float(inputs['loss_weights'][:,1:].sum())
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite training loss")
            counts.update(indices)
            observed["successful_microsteps"] += 1
            observed["supervised_tokens"] += supervised_tokens
            observed['weighted_token_mass']+=mass
            observed['relation_label_tokens']+=sum(label_counts[i] for i in indices)
            observed['longest_example_seen']|=longest_row_index in indices
            return loss

    class ReceiptCallback(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            for name in ("loss", "grad_norm", "learning_rate"):
                if name in (logs or {}) and not math.isfinite(float(logs[name])):
                    raise RuntimeError("Non-finite training statistic: " + name)
            progress = {"global_step": state.global_step, "epoch": state.epoch,
                        "elapsed_seconds": time.time() - started, "logs": logs or {},
                        "observed_exposure": {**exposure_summary(rows, counts), **observed}}
            write_json(output / "progress.json", progress)

    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, bf16=True, tf32=True,
        optim="adamw_torch", lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, max_grad_norm=1.0, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=args.logging_steps,
        logging_first_step=True, logging_nan_inf_filter=False, report_to=[],
        save_strategy="epoch", save_total_limit=2, save_only_model=True,
        dataloader_num_workers=0, dataloader_drop_last=False, remove_unused_columns=False,
        seed=args.seed, data_seed=args.seed, group_by_length=True, disable_tqdm=True)
    trainer = ObservedTrainer(model=model, args=training_args, train_dataset=Dataset(),
                              data_collator=tracked_collator, callbacks=[ReceiptCallback()])
    try:
        result = trainer.train()
        final = adapter_fingerprint(model, "default")
        if final == initial:
            raise RuntimeError("Adapter weights did not change")
        if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise RuntimeError("Non-finite saved adapter parameter")
        exposure = exposure_summary(rows, counts)
        if not args.smoke_only and exposure["min_visits"] < int(args.epochs):
            raise RuntimeError("Formal run did not process every row for every epoch")
        if not args.smoke_only:
            expected=int(args.epochs)
            if exposure['max_visits']!=expected or exposure['min_visits']!=expected:
                raise RuntimeError('Unexpected formal row exposure')
            if observed['supervised_tokens']!=expected*sum(target_counts) or observed['weighted_token_mass']!=expected*weighted_mass:
                raise RuntimeError('Observed supervised/weighted token count mismatch')
        if not observed['longest_example_seen']:
            raise RuntimeError('Longest full example not processed')
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        disk = load_file(str(output / "adapter_model.safetensors"), device="cpu")
        live = get_peft_model_state_dict(model)
        if disk.keys() != live.keys() or any(not torch.equal(disk[k], live[k].detach().cpu()) for k in disk):
            raise RuntimeError("Saved adapter weights differ from trained tensors")
        model.load_adapter(str(output),adapter_name='reload_check',is_trainable=False,local_files_only=True)
        if adapter_fingerprint(model,'reload_check')!=final:
            model.load_adapter(str(output),adapter_name='reload_check',is_trainable=False,local_files_only=True)
        if adapter_fingerprint(model,'reload_check')!=final:
            raise RuntimeError('Saved adapter reloaded into new slot differs from trained tensors')
        model.delete_adapter('reload_check')
        model.set_adapter('default')
        receipt.update({"completed": True, "finished_unix": time.time(),
                        "elapsed_seconds": time.time() - started, "observed_exposure": {**exposure, **observed},
                        "adapter_final_sha256": final, "adapter_files_sha256": adapter_files_digest(output),
                        "saved_weights_bitwise_verified": True, "metrics": result.metrics,
                        'saved_adapter_reloaded_exact':True,
                        'reload_check_scope':'new PEFT adapter slot on the same unmodified base model; not fresh base reload',
                        "global_step": trainer.state.global_step})
        trainer.save_state()
        write_json(output / "training_manifest.json", receipt)
        print(canonical({"completed": True, "output_dir": str(output), "stage": args.stage,
                         "exposure": exposure, "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)
    except BaseException as exc:
        receipt.update({"completed": False, "failed": True, "exception_type": type(exc).__name__,
                        "elapsed_seconds": time.time() - started,
                        "observed_exposure": {**exposure_summary(rows, counts), **observed}})
        write_json(output / "training_manifest.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", help="Expected SHA256 from the audited release manifest")
    parser.add_argument("--stage", choices=["entities", "relations"], required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument('--parent-adapter',help='Omit for fresh base; otherwise exact frozen SFT parent only')
    parser.add_argument('--label-weight',type=float,default=4.)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=.05)
    parser.add_argument("--warmup-ratio", type=float, default=.05)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke-only", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()

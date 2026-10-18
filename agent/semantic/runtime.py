"""Actual two-stage semantic-text inference with one base and two LoRA adapters.

Torch/Transformers/PEFT are imported only when a generator is constructed.
All generated text, failed windows and stopping reasons remain auditable.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from generative.chat import render_prompt
from generative.train import adapter_fingerprint
from kg_agent.choice_backend import adapter_files_digest, base_input_digest
from semantic.data import VARIANTS, canonical, sha, source_windows
from semantic import protocol as p


ENTITY_VARIANTS = ("full_source", "short_source")
ENTITY_TRAINING_INPUT_VIEWS = {"full_source": "full_source", "short_source": "short_source",
                             "short_source_complete_windows": "short_source"}


def protocol_digest():
    return hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()


def check_adapter_contract(path, stage, *, base_model):
    if stage not in ("entities", "relations"):
        raise ValueError("Unknown semantic adapter stage")
    receipt = json.loads((Path(path) / "training_manifest.json").read_text())
    if receipt.get("contract") != p.CONTRACT or receipt.get("stage") != stage:
        raise ValueError("Wrong semantic output contract or adapter stage")
    if receipt.get("completed") is not True or receipt.get("smoke_only") is not False or receipt.get("failed") is True:
        raise ValueError("Evaluation requires a completed non-smoke semantic adapter")
    if receipt.get("protocol_sha256") != protocol_digest():
        raise ValueError("Semantic protocol differs from the training receipt")
    if receipt.get("base_input_sha256") != base_input_digest(base_model):
        raise ValueError("Base tokenizer/configuration differs from the training receipt")
    if receipt.get("adapter_files_sha256") != adapter_files_digest(path):
        raise ValueError("Semantic adapter files differ from the training receipt")
    if not isinstance(receipt.get("adapter_final_sha256"), str) or len(receipt["adapter_final_sha256"]) != 64:
        raise ValueError("Training receipt has no saved in-memory adapter fingerprint")
    return receipt


def verify_title_context(metadata, title_sources, *, source_group_id):
    """Verify title receipt against original structured input, never its gold."""
    p._metadata(metadata)
    title = metadata.get("title_context")
    if title is None:
        return None
    if str(title["source_group_id"]) != str(source_group_id):
        raise ValueError("Title context crosses the source group")
    receipt = title_sources.get(title["source_record_sha256"])
    if not isinstance(receipt, dict) or str(receipt.get("source_group_id")) != str(source_group_id):
        raise ValueError("Missing or cross-group title source receipt")
    original = receipt["original_record"]
    if sha(original) != title["source_record_sha256"]:
        raise ValueError("Title original source hash mismatch")
    users = [m["content"] for m in original["messages"] if m["role"] == "user"]
    if len(users) != 1 or not isinstance(users[0], str):
        raise ValueError("Title receipt requires one original user source")
    raw = users[0]
    a, b = title["source_start"], title["source_end"]
    if b > len(raw) or raw[a:b] != title["text"]:
        raise ValueError("Title does not match its original raw source slice")
    from kg_agent.source import SourceRouter
    routed = SourceRouter().route(raw)
    if routed.source_type != "structured_icd" or routed.metadata.get("title") != title["text"]:
        raise ValueError("Title is not the structured input title")
    return deepcopy(title)


def decode_completion(tokenizer, token_ids, *, max_new_tokens, deadline_exhausted=False):
    ids = list(token_ids)
    eos = tokenizer.eos_token_id
    observed = eos in ids
    end = ids.index(eos) if observed else len(ids)
    text = tokenizer.decode(ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    trace = {"output_contract": p.CONTRACT, "generated_token_ids": ids,
             "generated_tokens": end + int(observed), "eos_observed": observed,
             "budget_exhausted": not observed and len(ids) >= max_new_tokens,
             "deadline_exhausted": bool(deadline_exhausted), "completion_text": text,
             "complete": False, "error": None}
    if deadline_exhausted:
        trace["error"] = "deadline_exhausted"
    elif not observed:
        trace["error"] = "missing_eos"
    elif any(token != tokenizer.pad_token_id for token in ids[end + 1:]):
        trace["error"] = "tokens_after_eos"
    else:
        trace["complete"] = True
    return {"completion_text": text, "trace": trace}


class SemanticGenerator:
    """Length-bucketed deterministic generation; OOM shrinks batches, not data."""

    def __init__(self, base_model, entity_adapter, relation_adapter, *, max_input_tokens=8192,
                 max_new_tokens=4096, batch_size=2, clock=time.monotonic):
        if any(type(value) is not int or value < 1 for value in (max_input_tokens, max_new_tokens, batch_size)):
            raise ValueError("Token budgets and batch size must be positive integers")
        self.receipts = {stage: check_adapter_contract(path, stage, base_model=base_model)
                         for stage, path in (("entities", entity_adapter), ("relations", relation_adapter))}
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA is required for actual semantic inference")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True)
        self.model = PeftModel.from_pretrained(model, entity_adapter, adapter_name="entities",
                                              is_trainable=False, local_files_only=True)
        self.weight_verification = {}
        def ensure_exact(stage, path):
            expected = self.receipts[stage]["adapter_final_sha256"]
            attempts = 1
            actual = adapter_fingerprint(self.model, stage)
            if actual != expected:
                # PEFT may initially create a BF16 slot before FP32 upcasting.
                # A second bounded load restores exact FP32 checkpoint values.
                self.model.load_adapter(path, adapter_name=stage, is_trainable=False, local_files_only=True)
                attempts += 1
                actual = adapter_fingerprint(self.model, stage)
            if actual != expected:
                raise RuntimeError(f"Loaded {stage} adapter differs from exact training fingerprint")
            self.weight_verification[stage] = {"expected_sha256": expected, "loaded_sha256": actual,
                                                "load_attempts": attempts}
        ensure_exact("entities", entity_adapter)
        self.model.load_adapter(relation_adapter, adapter_name="relations", is_trainable=False, local_files_only=True)
        ensure_exact("relations", relation_adapter)
        if adapter_fingerprint(self.model, "entities") != self.receipts["entities"]["adapter_final_sha256"]:
            raise RuntimeError("Loading relation adapter changed the entity adapter")
        self.model = self.model.to("cuda").eval()
        for stage in ("entities", "relations"):
            actual = adapter_fingerprint(self.model, stage)
            if actual != self.receipts[stage]["adapter_final_sha256"]:
                raise RuntimeError(f"CUDA transfer changed the {stage} adapter")
            self.weight_verification[stage]["cuda_sha256"] = actual
        for name, parameter in self.model.named_parameters():
            if "lora_" in name and not torch.isfinite(parameter).all().item():
                raise RuntimeError("Non-finite inference adapter parameter: " + name)
        self.max_input_tokens, self.max_new_tokens, self.batch_size = max_input_tokens, max_new_tokens, batch_size
        self.clock = clock

    def _failure(self, prompt, stage, error, **extra):
        return {"completion_text": "", "trace": {"output_contract": p.CONTRACT, "stage": stage,
            "prompt_sha256": sha(prompt), "complete": False, "eos_observed": False,
            "generated_tokens": 0, "generated_token_ids": [], "completion_text": "",
            "budget_exhausted": False, "deadline_exhausted": False, "error": error, **extra}}

    def generate(self, prompt, stage, *, deadline=None):
        return self.generate_many([prompt], stage, deadline=deadline)[0]

    def generate_many(self, prompts, stage, *, deadline=None):
        if stage not in ("entities", "relations"):
            raise ValueError("Unknown semantic inference stage")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be a finite monotonic timestamp")
        outputs, prepared = [None] * len(prompts), []
        context_max = getattr(self.model.config, "max_position_embeddings", None)
        for index, prompt in enumerate(prompts):
            rendered = render_prompt(self.tokenizer, prompt)
            length = len(self.tokenizer.encode(rendered, add_special_tokens=False))
            if length > self.max_input_tokens or (context_max and length + self.max_new_tokens > context_max):
                outputs[index] = self._failure(prompt, stage, "input_budget_exceeded_no_truncation", prompt_tokens=length)
            else:
                prepared.append((length, index, rendered))
        prepared.sort(key=lambda row: (row[0], row[1]))
        self.model.set_adapter(stage)
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        generator = self
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return deadline is not None and generator.clock() >= deadline
        config = GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, use_cache=True)

        def run_batch(batch, retries=0):
            if deadline is not None and self.clock() >= deadline:
                for length, index, _ in batch:
                    outputs[index] = self._failure(prompts[index], stage, "deadline_before_generation",
                        deadline_exhausted=True, prompt_tokens=length)
                return
            started = self.clock()
            retry_smaller = False
            try:
                inputs = self.tokenizer([row[2] for row in batch], add_special_tokens=False,
                    padding=True, truncation=False, return_tensors="pt").to(self.model.device)
                width = inputs["input_ids"].shape[1]
                with self.torch.inference_mode():
                    generated = self.model.generate(**inputs, generation_config=config,
                        stopping_criteria=StoppingCriteriaList([Deadline()]), use_model_defaults=False)
                suffixes = generated[:, width:].tolist()
                if len(suffixes) != len(batch):
                    raise RuntimeError("Generation returned the wrong batch size")
                expired = deadline is not None and self.clock() >= deadline
                for (length, index, rendered), suffix in zip(batch, suffixes):
                    result = decode_completion(self.tokenizer, suffix, max_new_tokens=self.max_new_tokens,
                                               deadline_exhausted=expired)
                    result["trace"].update(stage=stage, prompt_tokens=length, prompt_sha256=sha(prompts[index]),
                        rendered_prompt_sha256=sha(rendered), elapsed_seconds=max(0., self.clock() - started),
                        batch_size=len(batch), oom_retries=retries, max_new_tokens=self.max_new_tokens)
                    outputs[index] = result
            except Exception as exc:
                oom = isinstance(exc, self.torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if oom and len(batch) > 1:
                    retry_smaller = True
                else:
                    for length, index, _ in batch:
                        outputs[index] = self._failure(prompts[index], stage, f"{type(exc).__name__}: {exc}",
                            prompt_tokens=length, oom_retries=retries, elapsed_seconds=max(0., self.clock() - started))
            if retry_smaller:
                # Leave the except block first: its traceback may retain the
                # failed decoder's GPU KV cache and defeat smaller-batch retry.
                inputs = generated = None
                self.torch.cuda.empty_cache()
                middle = len(batch) // 2
                run_batch(batch[:middle], retries + 1)
                run_batch(batch[middle:], retries + 1)
        for start in range(0, len(prepared), self.batch_size):
            run_batch(prepared[start:start + self.batch_size])
        return outputs


def _invalid_attempts(text):
    return sum(bool(line.strip()) for line in (text or "").splitlines())


def _run_jobs(jobs, generator, stage, deadline, on_generation):
    if not jobs:
        return []
    runs = [None] * len(jobs)
    # Character-length buckets bound work between durable progress updates;
    # the generator further orders each bucket by actual rendered token length.
    ordered = sorted(range(len(jobs)), key=lambda index: (len(jobs[index]["prompt"]), index))
    batch_size = max(1, int(getattr(generator, "batch_size", 2)))
    def consume(index, output):
        job = jobs[index]
        trace = deepcopy(output.get("trace", {}))
        completion = output.get("completion_text", "")
        error = trace.get("error")
        if (trace.get("complete") is not True or trace.get("eos_observed") is not True
                or trace.get("budget_exhausted") or trace.get("deadline_exhausted")):
            error = error or "incomplete_generation"
        parsed = None
        if not error:
            try:
                parsed = (p.parse_entities(job["text"], completion, job["metadata"]) if stage == "entities"
                    else p.parse_relations(job["text"], job["entities"], completion, job["metadata"]))
                if not parsed["valid"]:
                    error = "invalid_semantic_records"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        row = {**deepcopy(job), "stage": stage, "completion_text": completion, "generation_trace": trace,
               "parse": parsed, "status": "failed" if error else "complete", "error": error,
               "invalid_attempt_records": _invalid_attempts(completion) if error else 0,
               "output_contract": p.CONTRACT}
        runs[index] = row
        if on_generation:
            on_generation(row)
    for start in range(0, len(ordered), batch_size):
        indices = ordered[start:start + batch_size]
        prompts = [jobs[index]["prompt"] for index in indices]
        try:
            if hasattr(generator, "generate_many"):
                generated = generator.generate_many(prompts, stage, deadline=deadline)
            else:
                generated = [generator.generate(prompt, stage, deadline=deadline) for prompt in prompts]
            if len(generated) != len(indices):
                raise ValueError("Generator must return exactly one result per job")
        except Exception as exc:
            generated = [{"completion_text": "", "trace": {"complete": False, "eos_observed": False,
                           "error": f"{type(exc).__name__}: {exc}"}} for _ in indices]
        for index, output in zip(indices, generated):
            consume(index, output)
    return runs


def _empty_entity_result():
    return {"valid": False, "entities": [], "filtered_entities": []}


def entity_strategy_contract(entity_variant, receipt=None, *, short_chars=2400):
    """Require an explicitly matched entity adapter when a receipt is present."""
    if entity_variant not in ENTITY_VARIANTS:
        raise ValueError("Unknown entity inference variant")
    trained = receipt.get("variant") if receipt is not None else None
    trained_view = ENTITY_TRAINING_INPUT_VIEWS.get(trained)
    if receipt is not None and trained_view != entity_variant:
        raise ValueError("Entity inference view differs from or cannot verify the entity adapter training variant")
    return {"entity_variant": entity_variant, "entity_prompt_variant": entity_variant,
            "entity_training_variant": trained, "entity_training_input_view": trained_view,
            "entity_prompt_matches_training_input_view": trained_view == entity_variant if receipt is not None else None,
            "entity_prompt_matches_training_variant":
                trained == entity_variant if receipt is not None else None,
            "entity_training_data_sha256": receipt.get("data_sha256") if receipt is not None else None,
            "entity_training_manifest_canonical_sha256": sha(receipt) if receipt is not None else None,
            "entity_window_char_budget": short_chars if entity_variant == "short_source" else None,
            "entity_window_overlap_chars": min(400, short_chars // 4) if entity_variant == "short_source" else None,
            "entity_window_boundaries_use_gold": False,
            "variant_match_scope": "Input view and source order only; it does not establish complete-window supervision. Exact supervision is identified by the training manifest and data SHA, including changed overlap/title exposure policies.",
            "entity_order": "source", "automatic_full_source_fallback": False}


def _merge_entity_windows(runs):
    """Union locally parsed successes; shift body anchors, never re-ground globally.

    Raw and filtered unions are independent. An entity must pass grounding in at
    least one complete local window to enter the relation table. Two failed
    grounding records from different windows cannot jointly manufacture one.
    """
    successes = [run for run in runs if run["status"] == "complete"]
    failed = [run for run in runs if run["status"] != "complete"]
    def merge(field):
        merged = {}
        for run in successes:
            for source in run["parse"][field]:
                entity = deepcopy(source)
                for name in ("mentions", "canonical_name_anchors"):
                    for anchor in entity[name]:
                        if anchor["context"] == "body":
                            anchor["start"] += run["window"]["start"]
                            anchor["end"] += run["window"]["start"]
                entity["source_windows"] = [{"window": deepcopy(run["window"]),
                    "generation_id": run["id"], "source_lines": deepcopy(source["source_lines"]),
                    "locally_grounded": source["grounding_verified"]}]
                key = entity["label"], p.normalize_name(entity["name"])
                if key not in merged:
                    merged[key] = entity
                    continue
                current = merged[key]
                for name in ("mentions", "canonical_name_anchors", "source_windows"):
                    for value in entity[name]:
                        if value not in current[name]: current[name].append(value)
                current["grounding_verified"] = current["grounding_verified"] or entity["grounding_verified"]
                current["rule_rejections"].extend(entity["rule_rejections"])
                # The source-window provenance above disambiguates local lines.
                current["source_lines"].extend(entity["source_lines"])
                if entity["canonical_name_source"] == "copied": current["canonical_name_source"] = "copied"
                elif current["canonical_name_source"] == "model_inferred":
                    current["canonical_name_source"] = entity["canonical_name_source"]
        result = list(merged.values())
        for entity in result:
            for name in ("mentions", "canonical_name_anchors"):
                entity[name].sort(key=lambda a: (a["context"] != "body", a["start"], a["end"]))
        return p._ordered(result, "source")
    raw, filtered = merge("entities"), merge("filtered_entities")
    filtered_ids = {entity["id"] for entity in filtered}
    return {"valid": bool(successes), "complete": not failed, "partial": bool(successes and failed),
        "output_contract": p.CONTRACT, "entities": raw, "raw_entities": deepcopy(raw),
        "filtered_entities": filtered, "validated_entities": deepcopy(filtered),
        "rejected_entities": [{"entity": deepcopy(entity), "reasons": deepcopy(entity["rule_rejections"])}
                              for entity in raw if entity["id"] not in filtered_ids],
        "errors": [{"generation_id": run["id"], "window": deepcopy(run["window"]), "reason": run["error"]} for run in failed],
        "trace": {"entity_variant": "short_source", "windows": len(runs), "successful_windows": len(successes),
            "failed_windows": len(failed), "accepted_entities": len(raw), "filtered_entities": len(filtered),
            "body_anchor_coordinates": "full original source payload; local offsets plus source-window start",
            "title_anchor_coordinates": "title-local start/end and unchanged original structured-source source_start/source_end",
            "merge_policy": "Typed normalized name union of successful windows; raw and locally grounded unions computed separately; no global re-grounding or gold-driven fallback",
            "generation_ids": [run["id"] for run in runs]}}


def _merge_relations(runs, field):
    merged = {}
    for run in runs:
        if run["status"] != "complete":
            continue
        for edge in run["parse"][field]:
            key = edge["source"], edge["relation"], edge["target"]
            item = merged.setdefault(key, {"source": key[0], "relation": key[1], "target": key[2], "evidence": []})
            for quote in edge["evidence_texts"]:
                if quote not in item["evidence"]:
                    item["evidence"].append(quote)
    return list(merged.values())


def _graph(entities, relations):
    ids = {entity["id"] for entity in entities}
    return {"entities": [{key: deepcopy(entity[key]) for key in ("id", "label", "name")} for entity in entities],
            "relations": [deepcopy(edge) for edge in relations if edge["source"] in ids and edge["target"] in ids]}


def run_documents(documents, generator, *, variant="full_source", entity_variant="full_source", mode="live", short_chars=2400,
                  deadline=None, on_generation=None, on_document=None):
    """Batch explicit full/short source entities, then all relation windows.

    documents contain text/metadata/id; oracle mode additionally requires an
    explicit oracle_entities table. No gold is inspected in live mode.
    """
    if variant not in VARIANTS or mode not in ("live", "oracle"):
        raise ValueError("Unknown semantic evaluation variant or entity mode")
    receipt = getattr(generator, "receipts", {}).get("entities") if mode == "live" else None
    entity_contract = entity_strategy_contract(entity_variant, receipt, short_chars=short_chars)
    order = "canonical" if variant == "short_canonical" else "source"
    jobs, states = [], []
    for index, document in enumerate(documents):
        state = {"id": document["id"], "document": document, "entity": _empty_entity_result(), "entity_runs": []}
        states.append(state)
        from kg_agent.source import SourceRouter, parse_icd_graph
        from kg_agent.assemble import assemble_graph as assemble_icd
        routed = SourceRouter().route(document["text"])
        if routed.source_type == "structured_icd":
            try:
                state["structured_graph"] = assemble_icd(parse_icd_graph(routed))
                state["structured_error"] = None
            except Exception as exc:
                state["structured_graph"] = {"entities": [], "relations": []}
                state["structured_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if mode == "oracle":
            supplied = deepcopy(document["oracle_entities"])
            p._input_entities(supplied)
            state["entity"] = {"valid": True, "entities": supplied, "filtered_entities": deepcopy(supplied),
                               "trace": {"entity_source": "oracle_no_model_prediction_or_grounding_claim"}}
        else:
            text, metadata = document["text"], document.get("metadata", {})
            windows = ([(0, len(text))] if entity_variant == "full_source"
                       else source_windows(text, short_chars, min(400, short_chars // 4)))
            for window_index, (a, b) in enumerate(windows):
                window = {"start": a, "end": b, "index": window_index}
                local_metadata = (metadata if entity_variant == "full_source" else
                    {**deepcopy(metadata), "variant": entity_variant, "order": "source",
                     "source_window": {**window, "full_source_fallback": False}, "relation_entity_table_scope": None})
                suffix = "" if entity_variant == "full_source" else f":{window_index}"
                jobs.append({"id": f"{document['id']}:entities{suffix}", "document_index": index, "document_id": document["id"],
                    "window": window, "text": text[a:b], "metadata": local_metadata, "entity_variant": entity_variant,
                    "prompt": p.build_entity_prompt(text[a:b], local_metadata, "source")})
    for run in _run_jobs(jobs, generator, "entities", deadline, on_generation):
        state = states[run["document_index"]]
        state["entity_runs"].append(run)
        if entity_variant == "full_source" and run["status"] == "complete":
            state["entity"] = run["parse"]
    if mode == "live" and entity_variant == "short_source":
        for state in states:
            if "structured_graph" not in state:
                state["entity"] = _merge_entity_windows(state["entity_runs"])
    relation_jobs = []
    for index, state in enumerate(states):
        document, entity_result = state["document"], state["entity"]
        table = entity_result["filtered_entities"]
        if not entity_result["valid"] or not table:
            continue
        text = document["text"]
        windows = [(0, len(text))] if variant == "full_source" else source_windows(text, short_chars, min(400, short_chars // 4))
        for window_index, (a, b) in enumerate(windows):
            metadata = {**document.get("metadata", {}), "source_window": {"start": a, "end": b, "index": window_index},
                        "relation_entity_table_scope": "complete_source_entities"}
            relation_jobs.append({"id": f"{document['id']}:relations:{window_index}", "document_index": index,
                "document_id": document["id"], "window": {"start": a, "end": b, "index": window_index},
                "text": text[a:b], "metadata": metadata, "entities": deepcopy(table),
                "prompt": p.build_relation_prompt(text[a:b], table, metadata, order)})
    relation_runs = _run_jobs(relation_jobs, generator, "relations", deadline, on_generation)
    grouped = {index: [] for index in range(len(states))}
    for run in relation_runs:
        grouped[run["document_index"]].append(run)
    results = []
    for index, state in enumerate(states):
        er, rr = state["entity"], grouped[index]
        if "structured_graph" in state:
            error = state["structured_error"]
            result = {"id": state["id"], "mode": mode, "variant": variant, "entity_variant": entity_variant, "output_contract": p.CONTRACT,
                "raw_graph": state["structured_graph"], "filtered_graph": deepcopy(state["structured_graph"]),
                "generations": [], "trace": {"entity_source": "deterministic_structured_icd", "generation_complete": error is None,
                    "generation_calls": 0, "failed_calls": int(error is not None), "error": error,
                    "invalid_entity_attempts": 0, "invalid_relation_attempts": 0, "budget_exhaustions": 0,
                    "deadline_exhausted": False, "chunks": []}}
            results.append(result)
            if on_document: on_document(result)
            continue
        runs = state["entity_runs"] + rr
        failed = [run for run in runs if run["status"] != "complete"]
        result = {"id": state["id"], "mode": mode, "variant": variant, "entity_variant": entity_variant, "output_contract": p.CONTRACT,
            "raw_graph": _graph(er["entities"], _merge_relations(rr, "relations")),
            "filtered_graph": _graph(er["filtered_entities"], _merge_relations(rr, "filtered_relations")),
            "trace": {"entity_source": mode, "entity_context": entity_variant if mode == "live" else "oracle",
                **entity_contract, "relation_order": order,
                "entity_strategy_applied": mode == "live", "entity_window_count": len(state["entity_runs"]),
                "entity_successful_windows": sum(run["status"] == "complete" for run in state["entity_runs"]),
                "entity_complete": all(run["status"] == "complete" for run in state["entity_runs"]),
                "relation_entity_table_scope": "complete_filtered_entity_stage_table" if mode == "live" else "complete_oracle_table",
                "generation_calls": len(runs), "failed_calls": len(failed), "generation_complete": not failed,
                "invalid_entity_attempts": sum(run["invalid_attempt_records"] for run in state["entity_runs"]),
                "invalid_relation_attempts": sum(run["invalid_attempt_records"] for run in rr),
                "budget_exhaustions": sum(bool(run["generation_trace"].get("budget_exhausted")) for run in runs),
                "deadline_exhausted": any(run["generation_trace"].get("deadline_exhausted") for run in runs),
                "chunks": [{"window": run["window"], "stage": run["stage"], "relation_errors": [run["error"]] if run["error"] else []} for run in runs],
                "entity_parse": er, "relation_run_ids": [run["id"] for run in rr]}, "generations": runs}
        results.append(result)
        if on_document:
            on_document(result)
    return results


def run_document(text, metadata, generator, *, variant="full_source", mode="live", oracle_entities=None, **kwargs):
    document = {"id": "document", "text": text, "metadata": metadata}
    if mode == "oracle":
        document["oracle_entities"] = oracle_entities
    return run_documents([document], generator, variant=variant, mode=mode, **kwargs)[0]

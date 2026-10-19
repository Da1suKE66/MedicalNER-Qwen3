"""Causal Qwen chooses finite actions; Python owns all graph serialization.

This is a new output contract, not a parser for historical JSON checkpoints.
No fallback to unconstrained generation is permitted.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unicodedata

from .contracts import Relation


CONTRACT = "bounded-choices-v1"


def protocol_implementation_digest():
    value = hashlib.sha256()
    value.update(unicodedata.unidata_version.encode())
    for name in ("choice_protocol.py", "entity_choices.py", "relation_choices.py", "contracts.py", "ontology.json"):
        value.update(name.encode())
        value.update(Path(__file__).with_name(name).read_bytes())
    value.update((Path(__file__).resolve().parents[1] / "generative" / "chat.py").read_bytes())
    return value.hexdigest()


def base_input_digest(path):
    """Bind tokenizer and model configuration, not a hash of the 8B weights."""
    value = hashlib.sha256()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        file = Path(path) / name
        if name in {"config.json", "tokenizer.json", "tokenizer_config.json"} and not file.is_file():
            raise ValueError(f"Missing base input artifact: {name}")
        value.update(name.encode())
        value.update(file.read_bytes() if file.is_file() else b"ABSENT")
    return value.hexdigest()


def adapter_files_digest(path):
    value = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        file = Path(path) / name
        if not file.is_file():
            raise ValueError(f"Missing adapter artifact: {name}")
        value.update(name.encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
    return value.hexdigest()


class ChoiceFailure(RuntimeError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def check_adapter_contract(path, stage, *, base_model=None):
    """Refuse old adapters before loading an expensive base model."""
    manifest = Path(path) / "choice_contract.json"
    if not manifest.is_file():
        raise ValueError("Adapter has no bounded-choice training receipt; legacy JSON adapters require retraining")
    value = json.loads(manifest.read_text())
    if (value.get("output_contract") != CONTRACT or stage not in value.get("stages", [])
            or value.get("completion_encoding") != "independent_integer_newline_tokens_then_eos"):
        raise ValueError(f"Adapter output contract does not support {stage}")
    if value.get("protocol_sha256") != protocol_implementation_digest():
        raise ValueError("Choice protocol implementation differs from the trained adapter")
    if base_model is not None and value.get("base_input_sha256") != base_input_digest(base_model):
        raise ValueError("Base tokenizer/model configuration differs from the training receipt")
    actual = adapter_files_digest(path)
    if not actual or actual != value.get("adapter_sha256"):
        raise ValueError("Choice adapter weights do not match their training receipt")
    return value


class ChoiceGenerator:
    """One sample at a time; entity/relation views may share the same base."""

    def __init__(self, backend, *, max_input_tokens=6144, max_new_tokens=2048):
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token budgets must be positive")
        self.backend = backend
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def run(self, task):
        import torch
        from transformers import GenerationConfig
        from generative.chat import render_prompt
        from .choice_protocol import ChoiceProtocol

        backend = self.backend
        tokenizer = backend.tokenizer
        if backend.adapter_name is not None:
            backend.model.set_adapter(backend.adapter_name)
        rendered = render_prompt(tokenizer, task.prompt)
        inputs = tokenizer(rendered, add_special_tokens=False, truncation=False,
                           return_tensors="pt").to(backend.model.device)
        length = inputs["input_ids"].shape[1]
        trace = {"output_contract": CONTRACT, "prompt_tokens": length,
                 "max_new_tokens": self.max_new_tokens,
                 "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
        if length > self.max_input_tokens:
            raise ChoiceFailure("Choice prompt exceeds input budget; no silent truncation", trace)
        maximum = getattr(backend.model.config, "max_position_embeddings", None)
        if maximum and length + self.max_new_tokens > maximum:
            raise ChoiceFailure("Choice generation would exceed model context", trace)
        codec = ChoiceProtocol(tokenizer, length, task.choices)
        # A fresh config prevents a checkpoint's forced EOS/min_length/bad-word
        # settings from overriding this protocol's permitted token paths.
        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        with torch.inference_mode():
            output = backend.model.generate(
                **inputs, generation_config=config,
                use_model_defaults=False,
                prefix_allowed_tokens_fn=codec.prefix_allowed_tokens_fn,
            )
        suffix = output[0, length:].tolist()
        result = codec.read_result(suffix, max_new_tokens=self.max_new_tokens)
        trace.update(asdict(result), generated_tokens=len(suffix), generated_token_ids=suffix)
        if result.error or not result.complete:
            raise ChoiceFailure("Incomplete or invalid bounded-choice decision", trace)
        return list(result.actions), trace


class ChoiceEntitySelector:
    output_contract = CONTRACT

    def __init__(self, generator):
        self.generator = generator

    def select(self, spans, *, target_name=None, max_candidates=64):
        from .entity_choices import EntityChoiceTask
        task = EntityChoiceTask(spans, target_name=target_name, max_candidates=max_candidates)
        actions, trace = self.generator.run(task)
        trace.update(task.trace(actions))
        return task.candidates(actions), trace


class ChoiceRelationSelector:
    output_contract = CONTRACT

    def __init__(self, generator, *, config=None):
        from generative.protocol import ProtocolConfig
        self.generator = generator
        self.config = config or ProtocolConfig(pairs_per_batch=4, order="source")
        # Evidence references must retain the live pipeline's source coordinates.
        if self.config.repair or self.config.matching != "legacy":
            raise ValueError("Live choice pipeline currently requires unchanged source coordinates")

    def classify(self, entities, document):
        from generative.protocol import build_batches
        from .relation_choices import RelationChoiceTask
        nodes = [asdict(entity) for entity in entities]
        batches, _, audit = build_batches(document.raw, nodes, self.config)
        edges, runs = [], []
        for batch in batches:
            task = RelationChoiceTask(batch)
            actions, trace = self.generator.run(task)
            pairs = {pair["pair_id"]: pair for pair in batch["pairs"]}
            for edge in task.edges(actions):
                pair = pairs[edge["pair_id"]]
                edges.append(Relation(
                    source=pair["source"], target=pair["target"], relation=edge["relation"],
                    evidence_ids=list(edge["evidence_span_ids"]),
                    evidence_text=[batch["original_spans"][sid] for sid in edge["evidence_span_ids"]],
                ))
            runs.append({"batch_id": batch["id"], **trace})
        return edges, {"output_contract": CONTRACT, **audit, "batches": runs,
                       "evidence_policy": "Selected source spans; source validity does not prove entailment"}

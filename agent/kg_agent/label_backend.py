"""Causal relation generation with literal labels, without numeric actions."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unicodedata

from .choice_backend import adapter_files_digest, base_input_digest, ChoiceFailure
from .contracts import Relation


CONTRACT = "relation-label-lines-v1"
COMPLETION_ENCODING = "literal_relation_lines_then_eos"


def protocol_implementation_digest():
    digest = hashlib.sha256(unicodedata.unidata_version.encode())
    root = Path(__file__).resolve().parents[1]
    for relative in ("kg_agent/relation_labels.py", "kg_agent/relation_choices.py",
                     "kg_agent/label_backend.py", "kg_agent/contracts.py", "kg_agent/ontology.json",
                     "generative/chat.py", "generative/label_data.py"):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def check_adapter_contract(path, stage="relations", *, base_model=None):
    receipt = Path(path) / "label_contract.json"
    if not receipt.is_file():
        raise ValueError("Adapter has no literal-label training receipt; numeric/JSON output weights are not interchangeable")
    value = json.loads(receipt.read_text())
    if (stage != "relations" or value.get("output_contract") != CONTRACT
            or value.get("stages") != ["relations"]
            or value.get("completion_encoding") != COMPLETION_ENCODING):
        raise ValueError("Adapter does not support literal relation labels")
    if value.get("protocol_sha256") != protocol_implementation_digest():
        raise ValueError("Literal-label protocol differs from the training receipt")
    if base_model is not None and value.get("base_input_sha256") != base_input_digest(base_model):
        raise ValueError("Base tokenizer/model configuration differs from the training receipt")
    if value.get("adapter_sha256") != adapter_files_digest(path):
        raise ValueError("Literal-label adapter weights do not match the training receipt")
    return value


def decode_completion(task, tokenizer, suffix, *, max_new_tokens):
    """Require an observed EOS even if a budget-limited prefix looks parsable."""
    tokens = list(suffix)
    eos = tokenizer.eos_token_id
    trace = {"output_contract": CONTRACT, "generated_tokens": len(tokens),
             "generated_token_ids": tokens, "eos_observed": eos in tokens,
             "budget_exhausted": eos not in tokens and len(tokens) >= max_new_tokens,
             "complete": False}
    end = tokens.index(eos) if eos in tokens else len(tokens)
    text = tokenizer.decode(tokens[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    trace["completion_text"] = text
    if eos not in tokens:
        trace["error"] = "missing_eos"
        raise ChoiceFailure("Incomplete literal-label response", trace)
    if any(token != tokenizer.pad_token_id for token in tokens[end + 1:]):
        trace["error"] = "tokens_after_eos"
        raise ChoiceFailure("Unexpected tokens after literal-label EOS", trace)
    try:
        edges = task.parse(text)
    except ValueError as exc:
        trace["error"] = str(exc)
        raise ChoiceFailure("Invalid literal-label relation response", trace) from exc
    trace.update(complete=True, error=None, relation_records=len(edges),
                 emitted_lines=len(text.splitlines()),
                 duplicate_records_merged=len(text.splitlines()) - len(edges))
    return edges, trace


class LabelGenerator:
    def __init__(self, backend, *, max_input_tokens=6144, max_new_tokens=2048):
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("Token budgets must be positive")
        self.backend, self.max_input_tokens, self.max_new_tokens = backend, max_input_tokens, max_new_tokens

    def run(self, task):
        import torch
        from transformers import GenerationConfig
        from generative.chat import render_prompt
        backend, tokenizer = self.backend, self.backend.tokenizer
        if backend.adapter_name is not None:
            backend.model.set_adapter(backend.adapter_name)
        rendered = render_prompt(tokenizer, task.prompt)
        inputs = tokenizer(rendered, add_special_tokens=False, truncation=False, return_tensors="pt").to(backend.model.device)
        length = inputs["input_ids"].shape[1]
        trace = {"output_contract": CONTRACT, "prompt_tokens": length, "max_new_tokens": self.max_new_tokens,
                 "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
        maximum = getattr(backend.model.config, "max_position_embeddings", None)
        if length > self.max_input_tokens or (maximum and length + self.max_new_tokens > maximum):
            raise ChoiceFailure("Literal-label prompt exceeds input/context budget; no silent truncation", trace)
        config = GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        with torch.inference_mode():
            output = backend.model.generate(**inputs, generation_config=config, use_model_defaults=False)
        try:
            edges, result = decode_completion(task, tokenizer, output[0, length:].tolist(), max_new_tokens=self.max_new_tokens)
        except ChoiceFailure as exc:
            exc.trace = {**trace, **exc.trace}
            raise
        return edges, {**trace, **result}


class LabelRelationSelector:
    output_contract = CONTRACT

    def __init__(self, generator, *, config=None):
        from generative.protocol import ProtocolConfig
        self.generator = generator
        self.config = config or ProtocolConfig(pairs_per_batch=4, order="source")
        if self.config.repair or self.config.matching != "legacy":
            raise ValueError("Live label pipeline requires unchanged source coordinates")

    def classify(self, entities, document):
        from generative.protocol import build_batches
        from .relation_labels import RelationLabelTask
        batches, _, audit = build_batches(document.raw, [asdict(e) for e in entities], self.config)
        relations, runs = [], []
        for batch in batches:
            task = RelationLabelTask(batch)
            edges, trace = self.generator.run(task)
            pairs = {pair["pair_id"]: pair for pair in batch["pairs"]}
            for edge in edges:
                pair = pairs[edge["pair_id"]]
                relations.append(Relation(source=pair["source"], target=pair["target"], relation=edge["relation"],
                    evidence_ids=list(edge["evidence_span_ids"]),
                    evidence_text=[batch["original_spans"][ref] for ref in edge["evidence_span_ids"]]))
            runs.append({"batch_id": batch["id"], **trace})
        return relations, {"output_contract": CONTRACT, **audit, "batches": runs,
            "evidence_policy": "Visible references are validated; entailment remains a model decision"}

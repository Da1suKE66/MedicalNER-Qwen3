"""Controlled two-stage extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .assemble import AssemblerConfig, assemble_graph
from .contracts import (
    Entity,
    EntityCandidate,
    Graph,
    Relation,
    SourceDocument,
    ValidationReport,
    allowed_relations,
)
from .normalize import normalize_candidates
from .prompts import (
    build_entity_prompt,
    build_relation_prompt,
    extract_json_object,
    parse_entity_candidates,
    parse_relation_decisions,
)
from .source import SourceRouter, chunk_spans, parse_icd_graph, resolve_span_ref
from .validate import validate_candidates, validate_graph


class LLMBackend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        ...


class RelationClassifierBackend(Protocol):
    def classify(self, entities: list[Entity], document: SourceDocument) -> tuple[list[Relation], dict[str, Any]]:
        ...


@dataclass(frozen=True)
class PipelineConfig:
    max_entity_candidates: int = 64
    max_pairs_per_chunk: int = 128
    max_new_tokens_entities: int = 1024
    max_new_tokens_relations: int = 768
    max_spans_per_chunk: int = 16
    max_chars_per_chunk: int = 6000
    entity_mention_copy: bool = False
    relation_emit_none: bool = False
    relation_batch_size: int = 128
    assembler: AssemblerConfig = field(default_factory=AssemblerConfig)


@dataclass
class PipelineResult:
    graph: Graph
    output: dict[str, list[dict[str, Any]]]
    validation: ValidationReport
    trace: dict[str, Any] = field(default_factory=dict)


def _span_rows(document: SourceDocument, spans: list[Any]) -> list[dict[str, str]]:
    return [
        {"span_id": span.id, "text": span.text, "section": span.section}
        for span in spans
    ]


def _base_span_id(reference: str) -> str:
    return reference.split(":", 1)[0]


def _fixed_entities(metadata: dict[str, Any] | None) -> list[Entity]:
    metadata = metadata or {}
    target = metadata.get("target_disease") or metadata.get("target") or metadata.get("disease")
    if isinstance(target, dict):
        name = str(target.get("name") or target.get("title") or "").strip()
        properties = {key: value for key, value in target.items() if key not in {"name", "title"}}
    else:
        name = str(target or "").strip()
        properties = {}
    if not name:
        return []
    return [Entity("D1", "Disease", name, properties=properties, importance="required")]


def _candidate_pairs(
    entities: list[Entity],
    active_span_ids: set[str],
    max_pairs: int,
) -> list[dict[str, Any]]:
    def active(entity: Entity) -> bool:
        # A fixed target has no source span and is visible in every chunk.
        return not entity.span_ids or any(
            _base_span_id(span_id) in active_span_ids for span_id in entity.span_ids
        )

    pairs: list[dict[str, Any]] = []
    for source in entities:
        if not active(source):
            continue
        for target in entities:
            if source.id == target.id or not active(target):
                continue
            allowed = allowed_relations(source.label, target.label)
            if not allowed:
                continue
            pairs.append(
                {
                    "pair_id": f"P{len(pairs) + 1:04d}",
                    "source": source.id,
                    "target": target.id,
                    "allowed_relations": list(allowed),
                }
            )
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _filter_valid_relations(graph: Graph, document: SourceDocument, trace: dict[str, Any]) -> Graph:
    accepted: list[Relation] = []
    for relation in graph.relations:
        report = validate_graph(Graph(graph.entities, [relation]), document)
        if report.valid:
            accepted.append(relation)
        else:
            trace.setdefault("rejected_relations", []).append(
                {"relation": relation.__dict__, "issues": [issue.__dict__ for issue in report.issues]}
            )
    return Graph(graph.entities, accepted)


class TwoStageKGAgent:
    """Orchestrate the fixed workflow; the backend only makes semantic calls."""

    def __init__(self, entity_backend: LLMBackend, relation_backend: LLMBackend | None = None, config: PipelineConfig | None = None, *, relation_classifier: RelationClassifierBackend | None = None):
        self.entity_backend = entity_backend
        self.relation_backend = relation_backend or entity_backend
        self.config = config or PipelineConfig()
        self.relation_classifier = relation_classifier
        self.router = SourceRouter()

    def run(self, raw: Any, *, metadata: dict[str, Any] | None = None) -> PipelineResult:
        document = self.router.route(raw, metadata)
        trace: dict[str, Any] = {"source_type": document.source_type, "chunks": []}

        if document.source_type == "structured_icd":
            graph = parse_icd_graph(document)
            validation = validate_graph(graph, document)
            return PipelineResult(
                graph=graph,
                output=assemble_graph(graph, self.config.assembler),
                validation=validation,
                trace={**trace, "mode": "deterministic_icd_parser"},
            )

        fixed = _fixed_entities({**document.metadata, **(metadata or {})})
        chunks = chunk_spans(
            document.spans,
            max_spans=self.config.max_spans_per_chunk,
            max_chars=self.config.max_chars_per_chunk,
        )
        all_candidates: list[EntityCandidate] = []
        for chunk_index, chunk in enumerate(chunks):
            prompt = build_entity_prompt(
                _span_rows(document, chunk),
                target_name=fixed[0].name if fixed else None,
                max_candidates=self.config.max_entity_candidates,
                mention_copy=self.config.entity_mention_copy,
            )
            call_trace: dict[str, Any] = {"chunk": chunk_index, "span_ids": [span.id for span in chunk]}
            try:
                response = self.entity_backend.generate(prompt, max_new_tokens=self.config.max_new_tokens_entities)
                candidates = parse_entity_candidates(response)
                valid_candidates, candidate_report = validate_candidates(candidates, document)
                all_candidates.extend(valid_candidates)
                call_trace["candidate_count"] = len(valid_candidates)
                call_trace["candidate_issues"] = [issue.__dict__ for issue in candidate_report.issues]
            except Exception as exc:  # a failed chunk is isolated and traceable
                call_trace["error"] = f"{type(exc).__name__}: {exc}"
            trace["chunks"].append(call_trace)

        entities, span_to_entity_id, dropped = normalize_candidates(all_candidates, document, fixed)
        trace["dropped_candidates"] = dropped
        id_to_entity = {entity.id: entity for entity in entities}
        relations: list[Relation] = []

        if self.relation_classifier is not None:
            # Same entity stage and assembler; only the relation decision module changes.
            # Classifier failures are raised, never hidden as confident NONE decisions.
            relations, trace["relation_classifier"] = self.relation_classifier.classify(entities, document)

        for chunk_index, chunk in enumerate(chunks if self.relation_classifier is None else []):
            active_span_ids = {span.id for span in chunk}
            pairs = _candidate_pairs(entities, active_span_ids, self.config.max_pairs_per_chunk)
            if not pairs:
                continue
            rows = [
                {
                    "id": entity.id,
                    "label": entity.label,
                    "name": entity.name,
                    "importance": entity.importance,
                }
                for entity in entities
                if (not entity.span_ids or any(
                    _base_span_id(span_id) in active_span_ids for span_id in entity.span_ids
                ))
            ]
            call_trace = trace["chunks"][chunk_index]
            batch_size = max(1, self.config.relation_batch_size)
            for batch_start in range(0, len(pairs), batch_size):
                pair_batch = pairs[batch_start : batch_start + batch_size]
                prompt = build_relation_prompt(
                    rows,
                    pair_batch,
                    _span_rows(document, chunk),
                    emit_none=self.config.relation_emit_none,
                )
                try:
                    response = self.relation_backend.generate(
                        prompt,
                        max_new_tokens=self.config.max_new_tokens_relations,
                    )
                    decisions = parse_relation_decisions(response)
                    pair_map = {pair["pair_id"]: pair for pair in pair_batch}
                    for decision in decisions:
                        pair = pair_map.get(decision["pair_id"])
                        if pair is None:
                            continue
                        if decision["relation"] not in pair["allowed_relations"]:
                            continue
                        evidence_id = decision.get("evidence_span_id", "")
                        evidence_span = resolve_span_ref(document, evidence_id)
                        if evidence_span is None or _base_span_id(evidence_id) not in active_span_ids:
                            continue
                        relations.append(
                            Relation(
                                source=pair["source"],
                                target=pair["target"],
                                relation=decision["relation"],
                                evidence_ids=[evidence_id],
                                evidence_text=[evidence_span.text],
                            )
                        )
                    call_trace["relation_decision_count"] = call_trace.get("relation_decision_count", 0) + len(decisions)
                    call_trace["relation_call_count"] = call_trace.get("relation_call_count", 0) + 1
                except Exception as exc:
                    call_trace.setdefault("relation_errors", []).append(f"{type(exc).__name__}: {exc}")

        # Stable exact dedup before validation.
        unique: dict[tuple[str, str, str], Relation] = {}
        for relation in relations:
            key = (relation.source, relation.target, relation.relation)
            if key not in unique:
                unique[key] = relation
            else:
                existing = unique[key]
                for evidence_id, evidence_text in zip(relation.evidence_ids, relation.evidence_text):
                    if evidence_id not in existing.evidence_ids:
                        existing.evidence_ids.append(evidence_id)
                    if evidence_text not in existing.evidence_text:
                        existing.evidence_text.append(evidence_text)
        graph = _filter_valid_relations(Graph(entities, list(unique.values())), document, trace)
        validation = validate_graph(graph, document)
        trace["mode"] = "entity_llm_relation_classifier" if self.relation_classifier is not None else "two_stage_llm_pipeline"
        trace["coverage"] = {
            "entities": len(graph.entities),
            "relations": len(graph.relations),
            "grounded_mentions": sum(len(entity.span_ids) for entity in graph.entities),
        }
        return PipelineResult(
            graph=graph,
            output=assemble_graph(graph, self.config.assembler),
            validation=validation,
            trace=trace,
        )


class TransformersBackend:
    """Optional Hugging Face backend; imported lazily for deterministic tests."""

    def __init__(
        self,
        base_model: str,
        adapter: str | None = None,
        *,
        load_in_4bit: bool = False,
        device_map: str | dict[str, int] = "auto",
        adapter_name: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": device_map}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter,
                adapter_name=adapter_name or "default",
            )
            self.model.set_adapter(adapter_name or "default")
        self.adapter_name = adapter_name
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def attach_adapter(self, adapter: str, adapter_name: str) -> None:
        """Load another LoRA onto this base model without duplicating the base."""
        if not adapter:
            raise ValueError("adapter path is required")
        if not hasattr(self.model, "load_adapter"):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter,
                adapter_name=adapter_name,
            )
        else:
            self.model.load_adapter(adapter, adapter_name=adapter_name)
        self.model.eval()

    def with_adapter(self, adapter_name: str) -> "TransformersBackend":
        """Return a lightweight view sharing this backend's model and tokenizer."""
        view = object.__new__(TransformersBackend)
        view.tokenizer = self.tokenizer
        view.model = self.model
        view.adapter_name = adapter_name
        return view

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if self.adapter_name is not None:
            if not hasattr(self.model, "set_adapter"):
                raise RuntimeError(f"model has no PEFT adapters; cannot select {self.adapter_name!r}")
            self.model.set_adapter(self.adapter_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        class JsonObjectStop(StoppingCriteria):
            """Stop once the generated suffix contains one complete JSON object."""

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                generated = input_ids[0, inputs["input_ids"].shape[1] :]
                text = self_tokenizer.decode(generated, skip_special_tokens=False)
                try:
                    extract_json_object(text)
                except ValueError:
                    return False
                return True

        self_tokenizer = self.tokenizer
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([JsonObjectStop()]),
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=False).strip()


def dump_result(result: PipelineResult, path: str) -> None:
    payload = {"output": result.output, "validation": result.validation.__dict__, "trace": result.trace}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

"""Versioned raw-source pipeline with finite LM choices in both model stages.

JSON is read/written by Python only. Old JSON adapters remain on run_pipeline.py;
they are rejected here unless retrained and supplied with the new receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path

from build_stage_data import _raw_user_payload
from generative.protocol import ProtocolConfig
from kg_agent.choice_backend import (
    ChoiceEntitySelector, ChoiceGenerator, ChoiceRelationSelector, check_adapter_contract,
)
from kg_agent.pipeline import PipelineConfig, TransformersBackend, TwoStageKGAgent
from kg_agent.source import SourceRouter


class NoTextGeneration:
    def generate(self, *args, **kwargs):
        raise AssertionError("Bounded-choice pipeline must never call the legacy JSON generator")


def build_agent(args):
    if not args.base_model:
        raise ValueError("Free text requires --base-model")
    relation_format = getattr(args, "relation_format", "choices")
    if relation_format == "labels":
        from kg_agent.label_backend import LabelGenerator, LabelRelationSelector
        from kg_agent.label_backend import check_adapter_contract as check_relation_contract
        relation_generator_type, relation_selector_type = LabelGenerator, LabelRelationSelector
    else:
        check_relation_contract = check_adapter_contract
        relation_generator_type, relation_selector_type = ChoiceGenerator, ChoiceRelationSelector
    for stage, path in (("entities", args.entity_adapter), ("relations", args.relation_adapter)):
        if not path:
            raise ValueError(f"Free text requires a trained {stage} choice adapter")
        checker = check_adapter_contract if stage == "entities" else check_relation_contract
        checker(path, stage, base_model=args.base_model)
    backend = TransformersBackend(args.base_model, args.entity_adapter, adapter_name="entities",
                                  load_in_4bit=args.load_in_4bit)
    if Path(args.relation_adapter).resolve() == Path(args.entity_adapter).resolve():
        relation_view = backend
    else:
        backend.attach_adapter(args.relation_adapter, "relations")
        relation_view = backend.with_adapter("relations")
    return TwoStageKGAgent(
        NoTextGeneration(),
        config=PipelineConfig(max_entity_candidates=args.max_entity_candidates,
                              max_spans_per_chunk=args.max_spans_per_chunk,
                              max_chars_per_chunk=args.max_chars_per_chunk),
        entity_selector=ChoiceEntitySelector(ChoiceGenerator(
            backend, max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_entity_tokens)),
        relation_classifier=relation_selector_type(relation_generator_type(
            relation_view, max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_relation_tokens),
            config=ProtocolConfig(pairs_per_batch=args.pairs_per_batch, order=args.order, context=args.context)),
    )


def main(argv=None, *, relation_format="choices"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL raw sources or {text, metadata}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--entity-adapter")
    parser.add_argument("--relation-adapter")
    parser.add_argument("--relation-format", choices=("choices", "labels"), default=relation_format)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-entity-tokens", type=int, default=2048)
    parser.add_argument("--max-relation-tokens", type=int, default=2048)
    parser.add_argument("--max-entity-candidates", type=int, default=64)
    parser.add_argument("--max-spans-per-chunk", type=int, default=8)
    parser.add_argument("--max-chars-per-chunk", type=int, default=2000)
    parser.add_argument("--pairs-per-batch", type=int, default=4)
    parser.add_argument("--order", choices=("source", "hash"), default="source")
    parser.add_argument("--context", choices=("full", "local"), default="full")
    args = parser.parse_args(argv)
    for key in ("max_input_tokens", "max_entity_tokens", "max_relation_tokens", "max_entity_candidates",
                "max_spans_per_chunk", "max_chars_per_chunk", "pairs_per_batch"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; preserve previous runs")
    values = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError("Empty input")
    inputs = []
    for value in values:
        metadata = value.get("metadata") or {} if isinstance(value, dict) else {}
        raw = value["text"] if isinstance(value, dict) and "text" in value else value
        raw = _raw_user_payload(raw) if isinstance(raw, str) else raw
        inputs.append((raw, metadata))
    router = SourceRouter()
    # Validate/load once. Configuration errors must not masquerade as model
    # failures on every sample. ICD-only input stays entirely CPU/deterministic.
    agent = (build_agent(args) if any(router.route(raw, meta).source_type == "free_text" for raw, meta in inputs)
             else TwoStageKGAgent(NoTextGeneration()))
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output.open("x") as handle:
        for index, (raw, metadata) in enumerate(inputs):
            identity = hashlib.sha256(json.dumps([raw, metadata], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            try:
                result = agent.run(raw, metadata=metadata)
                item = {"status": "complete", "output": result.output, "trace": result.trace}
            except Exception as exc:
                failures += 1
                # Preserve the row: scorers must retain all its gold targets as
                # FN. A failed stage is not a successful no-relation decision.
                item = {"status": "failed", "output": {"entities": [], "relations": []},
                        "error": f"{type(exc).__name__}: {exc}", "trace": getattr(exc, "trace", {})}
            item.update(sample_index=index, input_sha256=identity)
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
    print(json.dumps({"samples": len(inputs), "failed_samples": failures, "output": str(output)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

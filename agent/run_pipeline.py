#!/usr/bin/env python3
"""Run the deterministic ICD parser or two-stage LLM pipeline on JSONL input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kg_agent.pipeline import PipelineConfig, TransformersBackend, TwoStageKGAgent
from build_stage_data import _raw_user_payload


class NoLLMBackend:
    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        raise RuntimeError("no --base-model was supplied for a free-text sample")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL; each line is a raw source or {text, metadata}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--entity-adapter")
    parser.add_argument("--relation-adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens-entities", type=int, default=1024)
    parser.add_argument("--max-new-tokens-relations", type=int, default=768)
    parser.add_argument("--max-entity-candidates", type=int, default=64)
    parser.add_argument("--max-pairs-per-chunk", type=int, default=128)
    parser.add_argument("--max-spans-per-chunk", type=int, default=16)
    parser.add_argument("--max-chars-per-chunk", type=int, default=6000)
    parser.add_argument(
        "--entity-mention-copy",
        action="store_true",
        help="Ask the entity model to copy an exact source mention for deterministic grounding",
    )
    parser.add_argument(
        "--relation-emit-none",
        action="store_true",
        help="Ask the relation model to emit an explicit NONE decision for every candidate pair",
    )
    parser.add_argument("--relation-batch-size", type=int, default=128)
    args = parser.parse_args()

    backend = None
    if args.base_model:
        if args.entity_adapter and args.relation_adapter:
            # Keep one 8B base model resident and switch only the small LoRA
            # adapter between the entity and relation stages. Loading two
            # independent bases can exhaust an 80 GB GPU before inference.
            backend = TransformersBackend(
                args.base_model,
                args.entity_adapter,
                load_in_4bit=args.load_in_4bit,
                adapter_name="entities",
            )
            backend.attach_adapter(args.relation_adapter, "relations")
            relation_backend = backend.with_adapter("relations")
        else:
            backend = TransformersBackend(args.base_model, args.entity_adapter, load_in_4bit=args.load_in_4bit)
            relation_backend = backend
            if args.relation_adapter:
                relation_backend = TransformersBackend(args.base_model, args.relation_adapter, load_in_4bit=args.load_in_4bit)
        config = PipelineConfig(
            max_new_tokens_entities=args.max_new_tokens_entities,
            max_new_tokens_relations=args.max_new_tokens_relations,
            max_entity_candidates=args.max_entity_candidates,
            max_pairs_per_chunk=args.max_pairs_per_chunk,
            max_spans_per_chunk=args.max_spans_per_chunk,
            max_chars_per_chunk=args.max_chars_per_chunk,
            entity_mention_copy=args.entity_mention_copy,
            relation_emit_none=args.relation_emit_none,
            relation_batch_size=args.relation_batch_size,
        )
        agent = TwoStageKGAgent(backend, relation_backend, config=config)
    else:
        # This keeps the CLI useful for an ICD-only smoke test. Free-text rows
        # will be traced as isolated LLM backend errors rather than crashing
        # the whole JSONL run.
        config = PipelineConfig(
            max_new_tokens_entities=args.max_new_tokens_entities,
            max_new_tokens_relations=args.max_new_tokens_relations,
            max_entity_candidates=args.max_entity_candidates,
            max_pairs_per_chunk=args.max_pairs_per_chunk,
            max_spans_per_chunk=args.max_spans_per_chunk,
            max_chars_per_chunk=args.max_chars_per_chunk,
            entity_mention_copy=args.entity_mention_copy,
            relation_emit_none=args.relation_emit_none,
            relation_batch_size=args.relation_batch_size,
        )
        agent = TwoStageKGAgent(NoLLMBackend(), config=config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value: Any = json.loads(line)
            if isinstance(value, dict) and "text" in value:
                raw = _raw_user_payload(str(value["text"]))
                metadata = value.get("metadata") or {}
            else:
                raw = _raw_user_payload(value) if isinstance(value, str) else value
                metadata = {}
            result = agent.run(raw, metadata=metadata)
            handle.write(json.dumps({"output": result.output, "trace": result.trace}, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()

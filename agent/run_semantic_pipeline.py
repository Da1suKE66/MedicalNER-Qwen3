"""Run actual semantic entity -> relation inference on one original input.

Models emit escaped semantic TAB records. Python validates them and writes the
external graph JSON. This entry never reads gold, assistant graphs or an oracle
entity table. Raw input and generation traces stay in a separate 0600 file.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

from kg_agent.assemble import assemble_graph
from kg_agent.source import SourceRouter, parse_icd_graph
from semantic.data import VARIANTS, canonical, sha
from semantic.protocol import CONTRACT
from semantic.runtime import ENTITY_VARIANTS, SemanticGenerator, entity_strategy_contract, run_document, verify_title_context


def _input(path, observer=None):
    payload = Path(path).read_bytes()
    text = payload.decode("utf-8")  # read_text would translate CRLF offsets.
    if observer:
        observer({"text": text, "file_sha256": sha(payload)})
    if not text.strip():
        raise ValueError("Source input cannot be empty")
    parsed = None
    if text.lstrip().startswith(("{", "[")):
        def unique_fields(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate structured input field")
                result[key] = value
            return result
        def finite_constant(value):
            raise ValueError("Nonfinite structured input value")
        parsed = json.loads(text, object_pairs_hook=unique_fields, parse_constant=finite_constant)
        if not isinstance(parsed, dict):
            raise ValueError("Structured input must be one original source object")
        if set(parsed) & {"messages", "gold", "assistant", "original_record", "entities", "relations"}:
            raise ValueError("Chat records, gold and extracted graphs are not original source inputs")
        if SourceRouter().route(text).source_type != "structured_icd":
            raise ValueError("JSON input must be an original structured ICD record")
    return text, sha(payload), parsed


def title_context(raw, *, source_group_id, parsed=None):
    """Bind an explicitly supplied same-group header to its exact input slice.

    Group membership is the caller's explicit declaration, not a relationship
    inferred from matching disease names. A conflicting embedded group fails.
    """
    if not isinstance(source_group_id, str) or not source_group_id.strip():
        raise ValueError("A title source requires an explicit --source-group-id")
    group = source_group_id.strip()
    routed = SourceRouter().route(raw)
    if routed.source_type != "structured_icd":
        raise ValueError("Title source must be an original structured ICD header")
    for key in ("source_group_id", "group_id"):
        if isinstance(parsed, dict) and key in parsed and str(parsed[key]) != group:
            raise ValueError("Embedded title source group conflicts with the declared group")
    title = routed.metadata.get("title")
    if not isinstance(title, str) or not title or title not in raw:
        raise ValueError("Structured title must have a literal exact slice in its original input")
    start = raw.index(title)
    # This user-only wrapper gives the shared runtime verifier its existing
    # source receipt shape. No assistant response or graph is constructed/read.
    original = {"messages": [{"role": "user", "content": raw}]}
    digest = sha(original)
    context = {"text": title, "source_group_id": group,
               "source_record_sha256": digest, "source_start": start,
               "source_end": start + len(title)}
    receipts = {digest: {"source_group_id": group, "original_record": original}}
    verified = verify_title_context({"title_context": context}, receipts, source_group_id=group)
    if verified != context:
        raise ValueError("Shared runtime did not verify the exact title context")
    return context, receipts


def _trace_path(output):
    output = Path(output)
    return output.with_name(output.name + ".trace.json")


def _write_private(path, value):
    """Create exclusively; no overwrite, symlink following or permissive mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute(args, *, generator_factory=None):
    """Return result envelope, full private trace, and a process exit code."""
    generator_factory = generator_factory or SemanticGenerator
    entity_variant = getattr(args, "entity_variant", "full_source")
    trace = {"output_contract": CONTRACT, "status": "running", "started_unix": time.time(),
             "variant": args.variant, "entity_variant": entity_variant, "entity_strategy": None,
             "source_input": None, "title_source_input": None,
             "generation_events": [], "result": None, "exception": None}
    envelope = {"output_contract": CONTRACT, "status": "failed", "variant": args.variant,
                "entity_variant": entity_variant, "entity_strategy": None,
                "graph": None, "source_type": None, "input_identity": None,
                "trace_file": str(_trace_path(args.output)), "error": None}
    try:
        raw, digest, parsed = _input(args.source_file, lambda value: trace.update(source_input=value))
        trace["source_input"] = {"text": raw, "file_sha256": digest}
        routed = SourceRouter().route(raw)
        envelope["source_type"] = routed.source_type
        metadata = {"source_type": routed.source_type}
        if args.source_group_id:
            metadata["source_group_id"] = args.source_group_id.strip()
        title_file_sha = None
        if args.title_source_file:
            if routed.source_type == "structured_icd":
                raise ValueError("A structured primary source already supplies its own title")
            title_raw, title_file_sha, title_parsed = _input(args.title_source_file,
                lambda value: trace.update(title_source_input=value))
            trace["title_source_input"] = {"text": title_raw, "file_sha256": title_file_sha}
            context, receipts = title_context(title_raw, source_group_id=args.source_group_id,
                                               parsed=title_parsed)
            metadata["title_context"] = context
            trace["title_source_receipts"] = receipts
            trace["group_association"] = {"source_group_id": context["source_group_id"],
                "authority": "caller_explicit_same_group_declaration",
                "independent_cross_file_group_membership_proved": False}
        identity = {"source_file_sha256": digest, "source_text_sha256": sha(raw),
                    "source_group_id": metadata.get("source_group_id"),
                    "title_source_file_sha256": title_file_sha,
                    "inference_input_sha256": sha({"text": raw, "metadata": metadata})}
        envelope["input_identity"] = identity
        trace["input_identity"], trace["metadata"] = identity, deepcopy(metadata)
        if routed.source_type == "structured_icd":
            keys = {re.sub(r"[\s\-]+", "_", str(key).strip()).casefold(): value for key, value in routed.metadata.items()}
            if not any(isinstance(keys.get(key), str) and keys[key].strip()
                       for key in ("title", "canonical_title", "name", "label")):
                raise ValueError("Structured source needs an explicit nonempty title/name field")
            graph = assemble_graph(parse_icd_graph(routed))
            if not graph["entities"] or graph["entities"][0]["name"] == "Unnamed disease":
                raise ValueError("Structured source has no explicit canonical title")
            result = {"raw_graph": graph, "filtered_graph": deepcopy(graph), "generations": [],
                "trace": {"generation_complete": True, "generation_calls": 0, "failed_calls": 0,
                          "entity_source": "deterministic_structured_icd",
                          "parser": "kg_agent.source.parse_icd_graph"}}
            strategy = {**entity_strategy_contract(entity_variant, short_chars=args.short_chars),
                        "entity_strategy_applied": False, "reason": "deterministic_structured_icd_no_entity_adapter"}
        else:
            if not all((args.base_model, args.entity_adapter, args.relation_adapter)):
                raise ValueError("Free-text inference requires --base-model, --entity-adapter and --relation-adapter")
            generator = generator_factory(args.base_model, args.entity_adapter, args.relation_adapter,
                max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size)
            trace["adapter_receipts"] = deepcopy(getattr(generator, "receipts", {}))
            trace["adapter_weight_verification"] = deepcopy(getattr(generator, "weight_verification", {}))
            receipt = trace["adapter_receipts"].get("entities")
            if not isinstance(receipt, dict):
                raise ValueError("Live entity inference requires its verified training receipt")
            strategy = {**entity_strategy_contract(entity_variant, receipt, short_chars=args.short_chars),
                        "entity_strategy_applied": True}
            envelope["entity_strategy"], trace["entity_strategy"] = deepcopy(strategy), deepcopy(strategy)
            deadline = time.monotonic() + args.timeout_seconds if args.timeout_seconds else None
            result = run_document(raw, metadata, generator, variant=args.variant, entity_variant=entity_variant, mode="live",
                short_chars=args.short_chars, deadline=deadline,
                on_generation=lambda event: trace["generation_events"].append(deepcopy(event)))
        envelope["entity_strategy"], trace["entity_strategy"] = deepcopy(strategy), deepcopy(strategy)
        result["trace"]["entity_variant"] = entity_variant
        result["trace"]["entity_strategy"] = deepcopy(strategy)
        trace["result"] = result
        complete = result["trace"].get("generation_complete") is True
        graph = result["filtered_graph"]
        envelope["summary"] = {"entities": len(graph["entities"]), "relations": len(graph["relations"]),
            "raw_entities": len(result["raw_graph"]["entities"]),
            "raw_relations": len(result["raw_graph"]["relations"]),
            "generation_calls": result["trace"].get("generation_calls", 0),
            "failed_calls": result["trace"].get("failed_calls", 0),
            "generation_complete": complete, "empty_result": not graph["entities"] and not graph["relations"]}
        envelope["summary"]["entities_removed_by_rules"] = len(result["raw_graph"]["entities"]) - len(graph["entities"])
        envelope["summary"]["relations_removed_by_rules"] = len(result["raw_graph"]["relations"]) - len(graph["relations"])
        if envelope["summary"]["empty_result"]:
            envelope["summary"]["empty_result_reason"] = ("all_predictions_removed_by_validation"
                if result["raw_graph"]["entities"] or result["raw_graph"]["relations"]
                else "complete_empty_entity_sequence" if complete else "incomplete_generation")
        if complete:
            envelope["status"], envelope["graph"] = "complete", graph
            code = 0
        else:
            envelope["partial_graph"] = graph
            envelope["error"] = "incomplete_generation_or_invalid_semantic_records"
            code = 1
        trace["status"] = envelope["status"]
    except BaseException as exc:
        # Do not print exception messages: tokenizer/parser errors may quote
        # private input. Full details remain in the private trace artifact.
        trace["status"] = "failed"
        trace["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        envelope["error"] = type(exc).__name__
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    trace["finished_unix"] = time.time()
    return envelope, trace, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True, help="One original UTF-8 source, plain text or structured ICD JSON/header")
    parser.add_argument("--base-model")
    parser.add_argument("--entity-adapter")
    parser.add_argument("--relation-adapter")
    parser.add_argument("--output", required=True, help="New graph envelope JSON; raw trace is written beside it with mode 0600")
    parser.add_argument("--variant", choices=VARIANTS, default="full_source")
    parser.add_argument("--entity-variant", choices=ENTITY_VARIANTS, default="full_source",
                        help="Entity input view; must match the adapter's declared training input view")
    parser.add_argument("--title-source-file", help="Explicit same-group original structured header; never an assistant graph")
    parser.add_argument("--source-group-id", help="Explicit caller declaration that primary input and title source belong to this group")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--short-chars", type=int, default=2400)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args(argv)
    if args.timeout_seconds is not None and not 0 < args.timeout_seconds < float("inf"):
        parser.error("--timeout-seconds must be a finite positive duration")
    output, trace_path = Path(args.output), _trace_path(args.output)
    if output.exists() or output.is_symlink() or trace_path.exists() or trace_path.is_symlink():
        print("Refusing to overwrite an existing output or private trace.", file=sys.stderr)
        return 2
    inputs = [Path(args.source_file)] + ([Path(args.title_source_file)] if args.title_source_file else [])
    if any(p.resolve() in (output.resolve(), trace_path.resolve()) for p in inputs):
        print("Output paths must differ from input paths.", file=sys.stderr)
        return 2
    result, trace, code = execute(args)
    try:
        _write_private(trace_path, trace)
        _write_private(output, result)
    except OSError as exc:
        print(f"Cannot save private result artifacts ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(canonical({"status": result["status"], "output": str(output), "trace_file": str(trace_path),
                     "source_type": result["source_type"], "summary": result.get("summary", {}),
                     "error": result["error"]}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

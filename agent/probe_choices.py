"""Finite GPU probe of an artificial held-out bounded-choice contract.

Component metrics use each authored input task. The separate raw-source pipeline
smoke uses model-predicted entities and input-derived pairs, so its output must
never be merged into the component F1. No thresholds or training are adjusted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from generative.choice_data import compile_recovery_row, validate_row
from generative.protocol import ProtocolConfig
from generative.recovery_preferences import verify_case
from kg_agent.choice_backend import (
    ChoiceEntitySelector, ChoiceFailure, ChoiceGenerator, ChoiceRelationSelector,
    check_adapter_contract,
)
from kg_agent.choice_protocol import ChoiceProtocolError
from kg_agent.entity_choices import EntityChoiceTask
from kg_agent.pipeline import PipelineConfig, TwoStageKGAgent
from kg_agent.source import SourceRouter


SCOPE = "artificial_contract_only_not_medical"
MAX_PIPELINE_CALLS = 16


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_sources(path, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < limit:
        raise ValueError("Not enough held-out rows for the requested limit")
    seen = set()
    for row in rows:
        verify_case(row)
        if row["split"] != "mechanical_holdout" or row["training_view"] != "preference":
            raise ValueError("Probe accepts only raw recipe-verified mechanical_holdout preferences")
        if row["id"] in seen:
            raise ValueError("Duplicate held-out source ID")
        seen.add(row["id"])
        for stage in ("entities", "relations"):
            validate_row(compile_recovery_row(row, stage), require_train=False)
    return rows[:limit]


def _counts(gold, predicted):
    return {"tp": len(gold & predicted), "fp": len(predicted - gold), "fn": len(gold - predicted)}


def _metrics(counts):
    tp, fp, fn = (counts.get(key, 0) for key in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "gold_count": tp + fn, "predicted_count": tp + fp}


def _sets(task, actions, stage):
    if stage == "entities":
        return {(item.span_id, item.label) for item in task.candidates(actions)}, set()
    edges = task.edges(actions)
    return ({(edge["pair_id"], edge["relation"]) for edge in edges},
            {(edge["pair_id"], edge["relation"], span)
             for edge in edges for span in edge["evidence_span_ids"]})


def _failure(exc, *, protocol_phase=False):
    trace = deepcopy(getattr(exc, "trace", {}))
    if not isinstance(trace, dict):
        trace = {"exception_trace": str(trace)}
    grammar = (isinstance(exc, (ChoiceProtocolError, ValueError)) or protocol_phase
               or bool(trace.get("error")) or trace.get("complete") is False)
    return {"error": f"{type(exc).__name__}: {exc}", "trace": trace,
            "grammar_failure": grammar,
            "budget_exhausted": bool(trace.get("budget_exhausted"))}


class _NoFreeGeneration:
    def generate(self, *args, **kwargs):
        raise AssertionError("Probe must not call a free-text/JSON generator")


class _PipelineBudget:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = []

    def run(self, task):
        if len(self.calls) >= MAX_PIPELINE_CALLS:
            raise ChoiceFailure("Pipeline generation-call budget exhausted; candidates were not truncated",
                                {"call_budget_exhausted": True, "max_pipeline_calls": MAX_PIPELINE_CALLS})
        record = {"stage": "entities" if isinstance(task, EntityChoiceTask) else "relations"}
        self.calls.append(record)
        try:
            actions, trace = self.delegate.run(task)
            record.update(status="complete", actions=list(actions), trace=deepcopy(trace))
            return actions, trace
        except Exception as exc:
            record.update(status="failed", **_failure(exc))
            raise


def pipeline_smoke(source, generator):
    """Run one original source through both selectors; score no component F1."""
    raw = source["oracle_source_text"]
    document = SourceRouter().route(raw)
    limited = _PipelineBudget(generator)
    agent = TwoStageKGAgent(
        _NoFreeGeneration(),
        config=PipelineConfig(max_spans_per_chunk=max(1, len(document.spans)),
                              max_chars_per_chunk=max(1, len(raw)), max_entity_candidates=64),
        entity_selector=ChoiceEntitySelector(limited),
        relation_classifier=ChoiceRelationSelector(limited, config=ProtocolConfig(
            pairs_per_batch=4, order="source", context="full")),
    )
    record = {"stage": "pipeline_smoke", "sample_index": 0, "source_id": source["id"],
              "source_row_sha256": _digest(source), "scope": SCOPE,
              "candidate_policy": "all_input_derived_pairs_from_predicted_entities",
              "max_generation_calls": MAX_PIPELINE_CALLS, "not_in_component_metrics": True}
    try:
        result = agent.run(raw)
        record.update(status="complete", output=result.output, trace=result.trace,
                      validation=asdict(result.validation), grammar_failure=False,
                      engineering_pass=result.validation.valid and result.trace.get("mode") == "two_stage_bounded_choices")
    except Exception as exc:
        record.update(status="failed", output={"entities": [], "relations": []},
                      engineering_pass=False, **_failure(exc))
    record["generation_calls"] = limited.calls
    record["actions"] = [call.get("actions", []) for call in limited.calls]
    return record


def run_probe(sources, generator):
    """Attempt every requested component even after generation failures."""
    if not sources:
        raise ValueError("Probe needs at least one held-out source")
    # Public callers obey the same admission rule as the CLI before any inference.
    compiled, seen = [], set()
    for source in sources:
        verify_case(source)
        if source["split"] != "mechanical_holdout" or source["training_view"] != "preference":
            raise ValueError("Probe requires mechanical_holdout source rows")
        if source["id"] in seen:
            raise ValueError("Duplicate held-out source ID")
        seen.add(source["id"])
        compiled.append([(stage, row, validate_row(row, require_train=False))
                         for stage in ("entities", "relations")
                         for row in [compile_recovery_row(source, stage)]])
    predictions = []
    totals = {stage: Counter() for stage in ("entities", "relations", "relation_evidence")}
    failures, grammar_failures, budgets = Counter(), 0, 0
    for index, tasks in enumerate(compiled):
        source = sources[index]
        for stage, row, task in tasks:
            gold, gold_evidence = _sets(task, row["actions"], stage)
            predicted, predicted_evidence = set(), set()
            record = {"stage": stage, "sample_index": index, "source_id": source["id"],
                      "source_row_sha256": _digest(source), "scope": SCOPE, "actions": []}
            protocol_phase = False
            try:
                actions, trace = generator.run(task)
                record.update(actions=list(actions), trace=deepcopy(trace))
                if trace.get("error") or trace.get("complete") is False:
                    raise ChoiceFailure("Generator returned an incomplete/invalid task", trace)
                protocol_phase = True
                predicted, predicted_evidence = _sets(task, actions, stage)
                record.update(status="complete", grammar_failure=False)
            except Exception as exc:
                detail = _failure(exc, protocol_phase=protocol_phase)
                if not detail["trace"] and record.get("trace"):
                    detail["trace"] = record["trace"]
                record.update(status="failed", **detail)
                # Keep partial actions for diagnosis only; a failed integer task
                # is never converted to a successful sparse/empty graph.
                record["actions"] = detail["trace"].get("actions", record["actions"])
                failures[stage] += 1
                grammar_failures += detail["grammar_failure"]
                budgets += detail["budget_exhausted"]
            counts = _counts(gold, predicted)
            totals[stage].update(counts)
            record["counts"] = counts
            record["gold_count"] = len(gold)
            record["predicted_keys"] = sorted(predicted)
            if stage == "relations":
                evidence_counts = _counts(gold_evidence, predicted_evidence)
                totals["relation_evidence"].update(evidence_counts)
                record["evidence_counts"] = evidence_counts
            predictions.append(record)
    pipe = pipeline_smoke(sources[0], generator)
    predictions.append(pipe)
    grammar_failures += bool(pipe["grammar_failure"])
    all_attempted = len(predictions) - 1 == len(sources) * 2
    engineering = all_attempted and not sum(failures.values()) and not grammar_failures and pipe["engineering_pass"]
    report = {
        "status": "complete", "status_meaning": "all requested component and pipeline attempts finished; not a model-correctness claim",
        "scope": SCOPE, "source_samples": len(sources), "requested_component_attempts": len(sources) * 2,
        "completed_component_attempts": len(predictions) - 1,
        "components": {stage: {**_metrics(counts), "samples": len(sources),
                              "failed_samples": failures.get("relations" if stage == "relation_evidence" else stage, 0)}
                       for stage, counts in totals.items()},
        "metric_contract": {
            "aggregation": "micro sums of per-sample set TP/FP/FN; all failed rows retained as full gold FN",
            "entities": "exact source span reference plus label; importance not scored",
            "relations": "fixed candidate pair ID plus relation label (equivalent to a directed triple within its task)",
            "relation_evidence": "fixed pair ID, relation label, evidence span ID",
            "empty_denominator": "precision, recall and F1 are zero when their denominator is zero",
            "pipeline_smoke": "engineering only; different candidate construction, excluded from component metrics",
        },
        "pipeline_smoke": {key: pipe.get(key) for key in ("status", "engineering_pass", "grammar_failure", "error", "validation")},
        "probe_assertions": {"all_requested_attempted": all_attempted,
            "grammar_failures": grammar_failures, "component_failures": sum(failures.values()),
            "component_budget_exhaustions": budgets,
            "grammar_failure_definition": "illegal choice path or incomplete integer task; includes token-budget termination",
            "engineering_pass": bool(engineering)},
        "medical_quality_evaluated": False,
    }
    return report, predictions


def load_generator(args):
    receipts = {stage: check_adapter_contract(args.adapter, stage, base_model=args.base_model)
                for stage in ("entities", "relations")}
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This probe requires the authorized BF16 GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="default",
                                     is_trainable=False, local_files_only=True).to("cuda").eval()
    backend = SimpleNamespace(model=model, tokenizer=tokenizer, adapter_name="default")
    return ChoiceGenerator(backend, max_input_tokens=args.max_input_tokens,
                           max_new_tokens=args.max_new_tokens), receipts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args(argv)
    if any(getattr(args, key) < 1 for key in ("max_new_tokens", "max_input_tokens", "limit")):
        parser.error("Token budgets and limit must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Probe output must be new/empty; never overwrite prior results")
    sources = load_sources(args.data, args.limit)
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "loading", "scope": SCOPE, "arguments": vars(args),
              "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest()}
    metric_path = out / "metrics.json"
    metric_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        generator, receipts = load_generator(args)
        result, predictions = run_probe(sources, generator)
        report.update(result, adapter_receipts=receipts)
        (out / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      probe_assertions={"engineering_pass": False})
        raise
    finally:
        metric_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "scope": SCOPE,
                      "probe_assertions": report["probe_assertions"], "components": report["components"]}), flush=True)
    return report


if __name__ == "__main__":
    main()

"""Frozen medical oracle-entity screen with explicit relation output formats.

Separate from the artificial semantic holdout and from end-to-end evaluation.
All original gold triples, including unreachable endpoints, remain in scoring.
The model never emits JSON: Python serializes checked actions or literal labels
solely to reuse the established strict graph assembler and full-denominator
scorer. Historical integer choices remain the default.
"""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

from build_stage_data import _message
from evaluate import _graph
from evaluate_graphs import evaluate_records
from evaluation_identity import IDENTITY_KEY
from evaluate_choice_relations import load_generator, _check_format, checked_label_result
from generative.evaluate import run_records
from generative.protocol import dumps
from generative.rapid_screen import load_screen, sha_file
from kg_agent.choice_backend import ChoiceFailure, check_adapter_contract
from kg_agent.relation_choices import RelationChoiceTask
from kg_agent.relation_labels import RelationLabelTask
from kg_agent import label_backend


SCOPE = "frozen_medical_oracle_relation_subset_not_end_to_end"


def run_medical(records, indices, config, generator, *, relation_format="choices", max_seconds=None,
                on_generation=None, clock=time.monotonic):
    _check_format(relation_format)
    started = clock()
    def generate(batch):
        task = (RelationChoiceTask if relation_format == "choices" else RelationLabelTask)(batch)
        trace, actions, edges, error = {}, [], [], None
        attempted = False
        began = clock()
        try:
            if max_seconds is not None and clock() - started >= max_seconds:
                raise ChoiceFailure("Medical evaluation deadline reached", {"deadline_exhausted": True})
            attempted = True
            result, trace = generator.run(task)
            if relation_format == "choices":
                actions = result
            if max_seconds is not None and clock() - started > max_seconds:
                raise ChoiceFailure("Medical generation exceeded deadline", {**trace, "deadline_exhausted": True})
            if trace.get("error") or trace.get("complete") is False:
                raise ChoiceFailure("Incomplete bounded relation actions" if relation_format == "choices"
                                    else "Incomplete literal-label relation response", trace)
            edges = task.edges(actions) if relation_format == "choices" else checked_label_result(task, result, trace)
        except Exception as exc:
            trace = deepcopy(getattr(exc, "trace", trace))
            if relation_format == "choices":
                actions = list(trace.get("actions", actions))
            error = f"{type(exc).__name__}: {exc}"
            edges = []
        decisions = {pair["pair_id"]: [] for pair in batch["pairs"]}
        for edge in edges:
            decisions[edge["pair_id"]].append({key: edge[key] for key in ("relation", "evidence_span_ids")})
        # This JSON is produced by Python from checked output. It is never an
        # LM response or a repaired failed response. Failure remains explicit.
        output = {"text": dumps({"decisions": [{"pair_id": pid, "relations": relations}
                                                 for pid, relations in decisions.items()]}),
                  "actions": list(actions), "choice_trace": trace,
                  "choice_prompt": task.prompt, "choice_prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                  "predicted_edges": edges, "generation_attempted": attempted,
                  "tokens": trace.get("generated_tokens", 0),
                  "truncated": bool(trace.get("budget_exhausted")),
                  "elapsed_seconds": max(0., clock() - began),
                  "serialization_owner": "python_checked_actions"}
        if relation_format == "labels":
            for key in ("actions", "choice_trace", "choice_prompt", "choice_prompt_sha256"):
                output.pop(key)
            output.update(output_contract=label_backend.CONTRACT, relation_format="labels",
                          completion_text=trace.get("completion_text"), label_trace=trace,
                          label_prompt=task.prompt, label_prompt_sha256=hashlib.sha256(task.prompt.encode()).hexdigest(),
                          serialization_owner="python_checked_literal_labels")
        if error:
            output["error"] = error
        if on_generation:
            on_generation({"id": batch["id"], "sample_id": batch["sample_id"],
                           IDENTITY_KEY: deepcopy(batch[IDENTITY_KEY]), **output})
        return output
    predictions, raw, audit = run_records(records, None, config, generate, oracle=True,
                                          strict_identity=True, sample_indices=indices)
    scores = evaluate_records(records, predictions, strict_identity=True, sample_indices=indices)
    for index, record, prediction in zip(indices, records, predictions):
        prediction.update(id=f"medical:{index}", sample_index=index,
                          gold=deepcopy(_graph(_message(record, "assistant"))))
        if relation_format == "labels":
            prediction.update(output_contract=label_backend.CONTRACT, relation_format="labels")
    trace_key = "choice_trace" if relation_format == "choices" else "label_trace"
    deadline = any(row[trace_key].get("deadline_exhausted") for row in raw)
    failed_batches = sum(bool(row.get("error")) for row in raw)
    report = {"status": "incomplete" if deadline else "complete", "complete": not deadline,
              "scope": SCOPE, "samples": len(records), "sample_indices": list(indices),
              "entity_source": "oracle", "not_full_tune": True, "not_full_live_pipeline": True,
              "relation": scores["free_text"]["relation"],
              "evidence": scores["free_text"]["evidence_exact"],
              "evidence_compatible": scores["free_text"]["evidence_compatible"],
              "evidence_definition": scores["definitions"]["evidence_exact"],
              "failed_samples": scores["free_text"]["relation_stage_error_samples"],
              "failed_batches": failed_batches, "generation_complete": failed_batches == 0,
              "generation_batches": len(raw), "generation_attempted_batches": sum(row["generation_attempted"] for row in raw),
              "budget_exhaustions": sum(row["truncated"] for row in raw),
              "errors": dict(Counter(row["error"] for row in raw if row.get("error"))),
              "coverage": audit, "per_relation": scores["free_text"]["per_relation"],
              "all_metrics": scores, "config": asdict(config), "elapsed_seconds": max(0., clock() - started),
              "failure_policy": "Failed batches contribute no predictions; every document retains all gold FN including unreachable entities",
              "serialization_owner": "Python; model selects only finite integer actions"}
    if relation_format == "labels":
        report.update(output_contract=label_backend.CONTRACT, relation_format="labels",
                      serialization_owner="Python; model emits literal relation labels and input references")
    denominator = report["relation"]["tp"] + report["relation"]["fn"]
    if denominator != audit["gold_triples"]:
        raise ValueError("Medical complete gold denominator mismatch")
    report["coverage"]["unreachable_gold_triples"] = denominator - audit["candidate_covered_triples"]
    return report, predictions, raw


def main(argv=None, *, relation_format="choices"):
    _check_format(relation_format)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pairs-per-batch", type=int, choices=(4, 16), default=4)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seconds", type=float, default=600.)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--relation-format", choices=("choices", "labels"), default=relation_format)
    args = parser.parse_args(argv)
    relation_format = args.relation_format
    if min(args.max_input_tokens, args.max_new_tokens, args.max_seconds) <= 0 or not args.max_seconds < float("inf"):
        parser.error("Positive finite token/time budgets are required")
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new immutable output directory")
    records, indices, configs, frozen = load_screen(args.screen_dir)
    config = replace(configs["full_source16"], pairs_per_batch=args.pairs_per_batch)
    check = check_adapter_contract if relation_format == "choices" else label_backend.check_adapter_contract
    receipt = check(args.adapter, "relations", base_model=args.base_model)
    manifest = {"status": "preflight_complete" if args.preflight_only else "running", "args": vars(args),
                "scope": SCOPE, "samples": len(records), "sample_indices": indices, "config": asdict(config),
                "screen_manifest_sha256": sha_file(Path(args.screen_dir) / "manifest.json"),
                "records_sha256": frozen["records_sha256"], "sample_indices_sha256": frozen["sample_indices_sha256"],
                "denominators": frozen["denominators"], "adapter_sha256": receipt["adapter_sha256"],
                "protocol_sha256": receipt["protocol_sha256"], "base_input_sha256": receipt["base_input_sha256"]}
    if relation_format == "labels":
        manifest.update(output_contract=label_backend.CONTRACT, relation_format="labels")
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save("run_manifest.json", manifest)
    if args.preflight_only:
        print(json.dumps(manifest))
        return
    try:
        generator, _ = load_generator(args) if relation_format == "choices" else load_generator(args, relation_format="labels")
        with (out / "generations.jsonl").open("w", encoding="utf-8") as stream:
            def append(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            report, predictions, raw = run_medical(records, indices, config, generator,
                relation_format=relation_format, max_seconds=args.max_seconds, on_generation=append)
        if (report["relation"]["tp"] + report["relation"]["fn"] != frozen["denominators"]["gold_relation_triples"]
                or report["coverage"]["candidate_pairs"] != frozen["denominators"]["candidate_pairs"]
                or report["coverage"]["candidate_covered_triples"] != frozen["conditions"]["full_source16"]["candidate_covered_gold_triples"]):
            raise ValueError("Medical frozen denominator or candidate coverage changed")
        (out / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
        report.update(screen_manifest_sha256=manifest["screen_manifest_sha256"],
                      adapter_sha256=manifest["adapter_sha256"],
                      predictions_sha256=sha_file(out / "predictions.jsonl"),
                      generations_sha256=sha_file(out / "generations.jsonl"))
        save("metrics.json", report)
        manifest.update(status=report["status"], complete=report["complete"], metrics_sha256=sha_file(out / "metrics.json"))
        save("run_manifest.json", manifest)
        print(json.dumps(report, ensure_ascii=False))
    except Exception as exc:
        manifest.update(status="failed", complete=False, error_type=type(exc).__name__, error=str(exc))
        save("run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

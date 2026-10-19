#!/usr/bin/env python3
"""Independently recount local medical choice outputs; public aggregates only."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "src")]

from build_stage_data import _message
from data_repair.source_provenance import source_view
from evaluate import _graph, _relation_set, _validate_gold_graph
from evaluate_graphs import evidence_by_triple
from evaluation_identity import IDENTITY_KEY, validate_alignment
from generative.evaluate import candidate_ceiling
from generative.protocol import build_batches
from generative.rapid_screen import load_screen
from kg_agent.pairwise import relation_rejection
from kg_agent.relation_choices import RelationChoiceTask


JOBS = ("parent_eval", "sft_eval", "dpo_eval")
RULES = {"symptom_disease_type_mismatch", "different_disease_in_evidence", "enumerated_siblings"}


class InvalidReport(ValueError):
    """Use fixed public error codes, never exception strings containing text."""


def need(condition, code):
    if not condition:
        raise InvalidReport(code)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def match(reported, actual, code):
    need(isinstance(reported, dict), code)
    for key in ("tp", "fp", "fn"):
        need(type(reported.get(key)) is int and reported[key] == actual[key], code)
    for key in ("precision", "recall", "f1"):
        value = reported.get(key)
        need(type(value) in (int, float) and math.isfinite(value)
             and math.isclose(value, actual[key], abs_tol=1e-10, rel_tol=1e-10), code)


def recount(records, predictions, indices):
    """Scoring reads actual edge sets, with original complete gold per sample."""
    validate_alignment(records, predictions, strict_identity=True, sample_indices=indices)
    totals, evidence = Counter(), Counter()
    failed, invalid, examples = 0, 0, []
    for index, record, prediction in zip(indices, records, predictions):
        gold = _graph(_message(record, "assistant"))
        _validate_gold_graph(gold)
        graph = _graph(prediction)
        g, p = _relation_set(gold), _relation_set(graph) if graph is not None else set()
        totals.update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
        ge, pe = evidence_by_triple(gold), evidence_by_triple(graph) if graph is not None else {}
        exact = sum(bool(ge[k] & pe.get(k, set())) for k in p & g)
        evidence.update(tp=exact, fp=len(p) - exact, fn=len(g) - exact)
        errors = [error for chunk in prediction.get("trace", {}).get("chunks", [])
                  for error in chunk.get("relation_errors", [])]
        failed += bool(errors)
        invalid += graph is None
        if (g != p or errors) and len(examples) < 5:
            examples.append({"sample_index": index, IDENTITY_KEY: prediction[IDENTITY_KEY],
                "source": _message(record, "user"), "gold": gold, "predicted": graph,
                "tp": len(g & p), "fp": len(p - g), "fn": len(g - p),
                "false_positive_triples": sorted(p - g), "false_negative_triples": sorted(g - p),
                "generation_errors": errors})
    return {"samples": len(records), "relation": prf(**totals), "evidence": prf(**evidence),
            "failed_samples": failed, "invalid_predictions": invalid}, examples


def recount_rules(records, predictions, indices):
    reports = {}
    for name, enabled in [("type_guard", {"symptom_disease_type_mismatch"}), ("all_legacy_rules", RULES)]:
        filtered = deepcopy(predictions)
        removed_tp = removed_fp = removed_rows = 0
        reasons = Counter()
        for record, prediction in zip(records, filtered):
            graph = _graph(prediction)
            if graph is None:
                continue
            before = _relation_set(graph)
            gold = _relation_set(_graph(_message(record, "assistant")))
            kept = []
            for edge in graph["relations"]:
                # Retain malformed edges as FP; the historical rule assumes a
                # well-formed mapping and must not silently discard failures.
                reason = relation_rejection(edge, graph["entities"]) if isinstance(edge, dict) else None
                if reason in enabled:
                    reasons[reason] += 1
                    removed_rows += 1
                else:
                    kept.append(edge)
            graph["relations"] = kept
            after = _relation_set(graph)
            removed = before - after  # duplicate raw rows count only once in F1
            removed_tp += len(removed & gold)
            removed_fp += len(removed - gold)
            prediction["output"] = graph
        counted, _ = recount(records, filtered, indices)
        reports[name] = {"relation": counted["relation"], "removed_tp": removed_tp,
                         "removed_fp": removed_fp, "removed_raw_rows": removed_rows,
                         "reasons": dict(reasons), "posthoc_rule_diagnostic": True}
    return reports


def verify_generations(records, predictions, indices, raw, config):
    """Bind serialized predictions to successful finite actions; failures are empty."""
    expected, by_index = [], {}
    for index, record in zip(indices, records):
        gold = _graph(_message(record, "assistant"))
        batches, _, _ = build_batches(source_view(_message(record, "user"))["raw_payload"],
                                     gold["entities"], config, sample_id=index)
        expected.extend((index, record, batch) for batch in batches)
        by_index[index] = {"entities": gold["entities"], "relations": []}
    need(len(raw) == len(expected), "generation_denominator_mismatch")
    need(len({row.get("id") for row in raw}) == len(raw), "duplicate_generation_id")
    failed = budgets = deadlines = 0
    failed_indices = set()
    for row, (index, record, batch) in zip(raw, expected):
        need(row.get("id") == batch["id"] and row.get("sample_id") == index, "generation_order_or_identity_mismatch")
        validate_alignment([record], [row], strict_identity=True, sample_indices=[index])
        need(row.get("serialization_owner") == "python_checked_actions", "unexpected_serialization_owner")
        trace = row.get("choice_trace", {})
        budgets += bool(trace.get("budget_exhausted"))
        deadlines += bool(trace.get("deadline_exhausted"))
        need(bool(row.get("truncated")) == bool(trace.get("budget_exhausted")), "generation_budget_mismatch")
        task = RelationChoiceTask(batch)
        if row.get("error"):
            failed += 1
            failed_indices.add(index)
            edges = []
        else:
            need(trace.get("complete") is not False and not trace.get("error"), "failed_trace_without_error")
            edges = task.edges(row["actions"])
        need(row.get("predicted_edges") == edges, "actions_edge_mismatch")
        pairs = {pair["pair_id"]: pair for pair in batch["pairs"]}
        for edge in edges:
            pair = pairs[edge["pair_id"]]
            by_index[index]["relations"].append({"source": pair["source"], "target": pair["target"],
                "relation": edge["relation"], "evidence": [batch["original_spans"][sid] for sid in edge["evidence_span_ids"]]})
    for index, prediction in zip(indices, predictions):
        graph = _graph(prediction)
        need(graph is not None, "prediction_graph_missing")
        need(graph["entities"] == by_index[index]["entities"], "oracle_entities_changed")
        need(_relation_set(graph) == _relation_set(by_index[index]), "prediction_actions_mismatch")
        need(evidence_by_triple(graph) == evidence_by_triple(by_index[index]), "prediction_evidence_mismatch")
    return {"failed_batches": failed, "budget_exhaustions": budgets, "generation_batches": len(raw),
            "failed_generation_samples": len(failed_indices), "deadline_exhaustions": deadlines}


def build_report(root, screen_dir):
    records, indices, configs, frozen = load_screen(screen_dir)
    denominator = sum(len(_relation_set(_graph(_message(record, "assistant")))) for record in records)
    need(denominator == frozen["denominators"]["gold_relation_triples"], "frozen_gold_denominator_mismatch")
    covered = candidates = 0
    for index, record in zip(indices, records):
        gold = _graph(_message(record, "assistant"))
        batches, _, audit = build_batches(source_view(_message(record, "user"))["raw_payload"],
                                         gold["entities"], configs["full_source16"], sample_id=index)
        covered += len(_relation_set(gold) & candidate_ceiling(gold, batches))
        candidates += audit["candidate_pairs"]
    need(covered == frozen["conditions"]["full_source16"]["candidate_covered_gold_triples"]
         and candidates == frozen["denominators"]["candidate_pairs"], "frozen_candidate_coverage_mismatch")
    screen_sha = sha(Path(screen_dir) / "manifest.json")
    jobs, private, comparison = [], {}, []
    for name in JOBS:
        directory = Path(root) / name
        summary = {"job": name, "status": "missing", "metric_verified": False}
        jobs.append(summary)
        if not directory.exists():
            continue
        required = [directory / name for name in ("run_manifest.json", "metrics.json", "predictions.jsonl", "generations.jsonl")]
        if not all(path.is_file() for path in required):
            summary.update(status="incomplete", validation_error="required_artifacts_missing")
            continue
        try:
            run, metrics = read(required[0]), read(required[1])
            need(run.get("scope") == metrics.get("scope") == "frozen_medical_oracle_relation_subset_not_end_to_end", "wrong_evaluation_scope")
            need(run.get("screen_manifest_sha256") == metrics.get("screen_manifest_sha256") == screen_sha, "frozen_screen_hash_mismatch")
            need(run.get("records_sha256") == frozen["records_sha256"] and run.get("sample_indices_sha256") == frozen["sample_indices_sha256"], "frozen_inputs_hash_mismatch")
            need(run.get("metrics_sha256") == sha(required[1]), "metrics_file_hash_mismatch")
            need(metrics.get("predictions_sha256") == sha(required[2]) and metrics.get("generations_sha256") == sha(required[3]), "output_file_hash_mismatch")
            need(run.get("samples") == metrics.get("samples") == len(records), "sample_denominator_mismatch")
            need(run.get("sample_indices") == metrics.get("sample_indices") == indices, "sample_indices_mismatch")
            need(run.get("adapter_sha256") == metrics.get("adapter_sha256"), "adapter_hash_mismatch")
            predictions, raw = jsonl(required[2]), jsonl(required[3])
            need([row.get("id") for row in predictions] == [f"medical:{index}" for index in indices], "prediction_rows_missing_or_reordered")
            counted, examples = recount(records, predictions, indices)
            size = run["config"]["pairs_per_batch"]
            need(size in (4, 16), "unsupported_pair_batch_size")
            config = replace(configs["full_source16"], pairs_per_batch=size)
            need(run["config"] == metrics["config"] == asdict(config), "frozen_protocol_config_mismatch")
            telemetry = verify_generations(records, predictions, indices, raw, config)
            match(metrics.get("relation"), counted["relation"], "reported_relation_mismatch")
            match(metrics.get("evidence"), counted["evidence"], "reported_evidence_mismatch")
            need(metrics.get("failed_samples") == counted["failed_samples"], "reported_failed_samples_mismatch")
            need(telemetry["failed_generation_samples"] == counted["failed_samples"], "generation_failure_trace_mismatch")
            for key in ("failed_batches", "budget_exhaustions", "generation_batches"):
                need(metrics.get(key) == telemetry[key], "reported_generation_count_mismatch")
            need(counted["relation"]["tp"] + counted["relation"]["fn"] == denominator, "complete_gold_denominator_mismatch")
            private[name] = examples
            complete = run.get("status") == metrics.get("status") == "complete" and run.get("complete") is True and metrics.get("complete") is True
            need(not (complete and telemetry["deadline_exhaustions"]), "deadline_claimed_complete")
            if not complete:
                summary.update(status="incomplete", diagnostic_relation=counted["relation"], **telemetry)
                continue
            summary.update(status="verified", metric_verified=True, **counted, **telemetry,
                           rules=recount_rules(records, predictions, indices),
                           elapsed_seconds=metrics.get("elapsed_seconds"))
            comparison.append((run.get("base_input_sha256"), run.get("protocol_sha256"),
                size, run["args"].get("max_input_tokens"), run["args"].get("max_new_tokens"), run["args"].get("max_seconds")))
        except Exception as exc:
            summary.update(status="invalid", validation_error=str(exc) if isinstance(exc, InvalidReport) else type(exc).__name__)
    return {"scope": "medical_oracle_subset_independent_recount_not_end_to_end", "jobs": jobs,
            "samples": len(records), "gold_relation_triples": denominator,
            "candidate_covered_gold_triples": covered,
            "unreachable_gold_triples": denominator - covered,
            "screen_manifest_sha256": screen_sha,
            "comparison_settings_match": len(set(comparison)) <= 1,
            "all_three_verified": len(comparison) == len(JOBS),
            "limitations": ["Oracle entities, selected short/candidate-bounded subset; not full clinical or pipeline effectiveness.",
                "All original gold triples remain, including unreachable entities and failed generations.",
                "Rules are post-hoc diagnostics; deleted TP and FP are both reported.",
                "Evidence is exact normalized source text, not independent semantic entailment.",
                "No winner is asserted automatically; medical and artificial results remain separate."]}, private


def markdown(report):
    lines = [f"Medical oracle subset: {report['samples']} samples, {report['gold_relation_triples']} gold triples; "
             f"{report['candidate_covered_gold_triples']} reachable, {report['unreachable_gold_triples']} unreachable retained as FN.", "",
             "| Job | Status | TP/FP/FN | P | R | F1 | Failed samples | Type guard removed TP/FP | Type guard F1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in report["jobs"]:
        if row["metric_verified"]:
            m, rule = row["relation"], row["rules"]["type_guard"]
            lines.append(f"| {row['job']} | verified | {m['tp']}/{m['fp']}/{m['fn']} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {row['failed_samples']} | {rule['removed_tp']}/{rule['removed_fp']} | {rule['relation']['f1']:.4f} |")
        else:
            lines.append(f"| {row['job']} | {row['status']} | — | — | — | — | — | — | — |")
    lines += ["", f"Shared comparison settings match: {report['comparison_settings_match']}. All three verified: {report['all_three_verified']}.", ""]
    lines.extend("- " + text for text in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "screen-dir", "output-prefix"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    need(not root.is_relative_to(REPO), "private_error_examples_must_be_outside_repository")
    report, private = build_report(root, args.screen_dir)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + ".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    Path(str(prefix) + ".md").write_text(markdown(report))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "medical_error_examples_private.json"
    path.write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps({"jobs": [{"job": row["job"], "status": row["status"]} for row in report["jobs"]],
                      "gold_relation_triples": report["gold_relation_triples"]}))


if __name__ == "__main__":
    main()

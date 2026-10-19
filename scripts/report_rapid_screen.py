#!/usr/bin/env python3
"""Recount local rapid-screen predictions; publish aggregate-only comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "src")]

from build_stage_data import _message
from evaluate import _graph, _relation_set, _validate_gold_graph
from evaluation_identity import IDENTITY_KEY, validate_alignment
from generative.rapid_screen import load_screen
from kg_agent.pairwise import relation_rejection


class InvalidReport(ValueError):
    """Only fixed error codes enter public aggregates, never source content."""


def need(condition, code):
    if not condition:
        raise InvalidReport(code)


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def recount(records, predictions, indices):
    """Independent per-sample set accounting, not reported-metric arithmetic."""
    validate_alignment(records, predictions, strict_identity=True, sample_indices=indices)
    tp = fp = fn = invalid = stage_errors = 0
    for record, prediction in zip(records, predictions):
        gold = _graph(_message(record, "assistant"))
        _validate_gold_graph(gold)
        graph = _graph(prediction)
        invalid += graph is None
        expected = _relation_set(gold)
        actual = _relation_set(graph) if graph is not None else set()
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
        trace = prediction.get("trace", {}) if isinstance(prediction, dict) else {}
        stage_errors += any(chunk.get("relation_errors") for chunk in trace.get("chunks", []))
    return {"samples": len(records), "invalid_predictions": invalid,
            "relation_stage_error_samples": stage_errors, "relation": prf(tp, fp, fn)}


def match_metric(reported, actual, code):
    need(isinstance(reported, dict), code)
    for key in ("tp", "fp", "fn"):
        need(type(reported.get(key)) is int and reported[key] == actual[key], code)
    for key in ("precision", "recall", "f1"):
        value = reported.get(key)
        need(isinstance(value, (float, int)) and not isinstance(value, bool)
             and math.isfinite(value) and math.isclose(value, actual[key], abs_tol=1e-10, rel_tol=1e-10), code)


def recount_rules(records, predictions, indices):
    rules = {"symptom_disease_type_mismatch", "different_disease_in_evidence", "enumerated_siblings"}
    reports = {}
    for name, enabled in [("all_rules", rules)] + [(r, {r}) for r in sorted(rules)]:
        filtered = copy.deepcopy(predictions)
        removed = Counter()
        for record, row in zip(records, filtered):
            gold = _relation_set(_graph(_message(record, "assistant")))
            graph = _graph(row)
            if graph is None:
                continue
            accepted = []
            for edge in graph["relations"]:
                reason = relation_rejection(edge, graph["entities"])
                if reason in enabled:
                    triples = _relation_set({"entities": graph["entities"], "relations": [edge]})
                    removed["removed_tp" if triples & gold else "removed_fp"] += 1
                else:
                    accepted.append(edge)
            graph["relations"] = accepted
            row["output"] = graph
        reports[name] = {"removed_tp": removed["removed_tp"], "removed_fp": removed["removed_fp"],
                         "relation": recount(records, filtered, indices)["relation"]}
    return reports


def check_reported_metrics(metrics, counted):
    for scope in ("overall", "free_text"):
        need(metrics[scope]["samples"] == counted["samples"], "metric_sample_denominator_mismatch")
        match_metric(metrics[scope]["relation"], counted["relation"], "prediction_metric_mismatch")
        need(metrics[scope]["invalid_predictions"] == counted["invalid_predictions"], "invalid_count_mismatch")
    need(len(metrics["samples"]) == counted["samples"], "metric_sample_rows_missing")


def check_rules(reported, actual):
    for name, values in actual.items():
        need(name in reported, "rule_report_missing")
        removed = reported[name]["removed"]
        for key in ("removed_tp", "removed_fp"):
            need(removed.get(key, 0) == values[key], "rule_removal_count_mismatch")
        match_metric(reported[name]["metrics"]["free_text"]["relation"], values["relation"], "rule_metric_mismatch")


def check_generations(raw, records, indices, expected_batches, expected_per_sample=None):
    need(len(raw) == expected_batches, "generation_denominator_mismatch")
    lookup = dict(zip(indices, records))
    seen, counts = set(), Counter()
    for row in raw:
        index = row.get("sample_id")
        need(type(index) is int and index in lookup, "generation_sample_identity_mismatch")
        validate_alignment([lookup[index]], [row], strict_identity=True, sample_indices=[index])
        need(isinstance(row.get("id"), str) and row["id"] not in seen, "generation_duplicate_or_missing_id")
        seen.add(row["id"])
        counts[index] += 1
    if expected_per_sample is not None:
        need(counts == Counter(expected_per_sample), "generation_per_sample_denominator_mismatch")
    return {"batches": len(raw), "truncated_batches": sum(bool(r.get("truncated")) for r in raw),
            "parse_error_batches": sum(bool(r.get("parse_errors")) for r in raw),
            "runtime_error_batches": sum(bool(r.get("error")) for r in raw)}


def medical_condition(directory, name, records, indices, manifest, manifest_sha):
    run = read_json(directory / "run.json")
    row = {"condition": name, "job_status": run.get("status", "unknown"), "status": "incomplete"}
    summary = run.get("conditions", {}).get(name)
    metric_path = directory / name / "metrics.json"
    if not metric_path.is_file():
        row["status"] = "missing" if summary is None else "incomplete"
        return row
    report = read_json(metric_path)
    if not summary or summary.get("complete") is not True or report.get("complete") is not True:
        return row
    need(report.get("eligible_for_ranking") is True and report.get("status") == "complete", "condition_completion_disagreement")
    need(run.get("manifest_sha256") == manifest_sha and report.get("source_manifest_sha256") == manifest_sha,
         "screen_manifest_mismatch")
    out = directory / name
    for artifact in ("predictions.jsonl", "generations.jsonl", "raw_stream.jsonl"):
        need(report.get("artifact_sha256", {}).get(artifact) == digest(out / artifact), "artifact_sha256_mismatch")
    predictions, raw = jsonl(out / "predictions.jsonl"), jsonl(out / "generations.jsonl")
    counted = recount(records, predictions, indices)
    relation = counted["relation"]
    need(relation["tp"] + relation["fn"] == manifest["denominators"]["gold_relation_triples"], "gold_denominator_mismatch")
    check_reported_metrics(report["metrics_including_all_failed_rows"], counted)
    match_metric(report.get("main_relation_metric"), relation, "main_metric_mismatch")
    match_metric(summary.get("main_relation_metric"), relation, "run_metric_mismatch")
    plan = manifest["conditions"][name]
    expected = {r["sample_index"]: r["batches"] for r in plan["samples"]}
    generated = check_generations(raw, records, indices, plan["expected_batches"], expected)
    need(generated["runtime_error_batches"] == 0, "complete_condition_has_runtime_errors")
    for key in ("truncated_batches", "parse_error_batches"):
        need(report.get(key) == generated[key], "generation_counter_mismatch")
    rate = generated["truncated_batches"] / max(1, generated["batches"])
    need(math.isclose(report["truncation_rate"], rate, abs_tol=1e-10), "truncation_rate_mismatch")
    need(report.get("observed_samples") == report.get("expected_samples") == len(records), "condition_sample_denominator_mismatch")
    need(report.get("observed_batches") == report.get("expected_batches") == plan["expected_batches"], "condition_batch_denominator_mismatch")
    need(report.get("decode_limit") == manifest["decode"]["max_new_tokens"], "decode_limit_mismatch")
    need(report.get("config") == plan["config"], "condition_config_mismatch")
    rules = recount_rules(records, predictions, indices)
    check_rules(report["rule_ablation"], rules)
    elapsed = report.get("elapsed_seconds")
    need(isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= 0, "invalid_elapsed_time")
    row.update(status="verified", relation=relation, samples=counted["samples"],
               invalid_predictions=counted["invalid_predictions"],
               relation_stage_error_samples=counted["relation_stage_error_samples"],
               **generated, truncation_rate=rate, elapsed_seconds=elapsed, rules=rules)
    return row


def option(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def mechanical_condition(directory, job, records_path, queue):
    row = {"condition": "mechanical", "status": "missing", "scope": "artificial_mechanical_only"}
    if not (directory / "metrics.json").is_file():
        return row
    if records_path is None:
        return {**row, "status": "awaiting_mechanical_gold"}
    remote_gold = option(job["command"], "--gold")
    expected_hash = queue.get("expected_sha256", {}).get(remote_gold)
    need(expected_hash and digest(records_path) == expected_hash, "mechanical_gold_sha256_mismatch")
    records = read_json(records_path)
    indices = list(range(len(records)))
    need(len(records) == job["expected_samples"], "mechanical_gold_sample_mismatch")
    predictions = jsonl(directory / "predictions.jsonl")
    report = read_json(directory / "metrics.json")
    counted = recount(records, predictions, indices)
    check_reported_metrics(report["metrics"], counted)
    raw = jsonl(directory / "generations.jsonl")
    generated = check_generations(raw, records, indices, report["coverage_and_generation"]["generated_batches"])
    need(generated["runtime_error_batches"] == 0, "mechanical_generation_failure")
    rules = recount_rules(records, predictions, indices)
    check_rules(report["rule_ablation"], rules)
    row.update(status="verified", relation=counted["relation"], samples=counted["samples"],
               invalid_predictions=counted["invalid_predictions"],
               relation_stage_error_samples=counted["relation_stage_error_samples"],
               **generated, truncation_rate=generated["truncated_batches"] / max(1, generated["batches"]),
               elapsed_seconds=sum(r.get("amortized_seconds", 0) for r in raw),
               elapsed_scope="generation_time_only", rules=rules)
    return row


def training_exposure(root, jobs, training_manifest, queue):
    planned = read_json(training_manifest) if training_manifest else None
    if planned:
        hashes = [value for path, value in queue.get("expected_sha256", {}).items()
                  if Path(path).parent.name == "data" and Path(path).name == "manifest.json"]
        need(len(hashes) == 1 and hashes[0] == digest(training_manifest), "training_manifest_sha256_mismatch")
    result = []
    fields = {
        "rows": "sft_sequence_loss_exposures",
        "supervised_pairs": "sft_pair_loss_exposures",
        "candidate_pairs": "sft_candidate_pair_context_exposures",
        "unknown_pairs_context_only": "sft_unknown_pair_context_exposures",
        "input_tokens": "sft_input_token_loss_exposures",
        "supervised_completion_tokens": "sft_completion_token_loss_exposures",
        "prompt_tokens": "sft_prompt_token_loss_exposures",
        "masked_completion_tokens_context_only": "sft_masked_completion_token_context_exposures",
    }
    for job in jobs:
        if job["name"] not in {"matched_control", "matched_source", "matched_short4"}:
            continue
        epochs = job.get("expected_epochs", 1)
        need(isinstance(epochs, (int, float)) and not isinstance(epochs, bool)
             and math.isfinite(epochs) and epochs >= 1 and int(epochs) == epochs,
             "training_budget_requires_positive_whole_epochs")
        epochs = int(epochs)
        item = {"job": job["name"], "status": "missing", "planned_epochs": epochs,
                "per_class_exposure_directly_recorded": False}
        path = root / job["name"] / "run_manifest.json"
        if not path.is_file():
            result.append(item)
            continue
        receipt = read_json(path)
        item["status"] = receipt.get("status", "unknown")
        item["global_step"] = receipt.get("global_step")
        observed = receipt.get("observed_exposure", {})
        item["observed"] = {name: observed.get(key) for name, key in fields.items()}
        item["observed_positive_pairs"] = None
        item["observed_negative_pairs"] = None
        item["observed_positive_edges"] = None
        item["equal_positive_negative_exposure_verified"] = False
        if planned:
            basename = Path(option(job["command"], "--data")).name
            need(receipt.get("data_sha256") == planned["output_sha256"].get(basename), "training_data_sha256_mismatch")
            audit = planned["audits"][Path(basename).stem]
            admission, supervision = audit["admission"], audit["supervision"]
            item["expected_one_pass"] = {
                "rows": admission["rows"], "positive_pairs": admission["positive_pairs"],
                "negative_pairs": admission["negative_pairs"], "positive_edges": supervision["positive_edges"],
                "supervised_pairs": admission["supervised_pairs"], "candidate_pairs": admission["candidate_pairs"],
                "unknown_pairs_context_only": admission["unknown_pairs"],
                "input_tokens": (supervision["prompt_tokens_once"] + supervision["completion_tokens_once"]
                                 if supervision.get("prompt_tokens_once") is not None
                                 and supervision.get("completion_tokens_once") is not None else None),
                "supervised_completion_tokens": supervision.get("supervised_completion_tokens_once"),
                "prompt_tokens": supervision.get("prompt_tokens_once"),
                "masked_completion_tokens_context_only": supervision.get("masked_completion_tokens_once"),
            }
            comparable = {k: v for k, v in item["expected_one_pass"].items()
                          if k in item["observed"] and v is not None}
            item["recorded_totals_match_one_pass"] = (
                receipt.get("status") == "training_complete_evaluation_pending"
                and all(item["observed"][key] == value for key, value in comparable.items()))
            item["expected_training_budget"] = {
                key: value * epochs if value is not None else None
                for key, value in item["expected_one_pass"].items()}
            budget_comparable = {key: value for key, value in item["expected_training_budget"].items()
                                 if key in item["observed"] and value is not None}
            item["recorded_totals_match_training_budget"] = (
                receipt.get("status") == "training_complete_evaluation_pending"
                and all(item["observed"][key] == value for key, value in budget_comparable.items()))
        result.append(item)
    return result


def build_report(root, screen_dir, *, queue_config=None, mechanical_gold=None, training_manifest=None):
    root, screen_dir = Path(root), Path(screen_dir)
    records, indices, _, manifest = load_screen(screen_dir)
    manifest_sha = digest(screen_dir / "manifest.json")
    gold_count = sum(len(_relation_set(_graph(_message(r, "assistant")))) for r in records)
    need(gold_count == manifest["denominators"]["gold_relation_triples"], "frozen_gold_denominator_mismatch")
    state = read_json(root / "status.json") if (root / "status.json").is_file() else {}
    queue = read_json(queue_config) if queue_config else state.get("config", {})
    frozen_manifests = [value for path, value in queue.get("expected_sha256", {}).items()
                        if Path(path).parent.name == "evaluation" and Path(path).name == "manifest.json"]
    if frozen_manifests:
        need(len(frozen_manifests) == 1 and frozen_manifests[0] == manifest_sha, "queue_screen_manifest_mismatch")
    jobs = queue.get("jobs", [])
    if not jobs:
        jobs = [{"name": path.parent.name, "kind": "screen", "conditions":
                 read_json(path).get("arguments", {}).get("conditions", list(read_json(path).get("conditions", {})))}
                for path in sorted(root.glob("*/run.json"))]
    state_by_name = {j["name"]: j.get("status", "unknown") for j in state.get("jobs", [])}
    medical, mechanical = [], []
    for job in jobs:
        name = job["name"]
        need(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name), "unsafe_job_name")
        directory = root / name
        if job["kind"] == "screen":
            for condition in job["conditions"]:
                item = {"job": name, "condition": condition, "queue_status": state_by_name.get(name, "not_started"), "status": "missing"}
                try:
                    need(condition in manifest["conditions"], "unknown_condition")
                    if (directory / "run.json").is_file():
                        item.update(medical_condition(directory, condition, records, indices, manifest, manifest_sha))
                except InvalidReport as error:
                    item.update(status="invalid", validation_error=str(error))
                except (ValueError, KeyError, TypeError, OSError, IndexError):
                    item.update(status="invalid", validation_error="artifact_or_identity_validation_failed")
                medical.append(item)
        elif job["kind"] == "evaluation":
            item = {"job": name, "queue_status": state_by_name.get(name, "not_started")}
            try:
                item.update(mechanical_condition(directory, job, mechanical_gold, queue))
            except InvalidReport as error:
                item.update(status="invalid", validation_error=str(error))
            except (ValueError, KeyError, TypeError, OSError, IndexError):
                item.update(status="invalid", validation_error="artifact_or_identity_validation_failed")
            mechanical.append(item)
    covered = {name: p["candidate_covered_gold_triples"] for name, p in manifest["conditions"].items()}
    return {"schema_version": 1, "scope": "rapid_subset_not_final_pipeline_quality",
        "source_manifest_sha256": manifest_sha, "queue_status": state.get("status", "not_available"),
        "denominators": {"medical_samples": len(records), "medical_gold_triples": gold_count,
                         "medical_candidate_pairs": manifest["denominators"]["candidate_pairs"],
                         "candidate_covered_gold_triples_by_condition": covered},
        "medical": medical, "mechanical": mechanical,
        "matched_training_exposure": training_exposure(root, jobs, training_manifest, queue),
        "counts": {"medical_conditions": dict(Counter(r["status"] for r in medical)),
                   "mechanical_conditions": dict(Counter(r["status"] for r in mechanical))},
        "limitations": ["Medical gold and UNKNOWN handling are unchanged; every missing gold triple remains FN.",
                        "Candidates cover only the stated subset of gold; this is a recall ceiling, not measured recall.",
                        "Subset selection excludes zero-candidate and high-candidate documents; no full-130 or pipeline claim.",
                        "Mechanical preference tasks are artificial and never pooled with medical metrics.",
                        "Frozen runtime records total supervised/UNKNOWN/sequence exposure, not separate observed positive/negative/edge exposure; expected class counts are not measurements.",
                        "Incomplete, missing and invalid conditions have no accepted P/R/F1.",
                        "No automatic winner or causal direction conclusion is selected."]}


def markdown(report):
    d = report["denominators"]
    coverage = sorted(set(d["candidate_covered_gold_triples_by_condition"].values()))
    lines = ["Rapid screen: independently recounted aggregates", "",
             f"Medical subset: {d['medical_samples']} documents, {d['medical_gold_triples']} gold triples, "
             f"{d['medical_candidate_pairs']} candidates. Reachable gold: "
             + "/".join(map(str, coverage)) + f" of {d['medical_gold_triples']}; unreachable gold stays in FN.", ""]
    for title, rows in (("Medical relation subset", report["medical"]), ("Artificial mechanical tasks (separate)", report["mechanical"])):
        lines += [title, "", "| Job | Condition | Status | TP/FP/FN | P | R | F1 | Invalid graphs / parse-error batches | Truncated | Seconds | Rules removed TP/FP |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in rows:
            if row["status"] != "verified":
                lines.append(f"| {row['job']} | {row.get('condition', 'mechanical')} | {row['status']} | — | — | — | — | — | — | — | — |")
                continue
            m, rules = row["relation"], row["rules"]["all_rules"]
            lines.append(f"| {row['job']} | {row['condition']} | verified | {m['tp']}/{m['fp']}/{m['fn']} | "
                         f"{m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | "
                         f"{row['invalid_predictions']}/{row['parse_error_batches']} | "
                         f"{row['truncated_batches']}/{row['batches']} | {row['elapsed_seconds']:.1f} | "
                         f"{rules['removed_tp']}/{rules['removed_fp']} |")
        if not rows:
            lines.append("| — | — | missing | — | — | — | — | — | — | — | — |")
        lines.append("")
    lines += ["Matched-training exposure (measured totals versus planned epoch budget)", "",
              "| Job | Status | Planned epochs | Actual rows | Actual supervised pairs | Actual UNKNOWN context pairs | Expected budget positive/negative/edges | Recorded totals match budget |",
              "|---|---|---|---|---|---|---|---|"]
    for row in report["matched_training_exposure"]:
        observed, expected = row.get("observed", {}), row.get("expected_training_budget", row.get("expected_one_pass", {}))
        classes = "/".join(str(expected.get(k, "—")) for k in ("positive_pairs", "negative_pairs", "positive_edges"))
        lines.append(f"| {row['job']} | {row['status']} | {row.get('planned_epochs', 1)} | {observed.get('rows', '—')} | "
                     f"{observed.get('supervised_pairs', '—')} | {observed.get('unknown_pairs_context_only', '—')} | "
                     f"{classes} | {row.get('recorded_totals_match_training_budget', 'not verified')} |")
    lines.append("")
    lines += ["- " + text for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Local synchronized screen_v1 directory")
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--queue-config")
    parser.add_argument("--mechanical-gold")
    parser.add_argument("--training-manifest", help="Frozen rapid/data/manifest.json, for expected single-pass exposure")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    report = build_report(args.root, args.screen_dir, queue_config=args.queue_config,
                          mechanical_gold=args.mechanical_gold, training_manifest=args.training_manifest)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    prefix.with_suffix(".md").write_text(markdown(report))
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()

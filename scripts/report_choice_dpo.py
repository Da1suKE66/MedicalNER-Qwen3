#!/usr/bin/env python3
"""Independently recount all frozen artificial choice holdout rows and comparisons."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from generative.choice_preferences import validate_preference_row

ARMS = ("parent", "sft", "dpo")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def atoms(edges, evidence=False):
    if evidence:
        return {(edge["pair_id"], edge["relation"], span) for edge in edges for span in edge["evidence_span_ids"]}
    return {(edge["pair_id"], edge["relation"]) for edge in edges}


def counts(gold, predicted):
    return dict(tp=len(gold & predicted), fp=len(predicted - gold), fn=len(gold - predicted))


def rates(total):
    tp, fp, fn = (total.get(key, 0) for key in ("tp", "fp", "fn"))
    return dict(tp=tp, fp=fp, fn=fn, gold_count=tp + fn, predicted_count=tp + fp,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def expected_cases(path):
    cases = {}
    for source_index, row in enumerate(read_rows(path)):
        require(row["split"] == "mechanical_holdout", "data must be mechanical holdout")
        task = validate_preference_row(row, require_train=False)
        source, identity = row["source_row"], row["id"]
        require(identity not in cases, "duplicate data identity")
        cases[identity] = dict(task=task, data=row, gold=task.edges(row["chosen_actions"]),
            identity={"id": identity, "source_id": source["id"], "source_index": source_index,
                "input_row_sha256": sha(row), "source_row_sha256": sha(source), "kind": row["reason"],
                "task_input": row["task_input"], "task_input_sha256": sha(row["task_input"]),
                "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                "prompt_chars": len(task.prompt), "source_chars": len(source["oracle_source_text"]),
                "candidate_pairs": len(task.pairs), "gold_actions": row["chosen_actions"]})
    require(bool(cases), "empty frozen data")
    return cases


def group_metrics(rows):
    result = {"samples": len(rows), "failed_samples": sum(r["status"] != "complete" for r in rows)}
    for metric in ("relations", "relation_evidence"):
        total = Counter()
        for row in rows:
            total.update(row[metric + "_counts"])
        result[metric] = rates(total)
    return result


def recount(cases, records):
    require(len(records) == len(cases), "missing or extra prediction rows")
    require(len({r["id"] for r in records}) == len(records), "duplicate prediction identity")
    require({r["id"] for r in records} == set(cases), "prediction identity set differs")
    groups, per_label = defaultdict(list), defaultdict(lambda: {"relations": Counter(), "relation_evidence": Counter()})
    for row in records:
        case = cases[row["id"]]
        for key, expected in case["identity"].items():
            require(canonical(row.get(key)) == canonical(expected), f"prediction identity/input mismatch: {key}")
        require(canonical(row.get("gold_edges")) == canonical(case["gold"]), "stored gold differs from reconstructed oracle")
        require(row.get("status") in {"complete", "failed"}, "invalid generation status")
        require(type(row.get("generation_attempted")) is bool, "invalid generation attempt flag")
        predicted = case["task"].edges(row["actions"]) if row["status"] == "complete" else []
        require(canonical(row.get("predicted_edges")) == canonical(predicted), "stored prediction differs from actions or failed row materialized edges")
        for metric, evidence in (("relations", False), ("relation_evidence", True)):
            gold_set, predicted_set = atoms(case["gold"], evidence), atoms(predicted, evidence)
            actual = counts(gold_set, predicted_set)
            require(canonical(row.get(metric + "_counts")) == canonical(actual), "per-row counts differ from independent recount")
            for label in {atom[1] for atom in gold_set | predicted_set}:
                per_label[label][metric].update(counts({a for a in gold_set if a[1] == label},
                                                     {a for a in predicted_set if a[1] == label}))
        length_bin = next((f"le_{n}" for n in (1024, 4096, 8192, 16384) if row["prompt_chars"] <= n), "gt_16384")
        for category, key in (("kind", row["kind"]), ("prompt_chars", length_bin), ("candidate_pairs", str(row["candidate_pairs"]))):
            groups[category, key].append(row)
    metrics = group_metrics(records)
    metrics["relation"] = metrics.pop("relations")
    metrics["evidence"] = metrics.pop("relation_evidence")
    deadline = sum(bool(row["trace"].get("deadline_exhausted")) for row in records)
    metrics.update(status="incomplete" if deadline else "complete", complete=not bool(deadline),
        generation_complete=not bool(metrics["failed_samples"]), accounted_samples=len(records),
        generation_attempted_samples=sum(row["generation_attempted"] for row in records),
        complete_generation_count=len(records) - metrics["failed_samples"],
        budget_exhaustions=sum(bool(row["trace"].get("budget_exhausted")) for row in records),
        deadline_exhaustions=deadline,
        errors_by_type=dict(Counter(row["error_type"] for row in records if row.get("error_type"))),
        exact_relation_sets=sum(r["status"] == "complete" and not (r["relations_counts"]["fp"] or r["relations_counts"]["fn"]) for r in records),
        exact_evidence_sets=sum(r["status"] == "complete" and not (r["relation_evidence_counts"]["fp"] or r["relation_evidence_counts"]["fn"]) for r in records),
        per_relation={label: {metric: rates(value) for metric, value in values.items()} for label, values in per_label.items()},
        strata={category: {key: group_metrics(rows) for (cat, key), rows in groups.items() if cat == category}
                for category in ("kind", "prompt_chars", "candidate_pairs")})
    return metrics


def check_training(sft_path=None, dpo_path=None):
    if not sft_path and not dpo_path:
        return {"status": "not_provided", "same_parent_and_exposure_verified": False}
    require(bool(sft_path and dpo_path), "provide both training manifests")
    sft, dpo = (json.loads(Path(path).read_text()) for path in (sft_path, dpo_path))
    for key in ("parent_adapter_sha256", "parent_manifest_sha256", "data_sha256", "row_schedule_sha256",
                "paired_chosen_encoding_sha256", "optimizer_updates", "source_rows"):
        require(sft.get(key) is not None and sft.get(key) == dpo.get(key), f"training comparison mismatch: {key}")
    require(dpo.get("reference_unchanged") is True, "DPO reference is not verified frozen")
    for run in (sft, dpo):
        require(run.get("status") in {"complete", "training_complete_evaluation_pending"} and run.get("parent_initialization_verified") is True
                and run.get("saved_reload_exact") is True, "training receipt not complete/verified")
    require(dpo.get("reference_adapter_sha256_after") is not None
            and dpo["reference_adapter_sha256_after"] == dpo.get("policy_adapter_sha256_before")
            == dpo.get("expected_parent_parameter_sha256"), "DPO reference fingerprint differs from parent")
    require(sft.get("observed_exposure", {}).get("rows") == dpo.get("observed_exposure", {}).get("rows")
            and type(dpo.get("observed_exposure", {}).get("rows")) is int, "observed example exposure differs")
    return {"status": "verified", "same_parent_and_exposure_verified": True,
            "source_rows": dpo["source_rows"], "optimizer_updates": dpo["optimizer_updates"],
            "observed_rows": dpo["observed_exposure"]["rows"], "reference_unchanged": True}


def build_report(root, data, *, sft_manifest=None, dpo_manifest=None):
    cases, runs, private = expected_cases(data), {}, {}
    for arm in ARMS:
        directory = Path(root) / (arm + "_eval")
        predictions, metrics_path = directory / "predictions.jsonl", directory / "metrics.json"
        records, declared = read_rows(predictions), json.loads(metrics_path.read_text())
        actual = recount(cases, records)
        for key, value in actual.items():
            require(canonical(declared.get(key)) == canonical(value), f"declared metrics mismatch: {arm}.{key}")
        for alias, metric in (("relations", "relation"), ("relation_evidence", "evidence")):
            if alias in declared:
                require(canonical(declared[alias]) == canonical(actual[metric]), f"declared metric alias mismatch: {arm}.{alias}")
        if "predictions_sha256" in declared:
            require(declared["predictions_sha256"] == file_sha(predictions), "prediction file SHA mismatch")
        runs[arm] = {"verification": "independently_recounted", **actual,
                     "predictions_sha256": file_sha(predictions), "metrics_sha256": file_sha(metrics_path)}
        private[arm] = {row["id"]: row for row in records}
    comparisons, examples = {}, {}
    for arm in ("sft", "dpo"):
        by_kind = defaultdict(Counter)
        examples[arm] = {}
        for identity, case in cases.items():
            before, after = private["parent"][identity], private[arm][identity]
            before_errors = sum(before["relation_evidence_counts"][k] for k in ("fp", "fn"))
            after_errors = sum(after["relation_evidence_counts"][k] for k in ("fp", "fn"))
            change = "improved" if after_errors < before_errors else "regressed" if after_errors > before_errors else "unchanged_error_count"
            kind = case["identity"]["kind"]
            by_kind[kind][change] += 1
            if kind not in examples[arm] or change != "unchanged_error_count":
                examples[arm][kind] = {"id": identity, "change": change,
                    "source_row": case["data"]["source_row"], "gold_edges": case["gold"],
                    "parent": before, arm: after}
        comparisons[arm + "_vs_parent"] = {kind: dict(values) for kind, values in by_kind.items()}
    aggregate = {"scope": "adaptive_artificial_semantic_comparison_not_medical_not_full_pipeline",
        "data_sha256": file_sha(data), "samples_per_arm": len(cases), "runs": runs,
        "comparisons_by_error_kind": comparisons, "training": check_training(sft_manifest, dpo_manifest),
        "limitations": ["Names/groups differ across train/holdout but templates overlap.",
                        "Failure rows keep all gold FN; missing/duplicate/wrong gold rows reject the report.",
                        "Equal training rows do not equalize DPO/SFT token loss exposure or compute."]}
    return {"aggregate": aggregate, "private_error_examples": examples}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sft-manifest")
    parser.add_argument("--dpo-manifest")
    args = parser.parse_args(argv)
    prefix = Path(args.output_prefix).resolve()
    require(not prefix.is_relative_to(Path(__file__).resolve().parents[1]), "reports containing raw examples must remain outside repository")
    paths = [Path(str(prefix) + suffix) for suffix in (".json", ".aggregate.json")]
    require(not any(path.exists() for path in paths), "do not overwrite reports")
    report = build_report(args.root, args.data, sft_manifest=args.sft_manifest, dpo_manifest=args.dpo_manifest)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, value in zip(paths, (report, report["aggregate"])):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        path.chmod(0o600)
    print(json.dumps({"status": "verified", "samples": report["aggregate"]["samples_per_arm"],
                      "private_report": str(paths[0]), "aggregate_report": str(paths[1])}))


if __name__ == "__main__":
    main()

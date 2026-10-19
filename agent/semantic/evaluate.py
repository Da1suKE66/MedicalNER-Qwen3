"""Complete frozen tune evaluation: live/oracle, raw/filtered and strict failures.

The added trusted title is a versioned input change. Historical same-input
scores must not be silently compared as if they used this augmented context.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from build_stage_data import _message
from evaluate import _graph, _entity_set, _relation_set, _node_name, _validate_gold_graph
from evaluate_graphs import evaluate_records
from evaluation_identity import IDENTITY_KEY, make_input_identity
from kg_agent.assemble import assemble_graph as assemble_icd
from kg_agent.source import SourceRouter, parse_icd_graph
from pair_metrics import counts_metric
from semantic.data import DATA_VERSION, VARIANTS, canonical, sha
from semantic.protocol import CONTRACT, normalize_name
from semantic.runtime import ENTITY_VARIANTS, SemanticGenerator, check_adapter_contract, entity_strategy_contract, protocol_digest, run_documents, verify_title_context


def load_release(directory, *, expected_samples=None, evaluation_set="tune"):
    """Independently verify the complete release, exact original gold and inputs."""
    if evaluation_set == "historical_regression":
        from semantic.historical import load_release as load_historical
        return load_historical(directory, expected_samples=148 if expected_samples is None else expected_samples)
    if evaluation_set != "tune":
        raise ValueError("Unknown evaluation population")
    expected_samples = 130 if expected_samples is None else expected_samples
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("data_version") != DATA_VERSION or manifest.get("gold_unchanged") is not True:
        raise ValueError("Wrong semantic evaluation release contract")
    names = ("evaluation/tune_records.jsonl", "title_source_receipts.json")
    raw = {name: (directory / name).read_bytes() for name in names}
    for name, content in raw.items():
        if manifest.get("output_sha256", {}).get(name) != sha(content):
            raise ValueError("Frozen evaluation artifact SHA mismatch: " + name)
    cases = [json.loads(line) for line in raw[names[0]].decode().splitlines() if line.strip()]
    sources = json.loads(raw[names[1]])
    if len(cases) != expected_samples or not cases:
        raise ValueError(f"Expected all {expected_samples} frozen tune cases, got {len(cases)}")
    if len({case["id"] for case in cases}) != len(cases) or len({case["index"] for case in cases}) != len(cases):
        raise ValueError("Duplicate frozen sample identity")
    actual_counts = Counter()
    for case in cases:
        if case["split"] != "tune" or type(case["index"]) is not int or case["index"] < 0:
            raise ValueError("Only frozen tune identities may enter this evaluator")
        original, metadata = case["original_record"], case["metadata"]
        if sha(original) != case["original_record_sha256"] or metadata["source_record_sha256"] != sha(original):
            raise ValueError("Original record hash mismatch")
        original_gold = _graph(_message(original, "assistant"))
        if original_gold is None or case["gold"] != original_gold:
            raise ValueError("Gold changed or was reduced during evaluation compilation")
        _validate_gold_graph(original_gold)
        raw_user = _message(original, "user")
        if metadata["effective_user_sha256"] != sha(raw_user):
            raise ValueError("Tune source user was modified")
        a, b = metadata["source_payload_start"], metadata["source_payload_end"]
        if type(a) is not int or type(b) is not int or not 0 <= a <= b <= len(raw_user) or raw_user[a:b] != case["text"]:
            raise ValueError("Tune source text is not the exact original payload")
        if metadata.get("source_text_sha256") != sha(case["text"]):
            raise ValueError("Tune source text hash mismatch")
        if case["input_identity"].get("sha256") != sha({"text": case["text"], "metadata": metadata}):
            raise ValueError("Augmented semantic input identity mismatch")
        if str(case["source_group_id"]) != str(metadata["source_group_id"]):
            raise ValueError("Source group mismatch")
        verify_title_context(metadata, sources, source_group_id=case["source_group_id"])
        routed = SourceRouter().route(raw_user)
        if routed.source_type != case["source_type"] or metadata["source_type"] != case["source_type"]:
            raise ValueError("Source type differs from immutable original input")
        actual_counts["tune:" + case["source_type"]] += 1
    expected_counts = {key: value for key, value in manifest["record_counts"].items() if key.startswith("tune:")}
    if dict(actual_counts) != expected_counts:
        raise ValueError("Frozen tune strata are incomplete")
    return cases, {"manifest_sha256": sha(manifest_bytes),
                   "input_files_sha256": {name: sha(content) for name, content in raw.items()},
                   "record_counts": dict(actual_counts), "data_version": DATA_VERSION,
                   "source_artifacts": manifest.get("source_artifacts", {}), "short_chars": manifest["short_chars"]}


def _prediction(case, result, field, mode):
    return {"id": f"{mode}:{case['id']}", "sample_index": case["index"], "mode": mode,
        IDENTITY_KEY: make_input_identity(case["original_record"], case["index"]),
        "semantic_input_identity": deepcopy(case["input_identity"]),
        "output": deepcopy(result[field]), "trace": deepcopy(result["trace"]), "output_contract": CONTRACT}


def _strict_metrics(raw_metrics, results, cases):
    result = deepcopy(raw_metrics)
    for section in ("overall", "free_text", "structured_icd"):
        selected = [row for row, case in zip(results, cases) if section == "overall" or case["source_type"] == section]
        entity_attempts = sum(row["trace"].get("invalid_entity_attempts", 0) for row in selected)
        relation_attempts = sum(row["trace"].get("invalid_relation_attempts", 0) for row in selected)
        summary = result[section]
        for key, additional in (("entity", entity_attempts), ("relation", relation_attempts),
                                ("evidence_exact", relation_attempts), ("evidence_compatible", relation_attempts)):
            counts = summary[key]
            summary[key] = counts_metric(counts["tp"], counts["fp"] + additional, counts["fn"])
        summary["invalid_entity_attempt_fp"] = entity_attempts
        summary["invalid_relation_attempt_fp"] = relation_attempts
        if relation_attempts:
            summary["per_relation"]["__INVALID_ATTEMPT__"] = counts_metric(0, relation_attempts, 0)
    # Keep the original per-sample set scores in raw, not under a misleading
    # strict summary whose totals include the extra failed-record attempts.
    result.pop("samples", None)
    result["definitions"]["strict_raw"] = (
        "Mixed accounting: typed directed set TP/FN plus set FP and one additional invalid-attempt FP per nonblank model record in each failed generation/parser stage. This is not pure set F1. Partial rows are never materialized as graph edges."
    )
    result["definitions"]["invalid_attempt_unit"] = "nonblank physical completion line; even otherwise-valid lines count when the whole stage/window failed; blank transport failures add zero FP and retain all gold FN"
    return result


def score_results(cases, results, *, mode):
    if len(results) != len(cases) or [r["id"] for r in results] != [c["id"] for c in cases]:
        raise ValueError("Every frozen sample must be represented exactly once in original order")
    originals, indices = [case["original_record"] for case in cases], [case["index"] for case in cases]
    raw = [_prediction(case, row, "raw_graph", mode) for case, row in zip(cases, results)]
    filtered = [_prediction(case, row, "filtered_graph", mode) for case, row in zip(cases, results)]
    raw_scores = evaluate_records(originals, raw, strict_identity=True, sample_indices=indices)
    filtered_scores = evaluate_records(originals, filtered, strict_identity=True, sample_indices=indices)
    removed = {section: Counter() for section in ("overall", "free_text", "structured_icd")}
    coverage_rows = []
    for case, result in zip(cases, results):
        for kind, extract in (("entity", _entity_set), ("relation", _relation_set)):
            gold, before, after = extract(case["gold"]), extract(result["raw_graph"]), extract(result["filtered_graph"])
            if not after <= before:
                raise ValueError("Filtering invented predictions absent from raw output")
            gone = before - after
            for section in ("overall", case["source_type"]):
                removed[section].update({kind + "_removed_tp": len(gone & gold), kind + "_removed_fp": len(gone - gold)})
        # Diagnostics are deliberately computed only AFTER inference. Gold may
        # measure endpoint coverage, but never populates a live entity table.
        gold_edges = _relation_set(case["gold"])
        raw_entities = _entity_set(result["raw_graph"])
        table = result["trace"].get("entity_parse", {}).get("filtered_entities", result["filtered_graph"]["entities"])
        table_entities = _entity_set({"entities": table, "relations": []})
        raw_reachable = {edge for edge in gold_edges if edge[0] in raw_entities and edge[2] in raw_entities}
        reachable = {edge for edge in gold_edges if edge[0] in table_entities and edge[2] in table_entities}
        generations = result.get("generations", [])
        coverage_rows.append({"sample_index": case["index"], "source_type": case["source_type"],
            "gold_relations": len(gold_edges), "raw_entities": len(raw_entities), "actual_table_entities": len(table_entities),
            "gold_edges_with_both_raw_endpoints": len(raw_reachable),
            "gold_edges_with_both_actual_table_endpoints": len(reachable),
            "gold_endpoint_coverage_lost_to_entity_rules": len(raw_reachable - reachable),
            "zero_actual_table_entities": int(not table_entities),
            "zero_relation_generation_calls": int(not any(run["stage"] == "relations" for run in generations)),
            "input_budget_refusals": sum(run["generation_trace"].get("error") == "input_budget_exceeded_no_truncation" for run in generations)})
    strict = _strict_metrics(raw_scores, results, cases)
    for section in ("overall", "free_text", "structured_icd"):
        selected = [case for case in cases if section == "overall" or case["source_type"] == section]
        for metric, extract in (("entity", _entity_set), ("relation", _relation_set)):
            denominator = sum(len(extract(case["gold"])) for case in selected)
            for scores in (raw_scores, filtered_scores, strict):
                counts = scores[section][metric]
                if counts["tp"] + counts["fn"] != denominator:
                    raise ValueError("Evaluation changed a complete original gold denominator")
    coverage = {}
    for section in ("overall", "free_text", "structured_icd"):
        selected = [row for row in coverage_rows if section == "overall" or row["source_type"] == section]
        totals = Counter()
        for row in selected:
            totals.update({key: value for key, value in row.items() if key not in ("sample_index", "source_type")})
        coverage[section] = {"samples": len(selected), **dict(totals),
            "endpoint_only_recall_upper_bound": (totals["gold_edges_with_both_actual_table_endpoints"] / totals["gold_relations"]
                                                 if totals["gold_relations"] else 0.0)}
    coverage.update(samples=coverage_rows, definitions={
        "scope": "Post-inference diagnostics only; no gold influences live prompts or candidates",
        "actual_table": "Complete filtered full-source entity-stage table in live mode; full supplied gold entity table in oracle mode; deterministic output nodes for structured ICD",
        "endpoint_coverage": "Gold directed triples whose source and target typed normalized names both exist in the table; an upper bound ignoring relation-label and evidence difficulty",
        "zero_relation_generation_calls": "Structured ICD intentionally has no model calls and is reported in its separate stratum"})
    generation_calls = sum(row["trace"].get("generation_calls", 0) for row in results)
    failed_generation_calls = sum(row["trace"]["failed_calls"] for row in results if row["trace"].get("generation_calls", 0))
    return {"samples": len(cases), "mode": mode, "raw": raw_scores, "filtered": filtered_scores,
            "strict_raw": strict, "rule_deletions": {key: dict(value) for key, value in removed.items()},
            "coverage_diagnostics": coverage, "generation_calls": generation_calls,
            "semantic_record_delivery_success_rate": ((generation_calls - failed_generation_calls) / generation_calls if generation_calls else None),
            "failed_samples": sum(not row["trace"]["generation_complete"] for row in results),
            "failed_generation_calls": failed_generation_calls,
            "budget_exhaustions": sum(row["trace"].get("budget_exhaustions", 0) for row in results),
            "entity_windows": {
                "total": sum(row["trace"].get("entity_window_count", 0) for row in results),
                "successful": sum(row["trace"].get("entity_successful_windows", 0) for row in results),
                "documents_with_incomplete_entity_stage": sum(row["trace"].get("entity_complete") is False for row in results)},
            "deadline_exhaustions": sum(bool(row["trace"].get("deadline_exhausted")) for row in results)}, raw, filtered


def _structured(case):
    try:
        routed = SourceRouter().route(_message(case["original_record"], "user"))
        graph = assemble_icd(parse_icd_graph(routed))
        error = None
    except Exception as exc:
        graph, error = {"entities": [], "relations": []}, f"{type(exc).__name__}: {exc}"
    return {"id": case["id"], "raw_graph": graph, "filtered_graph": deepcopy(graph), "generations": [],
        "trace": {"entity_source": "deterministic_structured_icd", "generation_complete": error is None,
                  "generation_calls": 0, "failed_calls": int(error is not None), "error": error,
                  "invalid_entity_attempts": 0, "invalid_relation_attempts": 0, "budget_exhaustions": 0,
                  "deadline_exhausted": False, "chunks": []}}


def _oracle_entities(case):
    """Use one input reference for semantically identical historical gold IDs.

    Gold remains unchanged. Typed normalized name is also the scoring identity;
    multiple gold IDs with that identity cannot be disambiguated in the literal
    protocol and must not duplicate an otherwise identical prompt-table entry.
    """
    unique = {}
    for entity in case["gold"]["entities"]:
        name = _node_name(entity)
        key = (entity["label"], normalize_name(name))
        if key not in unique:
            unique[key] = {"id": str(entity["id"]), "label": entity["label"], "name": name, "entity_source": "oracle"}
    return list(unique.values())


def evaluate_cases(cases, generator, *, variant="full_source", entity_variant="full_source", modes=("live", "oracle"), short_chars=2400,
                   max_seconds=None, on_generation=None, on_document=None, clock=time.monotonic):
    if not cases or not modes or len(set(modes)) != len(modes) or any(mode not in ("live", "oracle") for mode in modes):
        raise ValueError("Nonempty cases and distinct live/oracle modes are required")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive and finite")
    receipt = getattr(generator, "receipts", {}).get("entities") if "live" in modes else None
    entity_contract = entity_strategy_contract(entity_variant, receipt, short_chars=short_chars)
    started = clock()
    deadline = started + max_seconds if max_seconds is not None else None
    reports, predictions, results_by_mode = {}, {}, {}
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("Duplicate evaluation document IDs")
    populations = {case.get("evaluation_set", case.get("split", "tune")) for case in cases}
    if len(populations) != 1:
        raise ValueError("Evaluation populations cannot be mixed")
    historical = populations == {"historical_regression"}
    for mode in modes:
        free, positions, results = [], [], [None] * len(cases)
        for index, case in enumerate(cases):
            if case["source_type"] == "structured_icd":
                results[index] = _structured(case)
                if on_document: on_document(mode, case, results[index])
                continue
            document = {"id": case["id"], "text": case["text"], "metadata": case["metadata"]}
            if mode == "oracle":
                document["oracle_entities"] = _oracle_entities(case)
            free.append(document)
            positions.append(index)
        def generation(row):
            if on_generation or hasattr(generator, "validate_generation_event"):
                source = by_id[row["document_id"]]
                event = {**row, "id": mode + ":" + row["id"], "mode": mode,
                    IDENTITY_KEY: make_input_identity(source["original_record"], source["index"]),
                    "semantic_input_identity": deepcopy(source["input_identity"])}
                if hasattr(generator, "validate_generation_event"):
                    generator.validate_generation_event(event)
                if on_generation: on_generation(event)
        free_results = run_documents(free, generator, variant=variant, entity_variant=entity_variant, mode=mode, short_chars=short_chars,
                                    deadline=deadline, on_generation=generation)
        for index, result in zip(positions, free_results):
            if mode == "oracle":
                result["trace"]["oracle_equivalent_gold_ids_merged"] = len(cases[index]["gold"]["entities"]) - len(_oracle_entities(cases[index]))
            results[index] = result
            if on_document: on_document(mode, cases[index], result)
        report, raw, filtered = score_results(cases, results, mode=mode)
        if historical:
            report["historical_context_strata"] = {}
            for has_title, name in ((True, "free_text_with_verified_title"), (False, "free_text_without_title")):
                selected = [(case, result) for case, result in zip(cases, results)
                            if case["source_type"] == "free_text" and bool(case["metadata"].get("title_context")) == has_title]
                stratum, _, _ = score_results([item[0] for item in selected], [item[1] for item in selected], mode=mode)
                report["historical_context_strata"][name] = stratum
        reports[mode], predictions[mode], results_by_mode[mode] = report, {"raw": raw, "filtered": filtered}, results
    expired = deadline is not None and clock() >= deadline
    incomplete = expired or any(report["deadline_exhaustions"] for report in reports.values())
    metrics = {"status": "incomplete" if incomplete else "complete", "complete": not incomplete,
        "samples": len(cases), "modes": reports, "variant": variant, "entity_variant": entity_variant,
        "entity_strategy": entity_contract, "output_contract": CONTRACT,
        "primary_metric": ("live" if "live" in modes else "oracle") + ".filtered.free_text.relation",
        "primary_metric_definition": "Typed directed micro precision/recall/F1 of the final rule-filtered graph against all original gold. Live is the complete pipeline; oracle-only runs are component evaluations, not live pipeline results. Structured ICD is reported separately.",
        "reported_metric_views": {
            "filtered": "Official final-graph set precision/recall/F1; every original gold item remains in the denominator, including failed generations",
            "raw": "Required companion set precision/recall/F1 before relation rules, under the declared filtered-entity-table input policy; every original gold item remains in the denominator",
            "strict_raw": "Diagnostic only: set TP/FN plus set FP and row-level invalid-attempt FP; mixed accounting, not pure graph-set F1 and not the official primary metric"},
        "strict_raw_is_diagnostic_only": True,
        "generation_complete": not any(report["failed_samples"] for report in reports.values()),
        "elapsed_seconds": max(0., clock() - started), "deadline_exhausted_after_scoring": expired,
        "gold_policy": "All original tune entities and relations retained, including ungrounded/unreachable endpoints and failed windows",
        "input_context": "trusted_same_group_structured_source_title_added_v1; not the historical same-input condition",
        "raw_pipeline_policy": "Live relation generation is conditional on the complete filtered full-source entity table. Raw entity predictions remain scored, and raw relation predictions precede relation rules; this is not a completely unfiltered two-stage pipeline.",
        "entity_order_control": "Full source, source order, and the same declared entity adapter for every relation variant",
        "gold_annotation_scope": "Frozen teacher silver annotations; matching scores are not human-adjudicated clinical correctness or proof that unannotated relations are false",
        "precision_is_not_accuracy": True, "no_true_negative_accuracy": "Open-world extraction has no defined finite true-negative universe"}
    if entity_variant == "short_source":
        metrics.update(entity_order_control="Source-only windows, source order, and the same declared short_source entity adapter for every relation variant",
            entity_window_policy="Same source_windows algorithm and literal prompt as data_v2 short_source, using the explicit character/overlap budget in entity_strategy. Inference visits all source-only windows, including windows without supervised training rows. No gold-driven or failure-driven full-source fallback; cross-boundary names can be missed. Successful local raw/filtered entities merge separately with global body offsets; any failed window keeps entity_complete/generation_complete false and every original gold remains scored.")
    if historical:
        metrics.update(evaluation_set="historical_regression", blind_test=False, training_allowed=False,
            gold_policy="All 148 original historical cases and all gold retained, including unmatched inputs and failed windows; this inspected historical regression is not a blind test",
            input_context="Verified original raw-source association plus same-source historical structured title for 57 free-text cases; 24 unmatched free-text cases retain original input without title. Versioned augmented input, not a historical same-input comparison.")
    if hasattr(generator, "finalize_cache"):
        metrics["entity_cache"] = generator.finalize_cache()
    return metrics, predictions, results_by_mode


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "base-model", "entity-adapter", "relation-adapter", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full_source")
    parser.add_argument("--entity-variant", choices=ENTITY_VARIANTS, default="full_source")
    parser.add_argument("--mode", choices=("live", "oracle", "both"), default="both")
    parser.add_argument("--evaluation-set", choices=("tune", "historical_regression"), default="tune")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--entity-cache-run", help="Explicit completed same-runtime full-130 live run; fixed entity replay only, relations still generated")
    args = parser.parse_args(argv)
    if args.expected_samples is None:
        args.expected_samples = 148 if args.evaluation_set == "historical_regression" else 130
    if args.evaluation_set == "historical_regression" and args.expected_samples != 148:
        parser.error("Historical regression requires all 148 original cases")
    if min(args.expected_samples, args.max_input_tokens, args.max_new_tokens, args.batch_size) < 1:
        parser.error("Sample/token/batch counts must be positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        parser.error("max-seconds must be positive and finite")
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Existing evaluation artifacts are immutable; use a new output directory")
    cases, data_manifest = load_release(args.release, expected_samples=args.expected_samples, evaluation_set=args.evaluation_set)
    receipts = {stage: check_adapter_contract(path, stage, base_model=args.base_model)
                for stage, path in (("entities", args.entity_adapter), ("relations", args.relation_adapter))}
    entity_contract = entity_strategy_contract(args.entity_variant, receipts["entities"], short_chars=data_manifest["short_chars"])
    cache_plan = None
    if args.entity_cache_run:
        from semantic.entity_cache import EntityCachePlan
        cache_plan = EntityCachePlan.load(args.entity_cache_run, cases, data_manifest, receipts["entities"],
            entity_variant=args.entity_variant, base_model=args.base_model, max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens, mode=args.mode)
    manifest = {"status": "preflight_complete" if args.preflight_only else "running", "complete": False, "completed": False, "failed": False,
                "args": vars(args), "input_release": data_manifest, "output_contract": CONTRACT,
                "protocol_sha256": protocol_digest(), "runtime_sha256": sha(Path(__file__).with_name("runtime.py").read_bytes()),
                "evaluator_sha256": sha(Path(__file__).read_bytes()), "samples": len(cases),
                "entity_variant": args.entity_variant, "entity_strategy": entity_contract,
                "relation_prompt_variant": args.variant, "relation_training_variant": receipts["relations"]["variant"],
                "relation_prompt_matches_training_variant": args.variant == receipts["relations"]["variant"],
                "base_input_hash_scope": "Model configuration and tokenizer files; not base weight bytes",
                "sample_indices": [case["index"] for case in cases],
                "adapters": {stage: {key: value[key] for key in ("stage", "variant", "adapter_files_sha256", "base_input_sha256", "data_sha256")}
                             for stage, value in receipts.items()}}
    if args.evaluation_set == "historical_regression":
        manifest.update(evaluation_set=args.evaluation_set, blind_test=False, training_allowed=False,
                        historical_builder_sha256=sha(Path(__file__).with_name("historical.py").read_bytes()))
    if cache_plan is not None:
        manifest["entity_cache"] = deepcopy(cache_plan.provenance)
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    _write(out / "run_manifest.json", manifest)
    _write(out / "manifest.json", manifest)
    if args.preflight_only:
        print(canonical(manifest))
        return manifest
    progress = {"status": "running", "generation_jobs_completed": 0, "document_mode_results_completed": 0,
                "expected_document_mode_results": len(cases) * (2 if args.mode == "both" else 1)}
    _write(out / "progress.json", progress)
    try:
        generator = SemanticGenerator(args.base_model, args.entity_adapter, args.relation_adapter,
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        manifest["loaded_adapter_verification"] = deepcopy(getattr(generator, "weight_verification", {}))
        if cache_plan is not None:
            generator = cache_plan.bind(generator)
            manifest["entity_cache"] = deepcopy(generator.provenance)
            progress.update(cached_entity_jobs_replayed=0, new_gpu_entity_jobs=0, new_gpu_relation_jobs=0)
        _write(out / "run_manifest.json", manifest)
        _write(out / "manifest.json", manifest)
        with (out / "generations.jsonl").open("x") as stream:
            (out / "generations.jsonl").chmod(0o600)
            def generation(row):
                stream.write(canonical(row) + "\n")
                stream.flush()
                progress.update(generation_jobs_completed=progress["generation_jobs_completed"] + 1,
                                latest_generation={key: row[key] for key in ("id", "mode", "stage", "status")})
                if cache_plan is not None:
                    key = "cached_entity_jobs_replayed" if row["generation_trace"].get("entity_cache") else ("new_gpu_entity_jobs" if row["stage"] == "entities" else "new_gpu_relation_jobs")
                    progress[key] = progress.get(key, 0) + 1
                _write(out / "progress.json", progress)
            def document(mode, case, result):
                progress.update(document_mode_results_completed=progress["document_mode_results_completed"] + 1,
                                latest_document={"id": case["id"], "mode": mode, "generation_complete": result["trace"]["generation_complete"]})
                _write(out / "progress.json", progress)
            metrics, predictions, _ = evaluate_cases(cases, generator, variant=args.variant, entity_variant=args.entity_variant,
                modes=("live", "oracle") if args.mode == "both" else (args.mode,), short_chars=data_manifest["short_chars"],
                max_seconds=args.max_seconds, on_generation=generation, on_document=document)
        for view in ("raw", "filtered"):
            path = out / f"predictions.{view}.jsonl"
            path.write_text("".join(canonical(row) + "\n" for mode in predictions for row in predictions[mode][view]))
            path.chmod(0o600)
        metrics.update(input_release_sha256=data_manifest["manifest_sha256"], protocol_sha256=manifest["protocol_sha256"],
            entity_strategy=entity_contract,
            relation_training_variant=manifest["relation_training_variant"],
            relation_prompt_matches_training_variant=manifest["relation_prompt_matches_training_variant"],
            artifacts_sha256={name: sha((out / name).read_bytes()) for name in ("generations.jsonl", "predictions.raw.jsonl", "predictions.filtered.jsonl")})
        _write(out / "metrics.json", metrics)
        manifest.update(status=metrics["status"], complete=metrics["complete"], completed=metrics["complete"],
                        failed=False, metrics_sha256=sha((out / "metrics.json").read_bytes()))
        if "entity_cache" in metrics:
            manifest["entity_cache"] = deepcopy(metrics["entity_cache"])
        progress.update(status=metrics["status"], complete=metrics["complete"])
        _write(out / "progress.json", progress)
        _write(out / "run_manifest.json", manifest)
        _write(out / "manifest.json", manifest)
        print(canonical({"status": metrics["status"], "samples": len(cases), "modes": list(metrics["modes"])}))
        return metrics
    except BaseException as exc:
        manifest.update(status="failed", complete=False, completed=False, failed=True, error=f"{type(exc).__name__}: {exc}")
        progress.update(status="failed", complete=False, error=manifest["error"])
        _write(out / "run_manifest.json", manifest)
        _write(out / "manifest.json", manifest)
        _write(out / "progress.json", progress)
        raise


if __name__ == "__main__":
    result = main()
    if result.get("status") == "incomplete":
        raise SystemExit(2)

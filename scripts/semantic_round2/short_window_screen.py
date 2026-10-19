"""Frozen-source semantic pipeline screening; original full gold is always scored.

Uses existing runtime and semantic text contract. This adds no candidate IDs,
JSON targets, gold-selected windows, caches, graph repair, or metric shortcuts.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from semantic.data import canonical, sha, source_windows
from semantic.evaluate import evaluate_cases, load_release, _write
from semantic.runtime import SemanticGenerator, check_adapter_contract, entity_strategy_contract, protocol_digest


def select_cases(cases, sample_count, selection_stratum="all"):
    """Source text alone selects pilot membership; final order stays frozen."""
    if sample_count == 0:
        if selection_stratum != "all":
            raise ValueError("Full130 must use the all stratum")
        return list(cases), {"policy": "all_original_130_tune_cases", "pilot": False,
                             "membership_uses_gold": False, "blind_test": False}
    if sample_count < 1:
        raise ValueError("sample-count must be positive for pilot or zero for full130")
    free = [case for case in cases if case["source_type"] == "free_text"]
    if selection_stratum == "long":
        free = [case for case in free if len(case["text"]) > 1600]
    elif selection_stratum != "all":
        raise ValueError("Unknown source-only selection stratum")
    if sample_count > len(free):
        raise ValueError("Requested pilot exceeds the free-text population")
    ranked = sorted(free, key=lambda case: (sha(case["text"]), str(case["id"])))
    selected_ids = {case["id"] for case in ranked[:sample_count]}
    selected = [case for case in cases if case["id"] in selected_ids]
    return selected, {"policy": "first_N_free_text_sorted_by_sha256_exact_source_text_then_id",
                      "pilot": True, "membership_uses_gold": False, "blind_test": False,
                      "selection_stratum": selection_stratum,
                      "source_length_threshold_exclusive": 1600 if selection_stratum == "long" else None,
                      "selection_order_source_hashes": [sha(case["text"]) for case in ranked[:sample_count]],
                      "nested_sample_counts": True,
                      "limitations": "Exploratory reused tune subset; not a full evaluation or independent test."}


def _jsonl(path, rows):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")


def run(args):
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Existing experiment artifacts cannot be overwritten")
    # Verify the entire original130 release before selecting the source-only pilot.
    all_cases, release = load_release(args.release, expected_samples=130)
    cases, selection = select_cases(all_cases, args.sample_count, args.selection_stratum)
    if args.short_chars < 32:
        raise ValueError("short-chars must be at least32")
    selection.update(samples=len(cases), original_release_samples=len(all_cases),
        original_release_manifest_sha256=release["manifest_sha256"],
        selected=[{"id": case["id"], "sample_index": case["index"],
                   "source_text_sha256": sha(case["text"]), "source_chars": len(case["text"]),
                   "original_record_sha256": case["original_record_sha256"]} for case in cases])
    selection_hash = sha(selection)
    receipts = {stage: check_adapter_contract(path, stage, base_model=args.base_model)
                for stage, path in (("entities", args.entity_adapter), ("relations", args.relation_adapter))}
    strategy = entity_strategy_contract("short_source", receipts["entities"], short_chars=args.short_chars)
    windows = {str(case["id"]): len(source_windows(case["text"], args.short_chars, min(400, args.short_chars//4)))
               for case in cases if case["source_type"] == "free_text"}
    from semantic import runtime, evaluate, data
    manifest = {"status": "preflight_complete" if args.preflight_only else "running",
        "complete": False, "completed": False, "failed": False,
        "args": vars(args), "input_release": release, "selection": selection,
        "selection_sha256": selection_hash, "samples": len(cases), "pilot": bool(args.sample_count),
        "entity_variant": "short_source", "entity_strategy": strategy,
        "relation_prompt_variant": "short_canonical", "relation_training_variant": receipts["relations"]["variant"],
        "output_contract": receipts["entities"]["contract"], "protocol_sha256": protocol_digest(),
        "runtime_sha256": sha(Path(runtime.__file__).read_bytes()),
        "evaluator_sha256": sha(Path(evaluate.__file__).read_bytes()),
        "source_window_code_sha256": sha(Path(data.__file__).read_bytes()),
        "screen_script_sha256": sha(Path(__file__).read_bytes()),
        "sample_indices": [case["index"] for case in cases],
        "adapters": {stage: {key: receipt[key] for key in
                     ("stage", "variant", "adapter_files_sha256", "base_input_sha256", "data_sha256", "adapter_final_sha256")}
                     for stage, receipt in receipts.items()},
        "window_policy": {"short_chars": args.short_chars, "overlap": min(400, args.short_chars//4),
                          "applies_to": ["entity_input", "relation_input"], "gold_used": False,
                          "source_only_windows": sum(windows.values()),
                          "source_only_windows_by_budget": {str(size): sum(len(source_windows(case["text"], size, min(400, size//4)))
                              for case in cases if case["source_type"] == "free_text") for size in (2400, 1600)},
                          "training_char_budget_is_not_changed": True,
                          "long_entity_preservation_is_not_guaranteed": True},
        "comparison_scope": "Fixed E/R weights and generation settings; both input window budgets change. This is inference-only; shorter windows can split long entity names. Batch membership may change with input lengths.",
        "base_input_hash_scope": "Tokenizer/configuration, not all8B weight bytes"}
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o700)
    _write(out/"selection.json", selection)
    _write(out/"run_manifest.json", manifest)
    _write(out/"manifest.json", manifest)
    if args.preflight_only:
        print(canonical({"status": manifest["status"], "samples": len(cases), "selection_sha256": selection_hash,
                         "source_only_windows": sum(windows.values())}))
        return manifest
    progress = {"status": "running", "generation_jobs_completed": 0, "document_mode_results_completed": 0,
                "expected_document_mode_results": len(cases), "started_unix": time.time()}
    _write(out/"progress.json", progress)
    try:
        generator = SemanticGenerator(args.base_model, args.entity_adapter, args.relation_adapter,
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        manifest["loaded_adapter_verification"] = deepcopy(generator.weight_verification)
        _write(out/"run_manifest.json", manifest)
        _write(out/"manifest.json", manifest)
        descriptor = os.open(out/"generations.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            def generation(row):
                stream.write(canonical(row)+"\n"); stream.flush()
                progress.update(generation_jobs_completed=progress["generation_jobs_completed"]+1,
                    latest_generation={key: row[key] for key in ("id", "mode", "stage", "status")})
                _write(out/"progress.json", progress)
            def document(mode, case, result):
                progress.update(document_mode_results_completed=progress["document_mode_results_completed"]+1,
                    latest_document={"id": case["id"], "mode": mode,
                                     "generation_complete": result["trace"]["generation_complete"]})
                _write(out/"progress.json", progress)
            metrics, predictions, results = evaluate_cases(cases, generator, variant="short_canonical",
                entity_variant="short_source", modes=("live",), short_chars=args.short_chars,
                max_seconds=args.max_seconds, on_generation=generation, on_document=document)
        for view in ("raw", "filtered"):
            _jsonl(out/f"predictions.{view}.jsonl", predictions["live"][view])
        _jsonl(out/"results.live.jsonl", results["live"])
        metrics.update(pilot=bool(args.sample_count), selection_sha256=selection_hash,
            original_population_samples=130, full_population_evaluation=args.sample_count == 0,
            input_release_sha256=release["manifest_sha256"], protocol_sha256=manifest["protocol_sha256"],
            window_policy=manifest["window_policy"], comparison_scope=manifest["comparison_scope"],
            artifacts_sha256={name: sha((out/name).read_bytes()) for name in
                ("selection.json", "generations.jsonl", "predictions.raw.jsonl", "predictions.filtered.jsonl", "results.live.jsonl")})
        _write(out/"metrics.json", metrics)
        manifest.update(status=metrics["status"], complete=metrics["complete"], completed=metrics["complete"],
            failed=False, generation_complete=metrics["generation_complete"],
            metrics_sha256=sha((out/"metrics.json").read_bytes()))
        progress.update(status=metrics["status"], complete=metrics["complete"])
        for name in ("run_manifest.json", "manifest.json"):
            _write(out/name, manifest)
        _write(out/"progress.json", progress)
        print(canonical({"status": metrics["status"], "pilot": bool(args.sample_count), "samples": len(cases),
            "short_chars": args.short_chars, "selection_sha256": selection_hash,
            "entity": metrics["modes"]["live"]["filtered"]["free_text"]["entity"],
            "relation": metrics["modes"]["live"]["filtered"]["free_text"]["relation"]}))
        return manifest
    except BaseException as exc:
        # Exception details stay private; public console never prints source text.
        manifest.update(status="failed", complete=False, completed=False, failed=True,
                        error_type=type(exc).__name__)
        _write(out/"failure.private.json", {"type": type(exc).__name__, "message": str(exc)})
        for name in ("run_manifest.json", "manifest.json"):
            _write(out/name, manifest)
        progress.update(status="failed", complete=False, error_type=type(exc).__name__)
        _write(out/"progress.json", progress)
        raise RuntimeError("Screen failed; inspect owner-private failure artifact") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "base-model", "entity-adapter", "relation-adapter", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--sample-count", type=int, default=12, help="12/24 nested free-text pilot;0=all130 original cases")
    parser.add_argument("--selection-stratum", choices=("all", "long"), default="all", help="long selects source-only text length>1600 before hashing")
    parser.add_argument("--short-chars", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    result = run(parser.parse_args())
    if result["status"] == "incomplete":
        raise SystemExit(2)

if __name__ == "__main__":
    main()

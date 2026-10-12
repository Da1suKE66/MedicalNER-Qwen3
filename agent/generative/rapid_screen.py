"""Frozen input-stratified screening, explicitly separate from full evaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

from build_stage_data import _message
from evaluate import _graph, _relation_set, _validate_gold_graph
from evaluate_graphs import evaluate_records
from evaluation_identity import IDENTITY_KEY, make_input_identity, resolve_sample_indices
from data_repair.source_provenance import source_view
from .chat import render_prompt
from .evaluate import candidate_ceiling, load_evaluation_model, rule_ablation, run_records
from .protocol import build_batches, dumps, load_variant

CONDITIONS = {
    "full_hash16": {"context": "full", "order": "hash", "pairs_per_batch": 16},
    "full_source16": {"context": "full", "order": "source", "pairs_per_batch": 16},
    "full_hash4": {"context": "full", "order": "hash", "pairs_per_batch": 4},
    "local_hash16": {"context": "local", "order": "hash", "pairs_per_batch": 16},
}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def condition_configs(config):
    if config.matching != "nfkc_v2":
        raise ValueError("Rapid V4 screening requires frozen nfkc_v2 matching")
    return {name: replace(config, **options) for name, options in CONDITIONS.items()}


def input_profiles(records, config):
    """Selection reads source text and supplied entities, never relation labels."""
    profiles = []
    for index, record in enumerate(records):
        view = source_view(_message(record, "user"))
        if view["source_type"] != "free_text":
            continue
        graph = _graph(_message(record, "assistant"))
        if graph is None:
            raise ValueError(f"Invalid graph {index}")
        _, _, audit = build_batches(view["raw_payload"], graph["entities"], config, sample_id=index)
        pairs, length = audit["candidate_pairs"], len(view["raw_payload"])
        profiles.append({"sample_index": index, "source_characters": length,
            "candidate_pairs": pairs,
            "stratum": f"length{0 if length < 1000 else 1 if length < 3000 else 2}"
                       f"_pairs{0 if pairs <= 12 else 1 if pairs <= 48 else 2}",
            IDENTITY_KEY: make_input_identity(record, index)})
    return profiles


def select_profiles(profiles, *, count=12, candidate_budget=240, per_record_candidates=96, seed=42):
    if min(count, candidate_budget, per_record_candidates) < 1:
        raise ValueError("Selection limits must be positive")
    eligible = [p for p in profiles if 0 < p["candidate_pairs"] <= per_record_candidates]
    if len(eligible) < count or sum(sorted(p["candidate_pairs"] for p in eligible)[:count]) > candidate_budget:
        raise ValueError("Cannot select requested count within candidate budget")
    rng, strata = random.Random(seed), defaultdict(list)
    for profile in eligible:
        strata[profile["stratum"]].append(profile)
    names = sorted(strata)
    rng.shuffle(names)
    for rows in strata.values():
        rng.shuffle(rows)
    selected, used = [], 0
    while len(selected) < count:
        progressed = False
        for name in names:
            for item in list(strata[name]):
                remaining = [p for rows in strata.values() for p in rows if p is not item]
                needed = count - len(selected) - 1
                reserve = sum(sorted(p["candidate_pairs"] for p in remaining)[:needed])
                if len(remaining) >= needed and used + item["candidate_pairs"] + reserve <= candidate_budget:
                    selected.append(item)
                    used += item["candidate_pairs"]
                    strata[name].remove(item)
                    progressed = True
                    break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError("Stratified selection stalled")
    return sorted(selected, key=lambda p: p["sample_index"])


def prepare_screen(gold_path, matrix_path, variant, output_dir, *, count=12,
                   candidate_budget=240, per_record_candidates=96, seed=42,
                   max_new_tokens=2048, condition_seconds=1500, inference_batch_size=4):
    if min(max_new_tokens, condition_seconds, inference_batch_size) < 1:
        raise ValueError("Decode limits must be positive")
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new frozen screen directory")
    config = load_variant(matrix_path, variant)
    configs = condition_configs(config)
    records = json.loads(Path(gold_path).read_text())
    profiles = input_profiles(records, config)
    chosen = select_profiles(profiles, count=count, candidate_budget=candidate_budget,
                             per_record_candidates=per_record_candidates, seed=seed)
    indices = [p["sample_index"] for p in chosen]
    selected = [records[i] for i in indices]
    for record in selected:
        _validate_gold_graph(_graph(_message(record, "assistant")))
    # Labels are consulted for reporting only after selection is frozen.
    denominator = sum(len(_relation_set(_graph(_message(r, "assistant")))) for r in selected)
    plans = {}
    for name, conf in configs.items():
        rows, covered = [], 0
        for record, index in zip(selected, indices):
            graph = _graph(_message(record, "assistant"))
            batches, _, audit = build_batches(source_view(_message(record, "user"))["raw_payload"],
                                               graph["entities"], conf, sample_id=index)
            covered += len(_relation_set(graph) & candidate_ceiling(graph, batches))
            rows.append({"sample_index": index, "batches": len(batches),
                         "candidate_pairs": audit["candidate_pairs"],
                         "prompt_characters": sum(len(b["prompt"]) for b in batches)})
        plans[name] = {"config": asdict(conf), "samples": rows,
                       "expected_batches": sum(r["batches"] for r in rows),
                       "candidate_pairs": sum(r["candidate_pairs"] for r in rows),
                       "candidate_covered_gold_triples": covered}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "records.json", selected)
    write_json(out / "sample_indices.json", indices)
    write_json(out / "matrix.json", json.loads(Path(matrix_path).read_text()))
    manifest = {
        "schema_version": 1, "purpose": "input_stratified_rapid_relation_screen",
        "not_full_tune": True, "not_full_live_pipeline": True, "entity_source": "oracle",
        "selection_uses_relation_labels_or_model_performance": False,
        "selection_entity_inputs": "gold_entities_supplied_to_oracle_relation_stage",
        "selection": {"seed": seed, "target_samples": count, "candidate_budget": candidate_budget,
                      "per_record_candidates": per_record_candidates,
                      "length_boundaries_characters": [1000, 3000], "pair_boundaries": [12, 48],
                      "algorithm": "seeded_stratum_round_robin_with_minimum_cost_reserve"},
        "source_sha256": sha_file(gold_path), "source_record_count": len(records),
        "source_free_text_count": len(profiles),
        "source_free_text_candidates": sum(p["candidate_pairs"] for p in profiles),
        "excluded_zero_candidates": sum(p["candidate_pairs"] == 0 for p in profiles),
        "excluded_over_per_record_budget": sum(p["candidate_pairs"] > per_record_candidates for p in profiles),
        "unselected_eligible": sum(0 < p["candidate_pairs"] <= per_record_candidates for p in profiles) - count,
        "variant": variant, "matrix_sha256": sha_file(out / "matrix.json"),
        "original_matrix_sha256": sha_file(matrix_path),
        "records_sha256": sha_file(out / "records.json"),
        "sample_indices_sha256": sha_file(out / "sample_indices.json"),
        "sample_indices": indices, "selected_profiles": chosen,
        "denominators": {"samples": len(selected), "gold_relation_triples": denominator,
                         "candidate_pairs": sum(p["candidate_pairs"] for p in chosen)},
        "decode": {"max_new_tokens": max_new_tokens, "condition_seconds": condition_seconds,
                   "inference_batch_size": inference_batch_size},
        "conditions": plans,
        "limitations": ["Excludes zero-candidate and high-candidate documents; cannot generalize to excluded population.",
                        "Oracle relation screening is not full entity-to-relation pipeline performance.",
                        "2048-token screening differs from unchanged formal 16000-token evaluation.",
                        "Inference-only format changes do not establish benefits of format-specific retraining.",
                        "Deadline/runtime failure keeps all denominators but disqualifies the condition from ranking."],
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def load_screen(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, key in (("records.json", "records_sha256"), ("sample_indices.json", "sample_indices_sha256"),
                      ("matrix.json", "matrix_sha256")):
        if sha_file(root / name) != manifest[key]:
            raise ValueError(f"Frozen screen hash mismatch: {name}")
    records = json.loads((root / "records.json").read_text())
    indices = resolve_sample_indices(records, json.loads((root / "sample_indices.json").read_text()))
    if indices != manifest["sample_indices"] or len(records) != manifest["denominators"]["samples"]:
        raise ValueError("Frozen screen denominator/index mismatch")
    if len(manifest["selected_profiles"]) != len(records):
        raise ValueError("Frozen screen profile count mismatch")
    for record, index, profile in zip(records, indices, manifest["selected_profiles"]):
        if make_input_identity(record, index) != profile[IDENTITY_KEY]:
            raise ValueError("Frozen screen identity mismatch")
    configs = condition_configs(load_variant(root / "matrix.json", manifest["variant"]))
    if {n: asdict(c) for n, c in configs.items()} != {n: p["config"] for n, p in manifest["conditions"].items()}:
        raise ValueError("Frozen screen condition mismatch")
    return records, indices, configs, manifest


def run_condition(records, indices, config, generate_group, output_dir, *,
                  expected_batches, max_new_tokens, condition_seconds, inference_batch_size=4,
                  manifest_sha256, clock=time.monotonic):
    """Preserve full denominators after failure; incomplete runs have no main F1."""
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new condition output directory")
    out.mkdir(parents=True, exist_ok=True)
    started, abort_reason = clock(), None
    deadline = started + condition_seconds

    def many(batches):
        nonlocal abort_reason
        results = []
        for offset in range(0, len(batches), inference_batch_size):
            group = batches[offset:offset + inference_batch_size]
            if clock() >= deadline:
                abort_reason = abort_reason or "condition_deadline_exceeded"
            if abort_reason:
                generated = [{"text": "", "tokens": 0, "error": abort_reason,
                              "not_generated": True} for _ in group]
            else:
                try:
                    generated = generate_group(group, deadline)
                    if len(generated) != len(group):
                        raise ValueError("Generation batch alignment mismatch")
                except (RuntimeError, TimeoutError) as error:
                    abort_reason = f"generation_failed:{type(error).__name__}"
                    generated = [{"text": "", "tokens": 0, "error": abort_reason,
                                  "error_detail": str(error)[:1000]} for _ in group]
                # The final generation can return after the deadline without
                # its stopping criterion observing the overrun (for example,
                # during device synchronization or decoding). Keep its output
                # for full-denominator diagnostics, but disqualify the run.
                if clock() >= deadline:
                    abort_reason = abort_reason or "condition_deadline_exceeded"
                    generated = [{**row, "error": row.get("error") or "condition_deadline_exceeded"}
                                 for row in generated]
                if any(r.get("error") == "condition_deadline_exceeded" for r in generated):
                    abort_reason = "condition_deadline_exceeded"
            with (out / "raw_stream.jsonl").open("a") as stream:
                for batch, result in zip(group, generated):
                    row = {"id": batch["id"], "sample_id": batch["sample_id"],
                           IDENTITY_KEY: copy.deepcopy(batch[IDENTITY_KEY]),
                           "prompt": batch["prompt"], **result}
                    stream.write(dumps(row) + "\n")
            results.extend(generated)
        return results

    predictions, raw, audit = run_records(records, None, config, None, oracle=True,
        generate_many=many, strict_identity=True, sample_indices=indices)
    metrics = evaluate_records(records, predictions, strict_identity=True, sample_indices=indices)
    if len(predictions) != len(records) or len(raw) != expected_batches:
        raise ValueError("Screen completion denominator mismatch")
    rule_reports = rule_ablation(records, predictions, strict_identity=True, sample_indices=indices)
    finished = clock()
    if finished >= deadline:
        abort_reason = abort_reason or "condition_deadline_exceeded"
    failures = Counter(r.get("error") for r in raw if r.get("error"))
    complete = not failures and not abort_reason
    report = {
        "status": "complete" if complete else "incomplete", "complete": complete,
        "eligible_for_ranking": complete, "main_relation_metric": metrics["free_text"]["relation"] if complete else None,
        "metrics_including_all_failed_rows": metrics, "coverage_and_generation": audit,
        "rule_ablation": rule_reports,
        "failure_counts": dict(failures), "abort_reason": abort_reason,
        "elapsed_seconds": finished - started, "expected_samples": len(records),
        "expected_batches": expected_batches, "observed_samples": len(predictions),
        "observed_batches": len(raw), "decode_limit": max_new_tokens,
        "truncated_batches": sum(bool(r.get("truncated")) for r in raw),
        "truncation_rate": sum(bool(r.get("truncated")) for r in raw) / max(1, len(raw)),
        "parse_error_batches": sum(bool(r.get("parse_errors")) for r in raw),
        "source_manifest_sha256": manifest_sha256,
        "config": asdict(config), "not_full_tune": True, "not_full_live_pipeline": True,
    }
    (out / "predictions.jsonl").write_text("".join(dumps(r) + "\n" for r in predictions))
    (out / "generations.jsonl").write_text("".join(dumps(r) + "\n" for r in raw))
    (out / "raw_stream.jsonl").touch(exist_ok=True)
    report["artifact_sha256"] = {name: sha_file(out / name) for name in (
        "predictions.jsonl", "generations.jsonl", "raw_stream.jsonl")}
    write_json(out / "metrics.json", report)
    print(dumps({"condition": out.name, "complete": complete,
                 "main_relation_metric": report["main_relation_metric"],
                 "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


def make_generator(model, tokenizer, max_new_tokens):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from kg_agent.prompts import extract_json_object

    def generate(group, deadline):
        encoded = tokenizer([render_prompt(tokenizer, b["prompt"]) for b in group],
                            padding=True, return_tensors="pt").to(model.device)
        n = encoded["input_ids"].shape[1]
        if n + max_new_tokens > model.config.max_position_embeddings:
            return [{"text": "", "tokens": 0, "error": "context_budget_exceeded"} for _ in group]
        timed_out = False

        class CompleteOrDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                nonlocal timed_out
                timed_out = time.monotonic() >= deadline
                if timed_out:
                    return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                done = []
                for row in input_ids[:, n:]:
                    try:
                        extract_json_object(tokenizer.decode(row, skip_special_tokens=True))
                        done.append(True)
                    except ValueError:
                        done.append(False)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([CompleteOrDeadline()]))[:, n:]
        results = []
        for row, mask in zip(output, encoded["attention_mask"]):
            ids = row.tolist()
            if tokenizer.pad_token_id in ids:
                ids = ids[:ids.index(tokenizer.pad_token_id)]
            result = {"text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                      "tokens": len(ids), "input_tokens": int(mask.sum()),
                      "truncated": len(ids) >= max_new_tokens,
                      "amortized_seconds": (time.monotonic() - started) / len(group)}
            if timed_out:
                result["error"] = "condition_deadline_exceeded"
            results.append(result)
        return results
    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    for flag in ("gold", "matrix", "variant", "output-dir"):
        prepare.add_argument("--" + flag, required=True)
    for flag, default in (("count", 12), ("candidate-budget", 240), ("per-record-candidates", 96),
                          ("seed", 42), ("max-new-tokens", 2048), ("condition-seconds", 1500),
                          ("inference-batch-size", 4)):
        prepare.add_argument("--" + flag, type=int, default=default)
    evaluate = sub.add_parser("evaluate")
    for flag in ("screen-dir", "base-model", "output-dir"):
        evaluate.add_argument("--" + flag, required=True)
    model = evaluate.add_mutually_exclusive_group(required=True)
    model.add_argument("--adapter")
    model.add_argument("--base-only", action="store_true")
    evaluate.add_argument("--grpo-sft-adapter")
    evaluate.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()
    if args.action == "prepare":
        options = vars(args).copy()
        for key in ("action", "gold", "matrix", "variant", "output_dir"):
            options.pop(key)
        manifest = prepare_screen(args.gold, args.matrix, args.variant, args.output_dir, **options)
        print(dumps({"denominators": manifest["denominators"], "sample_indices": manifest["sample_indices"],
                     "expected_batches": {n: p["expected_batches"] for n, p in manifest["conditions"].items()}}))
        return
    if args.base_only and args.grpo_sft_adapter:
        parser.error("--grpo-sft-adapter requires --adapter")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions must be unique")
    records, indices, configs, manifest = load_screen(args.screen_dir)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new screen output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest_sha = sha_file(Path(args.screen_dir) / "manifest.json")
    write_json(out / "run.json", {"status": "loading", "arguments": vars(args), "manifest_sha256": manifest_sha})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_evaluation_model(args.base_model, args.adapter, base_only=args.base_only,
                                  grpo_sft_adapter=args.grpo_sft_adapter)
    generate = make_generator(model, tokenizer, manifest["decode"]["max_new_tokens"])
    summaries = {}
    for name in args.conditions:
        report = run_condition(records, indices, configs[name], generate, out / name,
            expected_batches=manifest["conditions"][name]["expected_batches"],
            manifest_sha256=manifest_sha, **manifest["decode"])
        summaries[name] = {k: report[k] for k in ("complete", "main_relation_metric", "elapsed_seconds", "truncation_rate")}
        write_json(out / "run.json", {"status": "running", "arguments": vars(args),
                                      "manifest_sha256": manifest_sha, "conditions": summaries})
    write_json(out / "run.json", {"status": "complete" if all(x["complete"] for x in summaries.values()) else "incomplete",
                                  "arguments": vars(args), "conditions": summaries, "manifest_sha256": manifest_sha})


if __name__ == "__main__":
    main()

"""Full semantic artificial holdout, relation-only output evaluation.

This is a component comparison with authored entity candidates, not medical or
end-to-end quality evidence. No length, label, prediction, or success filtering
is permitted. Failed generations remain in the gold denominator as full FN.
Historical integer choices are the default; literal labels require an explicit
format selection or the evaluate_label_relations entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from generative.choice_data import compile_recovery_row, validate_row
from generative.choice_preferences import DATA_VERSION as PREFERENCE_VERSION, validate_preference_row
from generative.recovery_preferences import STRUCTURAL, verify_case
from kg_agent.choice_backend import ChoiceFailure, ChoiceGenerator, check_adapter_contract
from kg_agent import label_backend


SCOPE = "artificial_semantic_relation_holdout_not_medical_not_pipeline"
SELECTION = "all_recipe_verified_mechanical_holdout_preferences_except_four_structural_rejection_kinds"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _check_format(relation_format):
    if relation_format not in ("choices", "labels"):
        raise ValueError("relation_format must be choices or labels")


def prepare_cases(sources, *, relation_format="choices"):
    """Freeze all semantic cases before generation; structural exclusions listed."""
    _check_format(relation_format)
    if not isinstance(sources, list) or not sources:
        raise ValueError("A nonempty raw recovery holdout is required")
    cases, exclusions, seen = [], [], set()
    for index, input_row in enumerate(sources):
        if isinstance(input_row, dict) and input_row.get("data_version") == PREFERENCE_VERSION:
            validate_preference_row(input_row, require_train=False)
            source = input_row["source_row"]
        elif relation_format == "labels" and isinstance(input_row, dict) and input_row.get("data_version") == "artificial-literal-relation-data-v1":
            from generative.label_data import validate_row as validate_label_row
            validate_label_row(input_row, require_train=False)
            source = input_row["source_row"]
        else:
            source = input_row
        verify_case(source)
        if source["split"] != "mechanical_holdout" or source["training_view"] != "preference":
            raise ValueError("Only recipe-verified raw mechanical_holdout preferences are accepted")
        if source["id"] in seen:
            raise ValueError("Duplicate held-out source ID")
        seen.add(source["id"])
        # This exact reconstruction also checks source field types, including
        # bool/int distinctions. Compile before exclusions to verify every row.
        compiled = compile_recovery_row(source, "relations")
        task = validate_row(compiled, require_train=False)
        identity = {"id": input_row["id"], "source_id": source["id"], "source_index": index,
                    "input_row_sha256": digest(input_row),
                    "source_row_sha256": digest(source), "kind": source["reason"]}
        if source["reason"] in STRUCTURAL:
            exclusions.append({**identity, "reason": "structural_rejected_not_representable_as_legal_actions"})
            continue
        gold_edges = task.edges(compiled["actions"])
        target = {"gold_actions": list(compiled["actions"])}
        if relation_format == "labels":
            from generative.label_data import compile_recovery_row as compile_labels, validate_row as validate_labels
            literal = compile_labels(source)
            task = validate_labels(literal, require_train=False)
            if _sets(gold_edges) != _sets(literal["target"]):
                raise ValueError("Literal conversion changed certified gold relations or evidence")
            if compiled["task_input"] != literal["task_input"]:
                raise ValueError("Literal conversion changed frozen input candidates")
            target = {"gold_completion_text": literal["completion"],
                      "output_contract": label_backend.CONTRACT, "relation_format": "labels"}
        cases.append({**identity, "task": task, "task_input": compiled["task_input"],
                      "task_input_sha256": digest(compiled["task_input"]),
                      "prompt_sha256": hashlib.sha256(task.prompt.encode()).hexdigest(),
                      "prompt_chars": len(task.prompt), "source_chars": len(source["oracle_source_text"]),
                      **target, "gold_edges": gold_edges})
    if not cases:
        raise ValueError("No semantic holdout cases remain")
    manifest = {"scope": SCOPE, "selection_policy": SELECTION, "source_samples": len(sources),
                "samples": len(cases), "excluded": exclusions,
                "accepted": [{key: case[key] for key in ("id", "source_id", "source_index", "input_row_sha256", "source_row_sha256",
                    "kind", "task_input_sha256", "prompt_sha256")} for case in cases],
                "selection_uses_predictions": False, "selection_uses_lengths": False,
                "selection_uses_gold_performance": False}
    if relation_format == "labels":
        manifest.update(relation_format="labels", output_contract=label_backend.CONTRACT)
    manifest["selection_sha256"] = digest(manifest)
    return cases, manifest


def _sets(edges):
    return ({(edge["pair_id"], edge["relation"]) for edge in edges},
            {(edge["pair_id"], edge["relation"], span)
             for edge in edges for span in edge["evidence_span_ids"]})


def _counts(gold, predicted):
    return {"tp": len(gold & predicted), "fp": len(predicted - gold), "fn": len(gold - predicted)}


def _metrics(counts):
    tp, fp, fn = (counts.get(key, 0) for key in ("tp", "fp", "fn"))
    return {"tp": tp, "fp": fp, "fn": fn, "gold_count": tp + fn,
            "predicted_count": tp + fp,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def _length_bin(length):
    for bound in (1024, 4096, 8192, 16384):
        if length <= bound:
            return f"le_{bound}"
    return "gt_16384"


def summarize(records):
    """Recompute from stored edge sets, never average per-row F1 or drop errors."""
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Summary requires nonempty uniquely identified rows")
    totals = {"relations": Counter(), "relation_evidence": Counter()}
    per_label = defaultdict(lambda: {"relations": Counter(), "relation_evidence": Counter()})
    groups = defaultdict(list)
    for row in records:
        if row["status"] != "complete" and row["predicted_edges"]:
            raise ValueError("Failed action sequences may not materialize predictions")
        gold, gold_evidence = _sets(row["gold_edges"])
        predicted, predicted_evidence = _sets(row["predicted_edges"])
        for key, g, p in (("relations", gold, predicted),
                          ("relation_evidence", gold_evidence, predicted_evidence)):
            counts = _counts(g, p)
            if row[key + "_counts"] != counts:
                raise ValueError("Per-row counts disagree with edge sets")
            totals[key].update(counts)
            for label in {item[1] for item in g | p}:
                per_label[label][key].update(_counts({item for item in g if item[1] == label},
                                                   {item for item in p if item[1] == label}))
        for name, key in (("kind", row["kind"]), ("prompt_chars", _length_bin(row["prompt_chars"])),
                          ("candidate_pairs", str(row["candidate_pairs"]))):
            groups[name, key].append(row)
    strata = defaultdict(dict)
    for (name, key), rows in groups.items():
        strata[name][key] = {"samples": len(rows), "failed_samples": sum(r["status"] != "complete" for r in rows),
            **{metric: _metrics(sum((Counter(row[metric + "_counts"]) for row in rows), Counter()))
               for metric in totals}}
    failed = sum(row["status"] != "complete" for row in records)
    deadline = sum(bool(row["trace"].get("deadline_exhausted")) for row in records)
    return {"status": "incomplete" if deadline else "complete",
        "complete": not bool(deadline), "generation_complete": not bool(failed),
        "status_meaning": "All requested rows accounted for; generation failures are included in FN. A deadline makes the run incomplete.",
        "scope": SCOPE, "samples": len(records), "accounted_samples": len(records),
        "generation_attempted_samples": sum(row["generation_attempted"] for row in records),
        "complete_generation_count": len(records) - failed, "failed_samples": failed,
        "budget_exhaustions": sum(bool(row["trace"].get("budget_exhausted")) for row in records),
        "deadline_exhaustions": deadline,
        "errors_by_type": dict(Counter(row["error_type"] for row in records if row.get("error_type"))),
        "relation": _metrics(totals["relations"]),
        "evidence": _metrics(totals["relation_evidence"]),
        "exact_relation_sets": sum(row["relations_counts"]["fp"] == row["relations_counts"]["fn"] == 0
                                   and row["status"] == "complete" for row in records),
        "exact_evidence_sets": sum(row["relation_evidence_counts"]["fp"] == row["relation_evidence_counts"]["fn"] == 0
                                   and row["status"] == "complete" for row in records),
        "per_relation": {label: {key: _metrics(value) for key, value in counts.items()}
                         for label, counts in sorted(per_label.items())},
        "strata": dict(strata), "medical_quality_evaluated": False,
        "metric_contract": {"relations": "set of (fixed input pair_id, relation label), micro TP/FP/FN",
            "relation_evidence": "set of (pair_id, relation label, visible evidence span_id), micro TP/FP/FN",
            "failure_policy": "All failed rows keep all gold as FN; partial actions are diagnostic only",
            "empty_denominator": "Precision/recall/F1 are zero for their respective zero denominators; exact empty cases counted separately",
            "stratification": "All admitted rows retained; fixed prompt-character bins plus kind and candidate count",
            "scope": "Authored relation candidates only; no entity-stage or medical generalization claim"}}


def checked_label_result(task, returned_edges, trace):
    """Recheck the exact model text; incomplete or invalid prefixes yield no edges."""
    if (trace.get("output_contract") != label_backend.CONTRACT
            or trace.get("eos_observed") is not True or trace.get("complete") is not True
            or trace.get("error") or trace.get("budget_exhausted") or trace.get("deadline_exhausted")):
        raise ChoiceFailure("Incomplete literal-label response", trace)
    try:
        edges = task.parse(trace.get("completion_text"))
        # Equality is semantic: record and evidence ordering is unrestricted.
        if _sets(edges) != _sets(returned_edges):
            raise ValueError("Returned edges differ from literal completion text")
    except (ValueError, TypeError, KeyError) as exc:
        raise ChoiceFailure("Invalid literal-label response: " + str(exc), trace) from exc
    return edges


def evaluate_cases(cases, generator, *, relation_format="choices", max_seconds=None, on_record=None, clock=time.monotonic):
    _check_format(relation_format)
    if not cases:
        raise ValueError("No cases to evaluate")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    records, start = [], clock()
    for case in cases:
        row_start = clock()
        target_keys = ("gold_actions",) if relation_format == "choices" else ("gold_completion_text", "output_contract", "relation_format")
        record = {key: deepcopy(case[key]) for key in ("id", "source_id", "source_index", "input_row_sha256", "source_row_sha256", "kind",
            "task_input", "task_input_sha256", "prompt_sha256", "prompt_chars", "source_chars", "gold_edges") + target_keys}
        record.update({"actions": []} if relation_format == "choices" else {"completion_text": None})
        record.update(predicted_edges=[], candidate_pairs=len(case["task"].pairs),
                      trace={}, generation_attempted=False)
        try:
            if max_seconds is not None and clock() - start >= max_seconds:
                raise ChoiceFailure("Evaluation deadline reached before generation", {"deadline_exhausted": True})
            record["generation_attempted"] = True
            result, trace = generator.run(case["task"])
            record["trace"] = deepcopy(trace)
            if relation_format == "choices":
                record["actions"] = list(result)
            else:
                record["completion_text"] = trace.get("completion_text")
            if max_seconds is not None and clock() - start > max_seconds:
                raise ChoiceFailure("Generation exceeded evaluation deadline", {**trace, "deadline_exhausted": True})
            if trace.get("error") or trace.get("complete") is False:
                raise ChoiceFailure("Generator returned an incomplete action sequence", trace)
            record["predicted_edges"] = (case["task"].edges(result) if relation_format == "choices"
                                         else checked_label_result(case["task"], result, trace))
            record["status"] = "complete"
        except Exception as exc:
            trace = deepcopy(getattr(exc, "trace", record["trace"]))
            record.update(status="failed", predicted_edges=[], trace=trace,
                          error_type=type(exc).__name__, error=str(exc))
            if relation_format == "choices" and "actions" in trace:
                record["actions"] = list(trace["actions"])
            elif relation_format == "labels":
                record["completion_text"] = trace.get("completion_text")
        record["elapsed_seconds"] = max(0.0, clock() - row_start)
        gold, evidence = _sets(record["gold_edges"])
        predicted, predicted_evidence = _sets(record["predicted_edges"])
        record["relations_counts"] = _counts(gold, predicted)
        record["relation_evidence_counts"] = _counts(evidence, predicted_evidence)
        records.append(record)
        if on_record is not None:
            on_record(record)
    report = summarize(records)
    if relation_format == "labels":
        report.update(relation_format="labels", output_contract=label_backend.CONTRACT)
        report["metric_contract"]["failure_policy"] = "Every invalid, non-EOS or failed response keeps all gold as FN; partial literal text is diagnostic only"
    report["elapsed_seconds"] = max(0.0, clock() - start)
    return report, records


def load_generator(args, *, relation_format="choices"):
    _check_format(relation_format)
    check = check_adapter_contract if relation_format == "choices" else label_backend.check_adapter_contract
    receipt = check(args.adapter, "relations", base_model=args.base_model)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Authorized BF16 CUDA device is required")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="default",
        is_trainable=False, local_files_only=True).to("cuda").eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    backend = SimpleNamespace(model=model, tokenizer=tokenizer, adapter_name="default")
    generator_class = ChoiceGenerator if relation_format == "choices" else label_backend.LabelGenerator
    return generator_class(backend, max_input_tokens=args.max_input_tokens,
                           max_new_tokens=args.max_new_tokens), receipt


def parse_args(argv=None, *, relation_format="choices"):
    _check_format(relation_format)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "adapter", "base-model", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--relation-format", choices=("choices", "labels"), default=relation_format)
    args = parser.parse_args(argv)
    if args.max_input_tokens < 1 or args.max_new_tokens < 1 or (
            args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0)):
        parser.error("Token/time budgets must be positive and finite")
    return args


def main(argv=None, *, relation_format="choices"):
    args = parse_args(argv, relation_format=relation_format)
    relation_format = args.relation_format
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new output directory; existing runs are immutable")
    source_bytes = Path(args.data).read_bytes()
    sources = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    cases, selection = prepare_cases(sources, relation_format=relation_format)
    check = check_adapter_contract if relation_format == "choices" else label_backend.check_adapter_contract
    receipt = check(args.adapter, "relations", base_model=args.base_model)
    manifest = {"status": "preflight_complete" if args.preflight_only else "running", "scope": SCOPE,
                "args": vars(args), "data_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "adapter_sha256": receipt["adapter_sha256"], "protocol_sha256": receipt["protocol_sha256"],
                "base_input_sha256": receipt["base_input_sha256"], "selection": selection,
                "decoding": {"do_sample": False, "num_beams": 1,
                             "max_input_tokens": args.max_input_tokens, "max_new_tokens": args.max_new_tokens}}
    if relation_format == "labels":
        manifest.update(relation_format="labels", output_contract=label_backend.CONTRACT)
    out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save("run_manifest.json", manifest)
    if args.preflight_only:
        print(json.dumps(manifest, ensure_ascii=False))
        return
    try:
        generator, _ = load_generator(args) if relation_format == "choices" else load_generator(args, relation_format="labels")
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            def append(row):
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            metrics, _ = evaluate_cases(cases, generator, relation_format=relation_format,
                                        max_seconds=args.max_seconds, on_record=append)
        metrics.update(data_sha256=manifest["data_sha256"], adapter_sha256=manifest["adapter_sha256"],
                       selection_sha256=selection["selection_sha256"],
                       predictions_sha256=hashlib.sha256((out / "predictions.jsonl").read_bytes()).hexdigest())
        save("metrics.json", metrics)
        manifest.update(status=metrics["status"], complete=metrics["complete"],
                        metrics_sha256=hashlib.sha256((out / "metrics.json").read_bytes()).hexdigest())
        save("run_manifest.json", manifest)
        print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        manifest.update(status="failed", complete=False, error_type=type(exc).__name__, error=str(exc))
        save("run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

"""Full-document generation evaluation, strict graph metrics and rule ablation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

from build_stage_data import _message
from evaluate import _graph, _relation_set
from evaluate_graphs import evaluate_records
from kg_agent.normalize import normalize_text
from kg_agent.pairwise import relation_rejection
from kg_agent.source import SourceRouter, parse_icd_graph
from kg_agent.assemble import assemble_graph
from .protocol import build_batches, dumps, load_variant, parse_decisions
from .chat import render_prompt


def candidate_ceiling(graph, batches):
    nodes = {
        str(n["id"]): (n["label"], normalize_text(n["name"])) for n in graph["entities"]
    }
    return {
        (nodes[p["source"]], r, nodes[p["target"]])
        for b in batches
        for p in b["pairs"]
        for r in p["allowed"]
    }


def assemble_decisions(batch, text):
    parsed, errors = parse_decisions(text, batch)
    pairs = {p["pair_id"]: p for p in batch["pairs"]}
    result = []
    for edge in parsed:
        pair = pairs[edge["pair_id"]]
        # Bind only cited spans, never fabricate a contiguous range through
        # unseen text. Original-source spelling is restored after text repair.
        evidence = [batch["original_spans"][s] for s in edge["evidence_span_ids"]]
        result.append(
            {
                "source": pair["source"],
                "target": pair["target"],
                "relation": edge["relation"],
                "evidence": evidence[0] if len(evidence) == 1 else evidence,
            }
        )
    # Semantically invalid emitted edges must survive as FP in strict metrics.
    # Pure missing/order/JSON errors are stage failures and missed targets FN.
    edge_errors = {
        "invalid_decision",
        "unknown_pair",
        "invalid_relations",
        "invalid_edge",
        "unknown_relation",
        "invalid_evidence",
    }
    for i, reason in enumerate(errors):
        if reason in edge_errors:
            result.append(
                {
                    "source": "INVALID:" + batch["id"] + f":{i}",
                    "target": "INVALID",
                    "relation": "INVALID:" + reason,
                    "evidence": "",
                }
            )
    return result, errors


def run_records(records, cached, config, generate, *, oracle=False, generate_many=None):
    if cached is not None and len(cached) != len(records):
        raise ValueError("Cache/gold length mismatch; never guess sample alignment")
    if not oracle and cached is None:
        raise ValueError("Predicted-entity evaluation requires a full aligned cache")
    predictions, raw_generations = [], []
    totals = Counter()
    for i, record in enumerate(records):
        doc = SourceRouter().route(_message(record, "user"))
        gold = _graph(_message(record, "assistant"))
        if gold is None:
            raise ValueError(f"Invalid gold record {i}")
        if doc.source_type == "structured_icd":
            graph = (
                _graph(cached[i])
                if cached is not None
                else assemble_graph(parse_icd_graph(doc))
            )
            if graph is None:
                raise ValueError(f"Invalid ICD cached graph {i}")
            predictions.append(
                {"output": graph, "trace": {"source_type": "structured_icd"}}
            )
            continue
        supplied = gold if oracle else _graph(cached[i])
        if supplied is None:
            raise ValueError(f"Invalid entity cache row {i}")
        graph = {"entities": copy.deepcopy(supplied["entities"]), "relations": []}
        batches, _, audit = build_batches(
            doc.raw, graph["entities"], config, sample_id=i
        )
        totals["gold_triples"] += len(_relation_set(gold))
        totals["candidate_covered_triples"] += len(
            _relation_set(gold) & candidate_ceiling(graph, batches)
        )
        totals["candidate_pairs"] += audit["candidate_pairs"]
        totals["ungrounded_entities"] += len(audit["ungrounded_ids"])
        chunks = []
        generated_rows = (
            generate_many(batches)
            if generate_many is not None
            else [generate(batch) for batch in batches]
        )
        if len(generated_rows) != len(batches):
            raise ValueError("Generation batch alignment mismatch")
        for batch, generated in zip(batches, generated_rows):
            text = generated.get("text", "")
            edges, errors = assemble_decisions(batch, text)
            if generated.get("error"):
                errors.append(generated["error"])
            graph["relations"].extend(edges)
            chunks.append({"id": batch["id"], "relation_errors": errors})
            totals["generated_batches"] += 1
            totals["truncated_batches"] += bool(generated.get("truncated"))
            totals["generated_tokens"] += generated.get("tokens", 0)
            totals["empty_edge_batches"] += not edges
            raw_generations.append(
                {
                    "id": batch["id"],
                    "sample_id": i,
                    "prompt": batch["prompt"],
                    **generated,
                    "parse_errors": errors,
                }
            )
        predictions.append(
            {
                "output": graph,
                "trace": {
                    "source_type": "free_text",
                    "chunks": chunks,
                    "entity_source": "oracle" if oracle else "cached_predictions",
                    "grounding": audit,
                },
            }
        )
        print(
            dumps(
                {
                    "completed_sample": i,
                    "total_samples": len(records),
                    "candidate_pairs": audit["candidate_pairs"],
                    "relation_edges": len(graph["relations"]),
                }
            ),
            flush=True,
        )
    return predictions, raw_generations, dict(totals)


def rule_ablation(records, predictions):
    rules = {
        "symptom_disease_type_mismatch",
        "different_disease_in_evidence",
        "enumerated_siblings",
    }
    reports = {}
    for name, enabled in [("all_rules", rules)] + [(r, {r}) for r in sorted(rules)]:
        filtered = copy.deepcopy(predictions)
        removed = Counter()
        for record, row in zip(records, filtered):
            gold = _graph(_message(record, "assistant"))
            graph = row["output"]
            accepted = []
            for edge in graph["relations"]:
                reason = relation_rejection(edge, graph["entities"])
                if reason in enabled:
                    key = _relation_set(
                        {"entities": graph["entities"], "relations": [edge]}
                    )
                    removed[
                        "removed_tp" if key & _relation_set(gold) else "removed_fp"
                    ] += 1
                    removed[reason] += 1
                else:
                    accepted.append(edge)
            graph["relations"] = accepted
        reports[name] = {
            "removed": dict(removed),
            "metrics": evaluate_records(records, filtered),
        }
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--entity-predictions")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument(
        "--grpo-sft-adapter",
        help="Rebuild merged SFT base before loading residual GRPO LoRA",
    )
    p.add_argument("--variant", required=True)
    p.add_argument(
        "--matrix", default=str(Path(__file__).with_name("experiments.json"))
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=16000)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument(
        "--raw-prompt-diagnostic",
        action="store_true",
        help="Explicit inference-template mismatch diagnostic; never use to select normal checkpoints",
    )
    args = p.parse_args()
    if args.inference_batch_size < 1:
        p.error("inference-batch-size must be positive")
    records = json.loads(Path(args.gold).read_text())
    cached = (
        [
            json.loads(s)
            for s in Path(args.entity_predictions).read_text().splitlines()
            if s.strip()
        ]
        if args.entity_predictions
        else None
    )
    config = load_variant(args.matrix, args.variant)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new evaluation output directory")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from peft import PeftModel
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        StoppingCriteria,
        StoppingCriteriaList,
    )
    from kg_agent.prompts import extract_json_object

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if args.grpo_sft_adapter:
        model = PeftModel.from_pretrained(
            model, args.grpo_sft_adapter
        ).merge_and_unload()
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()

    def generate_group(batches):
        if not batches:
            return []
        encoded = tokenizer(
            [
                (
                    b["prompt"]
                    if args.raw_prompt_diagnostic
                    else render_prompt(tokenizer, b["prompt"])
                )
                for b in batches
            ],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        n = encoded["input_ids"].shape[1]
        if n + args.max_new_tokens > model.config.max_position_embeddings:
            return [
                {"text": "", "error": "context_budget_exceeded", "tokens": 0}
                for _ in batches
            ]

        class CompleteJSON(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = []
                for row in input_ids[:, n:]:
                    try:
                        extract_json_object(
                            tokenizer.decode(row, skip_special_tokens=True)
                        )
                        done.append(True)
                    except ValueError:
                        done.append(False)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([CompleteJSON()]),
            )[:, n:]
        elapsed = time.monotonic() - started
        results = []
        for batch, row, mask in zip(batches, output, encoded["attention_mask"]):
            ids = row.tolist()
            if tokenizer.pad_token_id in ids:
                ids = ids[: ids.index(tokenizer.pad_token_id)]
            result = {
                "text": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                "tokens": len(ids),
                "input_tokens": int(mask.sum()),
                "truncated": len(ids) >= args.max_new_tokens,
                "amortized_seconds": elapsed / len(batches),
            }
            results.append(result)
        with (out / "raw_stream.jsonl").open("a") as stream:
            for batch, result in zip(batches, results):
                stream.write(dumps({"id": batch["id"], **result}) + "\n")
        return results

    def generate_many(batches):
        results = []
        for i in range(0, len(batches), args.inference_batch_size):
            results.extend(generate_group(batches[i : i + args.inference_batch_size]))
        return results

    predictions, raw, audit = run_records(
        records, cached, config, None, oracle=args.oracle, generate_many=generate_many
    )
    metrics = evaluate_records(records, predictions)
    report = {
        "metrics": metrics,
        "coverage_and_generation": audit,
        "rule_ablation": rule_ablation(records, predictions),
        "arguments": vars(args),
        "entity_source": "oracle" if args.oracle else "fixed_predicted_cache",
        "not_full_live_pipeline": True,
    }
    (out / "predictions.jsonl").write_text(
        "".join(dumps(r) + "\n" for r in predictions)
    )
    (out / "generations.jsonl").write_text("".join(dumps(r) + "\n" for r in raw))
    (out / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(dumps({"free_text": metrics["free_text"]["relation"], "coverage": audit}))


if __name__ == "__main__":
    main()

"""Evaluate a selected relation head with cached predicted entities or oracle nodes."""

from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from build_stage_data import _message
from evaluate import _graph, _relation_set
from evaluate_graphs import evaluate_records
from kg_agent.normalize import normalize_text
from kg_agent.pairwise import ground_nodes, pair_inputs, relation_rejection
from kg_agent.source import SourceRouter
from kg_agent.relation_classifier import materialize_pair
from train_relation_classifier import predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--entity-predictions", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--rules", action="store_true")
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    records = json.loads(Path(args.gold).read_text())
    cached = [
        json.loads(s)
        for s in Path(args.entity_predictions).read_text().splitlines()
        if s.strip()
    ]
    if len(records) != len(cached):
        raise ValueError("Prediction/gold alignment mismatch")
    selection = json.loads(Path(args.selection).read_text())
    labels = selection["labels"]
    threshold = selection["threshold"]
    pairs = []
    graphs = []
    docs = []
    ids = []
    ceilings = Counter()
    started = time.time()
    for i, (record, cached_row) in enumerate(zip(records, cached)):
        doc = SourceRouter().route(_message(record, "user"))
        docs.append(doc)
        gold = _graph(_message(record, "assistant"))
        cached_graph = _graph(cached_row)
        if cached_graph is None:
            raise ValueError(f"Invalid cached entity graph at {i}")
        if doc.source_type == "structured_icd":
            graphs.append(cached_graph)
            continue
        nodes = gold["entities"] if args.oracle else cached_graph["entities"]
        entities, mapping, missing = ground_nodes(nodes, doc)
        # Keep the cached entity output unchanged: relation-only comparison.
        graphs.append({"entities": json.loads(json.dumps(nodes)), "relations": []})
        reverse = {new: old for old, new in mapping.items()}
        per = pair_inputs(entities, doc)
        goldkeys = _relation_set(gold)
        byid = {e.id: (e.label, normalize_text(e.name)) for e in entities}
        keys = {
            (byid[pair.source], label, byid[pair.target])
            for pair in per
            for label in pair.allowed
        }
        ceilings["gold_triples"] += len(goldkeys)
        ceilings["candidate_covered_triples"] += len(goldkeys & keys)
        ceilings["candidate_pairs"] += len(per)
        ceilings["ungrounded_cached_nodes"] += len(missing)
        for pair in per:
            row = {
                **asdict(pair),
                "prompt": pair.prompt(),
                "labels": [],
                "id": f"{i}:{pair.source}:{pair.target}",
            }
            pairs.append(row)
            ids.append((i, reverse[pair.source], reverse[pair.target]))
    tokenizer = AutoTokenizer.from_pretrained(
        selection["checkpoint"], local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, selection["checkpoint"]).to("cuda").eval()
    probs, excluded = predict(
        model, tokenizer, pairs, labels, selection["max_length"], 8
    )
    rejected = Counter()
    raw_records = []
    for pair, ps, (sample, source, target) in zip(pairs, probs, ids):
        for edge in materialize_pair(
            pair,
            ps,
            labels,
            threshold,
            docs[sample],
            source,
            target,
            label_thresholds=selection.get("label_thresholds"),
        ):
            reason = (
                relation_rejection(edge, graphs[sample]["entities"])
                if args.rules
                else None
            )
            if reason:
                rejected[reason] += 1
            else:
                graphs[sample]["relations"].append(edge)
        raw_records.append(
            {
                "sample_index": sample,
                "source": source,
                "target": target,
                "labels": labels,
                "probabilities": ps,
                "span_ids": pair["span_ids"],
            }
        )
    predictions = [
        {
            "output": graph,
            "trace": {
                "mode": "oracle_entities" if args.oracle else "cached_v4_entities",
                "relation_mode": "Qwen3_LoRA_multilabel_sequence_classifier",
                "threshold": threshold,
            },
        }
        for graph in graphs
    ]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
    )
    (out / "probabilities.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw_records)
    )
    metrics = evaluate_records(records, predictions)
    metrics["run"] = {
        "selection": selection,
        "oracle": args.oracle,
        "rules": args.rules,
        "rejected": dict(rejected),
        "overlength_pairs": excluded,
        "candidate_ceiling": dict(ceilings),
        "seconds": time.time() - started,
        "evidence_policy": "Deterministic retrieval span covering the selected local context; provenance is exact source text, semantic entailment is not separately verified.",
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: {
                    metric: metrics[k][metric]
                    for metric in ["samples", "relation", "evidence_compatible"]
                }
                for k in ["overall", "free_text", "structured_icd"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

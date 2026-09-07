"""Strict failure accounting and stratified graph/evidence metrics."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from evaluate import (
    _graph,
    _entity_set,
    _relation_set,
    _records_from_predictions,
    _load_predictions,
)
from build_stage_data import _message
from kg_agent.normalize import normalize_text
from kg_agent.pairwise import relation_rejection
from kg_agent.source import SourceRouter
from pair_metrics import counts_metric


EMPTY = {"entities": [], "relations": []}


def evidence_by_triple(graph):
    result = defaultdict(set)
    for edge in graph["relations"]:
        if not isinstance(edge, dict):
            continue
        key = next(
            iter(_relation_set({"entities": graph["entities"], "relations": [edge]})),
            None,
        )
        evidence = edge.get("evidence", "")
        for value in evidence if isinstance(evidence, list) else [evidence]:
            if value:
                result[key].add(normalize_text(str(value)))
    return result


def summarize_counts(rows):
    totals = {
        name: Counter()
        for name in ["entity", "relation", "evidence_exact", "evidence_compatible"]
    }
    types = defaultdict(Counter)
    for r in rows:
        for key in totals:
            totals[key].update({k: r[key][k] for k in ["tp", "fp", "fn"]})
        for label, c in r["per_relation"].items():
            types[label].update(c)
    report = {
        "samples": len(rows),
        "invalid_predictions": sum(r["invalid_prediction"] for r in rows),
        "json_closure_rate": (
            1 - sum(r["invalid_prediction"] for r in rows) / len(rows) if rows else 0
        ),
        "relation_stage_error_samples": sum(
            r["relation_stage_errors"] > 0 for r in rows
        ),
        "relation_stage_errors": sum(r["relation_stage_errors"] for r in rows),
    }
    for key, c in totals.items():
        report[key] = counts_metric(c["tp"], c["fp"], c["fn"])
    report["per_relation"] = {
        label: counts_metric(c["tp"], c["fp"], c["fn"])
        for label, c in sorted(types.items())
    }
    active = [c for c in report["per_relation"].values() if c["tp"] + c["fn"] > 0]
    report["macro_relation_f1"] = (
        sum(c["f1"] for c in active) / len(active) if active else 0
    )
    return report


def evaluate_records(gold_records, predictions):
    if len(gold_records) != len(predictions):
        raise ValueError(
            f"Expected {len(gold_records)} predictions, received {len(predictions)}; alignment cannot be guessed"
        )
    samples = []
    for i, (record, raw) in enumerate(zip(gold_records, predictions)):
        gold = _graph(_message(record, "assistant"))
        if gold is None:
            raise ValueError(f"Invalid gold graph at {i}")
        predicted = _graph(raw)
        invalid = predicted is None
        predicted = predicted or EMPTY
        trace = raw.get("trace", {}) if isinstance(raw, dict) else {}
        relation_errors = sum(
            len(chunk.get("relation_errors", [])) for chunk in trace.get("chunks", [])
        )
        e, g = _entity_set(predicted), _entity_set(gold)
        p, t = _relation_set(predicted), _relation_set(gold)
        pe, ge = evidence_by_triple(predicted), evidence_by_triple(gold)
        exact = sum(bool(pe[k] & ge[k]) for k in p & t)
        compatible = sum(
            any(a in b or b in a for a in pe[k] for b in ge[k]) for k in p & t
        )
        types = {
            label: {
                "tp": sum(k[1] == label for k in p & t),
                "fp": sum(k[1] == label for k in p - t),
                "fn": sum(k[1] == label for k in t - p),
            }
            for label in {k[1] for k in p | t}
        }
        samples.append(
            {
                "sample_index": i,
                "source_type": SourceRouter()
                .route(_message(record, "user"))
                .source_type,
                "invalid_prediction": invalid,
                "relation_stage_errors": relation_errors,
                "entity": counts_metric(len(e & g), len(e - g), len(g - e)),
                "relation": counts_metric(len(p & t), len(p - t), len(t - p)),
                "evidence_exact": counts_metric(exact, len(p) - exact, len(t) - exact),
                "evidence_compatible": counts_metric(
                    compatible, len(p) - compatible, len(t) - compatible
                ),
                "per_relation": types,
            }
        )
    return {
        "overall": summarize_counts(samples),
        "free_text": summarize_counts(
            [r for r in samples if r["source_type"] == "free_text"]
        ),
        "structured_icd": summarize_counts(
            [r for r in samples if r["source_type"] == "structured_icd"]
        ),
        "samples": samples,
        "definitions": {
            "relation": "Unique (source label/name, relation, target label/name), excluding evidence",
            "evidence_exact": "Same triple AND normalized evidence exact match",
            "evidence_compatible": "Same triple AND one nonempty normalized evidence contains the other; a lenient span metric, NOT semantic entailment",
            "empty_set": "Precision/recall/F1 are zero when undefined; invalid predictions contribute all gold false negatives",
            "macro_relation_f1": "Mean over gold-supported relation types; zero-support predicted types remain in micro FP",
        },
    }


def filter_records(predictions):
    output = []
    removed = []
    for i, raw in enumerate(predictions):
        record = json.loads(json.dumps(raw))
        graph = _graph(record)
        if graph is None:
            output.append(record)
            continue
        accepted = []
        for edge in graph["relations"]:
            reason = relation_rejection(edge, graph["entities"])
            if reason:
                removed.append({"sample_index": i, "reason": reason, "relation": edge})
            else:
                accepted.append(edge)
        graph["relations"] = accepted
        output.append({"output": graph, "trace": {"filter": "narrow_evidence_rules"}})
    return output, removed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--filter-output")
    args = p.parse_args()
    gold = json.loads(Path(args.gold).read_text())
    pred = _records_from_predictions(_load_predictions(args.predictions))
    result = evaluate_records(gold, pred)
    if args.filter_output:
        filtered, removed = filter_records(pred)
        result["filtered"] = evaluate_records(gold, filtered)
        result["filter_reasons"] = dict(Counter(r["reason"] for r in removed))
        Path(args.filter_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.filter_output).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filtered)
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: {
                    m: result[k][m]
                    for m in ["samples", "relation", "relation_stage_errors"]
                }
                for k in ["overall", "free_text", "structured_icd"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

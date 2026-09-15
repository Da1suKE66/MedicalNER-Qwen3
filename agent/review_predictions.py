"""Decompose graph errors and export private, fully inspectable examples."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from build_stage_data import _message
from build_pair_data import source_groups
from evaluate import _graph, _entity_set, _relation_set
from evaluate_graphs import evaluate_records, filter_records
from kg_agent.normalize import normalize_text
from kg_agent.pairwise import ground_nodes, pair_inputs
from kg_agent.source import SourceRouter
from pair_metrics import counts_metric


def named(key):
    source, relation, target = key
    return {
        "source": {"label": source[0], "name": source[1]},
        "relation": relation,
        "target": {"label": target[0], "name": target[1]},
    }


def fp_reason(key, gold_keys, gold_entities):
    source, relation, target = key
    if source not in gold_entities or target not in gold_entities:
        return "endpoint_concept_not_in_silver"
    if any(s == source and t == target for s, _, t in gold_keys):
        return "relation_label_mismatch"
    if any(s == source and r == relation for s, r, _ in gold_keys):
        return "target_binding_mismatch"
    if (target, relation, source) in gold_keys:
        return "reversed_endpoints"
    return "pair_not_annotated_in_silver"


def paired_bootstrap(gold, before, after, seed=42, draws=2000):
    groups = source_groups(gold)
    by_group = defaultdict(list)
    for i, row in enumerate(before["samples"]):
        if row["source_type"] == "free_text":
            by_group[groups[i]].append(i)
    keys = sorted(by_group)
    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        chosen = [
            i for group in rng.choices(keys, k=len(keys)) for i in by_group[group]
        ]
        values = []
        for report in [before, after]:
            counts = [
                sum(report["samples"][i]["relation"][k] for i in chosen)
                for k in ["tp", "fp", "fn"]
            ]
            values.append(counts_metric(*counts)["f1"])
        deltas.append(values[1] - values[0])
    deltas.sort()
    return {
        "seed": seed,
        "draws": draws,
        "groups_with_free_text": len(keys),
        "free_text_f1_delta_95_percentile_interval": [
            deltas[int(0.025 * draws)],
            deltas[int(0.975 * draws) - 1],
        ],
        "limitation": "Paired source-group bootstrap describes sample variability, not protection against historical test-set reuse or silver-label error.",
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ["gold", "baseline", "predictions", "selection", "output-dir"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    gold = json.loads(Path(args.gold).read_text())
    read = lambda path: [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    baseline = read(args.baseline)
    predictions = read(args.predictions)
    labels = set(json.loads(Path(args.selection).read_text())["labels"])
    before = evaluate_records(gold, baseline)
    after = evaluate_records(gold, predictions)
    filtered, removed = filter_records(predictions)
    filtered_metrics = evaluate_records(gold, filtered)
    failures = []
    counts = Counter()
    for i, (record, raw) in enumerate(zip(gold, predictions)):
        doc = SourceRouter().route(_message(record, "user"))
        if doc.source_type == "structured_icd":
            continue
        target = _graph(_message(record, "assistant"))
        predicted = _graph(raw) or {"entities": [], "relations": []}
        p, t = _relation_set(predicted), _relation_set(target)
        pe, te = _entity_set(predicted), _entity_set(target)
        grounded, _, _ = ground_nodes(predicted["entities"], doc)
        by_id = {e.id: (e.label, normalize_text(e.name)) for e in grounded}
        candidates = {
            (by_id[pair.source], rel, by_id[pair.target])
            for pair in pair_inputs(grounded, doc)
            for rel in pair.allowed
        }
        false_positive = []
        false_negative = []
        for key in sorted(p - t):
            reason = fp_reason(key, t, te)
            counts["FP:" + reason] += 1
            false_positive.append({**named(key), "reason": reason})
        for key in sorted(t - p):
            if key[0] not in pe or key[2] not in pe:
                reason = "entity_concept_missing"
            elif key not in candidates:
                reason = "no_local_grounded_candidate"
            elif key[1] not in labels:
                reason = "unseen_relation_label"
            else:
                reason = "classifier_decision"
            counts["FN:" + reason] += 1
            false_negative.append({**named(key), "reason": reason})
        failures.append(
            {
                "sample_index": i,
                "relation_metrics": after["samples"][i]["relation"],
                "tp_gain": after["samples"][i]["relation"]["tp"]
                - before["samples"][i]["relation"]["tp"],
                "false_positives": false_positive,
                "false_negatives": false_negative,
            }
        )
    improvements = sorted(
        failures,
        key=lambda r: (r["tp_gain"], -r["relation_metrics"]["fp"]),
        reverse=True,
    )[:2]
    problems = sorted(
        failures,
        key=lambda r: (r["relation_metrics"]["fp"], r["relation_metrics"]["fn"]),
        reverse=True,
    )[:2]
    selected = list(
        dict.fromkeys([2, *[r["sample_index"] for r in improvements + problems]])
    )
    examples = [
        {
            "sample_index": i,
            "source_text": SourceRouter().route(_message(gold[i], "user")).raw,
            "silver_graph": _graph(_message(gold[i], "assistant")),
            "baseline_graph": _graph(baseline[i]),
            "new_graph": _graph(predictions[i]),
            "errors": next(r for r in failures if r["sample_index"] == i),
        }
        for i in selected
    ]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "error_review.json").write_text(
        json.dumps(
            {"counts": dict(counts), "samples": failures}, ensure_ascii=False, indent=2
        )
        + "\n"
    )
    (out / "representative_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "predictions_rules.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in filtered)
    )
    (out / "rules_metrics.json").write_text(
        json.dumps(filtered_metrics, indent=2) + "\n"
    )
    public = {
        "error_counts": dict(counts),
        "selected_sample_indices": selected,
        "rules_removed": len(removed),
        "rules_removed_tp": after["free_text"]["relation"]["tp"]
        - filtered_metrics["free_text"]["relation"]["tp"],
        "rules_removed_fp": after["free_text"]["relation"]["fp"]
        - filtered_metrics["free_text"]["relation"]["fp"],
        "bootstrap": paired_bootstrap(gold, before, after),
        "example_counts": [
            {
                "sample_index": i,
                "old": before["samples"][i]["relation"],
                "new": after["samples"][i]["relation"],
            }
            for i in selected
        ],
    }
    (out / "review_summary.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()

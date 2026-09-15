"""Apply a tuning-fitted policy to cached scores without rerunning a model."""

import argparse
import json
from pathlib import Path

from build_stage_data import _message
from evaluate import _graph
from evaluate_graphs import evaluate_records
from kg_agent.contracts import allowed_relations
from kg_agent.relation_classifier import materialize_pair
from kg_agent.source import SourceRouter


def main():
    p = argparse.ArgumentParser()
    for name in ["gold", "predictions", "probabilities", "selection", "output-dir"]:
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    read = lambda p: [
        json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()
    ]
    gold = json.loads(Path(args.gold).read_text())
    predictions = read(args.predictions)
    scores = read(args.probabilities)
    selection = json.loads(Path(args.selection).read_text())
    if len(gold) != len(predictions):
        raise ValueError("Graph alignment mismatch")
    docs = [SourceRouter().route(_message(r, "user")) for r in gold]
    graphs = [_graph(r) for r in predictions]
    for doc, graph in zip(docs, graphs):
        if graph is None:
            raise ValueError("Invalid cached graph")
        if doc.source_type == "free_text":
            graph["relations"] = []
    for row in scores:
        index = row["sample_index"]
        graph = graphs[index]
        by_id = {n["id"]: n for n in graph["entities"]}
        source, target = row["source"], row["target"]
        if row["labels"] != selection["labels"]:
            raise ValueError("Classifier label order mismatch")
        pair = {
            "span_ids": row["span_ids"],
            "allowed": allowed_relations(
                by_id[source]["label"], by_id[target]["label"]
            ),
        }
        graph["relations"].extend(
            materialize_pair(
                pair,
                row["probabilities"],
                row["labels"],
                selection["threshold"],
                docs[index],
                source,
                target,
                label_thresholds=selection.get("label_thresholds"),
            )
        )
    results = [
        {
            "output": graph,
            "trace": {
                "relation_mode": "cached_scores_tuning_only_calibration",
                "threshold": selection["threshold"],
                "label_thresholds": selection.get("label_thresholds", {}),
            },
        }
        for graph in graphs
    ]
    metrics = evaluate_records(gold, results)
    metrics["calibration"] = selection.get("calibration", {})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: metrics[k]["relation"]
                for k in ["overall", "free_text", "structured_icd"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

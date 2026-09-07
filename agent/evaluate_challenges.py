"""Synthetic contract probes, never used for training or threshold selection."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from kg_agent.pairwise import ground_nodes, pair_inputs
from kg_agent.relation_classifier import QwenRelationClassifier
from kg_agent.source import SourceRouter
from pair_metrics import score_pairs
from train_relation_classifier import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "tests/fixtures/relation_challenges.json"),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = json.loads(Path(args.input).read_text())
    backend = QwenRelationClassifier(args.base_model, args.selection)
    rows = []
    for case in cases:
        doc = SourceRouter().route(case["text"])
        nodes = [{"id": "source", **case["source"]}, {"id": "target", **case["target"]}]
        entities, mapping, missing = ground_nodes(nodes, doc)
        assert not missing
        pair = next(
            p
            for p in pair_inputs(entities, doc)
            if p.source == mapping["source"] and p.target == mapping["target"]
        )
        rows.append(
            {
                **asdict(pair),
                "id": case["id"],
                "prompt": pair.prompt(),
                "labels": case["labels"],
            }
        )
    backend._load()
    selection = backend.selection
    probabilities, excluded = predict(
        backend.model,
        backend.tokenizer,
        rows,
        selection["labels"],
        selection["max_length"],
        8,
    )
    metrics = score_pairs(
        rows, probabilities, selection["labels"], selection["threshold"]
    )
    result = []
    for case, row, probs in zip(cases, rows, probabilities):
        predicted = [
            label
            for label, p in zip(selection["labels"], probs)
            if label in row["allowed"] and p >= selection["threshold"]
        ]
        result.append(
            {
                **case,
                "predicted": predicted,
                "exact_match": set(predicted) == set(case["labels"]),
                "scores": {
                    l: p
                    for l, p in zip(selection["labels"], probs)
                    if l in row["allowed"]
                },
            }
        )
    output = {
        "scope": "Eight synthetic diagnostic cases, not a representative test distribution; not used for model selection.",
        "threshold": selection["threshold"],
        "metrics": metrics,
        "overlength": excluded,
        "cases": result,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exact_cases": sum(r["exact_match"] for r in result),
                "total_cases": len(result),
                "cases": [
                    {
                        "id": r["id"],
                        "expected": r["labels"],
                        "predicted": r["predicted"],
                    }
                    for r in result
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

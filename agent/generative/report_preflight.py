"""Aggregate-only reproducible coverage report. NO model accuracy is inferred."""

import argparse
from collections import Counter
import json
from pathlib import Path

from build_stage_data import _message
from evaluate import _graph, _relation_set
from kg_agent.source import SourceRouter
from .protocol import ProtocolConfig, build_batches
from .evaluate import candidate_ceiling


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--entity-predictions", required=True)
    p.add_argument("--preferences-manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    gold = json.loads(Path(args.gold).read_text())
    cached = [
        json.loads(line)
        for line in Path(args.entity_predictions).read_text().splitlines()
        if line.strip()
    ]
    if len(gold) != len(cached):
        raise ValueError("Cache misalignment")
    coverage = {}
    for oracle in (False, True):
        totals = Counter()
        for i, record in enumerate(gold):
            doc = SourceRouter().route(_message(record, "user"))
            if doc.source_type != "free_text":
                continue
            graph = _graph(_message(record, "assistant"))
            supplied = graph if oracle else _graph(cached[i])
            batches, _, audit = build_batches(
                doc.raw, supplied["entities"], ProtocolConfig()
            )
            totals["gold_triples"] += len(_relation_set(graph))
            totals["covered_triples"] += len(
                _relation_set(graph) & candidate_ceiling(supplied, batches)
            )
            totals["candidate_pairs"] += audit["candidate_pairs"]
            totals["ungrounded_entities"] += len(audit["ungrounded_ids"])
        coverage["oracle" if oracle else "fixed_predicted_entities"] = dict(totals)
    data = json.loads(Path(args.data_dir, "preflight.json").read_text())
    report = {
        "status": "CPU_data_and_contract_preflight_only",
        "new_model_training_completed": False,
        "new_model_accuracy_evaluated": False,
        "coverage": coverage,
        "datasets": {
            name: {
                k: value[k]
                for k in [
                    "config",
                    "source_train_sha256",
                    "dataset_sha256",
                    "stats",
                    "label_counts",
                ]
            }
            for name, value in data.items()
        },
        "preferences": json.loads(Path(args.preferences_manifest).read_text()),
        "important": "Coverage upper bounds and successful unit tests are not model accuracy gains",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(coverage))


if __name__ == "__main__":
    main()

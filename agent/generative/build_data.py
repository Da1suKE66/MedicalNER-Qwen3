"""Build private generation supervision and a public aggregate preflight manifest.

python -m generative.build_data --train ... --output-dir ...
No GPU and no regression labels are required to construct training/tuning sets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from build_pair_data import source_groups
from build_stage_data import _message
from evaluate import _graph
from kg_agent.source import SourceRouter
from .protocol import build_batches, dumps, load_variant, supervision


def convert(record, index, group, config, *, sampled, seed=42):
    doc = SourceRouter().route(_message(record, "user"))
    if doc.source_type == "structured_icd":
        return [], {"icd_records": 1}
    # Keep the routed ORIGINAL source: build_batches may return a repaired doc.
    # Rebatching that repaired doc would falsely label repaired evidence original.
    source_text = doc.raw
    graph = _graph(_message(record, "assistant"))
    if graph is None:
        raise ValueError(f"Invalid teacher graph {index}")
    batches, doc, audit = build_batches(
        source_text, graph["entities"], config, sample_id=index
    )
    positive_pairs = {(str(r["source"]), str(r["target"])) for r in graph["relations"]}
    stats = Counter(
        records=1,
        gold_relations=len(graph["relations"]),
        gold_entities=len(graph["entities"]),
        ungrounded_entities=len(audit["ungrounded_ids"]),
        all_candidate_pairs=audit["candidate_pairs"],
        repaired_occurrences=len(audit["repairs"]),
        repaired_records=bool(audit["repairs"]),
    )
    if sampled:
        all_pairs = {(p["source"], p["target"]) for b in batches for p in b["pairs"]}
        negatives = sorted(all_pairs - positive_pairs)
        rng = random.Random(seed + index)
        rng.shuffle(negatives)
        # Sampling changes training prevalence, not inference candidate coverage.
        # Never call unannotated silver pairs verified true negatives.
        retained = (all_pairs & positive_pairs) | set(
            negatives[: max(8, config.negative_ratio * len(positive_pairs & all_pairs))]
        )
        batches, doc, _ = build_batches(
            source_text, graph["entities"], config, sample_id=index, selected_pairs=retained
        )
    rows = []
    for batch in batches:
        target, count = supervision(batch, graph, doc, config)
        stats.update({k: v for k, v in count.items() if isinstance(v, (int, float))})
        if target is None:
            stats["unknown_pairs"] += len(batch["pairs"])
            continue
        rows.append(
            {
                **batch,
                "group_id": group,
                "completion": dumps(target),
                "target": target,
                "sample_weight": 1.0,
                "annotation_source": "teacher_silver_not_human_gold",
            }
        )
    stats["rows"] = len(rows)
    stats["retained_pairs"] = sum(len(b["pairs"]) for b in rows)
    return rows, dict(stats)


def apply_rebalancing(rows):
    counts = Counter(
        e["relation"]
        for row in rows
        for d in row["target"]["decisions"]
        for e in d["relations"]
    )
    for row in rows:
        decisions = row["target"]["decisions"]
        labels = [e["relation"] for d in decisions for e in d["relations"]]
        multi = any(len(d["relations"]) > 1 for d in decisions)
        rare = any(counts[label] < 50 for label in labels)
        row["sample_weight"] = 4.0 if multi else 2.0 if rare else 1.0
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--matrix", default=str(Path(__file__).with_name("experiments.json"))
    )
    p.add_argument("--variants", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assume-closed-world-silver", action="store_true",
                   help="Historical reproduction only: explicitly treat every unannotated silver pair as negative")
    args = p.parse_args()
    if not args.assume_closed_world_silver:
        p.error("This legacy builder labels unannotated silver pairs as no relation. "
                "Use data_repair.release_v2 for UNKNOWN-aware supervision; "
                "only historical reproduction may opt in with --assume-closed-world-silver.")
    records = json.loads(Path(args.train).read_text())
    groups = source_groups(records)
    shuffled = sorted(set(groups))
    random.Random(args.seed).shuffle(shuffled)
    tune = set(shuffled[: round(len(shuffled) * 0.1)])
    destination = Path(args.output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Choose a new/empty data version; never overwrite existing training artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "tune_records.json").write_text(
        json.dumps(
            [r for r, g in zip(records, groups) if g in tune], ensure_ascii=False
        )
        + "\n"
    )
    spec = json.loads(Path(args.matrix).read_text())
    aggregate = {}
    for name in args.variants or list(spec["variants"]):
        config = load_variant(args.matrix, name)
        root = Path(args.output_dir) / name
        root.mkdir(parents=True, exist_ok=True)
        subsets = {"train": [], "tune": []}
        stats = {s: Counter() for s in subsets}
        for i, record in enumerate(records):
            split = "tune" if groups[i] in tune else "train"
            rows, counts = convert(
                record, i, groups[i], config, sampled=split == "train", seed=args.seed
            )
            for row in rows:
                row["split"] = split
            subsets[split].extend(rows)
            stats[split].update(counts)
        # Policy fitted from training only, not tuning counts.
        label_counts = (
            apply_rebalancing(subsets["train"])
            if config.rebalance
            else Counter(
                e["relation"]
                for r in subsets["train"]
                for d in r["target"]["decisions"]
                for e in d["relations"]
            )
        )
        hashes = {}
        for split, rows in subsets.items():
            path = root / f"{split}.jsonl"
            path.write_text("".join(dumps(row) + "\n" for row in rows))
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "variant": name,
            "config": asdict(config),
            "seed": args.seed,
            "supervision_policy": "legacy_closed_world_silver_explicit_opt_in",
            "unannotated_pairs_are_verified_negatives": False,
            "source_train_sha256": hashlib.sha256(
                Path(args.train).read_bytes()
            ).hexdigest(),
            "dataset_sha256": hashes,
            "train_groups": sorted(set(groups) - tune),
            "tune_groups": sorted(tune),
            "stats": {s: dict(v) for s, v in stats.items()},
            "label_counts": dict(label_counts),
            "effective_weighted_train_rows": sum(
                r["sample_weight"] for r in subsets["train"]
            ),
            "caveats": [
                "Silver labels are not adjudicated clinical truth",
                "Unknown batches excluded from supervision, never excluded from full graph evaluation",
                "Tuning pair targets are diagnostics, not a substitute for all-document graph generation",
                "Negative sampling is train-only; final evaluation uses ALL input-only candidates",
            ],
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        aggregate[name] = manifest
        print(dumps({"variant": name, "stats": manifest["stats"]}), flush=True)
    Path(args.output_dir, "preflight.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

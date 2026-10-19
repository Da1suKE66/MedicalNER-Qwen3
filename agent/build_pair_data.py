"""Create named-pair supervision with an input-only candidate builder."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from build_stage_data import _message
from evaluate import _graph
from kg_agent.normalize import normalize_text
from kg_agent.pairwise import evidence_spans, ground_nodes, pair_inputs
from kg_agent.source import SourceRouter


def source_groups(records):
    """The exported source order starts each group with its structured ICD row."""
    group = -1
    groups = []
    for r in records:
        doc = SourceRouter().route(_message(r, "user"))
        if doc.source_type == "structured_icd":
            group += 1
        if group < 0:
            raise ValueError("Missing source-group provenance: first record is not ICD")
        groups.append(group)
    return groups


def convert(record, index, group, *, sampled=False, seed=42, negative_ratio=2):
    doc = SourceRouter().route(_message(record, "user"))
    if doc.source_type == "structured_icd":
        return [], {"structured_icd": 1}
    gold = _graph(_message(record, "assistant"))
    if gold is None:
        raise ValueError(
            f"Invalid teacher graph at sample {index}; do not create negative supervision from a parse failure"
        )
    entities, mapping, missing = ground_nodes(gold["entities"], doc)
    pairs = pair_inputs(entities, doc)
    specs = defaultdict(list)
    for rel in gold["relations"]:
        if str(rel["source"]) in mapping and str(rel["target"]) in mapping:
            specs[(mapping[str(rel["source"])], mapping[str(rel["target"])])].append(
                rel
            )
    stats = Counter(
        records=1,
        gold_entities=len(gold["entities"]),
        ungrounded_entities=len(missing),
        gold_relations=len(gold["relations"]),
        candidate_pairs=len(pairs),
    )
    rows = []
    for pair in pairs:
        annotations = specs.get((pair.source, pair.target), [])
        labels = set()
        ev_ids = set()
        unresolved = False
        for rel in annotations:
            evidence = evidence_spans(
                rel.get("evidence")
                or rel.get("evidence_sentence")
                or rel.get("evidence_span_id"),
                doc,
            )
            overlap = evidence & set(pair.span_ids)
            if rel["relation"] in pair.allowed and overlap:
                labels.add(rel["relation"])
                ev_ids.update(overlap)
            else:
                unresolved = True
        if unresolved:
            stats["excluded_unknown_pairs"] += 1
            continue
        stats["covered_positive_triples"] += len(labels)
        rows.append(
            {
                "id": f"{index}:{pair.source}:{pair.target}",
                "sample_id": index,
                "group_id": group,
                "prompt": pair.prompt(),
                "labels": sorted(labels),
                "evidence_span_ids": sorted(ev_ids),
                **asdict(pair),
            }
        )
    if sampled:
        positives = [r for r in rows if r["labels"]]
        negatives = [r for r in rows if not r["labels"]]
        rng = random.Random(seed + index)
        rng.shuffle(negatives)
        signatures = {(r["source"], r["target_label"]) for r in positives} | {
            (r["target"], r["source_label"]) for r in positives
        }
        negatives.sort(key=lambda r: (r["source"], r["target_label"]) not in signatures)
        negatives = negatives[: max(8, len(positives) * negative_ratio)]
        # Order is independent of labels; pairs are classified independently.
        rows = sorted(
            positives + negatives,
            key=lambda r: (
                r["source_name"].casefold(),
                r["target_name"].casefold(),
                r["source_label"],
            ),
        )
    stats["output_pairs"] = len(rows)
    stats["none_pairs"] = sum(not r["labels"] for r in rows)
    return rows, dict(stats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = json.loads(Path(args.train).read_text())
    dev = json.loads(Path(args.dev).read_text())
    groups = source_groups(train)
    all_groups = sorted(set(groups))
    shuffled = all_groups[:]
    random.Random(args.seed).shuffle(shuffled)
    tune_groups = set(shuffled[: max(1, round(0.1 * len(shuffled)))])
    subsets = {"train": [], "tune": [], "dev_oracle": []}
    stats = {s: Counter() for s in subsets}
    for i, r in enumerate(train):
        split = "tune" if groups[i] in tune_groups else "train"
        rows, c = convert(r, i, groups[i], sampled=split == "train", seed=args.seed)
        subsets[split].extend(rows)
        stats[split].update(c)
    for i, r in enumerate(dev):
        rows, c = convert(r, i, i)
        subsets["dev_oracle"].extend(rows)
        stats["dev_oracle"].update(c)
    labels = sorted({label for r in subsets["train"] for label in r["labels"]})
    for split, rows in subsets.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    manifest = {
        "seed": args.seed,
        "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "dev_sha256": hashlib.sha256(Path(args.dev).read_bytes()).hexdigest(),
        "source_group_method": "ICD row starts a source group in the original grouped export; corroborated by 606 groups in upstream manifest",
        "train_groups": sorted(set(groups) - tune_groups),
        "tune_groups": sorted(tune_groups),
        "labels": labels,
        "stats": {s: dict(c) for s, c in stats.items()},
        "train_label_counts": dict(
            Counter(l for r in subsets["train"] for l in r["labels"])
        ),
        "historical_dev_role": "Previously inspected regression set, not a blind test",
        "unknown_pair_policy": "Any unsupported evidence annotation makes the pair unknown; never turn such pairs into NONE",
        "context_policy": {"max_distance": 2, "max_context_chars": 2200},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

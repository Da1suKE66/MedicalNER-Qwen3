"""Auditable DPO pairs: silver omissions + explicitly synthetic semantic errors.

No regression predictions are imported. Unknown real-world FP are not automatically
declared negative. Synthetic statements are counterfactual exercises, not medical facts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from .protocol import (
    ProtocolConfig,
    build_batches,
    digest,
    dumps,
    parse_decisions,
    reward_components,
    supervision,
)
from .train import read_rows


def omission_preferences(rows):
    for row in rows:
        chosen = row["target"]
        positives = [
            (i, j)
            for i, d in enumerate(chosen["decisions"])
            for j, _ in enumerate(d["relations"])
        ]
        if not positives:
            continue
        # One deterministic deletion per batch; preserve output syntax/order.
        i, j = positives[int(digest(row["id"]), 16) % len(positives)]
        rejected = copy.deepcopy(chosen)
        rejected["decisions"][i]["relations"].pop(j)
        yield {
            "id": "omission:" + row["id"],
            "group_id": row["group_id"],
            "split": "train",
            "prompt": row["prompt"],
            "chosen": dumps(chosen),
            "rejected": dumps(rejected),
            "reason": "delete_one_silver_supported_relation",
            "provenance": "teacher_silver_completeness_preference_not_human_adjudicated",
            "hard_semantic": False,
        }


def synthetic_preferences(groups, per_group=4):
    for group in sorted(groups):
        for variant in range(per_group):
            key = digest(f"counterfactual:{group}:{variant}")[:8]
            symptom, a, b = f"Feature {key}", f"Condition {key}A", f"Condition {key}B"
            nodes = [
                {"id": "S", "name": symptom, "label": "Symptom"},
                {"id": "A", "name": a, "label": "Disease"},
                {"id": "B", "name": b, "label": "Disease"},
            ]
            kind = variant % 4
            label = "is_core_symptom_of"
            if kind == 0:
                text = f"{symptom} is a core symptom of {a}. {symptom} is explicitly not a symptom of {b}."
                edges = [("S", "A", label)]
                wrong = ("S", "B", label)
            elif kind == 1:
                text = f"{symptom} is explicitly not a symptom of {a}. {b} is mentioned only as a separate condition."
                edges = []
                wrong = ("S", "A", label)
            elif kind == 2:
                text = f"{a} is a subtype of {b}. {b} is the parent condition, not a subtype of {a}. {symptom} is an unrelated feature."
                edges = [("A", "B", "subtype_of")]
                wrong = ("B", "A", "subtype_of")
            else:
                text = f"{symptom} is a core symptom of {a} and independently supports diagnosis of {a}. {symptom} is not linked to {b}."
                edges = [("S", "A", label), ("S", "A", "supports_diagnosis_of")]
                wrong = ("S", "B", "supports_diagnosis_of")
            graph = {
                "entities": nodes,
                "relations": [
                    {"source": s, "target": t, "relation": r, "evidence": text}
                    for s, t, r in edges
                ],
            }
            config = ProtocolConfig(
                strict_evidence=True, pairs_per_batch=4, order="source"
            )
            batches, doc, _ = build_batches(
                text, nodes, config, sample_id=f"synthetic:{group}:{variant}"
            )
            batch = next(
                batch
                for batch in batches
                if any((p["source"], p["target"]) == wrong[:2] for p in batch["pairs"])
            )
            chosen, _ = supervision(batch, graph, doc, config)
            if chosen is None:
                raise ValueError("Broken synthetic oracle")
            rejected = copy.deepcopy(chosen)
            pair = next(
                p for p in batch["pairs"] if (p["source"], p["target"]) == wrong[:2]
            )
            decision = next(
                d for d in rejected["decisions"] if d["pair_id"] == pair["pair_id"]
            )
            decision["relations"].append(
                {
                    "relation": wrong[2],
                    "evidence_span_ids": [s["span_id"] for s in batch["spans"]],
                }
            )
            if parse_decisions(dumps(rejected), batch)[1]:
                raise ValueError(
                    "Semantic negative accidentally violates output syntax"
                )
            if (
                reward_components(dumps(chosen), batch, chosen)["reward"]
                <= reward_components(dumps(rejected), batch, chosen)["reward"]
            ):
                raise ValueError("Preference/reward direction mismatch")
            yield {
                "id": batch["id"],
                "group_id": group,
                "split": "train",
                "prompt": batch["prompt"],
                "chosen": dumps(chosen),
                "rejected": dumps(rejected),
                "reason": [
                    "explicit_wrong_disease",
                    "explicit_negation",
                    "reversed_subtype",
                    "multi_relation_wrong_target",
                ][kind],
                "provenance": "synthetic_counterfactual_explicit_text_not_real_medical_fact",
                "hard_semantic": True,
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--synthetic-groups", type=int, default=64)
    args = p.parse_args()
    rows = read_rows(args.data)
    if any(r.get("split") != "train" for r in rows):
        raise ValueError(
            "Require explicitly train-tagged rows; no tune/regression preferences"
        )
    groups = sorted({r["group_id"] for r in rows}, key=lambda g: digest(f"pref:{g}"))[
        : args.synthetic_groups
    ]
    real = list(omission_preferences(rows))
    synthetic = list(synthetic_preferences(groups))
    if not real or not synthetic:
        raise ValueError(
            "Both omission and semantic counterfactual strata are required"
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Do not let 3k omission-only pairs overwhelm 256 semantic negatives and
    # inadvertently teach "emit more relations" to an already hallucinating LM.
    real.sort(key=lambda r: digest(r["id"]))
    if len(real) < 2 * len(synthetic):
        raise ValueError(
            "Insufficient distinct silver pairs for a size-matched no-hard control"
        )
    balanced = real[: len(synthetic)] + synthetic
    no_hard = real[: len(balanced)]
    for name, subset in [
        ("candidate_pool", real + synthetic),
        ("full", balanced),
        ("no_hard_semantic", no_hard),
    ]:
        (out / f"{name}.jsonl").write_text("".join(dumps(r) + "\n" for r in subset))
    manifest = {
        "source_data_sha256": digest(Path(args.data).read_text()),
        "rows": len(real) + len(synthetic),
        "balanced_training_rows": len(balanced),
        "size_matched_no_hard_rows": len(no_hard),
        "balanced_counts": dict(Counter(r["reason"] for r in balanced)),
        "counts": dict(Counter(r["reason"] for r in real + synthetic)),
        "tune_or_regression_rows": 0,
        "synthetic_group_ids": groups,
        "limitations": [
            "Real preferences test completeness of existing silver targets, not verified medical truth",
            "Semantic error preferences are explicit synthetic counterfactuals; no claim of human review",
            "No on-policy hard negatives yet: collect SFT train-only rollouts and adjudicate before a separate iteration",
            "These preferences alone do not prove real-corpus hallucination reduction",
            "512 balanced seed preferences are an initial pilot, not the final on-policy preference corpus",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dumps(manifest))


if __name__ == "__main__":
    main()

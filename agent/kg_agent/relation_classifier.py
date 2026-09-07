"""Drop-in non-generative relation backend for the existing two-stage agent."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from .contracts import Relation
from .pairwise import ground_nodes, pair_inputs, relation_rejection


def materialize_pair(
    pair,
    probabilities,
    labels,
    threshold,
    document,
    source,
    target,
    *,
    label_thresholds=None,
):
    """Bind decisions and exact retrieved text deterministically; do not generate IDs.

    Evidence is the retrieved source range, not a learned entailment guarantee.
    This identical routine is used by the cached-entity evaluation and live agent.
    """
    spans = [document.span_map()[sid] for sid in pair["span_ids"]]
    evidence = document.raw[min(s.start for s in spans) : max(s.end for s in spans)]
    return [
        {"source": source, "target": target, "relation": label, "evidence": evidence}
        for label, probability in zip(labels, probabilities)
        if label in pair["allowed"]
        and probability >= (label_thresholds or {}).get(label, threshold)
    ]


class QwenRelationClassifier:
    """Loads lazily, so ICD-only runs do not allocate a second base model."""

    def __init__(self, base_model, selection_path, *, rules=False, batch_size=8):
        self.base_model = base_model
        self.selection_path = str(selection_path)
        self.selection = json.loads(Path(selection_path).read_text())
        self.rules = rules
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        checkpoint = self.selection["checkpoint"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(self.selection["labels"]),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False
        self.model = PeftModel.from_pretrained(model, checkpoint).to("cuda").eval()

    def classify(self, entities, document):
        from train_relation_classifier import predict

        nodes = [asdict(entity) for entity in entities]
        grounded, mapping, missing = ground_nodes(nodes, document)
        reverse = {new: old for old, new in mapping.items()}
        pairs = [
            {
                **asdict(pair),
                "id": f"{pair.source}:{pair.target}",
                "prompt": pair.prompt(),
                "labels": [],
            }
            for pair in pair_inputs(grounded, document)
        ]
        trace = {
            "checkpoint": self.selection["checkpoint"],
            "threshold": self.selection["threshold"],
            "label_thresholds": self.selection.get("label_thresholds", {}),
            "candidate_pairs": len(pairs),
            "ungrounded_entities": missing,
            "rules": self.rules,
            "evidence_policy": "Exact retrieved source range; entailment is not separately verified.",
        }
        if not pairs:
            return [], {**trace, "overlength_pairs": [], "accepted_relations": 0}
        self._load()
        probabilities, excluded = predict(
            self.model,
            self.tokenizer,
            pairs,
            self.selection["labels"],
            self.selection["max_length"],
            self.batch_size,
        )
        accepted = []
        rejected = Counter()
        for pair, probs in zip(pairs, probabilities):
            decisions = materialize_pair(
                pair,
                probs,
                self.selection["labels"],
                self.selection["threshold"],
                document,
                reverse[pair["source"]],
                reverse[pair["target"]],
                label_thresholds=self.selection.get("label_thresholds"),
            )
            for edge in decisions:
                reason = relation_rejection(edge, nodes) if self.rules else None
                if reason:
                    rejected[reason] += 1
                    continue
                accepted.append(
                    Relation(
                        source=edge["source"],
                        target=edge["target"],
                        relation=edge["relation"],
                        evidence_ids=pair["span_ids"],
                        evidence_text=[edge["evidence"]],
                    )
                )
        trace.update(
            overlength_pairs=excluded,
            accepted_relations=len(accepted),
            rejected=dict(rejected),
        )
        return accepted, trace

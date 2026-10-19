"""Behavioral regressions for failure accounting and relation hallucinations."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import _graph, _prf, _relation_set
from build_pair_data import convert, source_groups
from kg_agent.pairwise import (
    ground_nodes,
    pair_inputs,
    evidence_spans,
    relation_rejection,
)
from kg_agent.source import SourceRouter, SentenceSpanSplitter
from kg_agent.contracts import allowed_relations
from pair_metrics import score_pairs
from evaluate_graphs import evaluate_records
from kg_agent.contracts import Entity, EntityCandidate, Relation
from kg_agent.normalize import normalize_candidates
from kg_agent.pipeline import TwoStageKGAgent
from kg_agent.relation_classifier import materialize_pair


class ReliabilityTests(unittest.TestCase):
    def test_symptom_disease_type_guard(self):
        nodes = [
            {"id": "s", "label": "Symptom", "name": "fatigue"},
            {"id": "d", "label": "Symptom", "name": "Condition A"},
        ]
        self.assertEqual(
            relation_rejection(
                {
                    "source": "s",
                    "target": "d",
                    "relation": "is_core_symptom_of",
                    "evidence": "fatigue in Condition A",
                },
                nodes,
            ),
            "symptom_disease_type_mismatch",
        )

    def test_type_threshold_abstains_without_relabeling(self):
        labels = ["is_core_symptom_of", "supports_diagnosis_of"]
        doc = SourceRouter().route("Fatigue supports Condition A.")
        edges = materialize_pair(
            {"span_ids": ["SENT_001"], "allowed": labels},
            [0.9, 0.9],
            labels,
            0.5,
            doc,
            "S1",
            "D1",
            label_thresholds={"supports_diagnosis_of": 1.01},
        )
        self.assertEqual([e["relation"] for e in edges], ["is_core_symptom_of"])
        m = score_pairs(
            [{"labels": labels, "allowed": labels}],
            [[0.9, 0.9]],
            labels,
            0.5,
            label_thresholds={"supports_diagnosis_of": 1.01},
        )
        self.assertEqual((m["tp"], m["fn"]), (1, 1))

    def test_invalid_teacher_never_becomes_none_supervision(self):
        with self.assertRaises(ValueError):
            convert(
                {
                    "messages": [
                        {"role": "user", "content": "Condition A has fatigue."},
                        {"role": "assistant", "content": '{"entities": ['},
                    ]
                },
                0,
                0,
            )

    def test_decimals_abbreviations_and_offsets(self):
        text = "Score is 2.3 (e.g., measured in adults). Onset is early, i.e., before age 10. See Table 6.1."
        spans = SentenceSpanSplitter().split(text)
        self.assertEqual(len(spans), 3)
        self.assertTrue(all(s.text == text[s.start : s.end] for s in spans))
        self.assertIn("2.3", spans[0].text)
        self.assertIn("e.g.", spans[0].text)

    def test_unknown_json_is_empty_for_scoring(self):
        self.assertIsNone(_graph('{"entities":['))
        self.assertEqual(_prf(set(), {"gold"})["fn"], 1)
        self.assertEqual(_prf(set(), {"gold"})["f1"], 0)

    def test_missing_endpoint_is_false_positive(self):
        graph = {
            "entities": [],
            "relations": [{"source": "X", "target": "Y", "relation": "causes"}],
        }
        self.assertEqual(_prf(_relation_set(graph), set())["fp"], 1)

    def test_shared_input_and_id_permutation(self):
        doc = SourceRouter().route(
            "Condition A has fatigue. Condition B is another diagnosis. Fatigue is associated with Condition A."
        )
        nodes = [
            {"id": "arbitrary", "label": "Disease", "name": "Condition A"},
            {"id": "other", "label": "Disease", "name": "Condition B"},
            {"id": "s", "label": "Symptom", "name": "fatigue"},
        ]
        entities, _, _ = ground_nodes(nodes, doc)
        prompts = sorted(p.prompt() for p in pair_inputs(entities, doc))
        for i, n in enumerate(nodes):
            n["id"] = str(91 - i)
        permuted, _, _ = ground_nodes(list(reversed(nodes)), doc)
        self.assertEqual(
            prompts, sorted(p.prompt() for p in pair_inputs(permuted, doc))
        )
        self.assertGreater(
            len(
                next(e for e in entities if e.name.casefold() == "condition a").span_ids
            ),
            1,
        )

    def test_no_evidence_fallback(self):
        doc = SourceRouter().route("A heading. Condition A is present.")
        self.assertEqual(evidence_spans("This sentence is absent.", doc), set())

    def test_multilabel_supervision(self):
        text = "Fatigue is a core symptom of Condition A and supports diagnosis of Condition A."
        graph = {
            "entities": [
                {"id": "s", "label": "Symptom", "name": "Fatigue"},
                {"id": "d", "label": "Disease", "name": "Condition A"},
            ],
            "relations": [
                {"source": "s", "target": "d", "relation": r, "evidence": text}
                for r in ["is_core_symptom_of", "supports_diagnosis_of"]
            ],
        }
        record = {
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": json.dumps(graph)},
            ]
        }
        rows, _ = convert(record, 0, 0)
        self.assertEqual(
            next(r for r in rows if r["source_label"] == "Symptom")["labels"],
            ["is_core_symptom_of", "supports_diagnosis_of"],
        )

    def test_unaligned_evidence_is_not_none(self):
        graph = {
            "entities": [
                {"id": "s", "label": "Symptom", "name": "Fatigue"},
                {"id": "d", "label": "Disease", "name": "Condition A"},
            ],
            "relations": [
                {
                    "source": "s",
                    "target": "d",
                    "relation": "is_core_symptom_of",
                    "evidence": "Evidence not in the source.",
                }
            ],
        }
        rows, stats = convert(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Fatigue and Condition A are mentioned.",
                    },
                    {"role": "assistant", "content": json.dumps(graph)},
                ]
            },
            0,
            0,
        )
        self.assertTrue(all(r["source_label"] != "Symptom" for r in rows))
        self.assertEqual(stats["excluded_unknown_pairs"], 1)

    def test_target_module_not_invented(self):
        self.assertIn("contributes_to", allowed_relations("Etiology", "Disease"))
        self.assertNotIn("presents_with", allowed_relations("Disease", "Symptom"))

    def test_sibling_rule_preserves_parent_child(self):
        nodes = [
            {"id": "s1", "label": "Symptom", "name": "aggression"},
            {"id": "s2", "label": "Symptom", "name": "self-injury"},
            {"id": "p", "label": "Symptom", "name": "problem behaviours"},
        ]
        evidence = "problem behaviours such as aggression, self-injury are frequent."
        self.assertEqual(
            relation_rejection(
                {
                    "source": "s1",
                    "target": "s2",
                    "relation": "has_manifestation",
                    "evidence": evidence,
                },
                nodes,
            ),
            "enumerated_siblings",
        )
        self.assertIsNone(
            relation_rejection(
                {
                    "source": "p",
                    "target": "s1",
                    "relation": "has_manifestation",
                    "evidence": evidence,
                },
                nodes,
            )
        )

    def test_none_does_not_inflate_f1(self):
        r = score_pairs(
            [{"labels": [], "allowed": ["causes"]}], [[0.1]], ["causes"], 0.5
        )
        self.assertEqual(r["pair_exact_accuracy"], 1)
        self.assertEqual(r["f1"], 0)

    def test_invalid_prediction_contributes_all_false_negatives(self):
        graph = {
            "entities": [
                {"id": "s", "name": "fatigue", "label": "Symptom"},
                {"id": "d", "name": "Condition A", "label": "Disease"},
            ],
            "relations": [
                {
                    "source": "s",
                    "target": "d",
                    "relation": "is_core_symptom_of",
                    "evidence": "fatigue in Condition A",
                }
            ],
        }
        gold = [
            {
                "messages": [
                    {"role": "user", "content": "fatigue in Condition A"},
                    {"role": "assistant", "content": json.dumps(graph)},
                ]
            }
        ]
        r = evaluate_records(gold, ['{"entities": ['])["overall"]
        self.assertEqual(r["relation"]["fn"], 1)
        self.assertEqual(r["entity"]["fn"], 2)
        self.assertEqual(r["invalid_predictions"], 1)
        with self.assertRaises(ValueError):
            evaluate_records(gold, [])

    def test_nondictionary_relation_counts_as_false_positive(self):
        self.assertEqual(
            len(_relation_set({"entities": [], "relations": ["garbage"]})), 1
        )

    def test_fixed_entity_retains_observed_mentions(self):
        doc = SourceRouter().route("Condition A causes fatigue.")
        entities, _, _ = normalize_candidates(
            [EntityCandidate("SENT_001", "Disease", "core", "Condition A")],
            doc,
            [Entity("D1", "Disease", "Condition A")],
        )
        self.assertEqual(len(entities), 1)
        self.assertTrue(entities[0].span_ids)

    def test_classifier_replaces_generation_only(self):
        class Backend:
            def generate(self, prompt, **kwargs):
                if "entity-span selector" not in prompt:
                    raise AssertionError("Relation generation must not run")
                return json.dumps(
                    {
                        "entity_candidates": [
                            {
                                "span_id": "SENT_001",
                                "label": "Symptom",
                                "mention": "Fatigue",
                            },
                            {
                                "span_id": "SENT_001",
                                "label": "Disease",
                                "mention": "Condition A",
                            },
                        ]
                    }
                )

        class Classifier:
            def classify(self, entities, document):
                by_label = {e.label: e.id for e in entities}
                return [
                    Relation(
                        by_label["Symptom"],
                        by_label["Disease"],
                        "is_core_symptom_of",
                        ["SENT_001"],
                        [document.raw],
                    )
                ], {"candidate_pairs": 2}

        result = TwoStageKGAgent(Backend(), relation_classifier=Classifier()).run(
            "Fatigue is a core symptom of Condition A."
        )
        self.assertTrue(result.validation.valid)
        self.assertEqual(result.trace["mode"], "entity_llm_relation_classifier")
        self.assertEqual(len(result.output["relations"]), 1)

    def test_materialization_preserves_multilabel_and_source_text(self):
        doc = SourceRouter().route("Fatigue is a core symptom of Condition A.")
        labels = ["is_core_symptom_of", "supports_diagnosis_of"]
        edges = materialize_pair(
            {"span_ids": ["SENT_001"], "allowed": labels},
            [0.9, 0.8],
            labels,
            0.5,
            doc,
            "S1",
            "D1",
        )
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(e["evidence"] == doc.raw for e in edges))


if __name__ == "__main__":
    unittest.main()

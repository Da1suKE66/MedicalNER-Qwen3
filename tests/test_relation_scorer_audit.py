from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_relation_scorer_20260819 import (  # noqa: E402
    AUDIT_INVERSE_CANDIDATES,
    core_entity_key,
    decompose,
    id_endpoint_audit,
    inverse_normalize,
    raw_id_relations,
    score_counters,
    semantic_relations,
    strict_entity_key,
)


def entity(entity_id: str, label: str, name: str, code: str = "") -> dict:
    properties = {"ICD-11 Code": code} if code else {}
    return {
        "id": entity_id,
        "label": label,
        "name": name,
        "properties": properties,
    }


def relation(source: str, relation_type: str, target: str) -> dict:
    return {
        "source": source,
        "relation": relation_type,
        "target": target,
        "evidence": "ignored by relation identity",
    }


class RelationScorerAuditTests(unittest.TestCase):
    def test_code_error_does_not_cascade_in_core_triple(self) -> None:
        target = {
            "entities": [
                entity("D1", "Disease", "Index disease", "6A00"),
                entity("S1", "Disease", "Subtype one"),
                entity("S2", "Disease", "Subtype two"),
            ],
            "relations": [
                relation("S1", "subtype_of", "D1"),
                relation("S2", "subtype_of", "D1"),
            ],
        }
        prediction = {
            "entities": [
                entity("D1", "Disease", "Index disease", "6A99"),
                entity("S1", "Disease", "Subtype one"),
                entity("S2", "Disease", "Subtype two"),
            ],
            "relations": target["relations"],
        }

        raw = score_counters(raw_id_relations(target), raw_id_relations(prediction))
        strict = score_counters(
            semantic_relations(target, strict_entity_key),
            semantic_relations(prediction, strict_entity_key),
        )
        core = score_counters(
            semantic_relations(target, core_entity_key),
            semantic_relations(prediction, core_entity_key),
        )
        self.assertEqual(raw["tp"], 2)
        self.assertEqual(strict["tp"], 0)
        self.assertEqual(core["tp"], 2)

    def test_raw_ids_can_false_match_different_entities(self) -> None:
        target = {
            "entities": [
                entity("S1", "Symptom", "Target symptom"),
                entity("D1", "Disease", "Target disease"),
            ],
            "relations": [relation("S1", "is_core_symptom_of", "D1")],
        }
        prediction = {
            "entities": [
                entity("S1", "Symptom", "Different symptom"),
                entity("D1", "Disease", "Different disease"),
            ],
            "relations": [relation("S1", "is_core_symptom_of", "D1")],
        }
        raw = score_counters(raw_id_relations(target), raw_id_relations(prediction))
        core = score_counters(
            semantic_relations(target, core_entity_key),
            semantic_relations(prediction, core_entity_key),
        )
        self.assertEqual(raw["tp"], 1)
        self.assertEqual(core["tp"], 0)
        endpoint_audit = id_endpoint_audit(target, prediction)
        self.assertEqual(
            endpoint_audit["matched_raw_id_relations_semantically_conflicting"], 1
        )

    def test_explicit_inverse_candidate_rescues_equivalent_triple(self) -> None:
        target = {
            "entities": [
                entity("D1", "Disease", "Disease"),
                entity("C1", "Diagnostic Criterion", "Criterion"),
            ],
            "relations": [relation("D1", "has_diagnostic_criterion", "C1")],
        }
        prediction = {
            "entities": target["entities"],
            "relations": [relation("C1", "required_for", "D1")],
        }
        target_core = semantic_relations(target, core_entity_key)
        pred_core = semantic_relations(prediction, core_entity_key)
        self.assertEqual(score_counters(target_core, pred_core)["tp"], 0)
        self.assertEqual(
            score_counters(
                inverse_normalize(target_core, AUDIT_INVERSE_CANDIDATES),
                inverse_normalize(pred_core, AUDIT_INVERSE_CANDIDATES),
            )["tp"],
            1,
        )
        self.assertEqual(
            decompose(target_core, pred_core, AUDIT_INVERSE_CANDIDATES)[
                "audit_inverse_candidate_rescued"
            ]["count"],
            1,
        )

    def test_decomposition_finds_reversal_and_wrong_type(self) -> None:
        entities = [
            entity("S1", "Symptom", "Symptom"),
            entity("D1", "Disease", "Disease"),
            entity("S2", "Symptom", "Other symptom"),
        ]
        target = {
            "entities": entities,
            "relations": [
                relation("S1", "supports_diagnosis_of", "D1"),
                relation("S2", "is_core_symptom_of", "D1"),
            ],
        }
        prediction = {
            "entities": entities,
            "relations": [
                relation("D1", "supports_diagnosis_of", "S1"),
                relation("S2", "is_associated_symptom_of", "D1"),
            ],
        }
        parts = decompose(
            semantic_relations(target, core_entity_key),
            semantic_relations(prediction, core_entity_key),
            AUDIT_INVERSE_CANDIDATES,
        )
        self.assertEqual(
            parts["same_relation_type_but_direction_reversed"]["count"], 1
        )
        self.assertEqual(
            parts["both_endpoints_correct_but_type_wrong"]["count"], 1
        )


if __name__ == "__main__":
    unittest.main()

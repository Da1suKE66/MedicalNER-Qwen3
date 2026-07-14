from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import load_schema  # noqa: E402
from kg_lora.semantic_audit import (  # noqa: E402
    REQUIRED_DIMENSIONS,
    CueAssessment,
    DimensionAssessment,
    EntityAssessment,
    ExactSpan,
    RelationAssessment,
    SemanticAuditResponse,
    SemanticOperation,
    build_semantic_audit_task,
)
from kg_lora.semantic_patch import (  # noqa: E402
    SemanticPatchError,
    apply_consensus_patch,
    compile_consensus_patch,
)


class SemanticPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()

    def make_record(self, *, reversed_relation: bool = True) -> dict[str, Any]:
        source_id = "http://id.who.int/icd/release/11/2025-01/mms/123456"
        text = "Canonical Disorder includes distractibility."
        start = text.index("distractibility")
        return {
            "schema_version": self.schema["schema_version"],
            "source_record_id": source_id,
            "source_code": "6A99",
            "source_title": "Canonical Disorder",
            "source_release": self.schema["source_release"],
            "input": text,
            "output": {
                "source_id": source_id,
                "code": "6A99",
                "title": "Canonical Disorder",
                "entities": [
                    {
                        "id": "D1",
                        "label": "Disease",
                        "name": "Canonical Disorder",
                        "properties": {
                            "icdcode": "6A99",
                            "coding_system": "ICD-11-MMS",
                            "icd_release": self.schema["source_release"],
                            "icd_uri": source_id,
                        },
                    },
                    {
                        "id": "S1",
                        "label": "Symptom",
                        "name": "distractibility",
                        "properties": {},
                    },
                ],
                "relations": [
                    {
                        "source": "D1" if reversed_relation else "S1",
                        "target": "S1" if reversed_relation else "D1",
                        "relation": "is_core_symptom_of",
                        "relation_name": "Core Symptom Of",
                        "relation_type": "legacy",
                        "evidence": "distractibility",
                        "evidence_span": {
                            "basis": "record.input",
                            "text": "distractibility",
                            "start": start,
                            "end": start + len("distractibility"),
                        },
                    }
                ],
            },
            "migration": {"status": "repaired"},
        }

    def response(
        self,
        task: dict[str, Any],
        *,
        operation: SemanticOperation | None,
        relation_verdict: str,
    ) -> SemanticAuditResponse:
        return SemanticAuditResponse(
            protocol_version=task["protocol_version"],
            phase=task["phase"],
            dataset_sha256=task["dataset_sha256"],
            schema_sha256=task["schema_sha256"],
            prompt_sha256=task["prompt_sha256"],
            task_sha256=task["task_sha256"],
            source_record_id=task["source_record_id"],
            record_sha256=task["record_sha256"],
            entity_assessments=[
                EntityAssessment(
                    entity_key=item["entity_key"],
                    verdict="correct",
                    reason="Correct grounded entity.",
                    confidence=0.99,
                )
                for item in task["entity_inventory"]
            ],
            relation_assessments=[
                RelationAssessment(
                    relation_key=item["relation_key"],
                    semantic_verdict=relation_verdict,
                    evidence_verdict="exact",
                    proposed_source_ref=(operation.source_ref if operation else None),
                    proposed_target_ref=(operation.target_ref if operation else None),
                    proposed_relation=(operation.replacement_relation if operation else None),
                    replacement_evidence=(operation.evidence if operation else None),
                    reason="The symptom relation direction must follow the schema.",
                    confidence=0.99,
                )
                for item in task["relation_inventory"]
            ],
            cue_assessments=[
                CueAssessment(
                    cue_key=item["cue_key"],
                    verdict="represented_correctly",
                    represented_by_relation_keys=[
                        relation["relation_key"] for relation in task["relation_inventory"]
                    ],
                    proposed_operation_refs=[],
                    reason="Cue inspected.",
                    confidence=0.99,
                )
                for item in task["cue_inventory"]
            ],
            dimension_assessments=[
                DimensionAssessment(
                    dimension=dimension,
                    status=(
                        "issues_found"
                        if operation and dimension == "relation_direction_and_endpoints"
                        else "pass"
                    ),
                    linked_relation_keys=(
                        [task["relation_inventory"][0]["relation_key"]]
                        if operation and dimension == "relation_direction_and_endpoints"
                        else []
                    ),
                    reason="Dimension inspected.",
                )
                for dimension in REQUIRED_DIMENSIONS
            ],
            proposed_operations=[operation] if operation else [],
            unresolved=[],
            overall_confidence=0.99,
        )

    def operations_for_tasks(
        self, primary_task: dict[str, Any], review_task: dict[str, Any]
    ) -> tuple[SemanticOperation, SemanticOperation]:
        def operation(task: dict[str, Any], ref: str) -> SemanticOperation:
            symptom = next(
                item for item in task["entity_inventory"] if item["id"] == "S1"
            )
            disease = next(
                item for item in task["entity_inventory"] if item["id"] == "D1"
            )
            relation = task["relation_inventory"][0]
            start = task["original_text"].index("distractibility")
            return SemanticOperation(
                operation_ref=ref,
                op="replace_relation",
                relation_key=relation["relation_key"],
                source_ref=symptom["entity_key"],
                target_ref=disease["entity_key"],
                replacement_relation="is_core_symptom_of",
                evidence=ExactSpan(
                    text="distractibility",
                    start=start,
                    end=start + len("distractibility"),
                ),
                reason="Use Symptom to Disease direction.",
                confidence=0.99,
            )

        return operation(primary_task, "op:primary"), operation(review_task, "op:review")

    def test_exact_consensus_compiles_and_applies_atomically(self) -> None:
        dataset_sha = "d" * 64
        record = self.make_record()
        original = copy.deepcopy(record)
        primary_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="primary"
        )
        review_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="blind_review"
        )
        primary_operation, review_operation = self.operations_for_tasks(
            primary_task, review_task
        )
        primary = self.response(
            primary_task, operation=primary_operation, relation_verdict="wrong_direction"
        )
        review = self.response(
            review_task, operation=review_operation, relation_verdict="wrong_direction"
        )
        patch = compile_consensus_patch(
            record=record,
            schema=self.schema,
            dataset_sha256=dataset_sha,
            primary=primary,
            review=review,
        )
        repaired, audit = apply_consensus_patch(
            record=record,
            schema=self.schema,
            dataset_sha256=dataset_sha,
            patch=patch,
        )
        self.assertEqual(record, original)
        relation = repaired["output"]["relations"][0]
        self.assertEqual(relation["source"], "S1")
        self.assertEqual(relation["target"], "D1")
        self.assertNotIn("relation_name", relation)
        self.assertNotIn("relation_type", relation)
        self.assertEqual(audit["operations_applied"], 1)
        self.assertIn("semantic_audit", repaired)

    def test_disagreement_and_low_confidence_never_compile(self) -> None:
        dataset_sha = "d" * 64
        record = self.make_record()
        primary_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="primary"
        )
        review_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="blind_review"
        )
        primary_operation, review_operation = self.operations_for_tasks(
            primary_task, review_task
        )
        primary = self.response(
            primary_task, operation=primary_operation, relation_verdict="wrong_direction"
        )
        review = self.response(review_task, operation=None, relation_verdict="correct")
        with self.assertRaisesRegex(SemanticPatchError, "conflict"):
            compile_consensus_patch(
                record=record,
                schema=self.schema,
                dataset_sha256=dataset_sha,
                primary=primary,
                review=review,
            )
        review = self.response(
            review_task, operation=review_operation, relation_verdict="wrong_direction"
        )
        review.proposed_operations[0].confidence = 0.5
        with self.assertRaisesRegex(SemanticPatchError, "confidence"):
            compile_consensus_patch(
                record=record,
                schema=self.schema,
                dataset_sha256=dataset_sha,
                primary=primary,
                review=review,
            )

    def test_patch_hash_or_record_drift_rolls_back(self) -> None:
        dataset_sha = "d" * 64
        record = self.make_record()
        primary_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="primary"
        )
        review_task = build_semantic_audit_task(
            record, self.schema, dataset_sha256=dataset_sha, phase="blind_review"
        )
        primary_operation, review_operation = self.operations_for_tasks(
            primary_task, review_task
        )
        patch = compile_consensus_patch(
            record=record,
            schema=self.schema,
            dataset_sha256=dataset_sha,
            primary=self.response(
                primary_task,
                operation=primary_operation,
                relation_verdict="wrong_direction",
            ),
            review=self.response(
                review_task,
                operation=review_operation,
                relation_verdict="wrong_direction",
            ),
        )
        changed = copy.deepcopy(record)
        changed["input"] += " changed"
        with self.assertRaisesRegex(SemanticPatchError, "precondition mismatch"):
            apply_consensus_patch(
                record=changed,
                schema=self.schema,
                dataset_sha256=dataset_sha,
                patch=patch,
            )


if __name__ == "__main__":
    unittest.main()

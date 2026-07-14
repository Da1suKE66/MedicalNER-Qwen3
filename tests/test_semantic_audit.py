from __future__ import annotations

import copy
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import load_schema, sha256_file, write_json  # noqa: E402
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
    evidence_span_inventory,
    entity_inventory,
    relation_inventory,
    response_requires_review,
    response_semantic_signature,
    validate_semantic_audit_response,
)


class SemanticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()

    def load_cli_module(self) -> Any:
        script_path = ROOT / "scripts/audit_schema_v2_semantics_deepseek.py"
        spec = importlib.util.spec_from_file_location("semantic_audit_cli_test", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def prepare_one_record_run(
        self,
        module: Any,
        directory: Path,
        *,
        timeout_seconds: float = 120.0,
    ) -> tuple[dict[str, Any], Path, Path, str, str]:
        record = self.make_record()
        input_path = directory / "input.json"
        state_db = directory / "state.sqlite3"
        write_json(input_path, [record])
        dataset_sha256 = sha256_file(input_path)
        request_config = {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "thinking": "disabled",
            "reasoning_effort": "high",
            "max_tokens": 32768,
            "timeout_seconds": timeout_seconds,
            "trust_environment_proxy": False,
            "response_format": "json_object",
        }
        run_id = module.prepare_run(
            state_db=state_db,
            input_path=input_path,
            records=[record],
            schema=self.schema,
            dataset_sha256=dataset_sha256,
            request_config=request_config,
            review_all=True,
        )
        return record, input_path, state_db, dataset_sha256, run_id

    def make_record(self) -> dict[str, Any]:
        source_id = "http://id.who.int/icd/release/11/2025-01/mms/123456"
        text = (
            "Canonical Disorder includes inattention, described as distractibility. "
            "In women, diagnostic presentation can differ. Huntington disease may "
            "cause Canonical Disorder and must be ruled out."
        )
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
                        "id": "D2",
                        "label": "Disease",
                        "name": "Huntington disease",
                        "properties": {},
                    },
                    {
                        "id": "S1",
                        "label": "Symptom",
                        "name": "inattention",
                        "properties": {
                            "description": "inattention described as distractibility"
                        },
                    },
                    {
                        "id": "S2",
                        "label": "Symptom",
                        "name": "distractibility",
                        "properties": {},
                    },
                    {
                        "id": "PI1",
                        "label": "Patient Information",
                        "name": "women",
                        "properties": {"special_conditions": "women"},
                    },
                ],
                "relations": [
                    {
                        "source": "S1",
                        "target": "D1",
                        "relation": "is_core_symptom_of",
                        "evidence": "inattention",
                        "evidence_span": {
                            "basis": "record.input",
                            "text": "inattention",
                            "start": text.index("inattention"),
                            "end": text.index("inattention") + len("inattention"),
                        },
                    },
                    {
                        "source": "PI1",
                        "target": "D1",
                        "relation": "affects_diagnosis_of",
                        "evidence": "women",
                        "evidence_span": {
                            "basis": "record.input",
                            "text": "women",
                            "start": text.index("women"),
                            "end": text.index("women") + len("women"),
                        },
                    },
                ],
            },
            "migration": {"status": "manual_review"},
        }

    def complete_response(
        self, task: dict[str, Any], *, operation: SemanticOperation | None = None
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
                    evidence=[],
                    needs_who_lookup=(
                        item["label"] == "Disease"
                        and item["entity_key"] != task["main_disease_entity_key"]
                    ),
                    reason="The entity is grounded and has the correct label.",
                    confidence=0.99,
                )
                for item in task["entity_inventory"]
            ],
            relation_assessments=[
                RelationAssessment(
                    relation_key=item["relation_key"],
                    semantic_verdict="correct",
                    evidence_verdict="exact",
                    reason="The predicate and endpoints are supported.",
                    confidence=0.99,
                )
                for item in task["relation_inventory"]
            ],
            cue_assessments=[
                CueAssessment(
                    cue_key=item["cue_key"],
                    verdict="not_graph_fact",
                    reason="The cue was reviewed and needs no additional graph fact.",
                    confidence=0.99,
                )
                for item in task["cue_inventory"]
            ],
            dimension_assessments=[
                DimensionAssessment(
                    dimension=dimension,
                    status="pass",
                    reason="The complete dimension was reviewed.",
                )
                for dimension in REQUIRED_DIMENSIONS
            ],
            proposed_operations=[operation] if operation else [],
            unresolved=[],
            overall_confidence=0.99,
        )

    def test_stable_keys_do_not_depend_on_distinct_relation_order(self) -> None:
        record = self.make_record()
        graph = record["output"]
        first_entities = entity_inventory(graph)
        first_relations = relation_inventory(graph, first_entities)
        reordered_graph = copy.deepcopy(graph)
        reordered_graph["entities"] = list(reversed(reordered_graph["entities"]))
        reordered_graph["relations"] = list(reversed(reordered_graph["relations"]))
        second_entities = entity_inventory(reordered_graph)
        second_relations = relation_inventory(reordered_graph, second_entities)
        self.assertEqual(
            {item["entity_key"] for item in first_entities},
            {item["entity_key"] for item in second_entities},
        )
        self.assertEqual(
            {item["relation_key"] for item in first_relations},
            {item["relation_key"] for item in second_relations},
        )

    def test_task_is_hash_bound_and_contains_forced_semantic_cues(self) -> None:
        record = self.make_record()
        record["output"]["relations"][1].pop("evidence_span")
        task = build_semantic_audit_task(
            record, self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        self.assertEqual(task["dataset_sha256"], "d" * 64)
        self.assertEqual(len(task["task_sha256"]), 64)
        self.assertEqual(len(task["entity_inventory"]), 5)
        self.assertEqual(len(task["relation_inventory"]), 2)
        categories = {cue["category"] for cue in task["cue_inventory"]}
        self.assertIn("description_fragment_candidate", categories)
        self.assertIn("patient_information", categories)
        self.assertIn("somatic_cause", categories)
        self.assertIn("exclusion", categories)
        self.assertIn("missing_relation_evidence", categories)
        self.assertEqual(task["risk_tier"], "high")

    def test_local_validation_derives_complete_coverage(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        response = self.complete_response(task)
        self.assertEqual(validate_semantic_audit_response(response, task), [])
        response.entity_assessments.pop()
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("missing entity assessment keys" in error for error in errors))

    def test_hash_span_schema_and_main_disease_guards(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        response = self.complete_response(task)
        response.task_sha256 = "0" * 64
        main = next(
            item
            for item in response.entity_assessments
            if item.entity_key == task["main_disease_entity_key"]
        )
        main.verdict = "wrong_label"
        main.recommended_label = "Symptom"
        errors = validate_semantic_audit_response(response, task)
        self.assertIn("task_sha256 mismatch", errors)
        self.assertTrue(any("canonical main Disease" in error for error in errors))

    def test_correct_entity_cannot_carry_a_hidden_recommendation(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        response = self.complete_response(task)
        target = next(
            item
            for item in response.entity_assessments
            if item.entity_key != task["main_disease_entity_key"]
        )
        target.recommended_label = "Symptom"
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("correct verdict requires null" in error for error in errors))

    def test_label_change_requires_lossless_target_schema_properties(self) -> None:
        record = self.make_record()
        record["output"]["entities"][3]["properties"] = {
            "description": "distractibility"
        }
        task = build_semantic_audit_task(
            record, self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        target = next(item for item in task["entity_inventory"] if item["id"] == "S2")
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if item["text"] == "distractibility"
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        operation = SemanticOperation(
            operation_ref="op:relabel-s2",
            op="update_entity",
            entity_key=target["entity_key"],
            replacement_label="Disease",
            replacement_properties={"core_features": "distractibility"},
            evidence=span,
            reason="Reclassify while preserving the description value for WHO lookup.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=operation)
        assessment = next(
            item for item in response.entity_assessments if item.entity_key == target["entity_key"]
        )
        assessment.verdict = "symptom_should_be_disease"
        assessment.recommended_label = "Disease"
        assessment.needs_who_lookup = True
        self.assertEqual(validate_semantic_audit_response(response, task), [])
        operation.replacement_properties = {}
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("preserve every non-empty value" in error for error in errors))

    def test_operation_forces_blind_review_and_exact_evidence(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        symptom = next(
            item for item in task["entity_inventory"] if item["id"] == "S2"
        )
        parent = next(
            item for item in task["entity_inventory"] if item["id"] == "S1"
        )
        text = task["original_text"]
        start = text.index("distractibility")
        operation = SemanticOperation(
            operation_ref="op:collapse-s2",
            op="collapse_symptom_into_manifestations",
            entity_key=symptom["entity_key"],
            parent_entity_key=parent["entity_key"],
            manifestation_text="distractibility",
            evidence=ExactSpan(
                text="distractibility", start=start, end=start + len("distractibility")
            ),
            reason="This is presented only as a manifestation of inattention.",
            confidence=0.98,
        )
        response = self.complete_response(task, operation=operation)
        child_assessment = next(
            item
            for item in response.entity_assessments
            if item.entity_key == symptom["entity_key"]
        )
        child_assessment.verdict = "description_fragment"
        child_assessment.parent_entity_key = parent["entity_key"]
        self.assertEqual(validate_semantic_audit_response(response, task), [])
        self.assertTrue(response_requires_review(response))
        response.proposed_operations[0].evidence.start = 0
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("offsets do not match" in error for error in errors))

    def test_evidence_must_be_selected_from_preindexed_inventory(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        child = next(item for item in task["entity_inventory"] if item["id"] == "S2")
        parent = next(item for item in task["entity_inventory"] if item["id"] == "S1")
        text = task["original_text"]
        start = text.index("includes")
        operation = SemanticOperation(
            operation_ref="op:not-in-inventory",
            op="collapse_symptom_into_manifestations",
            entity_key=child["entity_key"],
            parent_entity_key=parent["entity_key"],
            manifestation_text="distractibility",
            evidence=ExactSpan(text="includes", start=start, end=start + len("includes")),
            reason="Test an exact source span which was not pre-indexed.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=operation)
        assessment = next(
            item for item in response.entity_assessments if item.entity_key == child["entity_key"]
        )
        assessment.verdict = "description_fragment"
        assessment.parent_entity_key = parent["entity_key"]
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("not copied from evidence_span_inventory" in error for error in errors))

    def test_add_relation_closes_dimension_through_missing_fact_cue(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        source = next(item for item in task["entity_inventory"] if item["id"] == "D2")
        target = next(item for item in task["entity_inventory"] if item["id"] == "D1")
        cue = next(item for item in task["cue_inventory"] if item["category"] == "somatic_cause")
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if "Huntington disease" in item["text"] and "cause" in item["text"]
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        operation = SemanticOperation(
            operation_ref="op:add-somatic-relation",
            op="add_relation",
            source_ref=source["entity_key"],
            target_ref=target["entity_key"],
            replacement_relation="somatic_cause_of",
            evidence=span,
            reason="The sentence explicitly states the somatic causal relation.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=operation)
        cue_assessment = next(
            item for item in response.cue_assessments if item.cue_key == cue["cue_key"]
        )
        cue_assessment.verdict = "missing_graph_fact"
        cue_assessment.proposed_operation_refs = [operation.operation_ref]
        dimension = next(
            item for item in response.dimension_assessments if item.dimension == "somatic_causality"
        )
        dimension.status = "issues_found"
        dimension.linked_cue_keys = [cue["cue_key"]]
        self.assertEqual(validate_semantic_audit_response(response, task), [])

    def test_wrong_relation_type_cannot_be_closed_by_evidence_only_operation(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        relation = task["relation_inventory"][0]
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if item["text"] == "inattention"
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        operation = SemanticOperation(
            operation_ref="op:evidence-only",
            op="set_relation_evidence",
            relation_key=relation["relation_key"],
            evidence=span,
            reason="This operation does not repair the predicate.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=operation)
        assessment = next(
            item
            for item in response.relation_assessments
            if item.relation_key == relation["relation_key"]
        )
        assessment.semantic_verdict = "wrong_type"
        dimension = next(
            item for item in response.dimension_assessments if item.dimension == "relation_type"
        )
        dimension.status = "issues_found"
        dimension.linked_relation_keys = [relation["relation_key"]]
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("does not close semantic/evidence verdicts" in error for error in errors))

    def test_relation_findings_reject_semantic_and_evidence_noops(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        relation = task["relation_inventory"][0]
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if item["text"] == "inattention"
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        source = relation["source_entity_key"]
        target = relation["target_entity_key"]
        semantic_noop = SemanticOperation(
            operation_ref="op:semantic-noop",
            op="replace_relation",
            relation_key=relation["relation_key"],
            source_ref=source,
            target_ref=target,
            replacement_relation=relation["relation"],
            evidence=span,
            reason="This deliberately leaves the direction unchanged.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=semantic_noop)
        assessment = response.relation_assessments[0]
        assessment.semantic_verdict = "wrong_direction"
        assessment.proposed_source_ref = source
        assessment.proposed_target_ref = target
        assessment.proposed_relation = relation["relation"]
        assessment.replacement_evidence = span
        dimension = next(
            item
            for item in response.dimension_assessments
            if item.dimension == "relation_direction_and_endpoints"
        )
        dimension.status = "issues_found"
        dimension.linked_relation_keys = [relation["relation_key"]]
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("must reverse the endpoints" in error for error in errors))

        evidence_noop = SemanticOperation(
            operation_ref="op:evidence-noop",
            op="set_relation_evidence",
            relation_key=relation["relation_key"],
            evidence=span,
            reason="This deliberately reuses the already exact span.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=evidence_noop)
        assessment = response.relation_assessments[0]
        assessment.evidence_verdict = "repairable"
        assessment.replacement_evidence = span
        dimension = next(
            item for item in response.dimension_assessments if item.dimension == "data_quality"
        )
        dimension.status = "issues_found"
        dimension.linked_relation_keys = [relation["relation_key"]]
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("must change the exact span" in error for error in errors))

    def test_inactive_relation_with_missing_span_requires_full_replacement(self) -> None:
        record = self.make_record()
        record["output"]["relations"][0]["relation"] = "excludes_if_present"
        record["output"]["relations"][0].pop("evidence_span")
        record["output"]["relations"][0].pop("evidence")
        task = build_semantic_audit_task(
            record, self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        relation = task["relation_inventory"][0]
        source = next(item for item in task["entity_inventory"] if item["id"] == "S1")
        target = next(item for item in task["entity_inventory"] if item["id"] == "D1")
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if item["text"] == "inattention"
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        operation = SemanticOperation(
            operation_ref="op:replace-inactive",
            op="replace_relation",
            relation_key=relation["relation_key"],
            source_ref=source["entity_key"],
            target_ref=target["entity_key"],
            replacement_relation="excludes_diagnosis_of",
            evidence=span,
            reason="Replace the review-status predicate and restore exact evidence.",
            confidence=0.99,
        )
        response = self.complete_response(task, operation=operation)
        assessment = next(
            item
            for item in response.relation_assessments
            if item.relation_key == relation["relation_key"]
        )
        assessment.semantic_verdict = "wrong_type"
        assessment.evidence_verdict = "repairable"
        assessment.proposed_source_ref = operation.source_ref
        assessment.proposed_target_ref = operation.target_ref
        assessment.proposed_relation = operation.replacement_relation
        assessment.replacement_evidence = span
        dimension = next(
            item for item in response.dimension_assessments if item.dimension == "relation_type"
        )
        dimension.status = "issues_found"
        dimension.linked_relation_keys = [relation["relation_key"]]
        self.assertEqual(validate_semantic_audit_response(response, task), [])

    def test_collapse_parent_cannot_be_removed_in_the_same_patch(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        child = next(item for item in task["entity_inventory"] if item["id"] == "S2")
        parent = next(item for item in task["entity_inventory"] if item["id"] == "S1")
        span_data = next(
            item
            for item in task["evidence_span_inventory"]
            if item["text"] == "distractibility"
        )
        span = ExactSpan(**{key: span_data[key] for key in ("basis", "text", "start", "end")})
        collapse = SemanticOperation(
            operation_ref="op:collapse-child",
            op="collapse_symptom_into_manifestations",
            entity_key=child["entity_key"],
            parent_entity_key=parent["entity_key"],
            manifestation_text="distractibility",
            evidence=span,
            reason="Collapse the description fragment.",
            confidence=0.99,
        )
        remove_parent = SemanticOperation(
            operation_ref="op:remove-parent",
            op="remove_entity",
            entity_key=parent["entity_key"],
            evidence=span,
            reason="This deliberately conflicts with the collapse target.",
            confidence=0.99,
        )
        response = self.complete_response(task)
        response.proposed_operations = [collapse, remove_parent]
        child_assessment = next(
            item for item in response.entity_assessments if item.entity_key == child["entity_key"]
        )
        child_assessment.verdict = "description_fragment"
        child_assessment.parent_entity_key = parent["entity_key"]
        parent_assessment = next(
            item for item in response.entity_assessments if item.entity_key == parent["entity_key"]
        )
        parent_assessment.verdict = "unsupported"
        errors = validate_semantic_audit_response(response, task)
        self.assertTrue(any("collapse parent cannot also" in error for error in errors))

    def test_consensus_signature_includes_cue_and_dimension_links(self) -> None:
        task = build_semantic_audit_task(
            self.make_record(), self.schema, dataset_sha256="d" * 64, phase="primary"
        )
        first = self.complete_response(task)
        second = self.complete_response(task)
        relation_keys = [item["relation_key"] for item in task["relation_inventory"]]
        first.cue_assessments[0].verdict = "represented_correctly"
        second.cue_assessments[0].verdict = "represented_correctly"
        first.cue_assessments[0].represented_by_relation_keys = [relation_keys[0]]
        second.cue_assessments[0].represented_by_relation_keys = [relation_keys[1]]
        self.assertEqual(validate_semantic_audit_response(first, task), [])
        self.assertEqual(validate_semantic_audit_response(second, task), [])
        self.assertNotEqual(
            response_semantic_signature(first), response_semantic_signature(second)
        )
        second.cue_assessments[0].represented_by_relation_keys = [relation_keys[0]]
        first.dimension_assessments[0].linked_entity_keys = [
            task["entity_inventory"][0]["entity_key"]
        ]
        second.dimension_assessments[0].linked_entity_keys = [
            task["entity_inventory"][1]["entity_key"]
        ]
        self.assertNotEqual(
            response_semantic_signature(first), response_semantic_signature(second)
        )

    def test_entity_name_inventory_uses_word_boundaries(self) -> None:
        text = "development mental requirements Men are described here."
        spans = evidence_span_inventory(
            text,
            [{"name": "Men"}],
            [],
            [],
        )
        exact_names = [item for item in spans if item["text"].casefold() == "men"]
        self.assertEqual(len(exact_names), 1)
        self.assertEqual(exact_names[0]["start"], text.index("Men"))

    def test_plan_cli_needs_no_api_call_and_persists_all_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path = directory / "input.json"
            state_db = directory / "state.sqlite3"
            report_path = directory / "report.json"
            write_json(input_path, [self.make_record()])
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/audit_schema_v2_semantics_deepseek.py"),
                    "plan",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--report",
                    str(report_path),
                    "--env-file",
                    str(directory / "missing.env"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["expected"]["records"], 1)
            self.assertEqual(report["primary_states"], {"pending": 1})
            self.assertEqual(report["review_states"], {"pending": 1})
            self.assertFalse(report["semantic_audit_complete"])
            self.assertFalse(report["training_unlocked"])

    def test_run_identity_binds_non_secret_deepseek_request_config(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path = directory / "input.json"
            state_db = directory / "state.sqlite3"
            report_path = directory / "report.json"
            env_path = directory / "audit.env"
            write_json(input_path, [self.make_record()])
            clean_env = os.environ.copy()
            for key in list(clean_env):
                if key.startswith("DEEPSEEK_"):
                    clean_env.pop(key)
            command = [
                sys.executable,
                str(ROOT / "scripts/audit_schema_v2_semantics_deepseek.py"),
                "plan",
                "--input",
                str(input_path),
                "--state-db",
                str(state_db),
                "--report",
                str(report_path),
                "--env-file",
                str(env_path),
            ]
            env_path.write_text(
                "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
                "DEEPSEEK_MODEL=deepseek-v4-flash\n"
                "DEEPSEEK_MAX_TOKENS=32768\n",
                encoding="utf-8",
            )
            first = subprocess.run(
                command,
                cwd=ROOT,
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            first_report = json.loads(report_path.read_text(encoding="utf-8"))
            env_path.write_text(
                "DEEPSEEK_BASE_URL=https://another-provider.example/v1\n"
                "DEEPSEEK_MODEL=deepseek-v4-flash\n"
                "DEEPSEEK_MAX_TOKENS=32768\n",
                encoding="utf-8",
            )
            second = subprocess.run(
                command,
                cwd=ROOT,
                env=clean_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            second_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotEqual(first_report["run_id"], second_report["run_id"])
            self.assertNotEqual(
                first_report["request_config_sha256"],
                second_report["request_config_sha256"],
            )

    def test_reset_record_cli_forces_both_phases_pending_and_keeps_history(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path = directory / "input.json"
            state_db = directory / "state.sqlite3"
            report_path = directory / "report.json"
            missing_env = directory / "missing.env"
            record = self.make_record()
            source_record_id = record["source_record_id"]
            write_json(input_path, [record])
            script = str(ROOT / "scripts/audit_schema_v2_semantics_deepseek.py")
            planned = subprocess.run(
                [
                    sys.executable,
                    script,
                    "plan",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--report",
                    str(report_path),
                    "--env-file",
                    str(missing_env),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            run_id = json.loads(report_path.read_text(encoding="utf-8"))["run_id"]

            primary_digest = "1" * 64
            review_digest = "2" * 64
            with closing(sqlite3.connect(state_db)) as connection:
                row = connection.execute(
                    "SELECT primary_task_sha256, review_task_sha256 FROM records "
                    "WHERE run_id=? AND source_record_id=?",
                    (run_id, source_record_id),
                ).fetchone()
                self.assertIsNotNone(row)
                for digest, phase, task_sha256 in (
                    (primary_digest, "primary", row[0]),
                    (review_digest, "blind_review", row[1]),
                ):
                    connection.execute(
                        """
                        INSERT INTO responses (
                            response_sha256, run_id, source_record_id, phase,
                            attempt_no, task_sha256, response_json, created_at
                        ) VALUES (?, ?, ?, ?, 3, ?, '{}', 'test')
                        """,
                        (digest, run_id, source_record_id, phase, task_sha256),
                    )
                    connection.execute(
                        """
                        INSERT INTO api_attempts (
                            lease_token, run_id, source_record_id, phase,
                            attempt_no, state, response_sha256, usage_json,
                            started_at, finished_at
                        ) VALUES (?, ?, ?, ?, 3, 'done', ?, ?, 'test', 'test')
                        """,
                        (
                            f"lease-{phase}",
                            run_id,
                            source_record_id,
                            phase,
                            digest,
                            json.dumps({"total_tokens": 10}),
                        ),
                    )
                connection.execute(
                    """
                    UPDATE records SET
                        primary_state='done', review_state='done',
                        primary_attempts=3, review_attempts=2,
                        primary_response_sha256=?, review_response_sha256=?,
                        primary_error='old primary error',
                        review_error='old review error',
                        final_status='ready_to_patch'
                    WHERE run_id=? AND source_record_id=?
                    """,
                    (primary_digest, review_digest, run_id, source_record_id),
                )
                event_count_before = connection.execute(
                    "SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                connection.commit()

            reset = subprocess.run(
                [
                    sys.executable,
                    script,
                    "reset-record",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--report",
                    str(report_path),
                    "--env-file",
                    str(missing_env),
                    "--run-id",
                    run_id,
                    "--source-record-id",
                    source_record_id,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(reset.returncode, 0, reset.stderr)
            with closing(sqlite3.connect(state_db)) as connection:
                reset_row = connection.execute(
                    """
                    SELECT primary_state, review_state, primary_attempts,
                           review_attempts, primary_response_sha256,
                           review_response_sha256, primary_error, review_error,
                           final_status
                    FROM records WHERE run_id=? AND source_record_id=?
                    """,
                    (run_id, source_record_id),
                ).fetchone()
                self.assertEqual(
                    reset_row,
                    ("pending", "pending", 0, 0, None, None, None, None, None),
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM responses WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM api_attempts WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    event_count_before + 1,
                )
                details = json.loads(
                    connection.execute(
                        "SELECT details_json FROM events WHERE run_id=? "
                        "ORDER BY id DESC LIMIT 1",
                        (run_id,),
                    ).fetchone()[0]
                )
                self.assertEqual(details["reset_mode"], "record")
                self.assertEqual(details["previous_final_status"], "ready_to_patch")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["primary_states"], {"pending": 1})
            self.assertEqual(report["review_states"], {"pending": 1})
            self.assertEqual(
                report["api_request_ledger"]["total_attempted_requests"], 2
            )
            self.assertEqual(
                report["api_request_ledger"]["known_token_usage"]["total_tokens"],
                20,
            )

            missing_id = subprocess.run(
                [
                    sys.executable,
                    script,
                    "reset-record",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--env-file",
                    str(missing_env),
                    "--run-id",
                    run_id,
                    "--source-record-id",
                    "not-a-planned-record",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(missing_id.returncode, 0)
            self.assertIn("not planned", missing_id.stderr)

            missing_argument = subprocess.run(
                [
                    sys.executable,
                    script,
                    "reset-record",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--env-file",
                    str(missing_env),
                    "--run-id",
                    run_id,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(missing_argument.returncode, 0)
            self.assertIn("requires --source-record-id", missing_argument.stderr)

    def test_api_attempt_lease_prevents_stale_worker_state_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path = directory / "input.json"
            state_db = directory / "state.sqlite3"
            report_path = directory / "report.json"
            record = self.make_record()
            write_json(input_path, [record])
            script_path = ROOT / "scripts/audit_schema_v2_semantics_deepseek.py"
            planned = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "plan",
                    "--input",
                    str(input_path),
                    "--state-db",
                    str(state_db),
                    "--report",
                    str(report_path),
                    "--env-file",
                    str(directory / "missing.env"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            run_id = json.loads(report_path.read_text(encoding="utf-8"))["run_id"]

            spec = importlib.util.spec_from_file_location("semantic_audit_cli", script_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            source_id = record["source_record_id"]
            task = build_semantic_audit_task(
                record,
                self.schema,
                dataset_sha256=sha256_file(input_path),
                phase="primary",
            )
            attempt_no, lease_token = module.mark_running(
                state_db, run_id, source_id, "primary"
            )
            module.save_success(
                state_db=state_db,
                run_id=run_id,
                source_record_id=source_id,
                phase="primary",
                attempt_no=attempt_no,
                lease_token=lease_token,
                task=task,
                response=self.complete_response(task),
                meta={
                    "model": "deepseek-v4-flash",
                    "finish_reason": "stop",
                    "usage": {"total_tokens": 7},
                },
            )
            with closing(sqlite3.connect(state_db)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT primary_state FROM records WHERE run_id=?",
                        (run_id,),
                    ).fetchone()[0],
                    "done",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT state FROM api_attempts WHERE lease_token=?",
                        (lease_token,),
                    ).fetchone()[0],
                    "done",
                )
                connection.execute(
                    "UPDATE records SET primary_state='pending', "
                    "primary_response_sha256=NULL WHERE run_id=?",
                    (run_id,),
                )
                connection.commit()
            stale_attempt, stale_lease = module.mark_running(
                state_db, run_id, source_id, "primary"
            )
            with closing(sqlite3.connect(state_db)) as connection:
                connection.execute(
                    "UPDATE records SET primary_state='pending', "
                    "primary_lease_token=NULL, primary_started_at=NULL WHERE run_id=?",
                    (run_id,),
                )
                connection.commit()
            module.save_failure(
                state_db=state_db,
                run_id=run_id,
                source_record_id=source_id,
                phase="primary",
                attempt_no=stale_attempt,
                lease_token=stale_lease,
                state="api_error",
                error="late worker",
            )
            with closing(sqlite3.connect(state_db)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT primary_state FROM records WHERE run_id=?",
                        (run_id,),
                    ).fetchone()[0],
                    "pending",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT state FROM api_attempts WHERE lease_token=?",
                        (stale_lease,),
                    ).fetchone()[0],
                    "discarded_stale_failure",
                )

    def test_402_is_non_retryable(self) -> None:
        module = self.load_cli_module()
        calls = 0

        def fail_with_402(**_: Any) -> tuple[str, str, str]:
            nonlocal calls
            calls += 1
            return "record-1", "api_error", "DeepSeek API error status=402: balance"

        with mock.patch.object(module, "process_one", side_effect=fail_with_402), mock.patch.object(
            module.time, "sleep"
        ):
            result = module.process_with_retries(
                remaining_attempts=4,
                record={"source_record_id": "record-1"},
            )
        self.assertEqual(result[1], "api_error")
        self.assertEqual(calls, 1)

    def test_402_stops_unsent_records_across_the_batch(self) -> None:
        module = self.load_cli_module()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            records: list[dict[str, Any]] = []
            for index in range(3):
                record = copy.deepcopy(self.make_record())
                source_id = f"{record['source_record_id']}/batch-{index}"
                record["source_record_id"] = source_id
                record["output"]["source_id"] = source_id
                record["output"]["entities"][0]["properties"]["icd_uri"] = source_id
                records.append(record)
            input_path = directory / "input.json"
            state_db = directory / "state.sqlite3"
            write_json(input_path, records)
            dataset_sha256 = sha256_file(input_path)
            request_config = {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-flash",
                "thinking": "disabled",
                "reasoning_effort": "high",
                "max_tokens": 32768,
                "timeout_seconds": 120.0,
                "trust_environment_proxy": False,
                "response_format": "json_object",
            }
            run_id = module.prepare_run(
                state_db=state_db,
                input_path=input_path,
                records=records,
                schema=self.schema,
                dataset_sha256=dataset_sha256,
                request_config=request_config,
                review_all=True,
            )
            calls = 0

            def fail_with_402(**_: Any) -> tuple[str, str, str]:
                nonlocal calls
                calls += 1
                return records[0]["source_record_id"], "api_error", "status=402"

            with mock.patch.object(module, "process_one", side_effect=fail_with_402):
                results = module.run_phase(
                    state_db=state_db,
                    run_id=run_id,
                    records=records,
                    schema=self.schema,
                    dataset_sha256=dataset_sha256,
                    phase="primary",
                    config=module.DeepSeekConfig(api_key="test-key"),
                    workers=1,
                    max_attempts=4,
                    source_record_id=None,
                    limit=None,
                )
            self.assertEqual(calls, 1)
            self.assertEqual(results, {"api_error": 1, "cancelled": 2})

    def test_stale_lease_window_honors_frozen_request_timeout(self) -> None:
        module = self.load_cli_module()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record, _, state_db, _, run_id = self.prepare_one_record_run(
                module, directory, timeout_seconds=1200.0
            )
            source_id = record["source_record_id"]
            _, lease_token = module.mark_running(state_db, run_id, source_id, "primary")
            sixteen_minutes_ago = (
                module.datetime.now(module.timezone.utc)
                - module.timedelta(minutes=16)
            ).isoformat()
            with closing(sqlite3.connect(state_db)) as connection:
                connection.execute(
                    "UPDATE records SET primary_started_at=? WHERE run_id=?",
                    (sixteen_minutes_ago, run_id),
                )
                connection.commit()

            self.assertEqual(module.reset_stale_running(state_db, run_id, "primary"), 0)
            with closing(sqlite3.connect(state_db)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT primary_state FROM records WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    "running",
                )

                connection.execute(
                    "UPDATE records SET primary_started_at=? WHERE run_id=?",
                    (
                        (
                            module.datetime.now(module.timezone.utc)
                            - module.timedelta(minutes=26)
                        ).isoformat(),
                        run_id,
                    ),
                )
                connection.commit()

            self.assertEqual(module.reset_stale_running(state_db, run_id, "primary"), 1)
            with closing(sqlite3.connect(state_db)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT primary_state FROM records WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    "pending",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT state FROM api_attempts WHERE lease_token=?",
                        (lease_token,),
                    ).fetchone()[0],
                    "abandoned_stale",
                )

    def test_failed_reset_serializes_against_new_active_lease(self) -> None:
        module = self.load_cli_module()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record, _, state_db, _, run_id = self.prepare_one_record_run(module, directory)
            source_id = record["source_record_id"]
            with closing(sqlite3.connect(state_db)) as connection:
                connection.execute(
                    "UPDATE records SET primary_state='api_error' WHERE run_id=?",
                    (run_id,),
                )
                connection.commit()

            writer = sqlite3.connect(state_db)
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "UPDATE records SET primary_state='running', "
                "primary_lease_token='live-lease', primary_started_at=? WHERE run_id=?",
                (module.now_utc(), run_id),
            )
            result: list[int] = []
            errors: list[BaseException] = []

            def reset_in_parallel() -> None:
                try:
                    result.append(
                        module.reset_records_for_retry(
                            state_db=state_db,
                            run_id=run_id,
                            mode="failed",
                        )
                    )
                except BaseException as exc:  # preserve thread failures for assertion
                    errors.append(exc)

            thread = threading.Thread(target=reset_in_parallel)
            thread.start()
            time.sleep(0.1)
            self.assertTrue(thread.is_alive())
            writer.commit()
            writer.close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(result, [0])
            with closing(sqlite3.connect(state_db)) as connection:
                row = connection.execute(
                    "SELECT primary_state, primary_lease_token FROM records WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                self.assertEqual(row, ("running", "live-lease"))

    def test_validation_failure_preserves_provider_usage_in_ledger(self) -> None:
        module = self.load_cli_module()
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record, _, state_db, dataset_sha256, run_id = self.prepare_one_record_run(
                module, directory
            )
            task = build_semantic_audit_task(
                record,
                self.schema,
                dataset_sha256=dataset_sha256,
                phase="primary",
            )
            response = self.complete_response(task)
            response.record_sha256 = "0" * 64
            meta = {
                "model": "deepseek-v4-flash",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 6,
                    "total_tokens": 17,
                },
            }
            fake_client = mock.Mock()
            fake_client.propose_model.return_value = (response, meta)
            with mock.patch.object(module, "DeepSeekRepairClient", return_value=fake_client):
                result = module.process_one(
                    state_db=state_db,
                    run_id=run_id,
                    record=record,
                    schema=self.schema,
                    dataset_sha256=dataset_sha256,
                    phase="primary",
                    config=module.DeepSeekConfig(api_key="test-key"),
                )
            self.assertEqual(result[1], "invalid")
            report = module.status_report(state_db, run_id)
            self.assertEqual(
                report["api_request_ledger"]["known_token_usage"], meta["usage"]
            )
            self.assertEqual(
                report["api_request_ledger"]["attempts_with_provider_usage"], 1
            )


if __name__ == "__main__":
    unittest.main()

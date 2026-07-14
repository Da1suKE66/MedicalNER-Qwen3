from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.build_v2_splits import (  # noqa: E402
    build_hierarchy_groups,
    hierarchy_leakage_report,
)
from kg_lora.convert_schema_v2_to_llamafactory import (  # noqa: E402
    convert_records,
    evaluate_training_eligibility,
    load_schema,
    main as convert_main,
    validate_split_integrity,
)


class SchemaV2TrainingGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()

    def make_record(self, status: str = "repaired") -> dict:
        source_text = "Persistent inattention is an explicitly documented symptom."
        evidence = "Persistent inattention"
        start = source_text.index(evidence)
        source_id = "http://id.who.int/icd/release/11/2025-01/mms/example"
        graph = {
            "source_id": source_id,
            "code": "6A99",
            "title": "Example disorder",
            "entities": [
                {
                    "id": "D1",
                    "label": "Disease",
                    "name": "Example disorder",
                    "properties": {
                        "icdcode": "6A99",
                        "coding_system": "ICD-11-MMS",
                        "icd_release": "2025-01",
                        "icd_uri": source_id,
                    },
                },
                {
                    "id": "S1",
                    "label": "Symptom",
                    "name": "inattention",
                    "properties": {},
                },
            ],
            "relations": [
                {
                    "source": "S1",
                    "target": "D1",
                    "relation": "is_core_symptom_of",
                    "evidence": evidence,
                    "evidence_span": {
                        "basis": "record.input",
                        "text": evidence,
                        "start": start,
                        "end": start + len(evidence),
                    },
                }
            ],
        }
        return {
            "schema_version": self.schema["schema_version"],
            "source_record_id": source_id,
            "source_id": source_id,
            "source_code": "6A99",
            "source_title": "Example disorder",
            "source_release": "2025-01",
            "input": source_text,
            "output": graph,
            "icd_validation": {
                "status": "verified",
                "source": "WHO ICD-11 MMS API",
                "api_version": "v2",
                "release": "2025-01",
                "canonical_code": "6A99",
                "entity_uri": source_id,
                "title": "Example disorder",
                "errors": [],
            },
            "cot": "<think>This legacy reasoning must never be copied.</think>",
            "migration": {
                "status": status,
                "errors": [],
                "warnings": [],
                "unverified_codes": [],
            },
        }

    def test_shared_external_parent_is_a_virtual_union_find_node(self) -> None:
        raw = [
            {"id": "A", "parent": ["EXTERNAL-P"], "child": []},
            {"id": "B", "parent": ["EXTERNAL-P"], "child": []},
            {"id": "C", "parent": ["EXTERNAL-Q"], "child": []},
        ]
        groups, metadata = build_hierarchy_groups(raw)
        group_sets = [set(group) for group in groups]
        self.assertIn({"A", "B"}, group_sets)
        self.assertIn({"C"}, group_sets)
        self.assertEqual(metadata["external_parent_node_count"], 2)
        self.assertEqual(metadata["shared_external_parent_node_count"], 1)

        leaked = hierarchy_leakage_report(raw, {"A": "train", "B": "validation"})
        self.assertFalse(leaked["passed"])
        self.assertEqual(leaked["shared_external_parent_cross_split_count"], 1)

    def test_gate_accepts_only_approved_or_repaired_status(self) -> None:
        for status in ("approved", "repaired"):
            with self.subTest(status=status):
                result = evaluate_training_eligibility(
                    self.make_record(status), self.schema
                )
                self.assertTrue(result["eligible"])

        for status in ("invalid", "manual_review"):
            with self.subTest(status=status):
                result = evaluate_training_eligibility(
                    self.make_record(status), self.schema
                )
                self.assertFalse(result["eligible"])
                self.assertIn(
                    "status_not_trainable",
                    {reason["code"] for reason in result["excluded_reasons"]},
                )

    def test_gate_rejects_unverified_codes(self) -> None:
        record = self.make_record()
        record["migration"]["unverified_codes"] = [
            {"entity_id": "D2", "value": "6A98"}
        ]
        result = evaluate_training_eligibility(record, self.schema)
        self.assertFalse(result["eligible"])
        self.assertIn(
            "unverified_code",
            {reason["code"] for reason in result["excluded_reasons"]},
        )

    def test_gate_requires_verified_icd_validation(self) -> None:
        missing = self.make_record()
        missing.pop("icd_validation")
        missing_result = evaluate_training_eligibility(missing, self.schema)
        self.assertFalse(missing_result["eligible"])
        self.assertIn(
            "icd_validation_not_verified",
            {reason["code"] for reason in missing_result["excluded_reasons"]},
        )

        invalid = self.make_record()
        invalid["icd_validation"] = {
            "status": "invalid",
            "errors": [{"code": "canonical_icd_title_mismatch"}],
        }
        invalid_result = evaluate_training_eligibility(invalid, self.schema)
        self.assertFalse(invalid_result["eligible"])
        self.assertIn(
            "icd_validation_not_verified",
            {reason["code"] for reason in invalid_result["excluded_reasons"]},
        )

    def test_gate_rejects_any_schema_error_or_warning_and_inactive_relation(self) -> None:
        invalid = self.make_record()
        invalid["output"]["code"] = "WRONG"
        invalid_result = evaluate_training_eligibility(invalid, self.schema)
        self.assertFalse(invalid_result["eligible"])
        self.assertIn(
            "schema_validation_errors",
            {reason["code"] for reason in invalid_result["excluded_reasons"]},
        )

        warning = self.make_record()
        warning["output"]["relations"][0]["relation"] = "precedes"
        warning_result = evaluate_training_eligibility(warning, self.schema)
        warning_reasons = {
            reason["code"] for reason in warning_result["excluded_reasons"]
        }
        self.assertFalse(warning_result["eligible"])
        self.assertIn("schema_validation_warnings", warning_reasons)
        self.assertIn("relation_not_active", warning_reasons)

    def test_gate_requires_a_unique_canonical_main_entity_and_nonempty_graph(self) -> None:
        empty = self.make_record()
        empty["output"]["entities"] = []
        empty["output"]["relations"] = []
        empty_result = evaluate_training_eligibility(empty, self.schema)
        empty_reasons = {
            reason["code"] for reason in empty_result["excluded_reasons"]
        }
        self.assertIn("canonical_graph_empty", empty_reasons)
        self.assertIn("canonical_main_entity_missing", empty_reasons)

        duplicate = self.make_record()
        second_main = copy.deepcopy(duplicate["output"]["entities"][0])
        second_main["id"] = "D2"
        duplicate["output"]["entities"].append(second_main)
        duplicate_result = evaluate_training_eligibility(duplicate, self.schema)
        self.assertIn(
            "canonical_main_entity_not_unique",
            {reason["code"] for reason in duplicate_result["excluded_reasons"]},
        )

    def test_non_main_entity_requires_text_or_verified_relation_grounding(self) -> None:
        grounded_by_relation = self.make_record()
        grounded_by_relation["output"]["entities"][1]["name"] = (
            "not literally present"
        )
        relation_result = evaluate_training_eligibility(
            grounded_by_relation, self.schema
        )
        self.assertTrue(relation_result["eligible"])

        ungrounded = self.make_record()
        ungrounded["output"]["entities"][1]["name"] = "not literally present"
        ungrounded["output"]["relations"] = []
        ungrounded_result = evaluate_training_eligibility(ungrounded, self.schema)
        self.assertFalse(ungrounded_result["eligible"])
        self.assertIn(
            "ungrounded_non_main_entity",
            {reason["code"] for reason in ungrounded_result["excluded_reasons"]},
        )

        normalized_whitespace = self.make_record()
        normalized_whitespace["input"] = "Persistent\n\tinattention is documented."
        normalized_whitespace["output"]["entities"][1]["name"] = (
            "Persistent inattention"
        )
        normalized_whitespace["output"]["relations"] = []
        whitespace_result = evaluate_training_eligibility(
            normalized_whitespace, self.schema
        )
        self.assertTrue(whitespace_result["eligible"])

        partial_word = self.make_record()
        partial_word["output"]["entities"][1]["name"] = "attention"
        partial_word["output"]["relations"] = []
        partial_result = evaluate_training_eligibility(partial_word, self.schema)
        self.assertFalse(partial_result["eligible"])
        self.assertIn(
            "ungrounded_non_main_entity",
            {reason["code"] for reason in partial_result["excluded_reasons"]},
        )

    def test_gate_rejects_missing_or_invalid_evidence_spans(self) -> None:
        missing = self.make_record()
        del missing["output"]["relations"][0]["evidence_span"]
        missing_result = evaluate_training_eligibility(missing, self.schema)
        self.assertIn(
            "missing_verified_evidence_span",
            {reason["code"] for reason in missing_result["excluded_reasons"]},
        )

        invalid = self.make_record()
        invalid["output"]["relations"][0]["evidence_span"]["start"] += 1
        invalid_result = evaluate_training_eligibility(invalid, self.schema)
        self.assertIn(
            "invalid_evidence_span",
            {reason["code"] for reason in invalid_result["excluded_reasons"]},
        )

    def test_converter_uses_canonical_json_and_never_legacy_cot(self) -> None:
        approved = self.make_record("approved")
        rejected = self.make_record("manual_review")
        rejected["source_record_id"] += "/rejected"
        rejected["source_id"] = rejected["source_record_id"]
        converted, manifest = convert_records([approved, rejected], self.schema)

        self.assertEqual(len(converted), 1)
        self.assertEqual(manifest["counts"], {"input": 2, "eligible": 1, "excluded": 1})
        self.assertFalse(manifest["legacy_cot_reused"])
        self.assertFalse(manifest["strict_json_allow_nan"])
        self.assertEqual(set(converted[0]), {"messages"})
        assistant = converted[0]["messages"][-1]["content"]
        self.assertNotIn("<think>", assistant)
        self.assertNotIn("legacy reasoning", assistant)
        self.assertEqual(
            json.loads(assistant),
            {
                "entities": approved["output"]["entities"],
                "relations": approved["output"]["relations"],
            },
        )
        json.dumps(converted, ensure_ascii=False, allow_nan=False)

    def test_non_finite_canonical_output_is_rejected(self) -> None:
        record = self.make_record()
        record["output"]["entities"][1]["properties"]["score"] = float("nan")
        result = evaluate_training_eligibility(record, self.schema)
        self.assertFalse(result["eligible"])
        self.assertIn(
            "canonical_graph_not_strict_json",
            {reason["code"] for reason in result["excluded_reasons"]},
        )

    def test_split_integrity_requires_shared_parent_leakage_check(self) -> None:
        record = self.make_record()
        source_id = record["source_record_id"]
        manifest = {
            "source_ids": {"train": [source_id]},
            "hierarchy_leakage_check": {
                "passed": True,
                "direct_cross_split_edge_count": 0,
                "shared_external_parent_cross_split_count": 0,
            },
        }
        result = validate_split_integrity([record], manifest, "train")
        self.assertTrue(result["passed"])

        bad_manifest = copy.deepcopy(manifest)
        bad_manifest["hierarchy_leakage_check"][
            "shared_external_parent_cross_split_count"
        ] = 1
        with self.assertRaisesRegex(ValueError, "shared external-parent leakage"):
            validate_split_integrity([record], bad_manifest, "train")

    def test_cli_writes_auditable_manifest_and_dry_run_writes_nothing(self) -> None:
        record = self.make_record()
        source_id = record["source_record_id"]
        split_manifest = {
            "source_ids": {"train": [source_id]},
            "hierarchy_leakage_check": {
                "passed": True,
                "direct_cross_split_edge_count": 0,
                "shared_external_parent_cross_split_count": 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_path = base / "train.json"
            split_path = base / "split_manifest.json"
            output_path = base / "train_llamafactory.json"
            gate_path = base / "training_gate.json"
            input_path.write_text(
                json.dumps([record], ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            split_path.write_text(
                json.dumps(split_manifest, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )

            common_args = [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--manifest",
                str(gate_path),
                "--split-manifest",
                str(split_path),
            ]
            self.assertEqual(convert_main([*common_args, "--dry-run"]), 0)
            self.assertFalse(output_path.exists())
            self.assertFalse(gate_path.exists())

            self.assertEqual(convert_main(common_args), 0)
            self.assertEqual(len(json.loads(output_path.read_text(encoding="utf-8"))), 1)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertEqual(gate["counts"]["eligible"], 1)
            self.assertTrue(gate["split_integrity"]["passed"])
            self.assertIn("output_sha256", gate)


if __name__ == "__main__":
    unittest.main()

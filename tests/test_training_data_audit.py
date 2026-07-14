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

from scripts.audit_schema_v2_training_data import main as audit_main  # noqa: E402
from kg_lora.schema_v2 import load_json, load_schema, write_json  # noqa: E402
from kg_lora.training_data_audit import audit_training_data  # noqa: E402


class TrainingDataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()

    def make_record(self) -> dict:
        source_id = "http://id.who.int/icd/release/11/2025-01/mms/example"
        source_text = "Canonical Disorder includes grounded symptom."
        evidence = "grounded symptom"
        start = source_text.index(evidence)
        return {
            "schema_version": self.schema["schema_version"],
            "source_record_id": source_id,
            "source_code": "6A99",
            "source_title": "Canonical Disorder",
            "source_release": self.schema["source_release"],
            "input": source_text,
            "icd_validation": {"status": "verified", "errors": []},
            "migration": {
                "status": "repaired",
                "errors": [],
                "warnings": [],
                "unverified_codes": [],
            },
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
                        "name": evidence,
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
            },
        }

    def make_manifest(self, source_id: str) -> dict:
        return {
            "schema_version": self.schema["schema_version"],
            "counts": {
                "train": 1,
                "validation": 0,
                "test_v2_heldout": 0,
                "schema_regression_20": 0,
            },
            "source_ids": {
                "train": [source_id],
                "validation": [],
                "test_v2_heldout": [],
                "schema_regression_20": [],
            },
            "hierarchy_leakage_check": {
                "passed": True,
                "direct_cross_split_edge_count": 0,
                "direct_cross_split_edges": [],
                "shared_external_parent_cross_split_count": 0,
                "shared_external_parent_cross_splits": [],
            },
        }

    def audit(self, record: dict, manifest: dict | None = None) -> dict:
        return audit_training_data(
            [record],
            manifest or self.make_manifest(record["source_record_id"]),
            self.schema,
            expected_split="all",
        )

    def test_complete_train_ready_record_passes_every_gate(self) -> None:
        report = self.audit(self.make_record())
        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_gates"], [])
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["relation_count"], 1)
        self.assertTrue(all(gate["passed"] for gate in report["gates"].values()))
        self.assertEqual(
            report["gates"]["split_integrity"]["details"][
                "direct_cross_split_edge_count"
            ],
            0,
        )

    def test_each_required_quality_gate_rejects_its_failure(self) -> None:
        cases: list[tuple[str, dict, str]] = []

        non_finite = self.make_record()
        non_finite["output"]["entities"][1]["properties"]["description"] = float(
            "nan"
        )
        cases.append(("non_finite", non_finite, "strict_json"))

        warned_relation = self.make_record()
        warned_relation["output"]["relations"][0]["relation"] = "precedes"
        cases.append(("non_active_relation", warned_relation, "active_relation_schema"))

        wrong_main = self.make_record()
        wrong_main["output"]["entities"][0]["name"] = "Different Disorder"
        cases.append(("main_identity", wrong_main, "canonical_main_identity"))

        bad_icd = self.make_record()
        bad_icd["icd_validation"] = {
            "status": "invalid",
            "errors": ["WHO mismatch"],
        }
        bad_icd["migration"]["unverified_codes"] = [
            {"entity_id": "D2", "value": "6B00"}
        ]
        cases.append(("icd", bad_icd, "icd_verification"))

        bad_span = self.make_record()
        bad_span["output"]["relations"][0]["evidence_span"]["start"] += 1
        cases.append(("evidence", bad_span, "exact_evidence_spans"))

        damaged_word = self.make_record()
        damaged_word["output"]["entities"][1]["properties"][
            "description"
        ] = "pregnullcy"
        cases.append(("word_corruption", damaged_word, "known_word_corruption"))

        for label, record, expected_gate in cases:
            with self.subTest(case=label):
                report = self.audit(record)
                self.assertFalse(report["passed"])
                self.assertIn(expected_gate, report["failed_gates"])

        record = self.make_record()
        leaked_manifest = self.make_manifest(record["source_record_id"])
        leaked_manifest["hierarchy_leakage_check"][
            "direct_cross_split_edge_count"
        ] = 1
        leaked_manifest["hierarchy_leakage_check"]["passed"] = False
        report = self.audit(record, leaked_manifest)
        self.assertIn("split_integrity", report["failed_gates"])

    def test_relation_direction_and_validator_warning_are_gate_failures(self) -> None:
        reversed_record = self.make_record()
        relation = reversed_record["output"]["relations"][0]
        relation["source"], relation["target"] = relation["target"], relation["source"]
        report = self.audit(reversed_record)
        self.assertIn("schema_validation", report["failed_gates"])
        self.assertIn("active_relation_schema", report["failed_gates"])

        warning_record = self.make_record()
        warning_record["output"]["relations"][0]["relation"] = "precedes"
        report = self.audit(warning_record)
        self.assertIn("schema_validation", report["failed_gates"])
        self.assertGreater(
            report["gates"]["schema_validation"]["finding_count"], 0
        )

    def test_cli_writes_report_and_strict_mode_has_clear_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record = self.make_record()
            manifest = self.make_manifest(record["source_record_id"])
            input_path = directory / "train_ready.json"
            manifest_path = directory / "split_manifest.json"
            report_path = directory / "audit.json"
            write_json(input_path, [record])
            write_json(manifest_path, manifest)

            args = [
                "--input",
                str(input_path),
                "--split-manifest",
                str(manifest_path),
                "--report",
                str(report_path),
                "--expected-split",
                "all",
                "--strict",
            ]
            self.assertEqual(audit_main(args), 0)
            self.assertTrue(load_json(report_path)["passed"])

            damaged = copy.deepcopy(record)
            damaged["output"]["entities"][1]["properties"][
                "description"
            ] = "pregnullcy"
            write_json(input_path, [damaged])
            self.assertEqual(audit_main(args), 1)
            failed_report = load_json(report_path)
            self.assertFalse(failed_report["passed"])
            self.assertIn("known_word_corruption", failed_report["failed_gates"])

    def test_cli_reports_non_finite_json_and_never_overwrites_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            record = self.make_record()
            manifest = self.make_manifest(record["source_record_id"])
            input_path = directory / "non_finite.json"
            manifest_path = directory / "split_manifest.json"
            report_path = directory / "audit.json"
            non_finite = copy.deepcopy(record)
            non_finite["output"]["entities"][1]["properties"]["score"] = float(
                "inf"
            )
            input_path.write_text(json.dumps([non_finite]), encoding="utf-8")
            write_json(manifest_path, manifest)
            args = [
                "--input",
                str(input_path),
                "--split-manifest",
                str(manifest_path),
                "--report",
                str(report_path),
                "--strict",
            ]
            self.assertEqual(audit_main(args), 1)
            report = load_json(report_path)
            self.assertEqual(report["failed_gates"], ["strict_json"])
            error = report["gates"]["strict_json"]["findings"][0]["error"]
            self.assertIn("non-finite JSON number", error)

            self.assertEqual(
                audit_main(
                    [
                        "--input",
                        str(input_path),
                        "--split-manifest",
                        str(manifest_path),
                        "--report",
                        str(input_path),
                    ]
                ),
                2,
            )
            self.assertIn("Infinity", input_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

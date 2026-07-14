from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    MIGRATION_IMPLEMENTATION_VERSION,
    build_raw_indexes,
    load_json,
    load_schema,
    migrate_record,
    migration_config_fingerprint,
    validate_record,
)


class SchemaV2CoreRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()
        cls.source = {
            "id": "http://id.who.int/icd/release/11/2025-01/mms/123456789",
            "code": "6A99",
            "title": "Canonical Disorder",
        }
        cls.raw_indexes = build_raw_indexes([cls.source])

    def make_record(self) -> dict:
        return {
            "input": "Canonical Disorder includes distractibility.",
            "source_id": self.source["id"],
            "code": self.source["code"],
            "title": self.source["title"],
            "output": {
                "source_id": "model-invented-id",
                "code": "wrong-code",
                "title": "model invented title",
                "entities": [
                    {
                        "id": "D1",
                        "label": "Disease",
                        "name": self.source["title"],
                        "properties": {},
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
                        "source": "S1",
                        "target": "D1",
                        "relation": "is_core_symptom_of",
                        "evidence": "distractibility",
                    }
                ],
            },
        }

    def migrate(self, record: dict | None = None, **kwargs):
        return migrate_record(
            record if record is not None else self.make_record(),
            self.raw_indexes,
            self.schema,
            **kwargs,
        )

    def test_json_reader_rejects_every_non_finite_literal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for literal in ("NaN", "Infinity", "-Infinity", "1e400"):
                with self.subTest(literal=literal):
                    path.write_text(f'{{"value": {literal}}}', encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                        load_json(path)

    def test_nested_output_source_identity_is_canonical_and_validated(self) -> None:
        migrated, result = self.migrate()
        self.assertEqual(result["status"], "repaired")
        self.assertEqual(
            {
                key: migrated["output"][key] for key in ("source_id", "code", "title")
            },
            {
                "source_id": self.source["id"],
                "code": self.source["code"],
                "title": self.source["title"],
            },
        )
        self.assertFalse(validate_record(migrated, self.schema)["errors"])

        migrated["output"]["code"] = "not-canonical"
        errors = validate_record(migrated, self.schema)["errors"]
        self.assertIn(
            ("graph_source_identity_mismatch", "code"),
            {(error["code"], error.get("field")) for error in errors},
        )

    def test_evidence_text_and_verified_span_have_separate_strict_metrics(self) -> None:
        migrated, _ = self.migrate()
        result = validate_record(migrated, self.schema)
        self.assertEqual(
            result["metrics"]["relations_with_retained_evidence_text"], 1
        )
        self.assertEqual(
            result["metrics"]["relations_with_verified_evidence_span"], 1
        )

        retained_only = copy.deepcopy(migrated)
        retained_only["output"]["relations"][0].pop("evidence_span")
        retained_result = validate_record(retained_only, self.schema)
        self.assertEqual(
            retained_result["metrics"]["relations_with_retained_evidence_text"], 1
        )
        self.assertEqual(
            retained_result["metrics"]["relations_with_verified_evidence_span"], 0
        )

        corrupted = copy.deepcopy(migrated)
        corrupted["output"]["relations"][0]["evidence_span"]["text"] = "different"
        corrupted_result = validate_record(corrupted, self.schema)
        self.assertIn(
            "evidence_span_text_mismatch",
            {error["code"] for error in corrupted_result["errors"]},
        )
        self.assertEqual(
            corrupted_result["metrics"]["relations_with_verified_evidence_span"], 0
        )

        non_string = copy.deepcopy(migrated)
        non_string["output"]["relations"][0]["evidence"] = ["distractibility"]
        non_string_errors = validate_record(non_string, self.schema)["errors"]
        self.assertIn(
            "relation_evidence_not_string",
            {error["code"] for error in non_string_errors},
        )
        _, non_string_migration = self.migrate(
            non_string, force_renormalize=True
        )
        self.assertEqual(non_string_migration["status"], "invalid")
        self.assertIn(
            "relation_evidence_not_string",
            {error["code"] for error in non_string_migration["errors"]},
        )

    def test_skip_requires_implementation_and_config_and_force_reruns(self) -> None:
        migrated, result = self.migrate()
        expected_fingerprint = migration_config_fingerprint(
            self.schema, apply_high_confidence_collapses=False
        )
        self.assertEqual(
            result["implementation_version"], MIGRATION_IMPLEMENTATION_VERSION
        )
        self.assertEqual(result["config_fingerprint"], expected_fingerprint)

        old_marker = copy.deepcopy(migrated)
        old_marker["migration"].pop("implementation_version")
        old_marker["output"]["code"] = "stale"
        refreshed, refreshed_result = self.migrate(old_marker)
        self.assertEqual(refreshed["output"]["code"], self.source["code"])
        self.assertEqual(
            refreshed_result["implementation_version"],
            MIGRATION_IMPLEMENTATION_VERSION,
        )

        apparently_current = copy.deepcopy(refreshed)
        apparently_current["output"]["code"] = "externally-corrupted"
        skipped, _ = self.migrate(apparently_current)
        self.assertEqual(skipped["output"]["code"], "externally-corrupted")
        forced, _ = self.migrate(apparently_current, force_renormalize=True)
        self.assertEqual(forced["output"]["code"], self.source["code"])

        changed_config, changed_result = self.migrate(
            refreshed, apply_high_confidence_collapses=True
        )
        self.assertNotEqual(
            changed_result["config_fingerprint"], expected_fingerprint
        )
        self.assertEqual(changed_config["output"]["code"], self.source["code"])

    def test_missing_canonical_main_disease_is_injected_without_guessing(self) -> None:
        record = self.make_record()
        record["output"]["entities"] = [
            {
                "id": "D_CANONICAL_SOURCE",
                "label": "Disease",
                "name": "A nearby but different disorder",
                "properties": {},
            }
        ]
        record["output"]["relations"] = []
        migrated, result = self.migrate(record)
        injected = [
            entity
            for entity in migrated["output"]["entities"]
            if entity.get("name") == self.source["title"]
        ]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["id"], "D_CANONICAL_SOURCE_2")
        self.assertEqual(
            injected[0]["properties"]["icd_uri"], self.source["id"]
        )
        self.assertIn(
            "inject_canonical_main_disease",
            {change["op"] for change in result["changes"]},
        )
        self.assertFalse(validate_record(migrated, self.schema)["errors"])

    def test_multiple_exact_main_diseases_are_rejected_not_guessed(self) -> None:
        record = self.make_record()
        duplicate = copy.deepcopy(record["output"]["entities"][0])
        duplicate["id"] = "D2"
        record["output"]["entities"].append(duplicate)
        migrated, result = self.migrate(record)
        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "main_disease_ambiguous", {error["code"] for error in result["errors"]}
        )
        self.assertEqual(
            len(
                [
                    entity
                    for entity in migrated["output"]["entities"]
                    if entity.get("name") == self.source["title"]
                ]
            ),
            2,
        )

    def test_malformed_entities_and_relations_are_never_filtered_silently(self) -> None:
        record = self.make_record()
        record["output"]["entities"].append("not-an-entity")
        record["output"]["entities"][1]["properties"] = []
        record["output"]["relations"].extend(
            [42, {"source": "S1", "target": "D1"}]
        )
        migrated, migration = self.migrate(record)
        migration_codes = {error["code"] for error in migration["errors"]}
        self.assertTrue(
            {
                "entity_not_object",
                "entity_properties_not_object",
                "relation_not_object",
                "missing_relation_field",
            }.issubset(migration_codes)
        )
        self.assertEqual(migration["status"], "invalid")

        validation_codes = {
            error["code"] for error in validate_record(migrated, self.schema)["errors"]
        }
        self.assertIn("entity_not_object", validation_codes)
        self.assertIn("relation_not_object", validation_codes)
        self.assertIn("missing_relation_field", validation_codes)

        malformed_properties = copy.deepcopy(migrated)
        malformed_properties["output"]["entities"][0]["properties"] = []
        validation_codes = {
            error["code"]
            for error in validate_record(malformed_properties, self.schema)["errors"]
        }
        self.assertIn("entity_properties_not_object", validation_codes)


if __name__ == "__main__":
    unittest.main()

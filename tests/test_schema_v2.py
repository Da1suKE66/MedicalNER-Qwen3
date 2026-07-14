from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    build_raw_indexes,
    load_json,
    load_schema,
    migrate_record,
    write_json,
)


class SchemaV2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()
        cls.raw = load_json(ROOT / "data/raw/mental_disorders_20251125_165535.json")
        cls.raw_indexes = build_raw_indexes(cls.raw)
        cls.migrated = load_json(
            ROOT / "data/schema_v2/migrated/pro_cot_schema_v2.json"
        )
        cls.regression = load_json(
            ROOT / "data/schema_regression/schema_regression_20_v2_draft.json"
        )
        cls.split_manifest = load_json(
            ROOT / "data/schema_v2/splits/split_manifest.json"
        )

    def test_null_corruption_audit_is_reproducible(self) -> None:
        report = load_json(ROOT / "reports/null_corruption_audit.json")
        self.assertEqual(report["record_count"], 858)
        self.assertEqual(report["damaged_token_types"], 22)
        self.assertEqual(report["damaged_occurrences"], 305)
        self.assertEqual(report["json_null_count"], 858)
        self.assertEqual(
            report["null_paths"], {"$.entities[*].excel_metadata.browser_link": 858}
        )
        self.assertEqual(report["unexpected_tokens_containing_null"], {})

    def test_json_writer_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                write_json(output, {"value": float("nan")})
            self.assertFalse(output.exists())

    def test_full_migration_is_lossless_and_idempotent(self) -> None:
        report = load_json(ROOT / "reports/schema_v2_migration_report.json")
        self.assertEqual(report["input_record_count"], 858)
        self.assertEqual(report["output_record_count"], 858)
        self.assertTrue(report["no_silent_record_loss"])
        self.assertTrue(report["idempotent"])
        migrated_again, _ = migrate_record(
            self.migrated[0],
            self.raw_indexes,
            self.schema,
            apply_high_confidence_collapses=True,
        )
        self.assertEqual(migrated_again, self.migrated[0])

    def test_main_disease_code_has_provenance(self) -> None:
        resolved = 0
        for record in self.migrated:
            graph = record["output"]
            matches = [
                entity
                for entity in graph["entities"]
                if entity.get("label") == "Disease"
                and " ".join(entity.get("name", "").lower().split())
                == " ".join(record["source_title"].lower().split())
            ]
            if len(matches) != 1:
                continue
            resolved += 1
            properties = matches[0]["properties"]
            self.assertEqual(properties["icdcode"], record["source_code"])
            self.assertEqual(properties["coding_system"], "ICD-11-MMS")
            self.assertEqual(properties["icd_release"], "2025-01")
            self.assertEqual(properties["icd_uri"], record["source_record_id"])
        self.assertEqual(resolved, 858)

    def test_inattention_description_children_are_collapsed(self) -> None:
        record = next(item for item in self.regression if item["test_id"] == "KGTEST_003")
        entities = {entity["id"]: entity for entity in record["gold_output"]["entities"]}
        self.assertNotIn("S2", entities)
        self.assertNotIn("S3", entities)
        self.assertNotIn("S4", entities)
        self.assertEqual(
            entities["S1"]["properties"]["manifestations"],
            [
                "difficulty in sustaining attention to tasks that do not provide a high level of stimulation or frequent rewards",
                "distractibility",
                "problems with organisation",
            ],
        )
        relation_endpoints = {
            (relation["source"], relation["target"], relation["relation"])
            for relation in record["gold_output"]["relations"]
        }
        self.assertIn(("S1", "D1", "is_core_symptom_of"), relation_endpoints)
        collapse_operations = [
            change
            for change in record["migration"]["changes"]
            if change["op"] == "move_entity_to_manifestation"
        ]
        self.assertEqual(len(collapse_operations), 3)
        self.assertTrue(
            all(
                change["evidence"]["basis"] == "record.input"
                for change in collapse_operations
            )
        )

    def test_primary_splits_are_disjoint_and_hierarchy_isolated(self) -> None:
        source_ids = self.split_manifest["source_ids"]
        train = set(source_ids["train"])
        validation = set(source_ids["validation"])
        heldout = set(source_ids["test_v2_heldout"])
        regression = set(source_ids["schema_regression_20"])
        self.assertFalse(train & validation)
        self.assertFalse(train & heldout)
        self.assertFalse(validation & heldout)
        self.assertFalse((train | validation | heldout) & regression)
        self.assertEqual(len(train | validation | heldout | regression), 858)
        self.assertEqual(self.split_manifest["cross_split_hierarchy_edges"], [])


if __name__ == "__main__":
    unittest.main()

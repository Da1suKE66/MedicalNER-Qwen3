from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import load_schema  # noqa: E402
from kg_lora.training_sanitizer import sanitize_record_for_training  # noqa: E402


def _record() -> dict:
    schema = load_schema()
    uri = "http://id.who.int/icd/release/11/2025-01/mms/1"
    text = "Canonical Disorder includes grounded symptom."
    start = text.index("grounded symptom")
    return {
        "schema_version": schema["schema_version"],
        "source_record_id": uri,
        "source_code": "6A99",
        "source_title": "Canonical Disorder",
        "source_release": "2025-01",
        "input": text,
        "cot": "legacy reasoning that must never reach a training-ready record",
        "input_used": text,
        "response_had_think_tag": True,
        "icd_validation": {"status": "verified"},
        "migration": {"status": "manual_review", "unverified_codes": []},
        "output": {
            "source_id": uri,
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
                        "icd_release": "2025-01",
                        "icd_uri": uri,
                    },
                },
                {
                    "id": "S1",
                    "label": "Symptom",
                    "name": "grounded symptom",
                    "properties": {},
                },
                {
                    "id": "PI1",
                    "label": "Patient Information",
                    "name": "model-only paraphrase",
                    "properties": {},
                },
            ],
            "relations": [
                {
                    "source": "S1",
                    "target": "D1",
                    "relation": "is_core_symptom_of",
                    "evidence": "grounded symptom",
                    "evidence_span": {
                        "basis": "record.input",
                        "text": "grounded symptom",
                        "start": start,
                        "end": start + len("grounded symptom"),
                    },
                },
                {
                    "source": "S1",
                    "target": "D1",
                    "relation": "precedes",
                    "evidence": "grounded symptom",
                    "evidence_span": {
                        "basis": "record.input",
                        "text": "grounded symptom",
                        "start": start,
                        "end": start + len("grounded symptom"),
                    },
                },
            ],
        },
    }


def test_prunes_review_relation_and_ungrounded_orphan_with_audit() -> None:
    cleaned, audit = sanitize_record_for_training(_record(), load_schema())
    assert [relation["relation"] for relation in cleaned["output"]["relations"]] == [
        "is_core_symptom_of"
    ]
    assert [entity["id"] for entity in cleaned["output"]["entities"]] == ["D1", "S1"]
    assert audit["dropped_relations"][0]["reason"] == "relation_requires_medical_review"
    assert audit["dropped_entities"][0]["entity_id"] == "PI1"
    assert "cot" not in cleaned
    assert "input_used" not in cleaned
    assert "response_had_think_tag" not in cleaned
    assert audit["removed_legacy_fields"] == [
        "cot",
        "input_used",
        "response_had_think_tag",
    ]
    assert cleaned["migration"]["status"] == "repaired"


def test_unverified_icd_status_remains_invalid() -> None:
    record = _record()
    record["icd_validation"] = {"status": "invalid", "errors": ["mismatch"]}
    cleaned, _ = sanitize_record_for_training(record, load_schema())
    assert cleaned["migration"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_icd_not_verified"
        for error in cleaned["migration"]["errors"]
    )


def test_missing_icd_validation_fails_closed() -> None:
    record = _record()
    record.pop("icd_validation")
    cleaned, _ = sanitize_record_for_training(record, load_schema())
    assert cleaned["migration"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_icd_not_verified"
        and error["details"] == "icd_validation is missing"
        for error in cleaned["migration"]["errors"]
    )


def test_punctuation_normalized_orphan_is_not_treated_as_text_grounded() -> None:
    record = _record()
    record["input"] += " A transient physiological aftereffect (hangover effect) may occur."
    record["output"]["entities"].append(
        {
            "id": "S2",
            "label": "Symptom",
            "name": "transient physiological aftereffect ('hangover effect')",
            "properties": {},
        }
    )
    cleaned, audit = sanitize_record_for_training(record, load_schema())
    assert "S2" not in {entity["id"] for entity in cleaned["output"]["entities"]}
    assert any(
        item.get("entity_id") == "S2"
        and item["reason"]
        == "not_canonical_not_text_grounded_and_no_grounded_relation"
        for item in audit["dropped_entities"]
    )

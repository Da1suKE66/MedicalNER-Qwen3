from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.icd_validation import apply_icd_validation  # noqa: E402
from kg_lora.schema_v2 import load_schema  # noqa: E402
import scripts.apply_icd_validation as apply_cli  # noqa: E402


def _record() -> dict:
    schema = load_schema()
    uri = "http://id.who.int/icd/release/11/2025-01/mms/123"
    return {
        "schema_version": schema["schema_version"],
        "source_record_id": uri,
        "source_code": "6A99",
        "source_title": "Canonical Disorder",
        "source_release": "2025-01",
        "input": "Canonical Disorder text.",
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
                    "id": "D2",
                    "label": "Disease",
                    "name": "Other Disorder",
                    "properties": {},
                },
            ],
            "relations": [],
        },
        "migration": {
            "status": "manual_review",
            "errors": [],
            "warnings": [{"code": "non_main_code_requires_entity_linking"}],
            "unverified_codes": [
                {"entity_id": "D2", "property": "DSM-5 Code", "value": "6B00"}
            ],
        },
    }


def _report() -> dict:
    return {
        "who": {"release": "2025-01", "language": "en", "api_version": "v2"},
        "results": [
            {
                "requested_code": "6A99",
                "canonical_code": "6A99",
                "entity_uri": "https://id.who.int/icd/release/11/2025-01/mms/123",
                "title": "Canonical Disorder",
                "release": "2025-01",
                "language": "en",
                "valid": True,
            },
            {
                "requested_code": "6B00",
                "canonical_code": "6B00",
                "entity_uri": "http://id.who.int/icd/release/11/2025-01/mms/456",
                "title": "Other Disorder",
                "release": "2025-01",
                "language": "en",
                "valid": True,
            },
        ],
        "errors": [],
    }


def test_validates_main_and_discards_only_confirmed_non_main_code() -> None:
    cleaned, audit = apply_icd_validation([_record()], _report(), load_schema())
    record = cleaned[0]
    assert record["icd_validation"]["status"] == "verified"
    assert record["migration"]["unverified_codes"] == []
    discarded = record["migration"]["discarded_non_main_codes"]
    assert discarded[0]["code"] == "6B00"
    assert discarded[0]["title_matches"] is True
    assert record["migration"]["status"] == "repaired"
    assert audit["strict_ready"] is True
    assert audit["totals"]["canonical_verified"] == 1
    assert audit["totals"]["non_main_validated_and_discarded"] == 1


def test_missing_main_api_result_keeps_record_invalid() -> None:
    report = _report()
    report["results"] = report["results"][1:]
    cleaned, audit = apply_icd_validation([_record()], report, load_schema())
    assert cleaned[0]["migration"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_icd_api_validation_missing"
        for error in cleaned[0]["migration"]["errors"]
    )
    assert audit["records_with_icd_findings"] == 1


def test_non_main_title_mismatch_is_discarded_with_comparison_evidence() -> None:
    report = _report()
    report["results"][1]["title"] = "Different Official Title"
    cleaned, audit = apply_icd_validation([_record()], report, load_schema())
    record = cleaned[0]
    assert record["icd_validation"]["status"] == "verified"
    assert record["migration"]["status"] == "repaired"
    assert record["migration"]["unverified_codes"] == []
    discarded = record["migration"]["discarded_non_main_codes"]
    assert discarded[0]["action"] == "title_mismatch_not_attached"
    assert discarded[0]["entity_name"] == "Other Disorder"
    assert discarded[0]["who_title"] == "Different Official Title"
    assert discarded[0]["who_entity_uri"].endswith("/456")
    assert discarded[0]["title_comparison"] == {
        "expected_entity_name": "Other Disorder",
        "who_title": "Different Official Title",
        "normalized_expected": "other disorder",
        "normalized_who_title": "different official title",
        "matches": False,
    }
    assert audit["strict_ready"] is True
    assert audit["totals"]["non_main_title_mismatch_and_discarded"] == 1


def test_non_main_transient_api_error_stays_unverified() -> None:
    report = _report()
    report["results"] = report["results"][:1]
    report["errors"] = [
        {
            "code": "6B00",
            "http_status": 503,
            "endpoint": "https://id.who.int/example",
            "response_summary": "temporarily unavailable",
        }
    ]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    pending = cleaned[0]["migration"]["unverified_codes"]
    assert len(pending) == 1
    assert pending[0]["value"] == "6B00"
    assert cleaned[0]["migration"]["discarded_non_main_codes"] == []
    assert cleaned[0]["migration"]["status"] == "manual_review"


def test_non_main_missing_result_stays_unverified() -> None:
    report = _report()
    report["results"] = report["results"][:1]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    pending = cleaned[0]["migration"]["unverified_codes"]
    assert len(pending) == 1
    assert any(
        error["code"] == "non_main_icd_api_validation_missing"
        for error in pending[0]["validation_errors"]
    )


def test_non_main_404_is_definitively_invalid_and_does_not_block_record() -> None:
    report = _report()
    report["results"] = report["results"][:1]
    report["errors"] = [
        {
            "code": "6B00",
            "http_status": 404,
            "endpoint": "https://id.who.int/example",
            "response_summary": "not found",
        }
    ]
    cleaned, audit = apply_icd_validation([_record()], report, load_schema())
    record = cleaned[0]
    assert record["migration"]["unverified_codes"] == []
    discarded = record["migration"]["discarded_non_main_codes"]
    assert discarded[0]["action"] == "invalid_code_not_attached"
    assert discarded[0]["api_errors"][0]["http_status"] == 404
    assert record["migration"]["status"] == "repaired"
    assert audit["strict_ready"] is True
    assert audit["totals"]["non_main_definitively_invalid_and_discarded"] == 1


def test_non_main_explicit_does_not_exist_is_definitive() -> None:
    report = _report()
    report["results"] = report["results"][:1]
    report["errors"] = [
        {
            "code": "6B00",
            "http_status": 400,
            "endpoint": "https://id.who.int/example",
            "response_summary": "The code does not exist in this release",
        }
    ]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    assert cleaned[0]["migration"]["unverified_codes"] == []
    assert (
        cleaned[0]["migration"]["discarded_non_main_codes"][0]["action"]
        == "invalid_code_not_attached"
    )


def test_canonical_success_row_cannot_hide_api_error() -> None:
    report = _report()
    report["errors"] = [
        {
            "code": "6A99",
            "http_status": 503,
            "endpoint": "https://id.who.int/example",
            "response_summary": "temporarily unavailable",
        }
    ]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    record = cleaned[0]
    assert record["icd_validation"]["status"] == "invalid"
    assert record["migration"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_icd_api_error"
        for error in record["icd_validation"]["errors"]
    )


def test_canonical_404_is_always_invalid() -> None:
    report = _report()
    report["results"] = report["results"][1:]
    report["errors"] = [
        {
            "code": "6A99",
            "http_status": 404,
            "endpoint": "https://id.who.int/example",
            "response_summary": "not found",
        }
    ]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    assert cleaned[0]["icd_validation"]["status"] == "invalid"
    assert cleaned[0]["migration"]["status"] == "invalid"


def test_result_requires_explicit_valid_flag() -> None:
    report = _report()
    report["results"][0].pop("valid")
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    assert cleaned[0]["icd_validation"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_icd_result_not_explicitly_valid"
        for error in cleaned[0]["icd_validation"]["errors"]
    )


def test_canonical_entity_properties_must_match_verified_source() -> None:
    record = _record()
    record["output"]["entities"][0]["properties"]["icd_uri"] = (
        "http://id.who.int/icd/release/11/2025-01/mms/wrong"
    )
    cleaned, _ = apply_icd_validation([record], _report(), load_schema())
    assert cleaned[0]["icd_validation"]["status"] == "invalid"
    assert any(
        error["code"] == "canonical_main_disease_property_mismatch"
        for error in cleaned[0]["icd_validation"]["errors"]
    )


def test_previously_discarded_code_is_reopened_on_transient_error() -> None:
    first, _ = apply_icd_validation([_record()], _report(), load_schema())
    report = _report()
    report["results"] = report["results"][:1]
    report["errors"] = [
        {
            "code": "6B00",
            "http_status": 429,
            "endpoint": "https://id.who.int/example",
            "response_summary": "rate limited",
        }
    ]
    second, _ = apply_icd_validation(first, report, load_schema())
    assert second[0]["migration"]["discarded_non_main_codes"] == []
    assert second[0]["migration"]["unverified_codes"][0]["value"] == "6B00"


def test_missing_unverified_codes_field_does_not_create_fake_pending_entry() -> None:
    record = _record()
    record["migration"].pop("unverified_codes")
    report = _report()
    report["results"] = report["results"][:1]
    cleaned, audit = apply_icd_validation([record], report, load_schema())
    assert cleaned[0]["migration"]["unverified_codes"] == []
    assert cleaned[0]["migration"]["status"] == "repaired"
    assert audit["strict_ready"] is True


def test_schema_warning_is_retained_but_does_not_fail_who_strict_readiness() -> None:
    record = _record()
    evidence = "Canonical Disorder"
    record["output"]["relations"] = [
        {
            "source": "D1",
            "target": "D2",
            "relation": "precedes",
            "evidence": evidence,
            "evidence_span": {
                "basis": "record.input",
                "text": evidence,
                "start": 0,
                "end": len(evidence),
            },
        }
    ]
    cleaned, audit = apply_icd_validation([record], _report(), load_schema())
    assert cleaned[0]["icd_validation"]["status"] == "verified"
    assert cleaned[0]["migration"]["status"] == "manual_review"
    assert cleaned[0]["migration"]["warnings"]
    assert audit["strict_ready"] is True
    assert audit["records_with_icd_findings"] == 0
    assert audit["records_with_schema_warnings"] == 1


def test_mixed_404_and_transient_error_is_not_definitive() -> None:
    report = _report()
    report["results"] = report["results"][:1]
    report["errors"] = [
        {"code": "6B00", "http_status": 404, "response_summary": "not found"},
        {"code": "6B00", "http_status": 503, "response_summary": "retry"},
    ]
    cleaned, _ = apply_icd_validation([_record()], report, load_schema())
    assert cleaned[0]["migration"]["discarded_non_main_codes"] == []
    assert cleaned[0]["migration"]["unverified_codes"][0]["value"] == "6B00"


def test_strict_cli_does_not_write_training_input_when_any_record_is_not_ready() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.json"
        who_path = root / "who.json"
        output_path = root / "output.json"
        audit_path = root / "audit.json"
        input_path.write_text(json.dumps([_record()]), encoding="utf-8")
        report = _report()
        report["results"] = report["results"][1:]
        who_path.write_text(json.dumps(report), encoding="utf-8")

        exit_code = apply_cli.main(
            [
                "--input",
                str(input_path),
                "--who-report",
                str(who_path),
                "--output",
                str(output_path),
                "--report",
                str(audit_path),
                "--strict",
            ]
        )
        assert exit_code == 1
        assert not output_path.exists()
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["strict_ready"] is False
        assert audit["output_written"] is False


def test_strict_cli_writes_only_fully_verified_training_input() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.json"
        who_path = root / "who.json"
        output_path = root / "output.json"
        audit_path = root / "audit.json"
        input_path.write_text(json.dumps([_record()]), encoding="utf-8")
        who_path.write_text(json.dumps(_report()), encoding="utf-8")

        exit_code = apply_cli.main(
            [
                "--input",
                str(input_path),
                "--who-report",
                str(who_path),
                "--output",
                str(output_path),
                "--report",
                str(audit_path),
                "--strict",
            ]
        )
        assert exit_code == 0
        assert output_path.exists()
        output = json.loads(output_path.read_text(encoding="utf-8"))
        assert output[0]["icd_validation"]["status"] == "verified"
        assert output[0]["migration"]["status"] == "repaired"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["strict_ready"] is True
        assert audit["output_written"] is True


def test_release_mismatch_is_rejected() -> None:
    report = _report()
    report["who"]["release"] = "2026-01"
    try:
        apply_icd_validation([_record()], report, load_schema())
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("release mismatch was accepted")

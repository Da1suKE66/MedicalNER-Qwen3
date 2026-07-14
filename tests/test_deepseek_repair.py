from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.deepseek_repair import (  # noqa: E402
    DeepSeekAPIError,
    DeepSeekConfig,
    DeepSeekRepairClient,
    Evidence,
    PatchValidationError,
    RepairOperation,
    RepairPatch,
    apply_repair_patch,
    build_repair_task,
    cached_patch_path,
    record_sha256,
    save_safe_api_result,
    validate_patch_context,
)
from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    validate_record,
    write_json,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> Any:
        if self._payload is None:
            return json.loads(self.text)
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def post(self, endpoint: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"endpoint": endpoint, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class DeepSeekRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_schema()

    def make_record(self) -> dict[str, Any]:
        source_id = "http://id.who.int/icd/release/11/2025-01/mms/123456789"
        source_title = "Canonical Disorder"
        source_text = "Canonical Disorder includes distractibility."
        return {
            "schema_version": self.schema["schema_version"],
            "source_record_id": source_id,
            "source_code": "6A99",
            "source_title": source_title,
            "source_release": self.schema["source_release"],
            "input": source_text,
            "output": {
                "source_id": source_id,
                "code": "6A99",
                "title": source_title,
                "entities": [
                    {
                        "id": "D1",
                        "label": "Disease",
                        "name": source_title,
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
                # Deliberately reversed so a valid swap patch has one real error to fix.
                "relations": [
                    {
                        "source": "D1",
                        "target": "S1",
                        "relation": "is_core_symptom_of",
                        "evidence": "distractibility",
                    }
                ],
            },
            "migration": {"status": "invalid"},
        }

    def evidence(self, record: dict[str, Any], quote: str = "distractibility") -> Evidence:
        start = record["input"].index(quote)
        return Evidence(quote=quote, start=start, end=start + len(quote))

    def patch(
        self,
        record: dict[str, Any],
        operation: RepairOperation | None = None,
        **overrides: Any,
    ) -> RepairPatch:
        payload: dict[str, Any] = {
            "source_record_id": record["source_record_id"],
            "schema_version": record["schema_version"],
            "record_sha256": record_sha256(record),
            "operations": [operation] if operation else [],
            "needs_human_review": False,
            "review_reason": None,
        }
        payload.update(overrides)
        return RepairPatch.model_validate(payload)

    def swap_operation(
        self, record: dict[str, Any], *, confidence: float = 0.99
    ) -> RepairOperation:
        return RepairOperation(
            op="swap_relation_endpoints",
            relation_index=0,
            expected_relation_type="is_core_symptom_of",
            evidence=self.evidence(record),
            reason="The symptom must be the source.",
            confidence=confidence,
        )

    def test_v4_flash_json_request_is_exact_and_parseable(self) -> None:
        record = self.make_record()
        patch = self.patch(record, self.swap_operation(record))
        response = FakeResponse(
            payload={
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(patch.model_dump(mode="json"))
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        session = FakeSession(response=response)
        config = DeepSeekConfig(api_key="secret-test-key")
        client = DeepSeekRepairClient(config, session=session)
        task = build_repair_task(record, self.schema, validate_record(record, self.schema))

        actual_patch, meta = client.propose_patch(task)

        self.assertEqual(actual_patch, patch)
        self.assertEqual(meta["model"], "deepseek-v4-flash")
        self.assertEqual(len(session.calls), 1)
        call = session.calls[0]
        self.assertEqual(call["endpoint"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-test-key")
        self.assertEqual(call["json"]["model"], "deepseek-v4-flash")
        self.assertEqual(call["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(call["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(call["json"]["reasoning_effort"], "high")
        self.assertFalse(call["json"]["stream"])
        self.assertEqual(
            json.loads(call["json"]["messages"][1]["content"]), task
        )

    def test_api_and_network_errors_redact_api_key(self) -> None:
        key = "secret-test-key"
        task = {"record_sha256": "abc"}
        response = FakeResponse(
            status_code=401,
            text=f'{{"error":"Authorization Bearer {key}"}}',
        )
        client = DeepSeekRepairClient(
            DeepSeekConfig(api_key=key), session=FakeSession(response=response)
        )
        with self.assertRaises(DeepSeekAPIError) as caught:
            client.propose_patch(task)
        self.assertNotIn(key, str(caught.exception))
        self.assertIn("<redacted>", str(caught.exception))
        self.assertIn("status=401", str(caught.exception))

        client = DeepSeekRepairClient(
            DeepSeekConfig(api_key=key),
            session=FakeSession(error=OSError(f"connection failed for {key}")),
        )
        with self.assertRaises(DeepSeekAPIError) as caught:
            client.propose_patch(task)
        self.assertNotIn(key, str(caught.exception))
        self.assertIn("<redacted>", str(caught.exception))

    def test_record_hash_is_canonical_and_context_hash_is_enforced(self) -> None:
        record = self.make_record()
        reordered = {key: record[key] for key in reversed(record)}
        self.assertEqual(record_sha256(record), record_sha256(reordered))
        changed = copy.deepcopy(record)
        changed["source_code"] = "changed"
        self.assertNotEqual(record_sha256(record), record_sha256(changed))

        patch = self.patch(record, self.swap_operation(record))
        patch.record_sha256 = "0" * 64
        self.assertIn(
            "record_sha256 mismatch",
            validate_patch_context(patch, record, self.schema),
        )

    def test_evidence_relation_index_and_confidence_are_enforced(self) -> None:
        record = self.make_record()
        bad_evidence = RepairOperation(
            op="swap_relation_endpoints",
            relation_index=0,
            expected_relation_type="is_core_symptom_of",
            evidence=Evidence(quote="wrong", start=0, end=5),
            reason="bad evidence",
            confidence=0.99,
        )
        errors = validate_patch_context(
            self.patch(record, bad_evidence), record, self.schema
        )
        self.assertTrue(any("evidence offsets do not match" in error for error in errors))

        wrong_index = self.swap_operation(record)
        wrong_index.relation_index = 4
        errors = validate_patch_context(
            self.patch(record, wrong_index), record, self.schema
        )
        self.assertTrue(any("relation_index out of range" in error for error in errors))

        wrong_expected = self.swap_operation(record)
        wrong_expected.expected_relation_type = "subsumes"
        errors = validate_patch_context(
            self.patch(record, wrong_expected), record, self.schema
        )
        self.assertTrue(
            any("expected_relation_type mismatch" in error for error in errors)
        )

        low_confidence = self.patch(
            record, self.swap_operation(record, confidence=0.5)
        )
        errors = validate_patch_context(low_confidence, record, self.schema)
        self.assertTrue(any("confidence below" in error for error in errors))

    def test_protected_main_disease_and_illegal_relations_are_rejected(self) -> None:
        record = self.make_record()
        relabel_main = RepairOperation(
            op="relabel_entity",
            source_entity_id="D1",
            replacement_label="Symptom",
            evidence=self.evidence(record, "Canonical Disorder"),
            reason="unsafe",
            confidence=0.99,
        )
        errors = validate_patch_context(
            self.patch(record, relabel_main), record, self.schema
        )
        self.assertTrue(any("canonical main disease" in error for error in errors))

        unknown_relation = RepairOperation(
            op="replace_relation_type",
            relation_index=0,
            expected_relation_type="is_core_symptom_of",
            replacement_relation_type="invented_relation",
            evidence=self.evidence(record),
            reason="not in schema",
            confidence=0.99,
        )
        errors = validate_patch_context(
            self.patch(record, unknown_relation), record, self.schema
        )
        self.assertTrue(any("not in schema" in error for error in errors))

        wrong_domain = RepairOperation(
            op="add_relation",
            source_entity_id="D1",
            target_entity_id="S1",
            replacement_relation_type="is_core_symptom_of",
            evidence=self.evidence(record),
            reason="wrong domain",
            confidence=0.99,
        )
        errors = validate_patch_context(
            self.patch(record, wrong_domain), record, self.schema
        )
        self.assertTrue(any("domain/range mismatch" in error for error in errors))

    def test_valid_patch_removes_error_without_mutating_original(self) -> None:
        record = self.make_record()
        before = validate_record(record, self.schema)
        self.assertEqual(
            {error["code"] for error in before["errors"]},
            {"invalid_relation_domain_range"},
        )
        patch = self.patch(record, self.swap_operation(record))
        self.assertEqual(validate_patch_context(patch, record, self.schema), [])

        repaired, audit = apply_repair_patch(record, patch, self.schema)

        self.assertEqual(record["output"]["relations"][0]["source"], "D1")
        self.assertEqual(repaired["output"]["relations"][0]["source"], "S1")
        self.assertEqual(repaired["output"]["relations"][0]["target"], "D1")
        self.assertEqual(audit["validation_after"]["errors"], [])
        self.assertEqual(repaired["migration"]["status"], "repaired")

    def test_replace_relation_can_fix_type_and_endpoints_atomically(self) -> None:
        record = self.make_record()
        operation = RepairOperation(
            op="replace_relation",
            relation_index=0,
            expected_relation_type="is_core_symptom_of",
            source_entity_id="S1",
            target_entity_id="D1",
            replacement_relation_type="is_core_symptom_of",
            evidence=self.evidence(record),
            reason="Replace the invalid endpoint orientation in one guarded operation.",
            confidence=0.99,
        )
        patch = self.patch(record, operation)
        self.assertEqual(validate_patch_context(patch, record, self.schema), [])
        repaired, _ = apply_repair_patch(record, patch, self.schema)
        self.assertEqual(
            repaired["output"]["relations"][0],
            {
                **record["output"]["relations"][0],
                "source": "S1",
                "target": "D1",
            },
        )

    def test_manual_review_may_reference_relation_without_mutating_it(self) -> None:
        record = self.make_record()
        operation = RepairOperation(
            op="mark_manual_review",
            relation_index=0,
            reason="The source text does not determine a safe correction.",
            confidence=0.0,
        )
        patch = self.patch(
            record,
            operation,
            needs_human_review=True,
            review_reason="Ambiguous relation semantics",
        )
        self.assertEqual(validate_patch_context(patch, record, self.schema), [])

    def test_equal_error_count_with_a_new_error_is_rejected(self) -> None:
        record = self.make_record()
        relabel_symptom = RepairOperation(
            op="relabel_entity",
            source_entity_id="S1",
            replacement_label="Disease",
            evidence=self.evidence(record),
            reason="trades one validation error for another",
            confidence=0.99,
        )
        patch = self.patch(record, relabel_symptom)
        with self.assertRaisesRegex(PatchValidationError, "introduced new schema errors"):
            apply_repair_patch(record, patch, self.schema)

    def test_cache_key_changes_with_task_or_model(self) -> None:
        record = self.make_record()
        task = build_repair_task(record, self.schema, validate_record(record, self.schema))
        root = Path("cache")
        first = cached_patch_path(root, task, "deepseek-v4-flash")
        self.assertEqual(first, cached_patch_path(root, task, "deepseek-v4-flash"))
        changed = copy.deepcopy(task)
        changed["record_sha256"] = "different"
        self.assertNotEqual(first, cached_patch_path(root, changed, "deepseek-v4-flash"))
        self.assertNotEqual(first, cached_patch_path(root, task, "another-model"))

    def _prepare_cached_cli_case(
        self,
        directory: Path,
        *,
        valid: bool,
    ) -> tuple[Path, Path, Path]:
        record = self.make_record()
        input_path = directory / "input.json"
        report_path = directory / "report.json"
        cache_dir = directory / "cache"
        write_json(input_path, [record])
        findings = validate_record(record, self.schema)
        task = build_repair_task(record, self.schema, findings)
        operation = self.swap_operation(record, confidence=0.99 if valid else 0.1)
        patch = self.patch(record, operation)
        cache_path = cached_patch_path(cache_dir, task, "deepseek-v4-flash")
        save_safe_api_result(
            cache_path,
            {
                "model": "deepseek-v4-flash",
                "usage": {"cached": True},
                "finish_reason": "stop",
                "patch": patch.model_dump(mode="json"),
            },
        )
        return input_path, report_path, cache_dir

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("DEEPSEEK_API_KEY", None)
        environment.pop("DEEPSEEK_MODEL", None)
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/repair_schema_v2_deepseek.py"), *args],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_cli_cache_hit_needs_no_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path, report_path, cache_dir = self._prepare_cached_cli_case(
                directory, valid=True
            )
            completed = self._run_cli(
                "--input",
                str(input_path),
                "--report",
                str(report_path),
                "--cache-dir",
                str(cache_dir),
                "--env-file",
                str(directory / "missing.env"),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = load_json(report_path)
            self.assertEqual(report["result_counts"], {"valid_proposal": 1})
            self.assertEqual(report["results"][0]["cache"], "hit")

    def test_cli_rejects_overwrite_and_does_not_write_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            input_path, report_path, cache_dir = self._prepare_cached_cli_case(
                directory, valid=False
            )
            overwrite = self._run_cli(
                "--input",
                str(input_path),
                "--output",
                str(input_path),
                "--report",
                str(report_path),
                "--env-file",
                str(directory / "missing.env"),
                "--apply",
            )
            self.assertNotEqual(overwrite.returncode, 0)
            self.assertIn("refusing to overwrite input", overwrite.stderr)

            output_path = directory / "repaired.json"
            rejected = self._run_cli(
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--report",
                str(report_path),
                "--cache-dir",
                str(cache_dir),
                "--env-file",
                str(directory / "missing.env"),
                "--apply",
            )
            self.assertEqual(rejected.returncode, 1, rejected.stderr)
            self.assertFalse(output_path.exists())
            self.assertIn("did not write partial output", rejected.stderr)
            report = load_json(report_path)
            self.assertEqual(report["result_counts"], {"rejected_context": 1})


if __name__ == "__main__":
    unittest.main()

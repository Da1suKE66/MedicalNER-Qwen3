"""Read-only, fail-closed audit for final Schema v2 training data."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, Iterable

from .convert_schema_v2_to_llamafactory import (
    evaluate_training_eligibility,
    unverified_code_findings,
    validate_split_integrity,
    verify_evidence_span,
)
from .evidence_repair import span_is_exact
from .schema_v2 import TOKEN_RE, correct_text, graph_from_record, validate_record


RECORD_GATE_NAMES = (
    "schema_validation",
    "canonical_main_identity",
    "icd_verification",
    "active_relation_schema",
    "exact_evidence_spans",
    "known_word_corruption",
    "training_eligibility",
)
IDENTITY_ERROR_CODES = {
    "graph_source_identity_mismatch",
    "main_disease_unresolved",
    "main_disease_coding_mismatch",
}
RELATION_ERROR_CODES = {
    "relations_not_list",
    "relation_not_object",
    "missing_relation_field",
    "ghost_relation_endpoint",
    "undefined_relation",
    "invalid_relation_domain_range",
}


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)
    elif isinstance(value, str):
        yield value


def known_word_corruption_findings(
    value: Any, corrections: dict[str, str]
) -> list[dict[str, Any]]:
    """Find known nan-to-null corruptions and literal sentinel words in strings."""

    known_counts: Counter[str] = Counter()
    sentinel_counts: Counter[str] = Counter()
    unexpected_null_counts: Counter[str] = Counter()
    for text in _iter_strings(value):
        _, applied = correct_text(text, corrections)
        known_counts.update(applied)
        for token in TOKEN_RE.findall(text):
            lowered = token.lower()
            if lowered in {"nan", "null"}:
                sentinel_counts[lowered] += 1
            elif "null" in lowered and lowered not in corrections:
                unexpected_null_counts[lowered] += 1

    findings: list[dict[str, Any]] = []
    if known_counts:
        findings.append(
            {
                "code": "known_null_word_corruption",
                "tokens": dict(sorted(known_counts.items())),
                "occurrences": sum(known_counts.values()),
            }
        )
    if sentinel_counts:
        findings.append(
            {
                "code": "literal_nan_or_null_word",
                "tokens": dict(sorted(sentinel_counts.items())),
                "occurrences": sum(sentinel_counts.values()),
            }
        )
    if unexpected_null_counts:
        findings.append(
            {
                "code": "unexpected_word_containing_null",
                "tokens": dict(sorted(unexpected_null_counts.items())),
                "occurrences": sum(unexpected_null_counts.values()),
            }
        )
    return findings


def _canonical_identity_findings(
    validation: dict[str, Any], record: dict[str, Any]
) -> list[dict[str, Any]]:
    findings = []
    for error in validation["errors"]:
        code = error.get("code")
        if code in IDENTITY_ERROR_CODES:
            findings.append(error)
        elif code == "missing_record_field" and error.get("field") in {
            "source_record_id",
            "source_code",
            "source_title",
            "source_release",
        }:
            findings.append(error)
    if record.get("schema_version") in (None, ""):
        findings.append({"code": "canonical_schema_version_missing"})
    return findings


def _icd_findings(
    record: dict[str, Any], graph: dict[str, Any] | None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    validation = record.get("icd_validation")
    if not isinstance(validation, dict):
        findings.append({"code": "icd_validation_missing"})
    else:
        if validation.get("status") != "verified":
            findings.append(
                {
                    "code": "icd_validation_not_verified",
                    "actual": validation.get("status"),
                }
            )
        if validation.get("errors"):
            findings.append(
                {
                    "code": "icd_validation_has_errors",
                    "errors": validation.get("errors"),
                }
            )
    migration = record.get("migration")
    pending = migration.get("unverified_codes") if isinstance(migration, dict) else None
    if pending:
        findings.append(
            {
                "code": "migration_has_unverified_codes",
                "count": len(pending) if isinstance(pending, list) else 1,
            }
        )
    if graph is not None:
        code_findings = unverified_code_findings(record, graph)
        if code_findings:
            findings.append(
                {
                    "code": "graph_has_unverified_codes",
                    "findings": code_findings,
                }
            )
    return findings


def _relation_findings(
    graph: dict[str, Any] | None,
    validation: dict[str, Any],
    schema: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if graph is None:
        return [{"code": "canonical_graph_missing"}]
    relations = graph.get("relations")
    if not isinstance(relations, list):
        return [{"code": "relations_not_list"}]
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            findings.append({"code": "relation_not_object", "relation_index": index})
            continue
        relation_name = str(relation.get("relation") or "")
        spec = schema["relation_types"].get(relation_name)
        if spec is None:
            findings.append(
                {
                    "code": "undefined_relation",
                    "relation_index": index,
                    "relation": relation_name,
                }
            )
        elif spec.get("status") != "active":
            findings.append(
                {
                    "code": "relation_not_active",
                    "relation_index": index,
                    "relation": relation_name,
                    "status": spec.get("status"),
                }
            )
    findings.extend(
        error
        for error in validation["errors"]
        if error.get("code") in RELATION_ERROR_CODES
    )
    findings.extend(
        warning
        for warning in validation["warnings"]
        if warning.get("code") == "relation_semantics_need_medical_review"
    )
    return findings


def _evidence_findings(
    record: dict[str, Any], graph: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if graph is None:
        return [{"code": "canonical_graph_missing"}]
    source_text = record.get("input")
    if not isinstance(source_text, str):
        return [{"code": "record_input_not_string"}]
    relations = graph.get("relations")
    if not isinstance(relations, list):
        return [{"code": "relations_not_list"}]
    findings: list[dict[str, Any]] = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            findings.append({"code": "relation_not_object", "relation_index": index})
            continue
        valid, reason = verify_evidence_span(
            relation.get("evidence_span"), source_text
        )
        if valid and not span_is_exact(
            relation.get("evidence_span"), source_text, relation.get("evidence")
        ):
            valid, reason = False, "retained_evidence_text_mismatch"
        if not valid:
            findings.append(
                {
                    "code": "relation_evidence_span_not_exact",
                    "relation_index": index,
                    "reason": reason,
                }
            )
    return findings


def _prepare_manifest_for_expected_split(
    split_manifest: dict[str, Any], expected_split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = copy.deepcopy(split_manifest)
    source_ids = manifest.get("source_ids")
    findings: list[dict[str, Any]] = []
    if not isinstance(source_ids, dict):
        return manifest, [{"code": "split_source_ids_missing"}]

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        findings.append({"code": "split_counts_missing_or_invalid"})
        counts = {}
    seen: dict[str, str] = {}
    all_ids: list[str] = []
    for split, values in source_ids.items():
        if not isinstance(values, list):
            findings.append(
                {"code": "split_source_ids_not_list", "split": split}
            )
            continue
        expected_count = counts.get(split)
        if expected_count is not None and expected_count != len(values):
            findings.append(
                {
                    "code": "split_count_mismatch",
                    "split": split,
                    "actual": len(values),
                    "expected": expected_count,
                }
            )
        for value in values:
            source_id = str(value)
            if source_id in seen:
                findings.append(
                    {
                        "code": "source_id_in_multiple_splits",
                        "source_record_id": source_id,
                        "splits": [seen[source_id], split],
                    }
                )
            else:
                seen[source_id] = str(split)
                all_ids.append(source_id)
    if expected_split == "all":
        source_ids["all"] = all_ids
    return manifest, findings


def _split_findings(
    records: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    schema: dict[str, Any],
    expected_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    manifest, findings = _prepare_manifest_for_expected_split(
        split_manifest, expected_split
    )
    if manifest.get("schema_version") != schema.get("schema_version"):
        findings.append(
            {
                "code": "split_schema_version_mismatch",
                "actual": manifest.get("schema_version"),
                "expected": schema.get("schema_version"),
            }
        )
    integrity = None
    try:
        integrity = validate_split_integrity(records, manifest, expected_split)
    except (KeyError, TypeError, ValueError) as exc:
        findings.append(
            {
                "code": "split_integrity_failed",
                "error": str(exc),
            }
        )
    return findings, integrity


def audit_training_data(
    records: list[Any],
    split_manifest: dict[str, Any],
    schema: dict[str, Any],
    *,
    expected_split: str = "all",
) -> dict[str, Any]:
    """Audit final records without mutating them and return a JSON-safe report."""

    gate_failures: Counter[str] = Counter()
    gate_finding_counts: Counter[str] = Counter()
    findings: list[dict[str, Any]] = []
    relation_count = 0

    strict_json_findings: list[dict[str, Any]] = []
    try:
        json.dumps(records, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        strict_json_findings.append(
            {"code": "records_not_strict_json", "error": str(exc)}
        )

    typed_records: list[dict[str, Any]] = []
    for record_index, raw_record in enumerate(records):
        record_failures: dict[str, list[dict[str, Any]]] = {}
        if not isinstance(raw_record, dict):
            failure = {
                "code": "record_not_object",
                "actual_type": type(raw_record).__name__,
            }
            for gate in RECORD_GATE_NAMES:
                record_failures[gate] = [failure]
        else:
            record = raw_record
            typed_records.append(record)
            validation = validate_record(record, schema)
            schema_findings = [
                *validation["errors"],
                *validation["warnings"],
            ]
            if schema_findings:
                record_failures["schema_validation"] = schema_findings

            identity_findings = _canonical_identity_findings(validation, record)
            if identity_findings:
                record_failures["canonical_main_identity"] = identity_findings

            graph = graph_from_record(record)
            if isinstance(graph, dict) and isinstance(graph.get("relations"), list):
                relation_count += len(graph["relations"])

            icd_findings = _icd_findings(record, graph)
            if icd_findings:
                record_failures["icd_verification"] = icd_findings

            relation_findings = _relation_findings(graph, validation, schema)
            if relation_findings:
                record_failures["active_relation_schema"] = relation_findings

            evidence_findings = _evidence_findings(record, graph)
            if evidence_findings:
                record_failures["exact_evidence_spans"] = evidence_findings

            # Legacy CoT/audit metadata is never used as a training target. Scan the
            # canonical training surfaces so stale narrative text cannot create a
            # false positive while damaged input/entity/relation text is still gated.
            corruption_scope = {
                "input": record.get("input"),
                "source_title": record.get("source_title"),
                "graph": graph,
            }
            corruption_findings = known_word_corruption_findings(
                corruption_scope, schema["text_corrections"]
            )
            if corruption_findings:
                record_failures["known_word_corruption"] = corruption_findings

            eligibility = evaluate_training_eligibility(record, schema)
            if not eligibility["eligible"]:
                record_failures["training_eligibility"] = eligibility[
                    "excluded_reasons"
                ]

        if record_failures:
            for gate, gate_rows in record_failures.items():
                gate_failures[gate] += 1
                gate_finding_counts[gate] += len(gate_rows)
            findings.append(
                {
                    "record_index": record_index,
                    "source_record_id": (
                        raw_record.get("source_record_id")
                        if isinstance(raw_record, dict)
                        else None
                    ),
                    "failures": record_failures,
                }
            )

    split_findings: list[dict[str, Any]]
    split_integrity: dict[str, Any] | None
    if len(typed_records) != len(records):
        split_findings = [{"code": "split_not_checked_due_to_malformed_records"}]
        split_integrity = None
    else:
        split_findings, split_integrity = _split_findings(
            typed_records,
            split_manifest,
            schema,
            expected_split,
        )

    gates: dict[str, dict[str, Any]] = {
        "strict_json": {
            "passed": not strict_json_findings,
            "finding_count": len(strict_json_findings),
            "findings": strict_json_findings,
        }
    }
    for gate in RECORD_GATE_NAMES:
        gates[gate] = {
            "passed": gate_failures[gate] == 0,
            "records_checked": len(records),
            "failed_records": gate_failures[gate],
            "finding_count": gate_finding_counts[gate],
        }
    gates["split_integrity"] = {
        "passed": not split_findings,
        "finding_count": len(split_findings),
        "findings": split_findings,
        "details": split_integrity,
    }
    failed_gates = [name for name, gate in gates.items() if not gate["passed"]]
    return {
        "schema_version": schema.get("schema_version"),
        "record_count": len(records),
        "relation_count": relation_count,
        "expected_split": expected_split,
        "passed": not failed_gates,
        "failed_gates": failed_gates,
        "gates": gates,
        "records_with_findings": len(findings),
        "record_findings": findings,
    }

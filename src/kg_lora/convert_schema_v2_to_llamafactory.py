#!/usr/bin/env python3
"""Convert only training-eligible Schema v2 records to LLaMAFactory messages.

This converter is deliberately separate from the legacy CoT converter.  It never
copies historical reasoning traces: the assistant target is the canonical Schema v2
graph serialized as strict JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .schema_v2 import (
    graph_from_record,
    load_json,
    load_schema,
    normalize_text,
    record_list,
    sha256_file,
    validate_record,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MIGRATION_STATUSES = frozenset({"approved", "repaired"})
VERIFIED_CODE_STATUSES = frozenset({"approved", "verified"})


def source_id_for_record(record: dict[str, Any]) -> str:
    value = record.get("source_record_id") or record.get("source_id")
    return str(value) if value else ""


def _reason(code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, **details}


def verify_evidence_span(span: Any, source_text: str) -> tuple[bool, str]:
    """Verify that a relation span is an exact, reproducible slice of record.input."""

    if not isinstance(span, dict):
        return False, "span_not_object"
    if span.get("basis") != "record.input":
        return False, "basis_not_record_input"
    if span.get("verified") is False:
        return False, "explicitly_unverified"
    verification_status = span.get("verification_status")
    if verification_status is not None and str(verification_status).lower() not in {
        "approved",
        "verified",
    }:
        return False, "verification_status_not_verified"

    start = span.get("start")
    end = span.get("end")
    text = span.get("text")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        return False, "offset_not_integer"
    if start < 0 or end <= start or end > len(source_text):
        return False, "offset_out_of_bounds"
    if not isinstance(text, str) or not text:
        return False, "span_text_missing"
    if source_text[start:end] != text:
        return False, "span_text_mismatch"
    return True, "exact_record_input_slice"


def _explicitly_verified_codes(record: dict[str, Any]) -> set[tuple[str, str]]:
    """Collect entity/code pairs that have an explicit approved verification entry."""

    entries: list[Any] = []
    migration = record.get("migration")
    if isinstance(migration, dict):
        entries.extend(migration.get("verified_codes") or [])
    entries.extend(record.get("code_verifications") or [])

    verified: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or entry.get("verification_status") or "").lower()
        entity_id = str(entry.get("entity_id") or "")
        code = str(entry.get("code") or entry.get("icdcode") or "")
        if status in VERIFIED_CODE_STATUSES and entity_id and code:
            verified.add((entity_id, code))
    return verified


def unverified_code_findings(
    record: dict[str, Any], graph: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return every code that lacks canonical source identity or explicit approval."""

    findings: list[dict[str, Any]] = []
    migration = record.get("migration")
    if isinstance(migration, dict):
        pending = migration.get("unverified_codes") or []
        if pending:
            findings.append(
                {
                    "source": "migration.unverified_codes",
                    "count": len(pending),
                }
            )

    explicitly_verified = _explicitly_verified_codes(record)
    expected_code = str(record.get("source_code") or "")
    expected_system = "ICD-11-MMS"
    expected_release = str(record.get("source_release") or "")
    expected_uri = str(record.get("source_record_id") or "")

    for entity in graph.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = str(entity.get("id") or "")
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            continue
        code = str(properties.get("icdcode") or "").strip()
        if not code:
            continue
        canonical_source_code = (
            entity.get("label") == "Disease"
            and normalize_text(entity.get("name"))
            == normalize_text(record.get("source_title"))
            and code == expected_code
            and properties.get("coding_system") == expected_system
            and str(properties.get("icd_release") or "") == expected_release
            and str(properties.get("icd_uri") or "") == expected_uri
        )
        if not canonical_source_code and (entity_id, code) not in explicitly_verified:
            findings.append(
                {
                    "source": "entity.properties.icdcode",
                    "entity_id": entity_id,
                    "code": code,
                    "reason": "not_canonical_source_code_or_explicitly_verified",
                }
            )
    return findings


def _canonical_main_entities(
    record: dict[str, Any], graph: dict[str, Any]
) -> list[dict[str, Any]]:
    entities = graph.get("entities")
    if not isinstance(entities, list):
        return []
    source_title = normalize_text(record.get("source_title"))
    return [
        entity
        for entity in entities
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name")) == source_title
    ]


def _main_entity_is_canonical(
    record: dict[str, Any], entity: dict[str, Any]
) -> bool:
    properties = entity.get("properties")
    return bool(
        isinstance(properties, dict)
        and str(properties.get("icdcode") or "")
        == str(record.get("source_code") or "")
        and properties.get("coding_system") == "ICD-11-MMS"
        and str(properties.get("icd_release") or "")
        == str(record.get("source_release") or "")
        and str(properties.get("icd_uri") or "")
        == str(record.get("source_record_id") or "")
    )


def entity_name_is_text_grounded(source_text: str, entity_name: Any) -> bool:
    """Match an entity name exactly or after case/whitespace normalization."""

    if not isinstance(entity_name, str) or not entity_name.strip():
        return False

    def bounded_substring(haystack: str, needle: str) -> bool:
        start = haystack.find(needle)
        while start >= 0:
            end = start + len(needle)
            left_ok = (
                start == 0
                or not needle[0].isalnum()
                or not haystack[start - 1].isalnum()
            )
            right_ok = (
                end == len(haystack)
                or not needle[-1].isalnum()
                or not haystack[end].isalnum()
            )
            if left_ok and right_ok:
                return True
            start = haystack.find(needle, start + 1)
        return False

    if bounded_substring(source_text, entity_name):
        return True

    normalized_name = " ".join(entity_name.casefold().split())
    normalized_source = " ".join(source_text.casefold().split())
    if not normalized_name or not normalized_source:
        return False
    return bounded_substring(normalized_source, normalized_name)


def evaluate_training_eligibility(
    record: dict[str, Any], schema: dict[str, Any]
) -> dict[str, Any]:
    """Apply the strict Schema v2 training gate to one record."""

    exclusions: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    migration = record.get("migration")
    status = migration.get("status") if isinstance(migration, dict) else None
    if status not in ALLOWED_MIGRATION_STATUSES:
        exclusions.append(
            _reason(
                "status_not_trainable",
                actual=status,
                allowed=sorted(ALLOWED_MIGRATION_STATUSES),
            )
        )
    else:
        approvals.append(_reason("status_trainable", actual=status))

    if record.get("schema_version") != schema.get("schema_version"):
        exclusions.append(
            _reason(
                "schema_version_mismatch",
                actual=record.get("schema_version"),
                expected=schema.get("schema_version"),
            )
        )

    icd_validation = record.get("icd_validation")
    icd_status = (
        icd_validation.get("status") if isinstance(icd_validation, dict) else None
    )
    if icd_status != "verified":
        exclusions.append(
            _reason(
                "icd_validation_not_verified",
                actual=icd_status,
                errors=(
                    icd_validation.get("errors")
                    if isinstance(icd_validation, dict)
                    else None
                ),
            )
        )
    else:
        approvals.append(_reason("icd_validation_verified"))

    source_text = record.get("input")
    if not isinstance(source_text, str) or not source_text.strip():
        exclusions.append(_reason("missing_input_text"))
        source_text = "" if source_text is None else str(source_text)

    graph = graph_from_record(record)
    relation_count = 0
    verified_span_count = 0
    validation = validate_record(record, schema)
    if validation["errors"]:
        exclusions.append(
            _reason("schema_validation_errors", findings=validation["errors"])
        )
    else:
        approvals.append(_reason("schema_validation_zero_errors"))
    if validation["warnings"]:
        exclusions.append(
            _reason("schema_validation_warnings", findings=validation["warnings"])
        )
    else:
        approvals.append(_reason("schema_validation_zero_warnings"))

    if not isinstance(graph, dict):
        exclusions.append(_reason("canonical_graph_missing"))
    else:
        entities = graph.get("entities")
        if not isinstance(entities, list) or not entities:
            exclusions.append(_reason("canonical_graph_empty"))
            entities = []

        main_entities = _canonical_main_entities(record, graph)
        if not main_entities:
            exclusions.append(_reason("canonical_main_entity_missing"))
            main_entity_id = ""
        elif len(main_entities) > 1:
            exclusions.append(
                _reason(
                    "canonical_main_entity_not_unique",
                    entity_ids=[str(entity.get("id") or "") for entity in main_entities],
                )
            )
            main_entity_id = ""
        else:
            main_entity_id = str(main_entities[0].get("id") or "")
            if not main_entity_id or not _main_entity_is_canonical(
                record, main_entities[0]
            ):
                exclusions.append(
                    _reason(
                        "canonical_main_entity_not_canonical",
                        entity_id=main_entity_id,
                    )
                )
            else:
                approvals.append(
                    _reason(
                        "canonical_main_entity_verified",
                        entity_id=main_entity_id,
                    )
                )

        code_findings = unverified_code_findings(record, graph)
        if code_findings:
            exclusions.append(
                _reason("unverified_code", findings=code_findings)
            )
        else:
            approvals.append(_reason("all_codes_verified"))

        missing_spans: list[int] = []
        invalid_spans: list[dict[str, Any]] = []
        inactive_relations: list[dict[str, Any]] = []
        retained_relation_entity_ids: set[str] = set()
        relations = graph.get("relations")
        if not isinstance(relations, list):
            exclusions.append(_reason("relations_not_list"))
            relations = []
        relation_count = len(relations)
        for index, relation in enumerate(relations):
            if not isinstance(relation, dict):
                invalid_spans.append({"relation_index": index, "reason": "relation_not_object"})
                continue
            relation_name = str(relation.get("relation") or "")
            relation_spec = schema.get("relation_types", {}).get(relation_name)
            relation_is_active = bool(
                isinstance(relation_spec, dict)
                and relation_spec.get("status") == "active"
            )
            if not relation_is_active:
                inactive_relations.append(
                    {
                        "relation_index": index,
                        "relation": relation_name,
                        "status": (
                            relation_spec.get("status")
                            if isinstance(relation_spec, dict)
                            else "undefined"
                        ),
                    }
                )
            if "evidence_span" not in relation:
                missing_spans.append(index)
                continue
            valid, reason = verify_evidence_span(relation.get("evidence_span"), source_text)
            retained_evidence = relation.get("evidence")
            if valid and "evidence" in relation and not isinstance(retained_evidence, str):
                valid, reason = False, "retained_evidence_not_string"
            if (
                valid
                and isinstance(retained_evidence, str)
                and retained_evidence.strip()
                != relation["evidence_span"]["text"]
            ):
                valid, reason = False, "retained_evidence_text_mismatch"
            if valid:
                verified_span_count += 1
                if relation_is_active:
                    retained_relation_entity_ids.update(
                        {
                            str(relation.get("source") or ""),
                            str(relation.get("target") or ""),
                        }
                    )
            else:
                invalid_spans.append({"relation_index": index, "reason": reason})
        if inactive_relations:
            exclusions.append(
                _reason("relation_not_active", findings=inactive_relations)
            )
        else:
            approvals.append(
                _reason("all_relations_active", relation_count=relation_count)
            )
        if missing_spans:
            exclusions.append(
                _reason(
                    "missing_verified_evidence_span",
                    relation_indexes=missing_spans,
                )
            )
        if invalid_spans:
            exclusions.append(
                _reason("invalid_evidence_span", findings=invalid_spans)
            )
        if not missing_spans and not invalid_spans:
            approvals.append(
                _reason(
                    "all_relation_evidence_spans_verified",
                    relation_count=relation_count,
                    verified_span_count=verified_span_count,
                )
            )

        ungrounded_entities: list[dict[str, Any]] = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("id") or "")
            if main_entity_id and entity_id == main_entity_id:
                continue
            if entity_name_is_text_grounded(source_text, entity.get("name")):
                continue
            if entity_id and entity_id in retained_relation_entity_ids:
                continue
            ungrounded_entities.append(
                {
                    "entity_index": index,
                    "entity_id": entity_id,
                    "label": entity.get("label"),
                    "name": entity.get("name"),
                }
            )
        if ungrounded_entities:
            exclusions.append(
                _reason(
                    "ungrounded_non_main_entity",
                    findings=ungrounded_entities,
                )
            )
        else:
            approvals.append(
                _reason(
                    "all_non_main_entities_grounded",
                    by_text_or_verified_relation=True,
                )
            )

        try:
            json.dumps(graph, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            exclusions.append(
                _reason("canonical_graph_not_strict_json", error=str(exc))
            )

    return {
        "eligible": not exclusions,
        "status": status,
        "source_record_id": source_id_for_record(record),
        "relation_count": relation_count,
        "verified_evidence_span_count": verified_span_count,
        "eligible_reasons": approvals,
        "excluded_reasons": exclusions,
    }


def build_system_prompt(schema: dict[str, Any]) -> str:
    """Build a compact prompt directly from the reviewed Schema v2 contract."""

    entity_labels = ", ".join(sorted(schema.get("entity_types", {})))
    active_relations = ", ".join(
        sorted(
            name
            for name, spec in schema.get("relation_types", {}).items()
            if spec.get("status") == "active"
        )
    )
    return (
        "Extract a medical knowledge graph that conforms to Schema v2 "
        f"{schema.get('schema_version')}. Return strict JSON only, with exactly the "
        "top-level keys `entities` and `relations`. Use only explicitly stated "
        "information and ensure every relation endpoint names an emitted entity. "
        f"Allowed entity labels: {entity_labels}. "
        f"Approved relation names: {active_relations}."
    )


def build_user_prompt(source_text: str) -> str:
    return (
        "Extract the canonical Schema v2 graph from the following medical text. "
        "Return the JSON object directly; do not include reasoning, markdown, or "
        f"legacy CoT tags.\n\nMedical text:\n{source_text}"
    )


def build_messages(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Create one LLaMAFactory messages record from canonical v2 output."""

    graph = graph_from_record(record)
    if not isinstance(graph, dict):
        raise ValueError("canonical Schema v2 graph missing")
    assistant_target = {
        "entities": graph.get("entities", []),
        "relations": graph.get("relations", []),
    }
    assistant_json = json.dumps(
        assistant_target,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    return {
        "messages": [
            {"role": "system", "content": build_system_prompt(schema)},
            {"role": "user", "content": build_user_prompt(str(record["input"]))},
            {"role": "assistant", "content": assistant_json},
        ]
    }


def convert_records(
    records: Iterable[dict[str, Any]], schema: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert eligible records and return an auditable gate manifest."""

    converted: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    exclusion_reason_counts: Counter[str] = Counter()
    eligibility_reason_counts: Counter[str] = Counter()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            decision = {
                "record_index": index,
                "source_record_id": "",
                "status": None,
                "eligible": False,
                "eligible_reasons": [],
                "excluded_reasons": [_reason("record_not_object")],
                "relation_count": 0,
                "verified_evidence_span_count": 0,
            }
        else:
            decision = {
                "record_index": index,
                **evaluate_training_eligibility(record, schema),
            }
            if decision["eligible"]:
                converted.append(build_messages(record, schema))

        status_counts[str(decision.get("status") or "missing")] += 1
        for reason in decision["eligible_reasons"]:
            eligibility_reason_counts[reason["code"]] += 1
        for reason in decision["excluded_reasons"]:
            exclusion_reason_counts[reason["code"]] += 1
        decisions.append(decision)

    manifest = {
        "schema_version": schema.get("schema_version"),
        "format": "llamafactory_messages",
        "assistant_target": "canonical_schema_v2_graph_json",
        "legacy_cot_reused": False,
        "strict_json_allow_nan": False,
        "gate": {
            "allowed_migration_statuses": sorted(ALLOWED_MIGRATION_STATUSES),
            "require_icd_validation_status": "verified",
            "require_schema_validation_zero_errors": True,
            "require_schema_validation_zero_warnings": True,
            "reject_unverified_codes": True,
            "require_canonical_main_entity": True,
            "require_non_empty_entity_graph": True,
            "require_active_relations": True,
            "require_exact_evidence_span_for_every_relation": True,
            "require_non_main_entity_grounding": (
                "source_text_exact_or_normalized_whitespace_match_or_verified_relation"
            ),
        },
        "counts": {
            "input": len(decisions),
            "eligible": len(converted),
            "excluded": len(decisions) - len(converted),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_reason_counts": dict(sorted(eligibility_reason_counts.items())),
        "excluded_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "eligibility_decisions": decisions,
    }
    return converted, manifest


def validate_split_integrity(
    records: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    expected_split: str,
) -> dict[str, Any]:
    """Require a leakage-free split manifest and exact split membership."""

    leakage = split_manifest.get("hierarchy_leakage_check")
    if not isinstance(leakage, dict) or leakage.get("passed") is not True:
        raise ValueError(
            "split manifest has no passing hierarchy_leakage_check; rebuild v2 splits"
        )
    if leakage.get("direct_cross_split_edge_count") != 0:
        raise ValueError("split manifest reports direct hierarchy leakage")
    if leakage.get("shared_external_parent_cross_split_count") != 0:
        raise ValueError("split manifest reports shared external-parent leakage")

    expected_ids = split_manifest.get("source_ids", {}).get(expected_split)
    if not isinstance(expected_ids, list):
        raise ValueError(f"split manifest has no source_ids.{expected_split}")
    actual_ids = [source_id_for_record(record) for record in records]
    if any(not source_id for source_id in actual_ids):
        raise ValueError("input contains a record without source_record_id/source_id")
    duplicate_ids = sorted(
        source_id for source_id, count in Counter(actual_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"input contains duplicate source IDs: {duplicate_ids[:5]}")
    if set(actual_ids) != set(map(str, expected_ids)):
        missing = sorted(set(map(str, expected_ids)) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(map(str, expected_ids)))
        raise ValueError(
            f"input does not match {expected_split} split: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    return {
        "passed": True,
        "expected_split": expected_split,
        "record_count": len(actual_ids),
        "direct_cross_split_edge_count": 0,
        "shared_external_parent_cross_split_count": 0,
        "source_ids_match_manifest": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data/schema_v2/splits/train_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/llamafactory/schema_v2_train_llamafactory.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/llamafactory/schema_v2_train_manifest.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPO_ROOT / "data/schema_v2/splits/split_manifest.json",
    )
    parser.add_argument("--expected-split", default="train")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every gate and print counts without writing output files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = load_schema()
    records = record_list(load_json(args.input))
    split_manifest = load_json(args.split_manifest)
    split_integrity = validate_split_integrity(
        records, split_manifest, args.expected_split
    )
    converted, manifest = convert_records(records, schema)
    manifest.update(
        {
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "split_manifest_path": str(args.split_manifest),
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "split_integrity": split_integrity,
            "dry_run": bool(args.dry_run),
        }
    )

    if not converted:
        raise ValueError("training gate rejected every record; refusing empty dataset")

    if not args.dry_run:
        write_json(args.output, converted)
        manifest["output_path"] = str(args.output)
        manifest["output_sha256"] = sha256_file(args.output)
        write_json(args.manifest, manifest)
        print(f"wrote {len(converted)} records to {args.output}")
        print(f"wrote gate manifest to {args.manifest}")

    counts = manifest["counts"]
    print(
        f"training_gate input={counts['input']} eligible={counts['eligible']} "
        f"excluded={counts['excluded']}"
    )
    print(f"excluded_reasons={manifest['excluded_reason_counts']}")
    print("split_integrity=passed hierarchy_leakage=0 shared_parent_leakage=0")
    if args.dry_run:
        print("dry_run=true no files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

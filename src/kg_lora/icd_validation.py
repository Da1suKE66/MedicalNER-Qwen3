"""Apply an audited WHO ICD validation report to Schema v2 records.

This is a fail-closed boundary: a canonical disease is ``verified`` only when
one explicit successful WHO result agrees on code, localized title, MMS URI,
release, and language, and no API error exists for that code.  A non-main code
is not attached to the main-disease-only graph: a confirmed title mismatch is
therefore audited as a rejected association instead of blocking the record.
Transient API failures and contradictory reports never clear pending work.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any, Iterable
from urllib.parse import urlparse

from .schema_v2 import graph_from_record, normalize_text, validate_record


_DISCARDED_AUDIT_KEYS = frozenset(
    {
        "action",
        "api_errors",
        "api_version",
        "code",
        "entity_name",
        "reason",
        "title_matches",
        "title_comparison",
        "validation_errors",
        "who_entity_uri",
        "who_release",
        "who_title",
    }
)


def _uri_identity(value: Any) -> str:
    parsed = urlparse(str(value or ""))
    return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"


def _is_who_mms_uri(value: Any, release: str) -> bool:
    parsed = urlparse(str(value or ""))
    return (
        parsed.scheme.lower() in {"http", "https"}
        and parsed.hostname == "id.who.int"
        and parsed.path.startswith(f"/icd/release/11/{release}/mms/")
    )


def _normalize_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _report_indexes(
    report: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    set[str],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Index the report without allowing duplicate or malformed rows to vanish."""

    results: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    errors_by_code: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    global_issues: list[dict[str, Any]] = []

    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        global_issues.append({"code": "who_report_results_not_list"})
        raw_results = []
    for index, result in enumerate(raw_results):
        if not isinstance(result, dict):
            global_issues.append(
                {"code": "who_report_result_not_object", "result_index": index}
            )
            continue
        requested_code = _normalize_code(result.get("requested_code"))
        if not requested_code:
            global_issues.append(
                {"code": "who_report_result_requested_code_missing", "result_index": index}
            )
            continue
        if requested_code in results:
            duplicates.add(requested_code)
            continue
        results[requested_code] = result

    raw_errors = report.get("errors")
    if not isinstance(raw_errors, list):
        global_issues.append({"code": "who_report_errors_not_list"})
        raw_errors = []
    for index, error in enumerate(raw_errors):
        if not isinstance(error, dict):
            global_issues.append(
                {"code": "who_report_error_not_object", "error_index": index}
            )
            continue
        code = _normalize_code(error.get("code"))
        if code:
            errors_by_code[code].append(error)
        else:
            global_issues.append(
                {
                    "code": "who_report_global_api_error",
                    "error_index": index,
                    "api_error": error,
                }
            )

    summary = report.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("api_errors"), int):
        if summary["api_errors"] != len(raw_errors):
            global_issues.append(
                {
                    "code": "who_report_api_error_count_mismatch",
                    "summary_count": summary["api_errors"],
                    "error_rows": len(raw_errors),
                }
            )
    return results, duplicates, dict(errors_by_code), global_issues


def _definitive_not_found(error: dict[str, Any]) -> bool:
    status = error.get("http_status")
    try:
        if int(status) == 404:
            return True
    except (TypeError, ValueError):
        pass
    searchable = " ".join(
        str(error.get(key) or "")
        for key in ("message", "response_summary")
    ).casefold()
    return "does not exist" in searchable


def _only_definitive_not_found(errors: list[dict[str, Any]]) -> bool:
    return bool(errors) and all(_definitive_not_found(error) for error in errors)


def _result_validation_errors(
    *,
    prefix: str,
    expected_code: str,
    expected_title: str,
    expected_uri: str | None,
    result: dict[str, Any] | None,
    api_errors: list[dict[str, Any]],
    duplicated: bool,
    global_issues: list[dict[str, Any]],
    who: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if global_issues:
        errors.append(
            {
                "code": f"{prefix}_who_report_has_global_errors",
                "report_errors": global_issues,
            }
        )
    if api_errors:
        errors.append(
            {
                "code": f"{prefix}_api_error",
                "source_code": expected_code,
                "api_errors": api_errors,
            }
        )
    if duplicated:
        errors.append(
            {"code": f"{prefix}_duplicate_result", "source_code": expected_code}
        )
    if result is None:
        errors.append(
            {
                "code": f"{prefix}_api_validation_missing",
                "source_code": expected_code,
            }
        )
        return errors

    if result.get("valid") is not True:
        errors.append(
            {
                "code": f"{prefix}_result_not_explicitly_valid",
                "actual": result.get("valid"),
            }
        )
    requested_code = _normalize_code(result.get("requested_code"))
    if requested_code != expected_code:
        errors.append(
            {
                "code": f"{prefix}_requested_code_mismatch",
                "actual": requested_code,
                "expected": expected_code,
            }
        )
    canonical_code = _normalize_code(result.get("canonical_code"))
    if canonical_code != expected_code:
        errors.append(
            {
                "code": f"{prefix}_code_mismatch",
                "actual": canonical_code,
                "expected": expected_code,
            }
        )

    release = str(who.get("release") or "")
    language = str(who.get("language") or "")
    if result.get("release") != release:
        errors.append(
            {
                "code": f"{prefix}_result_release_mismatch",
                "actual": result.get("release"),
                "expected": release,
            }
        )
    if result.get("language") != language:
        errors.append(
            {
                "code": f"{prefix}_result_language_mismatch",
                "actual": result.get("language"),
                "expected": language,
            }
        )

    who_uri = str(result.get("entity_uri") or "")
    if not who_uri:
        errors.append({"code": f"{prefix}_entity_uri_missing"})
    elif not _is_who_mms_uri(who_uri, release):
        errors.append(
            {"code": f"{prefix}_entity_uri_not_who_mms", "actual": who_uri}
        )
    if expected_uri is not None:
        if not expected_uri:
            errors.append({"code": f"{prefix}_expected_uri_missing"})
        elif not _is_who_mms_uri(expected_uri, release):
            errors.append(
                {"code": f"{prefix}_expected_uri_not_who_mms", "actual": expected_uri}
            )
        elif who_uri and _uri_identity(who_uri) != _uri_identity(expected_uri):
            errors.append(
                {
                    "code": f"{prefix}_uri_mismatch",
                    "actual": who_uri,
                    "expected": expected_uri,
                }
            )

    who_title = str(result.get("title") or "")
    if not who_title:
        errors.append({"code": f"{prefix}_title_missing"})
    if not expected_title:
        errors.append({"code": f"{prefix}_expected_title_missing"})
    elif who_title and normalize_text(who_title) != normalize_text(expected_title):
        errors.append(
            {
                "code": f"{prefix}_title_mismatch",
                "actual": who_title,
                "expected": expected_title,
            }
        )
    return errors


def _canonical_entity_errors(
    record: dict[str, Any], schema: dict[str, Any]
) -> list[dict[str, Any]]:
    graph = graph_from_record(record)
    if not isinstance(graph, dict):
        return [{"code": "canonical_main_disease_graph_missing"}]
    matches = [
        entity
        for entity in graph.get("entities", [])
        if isinstance(entity, dict)
        and entity.get("label") == "Disease"
        and normalize_text(entity.get("name"))
        == normalize_text(record.get("source_title"))
    ]
    if len(matches) != 1:
        return [
            {
                "code": "canonical_main_disease_not_unique",
                "match_count": len(matches),
            }
        ]
    main = matches[0]
    properties = main.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    expected = {
        "icdcode": record.get("source_code"),
        "coding_system": schema["coding"]["coding_system"],
        "icd_release": record.get("source_release"),
        "icd_uri": record.get("source_record_id"),
    }
    errors: list[dict[str, Any]] = []
    for key, value in expected.items():
        if properties.get(key) != value:
            errors.append(
                {
                    "code": "canonical_main_disease_property_mismatch",
                    "entity_id": main.get("id"),
                    "property": key,
                    "actual": properties.get(key),
                    "expected": value,
                }
            )
    return errors


def _pending_base(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in _DISCARDED_AUDIT_KEYS}


def _candidate_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("entity_id") or ""),
        str(entry.get("property") or ""),
        _normalize_code(entry.get("value") or entry.get("code")),
    )


def _has_current_provenance(entry: dict[str, Any], who: dict[str, Any]) -> bool:
    return (
        entry.get("action")
        in {
            "not_attached_main_disease_only_schema",
            "invalid_code_not_attached",
            "title_mismatch_not_attached",
        }
        and entry.get("who_release") == who.get("release")
        and entry.get("api_version") == "v2"
    )


def apply_icd_validation(
    records: Iterable[dict[str, Any]],
    who_report: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_who = who_report.get("who")
    who = raw_who if isinstance(raw_who, dict) else {}
    if who.get("release") != schema.get("source_release"):
        raise ValueError(
            f"WHO report release {who.get('release')} does not match schema "
            f"{schema.get('source_release')}"
        )
    if who.get("api_version") != "v2":
        raise ValueError("WHO validation report must use API-Version v2")
    if not isinstance(who.get("language"), str) or not who["language"].strip():
        raise ValueError("WHO validation report must declare a language")

    results, duplicate_codes, api_errors, global_issues = _report_indexes(who_report)
    output: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    findings: list[dict[str, Any]] = []

    for record_index, original in enumerate(records):
        record = copy.deepcopy(original)
        source_code = _normalize_code(record.get("source_code"))
        main_result = results.get(source_code)
        icd_errors = _result_validation_errors(
            prefix="canonical_icd",
            expected_code=source_code,
            expected_title=str(record.get("source_title") or ""),
            expected_uri=str(record.get("source_record_id") or ""),
            result=main_result,
            api_errors=api_errors.get(source_code, []),
            duplicated=source_code in duplicate_codes,
            global_issues=global_issues,
            who=who,
        )
        icd_errors.extend(_canonical_entity_errors(record, schema))
        if icd_errors:
            totals["canonical_failed"] += 1
        else:
            totals["canonical_verified"] += 1
        record["icd_validation"] = {
            "status": "verified" if not icd_errors else "invalid",
            "source": "WHO ICD-11 MMS API",
            "api_version": "v2",
            "release": who.get("release"),
            "language": who.get("language"),
            "requested_code": (
                main_result.get("requested_code") if main_result else None
            ),
            "canonical_code": (
                main_result.get("canonical_code") if main_result else None
            ),
            "entity_uri": main_result.get("entity_uri") if main_result else None,
            "title": main_result.get("title") if main_result else None,
            "errors": icd_errors,
        }

        graph = graph_from_record(record) or {}
        entity_names = {
            str(entity.get("id")): str(entity.get("name") or "")
            for entity in graph.get("entities", [])
            if isinstance(entity, dict) and entity.get("id")
        }
        migration = record.get("migration")
        if not isinstance(migration, dict):
            migration = {}
            record["migration"] = migration

        raw_pending = migration.get("unverified_codes")
        pending = (
            raw_pending
            if isinstance(raw_pending, list)
            else []
            if raw_pending is None
            else [raw_pending]
        )
        raw_discarded = migration.get("discarded_non_main_codes")
        previous_discarded = raw_discarded if isinstance(raw_discarded, list) else []
        candidates: list[tuple[Any, bool]] = [(entry, False) for entry in pending]
        seen = {
            _candidate_key(entry)
            for entry in pending
            if isinstance(entry, dict)
        }
        for entry in previous_discarded:
            if not isinstance(entry, dict) or _candidate_key(entry) not in seen:
                candidates.append((entry, True))
                if isinstance(entry, dict):
                    seen.add(_candidate_key(entry))

        unresolved: list[dict[str, Any]] = []
        discarded: list[dict[str, Any]] = []
        for candidate, was_discarded in candidates:
            if not isinstance(candidate, dict):
                unresolved.append(
                    {"value": candidate, "reason": "invalid_pending_entry"}
                )
                totals["non_main_unresolved"] += 1
                continue
            entry = _pending_base(candidate)
            code = _normalize_code(entry.get("value") or candidate.get("code"))
            entity_id = str(entry.get("entity_id") or "")
            entity_name = entity_names.get(entity_id, "")
            result = results.get(code)
            code_api_errors = api_errors.get(code, [])

            if (
                was_discarded
                and result is None
                and not code_api_errors
                and code not in duplicate_codes
                and not global_issues
                and _has_current_provenance(candidate, who)
            ):
                discarded.append(candidate)
                totals["non_main_previously_confirmed_retained"] += 1
                continue

            if (
                result is None
                and code not in duplicate_codes
                and not global_issues
                and _only_definitive_not_found(code_api_errors)
            ):
                discarded.append(
                    {
                        **entry,
                        "code": code,
                        "entity_name": entity_name,
                        "action": "invalid_code_not_attached",
                        "who_release": who.get("release"),
                        "api_version": "v2",
                        "api_errors": code_api_errors,
                    }
                )
                totals["non_main_definitively_invalid_and_discarded"] += 1
                continue

            validation_errors = _result_validation_errors(
                prefix="non_main_icd",
                expected_code=code,
                expected_title=entity_name,
                expected_uri=None,
                result=result,
                api_errors=code_api_errors,
                duplicated=code in duplicate_codes,
                global_issues=global_issues,
                who=who,
            )
            title_mismatch_only = (
                len(validation_errors) == 1
                and validation_errors[0]["code"] == "non_main_icd_title_mismatch"
            )
            if title_mismatch_only:
                who_title = str(result.get("title") or "")
                discarded.append(
                    {
                        **entry,
                        "code": code,
                        "entity_name": entity_name,
                        "who_title": who_title,
                        "who_entity_uri": result.get("entity_uri"),
                        "title_matches": False,
                        "title_comparison": {
                            "expected_entity_name": entity_name,
                            "who_title": who_title,
                            "normalized_expected": normalize_text(entity_name),
                            "normalized_who_title": normalize_text(who_title),
                            "matches": False,
                        },
                        "action": "title_mismatch_not_attached",
                        "who_release": who.get("release"),
                        "api_version": "v2",
                    }
                )
                totals["non_main_title_mismatch_and_discarded"] += 1
                continue
            if validation_errors:
                unresolved.append(
                    {
                        **entry,
                        "reason": "WHO result did not confirm code, title, and URI",
                        "validation_errors": validation_errors,
                    }
                )
                totals["non_main_unresolved"] += 1
                for error in validation_errors:
                    totals[f"non_main_failure:{error['code']}"] += 1
                continue

            discarded.append(
                {
                    **entry,
                    "code": code,
                    "entity_name": entity_name,
                    "who_title": result.get("title"),
                    "who_entity_uri": result.get("entity_uri"),
                    "title_matches": True,
                    "action": "not_attached_main_disease_only_schema",
                    "who_release": who.get("release"),
                    "api_version": "v2",
                }
            )
            totals["non_main_validated_and_discarded"] += 1
            totals["non_main_exact_title_match"] += 1
        migration["unverified_codes"] = unresolved
        migration["discarded_non_main_codes"] = discarded

        validation = validate_record(record, schema)
        combined_errors = list(validation["errors"]) + icd_errors
        combined_warnings = list(validation["warnings"])
        migration["errors"] = combined_errors
        migration["warnings"] = combined_warnings
        migration["status"] = (
            "invalid"
            if combined_errors
            else "manual_review"
            if combined_warnings or unresolved
            else "repaired"
        )
        status_counts[migration["status"]] += 1
        if combined_errors or unresolved:
            findings.append(
                {
                    "record_index": record_index,
                    "source_record_id": record.get("source_record_id"),
                    "source_code": source_code,
                    "status": migration["status"],
                    "errors": combined_errors,
                    "warnings": combined_warnings,
                    "unresolved_non_main_codes": unresolved,
                }
            )
        output.append(record)

    filtered_who = {
        key: who.get(key)
        for key in ("release", "language", "api_version", "base_url")
        if key in who
    }
    records_with_schema_warnings = sum(
        bool((record.get("migration") or {}).get("warnings")) for record in output
    )
    strict_ready = bool(output) and all(
        (record.get("icd_validation") or {}).get("status") == "verified"
        and not (record.get("migration") or {}).get("errors")
        and not (record.get("migration") or {}).get("unverified_codes")
        for record in output
    )
    audit = {
        "schema_version": schema.get("schema_version"),
        "who": filtered_who,
        "record_count": len(output),
        "strict_ready": strict_ready,
        "status_counts": dict(sorted(status_counts.items())),
        "totals": dict(sorted(totals.items())),
        "records_with_icd_findings": len(findings),
        "records_with_schema_warnings": records_with_schema_warnings,
        "findings": findings,
    }
    return output, audit

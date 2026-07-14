#!/usr/bin/env python3
"""Validate local ICD-11 MMS codes against WHO codeinfo and entity endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.icd_api import (  # noqa: E402
    ICDAPIConfig,
    ICDAPIConfigurationError,
    ICDAPIError,
    WHOICDClient,
)


DEFAULT_SCHEMA_V2_INPUT = ROOT / "data/schema_v2/migrated/pro_cot_schema_v2.json"
DEFAULT_REPORT = ROOT / "reports/icd_api_validation_report.json"


def _load_local_environment() -> None:
    """Load the repository dotenv silently, with a tiny stdlib fallback."""

    dotenv_path = ROOT / ".env"
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        try:
            lines = dotenv_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return
    load_dotenv(dotenv_path, override=False, verbose=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--smoke-code",
        metavar="CODE",
        help="Validate one code, for example --smoke-code 6A05.0.",
    )
    source.add_argument(
        "--input",
        type=Path,
        help=f"Raw crawler JSON or Schema v2 JSON (typical: {DEFAULT_SCHEMA_V2_INPUT}).",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "raw", "schema-v2"),
        default="auto",
        help="Input structure; auto detects raw crawler data versus Schema v2.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Validate at most this many distinct codes (useful for staged checks).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass persistent public-response caches.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail when a local expected title differs from WHO's title.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
    os.replace(temporary, path)


def _normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _add_reference(
    found: "OrderedDict[str, dict[str, Any]]",
    code: Any,
    *,
    path: str,
    title: Any = "",
    record_id: Any = "",
) -> None:
    if not isinstance(code, str) or not code.strip():
        return
    normalized_code = code.strip().upper()
    entry = found.setdefault(
        normalized_code,
        {"code": normalized_code, "expected_titles": [], "references": []},
    )
    title_text = title.strip() if isinstance(title, str) else ""
    if title_text and title_text not in entry["expected_titles"]:
        entry["expected_titles"].append(title_text)
    reference = {"path": path}
    if record_id not in (None, ""):
        reference["record_id"] = str(record_id)
    if title_text:
        reference["expected_title"] = title_text
    if reference not in entry["references"]:
        entry["references"].append(reference)


def _detect_format(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        entities = payload["entities"]
        if entities and all(isinstance(item, dict) for item in entities):
            if any("code" in item and "title" in item for item in entities):
                return "raw"
        return "schema-v2"
    if isinstance(payload, list):
        samples = [item for item in payload[:20] if isinstance(item, dict)]
        if any(
            "source_code" in item
            or "schema_version" in item
            or isinstance(item.get("output"), dict)
            or isinstance(item.get("gold_output"), dict)
            for item in samples
        ):
            return "schema-v2"
        if any("code" in item and "title" in item for item in samples):
            return "raw"
    raise ValueError("could not detect raw or Schema v2 JSON structure")


def _raw_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        records = payload["entities"]
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("raw input must be a list or contain an entities/records list")
    return [record for record in records if isinstance(record, dict)]


def _schema_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("entities"), list):
        records = [payload]
    else:
        raise ValueError("Schema v2 input must be a list, graph, or records wrapper")
    return [record for record in records if isinstance(record, dict)]


def extract_code_references(
    payload: Any, input_format: str = "auto"
) -> tuple[str, list[dict[str, Any]]]:
    """Extract and deduplicate codes while retaining every source JSON path."""

    detected = _detect_format(payload) if input_format == "auto" else input_format
    found: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    if detected == "raw":
        records = _raw_records(payload)
        root = "$.entities" if isinstance(payload, dict) and "entities" in payload else "$"
        for index, record in enumerate(records):
            _add_reference(
                found,
                record.get("code"),
                path=f"{root}[{index}].code",
                title=record.get("title"),
                record_id=record.get("id"),
            )
        return detected, list(found.values())

    records = _schema_records(payload)
    for record_index, record in enumerate(records):
        base_path = "$" if len(records) == 1 and records[0] is payload else f"$[{record_index}]"
        record_id = record.get("source_record_id") or record.get("source_id")
        source_title = record.get("source_title") or record.get("title")
        if record.get("source_code"):
            _add_reference(
                found,
                record.get("source_code"),
                path=f"{base_path}.source_code",
                title=source_title,
                record_id=record_id,
            )
        elif record.get("code"):
            _add_reference(
                found,
                record.get("code"),
                path=f"{base_path}.code",
                title=source_title,
                record_id=record_id,
            )

        graph_name = ""
        graph: dict[str, Any] | None = None
        for candidate in ("output", "gold_output"):
            if isinstance(record.get(candidate), dict):
                graph_name = candidate
                graph = record[candidate]
                break
        if graph is None and isinstance(record.get("entities"), list):
            graph = record
        entity_names: dict[str, str] = {}
        if isinstance(graph, dict):
            entity_names = {
                str(entity.get("id")): str(entity.get("name") or "")
                for entity in graph.get("entities", [])
                if isinstance(entity, dict) and entity.get("id")
            }
        migration = record.get("migration")
        if isinstance(migration, dict):
            for code_index, pending in enumerate(
                migration.get("unverified_codes") or []
            ):
                if not isinstance(pending, dict):
                    continue
                entity_id = str(pending.get("entity_id") or "")
                _add_reference(
                    found,
                    pending.get("value"),
                    path=(
                        f"{base_path}.migration.unverified_codes[{code_index}].value"
                    ),
                    title=entity_names.get(entity_id, ""),
                    record_id=entity_id or record_id,
                )
        if graph is None:
            continue
        graph_path = f"{base_path}.{graph_name}" if graph_name else base_path
        for entity_index, entity in enumerate(graph.get("entities", [])):
            if not isinstance(entity, dict):
                continue
            properties = entity.get("properties")
            if not isinstance(properties, dict):
                continue
            _add_reference(
                found,
                properties.get("icdcode"),
                path=f"{graph_path}.entities[{entity_index}].properties.icdcode",
                title=entity.get("name"),
                record_id=entity.get("id") or record_id,
            )
    return detected, list(found.values())


def _title_comparisons(expected_titles: list[str], who_title: str) -> list[dict[str, Any]]:
    who_normalized = _normalize_title(who_title)
    return [
        {
            "expected_title": title,
            "matches_who_title": _normalize_title(title) == who_normalized,
        }
        for title in expected_titles
    ]


def validate_references(
    references: list[dict[str, Any]],
    client: WHOICDClient,
    *,
    force_refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for item in references:
        code = item["code"]
        try:
            lookup = client.lookup_code(code, force_refresh=force_refresh)
        except ICDAPIError as exc:
            error = {"code": code, **exc.as_dict(), "references": item["references"]}
            errors.append(error)
            print(f"ICD API error for {code}: {exc}", file=sys.stderr)
            continue
        comparisons = _title_comparisons(item["expected_titles"], lookup.title)
        result = lookup.as_dict()
        result.update(
            {
                "valid": True,
                "expected_titles": item["expected_titles"],
                "title_comparisons": comparisons,
                "all_expected_titles_match": all(
                    comparison["matches_who_title"] for comparison in comparisons
                ),
                "references": item["references"],
                "codeinfo_summary": {
                    "code": lookup.codeinfo.get("code"),
                    "stemCode": lookup.codeinfo.get("stemCode"),
                    "stemId": lookup.codeinfo.get("stemId"),
                },
            }
        )
        results.append(result)
    return results, errors


def _report(
    *,
    mode: str,
    source: str,
    input_format: str,
    config: ICDAPIConfig,
    references: list[dict[str, Any]],
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatch_count = sum(not result["all_expected_titles_match"] for result in results)
    return {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "source": source,
        "input_format": input_format,
        "who": {
            "release": config.release,
            "language": config.language,
            "api_version": config.api_version,
            "base_url": config.base_url,
        },
        "summary": {
            "distinct_codes": len(references),
            "valid_codes": len(results),
            "api_errors": len(errors),
            "codes_with_title_mismatch": mismatch_count,
        },
        "results": results,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # load_dotenv does not override credentials already supplied by a managed
    # environment.  Neither dotenv values nor the dotenv file are printed.
    _load_local_environment()
    try:
        config = ICDAPIConfig.from_env()
    except ICDAPIConfigurationError as exc:
        fatal = {
            "report_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {"distinct_codes": 0, "valid_codes": 0, "api_errors": 1},
            "results": [],
            "errors": [{"type": "configuration", "message": str(exc)}],
        }
        _write_json_atomic(args.output, fatal)
        print(f"configuration error: {exc}", file=sys.stderr)
        print(f"wrote {args.output}")
        return 2

    if args.smoke_code:
        input_format = "smoke"
        source = "command-line smoke code"
        references = [
            {"code": args.smoke_code.strip().upper(), "expected_titles": [], "references": []}
        ]
        mode = "smoke"
    else:
        source = str(args.input)
        try:
            payload = _read_json(args.input)
            input_format, references = extract_code_references(payload, args.input_format)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            fatal = {
                "report_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "summary": {"distinct_codes": 0, "valid_codes": 0, "api_errors": 1},
                "results": [],
                "errors": [
                    {
                        "type": "input",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }
            _write_json_atomic(args.output, fatal)
            print(f"input error: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"wrote {args.output}")
            return 2
        mode = "batch"

    if args.limit is not None:
        references = references[: args.limit]
    with WHOICDClient(config) as client:
        results, errors = validate_references(
            references, client, force_refresh=args.force_refresh
        )
    report = _report(
        mode=mode,
        source=source,
        input_format=input_format,
        config=config,
        references=references,
        results=results,
        errors=errors,
    )
    _write_json_atomic(args.output, report)
    summary = report["summary"]
    print(
        f"codes={summary['distinct_codes']} valid={summary['valid_codes']} "
        f"api_errors={summary['api_errors']} "
        f"title_mismatches={summary['codes_with_title_mismatch']}"
    )
    print(f"wrote {args.output}")
    if errors:
        return 1
    if args.strict and summary["codes_with_title_mismatch"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

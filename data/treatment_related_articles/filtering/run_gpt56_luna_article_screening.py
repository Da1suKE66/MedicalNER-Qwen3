#!/usr/bin/env python3
"""Screen every indexed treatment article with gpt-5.6-luna.

Requests run in deterministic worker partitions (``index % workers``). Every
attempt is appended to a durable JSONL checkpoint before more work is accepted;
reruns skip all attempted indices unless ``--retry-failed`` is supplied. Once
all records have valid outputs, a compact index/output JSON file is assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO


api_key = "sk-SCP0FiLRcsRfCwUX0b4aB1B512E94075913c8f0d2d273b38"
api_base = "https://api-2.xi-ai.cn/v1"

MODEL_NAME = "gpt-5.6-luna"
AUTOMATIC_RETRIES = 0
CHECKPOINT_SCHEMA_VERSION = "treatment-article-screening-jsonl-v1"

SCRIPT_DIR = Path(__file__).parent
ARTICLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = ARTICLE_DIR.parents[1]
DEFAULT_ARTICLES_PATH = ARTICLE_DIR / "articles.json"
DEFAULT_SYSTEM_PROMPT_PATH = (
    REPO_ROOT / "schemas" / "treatment_article_kg_screening_prompt.md"
)
DEFAULT_KG_SCHEMA_PATH = REPO_ROOT / "schemas" / "v2.0.0" / "schema.json"
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "gpt56_luna_article_screening_checkpoint.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "gpt56_luna_article_screening.json"

RECORDS_START_RE = re.compile(rb'^\s*"records"\s*:\s*\[\s*$')
RECORDS_END_RE = re.compile(rb"^\s*\]\s*,?\s*$")
STOP = object()

REASON_CODES = {
    "KEEP_SUPPORTED_SCHEMA_FACT",
    "REVIEW_AMBIGUOUS_SCHEMA_FACT",
    "REVIEW_AMBIGUOUS_ENTITY_TYPE",
    "REVIEW_UNCLEAR_ASSERTION_STATUS",
    "REVIEW_CONTRADICTORY_EVIDENCE",
    "REVIEW_INCOMPLETE_OR_CORRUPTED_TEXT",
    "DROP_NO_SCHEMA_FACT",
    "DROP_ENTITY_ONLY",
    "DROP_UNSUPPORTED_FACT_TYPE",
    "DROP_NONASSERTED_PLAN_ONLY",
    "DROP_OFF_TOPIC_OR_INSUFFICIENT_TEXT",
}
RELATION_CANDIDATE_KEYS = {
    "relation",
    "source_text",
    "source_type",
    "target_text",
    "target_type",
    "evidence_quote",
}
PROPERTY_CANDIDATE_KEYS = {
    "entity_text",
    "entity_type",
    "property",
    "value_text",
    "evidence_quote",
}


@dataclass(frozen=True)
class SchemaContract:
    version: str
    entity_properties: dict[str, set[str]]
    relation_pairs: dict[str, set[tuple[str, str]]]


@dataclass(frozen=True)
class AttemptState:
    status: str
    valid_output: bool
    offset: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def article_input_sha256(record: dict[str, Any]) -> str:
    return sha256_text(record["title"] + "\0" + record["abstract"])


def load_schema_contract(path: Path) -> SchemaContract:
    schema = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or not isinstance(schema.get("schema_version"), str):
        raise ValueError(f"Invalid KG schema: {path}")

    entity_specs = schema.get("entity_types")
    if not isinstance(entity_specs, dict) or not entity_specs:
        raise ValueError(f"KG schema has no entity_types object: {path}")
    entity_properties: dict[str, set[str]] = {}
    for entity_type, spec in entity_specs.items():
        properties = spec.get("properties") if isinstance(spec, dict) else None
        if (
            not isinstance(entity_type, str)
            or not isinstance(properties, list)
            or not all(isinstance(item, str) for item in properties)
        ):
            raise ValueError(f"Invalid entity property contract for {entity_type!r}")
        entity_properties[entity_type] = set(properties)

    relation_specs = schema.get("relation_types")
    if not isinstance(relation_specs, dict):
        raise ValueError(f"KG schema has no relation_types object: {path}")
    relation_pairs: dict[str, set[tuple[str, str]]] = {}
    for relation, spec in relation_specs.items():
        if not isinstance(spec, dict) or spec.get("status") != "active":
            continue
        raw_pairs = spec.get("allowed_pairs")
        if raw_pairs is None:
            sources = spec.get("source")
            targets = spec.get("target")
            if not isinstance(sources, list) or not isinstance(targets, list):
                raise ValueError(f"Active relation {relation!r} has no type contract")
            raw_pairs = [[source, target] for source in sources for target in targets]
        pairs = {
            (pair[0], pair[1])
            for pair in raw_pairs
            if isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(item, str) for item in pair)
        }
        if not pairs or len(pairs) != len(raw_pairs):
            raise ValueError(f"Invalid allowed pairs for active relation {relation!r}")
        relation_pairs[relation] = pairs
    return SchemaContract(schema["schema_version"], entity_properties, relation_pairs)


def iter_articles(path: Path) -> Iterator[dict[str, Any]]:
    found_records = False
    in_records = False
    finished_records = False
    expected_index = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            logical_line = line.rstrip(b"\r\n")
            if not in_records:
                if not found_records and RECORDS_START_RE.fullmatch(logical_line):
                    found_records = True
                    in_records = True
                continue
            if RECORDS_END_RE.fullmatch(logical_line):
                finished_records = True
                in_records = False
                break
            stripped = logical_line.strip()
            if not stripped:
                continue
            if stripped.endswith(b","):
                stripped = stripped[:-1].rstrip()
            try:
                record = json.loads(stripped)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Record line {line_number} is not a complete JSON object"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record line {line_number} is not an object")
            index = record.get("index")
            if type(index) is not int or index != expected_index:
                raise ValueError(
                    f"Record position {expected_index} has invalid index {index!r}; "
                    "run add_article_indices.py first"
                )
            for field in ("title", "abstract"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Record {index} has no non-empty {field}")
            yield record
            expected_index += 1
    if not found_records:
        raise ValueError(f'{path} has no top-level "records" array')
    if not finished_records:
        raise ValueError(f'{path} has an unterminated "records" array')
    if expected_index == 0:
        raise ValueError(f'{path} has an empty "records" array')


def request_messages(system_prompt: str, record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Title:\n{record['title']}\n\nAbstract:\n{record['abstract']}",
        },
    ]


def response_message_text(message: Any) -> tuple[str, str]:
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    reasoning = getattr(message, "reasoning_content", None)
    if not isinstance(reasoning, str):
        reasoning = ""
    return content.strip(), reasoning.strip()


def parse_response(content: str) -> tuple[dict[str, Any], bool]:
    if not content:
        raise ValueError("The response content is empty")
    try:
        parsed = json.loads(content)
        strict_json = True
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object was found in the response")
        parsed = json.loads(content[start : end + 1])
        strict_json = False
    if not isinstance(parsed, dict):
        raise ValueError("The parsed response must be a JSON object")
    return parsed, strict_json


def validate_output(
    output: dict[str, Any],
    title: str,
    abstract: str,
    contract: SchemaContract,
) -> list[str]:
    errors: list[str] = []
    required_keys = {
        "schema_version",
        "decision",
        "reason_code",
        "reason",
        "candidate_relations",
        "candidate_properties",
    }
    missing_keys = sorted(required_keys - set(output))
    extra_keys = sorted(set(output) - required_keys)
    if missing_keys:
        errors.append(f"missing top-level keys: {missing_keys}")
    if extra_keys:
        errors.append(f"unexpected top-level keys: {extra_keys}")
    if output.get("schema_version") != contract.version:
        errors.append(f"schema_version must be {contract.version!r}")

    decision = output.get("decision")
    if not isinstance(decision, str) or decision not in {"KEEP", "REVIEW", "DROP"}:
        errors.append("decision must be KEEP, REVIEW, or DROP")
    reason_code = output.get("reason_code")
    if not isinstance(reason_code, str) or reason_code not in REASON_CODES:
        errors.append("reason_code is not allowed")
    if isinstance(decision, str) and isinstance(reason_code, str):
        if not reason_code.startswith(decision + "_"):
            errors.append("reason_code prefix does not match decision")
    if not isinstance(output.get("reason"), str) or not output.get("reason", "").strip():
        errors.append("reason must be a non-empty string")

    relation_candidates = output.get("candidate_relations")
    property_candidates = output.get("candidate_properties")
    if not isinstance(relation_candidates, list):
        errors.append("candidate_relations must be a list")
        relation_candidates = []
    if not isinstance(property_candidates, list):
        errors.append("candidate_properties must be a list")
        property_candidates = []
    candidate_count = len(relation_candidates) + len(property_candidates)
    if candidate_count > 3:
        errors.append("candidate arrays contain more than three total items")
    if decision == "KEEP" and not 1 <= candidate_count <= 3:
        errors.append("KEEP must contain one to three total candidates")
    if decision == "DROP" and candidate_count:
        errors.append("DROP must contain two empty candidate arrays")

    entity_types = set(contract.entity_properties)
    evidence_source = title + "\n" + abstract
    for position, candidate in enumerate(relation_candidates):
        prefix = f"candidate_relations[{position}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(RELATION_CANDIDATE_KEYS - set(candidate))
        extra = sorted(set(candidate) - RELATION_CANDIDATE_KEYS)
        if missing:
            errors.append(f"{prefix} missing keys: {missing}")
        if extra:
            errors.append(f"{prefix} has unexpected keys: {extra}")
        relation = candidate.get("relation")
        source_type = candidate.get("source_type")
        target_type = candidate.get("target_type")
        if not isinstance(relation, str) or relation not in contract.relation_pairs:
            errors.append(f"{prefix}.relation is not allowed")
        if not isinstance(source_type, str) or source_type not in entity_types:
            errors.append(f"{prefix}.source_type is not allowed")
        if not isinstance(target_type, str) or target_type not in entity_types:
            errors.append(f"{prefix}.target_type is not allowed")
        if (
            isinstance(relation, str)
            and relation in contract.relation_pairs
            and isinstance(source_type, str)
            and isinstance(target_type, str)
            and (source_type, target_type) not in contract.relation_pairs[relation]
        ):
            errors.append(f"{prefix} has an invalid source/target type pair")
        for field in ("source_text", "target_text", "evidence_quote"):
            value = candidate.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
            elif value not in evidence_source:
                errors.append(f"{prefix}.{field} is not an exact input span")

    for position, candidate in enumerate(property_candidates):
        prefix = f"candidate_properties[{position}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(PROPERTY_CANDIDATE_KEYS - set(candidate))
        extra = sorted(set(candidate) - PROPERTY_CANDIDATE_KEYS)
        if missing:
            errors.append(f"{prefix} missing keys: {missing}")
        if extra:
            errors.append(f"{prefix} has unexpected keys: {extra}")
        entity_type = candidate.get("entity_type")
        property_name = candidate.get("property")
        if not isinstance(entity_type, str) or entity_type not in entity_types:
            errors.append(f"{prefix}.entity_type is not allowed")
        elif (
            not isinstance(property_name, str)
            or property_name not in contract.entity_properties[entity_type]
        ):
            errors.append(f"{prefix}.property is not allowed for {entity_type}")
        for field in ("entity_text", "value_text", "evidence_quote"):
            value = candidate.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
            elif value not in evidence_source:
                errors.append(f"{prefix}.{field} is not an exact input span")
    return errors


def usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def checkpoint_identity(
    articles_path: Path,
    articles_sha256: str,
    total_records: int,
    system_prompt_path: Path,
    system_prompt: str,
    kg_schema_path: Path,
    kg_schema_text: str,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "record_type": "run_metadata",
        "model": MODEL_NAME,
        "api_base": api_base,
        "automatic_retries": AUTOMATIC_RETRIES,
        "articles_file": str(articles_path),
        "articles_sha256": articles_sha256,
        "total_records": total_records,
        "system_prompt_file": str(system_prompt_path),
        "system_prompt_sha256": sha256_text(system_prompt),
        "kg_schema_file": str(kg_schema_path),
        "kg_schema_sha256": sha256_text(kg_schema_text),
    }


def load_checkpoint(
    path: Path, expected_identity: dict[str, Any]
) -> tuple[dict[int, AttemptState], dict[str, Any] | None]:
    if not path.exists():
        return {}, None
    states: dict[int, AttemptState] = {}
    with path.open("rb") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"Checkpoint is empty: {path}")
        try:
            metadata = json.loads(first_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint metadata is invalid: {path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint metadata must be a JSON object")
        for key, expected in expected_identity.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Checkpoint identity mismatch for {key}: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )

        line_number = 1
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            try:
                attempt = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid checkpoint JSON at line {line_number}; no requests "
                    "were started"
                ) from exc
            if not isinstance(attempt, dict) or attempt.get("record_type") != "attempt":
                raise ValueError(f"Invalid checkpoint record at line {line_number}")
            index = attempt.get("index")
            if type(index) is not int or not 0 <= index < expected_identity["total_records"]:
                raise ValueError(f"Invalid checkpoint index at line {line_number}")
            status = attempt.get("status")
            if status not in {
                "completed",
                "completed_with_validation_errors",
                "parse_failed",
                "request_failed",
            }:
                raise ValueError(f"Invalid checkpoint status at line {line_number}")
            states[index] = AttemptState(
                status=status,
                valid_output=attempt.get("valid_output") is True,
                offset=offset,
            )
    return states, metadata


def create_checkpoint(path: Path, identity: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {**identity, "created_at": utc_now()}
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_attempt(handle: TextIO, attempt: dict[str, Any]) -> None:
    handle.write(json.dumps(attempt, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def should_request(
    index: int, states: dict[int, AttemptState], retry_failed: bool
) -> bool:
    state = states.get(index)
    if state is None:
        return True
    return retry_failed and not state.valid_output


def request_article(
    client: Any,
    system_prompt: str,
    record: dict[str, Any],
    contract: SchemaContract,
) -> dict[str, Any]:
    index = record["index"]
    started_at = utc_now()
    raw_response = ""
    reasoning_content = ""
    common = {
        "record_type": "attempt",
        "index": index,
        "input_sha256": article_input_sha256(record),
        "attempt_started_at": started_at,
        "model": MODEL_NAME,
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=request_messages(system_prompt, record),
        )
        if not completion.choices:
            raise ValueError("The API response contains no choices")
        choice = completion.choices[0]
        raw_response, reasoning_content = response_message_text(choice.message)
        usage = usage_dict(completion)
        try:
            output, strict_json = parse_response(raw_response)
            validation_errors = validate_output(
                output, record["title"], record["abstract"], contract
            )
            valid = strict_json and not validation_errors
            return {
                **common,
                "status": "completed" if valid else "completed_with_validation_errors",
                "success": True,
                "valid_output": valid,
                "attempt_finished_at": utc_now(),
                "response_id": getattr(completion, "id", None),
                "response_model": getattr(completion, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "strict_json_response": strict_json,
                "output": output,
                "validation_errors": validation_errors,
                "raw_response": raw_response,
                "reasoning_content": reasoning_content,
                "usage": usage,
            }
        except Exception as exc:
            return {
                **common,
                "status": "parse_failed",
                "success": False,
                "valid_output": False,
                "attempt_finished_at": utc_now(),
                "response_id": getattr(completion, "id", None),
                "response_model": getattr(completion, "model", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_response": raw_response,
                "reasoning_content": reasoning_content,
                "usage": usage,
            }
    except Exception as exc:
        return {
            **common,
            "status": "request_failed",
            "success": False,
            "valid_output": False,
            "attempt_finished_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_response": raw_response,
            "reasoning_content": reasoning_content,
        }


def default_client_factory(worker_id: int, resolved_api_key: str) -> Any:
    del worker_id
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai") from exc
    return OpenAI(
        api_key=resolved_api_key,
        base_url=api_base,
        timeout=600.0,
        max_retries=AUTOMATIC_RETRIES,
    )


def worker_loop(
    worker_id: int,
    client: Any,
    task_queue: queue.Queue[Any],
    result_queue: queue.Queue[tuple[str, Any]],
    system_prompt: str,
    contract: SchemaContract,
    sleep_seconds: float,
) -> None:
    made_request = False
    try:
        while True:
            task = task_queue.get()
            try:
                if task is STOP:
                    return
                if made_request and sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                attempt = request_article(client, system_prompt, task, contract)
                made_request = True
                result_queue.put(("attempt", attempt))
            finally:
                task_queue.task_done()
    finally:
        result_queue.put(("worker_done", worker_id))


def producer_loop(
    articles_path: Path,
    task_queues: list[queue.Queue[Any]],
    states: dict[int, AttemptState],
    retry_failed: bool,
    start_index: int,
    end_index: int,
    result_queue: queue.Queue[tuple[str, Any]],
) -> None:
    dispatched = 0
    error: str | None = None
    try:
        for record in iter_articles(articles_path):
            index = record["index"]
            if index < start_index:
                continue
            if index >= end_index:
                break
            if not should_request(index, states, retry_failed):
                continue
            worker_id = index % len(task_queues)
            task_queues[worker_id].put(record)
            dispatched += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for task_queue in task_queues:
            task_queue.put(STOP)
        result_queue.put(
            ("producer_done", {"dispatched": dispatched, "error": error})
        )


def scan_request_plan(
    articles_path: Path,
    states: dict[int, AttemptState],
    retry_failed: bool,
    start_index: int,
    end_index: int | None,
    max_samples: int | None,
    workers: int,
) -> dict[str, Any]:
    total_records = 0
    selected_records = 0
    pending_requests = 0
    partition_counts = [0] * workers
    if end_index is not None:
        requested_end_exclusive = end_index + 1
    elif max_samples is not None:
        requested_end_exclusive = start_index + max_samples
    else:
        requested_end_exclusive = None
    for record in iter_articles(articles_path):
        index = record["index"]
        total_records += 1
        if index < start_index or (
            requested_end_exclusive is not None
            and index >= requested_end_exclusive
        ):
            continue
        selected_records += 1
        if should_request(index, states, retry_failed):
            pending_requests += 1
            partition_counts[index % workers] += 1
    if start_index >= total_records:
        raise ValueError(f"--start-index must be between 0 and {total_records - 1}")
    if end_index is not None and end_index >= total_records:
        raise ValueError(f"--end-index must be between {start_index} and {total_records - 1}")
    end_index_exclusive = (
        total_records
        if requested_end_exclusive is None
        else min(total_records, requested_end_exclusive)
    )
    return {
        "total_records": total_records,
        "selected_records": selected_records,
        "pending_requests": pending_requests,
        "start_index": start_index,
        "end_index": end_index_exclusive - 1,
        "end_index_exclusive": end_index_exclusive,
        "partition_counts": partition_counts,
    }


def summarize_states(total_records: int, states: dict[int, AttemptState]) -> dict[str, Any]:
    statuses = Counter(state.status for state in states.values())
    valid = sum(state.valid_output for state in states.values())
    return {
        "total_records": total_records,
        "attempted_records": len(states),
        "unattempted_records": total_records - len(states),
        "valid_outputs": valid,
        "invalid_or_failed_outputs": len(states) - valid,
        "status_counts": dict(sorted(statuses.items())),
    }


def build_final_output(
    checkpoint_path: Path,
    output_path: Path,
    states: dict[int, AttemptState],
    total_records: int,
) -> None:
    if len(states) != total_records or not all(
        state.valid_output for state in states.values()
    ):
        raise ValueError("Cannot finalize before every article has a valid output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_size = 1000
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.buckets.", dir=output_path.parent
    ) as bucket_dir_name:
        bucket_dir = Path(bucket_dir_name)
        bucket_handles: dict[int, Any] = {}
        try:
            with checkpoint_path.open("rb") as checkpoint:
                checkpoint.readline()
                while True:
                    offset = checkpoint.tell()
                    line = checkpoint.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    attempt = json.loads(line)
                    index = attempt.get("index")
                    state = states.get(index) if type(index) is int else None
                    if state is None or state.offset != offset:
                        continue
                    entry = {"index": index, "output": attempt["output"]}
                    bucket_number = index // bucket_size
                    bucket_handle = bucket_handles.get(bucket_number)
                    if bucket_handle is None:
                        bucket_handle = (bucket_dir / f"{bucket_number:06d}.jsonl").open(
                            "ab"
                        )
                        bucket_handles[bucket_number] = bucket_handle
                    bucket_handle.write(
                        json.dumps(
                            entry, ensure_ascii=False, separators=(",", ":")
                        ).encode("utf-8")
                        + b"\n"
                    )
        finally:
            for bucket_handle in bucket_handles.values():
                bucket_handle.close()

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
                delete=False,
            ) as output:
                temporary_name = output.name
                output.write("[\n")
                expected_index = 0
                first = True
                for bucket_path in sorted(bucket_dir.glob("*.jsonl")):
                    entries = [
                        json.loads(line)
                        for line in bucket_path.read_text(encoding="utf-8").splitlines()
                        if line
                    ]
                    entries.sort(key=lambda item: item["index"])
                    for entry in entries:
                        if entry["index"] != expected_index:
                            raise ValueError(
                                f"Final output index mismatch: {entry['index']} != "
                                f"{expected_index}"
                            )
                        if not first:
                            output.write(",\n")
                        output.write("  ")
                        output.write(
                            json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                        )
                        first = False
                        expected_index += 1
                if expected_index != total_records:
                    raise ValueError(
                        f"Final output has {expected_index} records; expected {total_records}"
                    )
                output.write("\n]\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(Path(temporary_name), output_path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass


def run(
    args: argparse.Namespace,
    client_factory: Callable[[int, str], Any] | None = None,
) -> dict[str, Any]:
    articles_path = args.articles
    system_prompt_path = args.system_prompt
    kg_schema_path = args.kg_schema
    checkpoint_path = args.checkpoint
    output_path = args.output

    if args.workers <= 0:
        raise ValueError("--workers must be greater than zero")
    if args.start_index < 0:
        raise ValueError("--start-index cannot be negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    if args.end_index is not None and args.max_samples is not None:
        raise ValueError("--end-index and --max-samples cannot be used together")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero")
    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative")
    if not articles_path.is_file():
        raise FileNotFoundError(f"Articles file does not exist: {articles_path}")

    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt is empty: {system_prompt_path}")
    kg_schema_text = kg_schema_path.read_text(encoding="utf-8")
    contract = load_schema_contract(kg_schema_path)
    articles_sha256 = sha256_file(articles_path)

    preliminary_total = sum(1 for _ in iter_articles(articles_path))
    identity = checkpoint_identity(
        articles_path,
        articles_sha256,
        preliminary_total,
        system_prompt_path,
        system_prompt,
        kg_schema_path,
        kg_schema_text,
    )
    states, metadata = load_checkpoint(checkpoint_path, identity)
    plan = scan_request_plan(
        articles_path,
        states,
        args.retry_failed,
        args.start_index,
        args.end_index,
        args.max_samples,
        args.workers,
    )
    print(f"Articles: {articles_path}")
    print(f"System prompt: {system_prompt_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {api_base}/chat/completions")
    print(f"Workers: {args.workers}")
    print(f"Index range (inclusive): {plan['start_index']}..{plan['end_index']}")
    print(f"Selected records: {plan['selected_records']}")
    print(f"Pending paid requests: {plan['pending_requests']}")
    print(f"Worker partitions: {plan['partition_counts']}")
    print(f"Automatic retries: {AUTOMATIC_RETRIES}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Final output: {output_path}")
    if args.preview:
        print("Preview completed; no checkpoint was changed and no API request was sent.")
        return {"plan": plan, "summary": summarize_states(preliminary_total, states)}

    if plan["pending_requests"]:
        resolved_api_key = os.getenv("XI_AI_API_KEY", "").strip() or api_key.strip()
        if not resolved_api_key and client_factory is None:
            raise ValueError(
                "API key is empty; fill api_key or set the XI_AI_API_KEY environment variable"
            )
        factory = client_factory or default_client_factory
        clients = [factory(worker_id, resolved_api_key) for worker_id in range(args.workers)]
        if metadata is None:
            create_checkpoint(checkpoint_path, identity)

        task_queues = [queue.Queue(maxsize=2) for _ in range(args.workers)]
        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        workers = [
            threading.Thread(
                target=worker_loop,
                name=f"screening-worker-{worker_id}",
                args=(
                    worker_id,
                    clients[worker_id],
                    task_queues[worker_id],
                    result_queue,
                    system_prompt,
                    contract,
                    args.sleep,
                ),
                daemon=True,
            )
            for worker_id in range(args.workers)
        ]
        for worker in workers:
            worker.start()
        producer = threading.Thread(
            target=producer_loop,
            name="screening-producer",
            args=(
                articles_path,
                task_queues,
                states,
                args.retry_failed,
                plan["start_index"],
                plan["end_index_exclusive"],
                result_queue,
            ),
            daemon=True,
        )
        producer.start()

        attempts_received = 0
        workers_done = 0
        producer_result: dict[str, Any] | None = None
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            while producer_result is None or workers_done < args.workers:
                message_type, payload = result_queue.get()
                if message_type == "attempt":
                    append_attempt(checkpoint, payload)
                    attempts_received += 1
                    print(
                        f"[{attempts_received}/{plan['pending_requests']}] "
                        f"index={payload['index']} status={payload['status']}"
                    )
                elif message_type == "worker_done":
                    workers_done += 1
                elif message_type == "producer_done":
                    producer_result = payload
                else:
                    raise RuntimeError(f"Unknown worker message: {message_type}")
        producer.join()
        for worker in workers:
            worker.join()
        if producer_result is None:
            raise RuntimeError("Producer did not report completion")
        if producer_result["error"]:
            raise RuntimeError(f"Producer failed: {producer_result['error']}")
        if attempts_received != producer_result["dispatched"]:
            raise RuntimeError(
                f"Checkpointed {attempts_received} attempts but dispatched "
                f"{producer_result['dispatched']}"
            )

    if checkpoint_path.exists():
        states, _ = load_checkpoint(checkpoint_path, identity)
    summary = summarize_states(preliminary_total, states)
    if summary["valid_outputs"] == preliminary_total:
        build_final_output(checkpoint_path, output_path, states, preliminary_total)
        summary["final_output"] = str(output_path)
    else:
        summary["final_output"] = None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"plan": plan, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen indexed treatment articles with gpt-5.6-luna"
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="Number of concurrent API workers (required)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First article index to include (default: 0)",
    )
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "--end-index",
        type=int,
        help="Last article index to include (inclusive)",
    )
    range_group.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of articles starting at --start-index",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between requests in each worker (default: 0)",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly retry failed or schema-invalid checkpoint records",
    )
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument(
        "--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT_PATH
    )
    parser.add_argument("--kg-schema", type=Path, default=DEFAULT_KG_SCHEMA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        raise SystemExit(f"Screening failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()

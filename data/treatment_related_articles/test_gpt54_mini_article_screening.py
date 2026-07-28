#!/usr/bin/env python3
"""Screen the 22 treatment abstracts with gpt-5.4-mini.

The script sends at most one request per pending sample, never retries
automatically, and atomically checkpoints a single complete JSON artifact after
every attempt. Existing attempted records, including failures, are skipped on
rerun unless --retry-failed or --overwrite is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

api_key = "sk-SCP0FiLRcsRfCwUX0b4aB1B512E94075913c8f0d2d273b38"
api_base = "https://api-2.xi-ai.cn/v1"


MODEL_NAME = "gpt-5.4-mini"
AUTOMATIC_RETRIES = 0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_PATH = (
    SCRIPT_DIR
    / "treatment_related_articles_predict_22_gemini_completed_llamafactory.json"
)
MANIFEST_PATH = SCRIPT_DIR / "treatment_related_articles_predict_22_manifest.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "schemas" / "treatment_article_kg_screening_prompt.md"
OUTPUT_PATH = SCRIPT_DIR / "treatment_related_articles_screening_gpt_5_4_mini_22.json"

MEDICAL_TEXT_MARKER = "Medical text:\n"

ENTITY_TYPES = {
    "Disease",
    "Symptom",
    "Diagnostic Criteria",
    "Interview Tool",
    "Patient Information",
    "Medication",
    "Communication Method",
    "Risk Information",
}

RELATION_PAIRS = {
    "subsumes": {
        ("Disease", "Disease"),
        ("Symptom", "Symptom"),
    },
    "differentiates_from": {("Disease", "Disease")},
    "co_occurs_with_frequency": {("Disease", "Disease")},
    "associated_with_poor_prognosis_in": {
        ("Disease", "Disease"),
        ("Symptom", "Disease"),
        ("Patient Information", "Disease"),
    },
    "is_core_symptom_of": {("Symptom", "Disease")},
    "is_associated_symptom_of": {("Symptom", "Disease")},
    "required_for_diagnosis_of": {("Diagnostic Criteria", "Disease")},
    "supports_subtyping_of": {
        ("Diagnostic Criteria", "Disease"),
        ("Symptom", "Disease"),
    },
    "first_line_for": {("Medication", "Disease")},
    "informed_by_patient_demographics": {
        ("Interview Tool", "Patient Information")
    },
    "affects_diagnosis_of": {("Patient Information", "Disease")},
    "must_be_ruled_out_for": {("Disease", "Disease")},
    "excludes_diagnosis_of": {
        ("Diagnostic Criteria", "Disease"),
        ("Symptom", "Disease"),
    },
    "somatic_cause_of": {("Disease", "Disease")},
    "assesses_for": {
        ("Interview Tool", "Disease"),
        ("Interview Tool", "Symptom"),
    },
    "recommended_for": {
        ("Communication Method", "Disease"),
        ("Communication Method", "Patient Information"),
    },
    "triggers_alert_when": {
        ("Risk Information", "Disease"),
        ("Risk Information", "Symptom"),
    },
}

REASON_CODES = {
    "KEEP_SUPPORTED_ACTIVE_RELATION",
    "REVIEW_AMBIGUOUS_RELATION",
    "REVIEW_AMBIGUOUS_ENTITY_TYPE",
    "REVIEW_UNCLEAR_ASSERTION_STATUS",
    "REVIEW_INCOMPLETE_OR_CORRUPTED_TEXT",
    "DROP_NO_ACTIVE_RELATION",
    "DROP_ENTITY_ONLY",
    "DROP_UNSUPPORTED_RELATION_TYPE",
    "DROP_NONASSERTED_PLAN_ONLY",
    "DROP_OFF_TOPIC_OR_INSUFFICIENT_TEXT",
}

CANDIDATE_KEYS = {
    "relation",
    "source_text",
    "source_type",
    "target_text",
    "target_type",
    "evidence_quote",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_abstract(sample: dict[str, Any], index: int) -> str:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Sample {index} has no messages list.")
    user_messages = [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if len(user_messages) != 1 or not isinstance(user_messages[0], str):
        raise ValueError(f"Sample {index} must contain exactly one user message.")
    user_content = user_messages[0]
    if MEDICAL_TEXT_MARKER not in user_content:
        raise ValueError(
            f"Sample {index} is missing the {MEDICAL_TEXT_MARKER!r} marker."
        )
    abstract = user_content.rsplit(MEDICAL_TEXT_MARKER, 1)[1].strip()
    if not abstract:
        raise ValueError(f"Sample {index} has an empty abstract.")
    return abstract


def build_inputs() -> tuple[list[dict[str, Any]], str]:
    source = load_json(SOURCE_PATH)
    manifest = load_json(MANIFEST_PATH)
    if not isinstance(source, list):
        raise ValueError("The source dataset must be a JSON list.")
    manifest_records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(manifest_records, list):
        raise ValueError("The manifest must contain a records list.")
    if len(source) != 22 or len(manifest_records) != 22:
        raise ValueError(
            f"Expected 22 source and manifest records; found "
            f"{len(source)} and {len(manifest_records)}."
        )

    records = []
    seen_abstracts: set[str] = set()
    for index, (sample, metadata) in enumerate(zip(source, manifest_records)):
        if not isinstance(sample, dict) or not isinstance(metadata, dict):
            raise ValueError(f"Sample or manifest record {index} is not an object.")
        abstract = extract_abstract(sample, index)
        expected_chars = metadata.get("abstract_chars")
        if isinstance(expected_chars, int) and len(abstract) != expected_chars:
            raise ValueError(
                f"Abstract length mismatch at index {index}: "
                f"{len(abstract)} != {expected_chars}."
            )
        if abstract in seen_abstracts:
            raise ValueError(f"Duplicate abstract at index {index}.")
        seen_abstracts.add(abstract)
        title = metadata.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Manifest record {index} has no title.")
        records.append(
            {
                "global_idx": index,
                "source": str(metadata.get("source") or ""),
                "source_index": metadata.get("source_index"),
                "doi": str(metadata.get("doi") or ""),
                "title": title.strip(),
                "abstract": abstract,
                "title_chars": len(title.strip()),
                "abstract_chars": len(abstract),
                "status": "pending",
                "success": False,
                "valid_output": False,
            }
        )

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError(f"System prompt is empty: {SYSTEM_PROMPT_PATH}")
    return records, system_prompt


def request_messages(system_prompt: str, record: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        f"Title:\n{record['title']}\n\n"
        f"Abstract:\n{record['abstract']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
        raise ValueError("The response content is empty.")
    try:
        parsed = json.loads(content)
        strict_json = True
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object was found in the response.")
        parsed = json.loads(content[start : end + 1])
        strict_json = False
    if not isinstance(parsed, dict):
        raise ValueError("The parsed response must be a JSON object.")
    return parsed, strict_json


def validate_output(
    output: dict[str, Any], title: str, abstract: str
) -> list[str]:
    errors = []
    required_keys = {
        "schema_version",
        "decision",
        "reason_code",
        "reason",
        "candidate_relations",
    }
    missing_keys = sorted(required_keys - set(output))
    extra_keys = sorted(set(output) - required_keys)
    if missing_keys:
        errors.append(f"missing top-level keys: {missing_keys}")
    if extra_keys:
        errors.append(f"unexpected top-level keys: {extra_keys}")

    if output.get("schema_version") != "2.0.0-draft.2":
        errors.append("schema_version must be '2.0.0-draft.2'")
    decision = output.get("decision")
    if decision not in {"KEEP", "REVIEW", "DROP"}:
        errors.append("decision must be KEEP, REVIEW, or DROP")
    reason_code = output.get("reason_code")
    if reason_code not in REASON_CODES:
        errors.append("reason_code is not allowed")
    if isinstance(decision, str) and isinstance(reason_code, str):
        if not reason_code.startswith(decision + "_"):
            errors.append("reason_code prefix does not match decision")
    if not isinstance(output.get("reason"), str) or not output.get("reason", "").strip():
        errors.append("reason must be a non-empty string")

    candidates = output.get("candidate_relations")
    if not isinstance(candidates, list):
        errors.append("candidate_relations must be a list")
        return errors
    if len(candidates) > 3:
        errors.append("candidate_relations contains more than three items")
    if decision == "KEEP" and not 1 <= len(candidates) <= 3:
        errors.append("KEEP must contain one to three candidate relations")
    if decision == "DROP" and candidates:
        errors.append("DROP must contain an empty candidate_relations list")

    evidence_source = title + "\n" + abstract
    for position, candidate in enumerate(candidates):
        prefix = f"candidate_relations[{position}]"
        if not isinstance(candidate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = sorted(CANDIDATE_KEYS - set(candidate))
        extra = sorted(set(candidate) - CANDIDATE_KEYS)
        if missing:
            errors.append(f"{prefix} missing keys: {missing}")
        if extra:
            errors.append(f"{prefix} has unexpected keys: {extra}")
        relation = candidate.get("relation")
        source_type = candidate.get("source_type")
        target_type = candidate.get("target_type")
        if relation not in RELATION_PAIRS:
            errors.append(f"{prefix}.relation is not allowed")
        if source_type not in ENTITY_TYPES:
            errors.append(f"{prefix}.source_type is not allowed")
        if target_type not in ENTITY_TYPES:
            errors.append(f"{prefix}.target_type is not allowed")
        if relation in RELATION_PAIRS and (source_type, target_type) not in RELATION_PAIRS[relation]:
            errors.append(f"{prefix} has an invalid source/target type pair")
        for field in ("source_text", "target_text", "evidence_quote"):
            value = candidate.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        evidence = candidate.get("evidence_quote")
        if isinstance(evidence, str) and evidence not in evidence_source:
            errors.append(f"{prefix}.evidence_quote is not an exact input quote")
    return errors


def usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    result = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def build_artifact(
    records: list[dict[str, Any]], system_prompt: str, created_at: str
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    decision_counts = {"KEEP": 0, "REVIEW": 0, "DROP": 0}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        output = record.get("output")
        if isinstance(output, dict) and output.get("decision") in decision_counts:
            decision_counts[output["decision"]] += 1
        usage = record.get("usage")
        if isinstance(usage, dict):
            for field in usage_totals:
                value = usage.get(field)
                if isinstance(value, int):
                    usage_totals[field] += value

    attempted = sum(record.get("status") != "pending" for record in records)
    valid = sum(record.get("valid_output") is True for record in records)
    return {
        "artifact_schema_version": "treatment-article-screening-run-v1",
        "created_at": created_at,
        "updated_at": utc_now(),
        "model": MODEL_NAME,
        "api_base": api_base,
        "automatic_retries": AUTOMATIC_RETRIES,
        "input_file": str(SOURCE_PATH),
        "manifest_file": str(MANIFEST_PATH),
        "system_prompt_file": str(SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": sha256_text(system_prompt),
        "summary": {
            "total_records": len(records),
            "attempted_records": attempted,
            "pending_records": len(records) - attempted,
            "valid_outputs": valid,
            "invalid_or_failed_outputs": attempted - valid,
            "status_counts": status_counts,
            "decision_counts": decision_counts,
            "usage": usage_totals,
        },
        "records": records,
    }


def checkpoint(records: list[dict[str, Any]], system_prompt: str, created_at: str) -> None:
    write_json_atomic(OUTPUT_PATH, build_artifact(records, system_prompt, created_at))


def load_checkpoint(
    fresh_records: list[dict[str, Any]], system_prompt: str, overwrite: bool
) -> tuple[list[dict[str, Any]], str]:
    if overwrite or not OUTPUT_PATH.exists():
        return fresh_records, utc_now()
    artifact = load_json(OUTPUT_PATH)
    if not isinstance(artifact, dict):
        raise ValueError(f"Existing output is not an object: {OUTPUT_PATH}")
    if artifact.get("model") != MODEL_NAME:
        raise ValueError("Existing output model does not match this run.")
    if artifact.get("system_prompt_sha256") != sha256_text(system_prompt):
        raise ValueError("System prompt changed since the existing checkpoint.")
    existing_records = artifact.get("records")
    if not isinstance(existing_records, list) or len(existing_records) != len(fresh_records):
        raise ValueError("Existing checkpoint does not contain 22 aligned records.")
    for index, (fresh, existing) in enumerate(zip(fresh_records, existing_records)):
        if not isinstance(existing, dict):
            raise ValueError(f"Checkpoint record {index} is not an object.")
        if fresh["abstract"] != existing.get("abstract") or fresh["title"] != existing.get("title"):
            raise ValueError(f"Checkpoint input mismatch at index {index}.")
    created_at = str(artifact.get("created_at") or utc_now())
    return existing_records, created_at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen 22 treatment-related abstracts with gpt-5.4-mini."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First zero-based index to consider (default: 0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of indices to consider from --start-index.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds between paid requests (default: 1).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Validate and display the request plan without calling the API.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Explicitly re-request records whose prior attempt failed or was invalid.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard the existing checkpoint and request selected records again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fresh_records, system_prompt = build_inputs()
    records, created_at = load_checkpoint(fresh_records, system_prompt, args.overwrite)

    if not 0 <= args.start_index < len(records):
        raise SystemExit(f"--start-index must be between 0 and {len(records) - 1}.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero.")
    end_index = len(records)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)
    selected_indices = list(range(args.start_index, end_index))

    terminal_statuses = {
        "completed",
        "completed_with_validation_errors",
        "parse_failed",
        "request_failed",
    }
    pending_indices = []
    for index in selected_indices:
        status = records[index].get("status")
        if status == "pending":
            pending_indices.append(index)
        elif args.retry_failed and status in {
            "completed_with_validation_errors",
            "parse_failed",
            "request_failed",
        }:
            pending_indices.append(index)
        elif status not in terminal_statuses:
            raise ValueError(f"Unknown checkpoint status at index {index}: {status!r}")

    print(f"Source: {SOURCE_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"System prompt: {SYSTEM_PROMPT_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {api_base}/chat/completions")
    print(f"Selected records: {len(selected_indices)}")
    print(f"Pending paid requests: {len(pending_indices)}")
    print(f"Automatic retries: {AUTOMATIC_RETRIES}")
    print(f"Output: {OUTPUT_PATH}")

    if args.preview:
        print("Preview completed; no API request was sent.")
        return

    if not pending_indices:
        checkpoint(records, system_prompt, created_at)
        print("No pending request; checkpoint is already complete for this selection.")
        return

    resolved_api_key = os.getenv("XI_AI_API_KEY", "").strip() or api_key.strip()
    if not resolved_api_key:
        raise SystemExit(
            "API key is empty. Set XI_AI_API_KEY or configure the existing "
            "test_gemini_chat_completion.py script."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install it with: "
            "python -m pip install openai"
        ) from exc

    client = OpenAI(api_key=resolved_api_key, base_url=api_base, timeout=600.0)
    for progress, index in enumerate(pending_indices, start=1):
        record = records[index]
        print(
            f"\n[{progress}/{len(pending_indices)}] index={index} "
            f"source={record.get('source')} abstract_chars={record.get('abstract_chars')}"
        )
        attempt_started_at = utc_now()
        raw_response = ""
        reasoning_content = ""
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=request_messages(system_prompt, record),
            )
            if not completion.choices:
                raise ValueError("The API response contains no choices.")
            choice = completion.choices[0]
            raw_response, reasoning_content = response_message_text(choice.message)
            usage = usage_dict(completion)
            try:
                output, strict_json = parse_response(raw_response)
                validation_errors = validate_output(
                    output, record["title"], record["abstract"]
                )
                record.update(
                    {
                        "status": (
                            "completed"
                            if strict_json and not validation_errors
                            else "completed_with_validation_errors"
                        ),
                        "success": True,
                        "valid_output": strict_json and not validation_errors,
                        "attempt_started_at": attempt_started_at,
                        "attempt_finished_at": utc_now(),
                        "model": MODEL_NAME,
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
                )
                print(
                    f"Parsed: decision={output.get('decision')} "
                    f"valid={record['valid_output']} errors={len(validation_errors)}"
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "parse_failed",
                        "success": False,
                        "valid_output": False,
                        "attempt_started_at": attempt_started_at,
                        "attempt_finished_at": utc_now(),
                        "model": MODEL_NAME,
                        "response_id": getattr(completion, "id", None),
                        "response_model": getattr(completion, "model", None),
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_response": raw_response,
                        "reasoning_content": reasoning_content,
                        "usage": usage,
                    }
                )
                print(f"Parse failed without retry: {type(exc).__name__}: {exc}")
        except Exception as exc:
            record.update(
                {
                    "status": "request_failed",
                    "success": False,
                    "valid_output": False,
                    "attempt_started_at": attempt_started_at,
                    "attempt_finished_at": utc_now(),
                    "model": MODEL_NAME,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_response": raw_response,
                    "reasoning_content": reasoning_content,
                }
            )
            print(f"Request failed without retry: {type(exc).__name__}: {exc}")

        checkpoint(records, system_prompt, created_at)
        if args.sleep > 0 and progress < len(pending_indices):
            time.sleep(args.sleep)

    artifact = build_artifact(records, system_prompt, created_at)
    write_json_atomic(OUTPUT_PATH, artifact)
    summary = artifact["summary"]
    print("\nScreening finished.")
    print(f"Attempted records: {summary['attempted_records']}/{summary['total_records']}")
    print(f"Valid outputs: {summary['valid_outputs']}")
    print(f"Decision counts: {summary['decision_counts']}")
    print(f"Status counts: {summary['status_counts']}")
    print(f"Usage: {summary['usage']}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

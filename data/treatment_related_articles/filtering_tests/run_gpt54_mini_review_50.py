#!/usr/bin/env python3
"""Screen the reproducible 50-article sample with gpt-5.4-mini.

Each pending record receives at most one paid request. The complete artifact is
atomically checkpointed after every attempt, and failures are never retried
unless --retry-failed is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).parent
ARTICLE_DIR = SCRIPT_DIR.parent
REPO_ROOT = ARTICLE_DIR.parents[1]
sys.path.insert(0, str(ARTICLE_DIR))

from test_gpt54_mini_article_screening import (  # noqa: E402
    AUTOMATIC_RETRIES,
    MODEL_NAME,
    api_base,
    api_key,
    parse_response,
    response_message_text,
    sha256_text,
    usage_dict,
    validate_output,
    write_json_atomic,
)


SAMPLE_PATH = SCRIPT_DIR / "random_articles_50.json"
MANUAL_REVIEW_PATH = SCRIPT_DIR / "manual_review_50.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "schemas" / "treatment_article_kg_screening_prompt.md"
OUTPUT_PATH = SCRIPT_DIR / "gpt56_sol_review_50.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_fresh_records() -> tuple[list[dict[str, Any]], str]:
    sample = load_json(SAMPLE_PATH)
    source_records = sample.get("records") if isinstance(sample, dict) else None
    if not isinstance(source_records, list) or len(source_records) != 50:
        raise ValueError("random_articles_50.json must contain exactly 50 records")
    records = []
    for expected_index, source in enumerate(source_records):
        if not isinstance(source, dict) or source.get("sample_index") != expected_index:
            raise ValueError(f"Sample alignment error at index {expected_index}")
        title = source.get("title")
        abstract = source.get("abstract")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Missing title at index {expected_index}")
        if not isinstance(abstract, str) or not abstract.strip():
            raise ValueError(f"Missing abstract at index {expected_index}")
        records.append(
            {
                "sample_index": expected_index,
                "source_record_index": source.get("source_record_index"),
                "source": source.get("source", ""),
                "disease_codes": source.get("disease_codes", []),
                "doi": source.get("doi", ""),
                "title": title,
                "abstract": abstract,
                "title_chars": len(title),
                "abstract_chars": len(abstract),
                "abstract_sha256": source.get("abstract_sha256", ""),
                "status": "pending",
                "success": False,
                "valid_output": False,
            }
        )
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ValueError("The system prompt is empty")
    return records, system_prompt


def request_messages(system_prompt: str, record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Title:\n{record['title']}\n\nAbstract:\n{record['abstract']}",
        },
    ]


def build_artifact(
    records: list[dict[str, Any]], system_prompt: str, created_at: str
) -> dict[str, Any]:
    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    decisions = Counter()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        output = record.get("output")
        if isinstance(output, dict) and output.get("decision") in {
            "KEEP",
            "REVIEW",
            "DROP",
        }:
            decisions[output["decision"]] += 1
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
        "sample_file": str(SAMPLE_PATH),
        "manual_review_file": str(MANUAL_REVIEW_PATH),
        "system_prompt_file": str(SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": sha256_text(system_prompt),
        "summary": {
            "total_records": len(records),
            "attempted_records": attempted,
            "pending_records": len(records) - attempted,
            "valid_outputs": valid,
            "invalid_or_failed_outputs": attempted - valid,
            "status_counts": dict(sorted(statuses.items())),
            "decision_counts": {
                decision: decisions.get(decision, 0)
                for decision in ("KEEP", "REVIEW", "DROP")
            },
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
        raise ValueError("Existing checkpoint must be an object")
    if artifact.get("model") != MODEL_NAME:
        raise ValueError("Existing checkpoint model mismatch")
    if artifact.get("system_prompt_sha256") != sha256_text(system_prompt):
        raise ValueError("System prompt changed since the existing checkpoint")
    existing = artifact.get("records")
    if not isinstance(existing, list) or len(existing) != len(fresh_records):
        raise ValueError("Existing checkpoint is not aligned to 50 samples")
    for index, (fresh, saved) in enumerate(zip(fresh_records, existing)):
        if not isinstance(saved, dict):
            raise ValueError(f"Checkpoint record {index} is not an object")
        if fresh["abstract_sha256"] != saved.get("abstract_sha256"):
            raise ValueError(f"Checkpoint input mismatch at index {index}")
    return existing, str(artifact.get("created_at") or utc_now())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen random_articles_50.json with gpt-5.4-mini"
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fresh_records, system_prompt = build_fresh_records()
    records, created_at = load_checkpoint(fresh_records, system_prompt, args.overwrite)
    if not 0 <= args.start_index < len(records):
        raise SystemExit(f"--start-index must be between 0 and {len(records) - 1}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be greater than zero")
    end_index = len(records)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + args.max_samples)
    selected = list(range(args.start_index, end_index))
    retryable = {
        "completed_with_validation_errors",
        "parse_failed",
        "request_failed",
    }
    terminal = {"completed", *retryable}
    pending = []
    for index in selected:
        status = records[index].get("status")
        if status == "pending" or (args.retry_failed and status in retryable):
            pending.append(index)
        elif status not in terminal:
            raise ValueError(f"Unknown status at index {index}: {status!r}")

    print(f"Sample: {SAMPLE_PATH}")
    print(f"Manual review: {MANUAL_REVIEW_PATH}")
    print(f"System prompt: {SYSTEM_PROMPT_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {api_base}/chat/completions")
    print(f"Selected records: {len(selected)}")
    print(f"Pending paid requests: {len(pending)}")
    print(f"Automatic retries: {AUTOMATIC_RETRIES}")
    print(f"Output: {OUTPUT_PATH}")
    if args.preview:
        print("Preview completed; no API request was sent.")
        return
    if not pending:
        checkpoint(records, system_prompt, created_at)
        print("No pending requests.")
        return

    resolved_api_key = os.getenv("XI_AI_API_KEY", "").strip() or api_key.strip()
    if not resolved_api_key:
        raise SystemExit("API key is empty; set XI_AI_API_KEY or the existing fallback key")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: openai") from exc
    client = OpenAI(api_key=resolved_api_key, base_url=api_base, timeout=600.0)

    for progress, index in enumerate(pending, start=1):
        record = records[index]
        print(
            f"\n[{progress}/{len(pending)}] index={index} "
            f"source={record['source']} chars={record['abstract_chars']}"
        )
        started_at = utc_now()
        raw_response = ""
        reasoning_content = ""
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
                    output, record["title"], record["abstract"]
                )
                valid = strict_json and not validation_errors
                record.update(
                    {
                        "status": "completed" if valid else "completed_with_validation_errors",
                        "success": True,
                        "valid_output": valid,
                        "attempt_started_at": started_at,
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
                    f"valid={valid} errors={len(validation_errors)}"
                )
            except Exception as exc:
                record.update(
                    {
                        "status": "parse_failed",
                        "success": False,
                        "valid_output": False,
                        "attempt_started_at": started_at,
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
                    "attempt_started_at": started_at,
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
        if args.sleep > 0 and progress < len(pending):
            time.sleep(args.sleep)

    artifact = build_artifact(records, system_prompt, created_at)
    write_json_atomic(OUTPUT_PATH, artifact)
    print("\nScreening finished.")
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

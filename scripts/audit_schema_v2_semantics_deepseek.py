#!/usr/bin/env python3
"""Run a resumable, full-coverage DeepSeek semantic audit of Schema v2 data."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.deepseek_repair import (  # noqa: E402
    DeepSeekAPIError,
    DeepSeekConfig,
    DeepSeekRepairClient,
)
from kg_lora.schema_v2 import (  # noqa: E402
    load_json,
    load_schema,
    record_list,
    sha256_file,
    write_json,
)
from kg_lora.semantic_audit import (  # noqa: E402
    SEMANTIC_AUDIT_PROTOCOL_VERSION,
    SEMANTIC_AUDIT_SYSTEM_PROMPT,
    SemanticAuditResponse,
    SemanticAuditValidationError,
    build_semantic_audit_task,
    require_valid_semantic_audit_response,
    response_requires_review,
    response_has_unresolved_semantics,
    response_semantic_signature,
    response_sha256,
    sha256_value,
)
from kg_lora.semantic_patch import (  # noqa: E402
    SemanticPatchError,
    apply_consensus_patch,
    compile_consensus_patch,
)


DEFAULT_INPUT = ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_icd_validated.json"
DEFAULT_STATE_DB = ROOT / ".cache/semantic-audit/state.sqlite3"
DEFAULT_REPORT = ROOT / "reports/semantic_audit_progress.json"
DEFAULT_PATCH_OUTPUT = ROOT / "data/schema_v2/semantic_audit/consensus_patches.json"
DEFAULT_CLEAN_OUTPUT = (
    ROOT / "data/schema_v2/cleaned/pro_cot_schema_v2_semantic_clean.json"
)
DEFAULT_APPLICATION_REPORT = ROOT / "reports/semantic_clean_application_report.json"

Phase = Literal["primary", "blind_review"]
_PRINT_LOCK = threading.Lock()
MINIMUM_STALE_LEASE_SECONDS = 15 * 60
STALE_LEASE_GRACE_SECONDS = 5 * 60


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


@contextmanager
def connect_database(path: Path) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=60000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database(path: Path) -> None:
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                protocol_version TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                schema_sha256 TEXT NOT NULL,
                prompt_sha256 TEXT NOT NULL,
                task_manifest_sha256 TEXT NOT NULL,
                request_config_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                input_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS records (
                run_id TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                record_index INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL,
                primary_task_sha256 TEXT NOT NULL,
                review_task_sha256 TEXT NOT NULL,
                risk_tier TEXT NOT NULL,
                risk_reasons_json TEXT NOT NULL,
                requires_review INTEGER NOT NULL DEFAULT 1,
                primary_state TEXT NOT NULL DEFAULT 'pending',
                review_state TEXT NOT NULL DEFAULT 'pending',
                primary_attempts INTEGER NOT NULL DEFAULT 0,
                review_attempts INTEGER NOT NULL DEFAULT 0,
                primary_response_sha256 TEXT,
                review_response_sha256 TEXT,
                primary_error TEXT,
                review_error TEXT,
                primary_lease_token TEXT,
                review_lease_token TEXT,
                primary_started_at TEXT,
                review_started_at TEXT,
                final_status TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, source_record_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS responses (
                response_sha256 TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                task_sha256 TEXT NOT NULL,
                model TEXT,
                finish_reason TEXT,
                usage_json TEXT,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                source_record_id TEXT,
                phase TEXT,
                old_state TEXT,
                new_state TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_attempts (
                lease_token TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt_no INTEGER NOT NULL,
                state TEXT NOT NULL,
                response_sha256 TEXT,
                model TEXT,
                finish_reason TEXT,
                usage_json TEXT,
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_records_run_primary
                ON records(run_id, primary_state);
            CREATE INDEX IF NOT EXISTS idx_records_run_review
                ON records(run_id, review_state);
            CREATE INDEX IF NOT EXISTS idx_responses_run_record
                ON responses(run_id, source_record_id, phase);
            CREATE INDEX IF NOT EXISTS idx_api_attempts_run
                ON api_attempts(run_id, phase, state);
            """
        )
        run_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
        if "task_manifest_sha256" not in run_columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN task_manifest_sha256 TEXT NOT NULL DEFAULT ''"
            )
        if "request_config_sha256" not in run_columns:
            connection.execute(
                "ALTER TABLE runs ADD COLUMN request_config_sha256 TEXT NOT NULL DEFAULT ''"
            )
        record_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(records)").fetchall()
        }
        for column_name in (
            "primary_lease_token",
            "review_lease_token",
            "primary_started_at",
            "review_started_at",
        ):
            if column_name not in record_columns:
                connection.execute(f"ALTER TABLE records ADD COLUMN {column_name} TEXT")


def derive_run_id(
    dataset_sha256: str,
    schema_sha256: str,
    prompt_sha256: str,
    task_manifest_sha256: str,
    request_config_sha256: str,
) -> str:
    return sha256_value(
        {
            "protocol": SEMANTIC_AUDIT_PROTOCOL_VERSION,
            "dataset_sha256": dataset_sha256,
            "schema_sha256": schema_sha256,
            "prompt_sha256": prompt_sha256,
            "task_manifest_sha256": task_manifest_sha256,
            "request_config_sha256": request_config_sha256,
        }
    )[:24]


def public_request_config(max_tokens: int) -> dict[str, Any]:
    """Return every non-secret request option that can change model output."""

    thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
    effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "high").strip().lower()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
    trust_proxy = os.getenv("DEEPSEEK_TRUST_ENV_PROXY", "false").strip().lower()
    if thinking not in {"enabled", "disabled"}:
        raise SystemExit("DEEPSEEK_THINKING must be enabled or disabled")
    if effort not in {"high", "max"}:
        raise SystemExit("DEEPSEEK_REASONING_EFFORT must be high or max")
    if not base_url.startswith(("https://", "http://")):
        raise SystemExit("DEEPSEEK_BASE_URL must be an HTTP(S) URL")
    if not model:
        raise SystemExit("DEEPSEEK_MODEL cannot be empty")
    try:
        timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))
    except ValueError as exc:
        raise SystemExit("DEEPSEEK_TIMEOUT_SECONDS must be numeric") from exc
    if timeout_seconds <= 0 or max_tokens < 1:
        raise SystemExit("request timeout and max tokens must be positive")
    return {
        "base_url": base_url,
        "model": model,
        "thinking": thinking,
        "reasoning_effort": effort,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "trust_environment_proxy": trust_proxy in {"1", "true", "yes"},
        "response_format": "json_object",
    }


def build_task_plan(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    planned_tasks = [
        (
            build_semantic_audit_task(
                record, schema, dataset_sha256=dataset_sha256, phase="primary"
            ),
            build_semantic_audit_task(
                record,
                schema,
                dataset_sha256=dataset_sha256,
                phase="blind_review",
            ),
        )
        for record in records
    ]
    manifest_sha256 = sha256_value(
        [
            {
                "source_record_id": primary["source_record_id"],
                "primary_task_sha256": primary["task_sha256"],
                "review_task_sha256": review["task_sha256"],
            }
            for primary, review in planned_tasks
        ]
    )
    return planned_tasks, manifest_sha256


def prepare_run(
    *,
    state_db: Path,
    input_path: Path,
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
    request_config: dict[str, Any],
    review_all: bool,
) -> str:
    if not records:
        raise SystemExit("input dataset is empty")
    planned_tasks, task_manifest_sha256 = build_task_plan(
        records, schema, dataset_sha256
    )
    first_task = planned_tasks[0][0]
    schema_sha256 = first_task["schema_sha256"]
    prompt_sha256 = first_task["prompt_sha256"]
    request_config_sha256 = sha256_value(request_config)
    run_id = derive_run_id(
        dataset_sha256,
        schema_sha256,
        prompt_sha256,
        task_manifest_sha256,
        request_config_sha256,
    )
    config = {
        "review_all": review_all,
        "request_config": request_config,
        "record_count": len(records),
        "entity_count": 0,
        "relation_count": 0,
        "cue_count": 0,
        "dimension_count": 0,
    }
    initialize_database(state_db)
    with connect_database(state_db) as connection:
        existing = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if existing:
            expected = {
                "dataset_sha256": dataset_sha256,
                "schema_sha256": schema_sha256,
                "prompt_sha256": prompt_sha256,
                "task_manifest_sha256": task_manifest_sha256,
                "request_config_sha256": request_config_sha256,
                "model": request_config["model"],
            }
            for key, value in expected.items():
                if existing[key] != value:
                    raise SystemExit(f"existing run metadata mismatch: {key}")
        else:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, protocol_version, dataset_sha256, schema_sha256,
                    prompt_sha256, task_manifest_sha256, request_config_sha256,
                    model, input_path, config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    SEMANTIC_AUDIT_PROTOCOL_VERSION,
                    dataset_sha256,
                    schema_sha256,
                    prompt_sha256,
                    task_manifest_sha256,
                    request_config_sha256,
                    request_config["model"],
                    str(input_path.resolve()),
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    now_utc(),
                ),
            )

        planned_ids: set[str] = set()
        entity_count = 0
        relation_count = 0
        cue_count = 0
        dimension_count = 0
        for index, (record, task_pair) in enumerate(zip(records, planned_tasks)):
            primary_task, review_task = task_pair
            source_record_id = primary_task["source_record_id"]
            if source_record_id in planned_ids:
                raise SystemExit(f"duplicate source_record_id: {source_record_id}")
            planned_ids.add(source_record_id)
            entity_count += len(primary_task["entity_inventory"])
            relation_count += len(primary_task["relation_inventory"])
            cue_count += len(primary_task["cue_inventory"])
            dimension_count += len(primary_task["required_dimensions"])
            requires_review = review_all or primary_task["risk_tier"] == "high"
            connection.execute(
                """
                INSERT INTO records (
                    run_id, source_record_id, record_index, record_sha256,
                    primary_task_sha256, review_task_sha256, risk_tier,
                    risk_reasons_json, requires_review, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, source_record_id) DO UPDATE SET
                    record_index=excluded.record_index,
                    requires_review=MAX(records.requires_review, excluded.requires_review),
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    source_record_id,
                    index,
                    primary_task["record_sha256"],
                    primary_task["task_sha256"],
                    review_task["task_sha256"],
                    primary_task["risk_tier"],
                    json.dumps(primary_task["risk_reasons"], ensure_ascii=False),
                    int(requires_review),
                    now_utc(),
                ),
            )
        actual_count = connection.execute(
            "SELECT COUNT(*) FROM records WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        if actual_count != len(records):
            raise SystemExit(
                f"planned record count mismatch: database={actual_count} input={len(records)}"
            )
        config.update(
            {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "cue_count": cue_count,
                "dimension_count": dimension_count,
            }
        )
        connection.execute(
            "UPDATE runs SET config_json=? WHERE run_id=?",
            (json.dumps(config, ensure_ascii=False, sort_keys=True), run_id),
        )
        connection.execute(
            """
            INSERT INTO events (
                run_id, source_record_id, phase, old_state, new_state,
                details_json, created_at
            ) VALUES (?, NULL, NULL, NULL, 'planned', ?, ?)
            """,
            (
                run_id,
                json.dumps(
                    {"record_count": len(records), "review_all": review_all},
                    ensure_ascii=False,
                ),
                now_utc(),
            ),
        )
    return run_id


def verify_run_context(
    *,
    state_db: Path,
    run_id: str,
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
    request_config: dict[str, Any],
) -> None:
    tasks, task_manifest_sha256 = build_task_plan(records, schema, dataset_sha256)
    if not tasks:
        raise SystemExit("input dataset is empty")
    first = tasks[0][0]
    expected_run = {
        "dataset_sha256": dataset_sha256,
        "schema_sha256": first["schema_sha256"],
        "prompt_sha256": first["prompt_sha256"],
        "task_manifest_sha256": task_manifest_sha256,
        "request_config_sha256": sha256_value(request_config),
        "model": request_config["model"],
    }
    with connect_database(state_db) as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise SystemExit(f"unknown run_id: {run_id}")
        for key, value in expected_run.items():
            if run[key] != value:
                raise SystemExit(
                    f"run context mismatch for {key}; run plan again for current input/schema/model"
                )
        rows = connection.execute(
            "SELECT * FROM records WHERE run_id=? ORDER BY record_index", (run_id,)
        ).fetchall()
    if len(rows) != len(tasks):
        raise SystemExit("run record count does not match current task plan")
    for index, (row, (primary, review)) in enumerate(zip(rows, tasks)):
        expected = {
            "record_index": index,
            "source_record_id": primary["source_record_id"],
            "record_sha256": primary["record_sha256"],
            "primary_task_sha256": primary["task_sha256"],
            "review_task_sha256": review["task_sha256"],
        }
        for key, value in expected.items():
            if row[key] != value:
                raise SystemExit(
                    f"run record context mismatch at index={index} field={key}"
                )


def _phase_columns(phase: Phase) -> tuple[str, str, str, str, str]:
    if phase == "primary":
        return (
            "primary_state",
            "primary_attempts",
            "primary_response_sha256",
            "primary_error",
            "primary_task_sha256",
        )
    return (
        "review_state",
        "review_attempts",
        "review_response_sha256",
        "review_error",
        "review_task_sha256",
    )


def _phase_lease_columns(phase: Phase) -> tuple[str, str]:
    if phase == "primary":
        return "primary_lease_token", "primary_started_at"
    return "review_lease_token", "review_started_at"


def mark_running(
    state_db: Path, run_id: str, source_record_id: str, phase: Phase
) -> tuple[int, str]:
    state_column, attempts_column, _, error_column, _ = _phase_columns(phase)
    lease_column, started_column = _phase_lease_columns(phase)
    lease_token = uuid.uuid4().hex
    started_at = now_utc()
    with connect_database(state_db) as connection:
        row = connection.execute(
            f"SELECT {state_column}, {attempts_column} FROM records "
            "WHERE run_id = ? AND source_record_id = ?",
            (run_id, source_record_id),
        ).fetchone()
        if row is None:
            raise RuntimeError("record is not planned")
        old_state = str(row[state_column])
        if old_state not in {"pending", "api_error", "invalid"}:
            raise RuntimeError(
                f"record cannot be claimed from state={old_state}"
            )
        attempt_no = int(row[attempts_column]) + 1
        cursor = connection.execute(
            f"UPDATE records SET {state_column}='running', "
            f"{attempts_column}=?, {error_column}=NULL, {lease_column}=?, "
            f"{started_column}=?, updated_at=? "
            f"WHERE run_id=? AND source_record_id=? AND {state_column}=? "
            f"AND {attempts_column}=?",
            (
                attempt_no,
                lease_token,
                started_at,
                started_at,
                run_id,
                source_record_id,
                old_state,
                int(row[attempts_column]),
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("record claim lost to another worker/process")
        connection.execute(
            """
            INSERT INTO api_attempts (
                lease_token, run_id, source_record_id, phase, attempt_no,
                state, started_at
            ) VALUES (?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                lease_token,
                run_id,
                source_record_id,
                phase,
                attempt_no,
                started_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO events (
                run_id, source_record_id, phase, old_state, new_state,
                details_json, created_at
            ) VALUES (?, ?, ?, ?, 'running', ?, ?)
            """,
            (
                run_id,
                source_record_id,
                phase,
                old_state,
                json.dumps({"attempt_no": attempt_no, "lease_token": lease_token}),
                started_at,
            ),
        )
    return attempt_no, lease_token


def save_failure(
    *,
    state_db: Path,
    run_id: str,
    source_record_id: str,
    phase: Phase,
    attempt_no: int,
    lease_token: str,
    state: Literal["api_error", "invalid"],
    error: str,
    meta: dict[str, Any] | None = None,
) -> None:
    state_column, _, _, error_column, _ = _phase_columns(phase)
    lease_column, started_column = _phase_lease_columns(phase)
    safe_error = error[:2000]
    finished_at = now_utc()
    safe_meta = meta or {}
    usage_json = (
        json.dumps(safe_meta.get("usage"), ensure_ascii=False, allow_nan=False)
        if safe_meta.get("usage") is not None
        else None
    )
    with connect_database(state_db) as connection:
        cursor = connection.execute(
            f"UPDATE records SET {state_column}=?, {error_column}=?, "
            f"{lease_column}=NULL, {started_column}=NULL, updated_at=? "
            f"WHERE run_id=? AND source_record_id=? AND {state_column}='running' "
            f"AND {lease_column}=?",
            (
                state,
                safe_error,
                finished_at,
                run_id,
                source_record_id,
                lease_token,
            ),
        )
        ledger_state = state if cursor.rowcount == 1 else "discarded_stale_failure"
        connection.execute(
            """
            UPDATE api_attempts SET state=?, error=?, model=COALESCE(?, model),
                finish_reason=COALESCE(?, finish_reason),
                usage_json=COALESCE(?, usage_json), finished_at=?
            WHERE lease_token=? AND attempt_no=?
            """,
            (
                ledger_state,
                safe_error,
                safe_meta.get("model"),
                safe_meta.get("finish_reason"),
                usage_json,
                finished_at,
                lease_token,
                attempt_no,
            ),
        )
        if cursor.rowcount != 1:
            return
        connection.execute(
            """
            INSERT INTO events (
                run_id, source_record_id, phase, old_state, new_state,
                details_json, created_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                run_id,
                source_record_id,
                phase,
                state,
                json.dumps({"error": safe_error}, ensure_ascii=False),
                finished_at,
            ),
        )


def save_success(
    *,
    state_db: Path,
    run_id: str,
    source_record_id: str,
    phase: Phase,
    attempt_no: int,
    lease_token: str,
    task: dict[str, Any],
    response: SemanticAuditResponse,
    meta: dict[str, Any],
) -> str:
    state_column, _, response_column, error_column, task_column = _phase_columns(phase)
    lease_column, started_column = _phase_lease_columns(phase)
    digest = response_sha256(response)
    response_json = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with connect_database(state_db) as connection:
        current = connection.execute(
            f"SELECT {state_column}, {task_column}, {lease_column}, "
            f"{'primary_attempts' if phase == 'primary' else 'review_attempts'} AS attempts "
            "FROM records WHERE run_id=? AND source_record_id=?",
            (run_id, source_record_id),
        ).fetchone()
        if current is None:
            raise RuntimeError("record disappeared before response persistence")
        if current[state_column] != "running" or current[lease_column] != lease_token:
            connection.execute(
                "UPDATE api_attempts SET state='discarded_stale_success', "
                "finished_at=? WHERE lease_token=?",
                (now_utc(), lease_token),
            )
            raise RuntimeError("response lease no longer owns the record")
        if current[task_column] != task["task_sha256"]:
            raise RuntimeError("response task hash does not match frozen run plan")
        if int(current["attempts"]) != attempt_no:
            raise RuntimeError("response attempt number does not match claimed attempt")
        connection.execute(
            """
            INSERT OR IGNORE INTO responses (
                response_sha256, run_id, source_record_id, phase, attempt_no,
                task_sha256, model, finish_reason, usage_json, response_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                run_id,
                source_record_id,
                phase,
                attempt_no,
                task["task_sha256"],
                meta.get("model"),
                meta.get("finish_reason"),
                json.dumps(meta.get("usage"), ensure_ascii=False, allow_nan=False),
                response_json,
                now_utc(),
            ),
        )
        cursor = connection.execute(
            f"UPDATE records SET {state_column}='done', {response_column}=?, "
            f"{error_column}=NULL, {lease_column}=NULL, {started_column}=NULL, "
            f"updated_at=? WHERE run_id=? AND source_record_id=? "
            f"AND {state_column}='running' AND {lease_column}=?",
            (digest, now_utc(), run_id, source_record_id, lease_token),
        )
        if cursor.rowcount != 1:
            connection.execute(
                "UPDATE api_attempts SET state='discarded_stale_success', "
                "finished_at=? WHERE lease_token=?",
                (now_utc(), lease_token),
            )
            raise RuntimeError("response lease lost during persistence")
        connection.execute(
            """
            UPDATE api_attempts SET
                state='done', response_sha256=?, model=?, finish_reason=?,
                usage_json=?, finished_at=?
            WHERE lease_token=? AND attempt_no=?
            """,
            (
                digest,
                meta.get("model"),
                meta.get("finish_reason"),
                json.dumps(meta.get("usage"), ensure_ascii=False, allow_nan=False),
                now_utc(),
                lease_token,
                attempt_no,
            ),
        )
        if phase == "primary" and response_requires_review(response):
            connection.execute(
                "UPDATE records SET requires_review=1 WHERE run_id=? AND source_record_id=?",
                (run_id, source_record_id),
            )
        connection.execute(
            """
            INSERT INTO events (
                run_id, source_record_id, phase, old_state, new_state,
                details_json, created_at
            ) VALUES (?, ?, ?, 'running', 'done', ?, ?)
            """,
            (
                run_id,
                source_record_id,
                phase,
                json.dumps({"response_sha256": digest}),
                now_utc(),
            ),
        )
    return digest


def reset_stale_running(state_db: Path, run_id: str, phase: Phase) -> int:
    state_column, _, _, _, _ = _phase_columns(phase)
    lease_column, started_column = _phase_lease_columns(phase)
    reset_count = 0
    with connect_database(state_db) as connection:
        connection.execute("BEGIN IMMEDIATE")
        run = connection.execute(
            "SELECT config_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise SystemExit(f"unknown run_id: {run_id}")
        try:
            run_config = json.loads(run["config_json"])
            timeout_seconds = float(
                run_config.get("request_config", {}).get("timeout_seconds", 120.0)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("run request timeout metadata is invalid") from exc
        stale_after_seconds = max(
            float(MINIMUM_STALE_LEASE_SECONDS),
            timeout_seconds + float(STALE_LEASE_GRACE_SECONDS),
        )
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        ).isoformat()
        rows = connection.execute(
            f"SELECT source_record_id, {lease_column} AS lease_token "
            f"FROM records WHERE run_id=? AND {state_column}='running' "
            f"AND {started_column} < ?",
            (run_id, cutoff),
        ).fetchall()
        for row in rows:
            lease_token = row["lease_token"]
            reset_at = now_utc()
            cursor = connection.execute(
                f"UPDATE records SET {state_column}='pending', {lease_column}=NULL, "
                f"{started_column}=NULL, updated_at=? WHERE run_id=? "
                f"AND source_record_id=? AND {state_column}='running' "
                f"AND {lease_column}=?",
                (reset_at, run_id, row["source_record_id"], lease_token),
            )
            if cursor.rowcount != 1:
                continue
            connection.execute(
                "UPDATE api_attempts SET state='abandoned_stale', finished_at=? "
                "WHERE lease_token=? AND state='running'",
                (reset_at, lease_token),
            )
            reset_count += 1
    return reset_count


def select_work(
    *,
    state_db: Path,
    run_id: str,
    phase: Phase,
    max_attempts: int,
    source_record_id: str | None,
    limit: int | None,
) -> list[sqlite3.Row]:
    state_column, attempts_column, _, _, _ = _phase_columns(phase)
    clauses = [
        "run_id = ?",
        f"{state_column} IN ('pending', 'api_error', 'invalid')",
        f"{attempts_column} < ?",
    ]
    parameters: list[Any] = [run_id, max_attempts]
    if phase == "blind_review":
        clauses.extend(["requires_review = 1", "primary_state = 'done'"])
    if source_record_id:
        clauses.append("source_record_id = ?")
        parameters.append(source_record_id)
    query = "SELECT * FROM records WHERE " + " AND ".join(clauses)
    query += " ORDER BY record_index"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    with connect_database(state_db) as connection:
        return list(connection.execute(query, parameters).fetchall())


def process_one(
    *,
    state_db: Path,
    run_id: str,
    record: dict[str, Any],
    schema: dict[str, Any],
    dataset_sha256: str,
    phase: Phase,
    config: DeepSeekConfig,
) -> tuple[str, str, str]:
    source_record_id = str(record.get("source_record_id") or "")
    task = build_semantic_audit_task(
        record, schema, dataset_sha256=dataset_sha256, phase=phase
    )
    client = DeepSeekRepairClient(config)
    attempt_no, lease_token = mark_running(
        state_db, run_id, source_record_id, phase
    )
    try:
        response, meta = client.propose_model(
            task,
            system_prompt=SEMANTIC_AUDIT_SYSTEM_PROMPT,
            response_model=SemanticAuditResponse,
        )
        require_valid_semantic_audit_response(response, task)
        digest = save_success(
            state_db=state_db,
            run_id=run_id,
            source_record_id=source_record_id,
            phase=phase,
            attempt_no=attempt_no,
            lease_token=lease_token,
            task=task,
            response=response,
            meta=meta,
        )
        return source_record_id, "done", digest
    except SemanticAuditValidationError as exc:
        save_failure(
            state_db=state_db,
            run_id=run_id,
            source_record_id=source_record_id,
            phase=phase,
            attempt_no=attempt_no,
            lease_token=lease_token,
            state="invalid",
            error=f"{type(exc).__name__}: {exc}",
            meta=meta,
        )
        return source_record_id, "invalid", str(exc)
    except DeepSeekAPIError as exc:
        state: Literal["api_error", "invalid"] = (
            "invalid" if "invalid structured response" in str(exc) else "api_error"
        )
        save_failure(
            state_db=state_db,
            run_id=run_id,
            source_record_id=source_record_id,
            phase=phase,
            attempt_no=attempt_no,
            lease_token=lease_token,
            state=state,
            error=f"{type(exc).__name__}: {exc}",
            meta=exc.meta,
        )
        return source_record_id, state, str(exc)
    except Exception as exc:  # defensive: keep the run resumable
        save_failure(
            state_db=state_db,
            run_id=run_id,
            source_record_id=source_record_id,
            phase=phase,
            attempt_no=attempt_no,
            lease_token=lease_token,
            state="api_error",
            error=f"{type(exc).__name__}: {exc}",
        )
        return source_record_id, "api_error", f"{type(exc).__name__}: {exc}"


def process_with_retries(
    *,
    remaining_attempts: int,
    batch_stop_event: threading.Event | None = None,
    **kwargs: Any,
) -> tuple[str, str, str]:
    source_record_id = str(kwargs["record"].get("source_record_id") or "")
    cancelled = (
        source_record_id,
        "cancelled",
        "batch stopped after a non-retryable API error",
    )
    if batch_stop_event is not None and batch_stop_event.is_set():
        return cancelled
    result = (source_record_id, "api_error", "not run")
    for retry_index in range(remaining_attempts):
        if batch_stop_event is not None and batch_stop_event.is_set():
            return cancelled
        result = process_one(**kwargs)
        if result[1] == "done":
            return result
        detail = result[2]
        non_retryable_statuses = (400, 401, 402, 403, 404, 405, 413, 422)
        if result[1] == "api_error" and any(
            f"status={status}" in detail for status in non_retryable_statuses
        ):
            if batch_stop_event is not None:
                batch_stop_event.set()
            return result
        if retry_index + 1 < remaining_attempts:
            delay = min(8.0, float(2**retry_index))
            if batch_stop_event is not None:
                if batch_stop_event.wait(delay):
                    return cancelled
            else:
                time.sleep(delay)
    return result


def run_phase(
    *,
    state_db: Path,
    run_id: str,
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
    phase: Phase,
    config: DeepSeekConfig,
    workers: int,
    max_attempts: int,
    source_record_id: str | None,
    limit: int | None,
) -> dict[str, int]:
    reset_stale_running(state_db, run_id, phase)
    state_column, attempts_column, _, _, _ = _phase_columns(phase)
    exhausted_clauses = [
        "run_id=?",
        f"{state_column} IN ('pending', 'api_error', 'invalid')",
        f"{attempts_column} >= ?",
    ]
    exhausted_parameters: list[Any] = [run_id, max_attempts]
    if phase == "blind_review":
        exhausted_clauses.extend(["requires_review=1", "primary_state='done'"])
    with connect_database(state_db) as connection:
        exhausted = connection.execute(
            "SELECT COUNT(*) FROM records WHERE "
            + " AND ".join(exhausted_clauses),
            exhausted_parameters,
        ).fetchone()[0]
    if exhausted:
        return {"exhausted": int(exhausted)}
    work = select_work(
        state_db=state_db,
        run_id=run_id,
        phase=phase,
        max_attempts=max_attempts,
        source_record_id=source_record_id,
        limit=limit,
    )
    by_id = {str(record.get("source_record_id") or ""): record for record in records}
    missing = [row["source_record_id"] for row in work if row["source_record_id"] not in by_id]
    if missing:
        raise SystemExit(f"planned records missing from current input: {missing[:3]}")
    counts: Counter[str] = Counter()
    total = len(work)
    if total == 0:
        return {}
    with _PRINT_LOCK:
        print(f"{phase}: scheduling {total} records with {workers} workers", flush=True)
    batch_stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_with_retries,
                remaining_attempts=max_attempts
                - int(
                    row[
                        "primary_attempts"
                        if phase == "primary"
                        else "review_attempts"
                    ]
                ),
                batch_stop_event=batch_stop_event,
                state_db=state_db,
                run_id=run_id,
                record=by_id[row["source_record_id"]],
                schema=schema,
                dataset_sha256=dataset_sha256,
                phase=phase,
                config=config,
            ): row["source_record_id"]
            for row in work
        }
        completed = 0
        for future in as_completed(futures):
            completed += 1
            source_id = futures[future]
            try:
                _, state, detail = future.result()
            except Exception as exc:
                state = "api_error"
                detail = f"{type(exc).__name__}: {exc}"
            counts[state] += 1
            with _PRINT_LOCK:
                if state == "done":
                    print(f"{phase}: {completed}/{total} done {source_id}", flush=True)
                else:
                    print(
                        f"{phase}: {completed}/{total} {state} {source_id}: {detail[:240]}",
                        flush=True,
                    )
    return dict(sorted(counts.items()))


def load_response(connection: sqlite3.Connection, digest: str | None) -> SemanticAuditResponse | None:
    if not digest:
        return None
    row = connection.execute(
        "SELECT response_json FROM responses WHERE response_sha256 = ?", (digest,)
    ).fetchone()
    if row is None:
        return None
    return SemanticAuditResponse.model_validate_json(row["response_json"])


def finalize_consensus(state_db: Path, run_id: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    with connect_database(state_db) as connection:
        rows = connection.execute(
            "SELECT * FROM records WHERE run_id = ? ORDER BY record_index", (run_id,)
        ).fetchall()
        for row in rows:
            final_status: str
            if row["primary_state"] != "done":
                final_status = "pending_primary"
            elif row["requires_review"] and row["review_state"] != "done":
                final_status = "pending_review"
            else:
                primary = load_response(connection, row["primary_response_sha256"])
                review = load_response(connection, row["review_response_sha256"])
                if primary is None:
                    final_status = "invalid_cache"
                elif row["requires_review"] and review is None:
                    final_status = "invalid_cache"
                elif review is None:
                    final_status = (
                        "unresolved" if primary.unresolved else "ready_to_patch"
                        if primary.proposed_operations
                        else "clean"
                    )
                elif response_semantic_signature(primary) != response_semantic_signature(review):
                    final_status = "conflict"
                elif response_has_unresolved_semantics(
                    primary
                ) or response_has_unresolved_semantics(review):
                    final_status = "unresolved"
                elif primary.proposed_operations:
                    if all(
                        operation.confidence >= 0.95
                        for operation in primary.proposed_operations
                    ) and all(
                        operation.confidence >= 0.95
                        for operation in review.proposed_operations
                    ):
                        final_status = "ready_to_patch"
                    else:
                        final_status = "unresolved"
                else:
                    final_status = "clean"
            counts[final_status] += 1
            connection.execute(
                "UPDATE records SET final_status=?, updated_at=? "
                "WHERE run_id=? AND source_record_id=?",
                (final_status, now_utc(), run_id, row["source_record_id"]),
            )
    return counts


def reset_records_for_retry(
    *,
    state_db: Path,
    run_id: str,
    mode: Literal["blocked", "failed"],
    source_record_id: str | None = None,
) -> int:
    """Reset only explicitly blocked/failed records while retaining response history."""

    clauses = ["run_id=?"]
    parameters: list[Any] = [run_id]
    if source_record_id:
        clauses.append("source_record_id=?")
        parameters.append(source_record_id)
    if mode == "blocked":
        clauses.append("final_status IN ('conflict', 'unresolved', 'invalid_cache')")
    else:
        clauses.append(
            "(primary_state IN ('api_error', 'invalid') "
            "OR review_state IN ('api_error', 'invalid'))"
        )
    where = " AND ".join(clauses)
    reset_count = 0
    with connect_database(state_db) as connection:
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            f"SELECT * FROM records WHERE {where} ORDER BY record_index", parameters
        ).fetchall()
        for row in rows:
            if (
                row["primary_state"] == "running"
                or row["review_state"] == "running"
                or row["primary_lease_token"] is not None
                or row["review_lease_token"] is not None
            ):
                raise SystemExit(
                    f"cannot reset a record with an active API lease: {row['source_record_id']}"
                )
            if mode == "blocked":
                cursor = connection.execute(
                    """
                    UPDATE records SET
                        primary_state='pending', review_state='pending',
                        primary_attempts=0, review_attempts=0,
                        primary_response_sha256=NULL, review_response_sha256=NULL,
                        primary_error=NULL, review_error=NULL, final_status=NULL,
                        primary_lease_token=NULL, review_lease_token=NULL,
                        primary_started_at=NULL, review_started_at=NULL,
                        updated_at=?
                    WHERE run_id=? AND source_record_id=?
                        AND primary_state=? AND review_state=?
                        AND primary_lease_token IS ? AND review_lease_token IS ?
                        AND final_status=?
                    """,
                    (
                        now_utc(),
                        run_id,
                        row["source_record_id"],
                        row["primary_state"],
                        row["review_state"],
                        row["primary_lease_token"],
                        row["review_lease_token"],
                        row["final_status"],
                    ),
                )
            else:
                updates: list[str] = ["final_status=NULL", "updated_at=?"]
                update_values: list[Any] = [now_utc()]
                if row["primary_state"] in {"api_error", "invalid"}:
                    updates.extend(
                        [
                            "primary_state='pending'",
                            "primary_attempts=0",
                            "primary_error=NULL",
                            "primary_response_sha256=NULL",
                            "primary_lease_token=NULL",
                            "primary_started_at=NULL",
                        ]
                    )
                if row["review_state"] in {"api_error", "invalid"}:
                    updates.extend(
                        [
                            "review_state='pending'",
                            "review_attempts=0",
                            "review_error=NULL",
                            "review_response_sha256=NULL",
                            "review_lease_token=NULL",
                            "review_started_at=NULL",
                        ]
                    )
                cursor = connection.execute(
                    f"UPDATE records SET {', '.join(updates)} "
                    "WHERE run_id=? AND source_record_id=? "
                    "AND primary_state=? AND review_state=? "
                    "AND primary_lease_token IS ? AND review_lease_token IS ?",
                    (
                        *update_values,
                        run_id,
                        row["source_record_id"],
                        row["primary_state"],
                        row["review_state"],
                        row["primary_lease_token"],
                        row["review_lease_token"],
                    ),
                )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"reset lease/state precondition changed: {row['source_record_id']}"
                )
            connection.execute(
                """
                INSERT INTO events (
                    run_id, source_record_id, phase, old_state, new_state,
                    details_json, created_at
                ) VALUES (?, ?, NULL, ?, 'pending', ?, ?)
                """,
                (
                    run_id,
                    row["source_record_id"],
                    str(row["final_status"] or "failed"),
                    json.dumps({"reset_mode": mode}),
                    now_utc(),
                ),
            )
            reset_count += 1
    return reset_count


def reset_record_for_retry(
    *,
    state_db: Path,
    run_id: str,
    source_record_id: str,
) -> None:
    """Force both phases of one planned record back to a pristine pending state."""

    with connect_database(state_db) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM records WHERE run_id=? AND source_record_id=?",
            (run_id, source_record_id),
        ).fetchone()
        if row is None:
            raise SystemExit(
                f"source_record_id is not planned for run {run_id}: {source_record_id}"
            )
        if row["primary_state"] == "running" or row["review_state"] == "running":
            raise SystemExit(
                "cannot reset-record while an API request holds an active lease"
            )
        connection.execute(
            """
            UPDATE records SET
                primary_state='pending', review_state='pending',
                primary_attempts=0, review_attempts=0,
                primary_response_sha256=NULL, review_response_sha256=NULL,
                primary_error=NULL, review_error=NULL, final_status=NULL,
                primary_lease_token=NULL, review_lease_token=NULL,
                primary_started_at=NULL, review_started_at=NULL,
                updated_at=?
            WHERE run_id=? AND source_record_id=?
            """,
            (now_utc(), run_id, source_record_id),
        )
        connection.execute(
            """
            INSERT INTO events (
                run_id, source_record_id, phase, old_state, new_state,
                details_json, created_at
            ) VALUES (?, ?, NULL, ?, 'pending', ?, ?)
            """,
            (
                run_id,
                source_record_id,
                str(row["final_status"] or "forced_record_reset"),
                json.dumps(
                    {
                        "reset_mode": "record",
                        "previous_primary_state": row["primary_state"],
                        "previous_review_state": row["review_state"],
                        "previous_final_status": row["final_status"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                now_utc(),
            ),
        )


def status_report(state_db: Path, run_id: str) -> dict[str, Any]:
    with connect_database(state_db) as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise SystemExit(f"unknown run_id: {run_id}")
        rows = connection.execute(
            "SELECT * FROM records WHERE run_id = ? ORDER BY record_index", (run_id,)
        ).fetchall()
        responses = connection.execute(
            """
            SELECT 'primary' AS phase, response.response_json
            FROM records AS record
            JOIN responses AS response
              ON response.response_sha256 = record.primary_response_sha256
            WHERE record.run_id = ?
            UNION ALL
            SELECT 'blind_review' AS phase, response.response_json
            FROM records AS record
            JOIN responses AS response
              ON response.response_sha256 = record.review_response_sha256
            WHERE record.run_id = ?
            """,
            (run_id, run_id),
        ).fetchall()
        api_attempt_rows = connection.execute(
            "SELECT phase, state, usage_json FROM api_attempts WHERE run_id=?",
            (run_id,),
        ).fetchall()
        actionable_records: list[dict[str, Any]] = []
        for row in rows:
            final_status = str(row["final_status"] or "not_finalized")
            has_phase_error = row["primary_state"] in {"api_error", "invalid"} or row[
                "review_state"
            ] in {"api_error", "invalid"}
            is_semantically_blocked = final_status in {
                "conflict",
                "unresolved",
                "invalid_cache",
            }
            if not has_phase_error and not is_semantically_blocked:
                continue
            item: dict[str, Any] = {
                "source_record_id": row["source_record_id"],
                "record_index": row["record_index"],
                "final_status": final_status,
                "primary_state": row["primary_state"],
                "review_state": row["review_state"],
                "primary_attempts": row["primary_attempts"],
                "review_attempts": row["review_attempts"],
                "primary_response_sha256": row["primary_response_sha256"],
                "review_response_sha256": row["review_response_sha256"],
                "primary_error": row["primary_error"],
                "review_error": row["review_error"],
            }
            if final_status == "conflict":
                primary = load_response(connection, row["primary_response_sha256"])
                review = load_response(connection, row["review_response_sha256"])
                if primary is not None and review is not None:
                    primary_signature = response_semantic_signature(primary)
                    review_signature = response_semantic_signature(review)
                    item["disagreement_sections"] = [
                        key
                        for key in primary_signature
                        if primary_signature[key] != review_signature[key]
                    ]
            actionable_records.append(item)
    primary_states = Counter(str(row["primary_state"]) for row in rows)
    review_states = Counter(str(row["review_state"]) for row in rows)
    risk_tiers = Counter(str(row["risk_tier"]) for row in rows)
    final_statuses = Counter(str(row["final_status"] or "not_finalized") for row in rows)
    coverage = {
        "primary": {"responses": 0, "entities": 0, "relations": 0, "cues": 0, "dimensions": 0},
        "blind_review": {"responses": 0, "entities": 0, "relations": 0, "cues": 0, "dimensions": 0},
    }
    usage: Counter[str] = Counter()
    api_attempt_states: Counter[str] = Counter()
    api_attempt_phases: Counter[str] = Counter()
    token_usage: Counter[str] = Counter()
    attempts_with_usage = 0
    for row in api_attempt_rows:
        api_attempt_states[str(row["state"])] += 1
        api_attempt_phases[str(row["phase"])] += 1
        if not row["usage_json"]:
            continue
        try:
            parsed_usage = json.loads(row["usage_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(parsed_usage, dict):
            continue
        attempts_with_usage += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = parsed_usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                token_usage[key] += value
    for row in responses:
        phase = str(row["phase"])
        response = SemanticAuditResponse.model_validate_json(row["response_json"])
        target = coverage[phase]
        target["responses"] += 1
        target["entities"] += len(response.entity_assessments)
        target["relations"] += len(response.relation_assessments)
        target["cues"] += len(response.cue_assessments)
        target["dimensions"] += len(response.dimension_assessments)
    for row in rows:
        usage["primary_attempts"] += int(row["primary_attempts"])
        usage["review_attempts"] += int(row["review_attempts"])
    expected_primary = len(rows)
    expected_review = sum(int(row["requires_review"]) for row in rows)
    run_config = json.loads(run["config_json"])
    expected_entities = int(run_config.get("entity_count", 0))
    expected_relations = int(run_config.get("relation_count", 0))
    expected_cues = int(run_config.get("cue_count", 0))
    expected_dimensions = int(run_config.get("dimension_count", 0))
    complete = (
        primary_states == Counter({"done": expected_primary})
        and review_states.get("done", 0) == expected_review
        and sum(
            count
            for state, count in review_states.items()
            if state not in {"done", "pending"}
        )
        == 0
        and sum(final_statuses.values()) == expected_primary
        and set(final_statuses).issubset({"clean", "ready_to_patch"})
        and coverage["primary"]
        == {
            "responses": expected_primary,
            "entities": expected_entities,
            "relations": expected_relations,
            "cues": expected_cues,
            "dimensions": expected_dimensions,
        }
        and coverage["blind_review"]
        == {
            "responses": expected_review,
            "entities": expected_entities if expected_review == expected_primary else coverage["blind_review"]["entities"],
            "relations": expected_relations if expected_review == expected_primary else coverage["blind_review"]["relations"],
            "cues": expected_cues if expected_review == expected_primary else coverage["blind_review"]["cues"],
            "dimensions": expected_dimensions if expected_review == expected_primary else coverage["blind_review"]["dimensions"],
        }
    )
    return {
        "protocol_version": run["protocol_version"],
        "run_id": run_id,
        "generated_at": now_utc(),
        "input_path": run["input_path"],
        "dataset_sha256": run["dataset_sha256"],
        "schema_sha256": run["schema_sha256"],
        "prompt_sha256": run["prompt_sha256"],
        "task_manifest_sha256": run["task_manifest_sha256"],
        "request_config_sha256": run["request_config_sha256"],
        "request_config": run_config.get("request_config", {}),
        "model": run["model"],
        "expected": {
            "records": len(rows),
            "primary_responses": expected_primary,
            "blind_review_responses": expected_review,
            "entities_per_full_pass": expected_entities,
            "relations_per_full_pass": expected_relations,
            "cues_per_full_pass": expected_cues,
            "dimensions_per_full_pass": expected_dimensions,
        },
        "risk_tiers": dict(sorted(risk_tiers.items())),
        "primary_states": dict(sorted(primary_states.items())),
        "review_states": dict(sorted(review_states.items())),
        "final_statuses": dict(sorted(final_statuses.items())),
        "coverage": coverage,
        "current_retry_window_attempts": dict(usage),
        "api_request_ledger": {
            "total_attempted_requests": len(api_attempt_rows),
            "by_phase": dict(sorted(api_attempt_phases.items())),
            "by_state": dict(sorted(api_attempt_states.items())),
            "attempts_with_provider_usage": attempts_with_usage,
            "attempts_without_provider_usage": len(api_attempt_rows)
            - attempts_with_usage,
            "known_token_usage": dict(sorted(token_usage.items())),
            "note": (
                "Request count is append-only across resets. Token totals include only "
                "responses for which the provider returned usage metadata."
            ),
        },
        "actionable_records": actionable_records,
        "semantic_audit_complete": complete,
        "training_unlocked": False,
        "training_unlock_note": (
            "Semantic audit completion is necessary but not sufficient; deterministic "
            "patch application, WHO validation, sanitization, conversion, and strict "
            "training-data audit must also pass."
        ),
    }


def compile_patch_bundle(
    *,
    state_db: Path,
    run_id: str,
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
) -> dict[str, Any]:
    statuses = finalize_consensus(state_db, run_id)
    blocking = {
        key: count
        for key, count in statuses.items()
        if key not in {"clean", "ready_to_patch"} and count
    }
    if blocking:
        raise SemanticPatchError(
            f"cannot compile while audit records are blocked: {dict(sorted(blocking.items()))}"
        )
    by_id = {str(record.get("source_record_id") or ""): record for record in records}
    patches: list[dict[str, Any]] = []
    with connect_database(state_db) as connection:
        rows = connection.execute(
            "SELECT * FROM records WHERE run_id=? ORDER BY record_index", (run_id,)
        ).fetchall()
        for row in rows:
            record = by_id.get(str(row["source_record_id"]))
            if record is None:
                raise SemanticPatchError(
                    f"record is missing from current input: {row['source_record_id']}"
                )
            primary = load_response(connection, row["primary_response_sha256"])
            review = load_response(connection, row["review_response_sha256"])
            if primary is None or review is None:
                raise SemanticPatchError(
                    f"consensus responses are missing: {row['source_record_id']}"
                )
            try:
                patch = compile_consensus_patch(
                    record=record,
                    schema=schema,
                    dataset_sha256=dataset_sha256,
                    primary=primary,
                    review=review,
                )
            except SemanticPatchError as exc:
                raise SemanticPatchError(
                    f"record {row['source_record_id']} failed patch compilation: {exc}"
                ) from exc
            patches.append(patch)
    if len(patches) != len(records):
        raise SemanticPatchError(
            f"patch coverage mismatch: patches={len(patches)} records={len(records)}"
        )
    bundle = {
        "bundle_protocol_version": "semantic-consensus-patch-bundle-v1",
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "record_count": len(records),
        "patches": patches,
    }
    bundle["bundle_sha256"] = sha256_value(bundle)
    return bundle


def apply_patch_bundle(
    *,
    bundle: dict[str, Any],
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    dataset_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if bundle.get("bundle_protocol_version") != "semantic-consensus-patch-bundle-v1":
        raise SemanticPatchError("unsupported patch-bundle protocol")
    payload = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if bundle.get("bundle_sha256") != sha256_value(payload):
        raise SemanticPatchError("patch-bundle hash mismatch")
    if bundle.get("dataset_sha256") != dataset_sha256:
        raise SemanticPatchError("patch-bundle dataset hash mismatch")
    patches = bundle.get("patches")
    if not isinstance(patches, list) or len(patches) != len(records):
        raise SemanticPatchError("patch-bundle record coverage mismatch")
    patch_by_id = {
        str(patch.get("source_record_id") or ""): patch
        for patch in patches
        if isinstance(patch, dict)
    }
    if len(patch_by_id) != len(records):
        raise SemanticPatchError("patch-bundle source IDs are missing or duplicated")
    repaired_records: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        source_id = str(record.get("source_record_id") or "")
        patch = patch_by_id.get(source_id)
        if patch is None:
            raise SemanticPatchError(f"record has no consensus patch: {source_id}")
        try:
            repaired, audit = apply_consensus_patch(
                record=record,
                schema=schema,
                dataset_sha256=dataset_sha256,
                patch=patch,
            )
        except SemanticPatchError as exc:
            raise SemanticPatchError(
                f"record {source_id} failed patch application: {exc}"
            ) from exc
        repaired_records.append(repaired)
        applications.append(
            {"record_index": index, "source_record_id": source_id, **audit}
        )
    report = {
        "protocol_version": "semantic-clean-application-v1",
        "run_id": bundle.get("run_id"),
        "dataset_sha256": dataset_sha256,
        "bundle_sha256": bundle["bundle_sha256"],
        "record_count": len(records),
        "records_applied": len(applications),
        "operation_count": sum(item["operations_applied"] for item in applications),
        "new_entity_count": sum(len(item["new_entity_ids"]) for item in applications),
        "removed_entity_count": sum(
            len(item["removed_entity_ids"]) for item in applications
        ),
        "collapse_count": sum(len(item["collapsed"]) for item in applications),
        "failures": [],
        "applications": applications,
    }
    return repaired_records, report


def find_run_id(state_db: Path, requested: str | None) -> str:
    if requested:
        return requested
    initialize_database(state_db)
    with connect_database(state_db) as connection:
        rows = connection.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC"
        ).fetchall()
    if len(rows) != 1:
        raise SystemExit("--run-id is required when the state database has zero or multiple runs")
    return str(rows[0]["run_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "primary",
            "review",
            "all",
            "finalize",
            "compile",
            "apply",
            "reset-record",
            "reset-blocked",
            "reset-failed",
            "status",
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schema-pointer", type=Path, default=ROOT / "schemas/current.json")
    parser.add_argument("--state-db", type=Path, default=DEFAULT_STATE_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--patch-output", type=Path, default=DEFAULT_PATCH_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CLEAN_OUTPUT)
    parser.add_argument(
        "--application-report", type=Path, default=DEFAULT_APPLICATION_REPORT
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-api-attempts", type=int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="override DEEPSEEK_MAX_TOKENS from .env (default 32768)",
    )
    parser.add_argument("--source-record-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--review-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="blind-review all records (default); --no-review-all keeps High + dynamic review",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.max_api_attempts < 1 or (
        args.max_tokens is not None and args.max_tokens < 1
    ):
        raise SystemExit("--max-api-attempts and --max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.command == "reset-record" and not args.source_record_id:
        raise SystemExit("reset-record requires --source-record-id")

    initialize_database(args.state_db)
    if args.command == "status":
        run_id = find_run_id(args.state_db, args.run_id)
        report = status_report(args.state_db, run_id)
        write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    input_payload = load_json(args.input)
    records = record_list(input_payload)
    schema = load_schema(args.schema_pointer)
    dataset_sha256 = sha256_file(args.input)
    load_env_file(args.env_file)
    if args.max_tokens is None:
        try:
            args.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "32768"))
        except ValueError as exc:
            raise SystemExit("DEEPSEEK_MAX_TOKENS must be an integer") from exc
    if args.max_tokens < 1:
        raise SystemExit("DEEPSEEK_MAX_TOKENS must be positive")
    request_config = public_request_config(args.max_tokens)
    run_id = args.run_id
    if args.command in {"plan", "all"}:
        run_id = prepare_run(
            state_db=args.state_db,
            input_path=args.input,
            records=records,
            schema=schema,
            dataset_sha256=dataset_sha256,
            request_config=request_config,
            review_all=args.review_all,
        )
        print(f"planned run_id={run_id} records={len(records)}", flush=True)
    else:
        run_id = find_run_id(args.state_db, run_id)

    if args.command == "plan":
        report = status_report(args.state_db, run_id)
        write_json(args.report, report)
        return 0

    verify_run_context(
        state_db=args.state_db,
        run_id=run_id,
        records=records,
        schema=schema,
        dataset_sha256=dataset_sha256,
        request_config=request_config,
    )

    if args.command == "finalize":
        counts = finalize_consensus(args.state_db, run_id)
        report = status_report(args.state_db, run_id)
        write_json(args.report, report)
        print(f"final statuses: {dict(sorted(counts.items()))}")
        return 0 if report["semantic_audit_complete"] else 1

    if args.command == "reset-record":
        reset_record_for_retry(
            state_db=args.state_db,
            run_id=run_id,
            source_record_id=args.source_record_id,
        )
        write_json(args.report, status_report(args.state_db, run_id))
        print(f"reset mode=record source_record_id={args.source_record_id}")
        return 0

    if args.command in {"reset-blocked", "reset-failed"}:
        mode: Literal["blocked", "failed"] = (
            "blocked" if args.command == "reset-blocked" else "failed"
        )
        count = reset_records_for_retry(
            state_db=args.state_db,
            run_id=run_id,
            mode=mode,
            source_record_id=args.source_record_id,
        )
        write_json(args.report, status_report(args.state_db, run_id))
        print(f"reset mode={mode} records={count}")
        return 0

    if args.command == "compile":
        bundle = compile_patch_bundle(
            state_db=args.state_db,
            run_id=run_id,
            records=records,
            schema=schema,
            dataset_sha256=dataset_sha256,
        )
        if args.patch_output.resolve() == args.input.resolve():
            raise SystemExit("refusing to overwrite the frozen input with a patch bundle")
        write_json(args.patch_output, bundle)
        print(
            f"compiled patches={bundle['record_count']} bundle={bundle['bundle_sha256']}",
            flush=True,
        )
        return 0

    if args.command == "apply":
        if args.output.resolve() in {args.input.resolve(), args.patch_output.resolve()}:
            raise SystemExit("refusing to overwrite input or patch bundle")
        bundle = load_json(args.patch_output)
        repaired, application_report = apply_patch_bundle(
            bundle=bundle,
            records=records,
            schema=schema,
            dataset_sha256=dataset_sha256,
        )
        # Both artifacts are written only after every record passed its transaction.
        write_json(args.output, repaired)
        application_report["output_path"] = str(args.output.resolve())
        application_report["output_sha256"] = sha256_file(args.output)
        write_json(args.application_report, application_report)
        print(
            f"applied records={len(repaired)} operations={application_report['operation_count']}",
            flush=True,
        )
        return 0

    config = replace(DeepSeekConfig.from_env(), max_tokens=args.max_tokens)
    commands: list[Phase] = []
    if args.command in {"primary", "all"}:
        commands.append("primary")
    if args.command in {"review", "all"}:
        commands.append("blind_review")
    phase_failed = False
    for phase in commands:
        workers = args.workers or (4 if phase == "primary" else 2)
        results = run_phase(
            state_db=args.state_db,
            run_id=run_id,
            records=records,
            schema=schema,
            dataset_sha256=dataset_sha256,
            phase=phase,
            config=config,
            workers=workers,
            max_attempts=args.max_api_attempts,
            source_record_id=args.source_record_id,
            limit=args.limit,
        )
        print(f"{phase} results: {results}", flush=True)
        write_json(args.report, status_report(args.state_db, run_id))
        if any(state != "done" and count for state, count in results.items()):
            phase_failed = True
            break
    return 1 if phase_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

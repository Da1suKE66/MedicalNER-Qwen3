"""Constrained DeepSeek repair patches for Schema v2 records.

The model is never allowed to replace a complete graph.  It may only propose a
small set of operations which are checked against the immutable input record,
the source text, and the active schema before they can be applied.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema_v2 import (
    _matches_direction,
    graph_from_record,
    normalize_text,
    validate_record,
    write_json,
)


_JSON = json
REPAIR_PROTOCOL_VERSION = "deepseek-repair-patch-v2"


class DeepSeekAPIError(RuntimeError):
    """A safe-to-display API error which never contains credentials."""


class PatchValidationError(ValueError):
    """Raised when an LLM patch fails deterministic local checks."""


def _redact(value: Any, *secrets: str) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text


class _ResponseLike(Protocol):
    ok: bool
    status_code: int
    text: str

    def json(self) -> Any: ...


class _SessionLike(Protocol):
    def post(
        self,
        endpoint: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: tuple[float, float],
    ) -> _ResponseLike: ...


class _UrllibResponse:
    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text)


class _UrllibSession:
    """Small stdlib transport so the repair client adds no undeclared dependency."""

    def __init__(self, *, trust_environment_proxy: bool) -> None:
        handlers: list[Any] = []
        if not trust_environment_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def post(
        self,
        endpoint: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: tuple[float, float],
    ) -> _UrllibResponse:
        request = urllib.request.Request(
            endpoint,
            data=bytes(
                _JSON.dumps(json, ensure_ascii=False, allow_nan=False),
                "utf-8",
            ),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=max(timeout)) as response:
                return _UrllibResponse(response.status, response.read())
        except urllib.error.HTTPError as exc:
            return _UrllibResponse(exc.code, exc.read())


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_bounds(self) -> "Evidence":
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        return self


RepairOperationName = Literal[
    "replace_relation",
    "replace_relation_type",
    "swap_relation_endpoints",
    "add_relation",
    "remove_relation",
    "relabel_entity",
    "move_description_to_manifestations",
    "mark_manual_review",
]


class RepairOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: RepairOperationName
    relation_index: int | None = Field(default=None, ge=0)
    source_entity_id: str | None = None
    target_entity_id: str | None = None
    description_entity_id: str | None = None
    parent_entity_id: str | None = None
    expected_relation_type: str | None = None
    replacement_relation_type: str | None = None
    replacement_label: str | None = None
    manifestation_text: str | None = None
    evidence: Evidence | None = None
    reason: str = Field(min_length=1, max_length=800)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_operation_fields(self) -> "RepairOperation":
        indexed = {
            "replace_relation",
            "replace_relation_type",
            "swap_relation_endpoints",
            "remove_relation",
        }
        if self.op in indexed:
            if self.relation_index is None or not self.expected_relation_type:
                raise ValueError(f"{self.op} requires relation_index and expected_relation_type")
        if self.op == "replace_relation_type" and not self.replacement_relation_type:
            raise ValueError("replace_relation_type requires replacement_relation_type")
        if self.op == "replace_relation" and not (
            self.source_entity_id
            and self.target_entity_id
            and self.replacement_relation_type
        ):
            raise ValueError(
                "replace_relation requires source, target, and replacement relation type"
            )
        if self.op == "add_relation":
            if not (
                self.source_entity_id
                and self.target_entity_id
                and self.replacement_relation_type
            ):
                raise ValueError("add_relation requires source, target, and relation type")
        if self.op == "relabel_entity" and not (
            self.source_entity_id and self.replacement_label
        ):
            raise ValueError("relabel_entity requires source_entity_id and replacement_label")
        if self.op == "move_description_to_manifestations" and not (
            self.description_entity_id
            and self.parent_entity_id
            and self.manifestation_text
        ):
            raise ValueError(
                "move_description_to_manifestations requires child, parent, and text"
            )
        if self.op != "mark_manual_review" and self.evidence is None:
            raise ValueError(f"{self.op} requires exact source evidence")
        return self


class RepairPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_record_id: str
    schema_version: str
    record_sha256: str
    operations: list[RepairOperation] = Field(max_length=20)
    needs_human_review: bool
    review_reason: str | None = Field(default=None, max_length=1000)


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    thinking: Literal["enabled", "disabled"] = "disabled"
    reasoning_effort: Literal["high", "max"] = "high"
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    trust_environment_proxy: bool = False

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise DeepSeekAPIError("DEEPSEEK_API_KEY is missing")
        thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
        if thinking not in {"enabled", "disabled"}:
            raise DeepSeekAPIError("DEEPSEEK_THINKING must be enabled or disabled")
        effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "high").strip().lower()
        if effort not in {"high", "max"}:
            raise DeepSeekAPIError("DEEPSEEK_REASONING_EFFORT must be high or max")
        trust_proxy = os.getenv("DEEPSEEK_TRUST_ENV_PROXY", "false").lower()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip(
            "/"
        )
        if not base_url.startswith(("https://", "http://")):
            raise DeepSeekAPIError("DEEPSEEK_BASE_URL must be an HTTP(S) URL")
        try:
            max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
            timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "120"))
        except ValueError as exc:
            raise DeepSeekAPIError(
                "DEEPSEEK_MAX_TOKENS and DEEPSEEK_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if max_tokens < 1 or timeout_seconds <= 0:
            raise DeepSeekAPIError(
                "DEEPSEEK_MAX_TOKENS and DEEPSEEK_TIMEOUT_SECONDS must be positive"
            )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            thinking=thinking,  # type: ignore[arg-type]
            reasoning_effort=effort,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            trust_environment_proxy=trust_proxy in {"1", "true", "yes"},
        )


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_repair_task(
    record: dict[str, Any],
    schema: dict[str, Any],
    findings: dict[str, Any],
    *,
    request_thinking_mode: str = "disabled",
) -> dict[str, Any]:
    source_record_id = str(record.get("source_record_id") or "")
    if not source_record_id:
        raise PatchValidationError("record has no source_record_id")
    graph = graph_from_record(record)
    if graph is None:
        raise PatchValidationError("record has no graph")
    return {
        "repair_protocol_version": REPAIR_PROTOCOL_VERSION,
        "request_thinking_mode": request_thinking_mode,
        "source_record_id": source_record_id,
        "schema_version": str(record.get("schema_version") or ""),
        "record_sha256": record_sha256(record),
        "original_text": str(record.get("input") or ""),
        "graph": graph,
        "validator_findings": findings,
        "allowed_entity_labels": sorted(schema["entity_types"]),
        "allowed_relations": schema["relation_types"],
        "protected_fields": [
            "source_record_id",
            "source_code",
            "source_title",
            "source_release",
            "schema_version",
        ],
    }


SYSTEM_PROMPT = """You repair a medical knowledge graph using only the supplied source text.
Return one JSON object matching the requested patch shape. Never return or rewrite the
complete graph. Use only the listed operations and schema labels/relations. Every
mutating operation needs an exact quote and zero-based [start,end) offsets such that
original_text[start:end] equals the quote. Never modify protected source identity or
ICD provenance. Prefer mark_manual_review over guessing. Keep operations minimal.

Patch shape:
{
  "source_record_id": "same as input",
  "schema_version": "same as input",
  "record_sha256": "same as input",
  "operations": [{
    "op": "replace_relation|replace_relation_type|swap_relation_endpoints|add_relation|remove_relation|relabel_entity|move_description_to_manifestations|mark_manual_review",
    "relation_index": 0,
    "source_entity_id": null,
    "target_entity_id": null,
    "description_entity_id": null,
    "parent_entity_id": null,
    "expected_relation_type": null,
    "replacement_relation_type": null,
    "replacement_label": null,
    "manifestation_text": null,
    "evidence": {"quote": "exact text", "start": 0, "end": 10},
    "reason": "short justification",
    "confidence": 0.0
  }],
  "needs_human_review": false,
  "review_reason": null
}
Use null for unused optional fields. The operations array may be empty.

Required fields by operation (use these exact field names; never use aliases such as
relation_type, new_target, or entity_id):
- replace_relation: relation_index, expected_relation_type, source_entity_id,
  target_entity_id, replacement_relation_type, evidence.
- replace_relation_type: relation_index, expected_relation_type,
  replacement_relation_type, evidence.
- swap_relation_endpoints: relation_index, expected_relation_type, evidence.
- remove_relation: relation_index, expected_relation_type, evidence.
- add_relation: source_entity_id, target_entity_id, replacement_relation_type,
  evidence; relation_index and expected_relation_type must be null.
- mark_manual_review: all mutation fields and evidence must be null; explain why in
  review_reason. Prefer replace_relation when one invalid relation needs a corrected
  endpoint and/or type.
"""


class DeepSeekRepairClient:
    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        session: _SessionLike | None = None,
    ) -> None:
        self.config = config
        self.session = session or _UrllibSession(
            trust_environment_proxy=config.trust_environment_proxy
        )

    def propose_patch(self, task: dict[str, Any]) -> tuple[RepairPatch, dict[str, Any]]:
        endpoint = f"{self.config.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(task, ensure_ascii=False, allow_nan=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": self.config.thinking},
            "reasoning_effort": self.config.reasoning_effort,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        try:
            response = self.session.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=(10, self.config.timeout_seconds),
            )
        except (OSError, urllib.error.URLError) as exc:
            safe_endpoint = _redact(endpoint, self.config.api_key)
            safe_error = _redact(exc, self.config.api_key)
            raise DeepSeekAPIError(
                f"DeepSeek network error endpoint={safe_endpoint}: "
                f"{type(exc).__name__}: {safe_error}"
            ) from exc
        if not response.ok:
            summary = _redact((response.text or "")[:1000], self.config.api_key)
            safe_endpoint = _redact(endpoint, self.config.api_key)
            raise DeepSeekAPIError(
                f"DeepSeek API error endpoint={safe_endpoint} "
                f"status={response.status_code}: {summary}"
            )
        content: Any = ""
        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            patch = RepairPatch.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            safe_error = _redact(exc, self.config.api_key)
            safe_content = _redact(repr(content)[:1000], self.config.api_key)
            raise DeepSeekAPIError(
                "DeepSeek returned an invalid repair response: "
                f"{type(exc).__name__}: {safe_error}; response={safe_content}"
            ) from exc
        safe_meta = {
            "model": envelope.get("model"),
            "usage": envelope.get("usage"),
            "finish_reason": envelope.get("choices", [{}])[0].get("finish_reason"),
            "patch": patch.model_dump(mode="json"),
        }
        return patch, safe_meta


def _evidence_matches(operation: RepairOperation, source_text: str) -> None:
    if operation.evidence is None:
        return
    evidence = operation.evidence
    if evidence.end > len(source_text):
        raise PatchValidationError(f"{operation.op}: evidence end is outside source text")
    if source_text[evidence.start : evidence.end] != evidence.quote:
        raise PatchValidationError(f"{operation.op}: evidence offsets do not match quote")


def _relation_domain_range_error(
    relation_name: str,
    source_id: str,
    target_id: str,
    by_id: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> str | None:
    spec = schema["relation_types"].get(relation_name)
    if spec is None:
        return "replacement relation is not in schema"
    expected_sources = spec.get("source")
    expected_targets = spec.get("target")
    allowed_pairs = spec.get("allowed_pairs")
    if not allowed_pairs and (not expected_sources or not expected_targets):
        return None
    source = by_id.get(source_id)
    target = by_id.get(target_id)
    if source is None or target is None:
        return None
    actual = (str(source.get("label", "")), str(target.get("label", "")))
    if not _matches_direction(spec, *actual):
        return (
            f"relation domain/range mismatch actual={actual} "
            f"expected=({expected_sources},{expected_targets})"
        )
    return None


def _validation_error_counter(validation: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for error in validation.get("errors", []):
        normalized_error = {
            key: value
            for key, value in error.items()
            if key not in {"entity_index", "relation_index", "record_index"}
        }
        key = json.dumps(
            normalized_error,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def validate_patch_context(
    patch: RepairPatch,
    record: dict[str, Any],
    schema: dict[str, Any],
    *,
    minimum_confidence: float = 0.90,
) -> list[str]:
    errors: list[str] = []
    if not 0.0 <= minimum_confidence <= 1.0:
        return ["minimum_confidence must be in [0,1]"]
    expected_id = str(record.get("source_record_id") or "")
    if patch.source_record_id != expected_id:
        errors.append("source_record_id mismatch")
    if patch.schema_version != str(record.get("schema_version") or ""):
        errors.append("schema_version mismatch")
    if patch.record_sha256 != record_sha256(record):
        errors.append("record_sha256 mismatch")

    graph = graph_from_record(record)
    if graph is None:
        return errors + ["record graph missing"]
    raw_entities = graph.get("entities")
    raw_relations = graph.get("relations")
    if not isinstance(raw_entities, list):
        return errors + ["record entities is not a list"]
    if not isinstance(raw_relations, list):
        return errors + ["record relations is not a list"]
    entities = [entity for entity in raw_entities if isinstance(entity, dict)]
    if len(entities) != len(raw_entities):
        errors.append("record contains a non-object entity")
    for relation_index, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            errors.append(f"record relation[{relation_index}] is not an object")
    entity_ids = [str(entity.get("id") or "") for entity in entities]
    by_id = {
        entity_id: entity
        for entity_id, entity in zip(entity_ids, entities)
        if entity_id
    }
    if not all(entity_ids) or len(by_id) != len(entities):
        errors.append("record entity IDs are missing or duplicated")
    source_value = record.get("input")
    if not isinstance(source_value, str):
        errors.append("record input is not a string")
        source_text = ""
    else:
        source_text = source_value
    allowed_relations = schema["relation_types"]
    allowed_labels = schema["entity_types"]
    protected_main_ids: set[str] = set()
    for entity_id, entity in by_id.items():
        properties = entity.get("properties")
        if not isinstance(properties, dict):
            errors.append(f"record entity {entity_id} properties is not an object")
            properties = {}
        if entity.get("label") == "Disease" and (
            normalize_text(entity.get("name"))
            == normalize_text(record.get("source_title"))
            or properties.get("icd_uri") == expected_id
        ):
            protected_main_ids.add(entity_id)
    indexed_operations: dict[int, int] = {}

    for index, operation in enumerate(patch.operations):
        prefix = f"operation[{index}] {operation.op}"
        if operation.confidence < minimum_confidence and operation.op != "mark_manual_review":
            errors.append(f"{prefix}: confidence below {minimum_confidence}")
        try:
            _evidence_matches(operation, source_text)
        except PatchValidationError as exc:
            errors.append(str(exc))
        indexed_mutation = operation.op in {
            "replace_relation",
            "replace_relation_type",
            "swap_relation_endpoints",
            "remove_relation",
        }
        if operation.relation_index is not None and indexed_mutation:
            relation_index = operation.relation_index
            if relation_index in indexed_operations:
                errors.append(
                    f"{prefix}: relation_index also mutated by "
                    f"operation[{indexed_operations[relation_index]}]"
                )
            else:
                indexed_operations[relation_index] = index
            if relation_index >= len(raw_relations):
                errors.append(f"{prefix}: relation_index out of range")
            else:
                relation = raw_relations[relation_index]
                if not isinstance(relation, dict):
                    errors.append(f"{prefix}: indexed relation is not an object")
                elif (
                    str(relation.get("relation"))
                    != operation.expected_relation_type
                ):
                    errors.append(f"{prefix}: expected_relation_type mismatch")
        if operation.replacement_relation_type and (
            operation.replacement_relation_type not in allowed_relations
        ):
            errors.append(f"{prefix}: replacement relation is not in schema")
        for field_name, entity_id in (
            ("source_entity_id", operation.source_entity_id),
            ("target_entity_id", operation.target_entity_id),
            ("description_entity_id", operation.description_entity_id),
            ("parent_entity_id", operation.parent_entity_id),
        ):
            if entity_id and entity_id not in by_id:
                errors.append(f"{prefix}: {field_name} does not exist")
        if operation.replacement_label and operation.replacement_label not in allowed_labels:
            errors.append(f"{prefix}: replacement label is not in schema")
        if operation.op == "relabel_entity" and operation.source_entity_id:
            if operation.source_entity_id in protected_main_ids:
                errors.append(f"{prefix}: canonical main disease cannot be relabelled")
        if operation.op in {
            "replace_relation",
            "replace_relation_type",
            "swap_relation_endpoints",
        }:
            relation_index = operation.relation_index
            if relation_index is not None and relation_index < len(raw_relations):
                relation = raw_relations[relation_index]
                if isinstance(relation, dict):
                    relation_name = str(relation.get("relation", ""))
                    source_id = str(relation.get("source", ""))
                    target_id = str(relation.get("target", ""))
                    if operation.op in {"replace_relation", "replace_relation_type"}:
                        relation_name = str(operation.replacement_relation_type)
                    if operation.op == "replace_relation":
                        source_id = str(operation.source_entity_id)
                        target_id = str(operation.target_entity_id)
                    if operation.op == "swap_relation_endpoints":
                        source_id, target_id = target_id, source_id
                    domain_error = _relation_domain_range_error(
                        relation_name, source_id, target_id, by_id, schema
                    )
                    if domain_error:
                        errors.append(f"{prefix}: {domain_error}")
        if operation.op == "add_relation":
            domain_error = _relation_domain_range_error(
                str(operation.replacement_relation_type),
                str(operation.source_entity_id),
                str(operation.target_entity_id),
                by_id,
                schema,
            )
            if domain_error:
                errors.append(f"{prefix}: {domain_error}")
            duplicate = any(
                isinstance(relation, dict)
                and str(relation.get("source")) == operation.source_entity_id
                and str(relation.get("target")) == operation.target_entity_id
                and str(relation.get("relation"))
                == operation.replacement_relation_type
                for relation in raw_relations
            )
            if duplicate:
                errors.append(f"{prefix}: duplicate relation")
        if operation.op == "move_description_to_manifestations":
            child = by_id.get(str(operation.description_entity_id))
            parent = by_id.get(str(operation.parent_entity_id))
            if child and child.get("label") != "Symptom":
                errors.append(f"{prefix}: description child is not Symptom")
            if parent and parent.get("label") != "Symptom":
                errors.append(f"{prefix}: manifestation parent is not Symptom")
            if operation.description_entity_id == operation.parent_entity_id:
                errors.append(f"{prefix}: child and parent must be different")
            if child and normalize_text(child.get("name")) != normalize_text(
                operation.manifestation_text
            ):
                errors.append(f"{prefix}: manifestation text must equal child name")
            if child and child.get("properties") not in ({}, None):
                errors.append(f"{prefix}: description child has properties that would be lost")
            if parent:
                parent_properties = parent.get("properties")
                if not isinstance(parent_properties, dict):
                    errors.append(f"{prefix}: parent properties is not an object")
                elif "manifestations" in parent_properties and not isinstance(
                    parent_properties["manifestations"], list
                ):
                    errors.append(f"{prefix}: parent manifestations is not a list")
            if child and parent:
                touching = [
                    relation
                    for relation in raw_relations
                    if isinstance(relation, dict)
                    and (
                        str(relation.get("source")) == operation.description_entity_id
                        or str(relation.get("target"))
                        == operation.description_entity_id
                    )
                ]
                if len(touching) != 1:
                    errors.append(
                        f"{prefix}: description child must have exactly one relation"
                    )
                else:
                    child_relation = touching[0]
                    has_parent_equivalent = any(
                        isinstance(relation, dict)
                        and str(relation.get("source")) == operation.parent_entity_id
                        and str(relation.get("target"))
                        == str(child_relation.get("target"))
                        and str(relation.get("relation"))
                        == str(child_relation.get("relation"))
                        for relation in raw_relations
                    )
                    if (
                        str(child_relation.get("source"))
                        != operation.description_entity_id
                        or child_relation.get("relation")
                        not in {"is_core_symptom_of", "is_associated_symptom_of"}
                        or not has_parent_equivalent
                    ):
                        errors.append(
                            f"{prefix}: description child relation is not redundant"
                        )
    if any(operation.op == "mark_manual_review" for operation in patch.operations):
        if not patch.needs_human_review:
            errors.append("mark_manual_review requires needs_human_review=true")
    if patch.needs_human_review and not patch.review_reason:
        errors.append("needs_human_review requires review_reason")
    return errors


def apply_repair_patch(
    record: dict[str, Any],
    patch: RepairPatch,
    schema: dict[str, Any],
    *,
    minimum_confidence: float = 0.90,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if patch.needs_human_review:
        raise PatchValidationError("patch requires human review and cannot be applied")
    context_errors = validate_patch_context(
        patch, record, schema, minimum_confidence=minimum_confidence
    )
    if context_errors:
        raise PatchValidationError("; ".join(context_errors))

    before_validation = validate_record(record, schema)
    repaired = copy.deepcopy(record)
    graph = graph_from_record(repaired)
    if graph is None:
        raise PatchValidationError("record graph missing")
    relations = graph["relations"]

    removals: list[int] = []
    deferred_additions: list[dict[str, Any]] = []
    moves: list[RepairOperation] = []
    applied: list[dict[str, Any]] = []
    for operation in patch.operations:
        if operation.op == "mark_manual_review":
            applied.append(operation.model_dump(mode="json"))
            continue
        if operation.op == "replace_relation":
            relation = relations[operation.relation_index]  # type: ignore[index]
            relation["source"] = operation.source_entity_id
            relation["target"] = operation.target_entity_id
            relation["relation"] = operation.replacement_relation_type
        elif operation.op == "swap_relation_endpoints":
            relation = relations[operation.relation_index]  # type: ignore[index]
            relation["source"], relation["target"] = relation.get("target"), relation.get("source")
        elif operation.op == "replace_relation_type":
            relation = relations[operation.relation_index]  # type: ignore[index]
            relation["relation"] = operation.replacement_relation_type
        elif operation.op == "remove_relation":
            removals.append(operation.relation_index)  # type: ignore[arg-type]
        elif operation.op == "add_relation":
            deferred_additions.append(
                {
                    "source": operation.source_entity_id,
                    "target": operation.target_entity_id,
                    "relation": operation.replacement_relation_type,
                    "evidence": operation.evidence.quote if operation.evidence else "",
                    "evidence_span": {
                        "basis": "record.input",
                        "text": operation.evidence.quote,
                        "start": operation.evidence.start,
                        "end": operation.evidence.end,
                    }
                    if operation.evidence
                    else None,
                }
            )
        elif operation.op == "relabel_entity":
            entity = next(
                item
                for item in graph["entities"]
                if str(item.get("id")) == operation.source_entity_id
            )
            entity["label"] = operation.replacement_label
        elif operation.op == "move_description_to_manifestations":
            moves.append(operation)
        applied.append(operation.model_dump(mode="json"))

    for relation_index in sorted(set(removals), reverse=True):
        del relations[relation_index]
    relations.extend(deferred_additions)

    for operation in moves:
        child_id = str(operation.description_entity_id)
        parent_id = str(operation.parent_entity_id)
        parent = next(
            item for item in graph["entities"] if str(item.get("id")) == parent_id
        )
        properties = parent.setdefault("properties", {})
        manifestations = properties.setdefault("manifestations", [])
        if operation.manifestation_text not in manifestations:
            manifestations.append(operation.manifestation_text)
        graph["entities"] = [
            item for item in graph["entities"] if str(item.get("id")) != child_id
        ]
        graph["relations"] = [
            relation
            for relation in graph["relations"]
            if str(relation.get("source")) != child_id
            and str(relation.get("target")) != child_id
        ]

    after_validation = validate_record(repaired, schema)
    before_errors = _validation_error_counter(before_validation)
    after_errors = _validation_error_counter(after_validation)
    introduced = {
        error: count - before_errors.get(error, 0)
        for error, count in after_errors.items()
        if count > before_errors.get(error, 0)
    }
    if introduced:
        raise PatchValidationError(
            "repair patch introduced new schema errors: "
            + ", ".join(sorted(introduced))
        )
    if before_errors and sum(after_errors.values()) >= sum(before_errors.values()):
        raise PatchValidationError(
            "repair patch did not reduce the record's schema error count"
        )
    migration = repaired.get("migration")
    if isinstance(migration, dict):
        migration["errors"] = after_validation["errors"]
        migration["warnings"] = after_validation["warnings"]
        has_unverified_codes = bool(migration.get("unverified_codes"))
        migration["status"] = (
            "invalid"
            if after_validation["errors"]
            else "manual_review"
            if after_validation["warnings"] or has_unverified_codes
            else "repaired"
        )
    repaired["deepseek_repair"] = {
        "model": "deepseek-v4-flash",
        "record_sha256_before": patch.record_sha256,
        "operations": applied,
        "needs_human_review": patch.needs_human_review,
        "review_reason": patch.review_reason,
        "validation_before": before_validation,
        "validation_after": after_validation,
    }
    return repaired, repaired["deepseek_repair"]


def cached_patch_path(cache_dir: Path, task: dict[str, Any], model: str) -> Path:
    identity = json.dumps(
        {"protocol": REPAIR_PROTOCOL_VERSION, "model": model, "task": task},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return cache_dir / f"{hashlib.sha256(identity).hexdigest()}.json"


def save_safe_api_result(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)

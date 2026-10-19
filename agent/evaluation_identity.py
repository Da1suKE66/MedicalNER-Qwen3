"""Input identity for evaluation artifacts, independent of graph scoring.

The digest covers the exact UTF-8 user message, BEFORE routing, extraction,
normalization or repair. Indices refer to the caller's evaluation dataset, not
to batch offsets. Subset callers must pass the same explicit ``sample_indices``
to generation and scoring; never infer the expected order from predictions.

Identity is alignment metadata, not an authenticity signature. Legacy artifacts
remain unverified, including new predictions that depend on an unverified cache.
"""

from __future__ import annotations

import hashlib
import re


IDENTITY_KEY = "input_identity"
SCHEMA_VERSION = 1


def user_content_sha256(record):
    messages = [m for m in record.get("messages", []) if m.get("role") == "user"]
    if len(messages) != 1 or not isinstance(messages[0].get("content"), str):
        raise ValueError("Input identity requires exactly one string user message")
    return hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()


def resolve_sample_indices(records, sample_indices=None):
    indices = (
        list(range(len(records))) if sample_indices is None else list(sample_indices)
    )
    if len(indices) != len(records):
        raise ValueError("sample_indices length mismatch; alignment cannot be guessed")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("sample_indices must contain nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("sample_indices must be unique; duplicate sample identity")
    return indices


def make_input_identity(record, sample_index, *, upstream=None):
    """Bind a newly produced artifact, without upgrading unverified ancestry."""
    resolve_sample_indices([record], [sample_index])
    status = upstream["status"] if upstream is not None else "verified"
    if status not in ("verified", "unverified"):
        raise ValueError("Invalid upstream identity status")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "sample_index": sample_index,
        "user_content_sha256": user_content_sha256(record),
        "status": status,
    }
    if status == "unverified":
        identity["reason"] = "unverified_entity_cache"
    return identity


def alignment_summary(rows):
    verified = sum(row["status"] == "verified" for row in rows)
    return {
        "status": "verified" if rows and verified == len(rows) else "unverified",
        "verified_samples": verified,
        "unverified_samples": len(rows) - verified,
    }


def validate_alignment(
    records,
    predictions,
    *,
    strict_identity=False,
    sample_indices=None,
    context="predictions",
):
    """Check every declared identity before scoring; return per-sample status.

    Missing metadata is a legacy case, not evidence of alignment. Contradictory
    or malformed metadata always raises, even in compatibility mode. Explicit
    ``unverified`` provenance is sticky across cache reuse and filtering.
    """
    if len(records) != len(predictions):
        raise ValueError(
            f"Expected {len(records)} {context}, received {len(predictions)}; "
            "length mismatch; alignment cannot be guessed"
        )
    indices = resolve_sample_indices(records, sample_indices)
    checked = []
    for position, (record, raw, index) in enumerate(zip(records, predictions, indices)):
        prefix = f"{context} row {position} (sample_index={index})"
        row = {
            "sample_index": index,
            "status": "unverified",
            "reason": "missing_input_identity",
        }
        # An index alone cannot verify the source text, but a known wrong index
        # is not made safe by the absence of a digest.
        if isinstance(raw, dict):
            for key in ("sample_index", "sample_id"):
                if key in raw and (type(raw[key]) is not int or raw[key] != index):
                    raise ValueError(
                        f"{prefix}: {key} mismatch; alignment cannot be guessed"
                    )
        if isinstance(raw, dict) and IDENTITY_KEY in raw:
            identity = raw[IDENTITY_KEY]
            if not isinstance(identity, dict):
                raise ValueError(f"{prefix}: malformed input identity")
            if (
                type(identity.get("schema_version")) is not int
                or identity["schema_version"] != SCHEMA_VERSION
            ):
                raise ValueError(f"{prefix}: unsupported input identity schema_version")
            claimed_index = identity.get("sample_index")
            if type(claimed_index) is not int or claimed_index != index:
                raise ValueError(f"{prefix}: input identity sample_index mismatch")
            digest = identity.get("user_content_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{prefix}: malformed user_content_sha256")
            if digest != user_content_sha256(record):
                raise ValueError(f"{prefix}: input identity user_content_sha256 mismatch")
            status = identity.get("status")
            if status not in ("verified", "unverified"):
                raise ValueError(f"{prefix}: malformed input identity status")
            row = {"sample_index": index, "status": status}
            if status == "unverified":
                row["reason"] = identity.get("reason") or "upstream_unverified"
        checked.append(row)
    unverified = [
        row["sample_index"] for row in checked if row["status"] != "verified"
    ]
    if strict_identity and unverified:
        raise ValueError(
            f"{context}: strict identity requires verified alignment; "
            f"unverified sample_indices={unverified[:10]}"
        )
    return checked

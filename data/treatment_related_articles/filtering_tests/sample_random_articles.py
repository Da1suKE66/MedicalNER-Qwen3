#!/usr/bin/env python3
"""Reservoir-sample 50 new articles from articles.json reproducibly."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SEED = 20260728
SAMPLE_SIZE = 50

SCRIPT_DIR = Path(__file__).parent
ARTICLE_DIR = SCRIPT_DIR.parent
ARTICLES_PATH = ARTICLE_DIR / "articles.json"
PREVIOUS_TEST_PATH = (
    ARTICLE_DIR / "treatment_related_articles_screening_gpt_5_4_mini_22_v2.json"
)
OUTPUT_PATH = SCRIPT_DIR / "random_articles_50.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(path)


def abstract_hash(abstract: str) -> str:
    return hashlib.sha256(abstract.strip().encode("utf-8")).hexdigest()


def load_excluded_hashes() -> set[str]:
    payload = json.loads(PREVIOUS_TEST_PATH.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"No records list in {PREVIOUS_TEST_PATH}")
    hashes = {
        abstract_hash(record["abstract"])
        for record in records
        if isinstance(record, dict) and isinstance(record.get("abstract"), str)
    }
    if len(hashes) != 22:
        raise ValueError(f"Expected 22 unique prior abstracts, found {len(hashes)}")
    return hashes


def iter_article_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    in_records = False
    record_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not in_records:
                if stripped == '"records": [':
                    in_records = True
                continue
            if stripped in {"]", "],"}:
                return
            if not stripped:
                continue
            serialized = stripped[:-1] if stripped.endswith(",") else stripped
            try:
                record = json.loads(serialized)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid record JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {record_index} is not an object")
            yield record_index, record
            record_index += 1
    raise ValueError(f"Could not find a complete records array in {path}")


def main() -> None:
    excluded_hashes = load_excluded_hashes()
    rng = random.Random(SEED)
    reservoir: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    excluded = 0

    for record_index, record in iter_article_records(ARTICLES_PATH):
        scanned += 1
        title = record.get("title")
        abstract = record.get("abstract")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(abstract, str) or not abstract.strip():
            continue
        digest = abstract_hash(abstract)
        if digest in excluded_hashes:
            excluded += 1
            continue
        eligible += 1
        sampled = {
            "source_record_index": record_index,
            "source": str(record.get("source") or ""),
            "disease_codes": record.get("disease_codes", []),
            "doi": str(record.get("doi") or ""),
            "title": title.strip(),
            "abstract": abstract.strip(),
            "title_chars": len(title.strip()),
            "abstract_chars": len(abstract.strip()),
            "abstract_sha256": digest,
        }
        if len(reservoir) < SAMPLE_SIZE:
            reservoir.append(sampled)
            continue
        replacement = rng.randrange(eligible)
        if replacement < SAMPLE_SIZE:
            reservoir[replacement] = sampled

    if len(reservoir) != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE} samples, found {len(reservoir)}")

    reservoir.sort(key=lambda record: record["source_record_index"])
    for sample_index, record in enumerate(reservoir):
        record["sample_index"] = sample_index

    payload = {
        "artifact_schema_version": "treatment-article-random-sample-v1",
        "created_at": utc_now(),
        "sampling_method": "uniform reservoir sampling without replacement",
        "random_seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "source_file": str(ARTICLES_PATH),
        "excluded_previous_test_file": str(PREVIOUS_TEST_PATH),
        "population_records_scanned": scanned,
        "eligible_records": eligible,
        "excluded_previous_abstract_matches": excluded,
        "sample_source_counts": dict(
            sorted(Counter(record["source"] for record in reservoir).items())
        ),
        "records": reservoir,
    }
    write_json_atomic(OUTPUT_PATH, payload)
    print(f"Scanned records: {scanned}")
    print(f"Eligible records: {eligible}")
    print(f"Excluded prior matches: {excluded}")
    print(f"Sample source counts: {payload['sample_source_counts']}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

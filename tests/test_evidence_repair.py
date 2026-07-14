from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.evidence_repair import (  # noqa: E402
    locate_evidence_span,
    repair_record_evidence,
)


def test_exact_duplicate_uses_reproducible_first_occurrence() -> None:
    source = "same evidence. middle. same evidence."
    located = locate_evidence_span(source, "same evidence")
    assert located is not None
    assert located.start == 0
    assert located.text == "same evidence"
    assert located.candidate_count == 2


def test_explicit_ellipsis_restores_exact_source_slice() -> None:
    source = (
        "Some disorders, such as Autism Spectrum Disorder and Depressive Disorders, "
        "occur more commonly than in the general population."
    )
    located = locate_evidence_span(
        source,
        "Some disorders, such as... Depressive Disorders... occur more commonly than in the general population.",
    )
    assert located is not None
    assert source[located.start : located.end] == located.text
    assert "Autism Spectrum Disorder" in located.text
    assert located.method.startswith("explicit_ellipsis_anchors")


def test_whitespace_and_punctuation_variants_are_exactly_grounded() -> None:
    source = "Required features:\nPersistent symptoms, with impairment."
    located = locate_evidence_span(
        source, "Required features: Persistent symptoms, with impairment."
    )
    assert located is not None
    assert located.text == source


def _record(evidence: str) -> dict:
    return {
        "input": "Alpha evidence. Beta text.",
        "output": {
            "entities": [
                {"id": "S1", "label": "Symptom", "name": "Alpha", "properties": {}},
                {"id": "D1", "label": "Disease", "name": "Disease", "properties": {}},
            ],
            "relations": [
                {
                    "source": "S1",
                    "target": "D1",
                    "relation": "is_core_symptom_of",
                    "evidence": evidence,
                }
            ],
        },
    }


def test_repair_replaces_abbreviation_with_exact_slice_and_audits_original() -> None:
    record = _record("Alpha... evidence appears in the text.")
    record["input"] = "Alpha clinical evidence appears in the text. Beta text."
    repaired, audit = repair_record_evidence(record)
    relation = repaired["output"]["relations"][0]
    assert relation["evidence"] == "Alpha clinical evidence appears in the text."
    assert relation["evidence_original"] == "Alpha... evidence appears in the text."
    span = relation["evidence_span"]
    assert repaired["input"][span["start"] : span["end"]] == span["text"]
    assert audit["repaired"] == 1
    assert audit["unresolved"] == 0


def test_unresolved_relation_is_retained_or_explicitly_dropped() -> None:
    record = _record("not present anywhere")
    retained, retained_audit = repair_record_evidence(record)
    assert len(retained["output"]["relations"]) == 1
    assert retained_audit["unresolved"] == 1
    assert retained_audit["dropped"] == 0

    dropped, dropped_audit = repair_record_evidence(record, drop_unresolved=True)
    assert dropped["output"]["relations"] == []
    assert dropped_audit["unresolved"] == 1
    assert dropped_audit["dropped"] == 1
    assert dropped_audit["unresolved_relations"][0]["action"] == "dropped"

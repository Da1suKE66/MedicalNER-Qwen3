from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg_lora.schema_v2 import (  # noqa: E402
    _add_grounded_patient_diagnosis_relations,
    _matches_direction,
    _repair_known_relation_shapes,
    _rewrite_relation,
    load_schema,
)


def test_draft_2_declares_requested_medical_relations() -> None:
    schema = load_schema()
    assert schema["schema_version"] == "2.0.0-draft.2"
    relations = schema["relation_types"]
    assert _matches_direction(
        relations["affects_diagnosis_of"], "Patient Information", "Disease"
    )
    assert _matches_direction(
        relations["must_be_ruled_out_for"], "Disease", "Disease"
    )
    assert _matches_direction(
        relations["excludes_diagnosis_of"], "Diagnostic Criteria", "Disease"
    )
    assert _matches_direction(relations["somatic_cause_of"], "Disease", "Disease")


def test_subsumes_only_allows_same_domain_hierarchy_pairs() -> None:
    spec = load_schema()["relation_types"]["subsumes"]
    assert _matches_direction(spec, "Disease", "Disease")
    assert _matches_direction(spec, "Symptom", "Symptom")
    assert not _matches_direction(spec, "Disease", "Symptom")
    assert not _matches_direction(spec, "Symptom", "Disease")


def test_exclusion_rewrite_is_endpoint_specific() -> None:
    schema = load_schema()
    assert (
        _rewrite_relation(
            "excludes_if_present", "Diagnostic Criteria", "Disease", schema
        )
        == "excludes_diagnosis_of"
    )
    assert (
        _rewrite_relation("excludes_if_present", "Disease", "Disease", schema)
        == "must_be_ruled_out_for"
    )
    assert (
        _rewrite_relation(
            "excludes_if_present", "Diagnostic Criteria", "Symptom", schema
        )
        == "excludes_if_present"
    )


def test_known_invalid_relation_shapes_are_repaired_deterministically() -> None:
    graph = {
        "entities": [
            {"id": "D1", "label": "Disease"},
            {"id": "DC1", "label": "Diagnostic Criteria"},
            {"id": "S1", "label": "Symptom"},
            {"id": "S2", "label": "Symptom"},
        ],
        "relations": [
            {
                "source": "DC1",
                "target": "S1",
                "relation": "required_for_diagnosis_of",
            },
            {
                "source": "D1",
                "target": "S2",
                "relation": "co_occurs_with_frequency",
            },
        ],
    }
    changes = []
    _repair_known_relation_shapes(graph, "D1", changes)
    assert graph["relations"][0] == {
        "source": "DC1",
        "target": "D1",
        "relation": "required_for_diagnosis_of",
    }
    assert graph["relations"][1] == {
        "source": "S2",
        "target": "D1",
        "relation": "is_associated_symptom_of",
    }
    assert [change["op"] for change in changes] == [
        "repair_known_relation_shape",
        "repair_known_relation_shape",
    ]


def test_patient_diagnosis_relation_requires_exact_source_grounding() -> None:
    graph = {
        "entities": [
            {"id": "D1", "label": "Disease", "name": "Disorder", "properties": {}},
            {
                "id": "PI1",
                "label": "Patient Information",
                "name": "adolescents",
                "properties": {},
            },
            {
                "id": "PI2",
                "label": "Patient Information",
                "name": "model-only detail",
                "properties": {},
            },
        ],
        "relations": [],
    }
    changes = []
    source = "The presentation in adolescents may affect diagnostic assessment."
    _add_grounded_patient_diagnosis_relations(graph, source, "D1", changes)
    assert graph["relations"] == [
        {
            "source": "PI1",
            "target": "D1",
            "relation": "affects_diagnosis_of",
            "evidence": "adolescents",
            "evidence_span": {
                "basis": "record.input",
                "text": "adolescents",
                "start": source.index("adolescents"),
                "end": source.index("adolescents") + len("adolescents"),
            },
        }
    ]
    assert changes[0]["op"] == "add_grounded_patient_diagnosis_relation"

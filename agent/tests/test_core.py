from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg_agent.contracts import EntityCandidate, Entity, Graph, Relation
from kg_agent.normalize import normalize_candidates
from kg_agent.pipeline import TwoStageKGAgent
from kg_agent.prompts import build_entity_prompt, build_relation_prompt, extract_json_object
from kg_agent.source import SentenceSpanSplitter, SourceRouter, parse_icd_graph, resolve_span_ref
from kg_agent.validate import validate_graph
from build_stage_data import make_records


def test_sentence_spans_are_grounded_and_stable() -> None:
    text = "Essential Features\nReduced intelligibility of speech.\nDysarthria is a differential diagnosis."
    spans = SentenceSpanSplitter().split(text)
    assert [span.id for span in spans] == ["SENT_001", "SENT_002", "SENT_003"]
    assert text[spans[1].start : spans[1].end] == spans[1].text
    assert spans[1].section == "Essential Features"


def test_icd_parser_does_not_expand_aliases_to_nodes() -> None:
    record = {
        "URI": "urn:icd:A01",
        "code": "A01",
        "title": "Example disease",
        "synonyms": ["Example syndrome"],
        "indexTerms": ["example disorder"],
        "descendants": [{"code": "A01.1", "title": "Example subtype"}],
        "exclusions": [{"code": "B02", "title": "Excluded condition"}],
    }
    graph = parse_icd_graph(record)
    assert [entity.id for entity in graph.entities] == ["D1", "D2", "D3"]
    assert graph.entities[0].properties["Aliases"] == ["example disorder", "Example syndrome"]
    assert {(item.source, item.target, item.relation) for item in graph.relations} == {
        ("D2", "D1", "subtype_of"),
        ("D1", "D3", "rules_out"),
    }


def test_legacy_medical_wrapper_routes_icd_deterministically() -> None:
    wrapped = """Read the following medical text.\n\nMedical text:\nhttps://id.who.int/icd/release/11/2025-01/mms/605267007\n6A00\nDisorders of intellectual development\ncategory\n[{\"foundationReference\":\"https://id.who.int/icd/entity/546689346\",\"label\":\"Dementia\"}]\n[{\"deprecated\":false,\"label\":\"Intellectual disability\"}]\n"""
    document = SourceRouter().route(wrapped)
    assert document.source_type == "structured_icd"
    graph = parse_icd_graph(document)
    assert [entity.name for entity in graph.entities] == ["Disorders of intellectual development", "Dementia"]
    assert graph.relations[0].relation == "rules_out"


def test_mixed_icd_reference_list_is_not_promoted_to_relations() -> None:
    wrapped = """Medical text:
https://id.who.int/icd/release/11/2025-01/mms/605267007
6A00
Disorders of intellectual development
category
[{"foundationReference":"","label":"Unresolved child"},{"foundationReference":"https://id.who.int/icd/entity/123","label":"Possible child"}]
[{"deprecated":false,"label":"Intellectual disability"}]
"""
    document = SourceRouter().route(wrapped)
    graph = parse_icd_graph(document)
    assert [entity.id for entity in graph.entities] == ["D1"]
    assert graph.relations == []


def test_legacy_medical_wrapper_strips_instructions_for_free_text() -> None:
    document = SourceRouter().route("Instructions\nMedical text:\nReduced speech intelligibility.")
    assert document.source_type == "free_text"
    assert document.raw == "Reduced speech intelligibility."
    assert [span.text for span in document.spans] == ["Reduced speech intelligibility."]


def test_normalization_merges_duplicate_mentions_and_rejects_unknown_spans() -> None:
    document = SourceRouter().route("Speech is reduced. Speech is reduced.")
    candidates = [
        EntityCandidate("SENT_001", "Symptom", "core"),
        EntityCandidate("SENT_002", "Symptom", "associated"),
        EntityCandidate("SENT_999", "Symptom", "core"),
    ]
    entities, span_map, dropped = normalize_candidates(candidates, document)
    assert len(entities) == 1
    assert span_map["SENT_001"] == span_map["SENT_002"] == "S1"
    assert dropped == [{"span_id": "SENT_999", "reason": "span_not_found"}]


def test_sentence_local_span_keeps_phrase_name() -> None:
    document = SourceRouter().route("Reduced speech intelligibility.")
    reference = "SENT_001:8-30"
    resolved = resolve_span_ref(document, reference)
    assert resolved and resolved.text == "speech intelligibility"
    entities, mapping, _ = normalize_candidates([EntityCandidate(reference, "Symptom")], document)
    assert entities[0].name == "speech intelligibility"
    assert mapping[reference] == "S1"


def test_mention_copy_is_grounded_to_exact_source_text() -> None:
    document = SourceRouter().route("Reduced speech intelligibility.")
    candidate = EntityCandidate("SENT_001", "Symptom", "core", "speech intelligibility")
    entities, mapping, dropped = normalize_candidates([candidate], document)
    assert not dropped
    assert entities[0].name == "speech intelligibility"
    assert entities[0].span_ids == ["SENT_001:8-30"]
    assert mapping["SENT_001"] == mapping["SENT_001:8-30"] == "S1"


def test_stage_prompts_support_copy_and_explicit_none_contracts() -> None:
    spans = [{"span_id": "SENT_001", "text": "A symptom.", "section": "body"}]
    entity_prompt = build_entity_prompt(spans, mention_copy=True)
    assert '"mention"' in entity_prompt
    assert "Do not generate character offsets" in entity_prompt
    relation_prompt = build_relation_prompt(
        [{"id": "S1", "label": "Symptom", "name": "A symptom", "importance": "core"}],
        [{"pair_id": "P0001", "source": "S1", "target": "D1", "allowed_relations": ["is_core_symptom_of"]}],
        spans,
        emit_none=True,
    )
    assert "Do not omit pairs" in relation_prompt
    assert "importance=core" in relation_prompt


def test_prompt_parser_handles_wrappers() -> None:
    value = extract_json_object("<output>```json\n{\"x\": 1}\n```</output>")
    assert value == {"x": 1}


class FakeBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        self.calls.append(prompt)
        if "entity-span selector" in prompt:
            return json.dumps(
                {
                    "entity_candidates": [
                        {"span_id": "SENT_001", "label": "Symptom", "importance": "core"},
                        {"span_id": "SENT_002", "label": "Disease", "importance": "differential"},
                    ]
                }
            )
        return json.dumps(
            {
                "relations": [
                    {"pair_id": "P0002", "relation": "is_core_symptom_of", "evidence_span_id": "SENT_001"}
                ]
            }
        )


def test_two_stage_pipeline_assembles_only_valid_relations() -> None:
    backend = FakeBackend()
    result = TwoStageKGAgent(backend).run("Reduced speech intelligibility. Developmental speech disorder.")
    assert result.validation.valid
    assert result.output["entities"][0]["id"] == "D1"
    assert result.output["relations"] == [] or result.output["relations"][0]["relation"] == "is_core_symptom_of"


def test_validator_rejects_wrong_relation_signature() -> None:
    document = SourceRouter().route("A symptom.")
    graph = Graph(
        entities=[Entity("S1", "Symptom", "A symptom", span_ids=["SENT_001"]), Entity("D1", "Disease", "A disease", span_ids=["SENT_001"])],
        relations=[Relation("D1", "S1", "is_core_symptom_of", ["SENT_001"], ["A symptom."])],
    )
    report = validate_graph(graph, document)
    assert not report.valid
    assert any(issue.kind == "invalid_signature" for issue in report.issues)


def test_teacher_conversion_creates_two_stage_targets() -> None:
    teacher = {
        "entities": [
            {"id": "D1", "label": "Disease", "name": "Developmental speech disorder"},
            {"id": "S1", "label": "Symptom", "name": "Reduced intelligibility of speech"},
        ],
        "relations": [
            {
                "source": "S1",
                "target": "D1",
                "relation": "is_core_symptom_of",
                "evidence_sentence": "SENT_001",
            }
        ],
    }
    record = {
        "messages": [
            {"role": "user", "content": "Reduced intelligibility of speech. Developmental speech disorder."},
            {"role": "assistant", "content": json.dumps(teacher)},
        ]
    }
    entities, relations, report = make_records(record, 0)
    assert entities and entities["records"]
    assert relations and relations["records"]
    assert report["grounded_entities"] == 2
    assert "is_core_symptom_of" in relations["records"][0]["completion"]


def test_teacher_conversion_can_emit_copy_mentions_and_none_pairs() -> None:
    teacher = {
        "entities": [
            {"id": "D1", "label": "Disease", "name": "Developmental speech disorder"},
            {"id": "S1", "label": "Symptom", "name": "Reduced intelligibility of speech"},
        ],
        "relations": [
            {"source": "S1", "target": "D1", "relation": "is_core_symptom_of", "evidence_sentence": "SENT_001"}
        ],
    }
    record = {
        "messages": [
            {"role": "user", "content": "Reduced intelligibility of speech. Developmental speech disorder."},
            {"role": "assistant", "content": json.dumps(teacher)},
        ]
    }
    entities, relations, _ = make_records(
        record,
        0,
        mention_copy=True,
        relation_emit_none=True,
        relation_batch_size=4,
        relation_negative_floor=2,
    )
    assert entities and '"mention":"Developmental speech disorder"' in entities["records"][0]["completion"]
    assert relations and '"relation":"NONE"' in relations["records"][0]["completion"]
    relation_targets = json.loads(relations["records"][0]["completion"])["relations"]
    assert relation_targets[0]["relation"] == "NONE"
    assert relation_targets[1]["relation"] == "is_core_symptom_of"

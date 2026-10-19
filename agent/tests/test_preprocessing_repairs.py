from generative.build_data import convert
from generative.protocol import ProtocolConfig, build_batches, disease_comma_hits, dumps, supervision


def test_original_evidence_survives_sampled_rebatch_and_repair():
    text = "Fatigue accompanies Condition Alpha during pregnullcy."
    graph = {"entities": [{"id": "S", "name": "Fatigue", "label": "Symptom"}, {"id": "D", "name": "Condition Alpha", "label": "Disease"}],
             "relations": [{"source": "S", "target": "D", "relation": "is_associated_symptom_of", "evidence": text}]}
    record = {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": dumps(graph)}]}
    rows, _ = convert(record, 1, 1, ProtocolConfig(repair=True, strict_evidence=True), sampled=True)
    assert rows and "pregnancy" in rows[0]["prompt"]
    assert text in rows[0]["original_spans"].values()


def test_strict_evidence_rejects_partial_list_instead_of_teaching_incomplete_positive():
    text = "Fatigue accompanies Condition Alpha."
    nodes = [{"id": "S", "name": "Fatigue", "label": "Symptom"}, {"id": "D", "name": "Condition Alpha", "label": "Disease"}]
    config = ProtocolConfig(strict_evidence=True)
    batches, doc, _ = build_batches(text, nodes, config)
    graph = {"entities": nodes, "relations": [{"source": "S", "target": "D", "relation": "is_associated_symptom_of", "evidence": [text, "A missing fragment."]}]}
    target, stats = supervision(batches[0], graph, doc, config)
    assert target is None and stats["reason"] == "missing_evidence_fragment"


def test_comma_fallback_has_exact_source_offsets_and_does_not_change_graph_names():
    text = "Disorder moderate, without psychosis includes fatigue."
    name = "Disorder moderate without psychosis"
    nodes = [{"id": "D", "name": name, "label": "Disease"}, {"id": "S", "name": "fatigue", "label": "Symptom"}]
    before, _, audit = build_batches(text, nodes, ProtocolConfig())
    assert audit["ungrounded_ids"] == ["D"] and not before
    after, _, audit = build_batches(text, nodes, ProtocolConfig(grounding="disease_comma", context="local"))
    assert len(after) == 1 and audit["ungrounded_ids"] == []
    assert audit["grounding_aliases"][0]["id"] == "D"
    assert after[0]["entities"][0]["name"] == name
    a, b = disease_comma_hits(name, text)[0]
    assert text[a:b] == "Disorder moderate, without psychosis"


def test_grounding_does_not_merge_numbers_negation_or_other_punctuation():
    assert not disease_comma_hits("1,000 mg", "1000 mg")
    assert not disease_comma_hits("without psychosis", "with psychosis")
    assert not disease_comma_hits("Disorder Alpha", "Disorder-Alpha")
    assert not disease_comma_hits("Bipolar I", "Bipolar II")

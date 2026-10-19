"""Synthetic-only, in-memory tests; no private source text or data-file edits."""

from copy import deepcopy
import json

import pytest

from data_repair.review_policy import (
    TYPE_REVIEW_REFERENCES,
    conflicting_inputs,
    review_record,
)
from kg_agent.contracts import RELATION_SIGNATURES, allowed_relations


NORMAL = "Many typically developing children show brief feature alpha."
POSITIVE = "Condition Alpha requires persistent feature beta with impairment."


def criterion_graph(name=NORMAL, evidence=NORMAL):
    return {
        "entities": [
            {"id": "D1", "label": "Disease", "name": "Condition Alpha"},
            {"id": "C1", "label": "Diagnostic Criterion", "name": name},
        ],
        "relations": [{"source": "D1", "target": "C1",
                       "relation": "has_diagnostic_criterion", "evidence": evidence}],
    }


def typed_graph(relation="is_core_symptom_of", source_type="Symptom", target_type="Disease"):
    return {
        "entities": [
            {"id": "a", "label": source_type, "name": "Feature Alpha", "properties": {}},
            {"id": "b", "label": target_type, "name": "Condition Alpha", "properties": {"Code": "X-A"}},
        ],
        "relations": [{"source": "a", "target": "b", "relation": relation,
                       "evidence": "Feature Alpha occurs in Condition Alpha."}],
    }


def record(graph=None, *, index=None, system="Full synthetic system instructions", user="Full synthetic user input", wrapper="output"):
    graph = typed_graph() if graph is None else graph
    answer = json.dumps(graph, ensure_ascii=False)
    if wrapper == "output":
        answer = "<output>\n" + answer + "\n</output>"
    elif wrapper == "fence":
        answer = "```json\n" + answer + "\n```"
    value = {"messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user},
                           {"role": "assistant", "content": answer}]}
    if index is not None:
        value["source_index"] = index
    return value


def assert_unblocked(result):
    assert result["blocked_pairs"] == []
    assert result["record_blocked"] is False
    assert all(issue["action"] == "hint" for issue in result["issues"])


@pytest.mark.parametrize("sentence", [
    NORMAL,
    "Typically-developing children show brief feature alpha.",
    "Healthy adults occasionally experience feature alpha.",
    "Feature alpha is common among normally developing children.",
    "正常儿童有时出现短暂甲现象。",
    "正常发育的儿童表现短暂甲现象。",
])
def test_explicit_normal_population_criterion_is_pair_review(sentence):
    graph = criterion_graph(sentence, sentence)
    before = deepcopy(graph)
    result = review_record(graph, sentence, source_index=123)
    assert set(result) == {"issues", "blocked_pairs", "record_blocked"}
    assert result["blocked_pairs"] == [["D1", "C1"]]
    assert result["record_blocked"] is False
    issue = result["issues"][0]
    assert issue["code"] == "normal_population_as_diagnostic_criterion"
    assert issue["severity"] == "review"
    assert issue["action"] == "quarantine_pair"
    assert issue["source_index"] == 123 and issue["relation_index"] == 0
    assert graph == before
    # Review provides neither a replacement graph nor an empty relation label.
    assert "relations" not in result and "relation" not in issue


def test_normality_pair_does_not_block_retained_clean_positives_or_entire_batch():
    graph = criterion_graph()
    graph["entities"].append({"id": "C2", "label": "Diagnostic Criterion", "name": POSITIVE})
    positive = {"source": "D1", "target": "C2", "relation": "has_diagnostic_criterion", "evidence": POSITIVE}
    graph["relations"].insert(0, deepcopy(positive))
    graph["relations"].append(deepcopy(graph["relations"][1]))
    before = deepcopy(graph)
    result = review_record(graph, POSITIVE + " Boundary with Normality: " + NORMAL)
    assert result["blocked_pairs"] == [["D1", "C1"]]  # deduplicated, ordered IDs
    assert [i["relation_index"] for i in result["issues"]] == [1, 2]
    assert result["record_blocked"] is False
    assert graph == before and graph["relations"][0] == positive


@pytest.mark.parametrize("sentence", [
    "Disruption of normal rhythm is required for Condition Alpha.",
    "Persistent feature beta exceeds the level in healthy children.",
    "Typically developing peers show less feature beta than the diagnostic threshold.",
    "Normal children show feature alpha, but diagnosis requires persistent impairment.",
    "Healthy children show feature alpha without impairment.",
    "Healthy children do not show persistent feature beta.",
    "Healthy children don't show feature beta.",
    "Healthy children can't show feature beta in this synthetic setup.",
    "Healthy adults show no feature beta.",
    "Healthy children show feature alpha and Condition Alpha requires feature beta.",
    "Healthy children show feature alpha. Condition Alpha involves feature beta.",
    "Healthy children show feature alpha, which can signal Condition Alpha.",
    "If healthy children show feature alpha, assess feature beta.",
    "Healthy children show feature alpha for at least 4 weeks.",
    "健康儿童出现甲现象，但诊断要求持续乙现象。",
    "正常儿童没有出现甲现象。",
])
def test_benign_threshold_negative_and_mixed_scopes_are_not_blanket_blocked(sentence):
    graph = criterion_graph(sentence, sentence)
    before = deepcopy(graph)
    result = review_record(graph, "Boundary with Normality (Threshold): " + sentence)
    assert_unblocked(result)
    assert graph == before


@pytest.mark.parametrize("evidence", [[NORMAL, POSITIVE], [NORMAL, "Healthy children do not show feature beta."]])
def test_multi_fragment_positive_negative_scope_gets_only_hint(evidence):
    result = review_record(criterion_graph(evidence=evidence), " ".join(evidence))
    assert_unblocked(result)
    assert result["issues"][0]["code"] == "normality_mixed_or_threshold_scope_hint"


@pytest.mark.parametrize("name,evidence", [
    ("Brief feature alpha", NORMAL),
    ("Unrelated feature beta", NORMAL),
    (NORMAL, "Healthy adults show feature beta."),
    (NORMAL, "The source cites feature alpha."),
])
def test_population_mention_or_short_phrase_binding_is_only_hint(name, evidence):
    result = review_record(criterion_graph(name, evidence), NORMAL)
    assert_unblocked(result)
    assert result["issues"][0]["code"] == "normality_context_or_binding_hint"


def test_section_heading_alone_does_not_invalidate_legitimate_criterion():
    result = review_record(criterion_graph(POSITIVE, "Boundary with Normality (Threshold)"),
                           "Boundary with Normality (Threshold): " + POSITIVE)
    assert_unblocked(result)
    assert result["issues"][0]["code"] == "normality_section_only_hint"


@pytest.mark.parametrize("evidence", [None, "", [], [NORMAL, None], 7])
def test_missing_or_malformed_evidence_is_left_to_callers_grounding(evidence):
    assert review_record(criterion_graph(evidence=evidence), NORMAL)["issues"] == []


def test_evidence_sentence_variant_and_comparison_normalization_do_not_rewrite_graph():
    graph = criterion_graph("  " + NORMAL + "  ")
    edge = graph["relations"][0]
    edge["evidence_sentence"] = edge.pop("evidence").replace(" ", "\u3000")
    before = deepcopy(graph)
    assert review_record(graph, NORMAL)["blocked_pairs"] == [["D1", "C1"]]
    assert graph == before


def test_unit_or_evidence_grounding_mismatches_are_not_classified_as_falsehood():
    graph = criterion_graph("Measurement under 4 kg/m²", "Measurement under 4 kg/m²")
    before = deepcopy(graph)
    assert review_record(graph, "Measurement under 4 kg/m2")["issues"] == []
    assert graph == before


@pytest.mark.parametrize("relation,source_type,target_type", [
    ("is_core_symptom_of", "Symptom", "Risk"),
    ("is_associated_symptom_of", "Symptom", "Symptom"),
    ("differentiates_from", "Disease", "Symptom"),
])
def test_type_reference_quarantine_does_not_change_production_ontology(relation, source_type, target_type):
    graph = typed_graph(relation, source_type, target_type)
    before, signatures = deepcopy(graph), deepcopy(RELATION_SIGNATURES)
    assert relation in allowed_relations(source_type, target_type)
    result = review_record(graph, "Synthetic differential comparison", source_index=78)
    assert result["blocked_pairs"] == [["a", "b"]]
    assert result["record_blocked"] is False
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["code"] == "relation_type_reference_requires_review"
    assert issue["reference_authority"] == "adjudication_reference_not_production_ontology"
    assert issue["observed_types"] == [source_type, target_type]
    assert graph == before and signatures == RELATION_SIGNATURES
    issue["reference_type_pairs"][0][1] = "not a type"
    assert TYPE_REVIEW_REFERENCES[relation][0][1] == "Disease"


@pytest.mark.parametrize("relation,source_type,target_type", [
    ("is_core_symptom_of", "Symptom", "Disease"),
    ("is_associated_symptom_of", "Symptom", "Disease"),
    ("differentiates_from", "Disease", "Disease"),
    ("is_core_symptom_of", "sign", "disorder"),
])
def test_reference_conforming_positive_edges_are_untouched(relation, source_type, target_type):
    graph = typed_graph(relation, source_type, target_type)
    before = deepcopy(graph)
    assert review_record(graph, "Synthetic input")["issues"] == []
    assert graph == before


def test_unannotated_pairs_are_never_generated_as_negative():
    graph = typed_graph()
    graph["entities"].append({"id": "unused", "label": "Symptom", "name": "Feature Beta"})
    assert review_record(graph, "Synthetic input")["issues"] == []
    graph["relations"] = []
    assert review_record(graph, "Synthetic input") == {"issues": [], "blocked_pairs": [], "record_blocked": False}
    assert review_record({"entities": [], "relations": []}, "")["record_blocked"] is False


@pytest.mark.parametrize("problem", ["duplicate_id", "missing_endpoint", "missing_entities", "invalid_id", "invalid_relation"])
def test_ambiguous_structure_is_whole_record_review_not_safe_empty(problem):
    graph = typed_graph()
    if problem == "duplicate_id":
        graph["entities"].append(deepcopy(graph["entities"][0]))
    elif problem == "missing_endpoint":
        graph["relations"][0]["target"] = "not present"
    elif problem == "missing_entities":
        del graph["entities"]
    elif problem == "invalid_id":
        graph["entities"][0]["id"] = False
    else:
        graph["relations"][0]["relation"] = []
    before = deepcopy(graph)
    result = review_record(graph, "Synthetic input")
    assert result["record_blocked"] is True
    assert result["issues"][0]["action"] == "quarantine_record"
    assert graph == before


def test_existing_source_module_failure_is_review_not_relabeling():
    graph = typed_graph("is_core_symptom_of", "Disease", "Disease")
    result = review_record(graph, "Synthetic input")
    assert "existing_source_module_requires_review" in {i["code"] for i in result["issues"]}
    assert result["blocked_pairs"] == [["a", "b"]]
    assert graph["relations"][0]["relation"] == "is_core_symptom_of"


def test_integer_ids_keep_order_and_are_reported_like_existing_pipeline():
    graph = typed_graph("differentiates_from", "Disease", "Symptom")
    graph["entities"][0]["id"], graph["entities"][1]["id"] = 2, 1
    graph["relations"][0].update(source=2, target=1)
    assert review_record(graph, "Synthetic input")["blocked_pairs"] == [["2", "1"]]


def test_conflicting_inputs_blocks_all_group_members_with_direct_integration_keys():
    general, subtype = typed_graph(), typed_graph()
    subtype["entities"][1]["name"] = "Condition Alpha, subtype Beta"
    subtype["entities"][1]["properties"]["Code"] = "X-B"
    values = [record(general, index=4), record(subtype, index=19), record(general, index=31)]
    before = deepcopy(values)
    result = conflicting_inputs(values)
    assert set(result) == {4, 19, 31}
    for index, info in result.items():
        assert info["code"] == "conflicting_identical_inputs"
        assert info["related_indices"] == sorted({4, 19, 31} - {index})
        assert info["conflict_group_source_indices"] == [4, 19, 31]
        assert info["record_blocked"] is True and info["blocked_pairs"] == []
        assert info["issues"][0]["distinct_valid_canonical_graphs"] == 2
        assert info["issues"][0]["source_index"] == index
    assert result == conflicting_inputs(list(reversed(values)))
    assert values == before
    result[4]["related_indices"].clear()
    assert result[19]["conflict_group_source_indices"] == [4, 19, 31]
    assert conflicting_inputs(values)[4]["related_indices"] == [19, 31]


@pytest.mark.parametrize("change", ["subtype", "alias", "code", "type", "direction", "relation", "evidence", "graph_property"])
def test_graph_differences_remain_reviewable_not_silently_merged(change):
    graph, changed = typed_graph(), typed_graph()
    if change == "subtype":
        changed["entities"][1]["name"] += ", subtype Beta"
    elif change == "alias":
        changed["entities"][1]["name"] = "Alpha Condition"
    elif change == "code":
        changed["entities"][1]["properties"]["Code"] = "X-OTHER"
    elif change == "type":
        changed["entities"][1]["label"] = "Symptom"
    elif change == "direction":
        changed["relations"][0].update(source="b", target="a")
    elif change == "relation":
        changed["relations"][0]["relation"] = "is_associated_symptom_of"
    elif change == "evidence":
        changed["relations"][0]["evidence"] = "A different supporting statement."
    else:
        changed["scope"] = "subtype Beta only"
    assert set(conflicting_inputs([record(graph), record(changed)])) == {0, 1}


def test_canonical_graph_ignores_arbitrary_ids_order_and_superficial_unicode():
    graph = typed_graph()
    graph["entities"][0]["name"] = "Feature 2"
    graph["relations"][0]["evidence"] = ["Synthetic feature 2.", "Synthetic support."]
    graph["relations"].append(dict(graph["relations"][0], relation="supports_diagnosis_of"))
    changed = deepcopy(graph)
    changed["entities"][0].update(id="renamed_7", name=" Feature ² ", label="symptom")
    changed["entities"][1]["id"] = "renamed_3"
    for edge in changed["relations"]:
        edge.update(source="renamed_7", target="renamed_3")
        edge["evidence"] = ["  Synthetic\tsupport.  ", "Synthetic feature ²."]
    changed["entities"].reverse()
    changed["relations"].reverse()
    changed["relations"].append(deepcopy(changed["relations"][0]))
    values = [record(graph), record(changed, wrapper="fence"), record(graph, wrapper="bare")]
    before = deepcopy(values)
    assert conflicting_inputs(values) == {}
    assert values == before


@pytest.mark.parametrize("field,content", [
    ("system", "Different system instructions"),
    ("system", "Full synthetic system instructions "),
    ("user", "Full synthetic user input "),
    ("user", "Full\u3000synthetic user input"),
    ("user", "Different header: Full synthetic user input"),
])
def test_complete_system_and_user_must_be_byte_identical(field, content):
    changed = typed_graph()
    changed["entities"][1]["name"] += ", subtype Beta"
    assert conflicting_inputs([record(), record(changed, **{field: content})]) == {}


@pytest.mark.parametrize("answer", [
    "<output>{broken}</output>",
    '<output>{"entities":[],"entities":[],"relations":[]}</output>',
    '<output>{"entities":[],"relations":[],"confidence":NaN}</output>',
    '<output>{"entities":[],"relations":[],"confidence":1e999}</output>',
    '<output>{"entities":[],"relations":[]}</output> extraneous commentary',
    '<output>{"entities":[],"relations":[{"source":"missing","target":"missing","relation":"subtype_of"}]}</output>',
])
def test_invalid_graphs_cannot_hide_in_consistent_input_group(answer):
    malformed = record(index=5)
    malformed["messages"][2]["content"] = answer
    result = conflicting_inputs([record(index=1), malformed])
    assert set(result) == {1, 5}
    assert all(info["record_blocked"] for info in result.values())
    assert result[5]["code"] == "invalid_graph_in_identical_input_group"
    assert result[5]["related_indices"] == [1]
    assert result[5]["issues"][0]["this_graph_invalid"] is True
    assert result[1]["issues"][0]["this_graph_invalid"] is False
    assert conflicting_inputs([malformed])[5]["record_blocked"] is True


@pytest.mark.parametrize("problem", ["missing_system", "wrong_order", "nonstring_content", "duplicate_index", "negative_index"])
def test_invalid_input_identity_fails_explicitly_without_partial_silent_output(problem):
    values = [record(index=1), record(index=2)]
    if problem == "missing_system":
        values[0]["messages"].pop(0)
    elif problem == "wrong_order":
        values[0]["messages"].reverse()
    elif problem == "nonstring_content":
        values[0]["messages"][0]["content"] = None
    elif problem == "duplicate_index":
        values[1]["source_index"] = 1
    else:
        values[0]["source_index"] = -1
    with pytest.raises(ValueError):
        conflicting_inputs(values)


def test_empty_conflict_collection_and_nonstring_review_text():
    assert conflicting_inputs([]) == {}
    with pytest.raises(TypeError):
        review_record(typed_graph(), None)

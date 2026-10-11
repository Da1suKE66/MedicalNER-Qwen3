"""Synthetic regression cases for previously hidden generated false positives."""

import copy

import pytest

from evaluate import _entity_set, _relation_set
from evaluate_graphs import evaluate_records
from generative.evaluate import assemble_decisions
from generative.protocol import dumps


GRAPH = {
    "entities": [
        {"id": "D1", "label": "Disease", "name": "Alpha"},
        {"id": "D2", "label": "Disease", "name": "Beta"},
    ],
    "relations": [{"source": "D1", "target": "D2", "relation": "subtype_of"}],
}


def score(predicted, gold=GRAPH):
    record = {"messages": [
        {"role": "user", "content": "Alpha is a subtype of Beta."},
        {"role": "assistant", "content": dumps(gold)},
    ]}
    return evaluate_records([record], [{"output": predicted}])["overall"]


@pytest.mark.parametrize("extra_label, expected_fp", [
    ("differentiates_from", 1), ("subtype_of", 0), ("INVALID_LABEL", 1),
])
def test_duplicate_pair_cannot_hide_a_new_false_edge(extra_label, expected_fp):
    batch = {
        "id": "synthetic:0",
        "pairs": [{"pair_id": "P0001", "source": "D1", "target": "D2",
                   "allowed": ["subtype_of", "differentiates_from"]}],
        "spans": [{"span_id": "SENT_001"}],
        "original_spans": {"SENT_001": "Alpha is a subtype of Beta."},
    }
    decisions = [{"pair_id": "P0001", "relations": [
        {"relation": label, "evidence_span_ids": ["SENT_001"]},
    ]} for label in ("subtype_of", extra_label)]
    edges, errors = assemble_decisions(batch, dumps({"decisions": decisions}))
    assert "duplicate_pair" in errors
    result = score({"entities": GRAPH["entities"], "relations": edges})["relation"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, expected_fp, 0)
    assert result["f1"] == 2 / (2 + expected_fp)


@pytest.mark.parametrize("malformed", [
    None,
    {"id": "bad", "label": "Disease", "name": ""},
    {"id": "bad", "label": "Disease", "name": [], "properties": None},
    {"id": "bad", "label": [], "name": "Extra"},
    {"id": None, "label": "Disease", "name": "Extra"},
])
def test_malformed_entity_is_fp_and_gold_is_rejected(malformed):
    bad = copy.deepcopy(GRAPH)
    bad["entities"].append(malformed)
    result = score(bad)
    assert result["malformed_entity_rows"] == 1
    assert (result["entity"]["tp"], result["entity"]["fp"], result["entity"]["fn"]) == (2, 1, 0)
    assert result["entity"]["f1"] == 0.8
    with pytest.raises(ValueError, match="Invalid gold graph"):
        score(GRAPH, bad)


def test_each_bad_row_counts_and_valid_concept_duplicates_still_dedupe():
    bad = copy.deepcopy(GRAPH)
    bad["entities"].extend([None, None])
    assert score(bad)["entity"]["fp"] == 2
    duplicate = copy.deepcopy(GRAPH)
    duplicate["entities"].append({**GRAPH["entities"][0], "id": "D3"})
    assert _entity_set(duplicate) == _entity_set(GRAPH)
    assert score(duplicate)["entity"]["f1"] == 1


def test_duplicate_ids_cannot_silently_rebind_relation_endpoints():
    duplicate = copy.deepcopy(GRAPH)
    duplicate["entities"].append({"id": "D1", "label": "Disease", "name": "Other"})
    result = score(duplicate)
    assert result["malformed_entity_rows"] == 2
    assert (result["relation"]["tp"], result["relation"]["fp"], result["relation"]["fn"]) == (0, 1, 1)
    assert not (_relation_set(duplicate) & _relation_set(GRAPH))


def test_properties_name_legacy_graph_remains_accepted():
    legacy = copy.deepcopy(GRAPH)
    for entity in legacy["entities"]:
        entity["properties"] = {"Name": entity.pop("name")}
    assert score(legacy)["entity"]["f1"] == score(legacy)["relation"]["f1"] == 1


def test_gold_dangling_relation_is_not_a_matching_failure_sentinel():
    bad = copy.deepcopy(GRAPH)
    bad["relations"][0]["source"] = "missing"
    with pytest.raises(ValueError, match="Invalid gold graph"):
        score(bad, bad)

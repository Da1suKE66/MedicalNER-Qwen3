"""Near-match diagnostics must not count one prediction/gold more than once."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from evaluation_identity import make_input_identity

SPEC = importlib.util.spec_from_file_location(
    "audit_relation_near_misses", Path(__file__).parents[1] / "scripts/audit_relation_near_misses.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    graph = {"entities": [
        {"id": "S", "label": "Symptom", "name": "Feature Alpha"},
        {"id": "A", "label": "Disease", "name": "Condition Alpha"},
        {"id": "B", "label": "Disease", "name": "Condition Alpha variant"}],
        "relations": [{"source": "S", "target": target, "relation": "is_core_symptom_of"}
                      for target in ("A", "B")]}
    record = {"messages": [{"role": "user", "content": "A fictional relation audit."},
                           {"role": "assistant", "content": json.dumps(graph)}]}
    prediction = {"output": deepcopy(graph), "input_identity": make_input_identity(record, 7)}
    return record, prediction


def test_augmenting_path_recovers_maximum_instead_of_greedy():
    allowed = {("A", "1"), ("A", "2"), ("B", "1")}
    assert len(audit.maximum_matching(["A", "B"], ["1", "2"], lambda a, b: (a, b) in allowed)) == 2


@pytest.mark.parametrize("left,right", [(["A"], ["1", "2"]), (["A", "B"], ["1"])])
def test_each_side_can_only_be_used_once(left, right):
    assert len(audit.maximum_matching(left, right, lambda a, b: True)) == 1


def test_strict_match_cannot_be_reused_to_recover_similar_missing_edge():
    record, prediction = fixture()
    prediction["output"]["relations"].pop()
    before = deepcopy((record, prediction))
    report, _ = audit.audit_job([record], [7], [prediction], thresholds=(0.5,))
    assert report["strict"]["tp"] == 1
    assert report["strict"]["fn"] == 1
    assert report["lexical_diagnostics"]["0.50"]["additional_matches_not_confirmed_synonyms"] == 0
    assert (record, prediction) == before


def test_invalid_prediction_keeps_all_gold_false_negatives():
    record, prediction = fixture()
    prediction["output"] = "broken JSON"
    report, _ = audit.audit_job([record], [7], [prediction])
    assert (report["strict"]["tp"], report["strict"]["fp"], report["strict"]["fn"]) == (0, 0, 2)
    assert report["structural_diagnostics_nonexclusive"]["invalid_graphs"] == 1


def test_label_or_type_mismatch_is_not_lexical_match():
    edge = (("Symptom", "feature"), "is_core_symptom_of", ("Disease", "condition"))
    label_error = (edge[0], "is_associated_symptom_of", edge[2])
    type_error = (("Disease", "feature"), edge[1], edge[2])
    assert not audit.lexical_compatible(edge, label_error, 0.8)
    assert not audit.lexical_compatible(edge, type_error, 0.8)


def test_wrong_sample_identity_is_rejected():
    record, prediction = fixture()
    with pytest.raises(ValueError):
        audit.audit_job([record], [8], [prediction])


def test_embedded_gold_cannot_override_frozen_gold():
    record, prediction = fixture()
    prediction["gold"] = {"entities": [], "relations": []}
    with pytest.raises(ValueError, match="Embedded gold differs"):
        audit.audit_job([record], [7], [prediction])

"""Medical component denominator and immutable identities, no model/GPU."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from evaluate_choice_medical import run_medical
from generative.protocol import ProtocolConfig
from generative.rapid_screen import load_screen
from kg_agent.choice_backend import ChoiceFailure


def record():
    text = "Fatigue is a core symptom of Condition Alpha."
    graph = {"entities": [
        {"id": "S1", "name": "Fatigue", "label": "Symptom"},
        {"id": "D1", "name": "Condition Alpha", "label": "Disease"},
        {"id": "D2", "name": "Absent Disease", "label": "Disease"}],
        "relations": [
            {"source": "S1", "target": target, "relation": "is_core_symptom_of", "evidence": text}
            for target in ("D1", "D2")]}
    return {"messages": [{"role": "user", "content": text},
                          {"role": "assistant", "content": json.dumps(graph)}]}


class PerfectReachable:
    def run(self, task):
        edges = [{"pair_id": pair["pair_id"], "relation": "is_core_symptom_of",
                  "evidence_span_ids": [task.spans[0]["span_id"]]}
                 for pair in task.pairs if pair["source"] == "S1" and pair["target"] == "D1"]
        return task.encode_target(edges), {"complete": True, "eos_observed": True}


def test_unreachable_gold_is_retained_and_original_identity_is_not_renumbered():
    metrics, predictions, raw = run_medical([record()], [42], ProtocolConfig(order="source"), PerfectReachable())
    assert metrics["relation"]["tp"] == metrics["relation"]["fn"] == 1
    assert metrics["evidence"]["tp"] == metrics["evidence"]["fn"] == 1
    assert metrics["coverage"]["unreachable_gold_triples"] == 1
    assert metrics["coverage"]["candidate_covered_triples"] == 1
    assert predictions[0]["id"] == "medical:42"
    assert predictions[0]["input_identity"]["sample_index"] == 42
    assert len(predictions[0]["gold"]["relations"]) == 2
    assert all(row["sample_id"] == 42 and row["serialization_owner"] == "python_checked_actions" for row in raw)


def test_failed_actions_never_become_partial_true_positives_or_lose_gold_fn():
    class Failing:
        def run(self, task):
            raise ChoiceFailure("cap", {"complete": False, "actions": [1, 1], "budget_exhausted": True})
    metrics, predictions, raw = run_medical([record()], [42], ProtocolConfig(), Failing())
    assert metrics["relation"]["tp"] == 0 and metrics["relation"]["fn"] == 2
    assert metrics["failed_samples"] == 1 and metrics["failed_batches"] == 1
    assert predictions[0]["output"]["relations"] == []
    assert raw[0]["actions"] == [1, 1] and raw[0]["predicted_edges"] == []


def test_real_frozen_twelve_document_denominator_includes_all_forty_gold():
    directory = Path(__file__).resolve().parents[3] / "audit_work/rapid_20261012/evaluation"
    if not directory.is_dir():
        pytest.skip("Private frozen screen is not present")
    records, indices, configs, manifest = load_screen(directory)
    class Empty:
        def run(self, task):
            return [0], {"complete": True, "eos_observed": True}
    metrics, predictions, raw = run_medical(records, indices,
        replace(configs["full_source16"], pairs_per_batch=4), Empty())
    assert metrics["samples"] == len(predictions) == 12
    assert metrics["relation"]["tp"] == 0 and metrics["relation"]["fn"] == 40
    assert metrics["coverage"]["candidate_covered_triples"] == 26
    assert metrics["coverage"]["unreachable_gold_triples"] == 14
    assert metrics["coverage"]["candidate_pairs"] == 240
    assert len(raw) == 64
    assert [row["sample_index"] for row in predictions] == manifest["sample_indices"]

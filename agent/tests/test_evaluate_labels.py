"""Literal outputs remain distinct from historical integer actions and JSON."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import evaluate_choice_relations as relations
import evaluate_choice_medical as medical
import evaluate_label_relations
import evaluate_label_medical
from generative.choice_preferences import compile_preference
from generative.label_data import compile_recovery_row as compile_labels
from generative.protocol import ProtocolConfig
from generative.rapid_screen import load_screen
from generative.recovery_preferences import KINDS, make_case
from kg_agent.choice_backend import ChoiceFailure
from kg_agent.label_backend import CONTRACT
from kg_agent.relation_labels import RelationLabelTask


def source(index=0, kind="omit_second_relation", pairs=2):
    return make_case(index, kind, partition="mechanical_holdout", pairs_per_batch=pairs)


def label_trace(text, **changes):
    return {"output_contract": CONTRACT, "complete": True, "eos_observed": True,
            "budget_exhausted": False, "completion_text": text, **changes}


class Perfect:
    def __init__(self, cases):
        self.cases = cases
        self.calls = 0

    def run(self, task):
        case = self.cases[self.calls]
        self.calls += 1
        assert isinstance(task, RelationLabelTask)
        # Reverse records and evidence; no ordering condition is imposed.
        edges = deepcopy(case["gold_edges"])[::-1]
        for edge in edges:
            edge["evidence_span_ids"].reverse()
        text = task.render_target(edges)
        return edges, label_trace(text)


def test_all_semantic_cases_and_gold_are_identical_to_historical_choices():
    rows = [source(i + 16 * repeat, kind, 2 if repeat == 0 else 8)
            for repeat in range(2) for i, kind in enumerate(KINDS)]
    historical, old_selection = relations.prepare_cases(rows)
    cases, selection = relations.prepare_cases(rows, relation_format="labels")
    assert selection["samples"] == len(cases) == 24
    assert selection["excluded"] == old_selection["excluded"]
    assert selection["output_contract"] == CONTRACT
    for old, case in zip(historical, cases):
        for key in ("id", "source_id", "source_index", "input_row_sha256", "source_row_sha256", "task_input", "gold_edges"):
            assert old[key] == case[key]
        assert "gold_actions" not in case
    metrics, predictions = relations.evaluate_cases(cases, Perfect(cases), relation_format="labels")
    assert metrics["status"] == "complete" and metrics["samples"] == 24
    assert metrics["relation"]["f1"] == metrics["evidence"]["f1"] == 1
    assert metrics["output_contract"] == CONTRACT
    assert all("actions" not in row and "gold_actions" not in row for row in predictions)
    assert all(isinstance(row["completion_text"], str) for row in predictions)


@pytest.mark.parametrize("compiler", [compile_preference, compile_labels])
def test_compiled_container_fully_verified_before_conversion(compiler):
    row = compiler(source())
    cases, _ = relations.prepare_cases([row], relation_format="labels")
    assert cases[0]["id"] == row["id"]
    assert cases[0]["input_row_sha256"] == relations.digest(row)
    assert cases[0]["source_row_sha256"] == relations.digest(row["source_row"])
    row["prompt"] += " tamper"
    with pytest.raises(ValueError, match="reconstruction"):
        relations.prepare_cases([row], relation_format="labels")


@pytest.mark.parametrize("failure", ["no_eos", "cap", "invalid_line", "forged_edges", "exception", "wrong_contract"])
def test_failures_preserve_raw_literal_text_and_all_gold_fn(failure):
    cases, _ = relations.prepare_cases([source(), source(1)], relation_format="labels")
    class Failing(Perfect):
        def run(self, task):
            edges, trace = super().run(task)
            if self.calls == 1:
                if failure in ("no_eos", "cap"):
                    trace.update(eos_observed=False, budget_exhausted=failure == "cap")
                elif failure == "invalid_line":
                    trace["completion_text"] += "E1\t123\tE2\tS1\n"
                elif failure == "forged_edges":
                    edges = []
                elif failure == "wrong_contract":
                    trace["output_contract"] = "bounded-choices-v1"
                else:
                    raise ChoiceFailure("failed backend", {**trace, "complete": False})
            return edges, trace
    generator = Failing(cases)
    metrics, rows = relations.evaluate_cases(cases, generator, relation_format="labels")
    assert generator.calls == 2 and metrics["failed_samples"] == 1
    assert rows[0]["predicted_edges"] == [] and rows[0]["completion_text"]
    assert rows[0]["relations_counts"] == {"tp": 0, "fp": 0, "fn": len(cases[0]["gold_edges"])}
    assert rows[1]["status"] == "complete"
    assert metrics["relation"]["gold_count"] == sum(len(row["gold_edges"]) for row in cases)
    assert all("actions" not in row for row in rows)


def test_empty_eos_only_is_valid_and_duplicate_lines_merge_without_order_constraint():
    cases, _ = relations.prepare_cases([source(1, "all_negative_hallucination"), source(2)], relation_format="labels")
    class Duplicate(Perfect):
        def run(self, task):
            edges, trace = super().run(task)
            trace["completion_text"] *= 2
            return edges, trace
    metrics, rows = relations.evaluate_cases(cases, Duplicate(cases), relation_format="labels")
    assert rows[0]["completion_text"] == "" and rows[0]["status"] == "complete"
    assert metrics["relation"]["f1"] == metrics["evidence"]["f1"] == 1


def test_multiple_evidence_refs_and_wrong_evidence_are_scored_as_sets():
    cases, _ = relations.prepare_cases([source(3, "wrong_evidence")], relation_format="labels")
    task = cases[0]["task"]
    gold = cases[0]["gold_edges"]
    refs = [span["span_id"] for span in task.spans]
    assert len(refs) > 1
    edges = deepcopy(gold)
    edges[0]["evidence_span_ids"] = refs[::-1]
    text = task.render_target(edges)
    assert "," in text
    assert relations.checked_label_result(task, edges, label_trace(text)) == task.parse(text)
    wrong = deepcopy(gold)
    wrong[0]["evidence_span_ids"] = [next(ref for ref in refs if ref not in gold[0]["evidence_span_ids"])]
    class WrongEvidence:
        def run(self, task):
            return wrong, label_trace(task.render_target(wrong))
    metrics, _ = relations.evaluate_cases(cases, WrongEvidence(), relation_format="labels")
    assert metrics["relation"]["f1"] == 1
    assert metrics["evidence"]["tp"] == sum(len(edge["evidence_span_ids"]) for edge in gold[1:])
    assert metrics["evidence"]["fp"] == 1
    assert metrics["evidence"]["fn"] == len(gold[0]["evidence_span_ids"])


def medical_record():
    text = "Fatigue is a core symptom of Condition Alpha."
    graph = {"entities": [{"id": "S1", "name": "Fatigue", "label": "Symptom"},
        {"id": "D1", "name": "Condition Alpha", "label": "Disease"},
        {"id": "D2", "name": "Absent Disease", "label": "Disease"}],
        "relations": [{"source": "S1", "target": target, "relation": "is_core_symptom_of", "evidence": text}
                      for target in ("D1", "D2")]}
    return {"messages": [{"role": "user", "content": text},
                          {"role": "assistant", "content": json.dumps(graph)}]}


class MedicalPerfect:
    def run(self, task):
        assert isinstance(task, RelationLabelTask)
        edges = [{"pair_id": pair["pair_id"], "relation": "is_core_symptom_of",
                  "evidence_span_ids": [task.spans[0]["span_id"]]}
                 for pair in task.pairs if pair["source"] == "S1" and pair["target"] == "D1"]
        return edges, label_trace(task.render_target(edges))


@pytest.mark.parametrize("fail", [False, True])
def test_medical_failures_no_invalid_fp_unreachable_gold_kept(fail):
    class Generator(MedicalPerfect):
        def run(self, task):
            edges, trace = super().run(task)
            if fail:
                trace.update(eos_observed=False, budget_exhausted=True)
            return edges, trace
    metrics, predictions, raw = medical.run_medical([medical_record()], [42], ProtocolConfig(),
        Generator(), relation_format="labels")
    assert metrics["relation"]["tp"] == int(not fail)
    assert metrics["relation"]["fp"] == 0
    assert metrics["relation"]["fn"] == 1 + int(fail)
    assert metrics["coverage"]["unreachable_gold_triples"] == 1
    assert metrics["failed_samples"] == int(fail)
    assert predictions[0]["input_identity"]["sample_index"] == 42
    assert all(row["output_contract"] == CONTRACT for row in (metrics, *raw, *predictions))
    assert all("actions" not in row and "choice_trace" not in row for row in raw)
    assert raw[0]["completion_text"] and raw[0]["label_trace"]["completion_text"]
    assert raw[0]["serialization_owner"] == "python_checked_literal_labels"
    if fail:
        assert predictions[0]["output"]["relations"] == [] and raw[0]["predicted_edges"] == []


def test_real_frozen_medical_labels_retain_forty_gold_and_original_indices():
    directory = Path(__file__).resolve().parents[3] / "audit_work/rapid_20261012/evaluation"
    if not directory.is_dir():
        pytest.skip("Private frozen screen is not present")
    records, indices, configs, frozen = load_screen(directory)
    class Empty:
        def run(self, task):
            return [], label_trace("")
    metrics, predictions, raw = medical.run_medical(records, indices,
        replace(configs["full_source16"], pairs_per_batch=4), Empty(), relation_format="labels")
    assert metrics["samples"] == len(predictions) == 12
    assert metrics["relation"]["tp"] == metrics["relation"]["fp"] == 0
    assert metrics["relation"]["fn"] == 40
    assert metrics["coverage"]["candidate_covered_triples"] == 26
    assert metrics["coverage"]["unreachable_gold_triples"] == 14
    assert metrics["coverage"]["candidate_pairs"] == 240 and len(raw) == 64
    assert [row["sample_index"] for row in predictions] == frozen["sample_indices"]


def test_artificial_label_wrapper_writes_explicit_contract_and_input_hashes(tmp_path, monkeypatch):
    rows = [compile_preference(source())]
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases, _ = relations.prepare_cases(rows, relation_format="labels")
    receipt = {"adapter_sha256": "a", "protocol_sha256": "p", "base_input_sha256": "b"}
    monkeypatch.setattr(relations.label_backend, "check_adapter_contract", lambda *a, **k: receipt)
    def loader(args, *, relation_format):
        assert relation_format == "labels"
        return Perfect(cases), receipt
    monkeypatch.setattr(relations, "load_generator", loader)
    out = tmp_path / "output"
    argv = ["--data", str(path), "--adapter", "adapter", "--base-model", "base", "--output-dir", str(out)]
    evaluate_label_relations.main(argv)
    metrics = json.loads((out / "metrics.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    prediction = json.loads((out / "predictions.jsonl").read_text())
    assert metrics["status"] == manifest["status"] == "complete"
    assert all(row["output_contract"] == CONTRACT for row in (metrics, manifest, prediction))
    assert metrics["data_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metrics["relation"]["f1"] == 1 and "actions" not in prediction
    with pytest.raises(ValueError, match="immutable"):
        evaluate_label_relations.main(argv)


@pytest.mark.parametrize("wrapper", [evaluate_label_relations, evaluate_label_medical])
def test_thin_wrappers_forward_argv_and_labels_default(wrapper, monkeypatch):
    calls = []
    monkeypatch.setattr(wrapper, "evaluate_main", lambda argv, **kwargs: calls.append((argv, kwargs)))
    wrapper.main()
    wrapper.main(["--preflight-only"])
    assert calls == [(None, {"relation_format": "labels"}), (["--preflight-only"], {"relation_format": "labels"})]

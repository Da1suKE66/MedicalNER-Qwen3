"""Complete denominators and source identity for semantic action evaluation."""

from copy import deepcopy
import hashlib
import json

import pytest

import evaluate_choice_relations as module
from generative.choice_preferences import compile_preference
from generative.recovery_preferences import KINDS, STRUCTURAL, make_case
from kg_agent.choice_backend import ChoiceFailure


def source(index=0, kind="omit_second_relation", *, pairs=4):
    return make_case(index, kind, partition="mechanical_holdout", pairs_per_batch=pairs)


class Generator:
    def __init__(self, cases, *, transform=None):
        self.cases = cases
        self.calls = 0
        self.transform = transform
    def run(self, task):
        case = self.cases[self.calls]
        self.calls += 1
        assert task is case["task"]
        actions = deepcopy(case["gold_actions"])
        if self.transform:
            actions = self.transform(task, actions, self.calls)
        return actions, {"complete": True, "eos_observed": True,
                         "generated_token_ids": [1, 2], "budget_exhausted": False}


def test_all_semantic_cases_kept_without_length_or_label_selection():
    sources = [source(i, kind, pairs=2 if repeat == 0 else 8)
               for repeat in range(2) for i, kind in enumerate(KINDS, repeat * 16)]
    cases, manifest = module.prepare_cases(sources)
    assert len(cases) == manifest["samples"] == 24
    assert len(manifest["excluded"]) == 8
    assert {row["kind"] for row in manifest["excluded"]} == set(STRUCTURAL)
    assert {len(case["task"].pairs) for case in cases} == {2, 8}
    assert [case["source_index"] for case in cases] == list(range(12)) + list(range(16, 28))
    generator = Generator(cases)
    metrics, predictions = module.evaluate_cases(cases, generator)
    assert metrics["status"] == "complete" and metrics["samples"] == generator.calls == 24
    assert metrics["relation"]["f1"] == metrics["evidence"]["f1"] == 1
    assert metrics["relation"]["tp"] == sum(len(case["gold_edges"]) for case in cases)
    assert metrics["failed_samples"] == 0
    assert len(metrics["strata"]["kind"]) == 12
    assert sum(item["samples"] for item in metrics["strata"]["prompt_chars"].values()) == 24
    assert metrics["exact_relation_sets"] == 24  # includes fully negative cases
    assert {row["id"] for row in predictions} == {case["id"] for case in cases}


def test_compiled_holdout_preserves_input_ids_and_matches_raw_gold_exactly():
    raw = [source(0), source(1, "wrong_evidence")]
    compiled = [compile_preference(row) for row in raw]
    cases, _ = module.prepare_cases(compiled)
    originals, _ = module.prepare_cases(raw)
    assert [case["id"] for case in cases] == [row["id"] for row in compiled]
    for case, original, input_row in zip(cases, originals, compiled):
        assert case["gold_actions"] == original["gold_actions"]
        assert case["task_input"] == original["task_input"]
        assert case["source_row_sha256"] == original["source_row_sha256"]
        assert case["input_row_sha256"] == module.digest(input_row)
    compiled[0]["chosen_actions"] = [0]
    with pytest.raises(ValueError, match="reconstruction"):
        module.prepare_cases(compiled)


def test_wrong_evidence_does_not_corrupt_relation_score():
    cases, _ = module.prepare_cases([source(1, "wrong_evidence")])
    def wrong(task, actions, count):
        edges = task.edges(actions)
        refs = edges[0]["evidence_span_ids"]
        edges[0]["evidence_span_ids"] = [next(span["span_id"] for span in task.spans if span["span_id"] not in refs)]
        return task.encode_target(edges)
    metrics, _ = module.evaluate_cases(cases, Generator(cases, transform=wrong))
    assert metrics["relation"]["tp"] == 1 and metrics["relation"]["f1"] == 1
    assert metrics["evidence"]["tp"] == 0
    assert metrics["evidence"]["fp"] == metrics["evidence"]["fn"] == 1


def test_missing_second_relation_is_fn_and_unsupported_negative_is_fp():
    cases, _ = module.prepare_cases([source(1), source(2, "all_negative_hallucination")])
    def corrupt(task, actions, count):
        if count == 1:
            return task.encode_target(task.edges(actions)[:1])
        pair = task.pairs[0]
        return task.encode_target([{"pair_id": pair["pair_id"], "relation": pair["allowed"][0],
                                    "evidence_span_ids": [task.spans[0]["span_id"]]}])
    metrics, records = module.evaluate_cases(cases, Generator(cases, transform=corrupt))
    assert metrics["relation"]["tp"] == metrics["relation"]["fp"] == metrics["relation"]["fn"] == 1
    assert metrics["relation"]["precision"] == metrics["relation"]["recall"] == metrics["relation"]["f1"] == .5
    assert records[1]["relations_counts"] == {"tp": 0, "fp": 1, "fn": 0}


@pytest.mark.parametrize("error", ["budget", "illegal_actions", "runtime"])
def test_failed_rows_keep_full_gold_and_later_rows_are_attempted(error):
    cases, _ = module.prepare_cases([source(0), source(1)])
    def fail_first(task, actions, count):
        if count == 1:
            if error == "budget":
                raise ChoiceFailure("cap", {"actions": [1], "complete": False, "budget_exhausted": True})
            if error == "runtime":
                raise RuntimeError("device error")
            return [1, 0]
        return actions
    generator = Generator(cases, transform=fail_first)
    metrics, records = module.evaluate_cases(cases, generator)
    assert generator.calls == metrics["samples"] == 2
    assert metrics["status"] == "complete" and not metrics["generation_complete"]
    assert metrics["failed_samples"] == 1
    assert metrics["relation"]["gold_count"] == sum(len(case["gold_edges"]) for case in cases)
    assert metrics["relation"]["fn"] == len(cases[0]["gold_edges"])
    assert records[0]["predicted_edges"] == [] and records[0]["status"] == "failed"
    assert records[1]["status"] == "complete"
    assert metrics["budget_exhaustions"] == int(error == "budget")


def test_deadline_accounts_for_every_row_without_claiming_complete_run():
    cases, _ = module.prepare_cases([source(0), source(1), source(2)])
    now = [0.]
    def slow(task, actions, count):
        now[0] = 6.
        return actions
    generator = Generator(cases, transform=slow)
    metrics, records = module.evaluate_cases(cases, generator, max_seconds=5, clock=lambda: now[0])
    assert metrics["status"] == "incomplete" and metrics["complete"] is False
    assert metrics["samples"] == metrics["failed_samples"] == 3
    assert generator.calls == metrics["generation_attempted_samples"] == 1
    assert metrics["relation"]["fn"] == sum(len(case["gold_edges"]) for case in cases)
    assert len(records) == 3 and all(not row["predicted_edges"] for row in records)


@pytest.mark.parametrize("change", ["train", "duplicate", "unknown", "tampered"])
def test_bad_holdout_admission_fails_before_generating(change):
    row = source()
    if change == "train":
        row = make_case(0, "omit_second_relation")
    elif change == "unknown":
        row["pair_supervision"] = {"P0001": "UNKNOWN_UNANNOTATED"}
    elif change == "tampered":
        row["oracle_source_text"] += " mutation"
    with pytest.raises(ValueError):
        module.prepare_cases([row, row] if change == "duplicate" else [row])


def test_summary_independently_recounts_and_rejects_duplicate_ids_or_wrong_counts():
    cases, _ = module.prepare_cases([source(0)])
    _, records = module.evaluate_cases(cases, Generator(cases))
    with pytest.raises(ValueError, match="uniquely"):
        module.summarize(records + records)
    records[0]["relations_counts"]["fn"] += 1
    with pytest.raises(ValueError, match="disagree"):
        module.summarize(records)


def test_cli_artifacts_bind_input_and_all_prediction_rows(tmp_path, monkeypatch):
    rows = [compile_preference(source(0)), compile_preference(source(1))]
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {"adapter_sha256": "adapter", "protocol_sha256": "protocol", "base_input_sha256": "base"}
    monkeypatch.setattr(module, "check_adapter_contract", lambda *a, **k: receipt)
    def loader(args):
        class CorrectGenerator:
            def run(self, task):
                row = next(row for row in rows if row["prompt"] == task.prompt)
                return row["chosen_actions"], {"complete": True, "eos_observed": True}
        return CorrectGenerator(), receipt
    monkeypatch.setattr(module, "load_generator", loader)
    out = tmp_path / "result"
    args = ["--data", str(path), "--adapter", "adapter", "--base-model", "base", "--output-dir", str(out)]
    module.main(args)
    metrics = json.loads((out / "metrics.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert metrics["status"] == manifest["status"] == "complete"
    assert metrics["samples"] == len(predictions) == 2
    assert metrics["relation"]["f1"] == metrics["evidence"]["f1"] == 1
    assert metrics["data_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metrics["predictions_sha256"] == hashlib.sha256((out / "predictions.jsonl").read_bytes()).hexdigest()
    assert {row["id"] for row in predictions} == {row["id"] for row in rows}
    assert all(row["gold_actions"] and row["gold_edges"] and row["task_input"] for row in predictions)
    with pytest.raises(ValueError, match="immutable"):
        module.main(args)

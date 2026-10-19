"""Offline probe scoring, failure retention and single-model loader wiring."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
import probe_choices as probe
from generative.choice_data import compile_recovery_row, validate_row
from generative.recovery_preferences import make_case
from kg_agent.choice_backend import ChoiceFailure
from kg_agent.entity_choices import EntityChoiceTask


def sources():
    return [make_case(index, kind, partition="mechanical_holdout")
            for index, kind in enumerate(("omit_second_relation", "omit_one_edge"))]


class OracleGenerator:
    """Only exact authored component answers; unmapped pipeline tasks choose 0."""
    def __init__(self, rows, fail_calls=()):
        self.calls = []
        self.fail_calls = set(fail_calls)
        self.targets = {}
        for row in rows:
            for stage in ("entities", "relations"):
                compiled = compile_recovery_row(row, stage)
                self.targets[compiled["prompt"]] = compiled["actions"]

    def run(self, task):
        index = len(self.calls)
        self.calls.append(task)
        if index in self.fail_calls:
            raise ChoiceFailure("test generation budget ended", {
                "complete": False, "budget_exhausted": True, "actions": [1],
                "generated_token_ids": [49], "partial_action": True})
        actions = list(self.targets.get(task.prompt, [0]))
        assert task.choices(tuple(actions)) is None
        return actions, {"complete": True, "eos_observed": True, "error": None,
                         "budget_exhausted": False, "test_fake_generator": True}


def write_source(tmp_path, rows):
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_perfect_components_are_micro_scored_and_pipeline_is_separate():
    rows = sources()
    generator = OracleGenerator(rows)
    report, predictions = probe.run_probe(rows, generator)
    assert report["status"] == "complete" and report["scope"] == probe.SCOPE
    assert report["requested_component_attempts"] == report["completed_component_attempts"] == 4
    assert len(predictions) == 5 and predictions[-1]["stage"] == "pipeline_smoke"
    assert all(report["components"][stage]["f1"] == 1 for stage in ("entities", "relations", "relation_evidence"))
    assert report["probe_assertions"]["engineering_pass"]
    assert report["probe_assertions"]["grammar_failures"] == 0
    assert report["pipeline_smoke"]["engineering_pass"]
    assert report["components"]["relations"]["tp"] == sum(p["gold_count"] for p in predictions if p["stage"] == "relations")
    assert predictions[-1]["not_in_component_metrics"]
    # The raw-source pipeline really regenerates input-derived candidates; it
    # does not reuse the four authored candidate pairs of each component task.
    pipeline_tasks = generator.calls[4:]
    assert isinstance(pipeline_tasks[0], EntityChoiceTask)
    pair_count = sum(len(task.pairs) for task in pipeline_tasks if not isinstance(task, EntityChoiceTask))
    assert pair_count > len(compile_recovery_row(rows[0], "relations")["task_input"]["batch"]["pairs"])


def test_failed_rows_keep_full_gold_fn_and_other_requested_rows_still_run():
    rows = sources()
    report, predictions = probe.run_probe(rows, OracleGenerator(rows, fail_calls=(0, 1)))
    for stage in ("entities", "relations"):
        stage_predictions = [p for p in predictions if p["stage"] == stage]
        failed, successful = stage_predictions
        metric = report["components"][stage]
        assert failed["status"] == "failed" and successful["status"] == "complete"
        assert failed["counts"] == {"tp": 0, "fp": 0, "fn": failed["gold_count"]}
        assert metric["fn"] == failed["gold_count"] and metric["tp"] == successful["gold_count"]
        assert metric["recall"] == successful["gold_count"] / (successful["gold_count"] + failed["gold_count"])
        assert metric["failed_samples"] == 1
        assert failed["actions"] == [1] and failed["trace"]["generated_token_ids"] == [49]
    assert report["status"] == "complete" and report["probe_assertions"]["all_requested_attempted"]
    assert report["probe_assertions"]["grammar_failures"] == 2
    assert report["probe_assertions"]["component_budget_exhaustions"] == 2
    assert not report["probe_assertions"]["engineering_pass"]


def test_every_failed_component_still_has_a_metric_denominator():
    rows = sources()
    report, predictions = probe.run_probe(rows, OracleGenerator(rows, fail_calls=range(4)))
    for stage, metric in report["components"].items():
        assert metric["tp"] == metric["fp"] == metric["f1"] == 0
        assert metric["gold_count"] == metric["fn"] > 0
        assert metric["samples"] == metric["failed_samples"] == 2
    assert len(predictions) == 5 and report["status"] == "complete"


def test_empty_sets_have_explicit_zero_denominators():
    assert probe._metrics({}) == {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0,
        "recall": 0.0, "f1": 0.0, "gold_count": 0, "predicted_count": 0}
    assert probe._counts({("p", "r")}, {("p", "r"), ("p", "wrong")}) == {"tp": 1, "fp": 1, "fn": 0}
    assert probe._metrics({"tp": 2, "fp": 1, "fn": 3})["f1"] == 4 / 8


def test_illegal_returned_actions_are_a_failure_and_preserve_generation_trace():
    rows = sources()
    generator = OracleGenerator(rows)
    original = generator.run
    first = True
    def run(task):
        nonlocal first
        if first:
            first = False
            generator.calls.append(task)
            return [999999], {"complete": True, "generated_token_ids": [12, 13]}
        return original(task)
    generator.run = run
    report, predictions = probe.run_probe(rows, generator)
    assert predictions[0]["status"] == "failed"
    assert predictions[0]["trace"]["generated_token_ids"] == [12, 13]
    assert predictions[0]["counts"]["fn"] == predictions[0]["gold_count"]
    assert report["probe_assertions"]["grammar_failures"] == 1
    assert not report["probe_assertions"]["engineering_pass"]


def test_pipeline_failure_is_not_merged_into_perfect_component_f1():
    rows = sources()
    report, predictions = probe.run_probe(rows, OracleGenerator(rows, fail_calls=(4,)))
    assert report["status"] == "complete"
    assert report["pipeline_smoke"]["status"] == "failed"
    assert report["components"]["entities"]["f1"] == report["components"]["relations"]["f1"] == 1
    assert report["probe_assertions"]["grammar_failures"] == 1
    assert not report["probe_assertions"]["engineering_pass"]
    assert predictions[-1]["generation_calls"][0]["status"] == "failed"


def test_pipeline_model_call_budget_fails_without_slicing_candidate_tables():
    source = sources()[0]
    task = validate_row(compile_recovery_row(source, "relations"), require_train=False)
    delegate = Mock()
    delegate.run.return_value = ([0], {"complete": True})
    bounded = probe._PipelineBudget(delegate)
    for _ in range(probe.MAX_PIPELINE_CALLS):
        bounded.run(task)
    with pytest.raises(ChoiceFailure, match="candidates were not truncated") as exc:
        bounded.run(task)
    assert exc.value.trace["call_budget_exhausted"]
    assert delegate.run.call_count == probe.MAX_PIPELINE_CALLS


def test_load_checks_all_input_rows_even_outside_requested_prefix(tmp_path):
    rows = sources()
    assert len(probe.load_sources(write_source(tmp_path, rows), 1)) == 1
    with pytest.raises(ValueError, match="requested limit"):
        probe.load_sources(write_source(tmp_path, rows), 3)
    with pytest.raises(ValueError, match="mechanical_holdout"):
        probe.load_sources(write_source(tmp_path, [rows[0], make_case(0, "omit_one_edge")]), 1)
    with pytest.raises(ValueError, match="Duplicate"):
        probe.load_sources(write_source(tmp_path, [rows[0], rows[0]]), 1)
    bad = deepcopy(rows[1]); bad["unknown_reasons"] = {"P0001": "unannotated"}
    with pytest.raises(ValueError):
        probe.load_sources(write_source(tmp_path, [rows[0], bad]), 1)


def test_callable_probe_rejects_nonheldout_before_generation():
    generator = Mock()
    with pytest.raises(ValueError, match="mechanical_holdout"):
        probe.run_probe([make_case(0, "omit_one_edge")], generator)
    generator.run.assert_not_called()
    row = sources()[0]
    with pytest.raises(ValueError, match="Duplicate"):
        probe.run_probe([row, row], generator)
    generator.run.assert_not_called()


def test_loader_checks_both_contracts_and_loads_one_base_and_one_adapter(monkeypatch):
    check = Mock(return_value={"output_contract": "bounded-choices-v1"})
    monkeypatch.setattr(probe, "check_adapter_contract", check)
    model, base = Mock(), Mock()
    model.to.return_value = model
    model.eval.return_value = model
    base_loader = Mock(return_value=base)
    adapter_loader = Mock(return_value=model)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16", cuda=SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: True)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=base_loader),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=adapter_loader)))
    args = SimpleNamespace(base_model="base", adapter="adapter", max_input_tokens=6144, max_new_tokens=2048)
    generator, receipts = probe.load_generator(args)
    assert [call.args for call in check.call_args_list] == [("adapter", "entities"), ("adapter", "relations")]
    assert all(call.kwargs == {"base_model": "base"} for call in check.call_args_list)
    base_loader.assert_called_once()
    adapter_loader.assert_called_once_with(base, "adapter", adapter_name="default", is_trainable=False, local_files_only=True)
    assert generator.backend.model is model and len(receipts) == 2


def test_cli_persists_all_predictions_and_complete_does_not_mean_quality(tmp_path, monkeypatch):
    rows = sources()
    path = write_source(tmp_path, rows)
    out = tmp_path / "probe"
    monkeypatch.setattr(probe, "load_generator", lambda args: (OracleGenerator(rows, fail_calls=(0,)), {}))
    argv = ["--data", str(path), "--base-model", "base", "--adapter", "adapter", "--output-dir", str(out)]
    args = probe.parse_args(argv)
    assert args.limit == 2 and args.max_new_tokens == 2048
    probe.main(argv)
    report = json.loads((out / "metrics.json").read_text())
    predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert report["status"] == "complete" and not report["probe_assertions"]["engineering_pass"]
    assert len(predictions) == 5 and predictions[0]["status"] == "failed"
    before = (out / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        probe.main(argv)
    assert (out / "metrics.json").read_bytes() == before


def test_configuration_failure_never_reports_all_attempts_complete(tmp_path, monkeypatch):
    path = write_source(tmp_path, sources())
    out = tmp_path / "probe"
    monkeypatch.setattr(probe, "load_generator", Mock(side_effect=ValueError("wrong adapter contract")))
    with pytest.raises(ValueError):
        probe.main(["--data", str(path), "--base-model", "base", "--adapter", "adapter", "--output-dir", str(out)])
    report = json.loads((out / "metrics.json").read_text())
    assert report["status"] == "failed" and not report["probe_assertions"]["engineering_pass"]

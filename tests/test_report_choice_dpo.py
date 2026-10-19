from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from evaluate_choice_relations import prepare_cases, evaluate_cases
from generative.choice_preferences import compile_preference
from generative.recovery_preferences import make_case
from kg_agent.choice_backend import ChoiceFailure

SPEC = importlib.util.spec_from_file_location("report_choice_dpo", Path(__file__).parents[1] / "scripts/report_choice_dpo.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


@pytest.fixture
def release(tmp_path):
    raw = [make_case(1, kind, partition="mechanical_holdout", pairs_per_batch=4)
           for kind in ("wrong_target", "all_negative_hallucination", "wrong_evidence", "omit_one_edge")]
    compiled = [compile_preference(row) for row in raw]
    cases, _ = prepare_cases(compiled)
    data = tmp_path / "held.jsonl"
    data.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in compiled))
    lookup = {row["prompt"]: (index, row) for index, row in enumerate(compiled)}
    for arm in reporter.ARMS:
        class Generator:
            def run(self, task):
                index, row = lookup[task.prompt]
                if arm == "parent" and index == 0:
                    raise ChoiceFailure("incomplete", {"complete": False, "actions": [1], "budget_exhausted": True})
                actions = row["rejected_actions"] if arm == "dpo" else row["chosen_actions"]
                return actions, {"complete": True}
        metrics, records = evaluate_cases(cases, Generator())
        directory = tmp_path / (arm + "_eval")
        directory.mkdir()
        (directory / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        (directory / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path, data


def test_independent_recount_includes_failures_and_agrees_with_both_metrics(release):
    root, data = release
    result = reporter.build_report(root, data)
    runs = result["aggregate"]["runs"]
    assert runs["parent"]["failed_samples"] == 1
    assert runs["parent"]["relation"]["fn"] > 0
    assert runs["sft"]["relation"]["f1"] == 1.0
    assert runs["sft"]["evidence"]["f1"] == 1.0
    assert len({run["relation"]["gold_count"] for run in runs.values()}) == 1
    aggregate_text = json.dumps(result["aggregate"])
    assert "Feature Cedar" not in aggregate_text and "oracle_source_text" not in aggregate_text
    assert result["private_error_examples"]["sft"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_id", "wrong_gold", "wrong_actions", "wrong_input", "wrong_counts"])
def test_prediction_tampering_is_rejected_before_reporting(release, mode):
    root, data = release
    path = root / "sft_eval" / "predictions.jsonl"
    rows = reporter.read_rows(path)
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows[0] = deepcopy(rows[1])
    elif mode == "wrong_id":
        rows[0]["id"] = "unrelated"
    elif mode == "wrong_gold":
        rows[0]["gold_edges"] = []
    elif mode == "wrong_actions":
        rows[0]["actions"] = [0]
    elif mode == "wrong_input":
        rows[0]["task_input_sha256"] = "different"
    elif mode == "wrong_counts":
        rows[0]["relations_counts"]["fn"] = 99
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        reporter.build_report(root, data)


def test_plausible_declared_f1_is_not_trusted(release):
    root, data = release
    path = root / "dpo_eval" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["relation"]["f1"] = 0.999
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="declared metrics mismatch"):
        reporter.build_report(root, data)


def test_failed_partial_actions_cannot_materialize_prediction(release):
    root, data = release
    path = root / "parent_eval" / "predictions.jsonl"
    rows = reporter.read_rows(path)
    rows[0]["predicted_edges"] = rows[0]["gold_edges"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="failed row materialized"):
        reporter.build_report(root, data)


def test_cli_writes_separate_private_and_aggregate_reports(release, capsys):
    root, data = release
    prefix = root / "report"
    reporter.main(["--root", str(root), "--data", str(data), "--output-prefix", str(prefix)])
    assert Path(str(prefix) + ".json").exists()
    public = json.loads(Path(str(prefix) + ".aggregate.json").read_text())
    assert "private_error_examples" not in public
    assert public["training"]["status"] == "not_provided"
    assert json.loads(capsys.readouterr().out)["samples"] == 4
    with pytest.raises(ValueError, match="overwrite"):
        reporter.main(["--root", str(root), "--data", str(data), "--output-prefix", str(prefix)])


@pytest.mark.parametrize("changed", [None, "row_schedule_sha256", "paired_chosen_encoding_sha256", "reference_adapter_sha256_after"])
def test_training_receipts_require_same_exposure_parent_and_frozen_reference(tmp_path, changed):
    common = {"status": "training_complete_evaluation_pending", "parent_initialization_verified": True,
        "saved_reload_exact": True, "source_rows": 128, "optimizer_updates": 32,
        "observed_exposure": {"rows": 128}, "reference_unchanged": True,
        "reference_adapter_sha256_after": "parent-parameters", "policy_adapter_sha256_before": "parent-parameters",
        "expected_parent_parameter_sha256": "parent-parameters"}
    for key in ("parent_adapter_sha256", "parent_manifest_sha256", "data_sha256", "row_schedule_sha256", "paired_chosen_encoding_sha256"):
        common[key] = key + "-fixed"
    a, b = tmp_path / "sft.json", tmp_path / "dpo.json"
    a.write_text(json.dumps(common))
    if changed:
        common[changed] = "different"
    b.write_text(json.dumps(common))
    if changed:
        with pytest.raises(ValueError):
            reporter.check_training(a, b)
    else:
        assert reporter.check_training(a, b)["same_parent_and_exposure_verified"]

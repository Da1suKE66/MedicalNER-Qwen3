"""Medical reports must derive their counts from aligned predictions/actions."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path

import pytest

from evaluate_choice_medical import run_medical, SCOPE
from evaluation_identity import make_input_identity
from generative.protocol import ProtocolConfig
from generative.rapid_screen import prepare_screen, load_screen
from kg_agent.choice_backend import ChoiceFailure

SPEC = importlib.util.spec_from_file_location("report_choice_medical", Path(__file__).parents[1] / "scripts/report_choice_medical.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def record(i):
    text = f"Feature {i} is a core symptom of Condition Alpha {i}."
    graph = {"entities": [{"id": "S", "label": "Symptom", "name": f"Feature {i}"},
                          {"id": "D", "label": "Disease", "name": f"Condition Alpha {i}"}],
             "relations": [{"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": text}]}
    return {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": json.dumps(graph)}]}


def save_job(release, *, fail=False):
    class Generator:
        def run(self, task):
            if fail:
                raise ChoiceFailure("cap", {"actions": [1], "complete": False, "budget_exhausted": True})
            edges = [{"pair_id": pair["pair_id"], "relation": "is_core_symptom_of",
                      "evidence_span_ids": [task.spans[0]["span_id"]]}
                     for pair in task.pairs if pair["source"] == "S"]
            return task.encode_target(edges), {"complete": True, "eos_observed": True}
    r = release
    metrics, predictions, raw = run_medical(r["records"], r["indices"], r["config"], Generator())
    job = r["root"] / "parent_eval"
    job.mkdir(parents=True, exist_ok=True)
    lines(job / "predictions.jsonl", predictions)
    lines(job / "generations.jsonl", raw)
    metrics.update(screen_manifest_sha256=reporter.sha(r["screen"] / "manifest.json"), adapter_sha256="adapter",
                   predictions_sha256=reporter.sha(job / "predictions.jsonl"), generations_sha256=reporter.sha(job / "generations.jsonl"))
    write(job / "metrics.json", metrics)
    frozen = r["frozen"]
    write(job / "run_manifest.json", {"scope": SCOPE, "status": "complete", "complete": True,
        "samples": len(r["records"]), "sample_indices": r["indices"], "config": asdict(r["config"]),
        "screen_manifest_sha256": metrics["screen_manifest_sha256"], "records_sha256": frozen["records_sha256"],
        "sample_indices_sha256": frozen["sample_indices_sha256"], "adapter_sha256": "adapter",
        "metrics_sha256": reporter.sha(job / "metrics.json"), "base_input_sha256": "base", "protocol_sha256": "protocol",
        "args": {"max_input_tokens": 6144, "max_new_tokens": 512, "max_seconds": 600}})
    return job


@pytest.fixture
def release(tmp_path):
    gold, matrix, screen = tmp_path / "gold.json", tmp_path / "matrix.json", tmp_path / "screen"
    write(gold, [record(i) for i in range(3)])
    write(matrix, {"defaults": asdict(ProtocolConfig(matching="nfkc_v2")), "variants": {"V4": {}}})
    prepare_screen(gold, matrix, "V4", screen, count=2)
    records, indices, configs, frozen = load_screen(screen)
    result = {"root": tmp_path / "runs", "screen": screen, "records": records,
              "indices": indices, "config": replace(configs["full_source16"], pairs_per_batch=4), "frozen": frozen}
    result["job"] = save_job(result)
    return result


def refresh(release):
    job = release["job"]
    metrics = reporter.read(job / "metrics.json")
    metrics.update(predictions_sha256=reporter.sha(job / "predictions.jsonl"),
                   generations_sha256=reporter.sha(job / "generations.jsonl"))
    write(job / "metrics.json", metrics)
    run = reporter.read(job / "run_manifest.json")
    run["metrics_sha256"] = reporter.sha(job / "metrics.json")
    write(job / "run_manifest.json", run)


def test_complete_independent_recount_missing_arms_and_no_public_text(release):
    report, private = reporter.build_report(release["root"], release["screen"])
    row = report["jobs"][0]
    assert row["status"] == "verified" and row["relation"]["tp"] == 2
    assert row["relation"]["fp"] == row["relation"]["fn"] == 0
    assert [row["status"] for row in report["jobs"]] == ["verified", "missing", "missing"]
    assert report["gold_relation_triples"] == 2 and not report["all_three_verified"]
    assert "Feature" not in json.dumps(report) and "Condition Alpha" not in reporter.markdown(report)
    assert private == {"parent_eval": []}


def test_forged_plausible_aggregate_is_rejected_even_with_rehashed_artifact(release):
    path = release["job"] / "metrics.json"
    metrics = reporter.read(path)
    metrics["relation"] = reporter.prf(2, 0, 1)
    write(path, metrics)
    refresh(release)
    row = reporter.build_report(release["root"], release["screen"])[0]["jobs"][0]
    assert row["status"] == "invalid" and row["validation_error"] == "reported_relation_mismatch"
    assert "relation" not in row


@pytest.mark.parametrize("change", ["missing", "identity", "actions"])
def test_prediction_or_action_tampering_cannot_pass_by_fixing_file_hashes(release, change):
    if change == "actions":
        path = release["job"] / "generations.jsonl"
        rows = reporter.jsonl(path)
        rows[0]["actions"] = [0]
    else:
        path = release["job"] / "predictions.jsonl"
        rows = reporter.jsonl(path)
        if change == "missing":
            rows.pop()
        else:
            rows[0]["input_identity"]["user_content_sha256"] = "0" * 64
    lines(path, rows)
    refresh(release)
    report, _ = reporter.build_report(release["root"], release["screen"])
    assert report["jobs"][0]["status"] == "invalid"
    assert report["gold_relation_triples"] == 2


def test_failed_generation_is_full_fn_without_invalid_false_positives(release):
    save_job(release, fail=True)
    report, private = reporter.build_report(release["root"], release["screen"])
    row = report["jobs"][0]
    assert row["status"] == "verified"
    assert row["relation"]["tp"] == row["relation"]["fp"] == 0 and row["relation"]["fn"] == 2
    assert row["failed_samples"] == row["failed_batches"] == row["budget_exhaustions"] == 2
    assert len(private["parent_eval"]) == 2 and "source" in private["parent_eval"][0]
    assert "Feature" not in json.dumps(report)


def test_rule_removed_tp_fp_count_unique_triples_and_expose_true_positive_harm():
    graph = {"entities": [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
                          {"id": "S2", "label": "Symptom", "name": "Dizziness"},
                          {"id": "D1", "label": "Disease", "name": "Condition Alpha"}],
        "relations": [{"source": "S1", "target": t, "relation": "is_core_symptom_of", "evidence": "Source"}
                      for t in ("S2", "D1")]}
    record = {"messages": [{"role": "user", "content": "Source"}, {"role": "assistant", "content": json.dumps(graph)}]}
    predicted = deepcopy(graph)
    fp = {"source": "D1", "target": "S2", "relation": "is_core_symptom_of", "evidence": "Source"}
    predicted["relations"].extend([fp, deepcopy(fp)])
    predictions = [{"output": predicted, "input_identity": make_input_identity(record, 17)}]
    rule = reporter.recount_rules([record], predictions, [17])["type_guard"]
    assert rule["removed_tp"] == rule["removed_fp"] == 1
    assert rule["removed_raw_rows"] == 3
    assert rule["relation"]["tp"] == rule["relation"]["fn"] == 1 and rule["relation"]["fp"] == 0


def test_incomplete_run_has_no_accepted_metric(release):
    path = release["job"] / "metrics.json"
    metrics = reporter.read(path)
    metrics.update(status="incomplete", complete=False)
    write(path, metrics)
    refresh(release)
    report, _ = reporter.build_report(release["root"], release["screen"])
    assert report["jobs"][0]["status"] == "incomplete"
    assert "relation" not in report["jobs"][0] and "diagnostic_relation" in report["jobs"][0]


def test_deadline_cannot_be_relabelled_complete_by_changing_summary(release):
    save_job(release, fail=True)
    path = release["job"] / "generations.jsonl"
    rows = reporter.jsonl(path)
    rows[0]["choice_trace"]["deadline_exhausted"] = True
    lines(path, rows)
    refresh(release)
    row = reporter.build_report(release["root"], release["screen"])[0]["jobs"][0]
    assert row["status"] == "invalid" and row["validation_error"] == "deadline_claimed_complete"


def test_cli_private_examples_permissions_and_aggregate_outputs(release, tmp_path):
    save_job(release, fail=True)
    prefix = tmp_path / "public" / "comparison"
    reporter.main(["--root", str(release["root"]), "--screen-dir", str(release["screen"]), "--output-prefix", str(prefix)])
    assert Path(str(prefix) + ".json").exists() and Path(str(prefix) + ".md").exists()
    private = release["root"] / "medical_error_examples_private.json"
    assert private.stat().st_mode & 0o777 == 0o600
    assert "Feature" in private.read_text() and "Feature" not in Path(str(prefix) + ".json").read_text()

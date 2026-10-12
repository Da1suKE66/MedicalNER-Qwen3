"""Independent recount must reject plausible-looking aggregate-only claims."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from generative.protocol import ProtocolConfig, dumps
from generative.rapid_screen import prepare_screen, load_screen, run_condition
from generative.evaluate import run_records, rule_ablation
from evaluate_graphs import evaluate_records

SPEC = importlib.util.spec_from_file_location("report_rapid_screen", Path(__file__).parents[1] / "scripts/report_rapid_screen.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def record(i):
    text = f"Feature {i} is a core symptom of Condition {i}."
    graph = {"entities": [{"id": "S", "label": "Symptom", "name": f"Feature {i}"},
                          {"id": "D", "label": "Disease", "name": f"Condition {i}"}],
             "relations": [{"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": text}]}
    return {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": dumps(graph)}]}


def generator(group, deadline=None):
    return [{"text": dumps({"decisions": [
        {"pair_id": p["pair_id"], "relations": ([{"relation": "is_core_symptom_of", "evidence_span_ids": ["SENT_001"]}]
         if p["source"] == "S" else [])} for p in b["pairs"]]}), "tokens": 30, "truncated": False}
        for b in group]


@pytest.fixture
def release(tmp_path):
    gold, matrix = tmp_path / "gold.json", tmp_path / "matrix.json"
    write(gold, [record(i) for i in range(15)])
    write(matrix, {"defaults": asdict(ProtocolConfig(matching="nfkc_v2")), "variants": {"V4": {}}})
    screen = tmp_path / "screen"
    prepare_screen(gold, matrix, "V4", screen)
    records, indices, configs, manifest = load_screen(screen)
    root = tmp_path / "screen_v1"
    job = root / "grounded64_screen"
    condition = "full_hash16"
    result = run_condition(records, indices, configs[condition], generator, job / condition,
        expected_batches=12, max_new_tokens=2048, condition_seconds=1500, inference_batch_size=4,
        manifest_sha256=reporter.digest(screen / "manifest.json"))
    write(job / "run.json", {"status": "complete", "manifest_sha256": reporter.digest(screen / "manifest.json"),
        "conditions": {condition: {"complete": True, "main_relation_metric": result["main_relation_metric"]}}})
    queue = tmp_path / "queue.json"
    write(queue, {"jobs": [{"name": "grounded64_screen", "kind": "screen", "conditions": [condition]},
                           {"name": "not_started_screen", "kind": "screen", "conditions": [condition]}]})
    return {"root": root, "screen": screen, "queue": queue, "job": job,
            "condition": condition, "records": records, "indices": indices, "configs": configs, "tmp": tmp_path}


def build(r):
    return reporter.build_report(r["root"], r["screen"], queue_config=r["queue"])


def test_complete_counts_from_predictions_and_missing_jobs_are_retained(release):
    result = build(release)
    row = result["medical"][0]
    assert row["status"] == "verified"
    assert row["relation"] == {"tp": 12, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert result["denominators"]["medical_gold_triples"] == 12
    assert result["medical"][1]["status"] == "missing"
    assert "relation" not in result["medical"][1]
    serialized = json.dumps(result)
    assert "Feature" not in serialized and "Condition 0" not in serialized
    assert "messages" not in serialized and "sample_index" not in serialized
    assert "grounded64_screen" in reporter.markdown(result)


def test_plausible_metric_with_wrong_gold_denominator_is_rejected(release):
    path = release["job"] / release["condition"] / "metrics.json"
    metrics = json.loads(path.read_text())
    wrong = reporter.prf(12, 0, 3)
    metrics["main_relation_metric"] = wrong
    for scope in ("overall", "free_text"):
        metrics["metrics_including_all_failed_rows"][scope]["relation"] = wrong
    write(path, metrics)
    result = build(release)["medical"][0]
    assert result["status"] == "invalid"
    assert result["validation_error"] == "prediction_metric_mismatch"
    assert "relation" not in result


def test_frozen_gold_denominator_is_independently_recounted(release):
    path = release["screen"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["denominators"]["gold_relation_triples"] = 40
    write(path, manifest)
    with pytest.raises(reporter.InvalidReport, match="frozen_gold_denominator"):
        build(release)


def test_missing_prediction_row_is_rejected_even_if_artifact_hash_updated(release):
    out = release["job"] / release["condition"]
    predictions = reporter.jsonl(out / "predictions.jsonl")
    (out / "predictions.jsonl").write_text("".join(dumps(r) + "\n" for r in predictions[:-1]))
    metrics = reporter.read_json(out / "metrics.json")
    metrics["artifact_sha256"]["predictions.jsonl"] = reporter.digest(out / "predictions.jsonl")
    write(out / "metrics.json", metrics)
    row = build(release)["medical"][0]
    assert row["status"] == "invalid" and "relation" not in row


def test_predictions_are_recounted_not_trusted_even_when_length_and_identity_match(release):
    out = release["job"] / release["condition"]
    predictions = reporter.jsonl(out / "predictions.jsonl")
    predictions[0]["output"]["relations"] = []
    (out / "predictions.jsonl").write_text("".join(dumps(r) + "\n" for r in predictions))
    metrics = reporter.read_json(out / "metrics.json")
    metrics["artifact_sha256"]["predictions.jsonl"] = reporter.digest(out / "predictions.jsonl")
    write(out / "metrics.json", metrics)
    row = build(release)["medical"][0]
    assert row["status"] == "invalid" and row["validation_error"] == "prediction_metric_mismatch"


def test_run_complete_and_metrics_complete_both_required(release):
    path = release["job"] / "run.json"
    run = reporter.read_json(path)
    run["conditions"][release["condition"]]["complete"] = False
    write(path, run)
    row = build(release)["medical"][0]
    assert row["status"] == "incomplete" and "relation" not in row


def test_completed_condition_can_be_reported_while_other_conditions_run(release):
    path = release["job"] / "run.json"
    run = reporter.read_json(path)
    run["status"] = "running"
    write(path, run)
    row = build(release)["medical"][0]
    assert row["status"] == "verified" and row["job_status"] == "running"


def test_generation_rows_cannot_hide_missing_batches(release):
    out = release["job"] / release["condition"]
    rows = reporter.jsonl(out / "generations.jsonl")
    (out / "generations.jsonl").write_text("".join(dumps(r) + "\n" for r in rows[:-1]))
    metrics = reporter.read_json(out / "metrics.json")
    metrics["artifact_sha256"]["generations.jsonl"] = reporter.digest(out / "generations.jsonl")
    write(out / "metrics.json", metrics)
    row = build(release)["medical"][0]
    assert row["status"] == "invalid" and row["validation_error"] == "generation_denominator_mismatch"


def test_mechanical_evidence_is_separate_and_recounted_against_own_gold(release):
    root, tmp = release["root"], release["tmp"]
    mechanical_gold = tmp / "mechanical.json"
    records = [record(77), record(78)]
    write(mechanical_gold, records)
    predictions, raw, audit = run_records(records, None, ProtocolConfig(matching="nfkc_v2"), None,
        oracle=True, generate_many=lambda group: generator(group), strict_identity=True)
    output = root / "dpo16_mechanical"
    output.mkdir()
    (output / "predictions.jsonl").write_text("".join(dumps(r) + "\n" for r in predictions))
    (output / "generations.jsonl").write_text("".join(dumps(r) + "\n" for r in raw))
    write(output / "metrics.json", {"metrics": evaluate_records(records, predictions, strict_identity=True),
        "coverage_and_generation": audit, "rule_ablation": rule_ablation(records, predictions, strict_identity=True)})
    queue = reporter.read_json(release["queue"])
    queue["expected_sha256"] = {"/remote/mechanical.json": reporter.digest(mechanical_gold)}
    queue["jobs"].append({"name": "dpo16_mechanical", "kind": "evaluation", "expected_samples": 2,
                          "command": ["python", "--gold", "/remote/mechanical.json"]})
    write(release["queue"], queue)
    result = reporter.build_report(root, release["screen"], queue_config=release["queue"], mechanical_gold=mechanical_gold)
    assert len(result["medical"]) == 2 and result["denominators"]["medical_gold_triples"] == 12
    assert result["mechanical"][0]["relation"]["tp"] == 2
    assert result["mechanical"][0]["scope"] == "artificial_mechanical_only"
    assert "Artificial mechanical tasks (separate)" in reporter.markdown(result)


def test_observed_exposure_never_invents_positive_negative_measurements(release):
    queue = reporter.read_json(release["queue"])
    queue["jobs"].append({"name": "matched_control", "kind": "train", "command": ["python", "--data", "/remote/data/V4_control.jsonl"]})
    data_manifest = release["tmp"] / "data_manifest.json"
    write(data_manifest, {"output_sha256": {"V4_control.jsonl": "a" * 64}, "audits": {"V4_control": {
        "admission": {"rows": 256, "positive_pairs": 818, "negative_pairs": 488, "supervised_pairs": 1306,
                      "candidate_pairs": 3544, "unknown_pairs": 2238}, "supervision": {"positive_edges": 821}}}})
    queue["expected_sha256"] = {"/remote/data/manifest.json": reporter.digest(data_manifest)}
    write(release["queue"], queue)
    write(release["root"] / "matched_control/run_manifest.json", {
        "status": "training_complete_evaluation_pending", "data_sha256": "a" * 64, "global_step": 16,
        "observed_exposure": {"sft_sequence_loss_exposures": 256, "sft_pair_loss_exposures": 1306,
                              "sft_candidate_pair_context_exposures": 3544, "sft_unknown_pair_context_exposures": 2238}})
    result = reporter.build_report(release["root"], release["screen"], queue_config=release["queue"], training_manifest=data_manifest)
    row = result["matched_training_exposure"][0]
    assert row["recorded_totals_match_one_pass"]
    assert row["planned_epochs"] == 1 and row["recorded_totals_match_training_budget"]
    assert row["expected_training_budget"] == row["expected_one_pass"]
    assert row["expected_one_pass"]["positive_pairs"] == 818
    assert row["observed_positive_pairs"] is None and not row["equal_positive_negative_exposure_verified"]


def test_independent_recount_includes_invalid_emitted_edge_as_false_positive():
    from evaluation_identity import make_input_identity
    records = [record(1), record(2)]
    predictions = []
    for i, item in enumerate(records):
        graph = json.loads(item["messages"][1]["content"])
        if i == 0:
            graph["relations"].append({"source": "MISSING", "target": "D", "relation": "invented"})
        else:
            graph["relations"] = []
        predictions.append({"output": graph, "input_identity": make_input_identity(item, i)})
    metric = reporter.recount(records, predictions, [0, 1])["relation"]
    assert metric == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_independent_recount_rejects_same_length_identity_swap():
    from evaluation_identity import make_input_identity
    records = [record(1), record(2)]
    predictions = [{"output": json.loads(item["messages"][1]["content"]),
                    "input_identity": make_input_identity(item, i)} for i, item in enumerate(records)]
    with pytest.raises(ValueError, match="sample_index mismatch"):
        reporter.recount(records, predictions[::-1], [0, 1])


@pytest.mark.parametrize("observed_passes, expected_match", [(3, True), (2, False)])
def test_three_epoch_exposure_budget_rejects_two_pass_totals(release, observed_passes, expected_match):
    queue = reporter.read_json(release["queue"])
    queue["jobs"].append({"name": "matched_control", "kind": "train", "expected_epochs": 3,
                          "command": ["python", "--data", "/remote/data/V4_control.jsonl"]})
    data_manifest = release["tmp"] / "data_manifest.json"
    write(data_manifest, {"output_sha256": {"V4_control.jsonl": "a" * 64}, "audits": {"V4_control": {
        "admission": {"rows": 256, "positive_pairs": 818, "negative_pairs": 488, "supervised_pairs": 1306,
                      "candidate_pairs": 3544, "unknown_pairs": 2238},
        "supervision": {"positive_edges": 821, "completion_tokens_once": 63184, "supervised_completion_tokens_once": 33487,
                        "prompt_tokens_once": 664548, "masked_completion_tokens_once": 29697}}}})
    queue["expected_sha256"] = {"/remote/data/manifest.json": reporter.digest(data_manifest)}
    write(release["queue"], queue)
    one_pass = {"sft_sequence_loss_exposures": 256, "sft_pair_loss_exposures": 1306,
                "sft_candidate_pair_context_exposures": 3544, "sft_unknown_pair_context_exposures": 2238,
                "sft_completion_token_loss_exposures": 33487, "sft_prompt_token_loss_exposures": 664548,
                "sft_input_token_loss_exposures": 727732,
                "sft_masked_completion_token_context_exposures": 29697}
    write(release["root"] / "matched_control/run_manifest.json", {
        "status": "training_complete_evaluation_pending", "data_sha256": "a" * 64,
        "global_step": observed_passes * 16,
        "observed_exposure": {key: value * observed_passes for key, value in one_pass.items()}})
    result = reporter.build_report(release["root"], release["screen"], queue_config=release["queue"], training_manifest=data_manifest)
    row = result["matched_training_exposure"][0]
    assert row["planned_epochs"] == 3
    assert row["expected_one_pass"]["positive_pairs"] == 818
    assert row["expected_training_budget"]["rows"] == 768
    assert row["expected_training_budget"]["positive_pairs"] == 2454
    assert row["expected_training_budget"]["negative_pairs"] == 1464
    assert row["expected_training_budget"]["positive_edges"] == 2463
    assert row["expected_training_budget"]["supervised_completion_tokens"] == 100461
    assert row["expected_training_budget"]["input_tokens"] == 2183196
    assert row["recorded_totals_match_training_budget"] is expected_match
    assert not row["recorded_totals_match_one_pass"]
    assert row["observed_positive_pairs"] is None and row["observed_negative_pairs"] is None
    assert row["observed_positive_edges"] is None and not row["equal_positive_negative_exposure_verified"]
    markdown = reporter.markdown(result)
    assert "Planned epochs" in markdown and "2454/1464/2463" in markdown

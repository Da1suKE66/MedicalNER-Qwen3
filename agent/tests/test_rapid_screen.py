"""Rapid-screen alignment, fair denominator and immutable selection checks."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from generative.protocol import ProtocolConfig, dumps
from generative.rapid_screen import (
    condition_configs, input_profiles, load_screen, prepare_screen,
    run_condition, select_profiles,
)
from evaluation_identity import IDENTITY_KEY


def record(index=0):
    text = f"Feature {index} is a core symptom of Condition {index}."
    graph = {"entities": [
        {"id": "S", "label": "Symptom", "name": f"Feature {index}"},
        {"id": "D", "label": "Disease", "name": f"Condition {index}"}],
        "relations": [{"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": text}]}
    return {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": dumps(graph)}]}


def empty(group, deadline):
    return [{"text": dumps({"decisions": [{"pair_id": p["pair_id"], "relations": []}
                                         for p in b["pairs"]]}), "tokens": 10} for b in group]


def condition_args(tmp_path):
    return dict(expected_batches=2, max_new_tokens=2048, condition_seconds=1500,
                inference_batch_size=4, manifest_sha256="a" * 64)


def test_labels_and_evidence_do_not_change_input_selection():
    rows = [record(i) for i in range(20)]
    changed = copy.deepcopy(rows)
    for row in changed:
        graph = json.loads(row["messages"][1]["content"])
        graph["relations"] = []
        row["messages"][1]["content"] = dumps(graph)
    conf = ProtocolConfig(matching="nfkc_v2")
    before = input_profiles(rows, conf)
    assert before == input_profiles(changed, conf)
    assert select_profiles(before) == select_profiles(before)
    selected = select_profiles(before)
    assert len(selected) == 12
    assert sum(p["candidate_pairs"] for p in selected) <= 240


def test_budget_reservation_keeps_exact_count_and_excludes_over_budget():
    profiles = [{"sample_index": i, "candidate_pairs": n, "stratum": f"s{i % 3}"}
                for i, n in enumerate([90, 0, 200, 2, 4, 6, 8, 10, 12, 20, 30, 40, 42, 50, 60])]
    selected = select_profiles(profiles, count=10, candidate_budget=200)
    assert len(selected) == 10
    assert sum(p["candidate_pairs"] for p in selected) <= 200
    assert all(0 < p["candidate_pairs"] <= 96 for p in selected)
    with pytest.raises(ValueError, match="Cannot select"):
        select_profiles(profiles, count=12, candidate_budget=1)


def test_conditions_inherit_all_frozen_v4_settings():
    config = ProtocolConfig(repair=True, strict_evidence=True, matching="nfkc_v2")
    variants = condition_configs(config)
    assert list(variants) == ["full_hash16", "full_source16", "full_hash4", "local_hash16"]
    assert all(c.matching == "nfkc_v2" and c.strict_evidence and c.repair for c in variants.values())
    assert variants["full_source16"].order == "source"
    assert variants["full_hash4"].pairs_per_batch == 4
    with pytest.raises(ValueError, match="nfkc_v2"):
        condition_configs(ProtocolConfig())


def test_frozen_records_indices_matrix_and_identity_are_verified(tmp_path):
    gold = tmp_path / "gold.json"
    matrix = tmp_path / "matrix.json"
    gold.write_text(json.dumps([record(i) for i in range(15)]))
    matrix.write_text(json.dumps({"defaults": asdict(ProtocolConfig(matching="nfkc_v2")), "variants": {"V4": {}}}))
    out = tmp_path / "screen"
    manifest = prepare_screen(gold, matrix, "V4", out)
    records, indices, configs, read = load_screen(out)
    assert len(records) == 12 and indices == manifest["sample_indices"]
    assert manifest["denominators"]["gold_relation_triples"] == 12
    assert all(plan["candidate_pairs"] == 24 for plan in manifest["conditions"].values())
    assert read == manifest and all(c.matching == "nfkc_v2" for c in configs.values())
    with pytest.raises(ValueError, match="new frozen"):
        prepare_screen(gold, matrix, "V4", out)
    (out / "records.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_screen(out)


def test_complete_condition_keeps_original_indices_and_all_gold_denominators(tmp_path):
    rows, indices = [record(1), record(2)], [9, 27]
    result = run_condition(rows, indices, ProtocolConfig(matching="nfkc_v2"), empty,
                           tmp_path / "done", **condition_args(tmp_path))
    assert result["complete"] and result["eligible_for_ranking"]
    assert result["main_relation_metric"]["fn"] == 2
    assert result["main_relation_metric"]["tp"] == 0
    predictions = [json.loads(line) for line in (tmp_path / "done/predictions.jsonl").read_text().splitlines()]
    assert [p[IDENTITY_KEY]["sample_index"] for p in predictions] == indices
    assert all(p[IDENTITY_KEY]["status"] == "verified" for p in predictions)
    assert result["observed_samples"] == 2 and result["observed_batches"] == 2
    assert set(result["artifact_sha256"]) == {"predictions.jsonl", "generations.jsonl", "raw_stream.jsonl"}


def test_expired_deadline_keeps_all_rows_but_has_no_main_f1(tmp_path):
    clock = Mock(side_effect=[0, 2000, 2000, 2000])
    generator = Mock()
    result = run_condition([record(1), record(2)], [9, 27], ProtocolConfig(matching="nfkc_v2"),
        generator, tmp_path / "timeout", clock=clock, **condition_args(tmp_path))
    generator.assert_not_called()
    assert not result["complete"] and not result["eligible_for_ranking"]
    assert result["main_relation_metric"] is None
    assert result["metrics_including_all_failed_rows"]["free_text"]["relation"]["fn"] == 2
    assert result["observed_samples"] == 2 and result["observed_batches"] == 2
    assert result["failure_counts"] == {"condition_deadline_exceeded": 2}


@pytest.mark.parametrize("returned_at", [1500, 1501])
def test_last_batch_deadline_overrun_preserves_predictions_but_cannot_rank(tmp_path, returned_at):
    now = [0]

    def generate(group, deadline):
        results = []
        for batch in group:
            decisions = [{"pair_id": p["pair_id"], "relations": [
                {"relation": "is_core_symptom_of", "evidence_span_ids": [batch["spans"][0]["span_id"]]}
            ] if p["source"] == "S" and p["target"] == "D" else []} for p in batch["pairs"]]
            results.append({"text": dumps({"decisions": decisions}), "tokens": 30})
        now[0] = returned_at
        return results

    options = {**condition_args(tmp_path), "expected_batches": 1}
    result = run_condition([record(1)], [9], ProtocolConfig(matching="nfkc_v2"), generate,
        tmp_path / "late", clock=lambda: now[0], **options)
    assert not result["complete"] and not result["eligible_for_ranking"]
    assert result["main_relation_metric"] is None
    assert result["abort_reason"] == "condition_deadline_exceeded"
    assert result["elapsed_seconds"] == returned_at
    assert result["failure_counts"] == {"condition_deadline_exceeded": 1}
    assert result["observed_samples"] == result["observed_batches"] == 1
    metrics = result["metrics_including_all_failed_rows"]["free_text"]["relation"]
    assert metrics["tp"] == 1 and metrics["fn"] == 0
    raw = json.loads((tmp_path / "late/generations.jsonl").read_text())
    assert raw["text"] and not raw.get("not_generated")
    assert raw["error"] == "condition_deadline_exceeded"


def test_condition_finalization_deadline_disqualifies_even_when_generation_finished(tmp_path, monkeypatch):
    import generative.rapid_screen as module
    now = [0]
    original_ablation = module.rule_ablation

    def delayed_ablation(*args, **kwargs):
        report = original_ablation(*args, **kwargs)
        now[0] = 1501
        return report

    monkeypatch.setattr(module, "rule_ablation", delayed_ablation)
    result = run_condition([record(1), record(2)], [9, 27], ProtocolConfig(matching="nfkc_v2"),
        empty, tmp_path / "late_scoring", clock=lambda: now[0], **condition_args(tmp_path))
    assert not result["complete"] and not result["eligible_for_ranking"]
    assert result["main_relation_metric"] is None
    assert result["abort_reason"] == "condition_deadline_exceeded"
    assert result["elapsed_seconds"] == 1501
    assert result["observed_samples"] == 2 and result["observed_batches"] == 2
    assert result["metrics_including_all_failed_rows"]["free_text"]["relation"]["fn"] == 2


def test_runtime_failure_aborts_remainder_without_dropping_gold(tmp_path):
    generator = Mock(side_effect=RuntimeError("synthetic OOM"))
    result = run_condition([record(1), record(2)], [9, 27], ProtocolConfig(matching="nfkc_v2"),
        generator, tmp_path / "failure", **condition_args(tmp_path))
    generator.assert_called_once()
    assert not result["complete"] and result["main_relation_metric"] is None
    assert result["metrics_including_all_failed_rows"]["free_text"]["relation"]["fn"] == 2
    assert result["failure_counts"] == {"generation_failed:RuntimeError": 2}


def test_decode_truncation_is_counted_and_all_rows_are_scored(tmp_path):
    def truncated(group, deadline):
        return [{"text": '{"decisions":[', "tokens": 2048, "truncated": True} for _ in group]
    result = run_condition([record(1), record(2)], [9, 27], ProtocolConfig(matching="nfkc_v2"),
        truncated, tmp_path / "truncated", **condition_args(tmp_path))
    assert result["complete"] and result["truncation_rate"] == 1.0
    assert result["parse_error_batches"] == 2
    assert result["main_relation_metric"]["fn"] == 2


def test_wrong_generation_count_cannot_produce_a_score(tmp_path):
    with pytest.raises(ValueError, match="alignment mismatch"):
        run_condition([record(1), record(2)], [9, 27], ProtocolConfig(matching="nfkc_v2"),
                      lambda group, deadline: [], tmp_path / "bad", **condition_args(tmp_path))
    assert not (tmp_path / "bad/metrics.json").exists()


def test_cli_loads_model_once_across_requested_conditions(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import generative.rapid_screen as module
    gold, matrix = tmp_path / "gold.json", tmp_path / "matrix.json"
    gold.write_text(json.dumps([record(i) for i in range(15)]))
    matrix.write_text(json.dumps({"defaults": asdict(ProtocolConfig(matching="nfkc_v2")), "variants": {"V4": {}}}))
    screen = tmp_path / "screen"
    prepare_screen(gold, matrix, "V4", screen)
    tokenizer = SimpleNamespace(pad_token=None, eos_token="EOS")
    tokenizer_loader = Mock(return_value=tokenizer)
    model = object()
    model_loader = Mock(return_value=model)
    generator = Mock(return_value=empty)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader)))
    monkeypatch.setattr(module, "load_evaluation_model", model_loader)
    monkeypatch.setattr(module, "make_generator", generator)
    monkeypatch.setattr(sys, "argv", ["rapid_screen", "evaluate", "--screen-dir", str(screen),
        "--base-model", "base", "--adapter", "adapter", "--output-dir", str(tmp_path / "results"),
        "--conditions", "full_hash16", "full_source16"])
    module.main()
    model_loader.assert_called_once_with("base", "adapter", base_only=False, grpo_sft_adapter=None)
    tokenizer_loader.assert_called_once_with("base", local_files_only=True)
    generator.assert_called_once_with(model, tokenizer, 2048)
    result = json.loads((tmp_path / "results/run.json").read_text())
    assert result["status"] == "complete"
    assert list(result["conditions"]) == ["full_hash16", "full_source16"]
    assert all(x["complete"] for x in result["conditions"].values())
    assert result["manifest_sha256"]

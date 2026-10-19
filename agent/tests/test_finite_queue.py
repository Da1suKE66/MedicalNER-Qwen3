import json

import pytest

from robustness.finite_queue import evaluation_command, publish, read_events, training_command, verify_inputs


def test_event_publication_does_not_require_rename(tmp_path, monkeypatch):
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("unsupported")))
    path = publish(tmp_path, {"status": "complete"})
    assert read_events(tmp_path)[0]["status"] == "complete"
    path.write_text("truncated")
    assert read_events(tmp_path) == []


def test_partial_event_is_not_complete_state(tmp_path):
    (tmp_path / "broken.json").write_text('{"status":')
    (tmp_path / "bad.json.sha256").write_text("wrong")
    assert read_events(tmp_path) == []


def test_frozen_input_drift_fails(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("mutated")
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_inputs({"inputs": {"train": {"path": str(path), "sha256": "wrong"}}})


def test_rl_smoke_still_uses_real_trainer_and_own_output():
    job = dict(mode="grpo", data="train.jsonl", base_model="base", max_steps=32, save_steps=16, sft_adapter="selected-sft", beta=.04)
    command = training_command(job, "smoke", smoke=True)
    assert command[command.index("--max-steps") + 1] == "2"
    assert command[command.index("--sft-adapter") + 1] == "selected-sft"
    assert "--verify-save" in command


def test_grpo_evaluation_reconstructs_sft_base_and_tune_never_uses_cache():
    job = dict(mode="grpo", tune_gold="tune.json", regression_gold="regression.json", entity_predictions="cache.jsonl", base_model="base", variant="G1_data", sft_adapter="selected-sft")
    command = evaluation_command(job, "residual", "output", tune=True)
    assert "--oracle" in command and "--entity-predictions" not in command
    assert command[command.index("--grpo-sft-adapter") + 1] == "selected-sft"

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from contextlib import nullcontext

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("rapid_queue_tests", SCRIPTS / "run_rapid_queue.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def config(tmp_path):
    return {"version": 1, "deadline": "2099-01-01T00:00:00Z", "python": sys.executable,
            "code_root": str(tmp_path / "code"), "output_dir": str(tmp_path / "runs"),
            "jobs": [{"name": "one", "kind": "screen", "conditions": ["full_hash16"],
                      "command": [sys.executable, "-m", "generative.rapid_screen"],
                      "timeout_seconds": 60, "output": str(tmp_path / "runs/one")}]}


def test_deadline_requires_timezone_and_future(tmp_path):
    c = config(tmp_path)
    c["deadline"] = "2099-01-01T00:00:00"
    with pytest.raises(ValueError, match="timezone"):
        queue.validate_config(c)
    c["deadline"] = "2000-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="passed"):
        queue.validate_config(c)


def test_dependency_must_already_be_registered(tmp_path):
    c = config(tmp_path)
    c["jobs"][0]["depends_on"] = ["missing"]
    with pytest.raises(ValueError, match="Dependencies"):
        queue.validate_config(c)


def test_output_escape_and_duplicate_rejected(tmp_path):
    c = config(tmp_path)
    c["jobs"][0]["output"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="inside"):
        queue.validate_config(c)
    c = config(tmp_path)
    c["jobs"].append({**c["jobs"][0], "name": "two"})
    with pytest.raises(ValueError, match="own output"):
        queue.validate_config(c)


def test_partial_screen_never_promoted(tmp_path):
    c = config(tmp_path);job = c["jobs"][0]
    out = Path(job["output"]);out.mkdir(parents=True)
    (out / "run.json").write_text(json.dumps({"status": "complete", "conditions": {"full_hash16": {"complete": False}}}))
    with pytest.raises(ValueError, match="Incomplete"):
        queue.validate_outputs(job)
    (out / "run.json").write_text(json.dumps({"status": "complete", "conditions": {"other": {"complete": True}}}))
    with pytest.raises(ValueError, match="conditions"):
        queue.validate_outputs(job)


def test_unchanged_train_weights_fail(tmp_path):
    out = tmp_path / "train";out.mkdir()
    (out / "run_manifest.json").write_text(json.dumps({"status": "training_complete_evaluation_pending", "policy_weights_changed": False}))
    with pytest.raises(ValueError, match="weight update"):
        queue.validate_train({"output": str(out)})


def test_zero_loader_requires_fingerprint_identity(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "zero_step_initial_policy_exported_verified", "global_step": 0,
                                                          "initial_policy_sha256": "a", "saved_adapter_sha256_verified": "b"}))
    with pytest.raises(ValueError, match="fingerprint"):
        queue.validate_outputs({"kind": "zero_control", "output": str(tmp_path)})


def test_epoch_control_rejects_accidental_step_cap(tmp_path):
    out = tmp_path / "train";out.mkdir()
    (out / "run_manifest.json").write_text(json.dumps({"status": "training_complete_evaluation_pending", "policy_weights_changed": True,
        "saved_adapter_sha256_verified": "a", "policy_adapter_sha256_after": "a", "arguments": {"epochs": 1, "max_steps": 16}}))
    with pytest.raises(ValueError, match="step cap"):
        queue.validate_train({"output": str(out), "expected_epochs": 1})


def test_independent_failure_retained_and_dependent_job_not_run(tmp_path, monkeypatch):
    c = config(tmp_path)
    original = c["jobs"][0]
    c["jobs"] = [original, {**original, "name": "dependent", "depends_on": ["one"], "output": str(tmp_path / "runs/dependent")},
                 {**original, "name": "independent", "output": str(tmp_path / "runs/independent")}]
    monkeypatch.setattr(queue, "verify_inputs", lambda c: None)
    monkeypatch.setattr(queue, "query_gpu", lambda x: {"uuid": "GPU-test"})
    monkeypatch.setattr(queue, "gpu_lock", lambda *a: nullcontext())
    monkeypatch.setattr(queue, "require_idle_gpu", lambda *a: None)
    calls = []
    def run(cmd, cwd, env, log, timeout):
        calls.append(log.stem)
        if log.stem == "one":
            raise RuntimeError("synthetic stage failure")
        out = tmp_path / "runs/independent";out.mkdir()
        (out / "run.json").write_text(json.dumps({"status": "complete", "conditions": {"full_hash16": {"complete": True}}}))
    monkeypatch.setattr(queue, "run_command", run)
    s = queue.execute(c)
    assert calls == ["one", "independent"]
    assert [j["status"] for j in s["jobs"]] == ["failed", "skipped_failed_dependency", "complete"]
    assert s["status"] == "finished_with_incomplete_jobs"


def test_deadline_prevents_new_job(tmp_path, monkeypatch):
    c = config(tmp_path)
    c["jobs"][0]["minimum_start_seconds"] = 120
    monkeypatch.setattr(queue, "seconds_left", lambda d: 30)
    monkeypatch.setattr(queue, "verify_inputs", lambda c: None)
    monkeypatch.setattr(queue, "query_gpu", lambda x: {"uuid": "GPU-test"})
    monkeypatch.setattr(queue, "gpu_lock", lambda *a: nullcontext())
    monkeypatch.setattr(queue, "run_command", lambda *a: pytest.fail("job started after deadline reserve"))
    s = queue.execute(c)
    assert s["jobs"][0]["status"] == "skipped_deadline"

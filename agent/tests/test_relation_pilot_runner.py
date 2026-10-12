"""CPU checks for pilot isolation, queue semantics, and artifact identity."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_relation_pilot.py"
spec = importlib.util.spec_from_file_location("relation_pilot_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config_file(tmp_path):
    code = tmp_path / "code"
    (code / "agent/generative").mkdir(parents=True)
    (code / "agent/generative/train.py").write_text("# synthetic code\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    files = {}
    for name, content in {
        "grounded.jsonl": '{"split":"train"}\n',
        "missing.jsonl": '{"split":"train"}\n',
        "matrix.json": json.dumps({"variants": {"V4_grounded_empty": {}, "V4_missing_only": {}}}),
        "tune.json": json.dumps([{}] * 130),
        "stress.jsonl": (json.dumps({"clean": {}, "noisy": {}}) + "\n") * 64,
    }.items():
        path = tmp_path / name
        path.write_text(content)
        files[name] = runner.sha256(path)
    config = {
        "code_root": "code", "python": sys.executable, "base_model": "base", "output_dir": "runs",
        "matrix": "matrix.json", "tune_gold": "tune.json", "stress_data": "stress.jsonl",
        "code_sha256": runner.code_digest(code), "expected_sha256": files,
        "variants": [
            {"name": "grounded", "variant": "V4_grounded_empty", "train_data": "grounded.jsonl"},
            {"name": "missing", "variant": "V4_missing_only", "train_data": "missing.jsonl"},
        ],
        "cuda_visible_devices": "2", "lock_dir": "locks",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def arg(command, flag):
    return command[command.index(flag) + 1]


def fake_artifact(stage, config):
    output = Path(stage["output"])
    output.mkdir(parents=True)
    if stage["kind"] == "train":
        adapter = Path(stage["adapter"])
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"fake test adapter, not a model")
        manifest = {
            "status": "training_complete_evaluation_pending", "global_step": 64,
            "final_adapter": str(adapter), "data_sha256": config["expected_sha256"][stage["variant"]["train_data"]],
            "policy_weights_changed": True, "saved_adapter_sha256_verified": "synthetic-weight-sha",
            "policy_adapter_sha256_after": "synthetic-weight-sha",
        }
        (output / "run_manifest.json").write_text(json.dumps(manifest))
    elif stage["kind"] == "tune":
        (output / "metrics.json").write_text(json.dumps({"metrics": {
            "overall": {"samples": 130}, "free_text": {"relation": {"tp": 1, "fp": 1, "fn": 1, "f1": .5}},
        }}))
        (output / "predictions.jsonl").write_text("{}\n" * 130)
    else:
        (output / "metrics.json").write_text(json.dumps({
            "complete_pairs": 64, "data_sha256": config["expected_sha256"][config["stress_data"]],
            "clean": {}, "noisy": {}, "noise_incident": {},
        }))


def mock_gpu(monkeypatch):
    idle = {"index": 2, "uuid": "GPU-device-two", "memory_mib": 0, "utilization_percent": 0}
    monkeypatch.setattr(runner, "query_gpu", lambda selector: idle)
    monkeypatch.setattr(runner, "require_idle_gpu", lambda config, uuid: idle)


def test_plan_has_finite_matched_budget_and_correct_adapter_order(tmp_path):
    config = runner.load_config(config_file(tmp_path))
    runner.verify_inputs(config)
    plan = runner.build_plan(config)
    assert [s["name"] for s in plan] == ["grounded-train", "base-only-tune", "grounded-tune", "grounded-stress", "missing-train", "missing-tune", "missing-stress"]
    for stage in (plan[0], plan[4]):
        command = stage["command"]
        assert arg(command, "--max-steps") == arg(command, "--save-steps") == "64"
        assert arg(command, "--batch-size") == "1"
        assert arg(command, "--gradient-accumulation") == "16"
        assert arg(command, "--seed") == "42"
        assert arg(command, "--max-length") == "6144"
        assert "--verify-save" in command
    assert "--base-only" in plan[1]["command"] and "--adapter" not in plan[1]["command"]
    for index, train_index in ((2, 0), (3, 0), (5, 4), (6, 4)):
        assert arg(plan[index]["command"], "--adapter") == plan[train_index]["adapter"]
    for stage in (plan[1], plan[2], plan[5]):
        assert "--oracle" in stage["command"] and "--strict-identity" in stage["command"]
        assert "--entity-predictions" not in stage["command"]
        assert arg(stage["command"], "--max-new-tokens") == "16000"


def test_config_preserves_venv_python_symlink(tmp_path):
    path = config_file(tmp_path)
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(sys.executable)
    config = json.loads(path.read_text())
    config["python"] = "venv-python"
    path.write_text(json.dumps(config))
    assert runner.load_config(path)["python"] == str(venv_python)


def test_reordered_variants_and_base_only_toggle(tmp_path):
    path = config_file(tmp_path)
    config = json.loads(path.read_text())
    config.update(order=["missing", "grounded"], base_only_tune=False)
    path.write_text(json.dumps(config))
    plan = runner.build_plan(runner.load_config(path))
    assert len(plan) == 6 and plan[0]["name"] == "missing-train"
    assert all("--base-only" not in s["command"] for s in plan)


@pytest.mark.parametrize("change", ["code", "data", "missing_hash"])
def test_immutable_input_hash_gate(tmp_path, change):
    config = runner.load_config(config_file(tmp_path))
    if change == "code":
        (Path(config["code_root"]) / "agent/generative/train.py").write_text("# changed\n")
    elif change == "data":
        Path(config["tune_gold"]).write_text("[]")
    else:
        config["expected_sha256"].pop(config["stress_data"])
    with pytest.raises(ValueError, match="SHA256"):
        runner.verify_inputs(config)


def test_default_dry_run_never_touches_gpu_or_output(tmp_path, monkeypatch, capsys):
    path = config_file(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(path)])
    monkeypatch.setattr(runner, "query_gpu", lambda *args: pytest.fail("dry-run touched GPU"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["execute"] is False
    assert not (tmp_path / "runs").exists()


def test_gpu_query_uses_requested_physical_device_and_resolves_uuid(monkeypatch):
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return "2, GPU-device-two, 12, 0\n"

    monkeypatch.setattr(runner.subprocess, "check_output", query)
    assert runner.query_gpu("2")["uuid"] == "GPU-device-two"
    assert arg(commands[0], "-i") == "2"


def test_active_compute_process_blocks_even_with_low_memory(monkeypatch):
    monkeypatch.setattr(runner, "query_gpu", lambda *a: {"memory_mib": 1, "utilization_percent": 0})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "GPU-device-two, 123\n")
    with pytest.raises(RuntimeError, match="not idle"):
        runner.require_idle_gpu({"idle_memory_mib": 1024, "idle_utilization_percent": 5}, "GPU-device-two")


def test_uuid_lock_prevents_concurrent_runner(tmp_path):
    with runner.gpu_lock(tmp_path, "GPU-one"):
        with pytest.raises(RuntimeError, match="holds"):
            with runner.gpu_lock(tmp_path, "GPU-one"):
                pytest.fail("second lock was admitted")


def test_complete_mock_queue_uses_one_uuid_and_records_all_stages(tmp_path, monkeypatch):
    config = runner.load_config(config_file(tmp_path))
    mock_gpu(monkeypatch)
    plan = runner.build_plan(config)
    invoked = []

    def run(command, cwd, env, log_path, timeout):
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-device-two"
        assert env["HF_HUB_OFFLINE"] == "1"
        stage = plan[len(invoked)]
        assert command == stage["command"]
        invoked.append(stage["name"])
        Path(log_path).write_text("synthetic execution\n")
        fake_artifact(stage, config)

    monkeypatch.setattr(runner, "run_command", run)
    result = runner.execute(config)
    assert result["status"] == "complete" and len(invoked) == 7
    saved = json.loads((Path(config["output_dir"]) / "status.json").read_text())
    assert all(stage["status"] == "complete" for stage in saved["stages"])
    assert len(list(Path(config["output_dir"]).glob("*.log"))) == 7


def test_stage_failure_stops_queue_and_records_failure_without_skipping(tmp_path, monkeypatch):
    config = runner.load_config(config_file(tmp_path))
    mock_gpu(monkeypatch)
    invoked = []

    def fail(*args):
        invoked.append(args[0])
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(runner, "run_command", fail)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        runner.execute(config)
    saved = json.loads((Path(config["output_dir"]) / "status.json").read_text())
    assert saved["status"] == saved["stages"][0]["status"] == "failed"
    assert len(invoked) == len(saved["stages"]) == 1
    with pytest.raises(ValueError, match="new/empty"):
        runner.execute(config)


def test_wrong_final_adapter_and_partial_evaluation_are_rejected(tmp_path):
    config = runner.load_config(config_file(tmp_path))
    plan = runner.build_plan(config)
    fake_artifact(plan[0], config)
    manifest_path = Path(plan[0]["output"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_adapter"] = "/a/different/old/adapter"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="different adapter"):
        runner.verify_stage(plan[0], config)
    fake_artifact(plan[1], config)
    (Path(plan[1]["output"]) / "predictions.jsonl").write_text("{}\n" * 129)
    with pytest.raises(RuntimeError, match="prediction count"):
        runner.verify_stage(plan[1], config)


@pytest.mark.parametrize("selector", ["0,1", "", "-1", "MIG-device"])
def test_reject_multiple_or_unsupported_gpu_selectors(tmp_path, selector):
    path = config_file(tmp_path)
    config = json.loads(path.read_text())
    config["cuda_visible_devices"] = selector
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exactly one"):
        runner.load_config(path)

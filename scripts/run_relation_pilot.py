#!/usr/bin/env python3
"""Finite single-GPU relation pilot; print a verified plan unless --execute.

Config paths are absolute or relative to the config file. Required fields:
code_root, python, base_model, output_dir, matrix, tune_gold, stress_data,
variants=[{name, variant, train_data}], expected_sha256={path: sha256},
code_sha256=code_digest(code_root). The variants list defines execution order;
optional order must list those names exactly once. Every data/matrix input must
have an expected hash. Code digest covers agent/src Python and JSON plus this
runner. Regenerate it after any code edit, never during execution.

Defaults: 64 SFT steps, seed 42, lr 5e-5, BF16 rank 16/alpha 32 (training entry),
batch 1, accumulation 16, length 6144, save/reload validation, inference batch 1,
16000 new tokens. First train -> base-only tune -> trained tune/stress -> next
train/tune/stress. This is a dense-output supervision diagnostic, not sparse
training, full end-to-end evaluation, or evidence of a quality improvement.

cuda_visible_devices selects ONE physical nvidia-smi index or GPU UUID. The
resolved UUID is locked and passed to every child, avoiding ordinal mismatch.
No training data, test labels, adapters, or existing outputs are overwritten.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_digest(root):
    root = Path(root).resolve()
    paths = set()
    for directory in (root / "agent", root / "src"):
        paths.update(path for path in directory.rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json"}
                     and not {"__pycache__", ".pytest_cache", "reports", "data", "outputs"}.intersection(path.relative_to(root).parts))
    runner = root / "scripts/run_relation_pilot.py"
    if runner.is_file():
        paths.add(runner)
    if not paths or not (root / "agent/generative/train.py").is_file():
        raise ValueError("code_root must contain the agent training implementation")
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".status-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("version", 1) != 1:
        raise ValueError("Unsupported runner config version")

    def resolve(value, *, follow_symlinks=True):
        value = Path(value).expanduser()
        value = value if value.is_absolute() else path.parent / value
        # Resolving a venv's python symlink would silently select base Python.
        return str(value.resolve() if follow_symlinks else Path(os.path.abspath(value)))

    for key in ("code_root", "python", "base_model", "output_dir", "matrix", "tune_gold", "stress_data"):
        config[key] = resolve(config[key], follow_symlinks=key != "python")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("At least one explicitly configured variant is required")
    names = []
    for variant in variants:
        name = variant["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Variant names must be safe single directory names")
        if not isinstance(variant["variant"], str) or not variant["variant"]:
            raise ValueError("Each evaluation matrix variant must be explicit")
        variant["train_data"] = resolve(variant["train_data"])
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate variant names")
    order = config.get("order", names)
    if len(order) != len(names) or set(order) != set(names):
        raise ValueError("order must list every variant exactly once")
    config["variants"] = [next(v for v in variants if v["name"] == name) for name in order]
    config["expected_sha256"] = {resolve(key): value for key, value in config["expected_sha256"].items()}
    for key, default in (("steps", 64), ("seed", 42), ("batch_size", 1),
                         ("gradient_accumulation", 16), ("max_length", 6144),
                         ("inference_batch_size", 1), ("max_new_tokens", 16000),
                         ("expected_tune_samples", 130), ("expected_stress_pairs", 64),
                         ("stage_timeout_seconds", 21600)):
        config.setdefault(key, default)
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a valid positive integer (seed may be zero)")
    config.setdefault("learning_rate", 5e-5)
    if not isinstance(config["learning_rate"], (int, float)) or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    config.setdefault("base_only_tune", True)
    if not isinstance(config["base_only_tune"], bool):
        raise ValueError("base_only_tune must be boolean")
    # Stress uses its audited fixed 16000 cap, so tune must retain the same cap.
    if config["max_new_tokens"] != 16000:
        raise ValueError("This pilot uses the complete historical 16000-token decoding budget")
    selector = str(config.get("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", selector):
        raise ValueError("Select exactly one physical GPU index or GPU UUID")
    config["cuda_visible_devices"] = selector
    config["lock_dir"] = resolve(config.get("lock_dir", "/tmp/medicalner-gpu-locks"))
    for key, default in (("idle_memory_mib", 1024), ("idle_utilization_percent", 5)):
        config.setdefault(key, default)
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    config["config_sha256"] = sha256(path)
    config["config_path"] = str(path)
    output = Path(config["output_dir"])
    for protected in (Path(config["code_root"]), Path(config["base_model"])):
        if output == protected or output.is_relative_to(protected):
            raise ValueError("Pilot outputs must be outside code and base-model directories")
    return config


def verify_inputs(config):
    if code_digest(config["code_root"]) != config["code_sha256"]:
        raise ValueError("Code SHA256 differs from the frozen config")
    files = {config[key] for key in ("matrix", "tune_gold", "stress_data")}
    files.update(v["train_data"] for v in config["variants"])
    expected = config["expected_sha256"]
    for path in files:
        if path not in expected:
            raise ValueError(f"Missing expected input SHA256: {path}")
    for path, digest in expected.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256(path) != digest:
            raise ValueError(f"Input SHA256 mismatch: {path}")
    matrix = json.loads(Path(config["matrix"]).read_text())
    if any(v["variant"] not in matrix["variants"] for v in config["variants"]):
        raise ValueError("An evaluation variant is missing from the frozen matrix")
    tune = json.loads(Path(config["tune_gold"]).read_text())
    stress = [json.loads(line) for line in Path(config["stress_data"]).read_text().splitlines() if line.strip()]
    if not isinstance(tune, list) or len(tune) != config["expected_tune_samples"]:
        raise ValueError("Tune set count differs from the declared complete set")
    if len(stress) != config["expected_stress_pairs"] or any(not isinstance(row, dict) or not {"clean", "noisy"} <= set(row) for row in stress):
        raise ValueError("Stress set must contain every declared clean/noisy pair")
    if not Path(config["python"]).is_file() or not (Path(config["base_model"]) / "config.json").is_file():
        raise ValueError("Missing Python runtime or local base-model config")


def check_empty_output(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new/empty pilot output directory; resume/overwrite is not supported")


def build_plan(config):
    stages = []
    prefix = [config["python"], "-u", "-m"]
    for index, variant in enumerate(config["variants"]):
        root = Path(config["output_dir"]) / variant["name"]
        final = str(root / "train/final")
        train = prefix + ["generative.train", "--mode", "sft", "--data", variant["train_data"],
                          "--base-model", config["base_model"], "--output-dir", str(root / "train"),
                          "--max-steps", str(config["steps"]), "--save-steps", str(config["steps"]),
                          "--save-total-limit", "1", "--verify-save"]
        for key in ("seed", "learning_rate", "batch_size", "gradient_accumulation", "max_length"):
            train += ["--" + key.replace("_", "-"), str(config[key])]
        stages.append({"name": variant["name"] + "-train", "kind": "train", "variant": variant,
                       "output": str(root / "train"), "adapter": final, "command": train})
        tune = prefix + ["generative.evaluate", "--gold", config["tune_gold"], "--oracle", "--strict-identity",
                         "--base-model", config["base_model"], "--variant", variant["variant"],
                         "--matrix", config["matrix"], "--max-new-tokens", str(config["max_new_tokens"]),
                         "--inference-batch-size", str(config["inference_batch_size"])]
        if index == 0 and config["base_only_tune"]:
            output = str(Path(config["output_dir"]) / "base-only-tune")
            stages.append({"name": "base-only-tune", "kind": "tune", "output": output,
                           "command": tune + ["--base-only", "--output-dir", output]})
        stages.append({"name": variant["name"] + "-tune", "kind": "tune", "output": str(root / "tune"),
                       "adapter": final, "command": tune + ["--adapter", final, "--output-dir", str(root / "tune")]})
        stress = prefix + ["robustness.noise_evaluation", "evaluate", "--data", config["stress_data"],
                           "--base-model", config["base_model"], "--adapter", final,
                           "--batch-size", str(config["inference_batch_size"]), "--output-dir", str(root / "stress")]
        stages.append({"name": variant["name"] + "-stress", "kind": "stress", "output": str(root / "stress"),
                       "adapter": final, "command": stress})
    return stages


def query_gpu(selector):
    output = subprocess.check_output([
        "nvidia-smi", "-i", selector, "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = [line.strip().split(",") for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError("GPU selector did not resolve to exactly one physical device")
    index, uuid, memory, utilization = [x.strip() for x in rows[0]]
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid):
        raise RuntimeError("GPU has no usable UUID")
    return {"index": int(index), "uuid": uuid, "memory_mib": int(memory), "utilization_percent": int(utilization)}


def require_idle_gpu(config, uuid):
    # nvidia-smi utilization may describe the final second of our exited child.
    # Wait briefly only for that stale counter; an active process fails at once.
    deadline = time.monotonic() + 10
    while True:
        gpu = query_gpu(uuid)
        processes = subprocess.check_output([
            "nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        if gpu["memory_mib"] > config["idle_memory_mib"] or processes:
            raise RuntimeError(f"Selected GPU is not idle: {gpu}; active_compute_processes={bool(processes)}")
        if gpu["utilization_percent"] <= config["idle_utilization_percent"]:
            return gpu
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Selected GPU remains utilized: {gpu}")
        time.sleep(1)


@contextmanager
def gpu_lock(directory, uuid):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (uuid + ".lock")).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another pilot holds the selected GPU UUID lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_command(command, cwd, env, log_path, timeout):
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        if code:
            raise RuntimeError(f"Stage exited with code {code}; inspect {log_path}")


def validate_adapter(path):
    path = Path(path)
    if not (path / "adapter_config.json").is_file() or not any((path / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise RuntimeError(f"Training did not produce a complete adapter: {path}")


def verify_stage(stage, config):
    output = Path(stage["output"])
    if stage["kind"] == "train":
        manifest = json.loads((output / "run_manifest.json").read_text())
        if manifest.get("status") != "training_complete_evaluation_pending" or manifest.get("global_step") != config["steps"]:
            raise RuntimeError("Training manifest did not confirm the complete step budget")
        if Path(manifest.get("final_adapter", "")).resolve() != Path(stage["adapter"]).resolve():
            raise RuntimeError("Training saved a different adapter than the evaluator would load")
        expected_data = config["expected_sha256"][stage["variant"]["train_data"]]
        if manifest.get("data_sha256") != expected_data or not manifest.get("policy_weights_changed"):
            raise RuntimeError("Training data or weight-update validation failed")
        if not manifest.get("saved_adapter_sha256_verified") or manifest["saved_adapter_sha256_verified"] != manifest.get("policy_adapter_sha256_after"):
            raise RuntimeError("Training lacks successful saved-weight verification")
        validate_adapter(stage["adapter"])
        return {"steps": manifest["global_step"], "adapter": stage["adapter"], "data_sha256": expected_data}
    metrics = json.loads((output / "metrics.json").read_text())
    if stage["kind"] == "tune":
        if metrics["metrics"]["overall"]["samples"] != config["expected_tune_samples"]:
            raise RuntimeError("Incomplete tuning evaluation; never promote partial results")
        predictions = [line for line in (output / "predictions.jsonl").read_text().splitlines() if line.strip()]
        if len(predictions) != config["expected_tune_samples"]:
            raise RuntimeError("Tuning prediction count differs from complete input count")
        return {"samples": len(predictions), "free_text_relation": metrics["metrics"]["free_text"]["relation"]}
    if metrics.get("complete_pairs") != config["expected_stress_pairs"] or metrics.get("data_sha256") != config["expected_sha256"][config["stress_data"]]:
        raise RuntimeError("Incomplete or wrong-input stress evaluation")
    return {"complete_pairs": metrics["complete_pairs"], "clean": metrics["clean"], "noisy": metrics["noisy"], "noise_incident": metrics["noise_incident"]}


def execute(config):
    output = Path(config["output_dir"])
    check_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive claim also prevents two different GPU jobs sharing one output.
    with (output / ".pilot-owner").open("x") as handle:
        handle.write(str(os.getpid()) + "\n")
    status = {"status": "preflight", "started_at": stamp(), "pid": os.getpid(),
              "config": config, "stages": [], "quality_claim": "No quality improvement claimed; dense supervision diagnostic"}
    status_path = output / "status.json"
    atomic_json(status_path, status)

    def terminated(signum, frame):
        raise InterruptedError(f"Pilot interrupted by signal {signum}")

    previous_termination_handler = signal.signal(signal.SIGTERM, terminated)
    try:
        verify_inputs(config)
        gpu = query_gpu(config["cuda_visible_devices"])
        with gpu_lock(config["lock_dir"], gpu["uuid"]):
            status["gpu"] = require_idle_gpu(config, gpu["uuid"])
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       PYTHONPATH=os.pathsep.join([str(Path(config["code_root"]) / "agent"), str(Path(config["code_root"]) / "src")]),
                       PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
            for stage in build_plan(config):
                verify_inputs(config)
                require_idle_gpu(config, gpu["uuid"])
                if Path(stage["output"]).exists():
                    raise RuntimeError("A stage output already exists; refusing overwrite or implicit resume")
                if stage["kind"] != "train" and stage.get("adapter"):
                    validate_adapter(stage["adapter"])
                log = output / (stage["name"] + ".log")
                record = {**stage, "status": "running", "started_at": stamp(), "log": str(log)}
                status["stages"].append(record)
                status.update(status="running", active_stage=stage["name"], updated_at=stamp())
                atomic_json(status_path, status)
                run_command(stage["command"], config["code_root"], env, log, config["stage_timeout_seconds"])
                record.update(status="complete", completed_at=stamp(), result=verify_stage(stage, config))
                atomic_json(status_path, status)
            status.update(status="complete", completed_at=stamp(), active_stage=None)
    except BaseException as exc:
        status.update(status="failed", failed_at=stamp(), error_type=type(exc).__name__, error=str(exc))
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1].update(status="failed", error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_termination_handler)
        atomic_json(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Default: verify immutable inputs and print the plan without writes/GPU calls")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.execute:
        execute(config)
    else:
        check_empty_output(config["output_dir"])
        verify_inputs(config)
        print(json.dumps({"execute": False, "stages": build_plan(config)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Bounded single-GPU experiment queues with immutable, verified state events.

Object-backed experiment storage need not support rename. An event is published
as a new JSON plus a separate SHA256 commit marker; partial files are not state.
There is no polling for dependency completion, retry loop, or automatic GPU claim.
Each launch owns one explicitly verified GPU UUID and stops on the first failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from generative.run_experiment import select_checkpoint


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(directory, value):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex}.json"
    path = directory / name
    payload = (json.dumps({"at": stamp(), **value}, ensure_ascii=False, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    digest = hashlib.sha256(payload).hexdigest()
    if sha(path) != digest:
        raise IOError("State event failed read-back verification")
    with Path(str(path) + ".sha256").open("x") as handle:
        handle.write(digest + "\n")
    return path


def read_events(directory):
    events = []
    for marker in sorted(Path(directory).glob("*.json.sha256")):
        path = Path(str(marker)[:-7])
        try:
            if sha(path) != marker.read_text().strip():
                continue
            events.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return events


def verify_inputs(job):
    for name, spec in job.get("inputs", {}).items():
        if sha(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    root = Path(job["code"])
    for name, expected in job.get("code_hashes", {}).items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")


def verify_evaluation(output, sample_count):
    output = Path(output)
    report = json.loads((output / "metrics.json").read_text())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != sample_count or report["metrics"]["overall"]["samples"] != sample_count:
        raise ValueError("Evaluation is incomplete; never publish partial F1 as full evaluation")
    for condition in ("overall", "free_text", "structured_icd"):
        if condition not in report["metrics"]:
            continue
        for kind in ("entity", "relation"):
            metric = report["metrics"][condition][kind]
            tp, fp, fn = (metric[k] for k in ("tp", "fp", "fn"))
            expected = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if abs(metric["f1"] - expected) > 1e-10:
                raise ValueError("Reported micro F1 is inconsistent with counts")
    return report


def evaluation_command(job, adapter, output, *, tune=False, oracle=False):
    cmd = [sys.executable, "-u", "-m", "generative.evaluate", "--gold", job["tune_gold"] if tune else job["regression_gold"],
           "--base-model", job["base_model"], "--adapter", str(adapter), "--variant", job["variant"],
           "--output-dir", str(output), "--max-new-tokens", "16000", "--inference-batch-size", str(job.get("inference_batch_size", 8))]
    if not tune:
        cmd += ["--entity-predictions", job["entity_predictions"]]
    if oracle or tune:
        cmd.append("--oracle")
    if job.get("matrix"):
        cmd += ["--matrix", job["matrix"]]
    if job["mode"] == "grpo":
        cmd += ["--grpo-sft-adapter", job["sft_adapter"]]
    return cmd


def training_command(job, output, *, smoke=False):
    cmd = [sys.executable, "-u", "-m", "generative.train", "--mode", job["mode"], "--data", job["data"],
           "--base-model", job["base_model"], "--output-dir", str(output), "--max-steps", str(2 if smoke else job["max_steps"]),
           "--save-steps", str(2 if smoke else job["save_steps"]), "--save-total-limit", "3",
           "--batch-size", str(job.get("batch_size", 2)), "--gradient-accumulation", str(job.get("gradient_accumulation", 8)),
           "--max-length", str(job.get("max_length", 6144)), "--seed", str(job.get("seed", 42)), "--verify-save"]
    for field in ("sft_adapter", "learning_rate", "beta", "evidence_weight", "num_generations", "max_completion_length"):
        if field in job:
            cmd += ["--" + field.replace("_", "-"), str(job[field])]
    return cmd


def verify_training(output, mode, steps):
    report = json.loads((Path(output) / "run_manifest.json").read_text())
    if report.get("status") != "training_complete_evaluation_pending" or report.get("global_step") != steps:
        raise ValueError("Training did not complete the declared optimizer-step budget")
    if not report.get("policy_weights_changed"):
        raise ValueError("Training did not update policy weights")
    delta = report.get("save_reload_max_logit_difference", float("nan"))
    if not math.isfinite(delta) or delta > .02:
        raise ValueError("Independent save/reload verification failed")
    if mode == "dpo" and report.get("reference_adapter_sha256_before") != report.get("reference_adapter_sha256_after"):
        raise ValueError("DPO frozen reference changed")
    if mode == "grpo" and not report.get("observed_exposure", {}).get("online_rollouts"):
        raise ValueError("GRPO did not generate real on-policy rollouts")
    return report


def run_job(job, queue_root, env):
    root = Path(job["output"])
    if root.exists():
        raise ValueError(f"Job output already exists; manual inspection/new run ID required: {root}")
    root.mkdir(parents=True)
    verify_inputs(job)
    publish(root / "events", {"status": "starting", "job": job})

    def run(stage, command, timeout=14400):
        publish(root / "events", {"status": "running", "stage": stage, "command": command})
        publish(queue_root / "events", {"status": "running", "job": job["name"], "stage": stage})
        started = time.monotonic()
        with (root / (stage + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=Path(job["code"]) / "agent", env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"{job['name']}/{stage} exit={result.returncode}; see {root / (stage + '.log')}")
        publish(root / "events", {"status": "stage_complete", "stage": stage, "seconds": time.monotonic() - started})

    if job["mode"] == "stress_only":
        selection = {"checkpoint": job["adapter"]}
    elif job["mode"] == "eval_only":
        output = root / "eval" / "predicted"
        run("regression_predicted", evaluation_command(job, job["adapter"], output))
        verify_evaluation(output, job["regression_count"])
        selection = {"checkpoint": job["adapter"]}
    else:
        if job["mode"] in {"dpo", "grpo"} or job.get("gpu_smoke", False):
            run("gpu_smoke", training_command(job, root / "smoke", smoke=True))
            verify_training(root / "smoke", job["mode"], 2)
        train = root / "train"
        if job.get("reuse_training"):
            if job["mode"] != "sft":
                raise ValueError("Post-hoc recovery is only registered for SFT")
            spec = root / "recovery_job.json"
            with spec.open("x") as handle:
                handle.write(json.dumps(job, indent=2) + "\n")
            run("verify_saved_training", [sys.executable, "-u", "-m", "robustness.recover_save_check",
                                         "--job-spec", str(spec), "--output-dir", str(train)])
            verify_training(train, job["mode"], job["max_steps"])
            train = Path(job["reuse_training"]["path"])
        else:
            run("train", training_command(job, train), timeout=43200 if job["mode"] == "grpo" else 28800)
            verify_training(train, job["mode"], job["max_steps"])
        checkpoints = sorted(train.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        expected = set(range(job["save_steps"], job["max_steps"] + 1, job["save_steps"])) | {job["max_steps"]}
        expected = set(sorted(expected)[-3:])
        if {int(p.name.split("-")[-1]) for p in checkpoints} != expected:
            raise ValueError("Unexpected checkpoint coverage for the declared finite budget")
        candidates = []
        for checkpoint in checkpoints:
            step = int(checkpoint.name.split("-")[-1])
            adapter = checkpoint / "policy" if job["mode"] == "dpo" else checkpoint
            output = root / "eval" / f"tune_{step}"
            run(f"tune_{step}", evaluation_command(job, adapter, output, tune=True))
            report = verify_evaluation(output, job["tune_count"])
            candidates.append({"checkpoint": str(adapter), "step": step, "relation": report["metrics"]["free_text"]["relation"]})
        selection = select_checkpoint(candidates)
        publish(root / "selection", {"selected": selection, "all_candidates": candidates, "test_used_for_selection": False})
        for mode in ("predicted", "oracle"):
            output = root / "eval" / mode
            run("regression_" + mode, evaluation_command(job, selection["checkpoint"], output, oracle=mode == "oracle"))
            verify_evaluation(output, job["regression_count"])
    if job.get("stress_data"):
        command = [sys.executable, "-u", "-m", "robustness.noise_evaluation", "evaluate", "--data", job["stress_data"],
                   "--output-dir", str(root / "eval" / "noise_stress"), "--base-model", job["base_model"], "--adapter", selection["checkpoint"]]
        if job["mode"] == "grpo":
            command += ["--grpo-sft-adapter", job["sft_adapter"]]
        run("noise_stress", command)
        report = json.loads((root / "eval" / "noise_stress" / "metrics.json").read_text())
        if report.get("complete_pairs") != job["stress_count"]:
            raise ValueError("Incomplete paired stress evaluation")
    publish(root / "events", {"status": "complete", "not_full_live_pipeline": True, "entity_conditions": "fixed cache / oracle; inspect job mode"})


def execute(path):
    path = Path(path).resolve()
    queue = json.loads(path.read_text())
    root = Path(queue["output"])
    root.mkdir(parents=True, exist_ok=True)
    for dependency in queue.get("required_completion", []):
        events = read_events(Path(dependency) / "events")
        if not events or events[-1].get("status") != "complete":
            raise ValueError("Prerequisite queue is not verified complete; do not claim this GPU")
    for dependency in queue.get("required_failure", []):
        events = read_events(Path(dependency) / "events")
        if not events or events[-1].get("status") != "failed":
            raise ValueError("Recovery predecessor has not failed; do not interrupt running work")
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    device, memory = gpu[0].strip(), int(gpu[1])
    if device != queue["gpu_uuid"] or memory > 1024:
        publish(root / "events", {"status": "blocked_gpu_not_idle_or_changed", "actual_gpu": device, "memory_mib": memory})
        return
    locks = Path(queue["lock_root"])
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (device + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(root / "events", {"status": "started", "pid": os.getpid(), "gpu_uuid": device, "host": socket.gethostname(), "queue_sha256": sha(path)})
        try:
            if not 1 <= len(queue["jobs"]) <= 8:
                raise ValueError("Each finite worker requires 1-8 explicitly listed jobs")
            for job in queue["jobs"]:
                env = dict(os.environ, PYTHONPATH=queue["dependencies"] + ":" + str(Path(job["code"]) / "agent"), PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
                run_job(job, root, env)
            publish(root / "events", {"status": "complete", "jobs": len(queue["jobs"])})
        except Exception as exc:
            publish(root / "events", {"status": "failed", "error": str(exc)})
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["launch", "execute", "status"])
    p.add_argument("--queue", required=True)
    args = p.parse_args()
    queue = json.loads(Path(args.queue).read_text())
    root = Path(queue["output"])
    if args.mode == "status":
        print(json.dumps(read_events(root / "events")[-1:], ensure_ascii=False))
        return
    if args.mode == "execute":
        execute(args.queue)
        return
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive launch marker prevents accidental retries of an uncertain SSH call.
    with (root / "launch.json").open("x") as marker:
        marker.write(json.dumps({"queue_sha256": sha(args.queue), "at": stamp()}))
    with (root / "worker.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "robustness.finite_queue", "execute", "--queue", str(Path(args.queue).resolve())], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    publish(root / "events", {"status": "launched_not_yet_completed", "pid": proc.pid})
    print(json.dumps({"pid": proc.pid, "queue": args.queue, "output": str(root)}))


if __name__ == "__main__":
    main()

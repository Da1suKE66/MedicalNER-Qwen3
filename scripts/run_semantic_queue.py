"""Run an appendable, deadline-bounded experiment plan on one GPU.

The plan contains explicit argv lists, never shell programs. A single advisory
lock prevents duplicate launch; every completed command has an immutable log
and a required success receipt. A changed plan cannot silently reuse a job ID.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def validate_plan(plan):
    if not isinstance(plan.get("deadline_unix"), (int, float)):
        raise ValueError("Explicit experiment deadline required")
    seen = set()
    for job in plan["jobs"]:
        if job["id"] in seen or not job["id"].replace("_", "").isalnum():
            raise ValueError("Job IDs must be unique simple names")
        seen.add(job["id"])
        if not job["argv"] or any(not isinstance(x, str) for x in job["argv"]):
            raise ValueError("Each job requires a literal argv list")
        if not Path(job["cwd"]).is_absolute() or not Path(job["receipt"]).is_absolute():
            raise ValueError("Job directories and receipts must be absolute")
        if job.get("estimated_seconds", 0) <= 0:
            raise ValueError("Each launch needs a declared time estimate")


def run(plan_path, state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    if state.get("current"):
        raise RuntimeError("Prior run was interrupted; reconcile its process before resuming")
    state.update({"runner_pid": os.getpid(), "status": "running"})
    write(state_path, state)
    while True:
        plan = json.loads(plan_path.read_text())
        validate_plan(plan)
        pending = []
        for job in plan["jobs"]:
            previous = state["jobs"].get(job["id"])
            if previous:
                if previous["job_sha256"] != digest(job):
                    raise RuntimeError("A previously launched job definition changed")
                if previous["status"] != "complete":
                    state["status"] = "needs_attention"
                    write(state_path, state)
                    return
            else:
                pending.append(job)
        now = time.time()
        if now >= plan["deadline_unix"]:
            state["status"] = "deadline_reached"
            write(state_path, state)
            return
        if (state_dir / "pause").exists() or not pending:
            state["status"] = "paused" if (state_dir / "pause").exists() else "waiting_for_plan"
            state["updated_unix"] = now
            write(state_path, state)
            if not pending and plan.get("stop_when_empty", False):
                return
            time.sleep(15)
            continue
        job = pending[0]
        if now + job["estimated_seconds"] > plan["deadline_unix"]:
            state.update({"status": "insufficient_remaining_time", "not_launched": job["id"]})
            write(state_path, state)
            return
        environment = dict(os.environ)
        environment.update(job.get("environment", {}))
        log_path = state_dir / (job["id"] + ".log")
        if log_path.exists() or Path(job["receipt"]).exists():
            raise RuntimeError("Refusing an untracked existing experiment artifact")
        with log_path.open("w") as log:
            process = subprocess.Popen(job["argv"], cwd=job["cwd"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            record = {"status": "running", "job_sha256": digest(job), "pid": process.pid,
                      "started_unix": now, "log": str(log_path), "receipt": job["receipt"]}
            state.update({"status": "running", "current": job["id"]})
            state["jobs"][job["id"]] = record
            write(state_path, state)
            try:
                return_code = process.wait(timeout=max(1, plan["deadline_unix"] - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return_code = -signal.SIGTERM
        record.update({"return_code": return_code, "finished_unix": time.time()})
        receipt_path = Path(job["receipt"])
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            receipt = {}
        success = return_code == 0 and receipt.get("completed") is True and not receipt.get("smoke_only", False)
        record["status"] = "complete" if success else "failed"
        if receipt_path.exists():
            record["receipt_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        state["current"] = None
        state["status"] = "running" if success else "needs_attention"
        write(state_path, state)
        if not success:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.state_dir)

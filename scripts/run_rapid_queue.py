#!/usr/bin/env python3
"""Deadline-bounded, single-GPU diagnostic queue; never promotes partial scores.

Commands and their input hashes must be explicitly frozen in a local JSON file.
Independent failures are retained; dependent jobs are skipped. This runner does
not alter the original pilot, infer quality improvements, or resume old outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

from run_relation_pilot import (
    atomic_json, code_digest, gpu_lock, query_gpu, require_idle_gpu,
    run_command, sha256, stamp, validate_adapter,
)


def seconds_left(deadline):
    end = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if end.tzinfo is None:
        raise ValueError("Deadline must have an explicit timezone")
    return (end - datetime.now(timezone.utc)).total_seconds()


def validate_config(config):
    if config.get("version") != 1:
        raise ValueError("Unsupported rapid queue version")
    if seconds_left(config["deadline"]) <= 0:
        raise ValueError("The frozen diagnostic deadline has passed")
    for key in ("python", "code_root", "output_dir"):
        if not Path(config[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    seen, outputs = set(), set()
    if not config.get("jobs"):
        raise ValueError("Queue has no jobs")
    for job in config["jobs"]:
        name = job["name"]
        if not name.replace("_", "").replace("-", "").isalnum() or name in seen:
            raise ValueError("Unsafe or duplicate job name")
        if not set(job.get("depends_on", [])) <= seen:
            raise ValueError("Dependencies must refer to earlier jobs")
        seen.add(name)
        if job["kind"] not in {"train", "screen", "zero_control", "evaluation", "choice_evaluation"}:
            raise ValueError("Unknown job kind")
        if not isinstance(job["command"], list) or not job["command"] or not all(isinstance(x, str) for x in job["command"]):
            raise ValueError("Command must be a nonempty argument vector")
        if job["command"][0] != config["python"]:
            raise ValueError("Use the frozen Python runtime")
        if not isinstance(job["timeout_seconds"], int) or job["timeout_seconds"] < 1:
            raise ValueError("Job timeout must be positive")
        output = Path(job["output"])
        if not output.is_absolute() or not output.is_relative_to(Path(config["output_dir"])):
            raise ValueError("Job outputs must be inside the new queue directory")
        if str(output) in outputs:
            raise ValueError("Each job needs its own output directory")
        outputs.add(str(output))
        if job["kind"] == "train" and job.get("mode") not in {"sft", "dpo"}:
            raise ValueError("This finite screen only registers SFT and DPO")
    return config


def verify_inputs(config):
    if code_digest(config["code_root"]) != config["code_sha256"]:
        raise ValueError("Frozen code hash mismatch")
    for path, digest in config["expected_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input hash mismatch: {path}")


def validate_train(job):
    out = Path(job["output"])
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest.get("status") != "training_complete_evaluation_pending":
        raise ValueError("Training did not finish its declared budget")
    if not manifest.get("policy_weights_changed"):
        raise ValueError("No policy weight update verified")
    if manifest.get("saved_adapter_sha256_verified") != manifest.get("policy_adapter_sha256_after") or not manifest.get("saved_adapter_sha256_verified"):
        raise ValueError("Saved adapter verification missing or mismatched")
    if job.get("expected_steps") is not None and manifest.get("global_step") != job["expected_steps"]:
        raise ValueError("Training step count differs from frozen budget")
    if job.get("expected_epochs") is not None:
        actual = manifest.get("train_metrics", {}).get("epoch")
        # Trainer versions serialize this metric differently; use the explicit
        # training contract plus completion, while retaining actual final step.
        args = manifest.get("arguments", {})
        if args.get("epochs") != job["expected_epochs"] or args.get("max_steps") != -1:
            raise ValueError("Epoch run was replaced by a finite step cap")
        if actual is not None and actual + 1e-6 < job["expected_epochs"]:
            raise ValueError("Epoch training ended prematurely")
    if job.get("mode") == "dpo":
        control = json.loads(Path(job["initial_control_manifest"]).read_text())
        if manifest.get("policy_adapter_sha256_before") != control.get("initial_policy_sha256"):
            raise ValueError("DPO did not start at the zero-control policy")
        before, after = (manifest.get("reference_adapter_sha256_" + k) for k in ("before", "after"))
        if not before or before != after:
            raise ValueError("DPO frozen reference changed")
    validate_adapter(manifest["final_adapter"])
    return {"steps": manifest["global_step"], "adapter": manifest["final_adapter"],
            "saved_adapter_sha256": manifest["saved_adapter_sha256_verified"],
            "observed_exposure": manifest.get("observed_exposure", {})}


def validate_outputs(job):
    for relative in job.get("required_files", []):
        if not (Path(job["output"]) / relative).is_file():
            raise ValueError(f"Missing output artifact: {relative}")
    if job["kind"] == "train":
        return validate_train(job)
    if job["kind"] == "choice_evaluation":
        out = Path(job["output"])
        report = json.loads((out / "metrics.json").read_text())
        rows = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines() if x.strip()]
        if report.get("status") != "complete" or report.get("samples") != job["expected_samples"] or len(rows) != job["expected_samples"]:
            raise ValueError("Choice evaluation did not process its full frozen denominator")
        identities = [row.get("id") for row in rows]
        if any(not identity for identity in identities) or len(set(identities)) != len(rows):
            raise ValueError("Choice evaluation has missing or duplicated identities")
        return {"samples": len(rows), "relation": report["relation"], "evidence": report["evidence"],
                "scope": report.get("scope"), "independent_recount_pending": True}
    if job["kind"] == "screen":
        report = json.loads((Path(job["output"]) / "run.json").read_text())
        if report.get("status") != "complete" or set(report.get("conditions", {})) != set(job["conditions"]):
            raise ValueError("Screen conditions did not all complete")
        if not all(c.get("complete") for c in report["conditions"].values()):
            raise ValueError("Incomplete condition cannot be promoted")
        return {"conditions": report["conditions"], "independent_recount_pending": True}
    if job["kind"] == "zero_control":
        manifest = json.loads((Path(job["output"]) / "run_manifest.json").read_text())
        if manifest.get("status") != "zero_step_initial_policy_exported_verified" or manifest.get("global_step") != 0:
            raise ValueError("Zero-step loader control was not verified")
        if manifest.get("initial_policy_sha256") != manifest.get("saved_adapter_sha256_verified") or not manifest.get("initial_policy_sha256"):
            raise ValueError("Zero-step fingerprint differs after reload")
        return {"initial_policy_sha256": manifest["initial_policy_sha256"]}
    if job["kind"] == "evaluation":
        out = Path(job["output"])
        report = json.loads((out / "metrics.json").read_text())
        rows = [x for x in (out / "predictions.jsonl").read_text().splitlines() if x.strip()]
        if report["metrics"]["overall"]["samples"] != job["expected_samples"] or len(rows) != job["expected_samples"]:
            raise ValueError("Mechanical evaluation missing samples")
        return {"samples": len(rows), "relation": report["metrics"]["free_text"]["relation"], "scope": "artificial_mechanical_only"}
    return {"artifact_checks_passed": True,
            "quality_metrics_require_complete_condition_and_independent_recount": True}


def verify_external_adapters(config):
    for item in config.get("external_adapter_receipts", []):
        manifest = json.loads(Path(item["manifest"]).read_text())
        if manifest.get("status") != "training_complete_evaluation_pending" or manifest.get("global_step") != item["steps"]:
            raise ValueError("External adapter training receipt is incomplete")
        if manifest.get("data_sha256") != item["data_sha256"] or not manifest.get("policy_weights_changed"):
            raise ValueError("External adapter has wrong data or no verified update")
        if manifest.get("saved_adapter_sha256_verified") != manifest.get("policy_adapter_sha256_after") or not manifest.get("saved_adapter_sha256_verified"):
            raise ValueError("External adapter save/reload is unverified")
        if manifest.get("final_adapter") != item["adapter"]:
            raise ValueError("External adapter path differs from receipt")
        validate_adapter(item["adapter"])


def execute(config):
    validate_config(config)
    verify_inputs(config)
    output = Path(config["output_dir"])
    if output.exists():
        raise ValueError("Choose a new queue output directory")
    output.mkdir(parents=True)
    with (output / ".owner").open("x") as f:
        f.write(str(os.getpid()))
    path = output / "status.json"
    status = {"status": "waiting_prerequisite", "pid": os.getpid(), "started_at": stamp(),
              "deadline": config["deadline"], "config": config, "jobs": [],
              "scope": "diagnostic_subset_not_final_quality"}
    atomic_json(path, status)

    def stop(signum, frame):
        raise InterruptedError(f"Rapid queue terminated by signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        prerequisite = config.get("wait_for_status")
        while prerequisite:
            state = json.loads(Path(prerequisite).read_text())
            if state["status"] == "complete":
                break
            if state["status"] == "failed":
                raise RuntimeError("Prerequisite training failed")
            if seconds_left(config["deadline"]) <= 0:
                raise TimeoutError("Deadline reached waiting for prerequisite")
            time.sleep(10)
        verify_external_adapters(config)
        gpu = query_gpu(config.get("cuda_visible_devices", "0"))
        limits = {"idle_memory_mib": 1024, "idle_utilization_percent": 5}
        with gpu_lock(config.get("lock_dir", "/tmp/medicalner-gpu-locks"), gpu["uuid"]):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"],
                       PYTHONPATH=config["code_root"] + "/agent:" + config["code_root"] + "/src",
                       PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
            outcomes = {}
            for job in config["jobs"]:
                record = {"name": job["name"], "kind": job["kind"], "output": job["output"]}
                status["jobs"].append(record)
                if any(outcomes.get(dep) != "complete" for dep in job.get("depends_on", [])):
                    record.update(status="skipped_failed_dependency")
                elif seconds_left(config["deadline"]) < job.get("minimum_start_seconds", 60):
                    record.update(status="skipped_deadline")
                else:
                    verify_inputs(config)
                    require_idle_gpu(limits, gpu["uuid"])
                    if Path(job["output"]).exists():
                        raise ValueError("Refusing to overwrite a job output")
                    record.update(status="running", started_at=stamp())
                    status.update(status="running", active_job=job["name"])
                    atomic_json(path, status)
                    try:
                        run_command(job["command"], config["code_root"], env,
                                    output / (job["name"] + ".log"),
                                    min(job["timeout_seconds"], max(1, int(seconds_left(config["deadline"])))) )
                        record.update(status="complete", result=validate_outputs(job), completed_at=stamp())
                    except (KeyboardInterrupt, InterruptedError):
                        raise
                    except Exception as exc:
                        record.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_at=stamp())
                outcomes[job["name"]] = record["status"]
                atomic_json(path, status)
            status.update(status="complete" if all(v == "complete" for v in outcomes.values()) else "finished_with_incomplete_jobs",
                          completed_at=stamp(), active_job=None)
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc), failed_at=stamp())
        if status["jobs"] and status["jobs"][-1].get("status") == "running":
            status["jobs"][-1].update(status="interrupted", error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        atomic_json(path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = validate_config(json.loads(Path(args.config).read_text()))
    config["config_sha256"] = sha256(args.config)
    if args.execute:
        execute(config)
    else:
        verify_inputs(config)
        print(json.dumps({"execute": False, "deadline": config["deadline"], "jobs": config["jobs"]}, indent=2))


if __name__ == "__main__":
    main()

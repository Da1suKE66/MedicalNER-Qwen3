"""Launch a finite isolated SFT -> tuning selection -> complete regression job.

This is not a recurring automation. Each launch snapshots code, owns one idle GPU,
and finishes with results or an explicit failed stage. No production model is replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def code_hash(root):
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def select_checkpoint(candidates):
    if not candidates:
        raise ValueError("No tuning checkpoints")
    eligible = [
        c
        for c in candidates
        if c["relation"]["precision"] >= 0.5 and c["relation"]["recall"] >= 0.1
    ]
    pool = eligible or candidates
    winner = max(
        pool,
        key=lambda c: (
            c["relation"]["f0_5"],
            c["relation"]["f1"],
            c["relation"]["recall"],
            -c["step"],
        ),
    )
    return {
        **winner,
        "eligibility_met": bool(eligible),
        "selection_rule": "Tune only: precision>=.5 and recall>=.1 when feasible, then maximize F0.5/F1/recall; fallback max F0.5 explicitly recorded",
    }


def execute(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    root = path.parent
    code = root / "code" / "agent"
    args = manifest["arguments"]
    lock_path = Path("/cache/liluchen/medicalner_generative_env_20260908/gpu0.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ]
            )
            .decode()
            .strip()
        )
        if memory > 1024:
            manifest.update(
                status="blocked_gpu_not_idle", gpu_memory_mib=memory, updated_at=stamp()
            )
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            return

        def run(stage, command):
            manifest.update(status="running", stage=stage, updated_at=stamp())
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"stage": stage, "at": stamp()}), flush=True)
            with (root / f"{stage}.log").open("w") as log:
                completed = subprocess.run(
                    command, cwd=code, stdout=log, stderr=subprocess.STDOUT
                )
            if completed.returncode:
                raise RuntimeError(
                    f"{stage} failed with exit code {completed.returncode}; inspect {stage}.log"
                )

        try:
            train = root / "train"
            run(
                "train",
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "generative.train",
                    "--mode",
                    "sft",
                    "--base-model",
                    args["base_model"],
                    "--data",
                    str(Path(args["data_dir"]) / "train.jsonl"),
                    "--output-dir",
                    str(train),
                    "--epochs",
                    str(args["epochs"]),
                    "--batch-size",
                    str(args["batch_size"]),
                    "--gradient-accumulation",
                    str(args["gradient_accumulation"]),
                    "--seed",
                    str(args["seed"]),
                    "--verify-save",
                ],
            )
            manifest["training_completed_at"] = stamp()
            checkpoints = sorted(
                train.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
            )
            if len(checkpoints) < int(args["epochs"]):
                raise RuntimeError(
                    "Missing per-epoch checkpoints; do not silently evaluate only the final model"
                )
            candidates = []
            for checkpoint in checkpoints:
                step = int(checkpoint.name.split("-")[-1])
                target = root / "eval" / f"tune_{step}"
                run(
                    f"tune_{step}",
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "generative.evaluate",
                        "--gold",
                        args["tune_gold"],
                        "--oracle",
                        "--base-model",
                        args["base_model"],
                        "--adapter",
                        str(checkpoint),
                        "--variant",
                        args["variant"],
                        "--output-dir",
                        str(target),
                        "--inference-batch-size",
                        str(args["inference_batch_size"]),
                    ],
                )
                report = json.loads((target / "metrics.json").read_text())
                candidates.append(
                    {
                        "checkpoint": str(checkpoint),
                        "step": step,
                        "relation": report["metrics"]["free_text"]["relation"],
                    }
                )
            selection = select_checkpoint(candidates)
            (root / "selection.json").write_text(
                json.dumps(
                    {
                        "selected": selection,
                        "all_candidates": candidates,
                        "test_used_for_selection": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            for mode in ["predicted", "oracle"]:
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "generative.evaluate",
                    "--gold",
                    args["regression_gold"],
                    "--entity-predictions",
                    args["entity_predictions"],
                    "--base-model",
                    args["base_model"],
                    "--adapter",
                    selection["checkpoint"],
                    "--variant",
                    args["variant"],
                    "--output-dir",
                    str(root / "eval" / mode),
                    "--inference-batch-size",
                    str(args["inference_batch_size"]),
                ]
                if mode == "oracle":
                    command.append("--oracle")
                run("regression_" + mode, command)
            manifest.update(
                status="sft_and_cached_oracle_evaluation_complete",
                selected_checkpoint=selection["checkpoint"],
                completed_at=stamp(),
                remaining="Cross-variant/seed statistics, live entity+relation pipeline, remaining ablations and RL are separate stages",
            )
        except Exception as exc:
            manifest.update(status="failed", error=str(exc), updated_at=stamp())
            path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute")
    p.add_argument("--output-dir")
    p.add_argument("--variant")
    p.add_argument("--data-dir")
    p.add_argument("--base-model")
    p.add_argument("--tune-gold")
    p.add_argument("--regression-gold")
    p.add_argument("--entity-predictions")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.execute:
        execute(args.execute)
        return
    for field in (
        "output_dir",
        "variant",
        "data_dir",
        "base_model",
        "tune_gold",
        "regression_gold",
        "entity_predictions",
    ):
        if not getattr(args, field):
            p.error(f"Missing --{field.replace('_','-')}")
    root = Path(args.output_dir).resolve()
    source = Path(__file__).resolve().parents[2]
    if root.is_relative_to(source):
        raise ValueError("Run directory cannot be inside the code snapshot source")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source,
        root / "code",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "reports"),
    )
    manifest = {
        "status": "launching",
        "created_at": stamp(),
        "host": socket.gethostname(),
        "arguments": vars(args),
        "code_sha256": code_hash(root / "code"),
        "data_sha256": hashlib.sha256(
            (Path(args.data_dir) / "train.jsonl").read_bytes()
        ).hexdigest(),
    }
    path = root / "supervisor.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with (root / "supervisor.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "generative.run_experiment",
                "--execute",
                str(path),
            ],
            cwd=root / "code" / "agent",
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # Separate launcher record avoids racing the child's stage manifest writes.
    (root / "process.json").write_text(
        json.dumps(
            {"pid": process.pid, "host": socket.gethostname(), "launched_at": stamp()}
        )
        + "\n"
    )
    print(json.dumps({"pid": process.pid, "run": str(root), "status": "launched"}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only extracted-snapshot preflight, isolated from the checkout imports."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

REQUIRED_RESOURCES = (
    "code/schemas/v2.0.0/schema.json", "code/schemas/current.json",
    "code/agent/kg_agent/ontology.json", "code/agent/generative/experiments.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or external snapshot resource: " + str(relative))
    return path


def worker(root):
    root = Path(root).resolve()
    resources = {name: sha(contained(root, name)) for name in REQUIRED_RESOURCES}
    pointer = read(root / "code/schemas/current.json")
    target = contained(root, "code/schemas/" + pointer["schema_path"])
    if read(target).get("schema_version") != pointer.get("schema_version"):
        raise ValueError("Schema pointer version mismatch")
    sys.path[:0] = [str(root / "code/agent"), str(root / "code/src")]
    from generative.protocol import dumps, load_variant, corrections
    from data_repair.text_normalization import normalize_with_offsets
    from generative.rapid_screen import load_screen
    from generative.evaluate import run_records
    from evaluate_graphs import evaluate_records
    from generative.supervision_guard import audit_partial_supervision
    from generative.train import validate_training_rows
    from kg_lora.schema_v2 import load_schema

    # Deliberately execute both lazy schema readers before any GPU work.
    corrections()
    normalize_with_offsets("Synthetic validation text.", repair=True)
    load_schema()
    records, indices, configs, manifest = load_screen(root / "evaluation")

    def empty_group(batches):
        return [{"text": dumps({"decisions": [{"pair_id": pair["pair_id"], "relations": []}
                                               for pair in batch["pairs"]]}), "tokens": 0}
                for batch in batches]

    def exercise(records, indices, config):
        with contextlib.redirect_stdout(io.StringIO()):
            predictions, raw, audit = run_records(records, None, config, None, oracle=True,
                generate_many=empty_group, strict_identity=True, sample_indices=indices)
        metric = evaluate_records(records, predictions, strict_identity=True, sample_indices=indices)
        if len(predictions) != len(records) or any(row["parse_errors"] for row in raw):
            raise ValueError("CPU decode-contract rehearsal failed")
        return {"samples": len(predictions), "batches": len(raw),
                "candidate_pairs": audit["candidate_pairs"],
                "gold_triples": audit["gold_triples"],
                "candidate_covered_triples": audit["candidate_covered_triples"],
                "gold_denominator_in_empty_prediction_rehearsal": metric["free_text"]["relation"]["fn"]}

    medical = {}
    for name, config in configs.items():
        result = exercise(records, indices, config)
        plan = manifest["conditions"][name]
        for key, expected in (("samples", manifest["denominators"]["samples"]),
                              ("batches", plan["expected_batches"]),
                              ("candidate_pairs", plan["candidate_pairs"]),
                              ("candidate_covered_triples", plan["candidate_covered_gold_triples"]),
                              ("gold_triples", manifest["denominators"]["gold_relation_triples"])):
            if result[key] != expected:
                raise ValueError("Frozen condition plan mismatch: " + name + ":" + key)
        if result["gold_denominator_in_empty_prediction_rehearsal"] != result["gold_triples"]:
            raise ValueError("Gold denominator lost in rehearsal")
        medical[name] = result
    data_manifest = read(contained(root, "data/manifest.json"))
    for name, expected in data_manifest["output_sha256"].items():
        if sha(contained(root, "data/" + name)) != expected:
            raise ValueError("Training artifact hash mismatch: " + name)
    training = {}
    for name in read(root / "data/experiments.json")["variants"]:
        rows = [json.loads(line) for line in contained(root, "data/" + name + ".jsonl").read_text().splitlines() if line.strip()]
        config = load_variant(root / "data/experiments.json", name)
        audit = audit_partial_supervision(rows)
        validate_training_rows(rows, "sft")
        if config.matching != "nfkc_v2" or audit["invalid_rows"]:
            raise ValueError("Training contract mismatch: " + name)
        training[name] = {k: audit[k] for k in ("rows", "positive_pairs", "negative_pairs", "unknown_pairs")}
    rl_manifest = read(contained(root, "rl/manifest.json"))
    for name, metadata in rl_manifest["files"].items():
        if sha(contained(root, "rl/" + name)) != metadata["sha256"]:
            raise ValueError("RL artifact hash mismatch: " + name)
    preferences = [json.loads(line) for line in (root / "rl/train.jsonl").read_text().splitlines() if line.strip()]
    validate_training_rows(preferences, "dpo")
    mechanical_records = read(root / "rl/mechanical_holdout_records.json")
    mechanical = exercise(mechanical_records, list(range(len(mechanical_records))),
                          load_variant(root / "rl/mechanical_matrix.json", "mechanical"))
    # Protect against accidentally validating checkout modules through PYTHONPATH.
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"generative", "data_repair", "kg_agent", "kg_lora", "robustness"}:
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root / "code"):
                raise ValueError("Runtime module imported outside extracted snapshot")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise ValueError("CPU contract preflight must not import the model runtime")
    return {"status": "passed", "scope": "CPU_contract_rehearsal_not_model_quality",
            "model_loaded": False, "resources_sha256": resources,
            "screen_manifest_sha256": sha(root / "evaluation/manifest.json"),
            "medical_conditions": medical, "training_variants": training,
            "dpo_preferences_validated": len(preferences), "mechanical": mechanical}


def validate_snapshot(snapshot_root, *, timeout=120):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    # -I ignores caller path/environment injection and imports only snapshot code.
    process = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker",
                              "--snapshot-root", str(Path(snapshot_root).resolve())],
                             env=env, capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "Extracted snapshot preflight failed")
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    result = worker(args.snapshot_root) if args.worker else validate_snapshot(args.snapshot_root)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

"""Export aggregate experiment facts only; never publish raw source/generations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from build_stage_data import _message


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    args = p.parse_args()
    work = Path(args.work_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda f: json.loads((work / f).read_text())
    blocks = lambda r: {k: r[k] for k in ["overall", "free_text", "structured_icd"]}
    audit = read("legacy_audit_with_tokens.json")
    variants = {
        name: blocks(read(path))
        for name, path in {
            "old_single_json": "old_metrics.json",
            "v4_generation_512": "baseline_metrics.json",
            "classifier_global": "eval_predicted/metrics.json",
            "classifier_type_thresholds": "eval_labelwise/metrics.json",
            "classifier_oracle_global": "eval_oracle/metrics.json",
            "controlled_problem_entities": "eval_problem_same_entities/metrics.json",
            "classifier_global_rules_posthoc": "review_global/rules_metrics.json",
            "classifier_type_thresholds_rules_posthoc": "review_labelwise/rules_metrics.json",
        }.items()
    }
    corruption = {}
    for name, path in [("train", args.train), ("dev", args.dev)]:
        count = 0
        tokens = Counter()
        for r in json.loads(Path(path).read_text()):
            found = [
                w.casefold()
                for w in re.findall(r"\b[A-Za-z]+null[A-Za-z]*\b", _message(r, "user"))
                if w.casefold()
                not in {"annul", "annuls", "annulled", "annulling", "annulment"}
            ]
            tokens.update(found)
            count += bool(found)
        corruption[name] = {"records": count, "tokens": dict(tokens)}
    run = read("training/run_manifest.json")
    selection = read("training/selection.json")
    calibration = read("training/selection_labelwise.json")
    private_args = {"base_model", "data_dir", "output_dir"}
    config = {k: v for k, v in run["args"].items() if k not in private_args}
    summary = {
        "scope": "Historical regression set, silver labels; fixed cached entity comparison, not a fresh full-pipeline blind test.",
        "model": "Qwen3-8B LoRA sequence classifier",
        "base_snapshot": "b968826d9c46dd6066d109eabc6255188de91218",
        "config": config,
        "versions": run["versions"],
        "selected_epoch": selection["epoch"],
        "global_threshold": selection["threshold"],
        "label_thresholds": calibration.get("label_thresholds", {}),
        "calibration": calibration["calibration"],
        "data_sha256": {
            "train": run["data_manifest"]["train_sha256"],
            "dev": run["data_manifest"]["dev_sha256"],
            **run["data_files_sha256"],
        },
        "data_statistics": run["data_manifest"]["stats"],
        "train_label_counts": run["data_manifest"]["train_label_counts"],
        "train_group_count": len(run["data_manifest"]["train_groups"]),
        "tune_group_count": len(run["data_manifest"]["tune_groups"]),
        "legacy_audit": {
            split: {k: v for k, v in audit[split].items() if k != "fallback_examples"}
            for split in ["train", "dev"]
        },
        "legacy_tokenization": audit["stage"],
        "known_source_corruption": corruption,
        "variants": variants,
        "candidate_ceiling": {
            key: read(path)["run"]["candidate_ceiling"]
            for key, path in [
                ("predicted", "eval_predicted/metrics.json"),
                ("oracle", "eval_oracle/metrics.json"),
            ]
        },
        "global_error_summary": read("review_global/review_summary.json"),
        "type_threshold_error_summary": read("review_labelwise/review_summary.json"),
        "synthetic_challenges": read("challenges.json"),
        "orthography_diagnostic": {
            "scope": "Post-hoc diagnostic, not the primary evaluation or model selection criterion",
            "free_text": read("identity_diagnostic.json")["orthography_only"][
                "free_text"
            ],
        },
        "controlled_problem_sample_2": read("eval_problem_same_entities/metrics.json")[
            "samples"
        ][2],
        "artifact_hashes": {
            path: hashlib.sha256((work / path).read_bytes()).hexdigest()
            for path in [
                "eval_predicted/predictions.jsonl",
                "eval_oracle/predictions.jsonl",
                "eval_labelwise/predictions.jsonl",
                "eval_problem_same_entities/predictions.jsonl",
                "training/training_history.json",
            ]
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (out / "training_history.json").write_text(
        json.dumps(read("training/training_history.json"), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "variants": {
                    k: v["free_text"]["relation"] for k, v in variants.items()
                },
                "source_corruption": corruption,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

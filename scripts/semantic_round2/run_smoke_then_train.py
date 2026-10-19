"""Run real GPU validation and then a fresh DPO run from the original SFT.

The smoke adapter is never used as the formal parent or accepted for evaluation.
The queue success receipt belongs only to the completed formal training run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    trainer = Path(__file__).with_name("semantic_dpo.py")
    common = config["common_argv"]
    assert "--output-dir" not in common and "--smoke-only" not in common
    pairs = Path(common[common.index("--candidates-root") + 1]) / "candidate_train.jsonl"
    content = pairs.read_bytes()
    assert hashlib.sha256(content).hexdigest() == config["expected_train_sha256"]
    assert len([line for line in content.splitlines() if line.strip()]) == config["expected_pairs"]
    for stage, output, extra in (("smoke", config["smoke_output"], ["--smoke-only"]),
                                  ("formal", config["formal_output"], [])):
        print(json.dumps({"stage": stage, "status": "starting", "parent": "unchanged_original_sft"}), flush=True)
        subprocess.run([sys.executable, str(trainer), "train", *common,
                        "--output-dir", output, *extra, "--execute"], check=True)
        receipt = json.loads((Path(output) / "training_manifest.json").read_text())
        assert receipt["completed"] is True and not receipt.get("failed")
        assert receipt["smoke_only"] is (stage == "smoke")
        assert receipt["reference_unchanged"] is True and receipt["policy_weights_changed"] is True
        assert receipt["saved_weights_bitwise_verified"] is True and receipt["saved_reload_exact"] is True
        if stage == "smoke":
            assert receipt["admission"]["smoke_subset_count"] == 4
        else:
            seen = receipt["observed_exposure"]
            assert seen["unique_pairs_seen"] == seen["pairs"] == config["expected_pairs"]
            assert seen["min_visits"] == seen["max_visits"] == 1
            assert receipt["global_step"] == config["expected_pairs"] // 4
            assert receipt["current_runtime_contract_check_passed"] is True
        print(json.dumps({"stage": stage, "status": "complete", "updates": receipt["global_step"],
                          "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()

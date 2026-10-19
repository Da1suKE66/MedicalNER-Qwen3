import copy
import json

import pytest

from data_repair.launch_release import build, read_jsonl, validate_frozen_inputs, validate_matched_arms
from data_repair.release_v2 import canonical, run, sha
from test_release_v2 import disease, free, message


def release_fixture(tmp_path):
    names = ("Feature Alpha", "Feature Beta", "Feature Gamma")
    sentences = [f"{name} is a core symptom of Condition A." for name in names]
    graph = {"entities": [{"id": "D", "name": "Condition A", "label": "Disease"}]
             + [{"id": str(i), "name": name, "label": "Symptom"} for i, name in enumerate(names)],
             "relations": [{"source": str(i), "target": "D", "relation": "is_core_symptom_of", "evidence": text}
                           for i, text in enumerate(sentences)]}
    records = [disease("A"), message(" ".join(sentences), graph), disease("B"), free("B")]
    source, split = tmp_path / "source.json", tmp_path / "split.json"
    source.write_text(canonical(records))
    split.write_text(canonical({"source_train_sha256": sha(source.read_bytes()), "train_groups": [0], "tune_groups": [1]}))
    release = tmp_path / "source_release"
    run(source, split, release, sha(source.read_bytes()))
    return release


def test_dense_launch_freezes_inputs_matches_arms_and_emits_only_aggregate_manifest(tmp_path):
    release = release_fixture(tmp_path)
    originals = {str(p.relative_to(release)): p.read_bytes() for p in release.rglob("*") if p.is_file()}
    out = tmp_path / "launch"
    report = build(release, out, max_cases=2)
    assert report["protocol"] == "dense_fixed_pair_diagnostic_not_sparse_positive"
    assert report["synthetic_cases_per_arm"] == 2
    assert report["grounded_negative_decisions_unmasked"] > 0
    assert report["source_group_counts"] == {"train": 1, "tune": 1}
    assert not report["evaluation_gold_modified"] and not report["tokenizer_checked"]
    assert "Feature Alpha" not in json.dumps(report) and str(tmp_path) not in json.dumps(report)
    for name, digest in report["output_sha256"].items():
        assert sha((out / name).read_bytes()) == digest
    control = read_jsonl(out / "V4_missing_only.jsonl")
    treatment = read_jsonl(out / "V4_grounded_empty.jsonl")
    base = read_jsonl(out / "base_unique_train.jsonl")
    assert len(control) == len(treatment) == len(base) + 2
    assert validate_matched_arms(base, {"V4_missing_only": control, "V4_grounded_empty": treatment}, {0}, {1}) > 0
    assert {str(p.relative_to(release)): p.read_bytes() for p in release.rglob("*") if p.is_file()} == originals
    with pytest.raises(ValueError, match="overwrite"):
        build(release, out, max_cases=2)


def test_source_hash_mismatch_refused_before_output(tmp_path):
    release = release_fixture(tmp_path)
    with (release / "frozen_split.jsonl").open("a") as handle:
        handle.write("\n")
    out = tmp_path / "launch"
    with pytest.raises(ValueError, match="hash mismatch"):
        build(release, out, max_cases=2)
    assert not out.exists()


def test_insufficient_case_budget_and_source_directory_output_refused(tmp_path):
    release = release_fixture(tmp_path)
    with pytest.raises(ValueError, match="Insufficient"):
        build(release, tmp_path / "too-many", max_cases=100)
    assert not (tmp_path / "too-many").exists()
    with pytest.raises(ValueError, match="outside"):
        build(release, release / "must-not-write", max_cases=1)
    assert not (release / "must-not-write").exists()


def test_frozen_group_identity_drift_and_tampered_arm_rejected(tmp_path):
    release = release_fixture(tmp_path)
    prepared = read_jsonl(release / "records.jsonl")
    assignments = read_jsonl(release / "frozen_split.jsonl")
    base = read_jsonl(release / "base/train.jsonl")
    tune = read_jsonl(release / "base/tune.jsonl")
    changed = copy.deepcopy(prepared)
    changed[0]["group_id"] = 99
    with pytest.raises(ValueError, match="identity"):
        validate_frozen_inputs(changed, assignments, base, tune)
    out = tmp_path / "launch"
    build(release, out, max_cases=2)
    arms = {name: read_jsonl(out / (name + ".jsonl")) for name in ("V4_missing_only", "V4_grounded_empty")}
    arms["V4_grounded_empty"][-1]["prompt"] += "different"
    with pytest.raises(ValueError, match="differ"):
        validate_matched_arms(read_jsonl(out / "base_unique_train.jsonl"), arms, {0}, {1})

"""Synthetic CPU-only noise contracts. No private clinical fixtures are embedded."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative.protocol import ProtocolConfig, build_batches, dumps, parse_decisions, supervision
from kg_agent.contracts import allowed_relations
from robustness.entity_noise import (
    NoiseConfig, NoiseError, NoEligibleMutation, augment_batch, augment_row, augment_heldout_row,
    audit_augmentation, fingerprint, prompt_payload, protocol_prefix, validate_row,
)
from robustness.build_noisy_data import (
    build_versions, build_heldout_version, exposure, main, noisy_count, plan_noise, sha256, validate_splits,
)


def sample(index=0, *, split="train", two_entities=False):
    text = (f"Synthetic case {index}. Sign Alpha supports diagnosis of Condition Alpha. "
            "Sign Beta supports diagnosis of Condition Beta.")
    nodes = [
        {"id": "E004", "name": "Sign Alpha", "label": "Symptom"},
        {"id": "P0001", "name": "Condition Alpha", "label": "Disease"},
        {"id": "E001", "name": "Sign Beta", "label": "Symptom"},
        {"id": "upstream-D", "name": "Condition Beta", "label": "Disease"},
    ]
    if two_entities:
        nodes = nodes[:2]
    graph = {"entities": nodes, "relations": [
        {"source": nodes[i]["id"], "target": nodes[i + 1]["id"],
         "relation": "supports_diagnosis_of", "evidence": text.split(". ")[1 + i // 2]}
        for i in range(0, len(nodes), 2)
    ]}
    # Multiple relations/evidence must survive on a nonincident pair.
    graph["relations"].append({**graph["relations"][0], "relation": "is_core_symptom_of"})
    config = ProtocolConfig(strict_evidence=True, order="source")
    batches, doc, _ = build_batches(text, nodes, config, sample_id=index)
    target, _ = supervision(batches[0], graph, doc, config)
    assert target is not None
    return {**batches[0], "target": target, "completion": dumps(target),
            "group_id": index, "split": split, "sample_weight": 1.0,
            "annotation_source": "synthetic_fixture_not_medical_gold"}


def refresh_prompt(row):
    row["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(row)) + "\n"


def write_source(root, *, count=16):
    root.mkdir()
    train, tune = [sample(i) for i in range(count)], [sample(10000, split="tune")]
    # Deliberately noncanonical outer JSON spacing: retain exact original bytes.
    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n").encode()
    tuning = (dumps(tune[0]) + "\n").encode()
    (root / "train.jsonl").write_bytes(data)
    (root / "tune.jsonl").write_bytes(tuning)
    manifest = {"variant": "synthetic", "config": {"context": "full"},
                "dataset_sha256": {"train": sha256(data), "tune": sha256(tuning)},
                "train_groups": list(range(count)), "tune_groups": [10000]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return train, tune, data, tuning


@pytest.mark.parametrize("operation", ["replace", "inject"])
@pytest.mark.parametrize("type_mode", ["keep", "change", "mixed"])
def test_clean_unchanged_and_nonincident_positives_evidence_conserved(operation, type_mode):
    clean = sample()
    original = deepcopy(clean)
    noisy = augment_row(clean, NoiseConfig(operation=operation, type_mode=type_mode))
    assert clean == original
    assert clean["prompt"] == original["prompt"]
    assert noisy["id"] != clean["id"]
    assert noisy["group_id"] == clean["group_id"]
    audit = audit_augmentation(clean, noisy)
    assert audit["retained_nonincident_positive_edges"] > 0
    assert audit["noise_entities"] == 1
    assert parse_decisions(noisy["completion"], noisy)[1] == []
    by_pid = {d["pair_id"]: d for d in noisy["target"]["decisions"]}
    for pid in noisy["entity_noise"]["incident_pair_ids"]:
        assert by_pid[pid]["relations"] == []
    assert noisy["spans"] == clean["spans"]
    assert noisy["original_spans"] == clean["original_spans"]


def test_both_source_and_target_use_upstream_mapping_not_local_id():
    clean = sample()
    noisy = augment_row(clean, NoiseConfig(type_mode="change"))
    mapping = noisy["entity_noise"]["id_mapping"]
    assert len(mapping) == 1
    old, new = next(iter(mapping.items()))
    assert old not in {e["id"] for e in noisy["entities"]}
    source_updates = target_updates = 0
    for before, after in zip(clean["pairs"], noisy["pairs"]):
        assert before["pair_id"] == after["pair_id"]
        for end in ("source", "target"):
            assert after[end] == mapping.get(before[end], before[end])
        source_updates += before["source"] == old
        target_updates += before["target"] == old
    assert source_updates and target_updates
    assert new != old


def test_type_corruption_updates_allowed_lists_in_both_prompt_and_pairs():
    clean = sample()
    noisy = augment_row(clean, NoiseConfig(type_mode="change"))
    assert noisy["entity_noise"]["old_label"] != noisy["entity_noise"]["new_label"]
    by_id = {e["id"]: e for e in noisy["entities"]}
    payload = json.loads(noisy["prompt"].rsplit("\n", 2)[1])
    changed_allowed = 0
    for before, pair, visible in zip(clean["pairs"], noisy["pairs"], payload["pairs"]):
        expected = list(allowed_relations(by_id[pair["source"]]["label"], by_id[pair["target"]]["label"]))
        assert pair["allowed"] == visible["allowed_relations"] == expected
        changed_allowed += pair["allowed"] != before["allowed"]
    assert changed_allowed


def test_error_entity_really_reaches_prompt_without_fake_grounding():
    clean = sample()
    noisy = augment_row(clean)
    bad = noisy["entity_noise"]["changed_entity_ids"][0]
    entity = next(e for e in noisy["entities"] if e["id"] == bad)
    payload = json.loads(noisy["prompt"].rsplit("\n", 2)[1])
    assert any(e["name"] == entity["name"] for e in payload["entities"])
    assert all(entity["name"].casefold() not in s["text"].casefold() for s in noisy["spans"])
    raw = " ".join(s["text"] for s in noisy["spans"])
    _, _, audit = build_batches(raw, noisy["entities"], ProtocolConfig())
    assert bad in audit["ungrounded_ids"]  # Demonstrates why we do NOT rebuild through grounding.


def test_metadata_quality_and_original_graph_ids_never_leak_to_prompt():
    clean = sample()
    clean["provenance"] = {"noise_origin": "TOP_SECRET_METADATA", "gold_quality": "GOLD_SECRET"}
    clean["entities"][0]["is_noise"] = "ENTITY_METADATA_SECRET"
    noisy = augment_row(clean)
    assert noisy["provenance"] == clean["provenance"]
    for forbidden in ("TOP_SECRET_METADATA", "GOLD_SECRET", "ENTITY_METADATA_SECRET",
                      "entity_noise", "is_noise", "annotation_source", "clinical_quality",
                      "PENDING_REVIEW", "source_row_sha256", "absent_synthetic_name"):
        assert forbidden not in noisy["prompt"]
    assert noisy["entity_noise"]["changed_entity_ids"][0] not in noisy["prompt"]


def test_injection_keeps_every_original_pair_and_handles_sparse_ids():
    clean = sample()
    for i, entity in enumerate(clean["entities"]):
        entity["entity_id"] = f"E{100 + i * 10:03d}"
    for i, (pair, decision) in enumerate(zip(clean["pairs"], clean["target"]["decisions"])):
        pair["pair_id"] = decision["pair_id"] = f"P{1000 + i * 10:04d}"
    clean["completion"] = dumps(clean["target"])
    refresh_prompt(clean)
    noisy = augment_row(clean, NoiseConfig(operation="inject", injected_pairs=3))
    assert noisy["pairs"][:len(clean["pairs"])] == clean["pairs"]
    assert noisy["target"]["decisions"][:len(clean["pairs"])] == clean["target"]["decisions"]
    assert len(noisy["pairs"]) == len(clean["pairs"]) + 3
    assert noisy["entities"][-1]["entity_id"] == "E131"
    assert noisy["entity_noise"]["id_mapping"] == {}


def test_replacement_cannot_turn_positive_batch_all_empty_but_injection_can_preserve_it():
    clean = sample(two_entities=True)
    with pytest.raises(NoEligibleMutation):
        augment_row(clean)
    noisy = augment_row(clean, NoiseConfig(operation="inject"))
    assert any(d["relations"] for d in noisy["target"]["decisions"])
    assert audit_augmentation(clean, noisy)["removed_incident_positive_edges"] == 0


def test_inherited_empty_target_is_not_reinterpreted_as_unknown():
    clean = sample()
    for d in clean["target"]["decisions"]:
        d["relations"] = []
    clean["completion"] = dumps(clean["target"])
    noisy = augment_row(clean)
    assert all(d["relations"] == [] for d in noisy["target"]["decisions"])
    assert "not_reclassified" in noisy["entity_noise"]["inherited_nonincident_empty"]


@pytest.mark.parametrize("split", ["tune", "dev", "test", "evaluation", None])
def test_training_only_split_required(split):
    clean = sample()
    clean["split"] = split
    with pytest.raises(NoiseError):
        augment_row(clean)


@pytest.mark.parametrize("corruption", ["unknown", "null_relations", "missing_decision",
                                       "stale_allowed", "bad_evidence", "duplicate_id",
                                       "duplicate_pair", "leaked_payload", "leaked_prefix"])
def test_invalid_or_unknown_contracts_fail_closed(corruption):
    clean = sample()
    if corruption == "unknown":
        clean["target"] = None
    elif corruption == "null_relations":
        clean["target"]["decisions"][0]["relations"] = None
    elif corruption == "missing_decision":
        clean["target"]["decisions"].pop()
    elif corruption == "stale_allowed":
        clean["pairs"][0]["allowed"] = []
        refresh_prompt(clean)
    elif corruption == "bad_evidence":
        next(d for d in clean["target"]["decisions"] if d["relations"])["relations"][0]["evidence_span_ids"] = ["SENT_999999"]
    elif corruption == "duplicate_id":
        clean["entities"][1]["id"] = clean["entities"][0]["id"]
    elif corruption == "duplicate_pair":
        clean["pairs"].append(deepcopy(clean["pairs"][0]))
    elif corruption == "leaked_payload":
        payload = prompt_payload(clean)
        payload["is_noise"] = True
        clean["prompt"] = protocol_prefix() + "\n" + dumps(payload) + "\n"
    elif corruption == "leaked_prefix":
        clean["prompt"] = "gold_quality=invalid " + clean["prompt"]
    clean["completion"] = dumps(clean["target"])
    with pytest.raises(NoiseError):
        augment_row(clean)


def test_recursive_augmentation_is_rejected():
    with pytest.raises(NoiseError, match="Recursive"):
        augment_row(augment_row(sample()))


def test_type_only_and_realistic_unreviewed_negative_modes_are_not_supported():
    for mode in ("type_only", "realistic", "predicted_difference", "synonym"):
        with pytest.raises(NoiseError, match="review"):
            NoiseConfig(type_mode=mode)


def test_direct_supervised_protocol_batch_adapter_and_unknown():
    row = sample()
    batch = {k: row[k] for k in ("id", "sample_id", "prompt", "entities", "pairs", "spans", "original_spans")}
    assert augment_batch(batch, row["target"], split="train", group_id=0)["entity_noise"]
    with pytest.raises(NoiseError):
        augment_batch(batch, None, split="train", group_id=0)


def test_seed_reproducibility_nested_rates_and_order_independence():
    rows = [sample(i) for i in range(30)]
    small, _ = plan_noise(rows, 3, NoiseConfig(seed=73))
    large, _ = plan_noise(list(reversed(rows)), 10, NoiseConfig(seed=73))
    assert small == large[:3]
    assert augment_row(rows[0], NoiseConfig(seed=17)) == augment_row(rows[0], NoiseConfig(seed=17))
    assert augment_row(rows[0], NoiseConfig(seed=17))["prompt"] != augment_row(rows[0], NoiseConfig(seed=73))["prompt"]
    assert len({r["entity_noise"]["source_row_id"] for r in large}) == 10


@pytest.mark.parametrize("n,rate,expected", [(1475, .1, 164), (1475, .25, 492),
                                            (4971, .1, 552), (4971, .25, 1657), (9, .1, 1), (16, 0, 0)])
def test_rate_denominator_and_rounding(n, rate, expected):
    assert noisy_count(n, rate) == expected
    assert abs(expected / (n + expected) - rate) <= 1 / (n + expected)


@pytest.mark.parametrize("rate", [-.1, .75, float("nan"), float("inf")])
def test_invalid_noise_rates(rate):
    with pytest.raises(NoiseError):
        noisy_count(100, rate)


def test_clean_duplicates_are_not_silently_deleted_but_no_duplicate_augmented_views():
    a, b = sample(), sample(1)
    duplicate = deepcopy(a)
    duplicate["id"] = "duplicate-view"
    rows = [a, duplicate, b]
    assert exposure(rows)["duplicate_training_views"] == 1
    pool, _ = plan_noise(rows, 2, NoiseConfig())
    assert len({r["entity_noise"]["source_training_view_sha256"] for r in pool}) == 2
    with pytest.raises(NoiseError, match="Insufficient"):
        plan_noise(rows, 3, NoiseConfig())


def test_split_group_sample_id_prompt_and_duplicate_row_checks():
    train, tune = [sample()], [sample(100, split="tune")]
    validate_splits(train, tune)
    for key in ("group_id", "sample_id", "id", "prompt"):
        leaked = deepcopy(tune)
        leaked[0][key] = train[0][key]
        if key == "prompt":
            leaked[0] = {**deepcopy(train[0]), "id": "other", "sample_id": 100,
                         "group_id": 100, "split": "tune"}
        with pytest.raises(NoiseError):
            validate_splits(train, leaked)
    with pytest.raises(NoiseError, match="Duplicate"):
        validate_splits(train * 2, tune)


def test_noisy_sampling_weight_does_not_inherit_rare_positive_oversampling():
    clean = sample()
    clean["sample_weight"] = 4
    noisy = augment_row(clean)
    assert clean["sample_weight"] == 4 and noisy["sample_weight"] == 1
    counts = exposure([clean, noisy])
    assert counts["noisy_row_fraction"] == .5
    assert counts["weighted_noisy_row_fraction"] == .2


def test_versions_private_immutable_hashes_clean_bytes_tune_and_aggregate_report(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    train, tune, original, tuning = write_source(source)
    report = build_versions(source, output)
    assert (source / "train.jsonl").read_bytes() == original
    assert (output / "clean/train.jsonl").read_bytes() == original
    hashes = json.loads((output / "SHA256SUMS.json").read_text())
    for path, digest in hashes["outputs"].items():
        assert sha256((output / path).read_bytes()) == digest
    small_parents = set()
    for name, manifest in report["variants"].items():
        data = (output / name / "train.jsonl").read_bytes()
        assert data.startswith(original)
        assert (output / name / "tune.jsonl").read_bytes() == tuning
        assert sha256(data) == manifest["dataset_sha256"]["train"]
        rows = [json.loads(line) for line in data.splitlines()]
        assert rows[:len(train)] == train
        assert all(r["split"] == "train" for r in rows)
        parents = {r["entity_noise"]["source_row_id"] for r in rows[len(train):]}
        assert len(parents) == manifest["appended_noisy_rows"]
        assert small_parents <= parents
        small_parents = parents
        assert manifest["token_audit"]["status"] == "NOT_RUN"
        assert (output / name / "train.jsonl").stat().st_mode & 0o777 == 0o600
    aggregate = (output / "generation_report.json").read_text()
    for private in ("Sign Alpha", "Condition Alpha", train[0]["prompt"], "upstream-D"):
        assert private not in aggregate
    assert "PENDING_REVIEW_NOT_TRAINED" in aggregate
    assert "NOT_MODEL_TRAINED" in report["status"]
    with pytest.raises(NoiseError, match="new path"):
        build_versions(source, output)


def test_byte_determinism_of_independent_builds(tmp_path):
    source = tmp_path / "input"
    write_source(source, count=8)
    a, b = tmp_path / "a", tmp_path / "b"
    assert build_versions(source, a) == build_versions(source, b)
    for path in a.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (b / path.relative_to(a)).read_bytes()


def test_source_hash_mismatch_and_repo_output_are_rejected(tmp_path):
    source = tmp_path / "input"
    write_source(source)
    (source / "train.jsonl").write_bytes((source / "train.jsonl").read_bytes() + b"\n")
    with pytest.raises(NoiseError):
        build_versions(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    with pytest.raises(NoiseError, match="outside"):
        build_versions(source, Path(__file__).resolve().parent / "private_noise_must_not_exist")


def test_cli_stdout_contains_only_aggregates(tmp_path, capsys):
    source, output = tmp_path / "input", tmp_path / "output"
    write_source(source, count=8)
    main(["--source-dir", str(source), "--output-dir", str(output), "--rates", "0", "0.25"])
    captured = capsys.readouterr().out
    assert json.loads(captured)["variant"] == "synthetic"
    assert "Sign Alpha" not in captured


def test_positive_density_guard_fails_before_writing(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    write_source(source)
    with pytest.raises(NoiseError, match="dilutes"):
        build_versions(source, output, rates=[.25], min_positive_density_retention=1)
    assert not output.exists()


@pytest.mark.parametrize("split", ["tune", "test"])
def test_heldout_explicit_split_readonly_target_and_train_rejection(split):
    clean = sample(split=split)
    before = deepcopy(clean)
    stress = augment_heldout_row(clean, split=split)
    assert clean == before
    assert stress["split"] == split
    assert stress["stress_evaluation"]["usage"] == "heldout_stress_only_not_training"
    assert audit_augmentation(clean, stress, expected_split=split)["noise_entities"] == 1
    assert "stress_evaluation" not in stress["prompt"]
    for action in (lambda: validate_row(stress), lambda: augment_row(stress),
                   lambda: audit_augmentation(clean, stress)):
        with pytest.raises(NoiseError):
            action()
    accidentally_retagged = {**stress, "split": "train"}
    with pytest.raises(NoiseError, match="Held-out"):
        validate_row(accidentally_retagged)


def test_heldout_wrapper_requires_exact_explicit_tune_or_test():
    clean = sample(split="tune")
    with pytest.raises(TypeError):
        augment_heldout_row(clean)
    for split in ("test", "train", "dev", None):
        with pytest.raises(NoiseError):
            augment_heldout_row(clean, split=split)


def test_nonce_namespace_independent_across_train_tune_test_with_identical_seed_and_id():
    config = NoiseConfig(seed=42)
    training = sample()
    rows = [augment_row(training, config)]
    for split in ("tune", "test"):
        heldout = {**deepcopy(training), "split": split}
        a = augment_heldout_row(heldout, split=split, config=config)
        assert a == augment_heldout_row(heldout, split=split, config=config)
        rows.append(a)
    names = []
    for row in rows:
        changed = set(row["entity_noise"]["changed_entity_ids"])
        names.extend(e["name"] for e in row["entities"] if e["id"] in changed)
    assert len(set(names)) == 3


def test_existing_training_entrypoint_rejects_heldout_before_model_loading(tmp_path, monkeypatch):
    from generative.train import main as train_main
    path = tmp_path / "heldout.jsonl"
    row = augment_heldout_row(sample(split="test"), split="test")
    path.write_text(dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", ["train", "--mode", "sft", "--data", str(path),
                                     "--base-model", "must-not-load", "--output-dir", str(tmp_path / "no-checkpoint")])
    with pytest.raises(ValueError, match="train"):
        train_main()
    assert not (tmp_path / "no-checkpoint").exists()


def test_shared_heldout_generation_hashes_pairing_separation_and_target_independent_selection(tmp_path):
    source = tmp_path / "input"
    train, _, _, _ = write_source(source)
    tune = [sample(10000 + i, split="tune", two_entities=(i == 0)) for i in range(4)]
    tuning = ("".join(dumps(r) + "\n" for r in tune)).encode()
    (source / "tune.jsonl").write_bytes(tuning)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["tune_groups"] = [r["group_id"] for r in tune]
    manifest["dataset_sha256"]["tune"] = sha256(tuning)
    (source / "manifest.json").write_text(json.dumps(manifest))
    train_noise = tmp_path / "training_noise.jsonl"
    train_noise.write_text("".join(dumps(augment_row(r)) + "\n" for r in train))
    source_hashes = {p.name: sha256(p.read_bytes()) for p in source.iterdir()}
    output = tmp_path / "stress"
    report = build_heldout_version(source, output, count=4, training_noise_files=[train_noise])
    assert {p.name: sha256(p.read_bytes()) for p in source.iterdir()} == source_hashes
    assert report["paired_rows"] == 4 and report["exact_nonce_overlap_with_supplied_training_noise"] == 0
    assert report["replacement_to_injection_fallbacks_without_reselection"] == 1
    assert not (output / "train.jsonl").exists()
    clean = [json.loads(s) for s in (output / "clean.jsonl").read_text().splitlines()]
    noise = [json.loads(s) for s in (output / "noisy.jsonl").read_text().splitlines()]
    pairs = [json.loads(s) for s in (output / "pairing.jsonl").read_text().splitlines()]
    for a, b, pair in zip(clean, noise, pairs):
        assert a["split"] == b["split"] == "tune"
        assert a in tune
        assert pair["clean_id"] == a["id"] and pair["noisy_id"] == b["id"]
        assert pair["clean_sha256"] == fingerprint(a)
        with pytest.raises(NoiseError):
            validate_row(b)
    hashes = json.loads((output / "SHA256SUMS.json").read_text())
    assert all(sha256((output / p).read_bytes()) == v for p, v in hashes["outputs"].items())
    aggregate = json.dumps(report)
    assert "Sign Alpha" not in aggregate and "Condition Alpha" not in aggregate
    # Change all reference labels without changing any input/ID; selection and
    # ordering MUST remain identical, even though the constructed stress differs.
    for row in tune:
        for decision in row["target"]["decisions"]:
            decision["relations"] = []
        row["completion"] = dumps(row["target"])
    tuning = "".join(dumps(r) + "\n" for r in tune).encode()
    (source / "tune.jsonl").write_bytes(tuning)
    manifest["dataset_sha256"]["tune"] = sha256(tuning)
    (source / "manifest.json").write_text(json.dumps(manifest))
    output2 = tmp_path / "stress_other_targets"
    build_heldout_version(source, output2, count=4)
    pairs2 = [json.loads(s) for s in (output2 / "pairing.jsonl").read_text().splitlines()]
    assert [p["clean_id"] for p in pairs] == [p["clean_id"] for p in pairs2]


def test_shared_heldout_cli_is_separate_and_immutable(tmp_path, capsys):
    source, output = tmp_path / "input", tmp_path / "heldout"
    write_source(source)
    main(["--source-dir", str(source), "--output-dir", str(output), "--heldout", "--heldout-count", "1"])
    assert json.loads(capsys.readouterr().out)["paired_rows"] == 1
    assert not (output / "train.jsonl").exists()
    with pytest.raises(NoiseError):
        build_heldout_version(source, output, count=1)

"""Offline audit tests use synthetic records only; no actual medical fixtures."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generative.build_data import convert
from generative.protocol import ProtocolConfig, build_batches, dumps, supervision
from kg_agent.source import SourceRouter
from robustness.data_audit import (
    Findings, audit_raw, audit_row, audit_saved, audit_source_overlap, audit_splits, canonical_sha,
    default_inputs, evidence_check, expected_original_spans, file_sha, main,
    overlap_audit, reproduce_bugs, run_audit, sha, synthetic_loss_mask_check,
    synthetic_record, trace_record, write_report,
)


TEXT = "Fatigue accompanies Condition Alpha during pregnullcy."
NODES = [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
         {"id": "D1", "label": "Disease", "name": "Condition Alpha"}]
EDGE = {"source": "S1", "target": "D1", "relation": "is_associated_symptom_of", "evidence": TEXT}
GRAPH = {"entities": NODES, "relations": [EDGE]}


@pytest.fixture
def fixture_inputs(tmp_path):
    data = tmp_path / "data"
    root = data / "Gsynthetic"
    root.mkdir(parents=True)
    config = ProtocolConfig(repair=True, strict_evidence=True)
    group_order = list(range(10))
    random.Random(42).shuffle(group_order)
    tune = {group_order[0]}
    records, groups, splits = [], [], []
    for group in range(10):
        records.extend([
            synthetic_record(dumps({"code": f"A{group:02d}", "title": f"SYNTHETIC_GROUP_{group}"}), {"entities": [], "relations": []}),
            synthetic_record(TEXT + f" Synthetic group {group}.", GRAPH),
        ])
        groups.extend([group, group])
        splits.extend(["tune" if group in tune else "train"] * 2)
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(records), encoding="utf-8")
    subsets = {"train": [], "tune": []}
    from collections import Counter
    stats = {"train": Counter(), "tune": Counter()}
    for i, (record, group, split) in enumerate(zip(records, groups, splits)):
        rows, counts = convert(record, i, group, config, sampled=split == "train", seed=42)
        stats[split].update(counts)
        for row in rows:
            row["split"] = split
        subsets[split].extend(rows)
    for split, rows in subsets.items():
        (root / f"{split}.jsonl").write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"variant": "Gsynthetic", "config": asdict(config), "seed": 42,
                "source_train_sha256": file_sha(source), "train_groups": sorted(set(groups) - tune), "tune_groups": sorted(tune),
                "dataset_sha256": {s: file_sha(root / f"{s}.jsonl") for s in subsets},
                "stats": {s: dict(c) for s, c in stats.items()}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (data / "tune_records.json").write_text(json.dumps([r for r, s in zip(records, splits) if s == "tune"]), encoding="utf-8")
    return {"raw": source, "data_dir": data, "root": root, "records": records, "groups": groups, "splits": splits,
            "config": config, "manifest": manifest, "subsets": subsets}


def row_context():
    config = ProtocolConfig(repair=True, strict_evidence=True)
    record = synthetic_record(TEXT, GRAPH)
    rows, _ = convert(record, 0, 0, config, sampled=False)
    row = {**rows[0], "split": "train"}
    return row, [SourceRouter().route(TEXT)], [GRAPH], [0], ["train"], config


def check_row(row, *context):
    issues = Findings()
    counts = audit_row(row, 1, *context, issues, {})
    return counts, issues.result()["counts"]


def test_all_evidence_fragments_required_by_independent_check():
    doc = SourceRouter().route(TEXT)
    result = evidence_check({**EDGE, "evidence": [TEXT, "ABSENT_SYNTHETIC_987"]}, doc)
    assert result["fragments"] == 2 and result["any_found"]
    assert result["partial_found"] and not result["all_found"]


@pytest.mark.parametrize("value", [[], "", [TEXT, ""], [TEXT, None], [TEXT, {"bad": True}], {"not": "text"}])
def test_malformed_evidence_cannot_be_complete(value):
    assert not evidence_check({**EDGE, "evidence": value}, SourceRouter().route(TEXT))["all_found"]


def test_span_ids_and_visibility_are_separate_from_full_document_location():
    doc = SourceRouter().route("Fatigue is present. Another synthetic sentence follows.")
    result = evidence_check({"evidence": doc.spans[1].id}, doc, visible={doc.spans[0].id})
    assert result["all_found"] and not result["all_visible"]


def test_raw_graph_integrity_keeps_indices_not_names():
    graph = copy.deepcopy(GRAPH)
    graph["entities"].append({"id": "S1", "label": "PRIVATE_UNKNOWN_LABEL", "name": "PRIVATE_UNGROUNDED_NAME"})
    graph["relations"].append({**EDGE, "target": "NO_SUCH_ID", "evidence": "PRIVATE_MISSING_EVIDENCE"})
    report, _, _ = audit_raw([synthetic_record(TEXT, graph)])
    issues = report["issues"]["counts"]
    assert issues["duplicate_entity_id"] == 1
    assert issues["unknown_entity_type"] == 1
    assert issues["dangling_relation_reference"] == 1
    assert "PRIVATE_" not in json.dumps(report)
    assert report["issues"]["examples"]["duplicate_entity_id"] == [{"source_index": 0, "entity_index": 2}]


def test_raw_multilabel_and_duplicate_annotations_are_distinct():
    graph = {"entities": NODES, "relations": [EDGE, copy.deepcopy(EDGE), {**EDGE, "relation": "supports_diagnosis_of"}]}
    report, _, _ = audit_raw([synthetic_record(TEXT, graph)])
    c = report["by_source_type"]["free_text"]
    assert (c["relations"], c["unique_relation_labels"], c["duplicate_relation_annotations"], c["multi_relation_pairs"]) == (3, 2, 1, 1)


def test_explicit_output_wrapper_is_not_reported_as_corrupt_graph():
    record = synthetic_record(TEXT, GRAPH)
    record["messages"][-1]["content"] = "<output>\n" + dumps(GRAPH) + "\n</output>"
    report, _, _ = audit_raw([record])
    assert report["counts"]["assistant_strict_json"] == 0
    assert report["counts"]["assistant_output_wrapper_json"] == 1
    assert report["counts"]["assistant_graphs_valid"] == 1
    assert not report["issues"]["counts"]


def test_distinct_finding_sources_not_limited_by_example_cap():
    issues = Findings(1)
    for index in (0, 0, 1, 2):
        issues.add("known_issue", source_index=index)
    result = issues.result()
    assert result["counts"]["known_issue"] == 4
    assert result["source_record_counts"]["known_issue"] == 3
    assert len(result["examples"]["known_issue"]) == 1


def test_raw_shape_failure_is_redacted():
    with pytest.raises(ValueError, match="redacted"):
        audit_raw([{"messages": [{"content": {"PRIVATE_MEDICAL": "secret"}}]}])


def test_normalized_overlap_and_repaired_overlap():
    docs = [SourceRouter().route("Ｆatigue   during pregnullcy."), SourceRouter().route("fatigue during pregnancy.")]
    assert not overlap_audit(docs, [0, 1], ["train", "tune"])["counts"].get("free_text_cross_split_text_hashes")
    overlap = overlap_audit(docs, [0, 1], ["train", "tune"], repair=True)
    assert overlap["counts"]["free_text_cross_split_text_hashes"] == 1
    assert overlap["cross_split_examples"][0]["source_indices"] == [0, 1]
    assert "Fatigue" not in json.dumps(overlap)


def test_frozen_split_validates_coverage_overlap_and_hash(fixture_inputs):
    f = fixture_inputs
    _, docs, _ = audit_raw(f["records"])
    result, groups, splits = audit_splits(f["records"], docs, f["manifest"], None, file_sha(f["raw"]))
    assert result["seed_split_matches"] and groups == f["groups"] and splits == f["splits"]
    with pytest.raises(ValueError, match="SHA256"):
        audit_splits(f["records"], docs, f["manifest"], None, "0" * 64)
    bad = {**f["manifest"], "train_groups": list(range(10))}
    with pytest.raises(ValueError, match="overlap"):
        audit_splits(f["records"], docs, bad, None, file_sha(f["raw"]))


def test_saved_shape_schema_and_provenance_valid_for_unsampled():
    row, *context = row_context()
    counts, issues = check_row(row, *context)
    assert not issues
    assert counts["schema_valid_rows"] == 1
    assert counts["positive_edges"] == 1 and counts["none_pairs"] == 1
    assert counts["unannotated_silver_pairs_taught_none"] == 1


@pytest.mark.parametrize("mutation,issue", [
    (lambda r: r.update(group_id=3), "row_source_group_or_split_mismatch"),
    (lambda r: r.update(completion="<think>PRIVATE_TEXT</think>"), "completion_invalid_json"),
    (lambda r: r.update(target={"decisions": []}), "completion_target_mismatch"),
    (lambda r: r.update(sample_weight=float("nan")), "invalid_sample_weight"),
    (lambda r: r["pairs"][0].update(source="MISSING"), "payload_reference_or_parser_error"),
    (lambda r: r["original_spans"].update(SENT_001="PRIVATE_REPLACEMENT"), "original_spans_not_original_source"),
    (lambda r: r.update(prompt=r["prompt"] + "{}\n"), "prompt_payload_mismatch"),
])
def test_stored_row_tampering_detected_without_text(mutation, issue):
    row, *context = row_context()
    mutation(row)
    _, issues = check_row(row, *context)
    assert issue in issues
    assert "PRIVATE_" not in json.dumps(issues)


def test_noise_or_gold_flags_cannot_leak_through_entity_table():
    row, *context = row_context()
    row["entities"][0]["is_noise"] = True
    payload = json.loads(row["prompt"].strip().splitlines()[-1])
    payload["entities"][0]["is_noise"] = True
    row["prompt"] = row["prompt"].splitlines()[0] + "\n" + dumps(payload) + "\n"
    _, issues = check_row(row, *context)
    assert issues["unexpected_entity_fields_in_prompt"] == 1


def test_missing_target_evidence_reference_detected():
    row, *context = row_context()
    next(d for d in row["target"]["decisions"] if d["relations"])["relations"][0]["evidence_span_ids"] = ["SENT_999"]
    row["completion"] = dumps(row["target"])
    _, issues = check_row(row, *context)
    assert issues["target_invalid_evidence"] == 1


def test_sampling_counts_and_unknown_collateral_conserve():
    nodes = NODES + [{"id": "D2", "label": "Disease", "name": "Condition Beta"}]
    graph = {"entities": nodes, "relations": [EDGE, {**EDGE, "target": "D2", "evidence": "MISSING_42"}]}
    record = synthetic_record(TEXT + " Condition Beta is also named.", graph)
    c = trace_record(record, 3, ProtocolConfig(strict_evidence=True), sampled=True)
    assert c["all_candidate_pairs"] == 6
    assert c["direct_unknown_pairs"] == 1 and c["collateral_positive_edges"] == 1
    assert c["collateral_none_pairs"] == 4 and c["retained_pairs"] == 0
    assert c["positive_edge_conservation_gap"] == c["pair_conservation_gap"] == 0


def test_negative_downsampling_is_train_only_and_seed_reproducible():
    nodes = [{"id": str(i), "label": "Disease", "name": f"Synthetic condition {i}"} for i in range(8)]
    text = ". ".join(n["name"] for n in nodes) + "."
    record = synthetic_record(text, {"entities": nodes, "relations": []})
    config = ProtocolConfig()
    train = trace_record(record, 8, config, sampled=True)
    tune = trace_record(record, 8, config, sampled=False)
    assert train == trace_record(record, 8, config, sampled=True)
    assert train["sampled_unannotated_pairs"] == 8 and train["downsampled_unannotated_pairs"] == 48
    assert tune["retained_none_pairs"] == 56 and tune["downsampled_unannotated_pairs"] == 0


def test_ungrounded_positives_are_counted_outside_candidates():
    graph = {"entities": NODES, "relations": [EDGE]}
    c = trace_record(synthetic_record("Only Fatigue is named.", graph), 1, ProtocolConfig(), sampled=True)
    assert c["positive_edges_outside_candidates"] == 1
    assert c["retained_positive_edges"] == 0 and c["positive_edge_conservation_gap"] == 0


def test_synthetic_bug_detector_tracks_current_code_without_locking_in_bug():
    report = reproduce_bugs()
    partial = report["partial_evidence_list"]
    assert not partial["all_fragments_found"] and partial["fragments"] == 2
    assert partial["bug_observed"] == partial["production_accepts"]
    provenance = report["sampled_original_span_provenance"]
    assert provenance["unsampled"]["original_span_mismatches"] == 0
    assert provenance["bug_observed"] == (provenance["sampled"]["original_span_mismatches"] > 0)
    # Core fixes may turn these booleans false without breaking the audit tests.
    assert TEXT not in json.dumps(report)


def test_masks_are_real_code_with_explicitly_synthetic_tokenizer():
    result = synthetic_loss_mask_check()
    assert result["all_passed"]
    assert not result["real_qwen_tokenization_verified"]
    assert not result["model_or_gpu_executed"]


def test_full_audit_is_deterministic_private_and_replays_actual_converter(fixture_inputs, tmp_path):
    f = fixture_inputs
    kwargs = {"raw": f["raw"], "data_dir": f["data_dir"], "expected_sha256": file_sha(f["raw"]), "expected_records": 20}
    report = run_audit(**kwargs)
    repeated = run_audit(**kwargs)
    assert report == repeated
    assert report["status"] == "audit_completed"
    assert report["splits"]["tune_records_exact_content_and_order"]
    entry = report["variants"]["Gsynthetic"]
    assert all(v["matches_saved_bytes"] and v["trace_matches_converter"] for v in entry["replay"]["splits"].values())
    assert all(v["manifest_hash_matches"] for v in entry["saved"].values())
    assert not report["code_changed_during_audit"] and not report["inputs_changed_during_audit"]
    output = tmp_path / "report"
    write_report(report, output)
    assert (output / "audit.json").stat().st_mode & 0o777 == 0o600
    content = (output / "audit.json").read_text()
    assert TEXT not in content and "SYNTHETIC_GROUP_" not in content
    with pytest.raises(ValueError, match="overwrite"):
        write_report(report, output)


def test_saved_file_hash_tampering_is_reported(fixture_inputs):
    f = fixture_inputs
    _, docs, graphs = audit_raw(f["records"])
    path = f["root"] / "train.jsonl"
    with path.open("a") as handle:
        handle.write("\nNOT_JSON\n")
    result = audit_saved(path, docs, graphs, f["groups"], f["splits"], f["config"], f["manifest"], "train")
    assert not result["manifest_hash_matches"]
    assert result["issues"]["counts"]["blank_line"] == 1
    assert result["issues"]["counts"]["invalid_json"] == 1


def test_fail_closed_source_hash_and_count(fixture_inputs):
    f = fixture_inputs
    with pytest.raises(ValueError, match="SHA256"):
        run_audit(raw=f["raw"], data_dir=f["data_dir"], expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="count"):
        run_audit(raw=f["raw"], data_dir=f["data_dir"], expected_sha256=file_sha(f["raw"]), expected_records=1345)


def test_cli_errors_redact_json_content(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text("PRIVATE_INVALID_MEDICAL_TEXT", encoding="utf-8")
    code = main(["--raw", str(path), "--expected-sha256", file_sha(path), "--output-dir", str(tmp_path / "new")])
    assert code == 2
    captured = capsys.readouterr()
    assert "audit_failed" in captured.err and "PRIVATE_" not in captured.err
    assert not (tmp_path / "new").exists()


def test_discovery_is_bounded_to_known_data_locations(tmp_path):
    inputs = default_inputs(tmp_path)
    assert all(str(p).startswith(str(tmp_path / "work")) for p in inputs.values())
    assert not any("tooling" in p.parts for p in inputs.values())


def test_bounded_dev_overlap_checks_hashes_and_preserves_dataset_indices(tmp_path):
    raw, dev = tmp_path / "train.json", tmp_path / "dev.json"
    raw.write_text(json.dumps([synthetic_record(TEXT, GRAPH)]), encoding="utf-8")
    dev.write_text(json.dumps([synthetic_record(TEXT.upper(), GRAPH)]), encoding="utf-8")
    kwargs = {"expected_sha256": file_sha(raw), "expected_dev_sha256": file_sha(dev), "expected_records": 1, "expected_dev_records": 1}
    result = audit_source_overlap(raw, dev, **kwargs)
    assert result["status"] == "source_overlap_completed"
    assert result["comparisons"]["exact_routed"]["counts"]["free_text_cross_split_text_hashes"] == 0
    normalized = result["comparisons"]["normalized_routed"]
    assert normalized["counts"]["free_text_cross_split_text_hashes"] == 1
    assert normalized["cross_split_examples"][0]["locations"] == [{"dataset": "train_source", "source_index": 0}, {"dataset": "historical_dev", "source_index": 0}]
    assert TEXT not in json.dumps(result)
    write_report(result, tmp_path / "overlap_report")
    assert (tmp_path / "overlap_report/overlap.json").is_file()
    with pytest.raises(ValueError, match="SHA256"):
        audit_source_overlap(raw, dev, **{**kwargs, "expected_dev_sha256": "0" * 64})


def test_bounded_overlap_does_not_read_labels_or_replay_variants(tmp_path, monkeypatch):
    raw, dev = tmp_path / "train.json", tmp_path / "dev.json"
    rows = [{"messages": [{"role": "user", "content": "SYNTHETIC_SOURCE_ONLY"}]}]
    for path in (raw, dev):
        path.write_text(json.dumps(rows), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("No graph processing allowed")
    monkeypatch.setattr("robustness.data_audit._graph", forbidden)
    monkeypatch.setattr("robustness.data_audit.convert", forbidden)
    result = audit_source_overlap(raw, dev, expected_sha256=file_sha(raw), expected_dev_sha256=file_sha(dev), expected_records=1, expected_dev_records=1)
    assert result["inputs_unchanged"]


def test_finding_locations_reject_content_instead_of_leaking_it():
    issues = Findings(1)
    with pytest.raises(ValueError):
        issues.add("example", source_index="PRIVATE_TEXT")
    with pytest.raises(ValueError):
        issues.add("other", text="PRIVATE_TEXT")
    assert sha("same") == sha("same")
    assert canonical_sha({"a": 1, "b": 2}) == canonical_sha({"b": 2, "a": 1})

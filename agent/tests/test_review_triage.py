"""Offline synthetic contracts plus opt-in, read-only 264-issue integration."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

import data_repair.review_triage as module
from data_repair.review_triage import (
    MECHANICAL, build_train_surface_candidates, canonical, classify_entity,
    run, sha, strict_json,
)


def record(index, name="Alpha condition, with feature", text="Alpha condition with feature.", split="train"):
    graph = {"entities": [{"id": "D1", "name": name, "label": "Disease"},
                          {"id": "S1", "name": "feature", "label": "Symptom"}],
             "relations": [{"source": "D1", "target": "S1", "relation": "unchanged_label",
                            "evidence": [text, "Second unlocated citation."]}]}
    original = {"messages": [{"role": "user", "content": text},
                              {"role": "assistant", "content": canonical(graph)}]}
    return {"source_index": index, "split": split, "group_id": index,
            "original_record": original, "original_record_sha256": sha(canonical(original)),
            "repaired_graph": graph, "source_type": "free_text",
            "source": {"original_text": text, "text": text, "original_sha256": sha(text), "sha256": sha(text)},
            "source_view": {"raw_payload": text, "labelled_lists": [{"raw": "untyped preserved"}], "issues": []},
            "entities": [{"id": "D1", "mentions": [], "status": "alias_or_context_review"}],
            "annotations": [{"index": 0, "annotation": deepcopy(graph["relations"][0]),
                             "evidence_fragments": [[], []], "all_evidence_located": False, "pair_blocked": True}],
            "issues": [], "record_blocked": True, "blocked_pairs": [["D1", "S1"]]}


def entity_issue(index, split="train", **extra):
    return {"source_index": index, "split": split, "entity_id": "D1", "kind": "entity",
            "reason": "original alias/context reason; not confirmed false", "decision": "PENDING",
            "not_a_negative": True, **extra}


def release(tmp_path, records=None, queue=None):
    records = records or [record(0), record(1, split="tune")]
    queue = queue or [entity_issue(r["source_index"], r["split"]) for r in records]
    path = tmp_path / "candidate"
    path.mkdir()
    data = {
        "records.jsonl": records,
        "review_queue.jsonl": queue,
        "frozen_split.jsonl": [{k: r[k] for k in ("source_index", "group_id", "split")} for r in records],
    }
    hashes = {}
    for name, rows in data.items():
        # Noncanonical formatting proves that the original queue is byte-preserved.
        blob = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
        (path / name).write_bytes(blob)
        hashes[name] = sha(blob)
    manifest = {"review_items": len(queue), "summary": {"records": len(records)},
                "output_sha256": hashes, "source_sha256": "frozen-teacher-hash"}
    (path / "manifest.json").write_text(canonical(manifest))
    return path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def refresh_hash(path, filename):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][filename] = sha((path / filename).read_bytes())
    manifest_path.write_text(canonical(manifest))


@pytest.mark.parametrize("name,text,group", [
    ("Alpha", "Alpha, not alphabet.", "entity_exact_surface"),
    ("A B", "a\t b", "entity_case_whitespace_surface"),
    ("kg/m²", "ＫＧ/ｍ2", "entity_unicode_surface"),
    ("café", "Cafe\u0301", "entity_unicode_surface"),
    ("Straße", "STRASSE", "entity_case_whitespace_surface"),
    ("Post traumatic stress disorder", "Post-Traumatic Stress Disorder", "entity_hyphen_boundary_surface"),
    ("Post-traumatic", "Post traumatic", "entity_hyphen_boundary_surface"),
    ("Alpha condition, with feature", "Alpha condition with feature", "entity_comma_surface"),
    ("Alpha condition with feature", "Alpha condition, with feature", "entity_comma_surface"),
    ("Alpha-beta, condition", "Alpha beta condition", "entity_mixed_separator_surface"),
    ("Manic Episode", "Hypomanic Episode", "entity_inside_word_review"),
    ("EEG", "EEGs", "entity_inside_word_review"),
    ("Alpha condition", "Alpha conditions", "entity_inside_word_review"),
    ("Alpha condition", "Alpha conditionand feature", "entity_inside_word_review"),
    ("Alpha condition, severe", "Alpha condition is discussed.", "entity_partial_prefix_review"),
    ("Unseen disorder", "Entirely unrelated source.", "entity_unlocated_review"),
])
def test_deterministic_mechanical_and_uncertain_categories(name, text, group):
    observed, evidence = classify_entity(name, text)
    assert observed == group
    assert classify_entity(name, text) == (observed, evidence)
    for hit in evidence.get("matches", []) + evidence.get("lexical_hints", []):
        assert text[hit["start"]:hit["end"]] == hit["text"]
        assert text[hit["context_start"]:hit["context_end"]] == hit["context"]
    if group not in MECHANICAL:
        assert evidence["matches"] == []


@pytest.mark.parametrize("name,text", [
    ("1 000 mg", "1,000 mg"), ("1.5 mg", "15 mg"), ("2 mg", "12 mg"),
    ("Alpha 2", "Alpha-2"), ("Alpha beta", "Alpha/beta"),
    ("Alpha beta", "Alpha+beta"), ("Alpha beta", "Alpha−beta"),
    ("Alpha beta", "Alpha. Beta"), ("Alpha beta", "Alpha; beta"),
    ("Alpha beta", "Alpha: beta"), ("Alpha beta", "Alpha (beta)"),
    ("Alpha beta", "Alpha's beta"), ("Alpha beta", "Alpha--beta"),
    ("Alpha, beta", "Alpha-beta"), ("(Alpha beta)", "Alpha beta"),
    ("Alpha beta", "Alphabeta"), ("with pain", "without pain"),
    ("Bipolar I", "Bipolar II"), ("pain", "pain\u0338"),
    ("s", "ß"), ("cafe", "café"), ("pregnancy", "pregnullcy"),
    ("Alpha condition", "Alphas condition"), ("Alpha condition", "condition Alpha"),
])
def test_no_unsafe_punctuation_substrings_morphology_alias_or_negation(name, text):
    category, evidence = classify_entity(name, text)
    assert category not in MECHANICAL
    assert evidence["matches"] == []


def test_unicode_expansion_preserves_raw_offsets_and_all_occurrences():
    raw = "💊Ａlpha， beta. Alpha beta."
    category, evidence = classify_entity("Alpha, beta", raw)
    # NFKC finds the first occurrence; the strict tier need not broaden further.
    assert category == "entity_unicode_surface"
    assert evidence["matches"][0]["start"] == 1
    assert evidence["matches"][0]["text"] == "Ａlpha， beta"
    category, evidence = classify_entity("Alpha, beta", "Alpha beta; Alpha beta")
    assert category == "entity_comma_surface" and len(evidence["matches"]) == 2


def test_lossless_issue_ids_originals_evidence_splits_and_reproducibility(tmp_path):
    records = [record(0), record(1, split="tune"), record(2, split="test")]
    queue = [entity_issue(0, issue_id="keep-this-id"), entity_issue(1, "tune"), entity_issue(2, "test"),
             {"source_index": 0, "split": "train", "kind": "relation", "relation_index": 0,
              "reason": "binding", "decision": "PENDING", "not_a_negative": True}]
    candidate = release(tmp_path, records, queue)
    before = {p.name: p.read_bytes() for p in candidate.iterdir()}
    one, two = tmp_path / "one", tmp_path / "two"
    manifest = run(candidate, one, expected_items=4, expected_manifest_sha256=sha(before["manifest.json"]))
    assert run(candidate, two, expected_items=4) == manifest
    assert {p.name: p.read_bytes() for p in candidate.iterdir()} == before
    assert (one / "original_review_queue.jsonl").read_bytes() == before["review_queue.jsonl"]
    rows = read_rows(one / "triage.jsonl")
    assert [r["original_issue"] for r in rows] == queue
    assert rows[0]["issue_id"] == "keep-this-id"
    assert rows[1]["issue_id"].startswith("rq-" + sha(before["review_queue.jsonl"]))
    assert [r["queue_line"] for r in rows] == list(range(1, 5))
    snapshots = read_rows(one / "source_records.jsonl")
    assert snapshots == records
    for row in rows:
        source = snapshots[row["record_ref"]["line"] - 1]
        assert sha(canonical(source)) == row["record_ref"]["prepared_record_sha256"]
        assert source["original_record"] == records[row["source_index"]]["original_record"]
        assert row["focus"]["existing_evidence_audit"] == source["annotations"]
        assert row["focus"]["relations"][0]["annotation"]["evidence"] == [source["source"]["text"], "Second unlocated citation."]
        assert row["focus"]["record_blocked"] and row["focus"]["blocked_pairs"] == [["D1", "S1"]]
        assert row["decision"] == "PENDING" and row["not_a_negative"] and row["human_review_required"]
        assert row["proposal_only"]
        if row["category"] == "entity_comma_surface":
            assert row["separator_scope_requires_human_review"]
        assert not row["gold"] and row["label_change"] is None
    assert manifest["counts"]["input_issues"] == manifest["counts"]["output_issues"] == 4
    assert manifest["counts"]["human_review_required"] == 4
    assert manifest["counts"]["automatically_closed"] == 0
    assert manifest["proposal_only"] and manifest["status"].startswith("PROPOSAL_ONLY")
    for relative, digest in manifest["output_sha256"].items():
        assert sha((one / relative).read_bytes()) == digest
        assert (one / relative).read_bytes() == (two / relative).read_bytes()
        assert (one / relative).stat().st_mode & 0o777 == 0o600
    assert (one / "manifest.json").read_bytes() == (two / "manifest.json").read_bytes()
    assert one.stat().st_mode & 0o777 == 0o700
    markdown = "".join(p.read_text() for p in (one / "groups").glob("*.md"))
    assert "|---|---:|\n|" in (one / "README.md").read_text()
    for row in rows:
        assert markdown.count("## " + row["triage_id"] + "\n") == 1
    candidates = read_rows(one / "train_surface_candidates.jsonl")
    assert candidates and all(p["split"] == "train" and p["source_index"] == 0
                              for c in candidates for p in c["provenance"])


def test_duplicate_issues_are_retained_not_deduplicated(tmp_path):
    issue = entity_issue(0, id="original-id")
    candidate = release(tmp_path, [record(0)], [issue, deepcopy(issue)])
    run(candidate, tmp_path / "out", expected_items=2)
    rows = read_rows(tmp_path / "out/triage.jsonl")
    assert len(rows) == 2 and rows[0]["triage_id"] != rows[1]["triage_id"]
    assert [r["issue_id"] for r in rows] == ["original-id", "original-id"]
    assert rows[0]["original_issue"] == rows[1]["original_issue"] == issue


@pytest.mark.parametrize("split", ["tune", "test", "historical_regression_audit_only", ""])
def test_train_builder_rejects_holdout_before_accessing_any_semantic_field(split):
    # A KeyError would expose unsafe reads of missing semantic fields.
    with pytest.raises(ValueError, match="train-only"):
        build_train_surface_candidates([{"split": "train"}, {"split": split}])


def test_holdout_changes_cannot_change_train_surface_candidates(tmp_path, monkeypatch):
    candidate = release(tmp_path)
    original_builder = module.build_train_surface_candidates
    calls = []

    def guarded(rows):
        rows = list(rows)
        assert rows and all(r["split"] == "train" for r in rows)
        calls.append(deepcopy(rows))
        return original_builder(rows)

    monkeypatch.setattr(module, "build_train_surface_candidates", guarded)
    run(candidate, tmp_path / "first", expected_items=2)
    records = read_rows(candidate / "records.jsonl")
    # Only non-training contents change; queue metadata and training input do not.
    records[1] = record(1, name="Poison invented alias", text="Poison-invented alias", split="tune")
    (candidate / "records.jsonl").write_text("".join(canonical(r) + "\n" for r in records))
    refresh_hash(candidate, "records.jsonl")
    run(candidate, tmp_path / "second", expected_items=2)
    assert len(calls) == 2
    assert (tmp_path / "first/train_surface_candidates.jsonl").read_bytes() == (
        tmp_path / "second/train_surface_candidates.jsonl").read_bytes()


def test_no_training_evaluation_or_external_files_read(tmp_path, monkeypatch):
    candidate = release(tmp_path)
    read = Path.read_bytes
    allowed = {candidate / name for name in module.INPUT_NAMES}
    allowed |= {Path(module.__file__), Path(module.__file__).with_name("text_normalization.py")}
    accesses = []

    def checked(path):
        assert path in allowed
        accesses.append(path)
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", checked)
    run(candidate, tmp_path / "out", expected_items=2)
    assert set(accesses) == allowed


@pytest.mark.parametrize("code,category", list(module.CODE_GROUPS.items()) + [("new-unknown-code", "other_review")])
def test_record_codes_remain_pending_without_relabel_or_extra_quarantine(tmp_path, code, category):
    issue = {"source_index": 0, "split": "train", "kind": "record_or_semantics",
             "details": {"code": code, "action": "hint", "reason": "Retain ``` raw evidence", "relation_index": 0},
             "decision": "PENDING", "not_a_negative": True}
    candidate = release(tmp_path, [record(0)], [issue])
    run(candidate, tmp_path / "out", expected_items=1)
    item = read_rows(tmp_path / "out/triage.jsonl")[0]
    assert item["category"] == category and item["original_issue"] == issue
    assert not item["mechanical_surface_located"]
    assert item["focus"]["existing_evidence_audit"][0]["pair_blocked"]
    assert "````json" in (tmp_path / f"out/groups/{category}.md").read_text()


@pytest.mark.parametrize("filename", ["records.jsonl", "review_queue.jsonl", "frozen_split.jsonl"])
def test_source_hash_mismatch_fails_before_output(tmp_path, filename):
    candidate = release(tmp_path)
    with (candidate / filename).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(candidate, tmp_path / "out", expected_items=2)
    assert not (tmp_path / "out").exists()


def test_expected_manifest_and_count_are_pinned_before_output(tmp_path):
    candidate = release(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA256"):
        run(candidate, tmp_path / "out", expected_items=2, expected_manifest_sha256="wrong")
    with pytest.raises(ValueError, match="count mismatch"):
        run(candidate, tmp_path / "out", expected_items=264)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["record_hash", "split", "missing_source", "missing_entity", "missing_relation", "no_relation_index",
                                    "duplicate_source", "duplicate_entity", "cross_split_group", "not_pending"])
def test_invalid_identity_or_provenance_fails_closed(tmp_path, problem):
    records = [record(0), record(1, split="tune")]
    queue = [entity_issue(0), entity_issue(1, "tune")]
    if problem == "record_hash":
        records[0]["original_record_sha256"] = "wrong"
    elif problem == "split":
        queue[0]["split"] = "tune"
    elif problem == "missing_source":
        queue[0]["source_index"] = 99
    elif problem == "missing_entity":
        queue[0]["entity_id"] = "missing"
    elif problem == "missing_relation":
        queue[0].update(kind="relation", relation_index=99)
    elif problem == "no_relation_index":
        queue[0]["kind"] = "relation"
    elif problem == "duplicate_source":
        records.append(deepcopy(records[0]))
    elif problem == "duplicate_entity":
        records[0]["repaired_graph"]["entities"].append(deepcopy(records[0]["repaired_graph"]["entities"][0]))
    elif problem == "cross_split_group":
        records[1]["group_id"] = 0
    elif problem == "not_pending":
        queue[0]["decision"] = "ACCEPTED"
    candidate = release(tmp_path, records, queue)
    with pytest.raises(ValueError):
        run(candidate, tmp_path / "out", expected_items=2)
    assert not (tmp_path / "out").exists()


def test_refuses_existing_directory_symlink_and_candidate_subdirectory(tmp_path):
    candidate = release(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("untouched")
    with pytest.raises(ValueError, match="overwrite"):
        run(candidate, out, expected_items=2)
    assert sentinel.read_text() == "untouched"
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "nonexistent", target_is_directory=True)
    with pytest.raises(ValueError, match="overwrite"):
        run(candidate, link, expected_items=2)
    with pytest.raises(ValueError, match="separate"):
        run(candidate, candidate / "new-output", expected_items=2)


def test_source_mutation_during_classification_fails_before_publication(tmp_path, monkeypatch):
    candidate = release(tmp_path)
    original = module.classify_entity

    def changed(name, text):
        with (candidate / "review_queue.jsonl").open("ab") as handle:
            handle.write(b" ")
        return original(name, text)

    monkeypatch.setattr(module, "classify_entity", changed)
    with pytest.raises(ValueError, match="source changed"):
        run(candidate, tmp_path / "out", expected_items=2)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["split_mismatch", "duplicate_frozen", "blank_queue", "source_symlink"])
def test_frozen_split_format_and_input_symlinks_are_checked(tmp_path, problem):
    candidate = release(tmp_path)
    frozen = candidate / "frozen_split.jsonl"
    if problem == "split_mismatch":
        rows = read_rows(frozen)
        rows[0]["split"] = "tune"
        frozen.write_text("".join(canonical(r) + "\n" for r in rows))
        refresh_hash(candidate, frozen.name)
    elif problem == "duplicate_frozen":
        with frozen.open("a") as handle:
            handle.write(canonical(read_rows(frozen)[0]) + "\n")
        refresh_hash(candidate, frozen.name)
    elif problem == "blank_queue":
        with (candidate / "review_queue.jsonl").open("a") as handle:
            handle.write("\n")
        refresh_hash(candidate, "review_queue.jsonl")
    else:
        original = candidate / "records.jsonl"
        original.rename(tmp_path / "moved-records")
        original.symlink_to(tmp_path / "moved-records")
    with pytest.raises(ValueError):
        run(candidate, tmp_path / "out", expected_items=2)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json_rejects_ambiguous_inputs(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_real_264_queue_integration(tmp_path):
    setting = os.environ.get("REVIEW_TRIAGE_REAL_CANDIDATE")
    if not setting:
        pytest.skip("set REVIEW_TRIAGE_REAL_CANDIDATE for offline private-corpus integration")
    candidate = Path(setting).resolve()
    original = {name: sha((candidate / name).read_bytes()) for name in module.INPUT_NAMES}
    result = run(candidate, tmp_path / "real")
    assert run(candidate, tmp_path / "real-repeat") == result
    counts = result["counts"]
    assert counts["input_records"] == 1345
    assert counts["input_issues"] == counts["output_issues"] == counts["human_review_required"] == 264
    assert counts["by_kind"] == {"entity": 251, "relation": 2, "record_or_semantics": 11}
    assert counts["by_split"] == {"train": 234, "tune": 30}
    assert counts["mechanically_located_entity_issues"] == 18
    assert counts["categories"]["entity_comma_surface"] == 13
    assert counts["categories"]["entity_hyphen_boundary_surface"] == 5
    assert counts["still_unlocated_entity_issues"] == 233
    assert counts["automatically_closed"] == counts["label_changes"] == 0
    queue = read_rows(candidate / "review_queue.jsonl")
    rows = read_rows(tmp_path / "real/triage.jsonl")
    assert [r["original_issue"] for r in rows] == queue
    assert sum(counts["categories"].values()) == 264
    records = {r["source_index"]: r for r in read_rows(candidate / "records.jsonl")}
    snapshots = read_rows(tmp_path / "real/source_records.jsonl")
    assert len(snapshots) == counts["preserved_review_records"]
    assert {r["source_index"] for r in snapshots} == {r["source_index"] for r in queue}
    for row in rows:
        source = snapshots[row["record_ref"]["line"] - 1]
        assert source == records[row["source_index"]]
        assert row["record_ref"]["prepared_record_sha256"] == sha(canonical(source))
        assert row["record_ref"]["original_record_sha256"] == sha(canonical(source["original_record"]))
        assert row["decision"] == "PENDING" and row["not_a_negative"] and row["human_review_required"]
        assert row["proposal_only"]
        if row["category"] in {"entity_comma_surface", "entity_hyphen_boundary_surface"}:
            assert row["separator_scope_requires_human_review"]
        assert row["label_change"] is None and not row["semantic_equivalence_confirmed"]
        for hit in row["observations"].get("matches", []) + row["observations"].get("lexical_hints", []):
            assert source["source"]["original_text"][hit["start"]:hit["end"]] == hit["text"]
    for relative, digest in result["output_sha256"].items():
        data = (tmp_path / "real" / relative).read_bytes()
        assert sha(data) == digest
        assert data == (tmp_path / "real-repeat" / relative).read_bytes()
    train_candidates = read_rows(tmp_path / "real/train_surface_candidates.jsonl")
    assert all(records[p["source_index"]]["split"] == "train"
               for item in train_candidates for p in item["provenance"])
    with pytest.raises(ValueError, match="overwrite"):
        run(candidate, tmp_path / "real")
    assert {name: sha((candidate / name).read_bytes()) for name in module.INPUT_NAMES} == original
    by_source = {(r["source_index"], r["original_issue"]["kind"]): r for r in rows}
    for index in (200, 479, 1329):
        assert by_source[(index, "entity")]["category"] == "entity_inside_word_review"
    for index in (904, 909):
        assert by_source[(index, "record_or_semantics")]["category"] == "conflicting_input_review"
        assert by_source[(index, "record_or_semantics")]["focus"]["record_blocked"]

"""Frozen chosen-evidence overlap expansion with invented source examples."""
from collections import Counter
from copy import deepcopy
import os

import pytest

from semantic.data import compile_source, source_windows, validate_row, encode_row, sha
from semantic.relation_windows import compile_source_rows, chosen_anchors, _write, VARIANTS
from test_semantic_data import prepared, FakeTokenizer


def fixture(*, repeated=True):
    prefix = "Alpha has beta. "
    text = prefix + "x" * (51 - len(prefix)) + " clue. " + "z" * 90
    if repeated:
        text += " clue. "
    item = prepared(text)
    hit = text.index("clue.")
    hits = [{"original_start": hit, "original_end": hit + 5, "original_text": "clue."}]
    if repeated:
        other = text.rindex("clue.")
        hits.append({"original_start": other, "original_end": other + 5, "original_text": "clue."})
    item["annotations"][0]["annotation"]["evidence"] = "clue."
    item["annotations"][0]["evidence_fragments"] = [hits]
    full = next(r for r in compile_source(item)[0] if r["stage"] == "relations")
    views = {v: [r for r in compile_source(item, variant=v, short_chars=64)[0] if r["stage"] == "relations"]
             for v in VARIANTS}
    return item, full, views


def compile_test(item, full, views):
    return compile_source_rows(full, item, views, {}, short_chars=64, overlap=16)


def test_chosen_overlap_expands_but_other_equal_evidence_occurrence_is_not_migrated():
    item, full, views = fixture()
    originals = deepcopy((item, full, views))
    rows, audit, source_audit = compile_test(item, full, views)
    assert (item, full, views) == originals
    anchor = audit[0]["chosen_anchor"]
    windows = source_windows(item["source"]["original_text"], 64, 16)
    containing = [i for i, (a, b) in enumerate(windows) if a <= anchor["start"] and anchor["end"] <= b]
    assert len(containing) == 2
    assert audit[0]["new_overlap_occurrences"] == 1
    for variant in VARIANTS:
        assert [r["metadata"]["source_window"]["index"] for r in rows[variant]] == containing
        assert len(rows[variant]) == len(views[variant]) + 1
        for row in rows[variant]:
            assert row["target"] == full["target"]
            assert row["task_input"]["entities"] == full["task_input"]["entities"]
            assert validate_row(row)
            encoded = encode_row(row, FakeTokenizer(), 10000)
            assert encoded["labels"][-1] == FakeTokenizer.eos_token_id
    other = item["source"]["original_text"].rindex("clue.")
    other_windows = [i for i,(a,b) in enumerate(windows) if a <= other and other+5 <= b]
    assert set(other_windows) - set(containing)
    assert set(other_windows) - set(containing) <= set(source_audit["unknown_regular_window_indices"])


def test_minimum_prepared_anchor_is_used_even_when_hit_list_is_unsorted():
    item, full, views = fixture()
    item["annotations"][0]["evidence_fragments"][0].reverse()
    anchors = chosen_anchors(full, item)
    assert next(iter(anchors.values()))["start"] == 52
    rows, _, _ = compile_test(item, full, views)
    assert len(rows["short_source"]) == 2


def test_original_cross_boundary_auxiliary_row_is_preserved_without_inference_fallback():
    text = "Alpha " + "unrelated material " * 8 + "beta."
    item = prepared(text)
    full = next(r for r in compile_source(item)[0] if r["stage"] == "relations")
    views = {v: [r for r in compile_source(item, variant=v, short_chars=64)[0] if r["stage"] == "relations"]
             for v in VARIANTS}
    rows, audit, source_audit = compile_test(item, full, views)
    for variant in VARIANTS:
        assert len(rows[variant]) == 1
        row, old = rows[variant][0], views[variant][0]
        assert row["metadata"]["source_window"]["index"] == -1
        for field in ("prompt", "completion", "target", "target_count", "target_sha256"):
            assert row[field] == old[field]
        policy = row["metadata"]["relation_window_supervision"]
        assert policy["auxiliary_full_source_training_only"] is True
        assert policy["runtime_gold_fallback_allowed"] is False
    assert audit[0]["auxiliary_full_source_only"] is True
    assert len(source_audit["unknown_regular_window_indices"]) == source_audit["source_only_windows"]


def test_views_have_equal_atoms_windows_eos_and_unchanged_endpoint_names():
    item, full, views = fixture()
    # Add a second original annotation/evidence identity with distinct evidence.
    annotation = deepcopy(item["annotations"][0])
    annotation["index"] = 1
    quote = "Alpha has beta."
    annotation["annotation"]["evidence"] = quote
    annotation["evidence_fragments"] = [[{"original_start": 0, "original_end": len(quote), "original_text": quote}]]
    item["annotations"].append(annotation)
    full = next(r for r in compile_source(item)[0] if r["stage"] == "relations")
    views = {v: [r for r in compile_source(item, variant=v, short_chars=64)[0] if r["stage"] == "relations"]
             for v in VARIANTS}
    rows, _, _ = compile_test(item, full, views)
    signature = lambda data: Counter((r["metadata"]["source_window"]["index"], a)
                                    for r in data for a in r["metadata"]["supervised_atom_ids"])
    assert signature(rows["short_source"]) == signature(rows["short_canonical"])
    assert len(rows["short_source"]) == len(rows["short_canonical"])
    for variant in VARIANTS:
        assert all(r["task_input"]["entities"] == full["task_input"]["entities"] for r in rows[variant])
        assert all(r["metadata"]["relation_entity_table_scope"] == "complete_source_entities" for r in rows[variant])


def test_tune_prepared_and_anchor_drift_rejected():
    item, full, views = fixture()
    wrong = deepcopy(item); wrong["split"] = "tune"
    with pytest.raises(ValueError, match="TRAIN source"):
        compile_test(wrong, full, views)
    wrong = deepcopy(item); wrong["annotations"][0]["evidence_fragments"][0][0]["original_start"] += 1
    with pytest.raises(ValueError, match="span differs"):
        compile_test(wrong, full, views)


def test_old_assigned_window_must_match_chosen_original_hit():
    item, full, views = fixture()
    wrong = deepcopy(views)
    wrong["short_source"] = []
    with pytest.raises(ValueError, match="row coverage"):
        compile_test(item, full, wrong)


def test_old_entity_table_drift_rejected_before_emitting_targets():
    item, full, views = fixture()
    wrong = deepcopy(views)
    row = wrong["short_source"][0]
    row["task_input"]["entities"][0]["name"] += " invented"
    row["input_sha256"] = sha(row["task_input"])
    with pytest.raises(ValueError):
        compile_test(item, full, wrong)


def test_private_writer_uses_0600_and_refuses_overwrite(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write(path, [{"aggregate": 1}], jsonl=True)
    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write(path, [{"aggregate": 2}], jsonl=True)

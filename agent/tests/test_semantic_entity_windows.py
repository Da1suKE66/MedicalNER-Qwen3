"""Entity-window supervision tests use invented names, never clinical fixtures."""
from collections import Counter
from copy import deepcopy

import pytest

from semantic.data import compile_source, sha, source_windows, validate_row, encode_row
from semantic.entity_windows import compile_source_row, VARIANT
from test_semantic_data import prepared, FakeTokenizer


def entity_row(item, title=None):
    return next(row for row in compile_source(item, title)[0] if row["stage"] == "entities")


def test_repeat_grounded_entities_in_every_source_only_window_without_new_unique_atoms():
    text = ("Alpha has beta. " + "Background. " * 3) * 4
    base = entity_row(prepared(text))
    unchanged = deepcopy(base)
    rows, audits = compile_source_row(base, {}, short_chars=80, overlap=20)
    assert base == unchanged
    windows = source_windows(text, 80, 20)
    assert [(a["source_window"]["start"], a["source_window"]["end"]) for a in audits] == windows
    original = set(base["metadata"]["supervised_atom_ids"])
    exposure = Counter(atom for row in rows for atom in row["metadata"]["supervised_atom_ids"])
    assert set(exposure) == original and all(count > 1 for count in exposure.values())
    for row in rows:
        assert row["variant"] == VARIANT
        assert validate_row(row)
        assert len(row["target"]) == len({(e["label"], e["name"]) for e in row["target"]})
        for target in row["target"]:
            for mention in target["mentions"]:
                assert mention["context"] == "body"
                assert row["task_input"]["text"][mention["start"]:mention["end"]] == mention["text"]
        encoded = encode_row(row, FakeTokenizer(), 10000)
        assert encoded["labels"][-1] == FakeTokenizer.eos_token_id


def test_trusted_title_entity_supervised_in_every_window_with_title_coordinates():
    text = "It has beta. " + "Background words. " * 20
    raw = '{"title":"Alpha","code":"ZZ"}'
    original = {"messages": [{"role": "user", "content": raw}]}
    title = {"text": "Alpha", "source_group_id": "3", "source_record_sha256": sha(original),
             "source_start": raw.index("Alpha"), "source_end": raw.index("Alpha") + 5}
    receipts = {sha(original): {"source_group_id": "3", "original_record": original}}
    base = entity_row(prepared(text), title)
    rows, audits = compile_source_row(base, receipts, short_chars=80, overlap=20)
    assert len(rows) == len(source_windows(text, 80, 20)) == len(audits)
    for row in rows:
        disease = next(e for e in row["target"] if e["name"] == "Alpha")
        assert all(m["context"] == "title_context" for m in disease["mentions"])
        assert all(m["source_record_sha256"] == sha(original) for m in disease["mentions"])
    with pytest.raises(ValueError):
        compile_source_row(base, {}, short_chars=80, overlap=20)


def test_unknown_windows_not_treated_as_empty_negative_targets():
    text = "Alpha has beta. " + "Unannotated material. " * 30
    rows, audits = compile_source_row(entity_row(prepared(text)), {}, short_chars=80, overlap=20)
    assert len(audits) > len(rows)
    assert all(row["completion"] and row["target"] for row in rows)
    assert all(a["status"] == "unknown_no_admitted_annotations_not_trained" for a in audits if not a["target_count"])


def test_local_canonical_surface_can_replace_unavailable_alias_without_changing_name():
    text = "This disease has beta. " + "Background material. " * 6 + " Alpha."
    item = prepared(text)
    item["entities"][0]["mentions"] = [{"original_start": 0, "original_end": 12, "original_text": "This disease"}]
    base = entity_row(item)
    rows, _ = compile_source_row(base, {}, short_chars=80, overlap=20)
    diseases = [e for row in rows for e in row["target"] if e["label"] == "Disease"]
    assert diseases and all(e["name"] == "Alpha" and e["mention"] == "Alpha" for e in diseases)
    assert any(row["metadata"]["entity_window_supervision"]["mention_selection_counts"].get("exact_local_canonical_name_surface") for row in rows)


def test_uncoverable_long_entity_blocks_instead_of_gold_selected_full_fallback():
    name = "X" * 150
    base = entity_row(prepared(name + " has beta.", title=name))
    with pytest.raises(ValueError, match="no gold fallback"):
        compile_source_row(base, {}, short_chars=64, overlap=16)


def test_heldout_and_source_hash_drift_rejected():
    with pytest.raises(ValueError, match="Held-out"):
        compile_source_row(entity_row(prepared(split="tune")), {})
    base = entity_row(prepared())
    base["metadata"]["source_text_sha256"] = "0" * 64
    base["input_sha256"] = sha(base["task_input"])
    with pytest.raises(ValueError, match="identity/window"):
        compile_source_row(base, {})


def test_order_uses_current_window_positions_not_document_first_mentions():
    text = "Alpha has beta. " + "Background. " * 6 + "beta then Alpha."
    rows, _ = compile_source_row(entity_row(prepared(text)), {}, short_chars=64, overlap=16)
    final = rows[-1]
    assert [e["name"] for e in final["target"]] == ["beta", "Alpha"]
    assert final["metadata"]["source_window"]["start"] > 0

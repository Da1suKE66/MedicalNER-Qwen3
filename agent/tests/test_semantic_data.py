"""Compiler invariants on invented nonclinical source strings only."""
from collections import Counter
from copy import deepcopy
import json

import pytest

from semantic.data import (DATA_VERSION, build_title_context, canonical, compile_eval_record,
                           compile_source, encode_row, sha, source_windows, validate_row)


def prepared(text="Alpha has beta.", *, title="Alpha", split="train"):
    graph = {"entities": [{"id": "d", "label": "Disease", "name": title},
                           {"id": "s", "label": "Symptom", "name": "beta"}],
             "relations": [{"source": "s", "target": "d", "relation": "is_associated_symptom_of", "evidence": text}]}
    original = {"messages": [{"role": "user", "content": text},
                              {"role": "assistant", "content": json.dumps(graph)}]}
    def locations(value):
        if value not in text:
            return []
        a = text.index(value)
        return [{"original_start": a, "original_end": a + len(value), "original_text": value}]
    entities = [{"id": e["id"], "canonical_name": e["name"], "label": e["label"], "mentions": locations(e["name"])} for e in graph["entities"]]
    return {"source_index": 7, "group_id": 3, "split": split, "source_type": "free_text",
            "original_record": original, "original_record_sha256": sha(original),
            "source_view_origin": "original_teacher_user", "source": {"original_text": text},
            "source_view": {"raw_payload": text}, "entities": entities,
            "annotations": [{"index": 0, "annotation": graph["relations"][0], "pair_blocked": False,
                             "all_evidence_located": True, "evidence_fragments": [locations(text)]}],
            "record_blocked": False}


def receipt(title="Alpha", group="3"):
    return {"text": title, "source_group_id": group, "source_record_sha256": "a" * 64,
            "source_start": 4, "source_end": 4 + len(title)}


class FakeTokenizer:
    eos_token_id = 900000
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return messages[0]["content"] + "</think>"
    def encode(self, value, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in value]


def atoms(rows):
    return Counter((r["stage"], a) for r in rows for a in r["metadata"]["supervised_atom_ids"])


def test_full_source_roundtrip_and_partial_supervision():
    rows, audit = compile_source(prepared())
    assert {r["stage"] for r in rows} == {"entities", "relations"}
    assert audit["entities_atoms"] == 2 and audit["relations_atoms"] == 1
    for row in rows:
        assert validate_row(json.loads(canonical(row)))
        assert row["data_version"] == DATA_VERSION
        assert "UNKNOWN" in row["metadata"]["unannotated_pair_policy"]
        assert "not_exhaustive" in row["metadata"]["eos_semantics"]
        assert not row["completion"].startswith("{")
        assert "NONE" not in row["completion"]


def test_trusted_title_recovers_unlocated_subject():
    rows, audit = compile_source(prepared("It has beta."), receipt())
    assert audit["entity_grounded_by_trusted_title"] == 1
    assert audit["relations_atoms"] == 1
    assert "SOURCE TITLE" in rows[0]["prompt"]


def test_no_title_no_gold_derived_alias():
    rows, audit = compile_source(prepared("It has beta."))
    assert audit["entity_canonical_name_not_source_supported"] == 1
    assert audit["relation_endpoint_unavailable"] == 1
    assert all(r["stage"] == "entities" for r in rows)


@pytest.mark.parametrize("variant", ["full_source", "short_source", "short_canonical"])
def test_counter_conservation_with_fallback(variant):
    text = "Alpha " + "unrelated sentence. " * 12 + "has beta."
    item = prepared(text)
    full, _ = compile_source(item)
    rows, audit = compile_source(item, variant=variant, short_chars=64)
    assert atoms(full) == atoms(rows)
    if variant != "full_source":
        assert audit["relations_fallback_atoms"] == 1
        assert any(r["metadata"]["source_window"]["full_source_fallback"] for r in rows)


def test_windows_source_only_exact_coverage():
    text = "A sentence. " * 100
    windows = source_windows(text, 80, 12)
    coverage = set()
    for a, b in windows:
        assert 0 <= a < b <= len(text)
        assert b - a <= 80
        coverage.update(range(a, b))
    assert coverage == set(range(len(text)))


@pytest.mark.parametrize("size,overlap", [(0, 0), (31, 2), (64, 64), (64, -1)])
def test_invalid_window_settings(size, overlap):
    with pytest.raises(ValueError):
        source_windows("x", size, overlap)


def test_duplicate_type_name_ids_are_merged_and_edges_remapped():
    item = prepared()
    extra = deepcopy(item["entities"][0]); extra["id"] = "d_alias"
    item["entities"].append(extra)
    item["annotations"][0]["annotation"]["target"] = "d_alias"
    rows, audit = compile_source(item)
    assert audit["entity_duplicate_type_name_merged"] == 1
    assert next(r for r in rows if r["stage"] == "relations")["target"][0]["target"] == "d"


def test_no_empty_silver_supervision_after_blocked_edge():
    item = prepared(); item["annotations"][0]["pair_blocked"] = True
    rows, audit = compile_source(item)
    assert len(rows) == 1 and rows[0]["stage"] == "entities"
    assert audit["relation_pair_blocked"] == 1
    assert audit["relations_no_admitted_positive_targets_no_empty_supervision"] == 1


def test_quarantine_record_is_processed_without_supervision():
    item = prepared(); item["record_blocked"] = True
    rows, audit = compile_source(item)
    assert rows == [] and audit["source_records_processed"] == 1
    assert audit["quarantined_record_no_training_supervision"] == 1


def test_unlocated_evidence_not_silently_used():
    item = prepared(); item["annotations"][0]["all_evidence_located"] = False
    rows, audit = compile_source(item)
    assert audit["relation_evidence_unlocated"] == 1
    assert len(rows) == 1


def test_negative_qualifier_preserved_in_source_evidence():
    text = "Alpha has beta but is not severe."
    rows, _ = compile_source(prepared(text))
    edge = next(r for r in rows if r["stage"] == "relations")
    assert "not severe" in edge["completion"] and text == edge["task_input"]["text"]


def test_target_type_rejection_audited_not_negative():
    item = prepared()
    item["annotations"][0]["annotation"].update(source="d", target="s", relation="subtype_of")
    rows, audit = compile_source(item)
    assert audit["relation_rejection:target_type_mismatch"] == 1
    assert all(r["stage"] == "entities" for r in rows)


def test_encoder_masks_prompt_and_single_eos():
    row = compile_source(prepared())[0][0]
    tokens = encode_row(row, FakeTokenizer(), 100000)
    n = tokens["_choice_counts"]["prompt_tokens"]
    assert tokens["labels"][:n] == [-100] * n
    assert tokens["labels"][n:] == tokens["input_ids"][n:]
    assert tokens["input_ids"][-1] == 900000
    assert tokens["input_ids"].count(900000) == 1
    assert tokens["_choice_counts"]["action_count_unit"] == "semantic_annotation_records"


def test_encoder_refuses_budget_and_tune():
    row = compile_source(prepared())[0][0]
    with pytest.raises(ValueError, match="Untruncated"):
        encode_row(row, FakeTokenizer(), 1)
    tune = compile_source(prepared(split="tune"))[0][0]
    with pytest.raises(ValueError, match="Held-out"):
        encode_row(tune, FakeTokenizer(), 100000)
    assert encode_row(tune, FakeTokenizer(), 100000, require_train=False)


@pytest.mark.parametrize("field", ["prompt", "completion", "input_sha256", "target_sha256"])
def test_mutation_rejected(field):
    row = compile_source(prepared())[0][0]
    row[field] += "x"
    with pytest.raises(ValueError):
        validate_row(row)


def test_eval_keeps_full_original_gold_when_training_has_quarantine():
    item = prepared(); item["record_blocked"] = True
    evaluation = compile_eval_record(item)
    assert evaluation["gold"]["relations"] == [item["annotations"][0]["annotation"]]
    assert evaluation["original_record"] == item["original_record"]
    assert evaluation["input_identity"]["sha256"] == sha({"text": evaluation["text"], "metadata": evaluation["metadata"]})


def test_cross_group_title_and_original_sha_drift_rejected():
    with pytest.raises(ValueError, match="Cross-group"):
        compile_source(prepared(), receipt(group="99"))
    item = prepared(); item["original_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA drift"):
        compile_source(item)


def test_title_receipt_reads_original_user_only_and_preserves_offsets():
    item = prepared(); item["source_type"] = "structured_icd"
    item["source_view"]["routed"] = {"metadata": {"title": "Alpha"}}
    title = build_title_context(item)
    raw = item["original_record"]["messages"][0]["content"]
    assert raw[title["source_start"]:title["source_end"]] == title["text"] == "Alpha"
    assert title["source_record_sha256"] == sha(item["original_record"])
    item["source_view"]["routed"]["metadata"]["title"] = "gold-only alias"
    with pytest.raises(ValueError, match="original source"):
        build_title_context(item)

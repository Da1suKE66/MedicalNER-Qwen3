"""Non-JSON semantic preference admission and exact constrained token paths."""
from copy import deepcopy
import hashlib
import json

import pytest

from generative.choice_data import compile_recovery_row, encode_row
from generative.choice_preferences import (
    EXCLUDED_KINDS, build, compile_preference, encode_preference_row, validate_preference_row,
)
from generative.recovery_preferences import MORE, FEWER, EQUAL, build as build_source, make_case
from kg_agent.choice_protocol import ChoiceProtocol


class Tokenizer:
    eos_token_id = 0
    all_special_ids = [0]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<chat>" + messages[0]["content"] + "</think>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text.startswith("<chat>"):
            return [700, 701, 702]
        # Deliberately forbid whole-completion tokenization. Production must
        # independently encode each integer/newline before concatenating IDs.
        assert text.count("\n") == 1 and text.endswith("\n") and text[:-1].isdigit()
        return [ord(character) for character in text]


@pytest.mark.parametrize("kind", MORE + FEWER + EQUAL)
@pytest.mark.parametrize("size", [2, 4, 8])
def test_each_semantic_kind_has_distinct_legal_targets_and_exact_chosen_sft(kind, size):
    source = make_case(1, kind, pairs_per_batch=size, template=4)
    row = compile_preference(source)
    task = validate_preference_row(row)
    assert row["chosen_actions"] != row["rejected_actions"]
    assert task.choices(row["chosen_actions"]) is None
    assert task.choices(row["rejected_actions"]) is None
    assert task.edges(row["chosen_actions"]) != task.edges(row["rejected_actions"])
    assert row["chosen_actions"] == compile_recovery_row(source, "relations")["actions"]
    assert len(task.edges(row["chosen_actions"])) == len(source["oracle"]["relations"])
    assert row["admission"]["complete_oracle_recovered"]


@pytest.mark.parametrize("kind", EXCLUDED_KINDS)
def test_structural_or_fabricated_identity_rejects_are_excluded_not_repaired(kind):
    with pytest.raises(ValueError, match="excluded"):
        compile_preference(make_case(1, kind))


def test_completions_use_independent_integer_tokens_and_single_eos():
    source = make_case(1, "wrong_target", pairs_per_batch=8)
    row, tokenizer = compile_preference(source), Tokenizer()
    encoded = encode_preference_row(row, tokenizer, 3072)
    task = validate_preference_row(row)
    codec = ChoiceProtocol(tokenizer, len(encoded["prompt_ids"]), task.choices)
    for side in ("chosen", "rejected"):
        ids = encoded[side + "_ids"]
        assert ids == codec.encode_actions(row[side + "_actions"])
        assert ids[-1] == 0 and ids.count(0) == 1
        replay = codec.read_result(ids)
        assert replay.complete and replay.eos_observed and replay.error is None
    sft = encode_row(compile_recovery_row(source, "relations"), tokenizer, 3072)
    assert sft["input_ids"] == encoded["prompt_ids"] + encoded["chosen_ids"]
    assert sft["labels"] == [-100] * len(encoded["prompt_ids"]) + encoded["chosen_ids"]


@pytest.mark.parametrize("field,value", [
    ("prompt", "replacement clinical content"), ("chosen_actions", [0]),
    ("rejected_actions", [0]), ("reason", "different"),
    ("pair_supervision", {"P0001": "UNKNOWN"}),
])
def test_outer_tampering_cannot_enter_training(field, value):
    row = compile_preference(make_case(1, "wrong_target"))
    row[field] = value
    with pytest.raises(ValueError, match="reconstruction"):
        validate_preference_row(row)


def test_source_unknown_and_type_confusion_are_not_admitted():
    source = make_case(1, "wrong_target")
    source["pair_supervision"] = {"P0001": "UNKNOWN"}
    with pytest.raises(ValueError, match="reconstruction"):
        compile_preference(source)
    row = compile_preference(make_case(1, "wrong_target"))
    row["source_row"]["human_adjudicated"] = 0
    with pytest.raises(ValueError, match="typed artificial recipe"):
        validate_preference_row(row)
    row = compile_preference(make_case(1, "wrong_target"))
    row["chosen_actions"][0] = True
    with pytest.raises(ValueError, match="reconstruction"):
        validate_preference_row(row)


def test_holdout_can_be_evaluated_but_never_encoded_as_training():
    row = compile_preference(make_case(1, "wrong_target", partition="mechanical_holdout"))
    assert validate_preference_row(row, require_train=False)
    with pytest.raises(ValueError, match="Only train"):
        encode_preference_row(row, Tokenizer(), 3072)
    assert encode_preference_row(row, Tokenizer(), 3072, require_train=False)["chosen_ids"]


def test_max_length_does_not_silently_truncate():
    row = compile_preference(make_case(1, "wrong_target"))
    with pytest.raises(ValueError, match="no silent truncation"):
        encode_preference_row(row, Tokenizer(), 3)


def test_canonical_json_disk_roundtrip_preserves_recipe_world_and_actions(tmp_path):
    original = compile_preference(make_case(1, "wrong_target", partition="mechanical_holdout"))
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(original, sort_keys=True))
    restored = json.loads(path.read_text())
    task = validate_preference_row(restored, require_train=False)
    assert restored["source_row"]["id"] == original["source_row"]["id"]
    assert task.edges(restored["chosen_actions"]) == task.edges(original["chosen_actions"])


def test_release_freezes_balanced_matched128_and_keeps_holdout_disjoint(tmp_path):
    original = tmp_path / "original"
    build_source(original)
    out = tmp_path / "release"
    manifest = build(original, out, tokenizer=Tokenizer())
    assert manifest["audits"]["train"]["rows"] == 128
    assert manifest["audits"]["train_all_semantic"]["rows"] == 192
    assert manifest["audits"]["mechanical_holdout"]["rows"] == 24
    assert manifest["audits"]["train"]["edge_count_direction"] == {"chosen_more": 48, "chosen_fewer": 48, "equal": 32}
    assert sum(manifest["excluded"]["train"].values()) == 64
    assert sum(manifest["excluded"]["mechanical_holdout"].values()) == 8
    read = lambda name: [json.loads(line) for line in (out / (name + ".jsonl")).read_text().splitlines()]
    rows, sft, held = read("train"), read("sft_chosen"), read("mechanical_holdout_preferences")
    assert not {r["group_id"] for r in rows} & {r["group_id"] for r in held}
    assert not {r["source_row_sha256"] for r in rows} & {r["source_row_sha256"] for r in held}
    assert [(r["prompt"], r["chosen_actions"], r["group_id"]) for r in rows] == [
        (r["prompt"], r["actions"], r["group_id"]) for r in sft]
    for name, metadata in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == metadata["sha256"]
    again = build(original, tmp_path / "repeat", tokenizer=Tokenizer())
    assert manifest == again
    with pytest.raises(ValueError, match="overwrite"):
        build(original, out)
    (original / "train.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="source hash mismatch"):
        build(original, tmp_path / "tampered")

"""Closed-recipe admission and exact non-JSON LM targets; no torch/model needed."""

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from generative.choice_data import compile_recovery_row, encode_row, validate_row
from generative.protocol import exact_hits
from generative.recovery_preferences import KINDS, _construct, as_sft, make_case
from kg_agent.choice_protocol import ChoiceProtocol


class Tokenizer:
    eos_token_id = 999
    all_special_ids = [999]

    def __init__(self):
        self.encoded = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<|im_start|>user\n" + messages[0]["content"] + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.encoded.append(text)
        return list(map(ord, text))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stage", ["entities", "relations"])
def test_all_families_compile_into_complete_actions_and_json_disk_roundtrip(kind, stage):
    source = make_case(0, kind)
    untouched = copy.deepcopy(source)
    row = compile_recovery_row(source, stage)
    row = json.loads(json.dumps(row))
    task = validate_row(row)
    assert task.prompt == row["prompt"] and task.choices(tuple(row["actions"])) is None
    assert row["source_row"] == source == untouched
    assert row["split"] == "train" and row["stage"] == stage
    assert "completion" not in row and all(type(action) is int for action in row["actions"])
    assert not row["admission"]["unknown_supervision_allowed"]
    assert row["admission"]["medical_source_records"] == 0
    if stage == "relations":
        _, batch, chosen = _construct(source["construction_recipe"])
        expected = [{"pair_id": d["pair_id"], **edge} for d in chosen["decisions"] for edge in d["relations"]]
        key = lambda edge: (edge["pair_id"], edge["relation"], tuple(edge["evidence_span_ids"]))
        assert sorted(task.edges(row["actions"]), key=key) == sorted(expected, key=key)
        assert row["task_input"]["batch"]["pairs"] == [{k: p[k] for k in ("pair_id", "source", "target", "allowed")} for p in batch["pairs"]]
        assert {e["id"] for e in row["task_input"]["batch"]["entities"]} == {e["id"] for e in batch["entities"]}


def test_entity_targets_include_all_repeated_source_mentions_with_exact_offsets():
    source = make_case(0, "omit_one_edge")
    row = compile_recovery_row(source, "entities")
    task = validate_row(row)
    expected = set()
    for entity in source["oracle"]["entities"]:
        for span in row["task_input"]["spans"]:
            for start, end in exact_hits(entity["name"], span["text"]):
                expected.add((f"{span['span_id']}:{start}-{end}", span["text"][start:end], entity["label"]))
    observed = {(candidate.span_id, candidate.name, candidate.label) for candidate in task.candidates(row["actions"])}
    assert observed == expected and len(observed) > len(source["oracle"]["entities"])
    assert all(candidate.importance == "associated" for candidate in task.candidates(row["actions"]))
    assert row["admission"]["entity_importance_policy"] == "constant_associated_not_clinical_annotation"


@pytest.mark.parametrize("stage", ["entities", "relations"])
def test_encoder_masks_only_prompt_and_supervises_exact_integer_actions_plus_one_eos(stage):
    source = make_case(0, "terminal_bracket_reversal")
    row = compile_recovery_row(source, stage)
    tokenizer = Tokenizer()
    encoded = encode_row(row, tokenizer, 100000)
    counts = encoded["_choice_counts"]
    target = [token for action in row["actions"] for token in map(ord, f"{action}\n")] + [999]
    assert encoded["labels"] == [-100] * counts["prompt_tokens"] + target
    assert encoded["input_ids"][counts["prompt_tokens"]:] == target
    assert encoded["attention_mask"] == [1] * len(encoded["labels"])
    assert counts == {"prompt_tokens": len(encoded["input_ids"]) - len(target),
        "completion_tokens": len(target), "action_count": len(row["actions"]), "stage": stage}
    assert source["rejected"] not in tokenizer.encoded and source["chosen"] not in tokenizer.encoded
    assert all(text == str(int(text[:-1])) + "\n" for text in tokenizer.encoded[1:])
    task = validate_row(row)
    protocol = ChoiceProtocol(tokenizer, counts["prompt_tokens"], task.choices)
    prefix = encoded["input_ids"][:counts["prompt_tokens"]]
    for token in target:
        assert token in protocol.prefix_allowed_tokens_fn(0, prefix)
        prefix.append(token)
    assert protocol.read_result(target).complete


@pytest.mark.parametrize("stage", ["entities", "relations"])
def test_max_length_fails_instead_of_silently_truncating_actions_or_source(stage):
    row = compile_recovery_row(make_case(0, "omit_second_relation"), stage)
    n = len(encode_row(row, Tokenizer(), 100000)["input_ids"])
    assert len(encode_row(row, Tokenizer(), n)["input_ids"]) == n
    with pytest.raises(ValueError, match="no silent truncation"):
        encode_row(row, Tokenizer(), n - 1)


@pytest.mark.parametrize("stage", ["entities", "relations"])
def test_holdout_cannot_be_laundered_into_training(stage):
    source = make_case(0, "wrong_target", partition="mechanical_holdout")
    row = compile_recovery_row(source, stage)
    assert row["split"] == "mechanical_holdout"
    with pytest.raises(ValueError, match="Only train"):
        encode_row(row, Tokenizer(), 100000)
    assert encode_row(row, Tokenizer(), 100000, require_train=False)
    changed = {**row, "split": "train"}
    with pytest.raises(ValueError, match="reconstruction"):
        encode_row(changed, Tokenizer(), 100000)
    source["split"] = "train"
    with pytest.raises(ValueError):
        compile_recovery_row(source, stage)


@pytest.mark.parametrize("stage", ["entities", "relations"])
@pytest.mark.parametrize("field,value", [
    ("loss_mask_contract", "pair-unknown-loss-mask-v1"),
    ("loss_mask_spans", [[0, 1]]),
    ("pair_supervision", {"P0001": "UNKNOWN_UNANNOTATED"}),
    ("unknown_reasons", {"P0001": "silver_absence"}),
    ("data_version", "teacher-silver-repair-v2"),
])
def test_partial_or_unknown_rows_cannot_change_format_to_bypass_admission(stage, field, value):
    source = {**make_case(0, "omit_one_edge"), field: value}
    with pytest.raises(ValueError):
        compile_recovery_row(source, stage)


@pytest.mark.parametrize("field", ["actions", "prompt", "task_input", "source_row", "source_row_sha256", "admission", "split", "extra"])
def test_every_compiled_field_is_checked_against_reconstruction(field):
    row = compile_recovery_row(make_case(0, "wrong_target_type"), "relations")
    changed = copy.deepcopy(row)
    if field == "actions":
        changed[field] = [0]
    elif field == "task_input":
        changed[field]["batch"]["entities"][0]["name"] = "changed external text"
    elif field == "source_row":
        changed[field]["oracle_source_text"] += " external text"
    elif field == "admission":
        changed[field]["medical_source_records"] = False  # Python False == 0 must not bypass typed equality.
    else:
        changed[field] = "tampered"
    with pytest.raises(ValueError):
        encode_row(changed, Tokenizer(), 100000)


def test_typed_action_boolean_is_not_equal_to_integer_one_for_admission():
    row = compile_recovery_row(make_case(0, "omit_one_edge"), "relations")
    row["actions"][0] = True
    with pytest.raises(ValueError, match="reconstruction"):
        encode_row(row, Tokenizer(), 100000)


def test_both_validated_preference_and_chosen_sft_source_views_keep_same_targets():
    source = make_case(0, "fabricated_pair_id")
    for stage in ("entities", "relations"):
        a, b = [compile_recovery_row(view, stage) for view in (source, as_sft(source))]
        assert a["actions"] == b["actions"] and a["prompt"] == b["prompt"]
        assert a["source_row_sha256"] != b["source_row_sha256"]
        validate_row(a)
        validate_row(b)


def test_unverified_real_silver_source_and_bad_stage_are_rejected():
    with pytest.raises(ValueError):
        compile_recovery_row({"split": "train", "oracle_source_text": "unverified", "oracle": {"entities": []}}, "entities")
    with pytest.raises(ValueError, match="stage"):
        compile_recovery_row(make_case(0, "wrong_target"), "entity_json")

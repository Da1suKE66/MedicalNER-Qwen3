"""Literal semantic labels remain causal-LM text, with recipe-verified supervision."""
from copy import deepcopy
import json

import pytest

from generative.label_data import (
    COMPLETION_ENCODING, DATA_VERSION, compile_recovery_row, encode_row, validate_row,
)
from generative.recovery_preferences import KINDS, as_sft, make_case
from kg_agent.label_backend import CONTRACT
from train_choices import _exposure_delta


class Tokenizer:
    eos_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<chat>" + messages[0]["content"] + "</think>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(character) + 1 for character in text]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("size", [2, 4, 8])
def test_recipe_chosen_targets_are_full_literal_semantic_records(kind, size):
    source = make_case(1, kind, pairs_per_batch=size, template=4)
    row = compile_recovery_row(source)
    task = validate_row(row)
    assert row["data_version"] == DATA_VERSION and row["output_contract"] == CONTRACT
    assert "actions" not in row and "chosen_actions" not in row
    assert task.parse(row["completion"]) == row["target"]
    assert len(row["target"]) == len(source["oracle"]["relations"])
    assert row["admission"]["complete_oracle_recovered"]
    local_ids = {entity["entity_id"] for entity in row["task_input"]["batch"]["entities"]}
    labels = {edge["relation"] for edge in row["target"]}
    visible = {span["span_id"] for span in row["task_input"]["batch"]["spans"]}
    for line in row["completion"].splitlines():
        fields = line.split("\t")
        assert len(fields) == 4
        source_id, label, target_id, evidence = fields
        assert source_id in local_ids and target_id in local_ids
        assert label in labels and not label.isdigit() and label != "NONE"
        assert set(evidence.split(",")) <= visible
    if not row["target"]:
        assert row["completion"] == ""


def test_complete_completion_is_tokenized_once_with_eos_and_prompt_mask():
    row = compile_recovery_row(make_case(1, "omit_second_relation"))
    tokenizer = Tokenizer()
    encoded = encode_row(row, tokenizer, 100000)
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[1] == row["completion"]
    assert "\t" in tokenizer.calls[1] and len(tokenizer.calls[1].splitlines()) == 2
    prompt_length = len(tokenizer.calls[0])
    target_ids = [ord(character) + 1 for character in row["completion"]] + [0]
    assert encoded["input_ids"][prompt_length:] == target_ids
    assert encoded["labels"] == [-100] * prompt_length + target_ids
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])
    metadata = encoded["_choice_counts"]
    assert metadata["action_count"] == 2
    assert metadata["action_count_unit"] == "relation_records"
    assert metadata["completion_encoding"] == COMPLETION_ENCODING
    exposure = _exposure_delta([metadata])
    assert exposure["relations_rows"] == 1 and exposure["action_count"] == 2


def test_verified_all_negative_target_supervises_only_eos():
    row = compile_recovery_row(make_case(0, "all_negative_hallucination"))
    tokenizer = Tokenizer()
    encoded = encode_row(row, tokenizer, 100000)
    assert tokenizer.calls[1] == ""
    assert encoded["labels"].count(0) == 1
    assert set(encoded["labels"][:-1]) == {-100}
    assert encoded["_choice_counts"]["completion_tokens"] == 1
    assert encoded["_choice_counts"]["action_count"] == 0


def test_multirelation_gold_is_retained_without_numeric_label_indices():
    source = make_case(1, "omit_second_relation")
    row = compile_recovery_row(source)
    assert len(row["completion"].splitlines()) == 2
    assert "is_core_symptom_of" in row["completion"]
    assert "supports_diagnosis_of" in row["completion"]
    reversed_lines = "\n".join(reversed(row["completion"].splitlines()))
    task = validate_row(row)
    signature = lambda edges: {(e["pair_id"], e["relation"], tuple(e["evidence_span_ids"])) for e in edges}
    assert signature(task.parse(reversed_lines)) == signature(row["target"])


def test_heldout_is_evaluation_only_and_stage_must_be_relations():
    source = make_case(1, "wrong_target", partition="mechanical_holdout")
    row = compile_recovery_row(source)
    assert validate_row(row, require_train=False)
    assert encode_row(row, Tokenizer(), 100000, require_train=False)
    with pytest.raises(ValueError, match="Only train"):
        encode_row(row, Tokenizer(), 100000)
    with pytest.raises(ValueError, match="only the relations stage"):
        compile_recovery_row(source, "entities")


@pytest.mark.parametrize("field,value", [
    ("completion", "E001\t1\tE002\tSENT_001"),
    ("target", []), ("prompt", "replacement input"), ("output_contract", "old-numeric-contract"),
    ("pair_supervision", {"P0001": "UNKNOWN"}), ("loss_mask_spans", []),
])
def test_target_prompt_or_unknown_metadata_tampering_fails_reconstruction(field, value):
    row = compile_recovery_row(make_case(1, "wrong_target"))
    row[field] = value
    with pytest.raises(ValueError, match="reconstruction"):
        validate_row(row)


@pytest.mark.parametrize("field,value", [
    ("pair_supervision", {"P0001": "UNKNOWN"}), ("medical_source_records", 1),
    ("provenance", "clinical_silver"), ("oracle_source_text", "arbitrary clinical source"),
])
def test_partial_or_clinical_source_cannot_be_admitted(field, value):
    source = make_case(1, "wrong_target")
    source[field] = value
    with pytest.raises(ValueError, match="reconstruction"):
        compile_recovery_row(source)


def test_canonical_disk_roundtrip_preserves_source_recipe_and_target(tmp_path):
    row = compile_recovery_row(make_case(1, "wrong_evidence"))
    path = tmp_path / "row.json"
    path.write_text(json.dumps(row, sort_keys=True))
    restored = json.loads(path.read_text())
    assert validate_row(restored).parse(restored["completion"]) == row["target"]
    changed = deepcopy(restored)
    changed["source_row"]["human_adjudicated"] = 0
    with pytest.raises(ValueError, match="typed artificial recipe"):
        validate_row(changed)


def test_existing_sft_source_provenance_is_reconstructed_without_rejected_training():
    source = as_sft(make_case(1, "terminal_bracket_reversal"))
    row = compile_recovery_row(source)
    tokenizer = Tokenizer()
    encode_row(row, tokenizer, 100000)
    assert source["rejected"] not in tokenizer.calls
    assert source["chosen"] not in tokenizer.calls
    assert tokenizer.calls[1] == row["completion"]


def test_overlength_row_is_rejected_without_truncating_targets():
    row = compile_recovery_row(make_case(1, "wrong_target"))
    with pytest.raises(ValueError, match="no silent truncation"):
        encode_row(row, Tokenizer(), 5)


@pytest.mark.parametrize("eos", [None, True, -1, 1.5])
def test_invalid_eos_is_rejected(eos):
    row = compile_recovery_row(make_case(1, "wrong_target"))
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = eos
    with pytest.raises(ValueError, match="valid EOS"):
        encode_row(row, tokenizer, 100000)


def test_embedded_eos_cannot_create_premature_completion():
    row = compile_recovery_row(make_case(1, "wrong_target"))
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = ord("\t") + 1
    with pytest.raises(ValueError, match="contains EOS"):
        encode_row(row, tokenizer, 100000)

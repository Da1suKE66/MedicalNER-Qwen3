"""CPU-only continuation identity and narrowly scoped recovery-data admission."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative import train
from generative.recovery_preferences import KINDS, STRUCTURAL, as_sft, make_case


def arguments(tmp_path, *extra):
    return train.parse_args([
        "--mode", "sft", "--data", str(tmp_path / "data.jsonl"),
        "--base-model", str(tmp_path / "base"), "--output-dir", str(tmp_path / "out"),
        "--sft-adapter", str(tmp_path / "parent" / "final"), *extra,
    ])


def receipt(tmp_path, **overrides):
    path = tmp_path / "parent" / "run_manifest.json"
    path.parent.mkdir(exist_ok=True)
    payload = {"status": "training_complete_evaluation_pending", "global_step": 64,
        "arguments": {"mode": "sft", "base_model": str(tmp_path / "base")},
        "policy_adapter_sha256_after": "a" * 64, "data_sha256": "b" * 64, **overrides}
    path.write_text(json.dumps(payload))
    return path


def test_continuation_auto_requires_reload_and_records_actual_parent(tmp_path):
    args = arguments(tmp_path)
    assert args.verify_save
    with pytest.raises(FileNotFoundError):
        train.starting_policy_receipt(args)
    path = receipt(tmp_path)
    parent = train.starting_policy_receipt(args)
    assert parent["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert parent["global_step"] == 64
    assert not parent["initialization_verified"] and not parent["optimizer_state_resumed"]


@pytest.mark.parametrize("change", [
    {"status": "running"}, {"policy_adapter_sha256_after": None},
    {"policy_adapter_sha256_after": "x" * 64},
    {"arguments": {"mode": "dpo", "base_model": "base"}},
    {"arguments": {"mode": "sft", "base_model": "another-base"}},
])
def test_continuation_rejects_unfinished_wrong_parent_or_base(tmp_path, change):
    receipt(tmp_path, **change)
    with pytest.raises(ValueError):
        train.starting_policy_receipt(arguments(tmp_path))


def test_existing_weights_are_loaded_trainable_and_must_match_parent(monkeypatch):
    base, loaded, loader = object(), Mock(), Mock()
    loader.from_pretrained.return_value = loaded
    monkeypatch.setattr(train, "adapter_fingerprint", lambda model, active: "actual")
    parent = {"policy_adapter_sha256": "actual"}
    assert train.initialize_continued_sft(base, "grounded64/final", parent, loader) is loaded
    loader.from_pretrained.assert_called_once_with(base, "grounded64/final", adapter_name="default",
        is_trainable=True, local_files_only=True)
    loaded.set_adapter.assert_called_once_with("default")
    assert parent["initialization_verified"]
    with pytest.raises(RuntimeError, match="parent receipt fingerprint"):
        train.initialize_continued_sft(base, "wrong/final", {"policy_adapter_sha256": "wrong"}, loader)
    with pytest.raises(ValueError, match="parent receipt"):
        train.initialize_continued_sft(base, "anything", None, loader)


def test_unchanged_continuation_cannot_be_marked_complete():
    with pytest.raises(RuntimeError, match="did not update"):
        train.require_policy_update("same", "same")
    train.require_policy_update("parent", "updated")


def test_sft_dropout_match_is_explicit_and_never_changes_other_modules(tmp_path):
    class Dropout:
        def __init__(self, p):
            self.p = p
    modules = [Dropout(.05), Dropout(0), SimpleNamespace(p=.25)]
    model = SimpleNamespace(modules=lambda: modules)
    assert train.disable_sft_dropout(model, Dropout) == {
        "disabled": True, "dropout_modules": 2, "nonzero_before": 1}
    assert [m.p for m in modules] == [0, 0, .25]
    assert not arguments(tmp_path).sft_disable_dropout
    assert arguments(tmp_path, "--sft-disable-dropout").sft_disable_dropout
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--mode", "dpo", "--sft-disable-dropout")


def test_dpo_receipt_is_optional_and_legacy_initialization_is_unchanged(tmp_path):
    args = SimpleNamespace(mode="dpo", sft_adapter="parent", parent_receipt=None)
    assert train.starting_policy_receipt(args) is None
    with pytest.raises(SystemExit):
        arguments(tmp_path, "--mode", "grpo", "--parent-receipt", "parent.json")


@pytest.mark.parametrize("kind", KINDS)
def test_recipe_verified_preferences_and_matching_chosen_sft_are_admitted(kind):
    row = make_case(0, kind)
    original = copy.deepcopy(row)
    train.validate_training_rows([row], "dpo")
    sft = as_sft(row)
    train.validate_training_rows([sft], "sft")
    assert sft["completion"] == row["chosen"]
    assert sft["prompt"] == row["prompt"] and row == original
    with pytest.raises(ValueError, match="training view"):
        train.validate_training_rows([row], "sft")
    with pytest.raises(ValueError, match="training view"):
        train.validate_training_rows([sft], "dpo")


@pytest.mark.parametrize("kind", STRUCTURAL)
def test_structural_rejected_allowance_cannot_be_reused_for_generic_data(kind):
    row = make_case(0, kind)
    plain = {k: v for k, v in row.items() if k != "preference_contract"}
    with pytest.raises(ValueError, match="Invalid rejected"):
        train.validate_training_rows([plain], "dpo")
    for field, value in (("chosen", "{"), ("rejected", "{}"), ("prompt", row["prompt"] + "unexpected")):
        changed = {**row, field: value}
        changed[field + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        with pytest.raises(ValueError):
            train.validate_training_rows([changed], "dpo")


@pytest.mark.parametrize("mode", ["sft", "dpo"])
def test_recovery_never_bypasses_unknown_or_heldout_gate(mode):
    row = make_case(0, "terminal_bracket_reversal")
    if mode == "sft":
        row = as_sft(row)
    for split in ("mechanical_holdout", "tune", "regression"):
        with pytest.raises(ValueError, match="train-tagged"):
            train.validate_training_rows([{**row, "split": split}], mode)
    with pytest.raises(ValueError):
        train.validate_training_rows([{**row, "pair_supervision": {"P0001": "UNKNOWN_UNANNOTATED"}}], mode)
    with pytest.raises(ValueError):
        train.validate_training_rows([{**row, "unknown_reasons": {"P0001": "unannotated"}}], mode)
    if mode == "sft":
        with pytest.raises(ValueError):
            train.validate_training_rows([{**row, "completion": "{}"}], mode)


def test_same_chosen_control_alignment_is_audited_without_rejected_training():
    rows = [make_case(0, kind) for kind in KINDS]
    sft = [as_sft(row) for row in rows]
    a, b = train.recovery_admission_audit(rows), train.recovery_admission_audit(sft)
    assert a["prompt_chosen_alignment_sha256"] == b["prompt_chosen_alignment_sha256"]
    assert a["structural_rejected_rows"] == b["structural_rejected_rows"] == 4
    assert b["training_view"] == "chosen_sft"
    with pytest.raises(ValueError, match="Do not mix"):
        train.recovery_admission_audit([rows[0], {"id": "legacy"}])


def test_sft_control_dataset_never_tokenizes_rejected_receipt(monkeypatch):
    from train_sft import JsonlSFTDataset
    # Dataset construction stores the torch reference but creates Python lists;
    # the collator/model are deliberately not exercised by this CPU-only test.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    tokenized = []
    class Tokenizer:
        eos_token_id = 99999
        def __call__(self, text, **kwargs):
            tokenized.append(text)
            return {"input_ids": list(map(ord, text))}
    row = as_sft(make_case(0, "terminal_bracket_reversal"))
    item = JsonlSFTDataset([row], Tokenizer(), 20000)[0]
    assert tokenized == [row["prompt"], row["chosen"]]
    assert item["labels"] == [-100] * len(row["prompt"]) + list(map(ord, row["chosen"])) + [99999]

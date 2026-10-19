"""Legacy replay must not bypass current relation supervision admission."""

import builtins
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_sft import main, validate_legacy_training_rows, validate_output_directory


def legacy_row(stage="relations", **extra):
    return {
        "id": "synthetic:1",
        "stage": stage,
        "prompt": "Return the requested synthetic stage JSON.",
        "completion": '{"relations":[]}',
        **extra,
    }


@pytest.mark.parametrize("stage", ["entities", "relations"])
def test_historical_stage_exports_remain_admissible(stage):
    validate_legacy_training_rows([legacy_row(stage)], stage)
    validate_legacy_training_rows([legacy_row(stage, split="train")], stage)


@pytest.mark.parametrize("split", ["tune", "dev", "test", "validation", None])
def test_explicit_held_out_split_is_rejected(split):
    with pytest.raises(ValueError, match="split='train'"):
        validate_legacy_training_rows([legacy_row(split=split)], "relations")


@pytest.mark.parametrize("fields", [
    {"data_version": "teacher-silver-repair-v2"},
    {"pair_supervision": {"P1": "UNKNOWN_UNANNOTATED"}},
    {"loss_mask_contract": "pair-unknown-loss-mask-v1"},
    {"loss_mask_spans": []},
    {"pairs": []},
    {"target": {"decisions": []}},
    {"output_contract": "sparse-relations"},
])
def test_current_relation_protocol_cannot_use_legacy_entry(fields):
    with pytest.raises(ValueError, match="generative.train"):
        validate_legacy_training_rows([legacy_row(**fields)], "relations")


def test_stage_mismatch_and_missing_stage_are_rejected():
    with pytest.raises(ValueError, match="stage='relations'"):
        validate_legacy_training_rows([legacy_row("entities")], "relations")
    row = legacy_row()
    row.pop("stage")
    with pytest.raises(ValueError, match="generative.train"):
        validate_legacy_training_rows([row], "relations")


def test_chat_rendered_data_cannot_silently_change_legacy_replay():
    with pytest.raises(ValueError, match="raw prompts"):
        validate_legacy_training_rows([legacy_row(prompt="<|im_start|>user\nexample")], "relations")


def test_output_requires_new_or_empty_directory(tmp_path):
    validate_output_directory(tmp_path / "new")
    empty = tmp_path / "empty"
    empty.mkdir()
    validate_output_directory(empty)
    checkpoint = empty / "adapter_model.safetensors"
    checkpoint.write_bytes(b"synthetic preserved checkpoint")
    with pytest.raises(ValueError, match="never overwrite"):
        validate_output_directory(empty)
    with pytest.raises(ValueError, match="never overwrite"):
        validate_output_directory(checkpoint)
    assert checkpoint.read_bytes() == b"synthetic preserved checkpoint"


@pytest.mark.parametrize("failure", ["partial", "held_out", "overwrite"])
def test_cli_fails_before_importing_ml_runtime_or_writing_output(tmp_path, monkeypatch, failure):
    output = tmp_path / "output"
    row = legacy_row()
    if failure == "partial":
        row["pair_supervision"] = {"P1": "SUPPORTED_SILVER"}
    elif failure == "held_out":
        row["split"] = "test"
    else:
        output.mkdir()
        (output / "existing.txt").write_text("preserve")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "train_sft.py", "--stage", "relations", "--data", str(data),
        "--base-model", "must-not-load", "--output-dir", str(output),
    ])
    real_import = builtins.__import__

    def reject_ml_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "peft", "transformers"}:
            raise AssertionError("GPU runtime imported before admission")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ml_import)
    with pytest.raises(ValueError):
        main()
    if failure == "overwrite":
        assert (output / "existing.txt").read_text() == "preserve"
    else:
        assert not output.exists()

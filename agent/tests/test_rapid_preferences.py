"""CPU contracts for synthetic RL screens; these tests claim no model quality."""

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative import rapid_preferences as rapid
from generative.protocol import digest
from generative.train import validate_training_rows


@pytest.mark.parametrize("kind", rapid.KINDS)
def test_each_mutation_is_valid_and_has_positive_mechanical_reward_margin(kind):
    row, record = rapid.make_case(0, kind)
    validate_training_rows([row], "dpo")
    result = rapid.verify_case(row)
    assert result["reward_margin"] > 0
    assert row["human_adjudicated"] is False
    assert row["semantic_claim"] == "mechanical_extraction_only"
    assert json.loads(record["messages"][1]["content"]) == row["oracle"]
    assert record["messages"][0]["content"] == row["oracle_source_text"]


def test_preference_directions_and_errors_do_not_only_reward_length():
    rows = [rapid.make_case(0, kind)[0] for kind in rapid.KINDS]
    result = rapid.summarize(rows)
    assert result["edge_count_direction"] == {"chosen_more": 2, "chosen_fewer": 2, "equal": 4}
    assert result["character_length_direction"] == {"chosen_longer": 2, "chosen_shorter": 3, "equal": 3}
    assert result["reasons"] == dict(Counter(rapid.KINDS))
    assert result["medical_source_records"] == 0


def test_mechanical_holdout_cannot_pass_training_admission():
    train, _ = rapid.make_case(0, rapid.KINDS[0])
    held, _ = rapid.make_case(0, rapid.KINDS[0], partition="mechanical_holdout")
    assert train["group_id"] != held["group_id"]
    assert train["prompt_sha256"] != held["prompt_sha256"]
    with pytest.raises(ValueError, match="train-tagged"):
        validate_training_rows([held], "dpo")


def test_tampered_chosen_is_rejected_even_if_hash_resigned():
    row, _ = rapid.make_case(0, "omit_supported_edge")
    row["chosen"] = row["rejected"]
    row["chosen_sha256"] = digest(row["chosen"])
    with pytest.raises(ValueError, match="authored exhaustive oracle"):
        rapid.verify_case(row)


@pytest.mark.parametrize("field", ["prompt", "chosen", "rejected"])
def test_hashes_cover_every_visible_training_field(field):
    row, _ = rapid.make_case(0, rapid.KINDS[0])
    row[f"{field}_sha256"] = "invalid"
    with pytest.raises(ValueError, match="hash mismatch"):
        rapid.verify_case(row)


def test_medical_or_human_relabelling_is_rejected():
    row, _ = rapid.make_case(0, rapid.KINDS[0])
    with pytest.raises(ValueError, match="fully artificial"):
        rapid.verify_case({**row, "provenance": "real_medical_gold"})
    with pytest.raises(ValueError, match="medically validated"):
        rapid.verify_case({**row, "human_adjudicated": True})


def test_release_is_deterministic_hashed_and_refuses_overwrite(tmp_path):
    a = rapid.build(tmp_path / "a", repeats=1, holdout_repeats=1)
    b = rapid.build(tmp_path / "b", repeats=1, holdout_repeats=1)
    assert a == b
    assert a["train"]["rows"] == 8
    assert a["mechanical_holdout"]["rows"] == 8
    for name, info in a["files"].items():
        assert info["sha256"] == rapid._sha(tmp_path / "a" / name)
    with pytest.raises(ValueError, match="refuse to overwrite"):
        rapid.build(tmp_path / "a", repeats=1, holdout_repeats=1)


@pytest.mark.parametrize("index,kind,partition", [
    (-1, rapid.KINDS[0], "train"), (True, rapid.KINDS[0], "train"),
    (0, "arbitrary", "train"), (0, rapid.KINDS[0], "clinical_gold"),
])
def test_invalid_construction_inputs_fail(index, kind, partition):
    with pytest.raises(ValueError):
        rapid.make_case(index, kind, partition=partition)


def test_extra_edge_and_missing_edge_controls_change_exactly_one_count():
    for kind, expected in [("omit_supported_edge", 1), ("omit_second_relation", 1),
                           ("add_unsupported_edge", -1), ("contradict_explicit_negation", -1)]:
        result = rapid.verify_case(rapid.make_case(0, kind)[0])
        assert result["chosen_edges"] - result["rejected_edges"] == expected


def test_wrong_evidence_changes_only_citations():
    row, _ = rapid.make_case(0, "wrong_evidence")
    a, b = [json.loads(row[k]) for k in ("chosen", "rejected")]
    assert a != b
    for obj in (a, b):
        for decision in obj["decisions"]:
            for edge in decision["relations"]:
                edge.pop("evidence_span_ids")
    assert a == b


@pytest.mark.parametrize("mismatch", [None, "weights", "logits"])
def test_zero_step_control_exports_and_rejects_reload_mismatch(tmp_path, monkeypatch, mismatch):
    """Mock only runtime; check zero optimization and persisted failure states."""
    from generative import evaluate

    class Tensor:
        def __init__(self, value=0.0):
            self.value = value

        def __getitem__(self, _):
            return self

        def float(self):
            return self

        cpu = abs = float

        def __sub__(self, other):
            return Tensor(abs(self.value - other.value))

        def max(self):
            return self.value

    class Inputs(dict):
        def to(self, _):
            return self

    class Model:
        device = "cuda"

        def __init__(self, value=0.0):
            self.value = value
            self.loaded_from = None

        def set_adapter(self, name):
            assert name == "policy"

        def to(self, _):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=Tensor(self.value))

        def save_pretrained(self, path, selected_adapters):
            assert selected_adapters == ["policy"]
            (path / "policy").mkdir(parents=True)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "<|im_start|>user\n" + messages[0]["content"] + "\n</think>\n"

        def __call__(self, *args, **kwargs):
            return Inputs(input_ids=Tensor())

        def save_pretrained(self, path):
            pass

    model = Model()
    loader = Mock(return_value=model)
    peft_loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="mock_bf16", inference_mode=nullcontext,
        autocast=lambda *args, **kwargs: nullcontext(),
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True, empty_cache=lambda: None),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=peft_loader)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
    ))
    monkeypatch.setattr(rapid.importlib.metadata, "version", lambda name: {"peft": "0.17.1", "transformers": "4.56.2"}[name])
    initialize = Mock(return_value={"initial_adapters_bitwise_equal": True})
    monkeypatch.setattr(rapid, "initialize_dpo_reference", initialize)
    monkeypatch.setattr(rapid, "adapter_fingerprint", lambda model, name: "initial_exact")
    eval_loader = Mock(return_value=Model(1.0 if mismatch == "logits" else 0.0))
    monkeypatch.setattr(evaluate, "load_evaluation_model", eval_loader)

    def require_equal(model, expected):
        assert expected == "initial_exact"
        if mismatch == "weights":
            raise RuntimeError("Saved/reloaded adapter differs")
        return expected

    monkeypatch.setattr(rapid, "require_saved_weight_equality", require_equal)
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(rapid.make_case(0, "wrong_target")[0]) + "\n")
    out = tmp_path / "zero"
    if mismatch:
        with pytest.raises(RuntimeError):
            rapid.export_initial("base", "sft64", data, out)
    else:
        rapid.export_initial("base", "sft64", data, out)
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["global_step"] == 0 and manifest["optimizer_created"] is False
    assert manifest["status"] == ("failed" if mismatch else "zero_step_initial_policy_exported_verified")
    assert peft_loader.call_args.kwargs == {"adapter_name": "policy", "is_trainable": True, "local_files_only": True}
    initialize.assert_called_once_with(model, "sft64")
    eval_loader.assert_called_once_with("base", str(out / "final" / "policy"))

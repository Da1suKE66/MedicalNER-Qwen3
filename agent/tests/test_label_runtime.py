"""Literal SFT and runtime wiring, without claiming mock runs train a model."""

from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest

import train_label_relations
from kg_agent.choice_backend import ChoiceFailure
from kg_agent.label_backend import CONTRACT, LabelGenerator, check_adapter_contract
from kg_agent.relation_labels import RelationLabelTask
from test_train_choices import arguments, runtime
from test_choice_backend import FakeGenerationConfig, FakeInputs, FakeTensor, FakeTokenizer


def task():
    return RelationLabelTask({
        "entities": [{"id": "S", "entity_id": "E1", "label": "Symptom", "name": "Fatigue"},
                     {"id": "D", "entity_id": "E2", "label": "Disease", "name": "Condition"}],
        "pairs": [{"pair_id": "P1", "source": "S", "target": "D",
                   "allowed": ["is_core_symptom_of", "supports_diagnosis_of"]}],
        "spans": [{"span_id": "SENT_1", "text": "Fatigue is a core symptom of Condition."}],
    })


def test_literal_training_entry_uses_text_and_writes_distinct_verified_receipt(tmp_path, monkeypatch, runtime):
    _, out = arguments(monkeypatch, tmp_path)
    train_label_relations.main()
    report = json.loads((out / "run_manifest.json").read_text())
    assert report["output_contract"] == CONTRACT
    assert report["stages"] == ["relations"] and report["training_rows"] == 1
    assert report["completion_encoding"] == "literal_relation_lines_then_eos"
    assert report["observed_exposure"]["relations_rows"] == 8
    assert report["saved_reload_exact"] and report["policy_weights_changed"]
    assert any("\tis_core_symptom_of\t" in value for value in runtime.tokenizer.encoded_texts)
    assert not (out / "final/choice_contract.json").exists()
    receipt = check_adapter_contract(out / "final", base_model=tmp_path / "base")
    assert receipt["output_contract"] == CONTRACT and not receipt["medical_quality_evaluated"]
    with pytest.raises(ValueError, match="literal relation"):
        check_adapter_contract(out / "final", "entities")


def test_literal_training_preflight_does_not_load_weights(tmp_path, monkeypatch, runtime):
    _, out = arguments(monkeypatch, tmp_path, None, "--preflight-only")
    train_label_relations.main()
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "preflight_complete" and manifest["stages"] == ["relations"]
    runtime.base_loader.assert_not_called()
    assert not runtime.trainers


def test_label_entry_rejects_unsupported_entity_training(tmp_path, monkeypatch, runtime):
    _, out = arguments(monkeypatch, tmp_path, None, "--stages", "entities", "relations")
    with pytest.raises(SystemExit):
        train_label_relations.main()
    assert not out.exists()


def fake_backend(monkeypatch, text, *, eos=True):
    class Tokenizer(FakeTokenizer):
        def decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            return "".join(map(chr, ids))
    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=10000)
        def __init__(self):
            self.calls = []
        def set_adapter(self, name):
            assert name == "relations"
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            assert "prefix_allowed_tokens_fn" not in kwargs
            assert kwargs["use_model_defaults"] is False
            prefix = kwargs["input_ids"].values[0]
            suffix = list(map(ord, text)) + ([900] if eos else [])
            return FakeTensor([prefix + suffix])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=FakeGenerationConfig))
    return SimpleNamespace(tokenizer=Tokenizer(), model=Model(), adapter_name="relations")


def test_generator_emits_literal_labels_and_has_no_numeric_constraint(monkeypatch):
    text = "E1\tsupports_diagnosis_of\tE2\tSENT_1\nE1\tis_core_symptom_of\tE2\tSENT_1\n"
    backend = fake_backend(monkeypatch, text)
    edges, trace = LabelGenerator(backend).run(task())
    assert [edge["relation"] for edge in edges] == ["supports_diagnosis_of", "is_core_symptom_of"]
    assert trace["complete"] and trace["completion_text"] == text
    assert trace["prompt_tokens"] == 8
    config = backend.model.calls[0]["generation_config"]
    assert config.do_sample is False and config.num_beams == 1


def test_legal_looking_budget_prefix_is_not_accepted(monkeypatch):
    text = "E1\tis_core_symptom_of\tE2\tSENT_1\n"
    backend = fake_backend(monkeypatch, text, eos=False)
    with pytest.raises(ChoiceFailure) as error:
        LabelGenerator(backend, max_new_tokens=len(text)).run(task())
    assert error.value.trace["budget_exhausted"] and not error.value.trace["complete"]
    assert error.value.trace["prompt_tokens"] == 8


def test_invalid_generation_keeps_raw_text_and_fails(monkeypatch):
    text = "E1\t1\tE2\tSENT_1\n"
    backend = fake_backend(monkeypatch, text)
    with pytest.raises(ChoiceFailure) as error:
        LabelGenerator(backend).run(task())
    assert error.value.trace["completion_text"] == text
    assert error.value.trace["eos_observed"] and not error.value.trace["complete"]


def test_prompt_budget_is_checked_before_any_generation(monkeypatch):
    backend = fake_backend(monkeypatch, "")
    with pytest.raises(ChoiceFailure, match="no silent truncation"):
        LabelGenerator(backend, max_input_tokens=7).run(task())
    assert backend.model.calls == []

"""CPU scoring-contract tests; no model-quality or GPU claims."""
import contextlib
import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from generative import recovery_preference_eval as module
from generative.recovery_preferences import make_case


class Tokenizer:
    eos_token_id = 999

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[0]["content"] + "</think>\n\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode())


def held(index=0, kind="omit_one_edge"):
    return make_case(index, kind, partition="mechanical_holdout", pairs_per_batch=2)


def test_only_exact_recipe_holdout_preferences_admitted(tmp_path):
    path = tmp_path / "held.jsonl"
    row = held()
    path.write_text(json.dumps(row) + "\n")
    rows, details = module.load_holdout(path)
    assert rows == [row] and details[0]["chosen_edges"] > 0
    path.write_text(json.dumps(make_case(0, "omit_one_edge")) + "\n")
    with pytest.raises(ValueError, match="mechanical_holdout"):
        module.load_holdout(path)
    row["prompt"] += " injected"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="recipe"):
        module.load_holdout(path)


def test_duplicate_holdout_ids_rejected(tmp_path):
    path = tmp_path / "held.jsonl"
    path.write_text((json.dumps(held()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        module.load_holdout(path)


def test_dpo_encoding_renders_once_keeps_prompt_and_adds_one_eos():
    row = {"prompt": "prompt", "chosen": "yes", "rejected": "no"}
    encoded = module.encode_preference(row, Tokenizer(), 100)
    assert bytes(encoded["prompt"]).decode() == "USER:prompt</think>\n\n"
    assert encoded["chosen"] == [121, 101, 115, 999]
    assert encoded["rejected"] == [110, 111, 999]
    exact = len(encoded["prompt"]) + len(encoded["chosen"])
    assert module.encode_preference(row, Tokenizer(), exact) == encoded
    with pytest.raises(ValueError, match="truncation is forbidden"):
        module.encode_preference(row, Tokenizer(), exact - 1)


def test_zero_step_all_ties_are_not_zero_accuracy():
    score = module.score_row("zero", (4, 8), (-6., -9.), (-6., -9.), 1e-6)
    report = module.summarize([score])
    assert report["adjusted"] == {"wins": 0, "ties": 1, "losses": 0,
                                   "decisive_win_rate": None, "tie_fraction": 1.0, "mean_margin": 0.0}
    assert report["raw"]["wins"] == 1
    assert report["length_normalized"]["losses"] == 1


def test_adjusted_margin_subtracts_reference_and_normalization_uses_each_length():
    score = module.score_row("comparison", (2, 4), (-4., -6.), (-5., -5.), 1e-6)
    assert score["raw_margin"] == 2
    assert score["adjusted_margin"] == 2
    assert score["length_normalized_margin"] == -0.5
    assert score["length_normalized_adjusted_margin"] == 0.75
    assert score["adjusted_outcome"] == 1
    assert module.classify(1e-6, 1e-6) == 0
    assert module.classify(-2e-6, 1e-6) == -1
    with pytest.raises(ValueError, match="finite"):
        module.score_row("bad", (2, 4), (float("nan"), -6), (-5, -5), 1e-6)


def test_scores_have_no_source_text_and_strata_keep_categories():
    rows = [held(0), held(1, "terminal_bracket_reversal")]
    details = [module.verify_case(row) for row in rows]
    scores = [module.score_row(row["id"], (3, 5), (-2, -3), (-2, -3), 1e-6) for row in rows]
    for score in scores:
        assert all(isinstance(v, (int, float)) and math.isfinite(v) for k, v in score.items() if k != "id")
        assert not {"prompt", "chosen", "rejected", "oracle"} & score.keys()
    strata = module.stratify(rows, details, scores)
    assert set(strata["reason"]) == {"omit_one_edge", "terminal_bracket_reversal"}
    assert strata["token_length_direction"]["chosen_shorter"]["samples"] == 2


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def __getitem__(self, index): return Tensor(self.value[index])
    @property
    def shape(self): return self.value.shape
    def float(self): return Tensor(self.value.astype(float))
    def unsqueeze(self, dim): return Tensor(np.expand_dims(self.value, dim))
    def squeeze(self, dim): return Tensor(np.squeeze(self.value, dim))
    def gather(self, dim, indices): return Tensor(np.take_along_axis(self.value, indices.value, axis=dim))
    def sum(self): return Tensor(self.value.sum())
    def item(self): return self.value.item()


def test_completion_logp_excludes_prompt_positions_and_includes_eos(monkeypatch):
    import sys
    def log_softmax(tensor, dim):
        values = tensor.value
        maximum = values.max(axis=dim, keepdims=True)
        return Tensor(values - maximum - np.log(np.exp(values - maximum).sum(axis=dim, keepdims=True)))
    torch = SimpleNamespace(tensor=lambda value, **kw: Tensor(value), long="long",
        ones_like=lambda tensor: Tensor(np.ones_like(tensor.value)),
        inference_mode=contextlib.nullcontext, autocast=lambda *a, **k: contextlib.nullcontext(),
        log_softmax=log_softmax)
    monkeypatch.setitem(sys.modules, "torch", torch)
    logits = np.zeros((1, 5, 5))
    logits[0, 0, 1] = 1000  # Prompt-only probability must not enter the sum.
    logits[0, 1, 2] = 2
    logits[0, 2, 3] = 3
    logits[0, 3, 4] = 4  # EOS contributes too.
    logits[0, 4, :] = float("nan")  # Final unused logit must never be scored.
    class Model:
        device = SimpleNamespace(type="cpu")
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].value.tolist() == [[0, 1, 2, 3, 4]]
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=Tensor(logits))
    result = module.completion_logp(Model(), [0, 1], [2, 3, 4])
    expected = sum(value - math.log(math.exp(value) + 4) for value in [2, 3, 4])
    assert result == pytest.approx(expected)


def fake_model():
    class Model:
        active = None
        def set_adapter(self, name): self.active = name
        def eval(self): return self
        def parameters(self): return [Mock()]
    return Model()


def test_both_adapters_scored_without_mutation_and_reference_change_rejected():
    model = fake_model()
    rows = [{"id": "sample"}]
    encoded = [{"prompt": [1, 2], "chosen": [3, 9], "rejected": [4, 5, 9]}]
    calls = []
    def scorer(model, prompt, completion):
        calls.append((model.active, list(completion)))
        return -len(completion)
    scores, identities = module.evaluate_scores(model, encoded, rows, [{}], scorer=scorer,
        fingerprint=lambda model, name: "unchanged-" + name)
    assert calls == [("policy", [3, 9]), ("policy", [4, 5, 9]),
                     ("reference", [3, 9]), ("reference", [4, 5, 9])]
    assert scores[0]["adjusted_outcome"] == 0
    assert identities["reference_adapter_sha256_before"] == identities["reference_adapter_sha256_after"]
    fingerprint = Mock(side_effect=["p", "r", "p", "CHANGED"])
    with pytest.raises(RuntimeError, match="changed adapter"):
        module.evaluate_scores(model, encoded, rows, [{}], scorer=scorer, fingerprint=fingerprint)


def test_shared_loader_uses_one_base_and_restores_second_adapter_precision(monkeypatch):
    import sys
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = []
    base = object()
    base_loader = Mock(return_value=base)
    policy_loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=base_loader)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=policy_loader)))
    monkeypatch.setattr(module, "adapter_fingerprint", lambda model, name: "fixed-policy")
    monkeypatch.setattr(module, "checkpoint_equal", Mock(side_effect=[True, False, True]))
    loaded, receipt = module.load_shared_model("base", "policy_path", "reference_path")
    assert loaded is model and receipt["reference_load_attempts"] == 2
    base_loader.assert_called_once()
    policy_loader.assert_called_once_with(base, "policy_path", adapter_name="policy", is_trainable=False, local_files_only=True)
    assert model.load_adapter.call_count == 2

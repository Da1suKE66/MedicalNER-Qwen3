"""CPU checks of token shifts, objective, PEFT switching and certified admission."""
import copy
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import train_choice_dpo as train
from generative.choice_preferences import compile_preference, encode_preference_row
from generative.recovery_preferences import make_case
from kg_agent import choice_backend


class Tokenizer:
    eos_token_id = 999999
    all_special_ids = [999999]
    pad_token_id = 999998

    def apply_chat_template(self, messages, **kwargs):
        return "USER:" + messages[0]["content"] + "\nASSISTANT:</think>\n\n"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))


class Array(np.ndarray):
    """Tiny NumPy adapter exercising production indexing without local Torch."""
    def float(self):
        return self.astype(np.float32)

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim).view(Array)

    def squeeze(self, dim=None):
        return np.asarray(self).squeeze(axis=dim).view(Array)

    def sum(self, dim=None):
        return np.asarray(self).sum(axis=dim).view(Array)

    def gather(self, dim, indices):
        return np.take_along_axis(self, indices, axis=dim).view(Array)


def test_actual_completion_function_shifts_one_and_ignores_prompt_logits():
    logits = np.array([[[8, 0, 0], [0, 8, 0], [0, 0, 8], [0, 8, 0], [0, 0, 8]]], dtype=float)
    module = SimpleNamespace(long=np.int64,
        tensor=lambda x, **kw: np.array(x).view(Array),
        ones_like=lambda x: np.ones_like(x),
        nn=SimpleNamespace(functional=SimpleNamespace(log_softmax=lambda a, dim:
            (np.asarray(a) - np.log(np.exp(np.asarray(a)).sum(axis=dim, keepdims=True))).view(Array))))
    calls = []
    class Model:
        device = "cpu"
        def __call__(self, **kw):
            calls.append(kw)
            return SimpleNamespace(logits=logits.view(Array))
    # Prompt length three: logits positions 2 and 3 predict tokens [2,1].
    actual = train.completion_logps(Model(), [0, 0, 0], [2, 1], torch_module=module)
    expected = 2 * (8 - math.log(math.exp(8) + 2))
    assert float(actual) == pytest.approx(expected, abs=1e-6)
    logits[0, :2] = [1e4, -1e4, 1e4]
    assert float(train.completion_logps(Model(), [0, 0, 0], [2, 1], torch_module=module)) == actual
    assert calls[0]["input_ids"].tolist() == [[0, 0, 0, 2, 1]]
    assert calls[0]["use_cache"] is False


def test_dpo_sigmoid_sum_objective_equal_start_and_preference_direction():
    functional = SimpleNamespace(logsigmoid=lambda x: -np.logaddexp(0, -x))
    f = lambda c, r: float(train.dpo_loss(c, r, -10., -8., .1, functional=functional))
    assert f(-10, -8) == pytest.approx(math.log(2))
    assert f(-9, -8) < f(-10, -8) < f(-11, -8)
    assert f(-10, -9) < f(-10, -8) < f(-10, -7)


def test_switch_back_refreezes_reference_even_when_peft_enables_selected_weights():
    class Parameter:
        def __init__(self):
            self.requires_grad = False
        def requires_grad_(self, value):
            self.requires_grad = value
    class Model:
        def __init__(self):
            self.p = {n: Parameter() for n in ("x.lora_A.policy.weight", "x.lora_A.reference.weight", "x.weight")}
        def set_adapter(self, name):
            for n, p in self.p.items():
                p.requires_grad = f".{name}." in n
        def named_parameters(self):
            return self.p.items()
        def train(self, mode):
            self.training = mode
    model = Model()
    train.activate(model, "reference", training=False)
    assert not any(p.requires_grad for p in model.p.values()) and not model.training
    selected = train.activate(model, "policy", training=True)
    assert selected == [model.p["x.lora_A.policy.weight"]]
    assert not model.p["x.lora_A.reference.weight"].requires_grad
    assert not model.p["x.weight"].requires_grad and model.training


def test_schedule_equal_full_pass_exposure_and_seed():
    a = train.row_schedule(128, 128, 42)
    assert sorted(a) == list(range(128)) and a == train.row_schedule(128, 128, 42)
    assert a != train.row_schedule(128, 128, 43)
    assert train.row_schedule(2, 5, 42)[:2] in ([0, 1], [1, 0])


def test_encoding_is_same_independent_action_tokens_for_both_objectives():
    row = compile_preference(make_case(0, "omit_one_edge"))
    tokenizer = Tokenizer()
    encoded = encode_preference_row(row, tokenizer, 100000)
    train.validate_encoded(encoded, tokenizer.eos_token_id)
    assert encoded["chosen_ids"] == [token for n in row["chosen_actions"] for token in map(ord, f"{n}\n")] + [999999]
    assert train.exposure_delta(encoded, "continued-sft")["rejected_completion_tokens"] == 0
    dpo = train.exposure_delta(encoded, "dpo")
    sft = train.exposure_delta(encoded, "continued-sft")
    assert dpo["chosen_completion_tokens"] == sft["chosen_completion_tokens"]
    assert dpo["prompt_loss_tokens"] == sft["prompt_loss_tokens"] == 0
    assert dpo["policy_completion_loss_tokens"] == len(encoded["chosen_ids"]) + len(encoded["rejected_ids"])


@pytest.mark.parametrize("mutation", ["holdout", "unknown", "actions", "prompt"])
def test_compiler_is_not_bypassed_for_train_admission(mutation):
    source = make_case(0, "omit_one_edge", partition="mechanical_holdout" if mutation == "holdout" else "train")
    row = compile_preference(source)
    if mutation == "unknown":
        row["source_row"]["unknown"] = ["pair"]
    elif mutation == "actions":
        row["chosen_actions"] = [0]
    elif mutation == "prompt":
        row["prompt"] += "tamper"
    with pytest.raises(ValueError):
        encode_preference_row(row, Tokenizer(), 100000, require_train=True)


@pytest.mark.parametrize("bad", [[1], [999999, 999999], [True, 999999], []])
def test_malformed_completion_encoding_rejected(bad):
    with pytest.raises(ValueError):
        train.validate_encoded(dict(prompt_ids=[1], chosen_ids=bad, rejected_ids=[2, 999999], stage="relations"), 999999)


def test_parent_receipt_binds_loaded_parameter_fingerprint_and_manifest(tmp_path, monkeypatch):
    final = tmp_path / "final"
    final.mkdir()
    receipt = dict(status="training_complete_evaluation_pending", saved_reload_exact=True,
        policy_adapter_sha256_after="a" * 64, saved_adapter_sha256_verified="a" * 64)
    payload = json.dumps(receipt).encode()
    manifest = tmp_path / "run_manifest.json"
    manifest.write_bytes(payload)
    seen = []
    def check(path, stage, *, base_model):
        seen.append((stage, base_model))
        return {"training_manifest_sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(choice_backend, "check_adapter_contract", check)
    assert train.parent_receipt(final, "base")[1] == receipt
    assert seen == [("entities", "base"), ("relations", "base")]
    manifest.write_text(json.dumps({**receipt, "status": "training"}))
    with pytest.raises(ValueError, match="manifest"):
        train.parent_receipt(final, "base")


def test_cli_defaults_and_serial_only():
    required = ["--data", "x", "--base-model", "base", "--parent-adapter", "parent", "--output-dir", "out"]
    args = train.parse_args(required)
    assert (args.max_steps, args.gradient_accumulation, args.learning_rate, args.beta) == (32, 4, 5e-6, .1)
    assert train.parse_args(required + ["--mode", "continued-sft"]).mode == "continued-sft"
    with pytest.raises(SystemExit):
        train.parse_args(required + ["--batch-size", "2"])


def test_preflight_main_uses_same_certified_data_order_and_chosen_for_both_modes(tmp_path, monkeypatch):
    rows = [compile_preference(make_case(i, kind)) for i, kind in enumerate(("omit_one_edge", "wrong_evidence"))]
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    contract = {"stages": ["entities", "relations"], "adapter_sha256": "b" * 64}
    parent = {"saved_adapter_sha256_verified": "a" * 64}
    monkeypatch.setattr(train, "parent_receipt", lambda *args: (contract, parent, "c" * 64))
    monkeypatch.setattr(choice_backend, "base_input_digest", lambda path: "d" * 64)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer())))
    manifests = []
    for mode in ("dpo", "continued-sft"):
        out = tmp_path / mode
        train.main(["--mode", mode, "--data", str(source), "--base-model", "base",
            "--parent-adapter", "parent", "--output-dir", str(out), "--max-length", "100000",
            "--max-steps", "1", "--gradient-accumulation", "2", "--preflight-only"])
        manifests.append(json.loads((out / "run_manifest.json").read_text()))
    for key in ("row_schedule_sha256", "paired_chosen_encoding_sha256", "data_sha256",
                "parent_adapter_sha256", "scheduled_rows", "scheduled_unique_rows"):
        assert manifests[0][key] == manifests[1][key]
    assert manifests[0]["status"] == "preflight_complete"
    assert manifests[0]["scheduled_unique_rows"] == 2
    assert "observed_exposure" not in manifests[0]


def test_real_torch_causal_gradients_if_installed():
    torch = pytest.importorskip("torch")
    logits = torch.randn(1, 5, 4, requires_grad=True)
    class Model:
        device = "cpu"
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=logits)
    score = train.completion_logps(Model(), [0, 1, 2], [3, 0])
    expected = torch.log_softmax(logits[:, 2:4], -1)[0, 0, 3] + torch.log_softmax(logits[:, 2:4], -1)[0, 1, 0]
    assert torch.allclose(score, expected)
    (-score).backward()
    assert logits.grad[:, :2].abs().sum() == 0 and logits.grad[:, 4:].abs().sum() == 0
    assert logits.grad[:, 2:4].abs().sum() > 0

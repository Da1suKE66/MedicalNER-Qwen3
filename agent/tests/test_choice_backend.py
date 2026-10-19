"""CPU-only contract tests: no model download, torch runtime, or GPU required."""

from contextlib import nullcontext
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import kg_agent.choice_backend as backend_module
from kg_agent.choice_backend import ChoiceFailure, ChoiceGenerator, adapter_files_digest, check_adapter_contract
from kg_agent.entity_choices import EntityChoiceTask
import run_choice_pipeline as cli


class FakeTensor:
    def __init__(self, values):
        self.values = values

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, columns = key
            return FakeTensor(self.values[row][columns])
        return FakeTensor(self.values[key])

    def tolist(self):
        return self.values


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 900
    pad_token_id = 901
    all_special_ids = [900, 901]

    def __init__(self, prompt_length=8):
        self.prompt_length = prompt_length
        self.tokenize_calls = []
        self.chat_calls = []

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "rendered:" + messages[0]["content"] + "\n</think>"

    def __call__(self, text, **kwargs):
        self.tokenize_calls.append((text, kwargs))
        return FakeInputs(input_ids=FakeTensor([[700] * self.prompt_length]),
                          attention_mask=FakeTensor([[1] * self.prompt_length]))


class FakeGenerationConfig:
    instances = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.instances.append(self)

    @classmethod
    def from_model_config(cls, *args, **kwargs):
        raise AssertionError("Do not inherit a checkpoint generation configuration")

    from_pretrained = from_model_config


class FakeModel:
    device = "cpu"

    def __init__(self, suffix, *, maximum=4096, enforce_prefix=True):
        self.config = SimpleNamespace(max_position_embeddings=maximum)
        self.generation_config = SimpleNamespace(forced_eos_token_id=900, min_length=4000,
                                                 do_sample=True, bad_words_ids=[[48]])
        self.suffix = suffix
        self.enforce_prefix = enforce_prefix
        self.generate_calls = []
        self.adapter_calls = []
        self.constraint_calls = 0

    def set_adapter(self, name):
        self.adapter_calls.append(name)

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        allowed = kwargs["prefix_allowed_tokens_fn"]
        assert callable(allowed)
        prompt = kwargs["input_ids"].values[0]
        if self.enforce_prefix:
            for index, token in enumerate(self.suffix):
                assert token in allowed(0, FakeTensor(prompt + self.suffix[:index]))
                self.constraint_calls += 1
        return FakeTensor([prompt + self.suffix])


@pytest.fixture
def fake_runtime(monkeypatch):
    FakeGenerationConfig.instances = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=FakeGenerationConfig))

    def make(suffix=None, *, prompt_length=8, maximum=4096, enforce_prefix=True, adapter_name="entities"):
        return SimpleNamespace(tokenizer=FakeTokenizer(prompt_length), adapter_name=adapter_name,
                               model=FakeModel([48, 10, 900] if suffix is None else suffix,
                                               maximum=maximum, enforce_prefix=enforce_prefix))
    return make


def empty_task():
    return EntityChoiceTask([])


def test_generation_uses_fresh_config_explicit_no_model_defaults_and_constraint(fake_runtime):
    backend = fake_runtime()
    actions, trace = ChoiceGenerator(backend, max_input_tokens=8, max_new_tokens=3).run(empty_task())
    assert actions == [0] and trace["complete"] and trace["eos_observed"]
    assert trace["generated_token_ids"] == [48, 10, 900]
    assert backend.model.adapter_calls == ["entities"]
    assert backend.model.constraint_calls == 3
    call = backend.model.generate_calls[0]
    config = call["generation_config"]
    assert config is FakeGenerationConfig.instances[0] and config is not backend.model.generation_config
    assert config.__dict__ == {"max_new_tokens": 3, "do_sample": False, "num_beams": 1,
                               "eos_token_id": 900, "pad_token_id": 901, "use_cache": True}
    # transformers >= 4.50 otherwise merges model defaults into globally-default
    # fields of even a fresh GenerationConfig (e.g. forced_eos/min_length).
    assert call.get("use_model_defaults") is False
    assert backend.tokenizer.tokenize_calls[0][1] == {
        "add_special_tokens": False, "truncation": False, "return_tensors": "pt"}


def test_base_only_view_does_not_switch_adapters(fake_runtime):
    backend = fake_runtime(adapter_name=None)
    ChoiceGenerator(backend).run(empty_task())
    assert backend.model.adapter_calls == []
    assert callable(backend.model.generate_calls[0]["prefix_allowed_tokens_fn"])


def test_prompt_budget_failure_occurs_before_generation_without_truncation(fake_runtime):
    backend = fake_runtime(prompt_length=9)
    with pytest.raises(ChoiceFailure, match="no silent truncation") as error:
        ChoiceGenerator(backend, max_input_tokens=8).run(empty_task())
    assert error.value.trace["prompt_tokens"] == 9
    assert backend.model.generate_calls == []
    assert backend.tokenizer.tokenize_calls[0][1]["truncation"] is False


def test_total_context_budget_is_checked_before_generation(fake_runtime):
    backend = fake_runtime(prompt_length=8, maximum=10)
    with pytest.raises(ChoiceFailure, match="model context"):
        ChoiceGenerator(backend, max_new_tokens=3).run(empty_task())
    assert not backend.model.generate_calls
    backend = fake_runtime(prompt_length=8, maximum=11)
    assert ChoiceGenerator(backend, max_new_tokens=3).run(empty_task())[0] == [0]


@pytest.mark.parametrize("suffix, expected_error", [
    ([49], None),  # part of first start action
    ([49, 10], None),  # legal start, but end/label/importance/stop missing
    ([49, 10, 49, 10, 49, 10, 49, 10], None),  # entity without task stop
    ([900], "eos_before_terminal"),
    ([48, 10, 88], "tokens_after_terminal"),
    ([48, 10, 900, 900], "tokens_after_eos"),
    ([88], "token_outside_current_choices"),
])
def test_incomplete_and_invalid_results_raise_choice_failure_with_replay_trace(fake_runtime, suffix, expected_error):
    backend = fake_runtime(suffix, enforce_prefix=expected_error is None)
    task = EntityChoiceTask([{"span_id": "SENT_001", "text": "alpha"}])
    with pytest.raises(ChoiceFailure, match="Incomplete or invalid") as error:
        ChoiceGenerator(backend, max_new_tokens=len(suffix)).run(task)
    trace = error.value.trace
    assert trace["generated_token_ids"] == suffix and not trace["complete"]
    assert trace["generated_tokens"] == len(suffix)
    if expected_error is not None:
        assert expected_error in trace["error"]
    else:
        assert trace["error"] is None and trace["budget_exhausted"]
    assert len(backend.model.generate_calls) == 1  # No unconstrained fallback.


def receipt(tmp_path, **changes):
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"fake-weights-not-loadable")
    value = {
        "output_contract": backend_module.CONTRACT,
        "stages": ["entities", "relations"],
        "completion_encoding": "independent_integer_newline_tokens_then_eos",
        "adapter_sha256": adapter_files_digest(adapter),
        "protocol_sha256": backend_module.protocol_implementation_digest(),
    }
    value.update(changes)
    (adapter / "choice_contract.json").write_text(json.dumps(value))
    return adapter, value


def test_adapter_receipt_accepts_matching_version_stage_weights_and_protocol(tmp_path):
    adapter, value = receipt(tmp_path)
    assert check_adapter_contract(adapter, "entities") == value
    assert check_adapter_contract(adapter, "relations") == value


def test_adapter_receipt_rejects_runtime_protocol_implementation_change(tmp_path, monkeypatch):
    adapter, _ = receipt(tmp_path)
    monkeypatch.setattr(backend_module, "protocol_implementation_digest", lambda: "new-runtime-implementation")
    with pytest.raises(ValueError, match="protocol implementation differs"):
        check_adapter_contract(adapter, "entities")


@pytest.mark.parametrize("dependency", [
    "choice_protocol.py", "entity_choices.py", "relation_choices.py",
    "contracts.py", "ontology.json", "chat.py",
])
def test_protocol_digest_covers_grammar_label_order_contracts_and_chat_bytes(tmp_path, monkeypatch, dependency):
    adapter, manifest = receipt(tmp_path)
    original_read_bytes = backend_module.Path.read_bytes

    def changed_read_bytes(path):
        data = original_read_bytes(path)
        if path.name != dependency:
            return data
        if dependency == "ontology.json":
            ontology = json.loads(data)
            assert len(ontology["node_labels"]) > 1
            # Same vocabulary, different integer meanings: an old receipt
            # must fail even though every label name is still valid.
            ontology["node_labels"] = list(reversed(ontology["node_labels"]))
            return json.dumps(ontology, ensure_ascii=False).encode()
        return data + b"\n# simulated deployed implementation change\n"

    monkeypatch.setattr(backend_module.Path, "read_bytes", changed_read_bytes)
    assert backend_module.protocol_implementation_digest() != manifest["protocol_sha256"]
    with pytest.raises(ValueError, match="protocol implementation differs"):
        check_adapter_contract(adapter, "entities")


def test_protocol_digest_covers_unicode_boundary_version(tmp_path, monkeypatch):
    adapter, manifest = receipt(tmp_path)
    monkeypatch.setattr(backend_module.unicodedata, "unidata_version", "changed-unicode-tables")
    assert backend_module.protocol_implementation_digest() != manifest["protocol_sha256"]
    with pytest.raises(ValueError, match="protocol implementation differs"):
        check_adapter_contract(adapter, "entities")


def base_inputs(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (base / name).write_text(json.dumps({"fixture": name}))
    return base


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_base_input_digest_and_receipt_reject_required_file_drift(tmp_path, name):
    base = base_inputs(tmp_path)
    original = backend_module.base_input_digest(base)
    adapter, value = receipt(tmp_path, base_input_sha256=original)
    assert check_adapter_contract(adapter, "entities", base_model=base) == value
    (base / name).write_text(json.dumps({"fixture": name, "changed": True}))
    assert backend_module.base_input_digest(base) != original
    with pytest.raises(ValueError, match="Base tokenizer/model configuration differs"):
        check_adapter_contract(adapter, "entities", base_model=base)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_base_input_digest_requires_each_mandatory_file(tmp_path, name):
    base = base_inputs(tmp_path)
    (base / name).unlink()
    with pytest.raises(ValueError, match=f"Missing base input artifact: {name}"):
        backend_module.base_input_digest(base)


@pytest.mark.parametrize("name", ["special_tokens_map.json", "added_tokens.json"])
def test_optional_base_token_files_presence_and_content_are_bound(tmp_path, name):
    base = base_inputs(tmp_path)
    absent_digest = backend_module.base_input_digest(base)
    adapter, _ = receipt(tmp_path, base_input_sha256=absent_digest)
    (base / name).write_text("{}")
    present_digest = backend_module.base_input_digest(base)
    assert present_digest != absent_digest
    with pytest.raises(ValueError, match="Base tokenizer/model configuration differs"):
        check_adapter_contract(adapter, "relations", base_model=base)
    (base / name).write_text('{"changed":true}')
    assert backend_module.base_input_digest(base) not in {absent_digest, present_digest}
    (base / name).unlink()
    assert backend_module.base_input_digest(base) == absent_digest


def test_serving_with_base_requires_its_input_digest_in_receipt(tmp_path):
    base = base_inputs(tmp_path)
    adapter, _ = receipt(tmp_path)
    with pytest.raises(ValueError, match="Base tokenizer/model configuration differs"):
        check_adapter_contract(adapter, "entities", base_model=base)


def test_old_json_adapter_without_receipt_is_rejected_before_model_load(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy adapter admission must fail before GPU/model construction")
    monkeypatch.setattr(cli, "TransformersBackend", forbidden)
    args = SimpleNamespace(base_model="unused", entity_adapter=str(tmp_path), relation_adapter=str(tmp_path))
    with pytest.raises(ValueError, match="legacy JSON adapters require retraining"):
        cli.build_agent(args)


@pytest.mark.parametrize("changes", [
    {"output_contract": "legacy-json"}, {"stages": ["relations"]},
    {"completion_encoding": "tokenize_concatenated_text"},
    {"adapter_sha256": "different-weights"}, {"protocol_sha256": "different-source-code"},
])
def test_adapter_receipt_rejects_contract_stage_encoding_or_digest_drift(tmp_path, changes):
    adapter, _ = receipt(tmp_path, **changes)
    with pytest.raises(ValueError):
        check_adapter_contract(adapter, "entities")


@pytest.mark.parametrize("name", ["adapter_config.json", "adapter_model.safetensors"])
def test_adapter_receipt_rejects_mutated_or_missing_artifact(tmp_path, name):
    adapter, _ = receipt(tmp_path)
    artifact = adapter / name
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="weights do not match"):
        check_adapter_contract(adapter, "entities")
    artifact.unlink()
    with pytest.raises(ValueError, match="Missing adapter artifact"):
        check_adapter_contract(adapter, "entities")


def prepare_cli(monkeypatch, tmp_path, rows, *extra):
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(sys, "argv", ["run_choice_pipeline", "--input", str(source), "--output", str(output), *extra])
    return output


def test_cli_preserves_completed_and_failed_rows_and_continues_after_failure(monkeypatch, tmp_path, capsys):
    rows = [{"text": text, "metadata": {"case": index}} for index, text in enumerate(("alpha", "beta", "gamma"))]
    calls = []

    class Agent:
        def run(self, raw, metadata):
            calls.append((raw, metadata))
            if raw == "beta":
                raise ChoiceFailure("engineering failure", {"generated_token_ids": [49], "complete": False})
            return SimpleNamespace(output={"entities": [{"name": raw}], "relations": []}, trace={"complete": True})

    monkeypatch.setattr(cli, "build_agent", lambda args: Agent())
    output = prepare_cli(monkeypatch, tmp_path, rows)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == len(rows) == len(calls)
    assert [row["sample_index"] for row in saved] == [0, 1, 2]
    assert [row["status"] for row in saved] == ["complete", "failed", "complete"]
    assert saved[1]["output"] == {"entities": [], "relations": []}
    assert saved[1]["trace"]["generated_token_ids"] == [49]
    assert "ChoiceFailure" in saved[1]["error"]
    for input_row, output_row in zip(rows, saved):
        expected = hashlib.sha256(json.dumps([input_row["text"], input_row["metadata"]], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert output_row["input_sha256"] == expected
    summary = json.loads(capsys.readouterr().out)
    assert summary["samples"] == 3 and summary["failed_samples"] == 1


def test_cli_icd_only_runs_without_adapter_or_gpu_construction(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("ICD input must not construct a model, generator, or load torch")
    monkeypatch.setattr(cli, "build_agent", forbidden)
    monkeypatch.setattr(cli, "TransformersBackend", forbidden)
    monkeypatch.setattr(cli.NoTextGeneration, "generate", forbidden)
    rows = [{"code": "X01", "title": "Condition Alpha"}, {"code": "X02", "title": "Condition Beta"}]
    output = prepare_cli(monkeypatch, tmp_path, rows)
    cli.main()
    saved = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(saved) == 2 and all(row["status"] == "complete" for row in saved)
    assert all(row["trace"]["mode"] == "deterministic_icd_parser" for row in saved)
    assert json.loads(capsys.readouterr().out)["failed_samples"] == 0


def test_cli_configuration_error_is_not_converted_to_per_sample_model_failure(monkeypatch, tmp_path):
    output = prepare_cli(monkeypatch, tmp_path, ["alpha", "beta"])
    with pytest.raises(ValueError, match="requires --base-model"):
        cli.main()
    assert not output.exists()


def test_cli_does_not_overwrite_previous_outputs(monkeypatch, tmp_path):
    output = prepare_cli(monkeypatch, tmp_path, ["alpha"])
    output.write_text("previous-results\n")
    with pytest.raises(ValueError, match="already exists"):
        cli.main()
    assert output.read_text() == "previous-results\n"

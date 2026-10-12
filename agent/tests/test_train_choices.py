"""CPU admission and trainer-orchestration tests, not GPU learning evidence."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import train_choices
from generative import train as legacy_train
from generative.recovery_preferences import make_case
from kg_agent.choice_backend import check_adapter_contract


class Tokenizer:
    eos_token_id = 1000001
    eos_token = "<EOS>"
    pad_token_id = 1000002
    padding_side = "left"

    def __init__(self):
        self.encoded_texts = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "USER:" + messages[0]["content"] + "</think>\n\n"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        return list(map(ord, text))

    def save_pretrained(self, destination):
        (Path(destination) / "tokenizer_config.json").write_text('{"test_mock":true}')


def arguments(monkeypatch, tmp_path, rows=None, *extra):
    rows = [make_case(0, "terminal_bracket_reversal")] if rows is None else rows
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (base / name).write_text("{}")
    destination = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["train_choices", "--data", str(source),
        "--base-model", str(tmp_path / "base"), "--output-dir", str(destination),
        "--max-steps", "2", "--max-length", "100000", *extra])
    return source, destination


@pytest.fixture
def runtime(monkeypatch):
    """Mock GPU execution; keep real recipe/compiler/codec/collator/receipts."""
    tokenizer = Tokenizer()
    state = SimpleNamespace(tokenizer=tokenizer, models=[], trainers=[], lora_configs=[],
                            reloads=[], collated=[], unchanged=False, short_steps=False,
                            loss=0.5, reload_mismatch=False, finite_parameters=True,
                            prefetch_rows=0, prefetched=[], training_step_calls=[], fail_step=None)

    class Model:
        def __init__(self, fingerprint="before"):
            self.fingerprint = fingerprint
            self.config = SimpleNamespace(pad_token_id=None, use_cache=True)
            self.device = "cpu"
            self.enabled_grad_inputs = False
        def enable_input_require_grads(self):
            self.enabled_grad_inputs = True
        def to(self, device):
            self.device = device
            return self
        def eval(self):
            return self
        def set_adapter(self, name):
            assert name == "default"
        def named_parameters(self):
            return [("model.lora_A.default.weight", SimpleNamespace(requires_grad=True))]
        def save_pretrained(self, path, *, safe_serialization):
            assert safe_serialization is True
            path = Path(path)
            path.mkdir()
            (path / "adapter_config.json").write_text('{"task_type":"CAUSAL_LM","test_mock":true}')
            (path / "adapter_model.safetensors").write_bytes(self.fingerprint.encode())

    def load_base(path, **kwargs):
        assert kwargs == {"torch_dtype": "bf16", "attn_implementation": "sdpa", "local_files_only": True}
        model = Model()
        state.models.append(model)
        return model

    def lora_config(**kwargs):
        state.lora_configs.append(kwargs)
        return kwargs

    def add_lora(model, config):
        assert config["task_type"] == "CAUSAL_LM"
        assert set(config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        return model

    def reload_adapter(model, path, **kwargs):
        assert kwargs == {"local_files_only": True}
        assert isinstance(path, str)
        model.fingerprint = "mismatch" if state.reload_mismatch else (Path(path) / "adapter_model.safetensors").read_text()
        state.reloads.append((model, path))
        return model

    def verify_saved(model, expected):
        if model.fingerprint != expected:
            raise RuntimeError("Saved/reloaded adapter differs from the recorded trained weights")
        return expected

    class Trainer:
        def __init__(self, *, model, args, train_dataset, data_collator):
            self.model, self.args, self.dataset, self.collator = model, args, train_dataset, data_collator
            self.state = SimpleNamespace(global_step=0)
            def unwrap(value, *, keep_fp32_wrapper):
                assert value is model and keep_fp32_wrapper is False
                return value
            self.accelerator = SimpleNamespace(unwrap_model=unwrap)
            state.trainers.append(self)
        def train(self):
            offset = 0
            for step in range(self.args.max_steps):
                for _ in range(self.args.gradient_accumulation_steps):
                    items = [self.dataset[(offset + i) % len(self.dataset)]
                             for i in range(self.args.per_device_train_batch_size)]
                    offset += len(items)
                    collated = self.collator(items)
                    assert set(collated) == {"input_ids", "attention_mask", "labels", "_choice_counts_batch"}
                    self.training_step(self.model, collated, num_items_in_batch=len(items))
                self.state.global_step = step + 1
            for _ in range(state.prefetch_rows):
                state.prefetched.append(self.collator([self.dataset[offset % len(self.dataset)]]))
                offset += 1
            if state.short_steps:
                self.state.global_step -= 1
            if not state.unchanged:
                self.model.fingerprint = "after"
            return SimpleNamespace(metrics={"train_loss": state.loss, "train_runtime": 1.0})
        def training_step(self, model, inputs, num_items_in_batch=None):
            assert model is self.model
            assert set(inputs) == {"input_ids", "attention_mask", "labels"}
            assert num_items_in_batch == len(inputs["input_ids"])
            state.training_step_calls.append(dict(self.observed_exposure))
            if len(state.training_step_calls) == state.fail_step:
                raise RuntimeError("simulated training_step failed")
            state.collated.append(inputs)
            return state.loss
        def save_state(self):
            (Path(self.args.output_dir) / "trainer_state.json").write_text(json.dumps({"global_step": self.state.global_step}))

    state.base_loader = Mock(side_effect=load_base)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=state.base_loader),
        Trainer=Trainer, TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs),
        set_seed=Mock(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        bfloat16="bf16", long="long", tensor=lambda value, **kwargs: value,
        cuda=SimpleNamespace(empty_cache=Mock()),
        isfinite=lambda value: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: state.finite_parameters)),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=lora_config, get_peft_model=add_lora,
        PeftModel=SimpleNamespace(from_pretrained=reload_adapter),
    ))
    monkeypatch.setattr(legacy_train, "adapter_fingerprint", lambda model, adapter: model.fingerprint)
    monkeypatch.setattr(legacy_train, "require_saved_weight_equality", verify_saved)
    return state


def test_preflight_real_recipe_data_masks_prompt_and_never_encodes_old_json(tmp_path, monkeypatch, runtime):
    source, out = arguments(monkeypatch, tmp_path, None, "--preflight-only")
    row = json.loads(source.read_text())
    train_choices.main()
    report = json.loads((out / "run_manifest.json").read_text())
    assert report["status"] == "preflight_complete" and report["training_rows"] == 2
    assert report["source_rows"] == 1 and report["stages"] == ["entities", "relations"]
    assert report["medical_training"] is False and report["clinical_quality_claim"] is False
    assert report["data_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["completion_encoding"] == "independent_integer_newline_tokens_then_eos"
    assert not (out / "final").exists()
    runtime.base_loader.assert_not_called()
    assert not runtime.trainers
    assert row["chosen"] not in runtime.tokenizer.encoded_texts
    assert row["rejected"] not in runtime.tokenizer.encoded_texts
    assert all(text.startswith("USER:") or (text.endswith("\n") and text[:-1].isdigit())
               for text in runtime.tokenizer.encoded_texts)


def test_training_budget_lora_change_save_reload_and_receipt_are_bound(tmp_path, monkeypatch, runtime):
    source, out = arguments(monkeypatch, tmp_path)
    train_choices.main()
    report = json.loads((out / "run_manifest.json").read_text())
    trainer = runtime.trainers[0]
    assert trainer.args.max_steps == report["optimizer_updates"] == 2
    assert trainer.args.remove_unused_columns is False
    assert trainer.args.gradient_accumulation_steps == 4
    assert trainer.args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert trainer.args.logging_nan_inf_filter is False
    assert trainer.args.dataloader_num_workers == 0 and trainer.args.save_strategy == "no"
    assert report["realized_collator_exposures"]["rows"] == 8
    assert report["realized_collator_exposures"]["entities_rows"] == 4
    assert report["realized_collator_exposures"]["relations_rows"] == 4
    assert report["adapter_parameters_before"] != report["adapter_parameters_after"]
    assert report["status"] == "training_complete_evaluation_pending" and report["saved_reload_exact"] is True
    assert report["policy_weights_changed"] is True and report["global_step"] == 2
    assert report["policy_adapter_sha256_after"] == report["saved_adapter_sha256_verified"]
    assert report["policy_adapter_sha256_before"] != report["policy_adapter_sha256_after"]
    assert report["observed_exposure"] == report["realized_collator_exposures"]
    assert report["collator_exposures"] == report["realized_collator_exposures"]
    assert report["observed_exposure"]["batches"] == 8
    assert report["exposure_counting_version"] == "successful_training_step_v2"
    assert runtime.models[0].enabled_grad_inputs and runtime.models[0].device == "cuda"
    assert len(runtime.models) == 2 and len(runtime.reloads) == 1
    assert len(runtime.lora_configs) == 1 and runtime.lora_configs[0]["lora_dropout"] == 0.0
    # Actual encoded rows and real collator preserve completion-only labels/EOS.
    for item in trainer.dataset:
        count = item["_choice_counts"]["prompt_tokens"]
        assert item["labels"][:count] == [-100] * count
        assert item["labels"][count:] == item["input_ids"][count:]
        assert item["input_ids"][-1] == runtime.tokenizer.eos_token_id
        text = "".join(map(chr, item["input_ids"][count:-1]))
        assert text and all(line.isdigit() for line in text.splitlines())
        assert text.endswith("0\n")
    for item in runtime.collated:
        for ids, mask, labels in zip(item["input_ids"], item["attention_mask"], item["labels"]):
            assert len(ids) == len(mask) == len(labels)
            assert labels[-1] == runtime.tokenizer.eos_token_id
    receipt = check_adapter_contract(out / "final", "entities", base_model=tmp_path / "base")
    assert receipt == check_adapter_contract(out / "final", "relations")
    assert receipt["base_input_sha256"] == report["base_input_sha256"]
    assert receipt["training_manifest_sha256"] == hashlib.sha256((out / "run_manifest.json").read_bytes()).hexdigest()
    assert receipt["training_data_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["medical_quality_evaluated"] is False
    (tmp_path / "base" / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="[Bb]ase|tokenizer"):
        check_adapter_contract(out / "final", "relations", base_model=tmp_path / "base")
    (out / "final" / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights do not match"):
        check_adapter_contract(out / "final", "relations")


@pytest.mark.parametrize("prefetch_rows", [1, 3])
def test_collator_prefetch_never_inflates_successful_training_exposure(tmp_path, monkeypatch, runtime, prefetch_rows):
    _, out = arguments(monkeypatch, tmp_path)
    runtime.prefetch_rows = prefetch_rows
    train_choices.main()
    report = json.loads((out / "run_manifest.json").read_text())
    assert report["observed_exposure"]["rows"] == report["observed_exposure"]["batches"] == 8
    assert report["observed_exposure"]["entities_rows"] == 4
    assert report["observed_exposure"]["relations_rows"] == 4
    assert report["collator_exposures"]["rows"] == 8 + prefetch_rows
    assert report["realized_collator_exposures"] == report["collator_exposures"]
    assert len(runtime.training_step_calls) == 8
    assert len(runtime.prefetched) == prefetch_rows
    assert all("_choice_counts_batch" in item for item in runtime.prefetched)
    actual = report["observed_exposure"]
    prepared = report["collator_exposures"]
    for key in ("prompt_tokens", "completion_tokens", "action_count"):
        assert prepared[key] - actual[key] == sum(
            meta[key] for batch in runtime.prefetched for meta in batch["_choice_counts_batch"])


@pytest.mark.parametrize("fail_step", [1, 3])
def test_failed_training_step_is_not_counted_and_partial_telemetry_is_saved(tmp_path, monkeypatch, runtime, fail_step):
    _, out = arguments(monkeypatch, tmp_path)
    runtime.fail_step = fail_step
    with pytest.raises(RuntimeError, match="simulated training_step failed"):
        train_choices.main()
    report = json.loads((out / "run_manifest.json").read_text())
    assert report["observed_exposure"].get("rows", 0) == fail_step - 1
    assert report["observed_exposure"].get("batches", 0) == fail_step - 1
    assert report["collator_exposures"]["rows"] == fail_step
    assert dict(runtime.trainers[0].observed_exposure) == report["observed_exposure"]
    assert [counts.get("rows", 0) for counts in runtime.training_step_calls] == list(range(fail_step))
    assert len(runtime.collated) == fail_step - 1
    assert report["status"] == "training"
    assert not (out / "final" / "choice_contract.json").exists()


def test_training_step_preserves_hf_positional_and_keyword_arguments():
    sentinel = object()
    forwarded = []
    class BaseTrainer:
        def training_step(self, model, inputs, *args, **kwargs):
            assert "_choice_counts_batch" not in inputs
            forwarded.append((model, inputs, args, kwargs))
            return sentinel
    class Trainer(train_choices.ChoiceExposureMixin, BaseTrainer):
        pass
    trainer = Trainer()
    meta = {"stage": "entities", "prompt_tokens": 3, "completion_tokens": 2, "action_count": 1}
    batch = {"input_ids": [[10, 11, 12, 13, 14]], "_choice_counts_batch": [meta]}
    model = object()
    assert trainer.training_step(model, batch, 7, hf_future_option=True) is sentinel
    assert forwarded == [(model, {"input_ids": [[10, 11, 12, 13, 14]]}, (7,), {"hf_future_option": True})]
    assert trainer.observed_exposure["rows"] == 1


@pytest.mark.parametrize("setting,value,match", [
    ("unchanged", True, "failed to update"),
    ("short_steps", True, "failed to update"),
    ("loss", float("nan"), "Nonfinite"),
    ("loss", float("inf"), "Nonfinite"),
    ("finite_parameters", False, "Nonfinite choice adapter parameters"),
    ("reload_mismatch", True, "Saved/reloaded adapter differs"),
])
def test_failed_training_or_reload_never_creates_usable_receipt(tmp_path, monkeypatch, runtime, setting, value, match):
    _, out = arguments(monkeypatch, tmp_path)
    setattr(runtime, setting, value)
    with pytest.raises(RuntimeError, match=match):
        train_choices.main()
    assert not (out / "final" / "choice_contract.json").exists()
    assert json.loads((out / "run_manifest.json").read_text())["status"] not in {
        "complete", "training_complete_evaluation_pending"}


@pytest.mark.parametrize("kind", ["heldout", "unknown", "changed_source", "generic_json", "duplicate"])
def test_preflight_rejects_nonreconstructible_nontraining_or_duplicate_source(tmp_path, monkeypatch, runtime, kind):
    row = make_case(0, "wrong_target", partition="mechanical_holdout" if kind == "heldout" else "train")
    if kind == "unknown":
        row["unknown_reasons"] = {"P0001": "unannotated"}
    if kind == "changed_source":
        row["oracle_source_text"] += " Unverified medical text."
    if kind == "generic_json":
        row = {"id": "generic", "split": "train", "prompt": "medical", "chosen": "{}", "rejected": "{}"}
    rows = [row, row] if kind == "duplicate" else [row]
    _, out = arguments(monkeypatch, tmp_path, rows, "--preflight-only")
    with pytest.raises((ValueError, KeyError)):
        train_choices.main()
    runtime.base_loader.assert_not_called()
    assert not out.exists()


@pytest.mark.parametrize("extra", [
    ("--max-steps", "0"), ("--max-steps", "-1"), ("--learning-rate", "nan"),
    ("--learning-rate", "inf"), ("--learning-rate", "0"), ("--batch-size", "0"),
    ("--gradient-accumulation", "0"), ("--stages", "entities", "entities"),
])
def test_invalid_or_unbounded_training_arguments_fail_before_model(tmp_path, monkeypatch, runtime, extra):
    _, out = arguments(monkeypatch, tmp_path, None, *extra)
    with pytest.raises(SystemExit):
        train_choices.main()
    runtime.base_loader.assert_not_called()
    assert not out.exists()


def test_existing_run_is_never_overwritten(tmp_path, monkeypatch, runtime):
    _, out = arguments(monkeypatch, tmp_path)
    out.mkdir()
    (out / "run_manifest.json").write_text("previous")
    with pytest.raises(ValueError, match="immutable"):
        train_choices.main()
    assert (out / "run_manifest.json").read_text() == "previous"
    runtime.base_loader.assert_not_called()


def test_trainingarguments_keywords_exist_in_installed_real_hf_api():
    # Import the installed real API in this test (the runtime fixture is absent).
    transformers = pytest.importorskip("transformers")
    signature = inspect.signature(transformers.TrainingArguments)
    tree = ast.parse(Path(train_choices.__file__).read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments")
    assert all(keyword.arg in signature.parameters for keyword in call.keywords)

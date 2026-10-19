"""CPU/mock safety contracts. These tests do NOT run a model or establish RL quality."""

from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative import build_preferences as preferences
from generative import train
from generative.protocol import ProtocolConfig, build_batches, dumps, supervision


def silver_row(index="train:0", group=0):
    text = "Feature A is a core symptom of Condition B and supports diagnosis of Condition B."
    entities = [
        {"id": "S", "name": "Feature A", "label": "Symptom"},
        {"id": "D", "name": "Condition B", "label": "Disease"},
    ]
    graph = {
        "entities": entities,
        "relations": [
            {"source": "S", "target": "D", "relation": label, "evidence": text}
            for label in ("is_core_symptom_of", "supports_diagnosis_of")
        ],
    }
    config = ProtocolConfig(strict_evidence=True)
    batches, doc, _ = build_batches(text, entities, config, sample_id=index)
    target, _ = supervision(batches[0], graph, doc, config)
    return {
        **batches[0],
        "group_id": group,
        "split": "train",
        "target": target,
        "completion": dumps(target),
    }


def argv(tmp_path, mode="dpo", *extra):
    return [
        "--mode",
        mode,
        "--data",
        str(tmp_path / "data.jsonl"),
        "--base-model",
        str(tmp_path / "base"),
        *(["--sft-adapter", str(tmp_path / "sft")] if mode != "sft" else []),
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


class Tokenizer:
    eos_token_id = 999
    eos_token = "EOS"
    pad_token = None

    def encode(self, text, add_special_tokens=False):
        return [1] * (len(text) + int(add_special_tokens))

    def __call__(self, text, add_special_tokens=False, **kwargs):
        class Inputs(dict):
            def to(self, device):
                return self
        return Inputs(input_ids=self.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return (
            "<|im_start|>user\n"
            + messages[0]["content"]
            + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def test_dpo_disables_the_actual_512_token_truncation_default(tmp_path):
    args = train.parse_args(argv(tmp_path))
    config = train.rl_config_kwargs(args)
    assert config["max_prompt_length"] is None
    assert config["max_completion_length"] is None
    assert config["max_length"] == 6144
    assert (
        config["model_adapter_name"] == "policy"
        and config["ref_adapter_name"] == "reference"
    )
    assert config["precompute_ref_log_probs"] and not config["reference_free"]
    assert not config["sync_ref_model"] and config["disable_dropout"]
    row = {"prompt": "P" * 600, "chosen": "C" * 10, "rejected": "R" * 5}
    audit = train.length_audit([row], Tokenizer(), 611, "dpo")
    assert audit["sequences"] == 2 and audit["max_tokens"] == 611
    assert audit["prompt_tokens_once"] == 600 and audit["completion_tokens_once"] == 17
    with pytest.raises(ValueError, match="Overlength"):
        train.length_audit([row], Tokenizer(), 610, "dpo")


def test_sft_audit_matches_existing_dataset_special_tokens():
    audit = train.length_audit(
        [{"prompt": "abc", "completion": "de"}], Tokenizer(), 7, "sft"
    )
    assert audit["max_tokens"] == 7  # one prompt special + one completion EOS


def test_grpo_reserves_full_rollout_space_and_checks_target(tmp_path):
    args = train.parse_args(
        argv(tmp_path, "grpo", "--max-length", "100", "--max-completion-length", "20")
    )
    config = train.rl_config_kwargs(args)
    assert config["max_prompt_length"] == 80 and config["max_completion_length"] == 20
    assert config["num_iterations"] == 1 and config["steps_per_generation"] == 16
    assert config["loss_type"] == "dr_grpo" and not config["scale_rewards"]
    assert not config["use_vllm"] and not config["sync_ref_model"]
    row = {"prompt": "P" * 80, "completion": "C" * 19}
    assert train.length_audit([row], Tokenizer(), 100, "grpo", 20)["max_tokens"] == 100
    with pytest.raises(ValueError, match="rollout budget"):
        train.length_audit([{**row, "prompt": "P" * 81}], Tokenizer(), 100, "grpo", 20)
    with pytest.raises(ValueError, match="GRPO target"):
        train.length_audit(
            [{**row, "completion": "C" * 20}], Tokenizer(), 100, "grpo", 20
        )


def test_beta_zero_is_a_real_grpo_no_kl_ablation(tmp_path):
    args = train.parse_args(argv(tmp_path, "grpo", "--beta", "0"))
    assert train.rl_config_kwargs(args)["beta"] == 0
    with pytest.raises(SystemExit):
        train.parse_args(argv(tmp_path, "dpo", "--beta", "0"))


@pytest.mark.parametrize(
    "options",
    [
        ("--beta", "nan"),
        ("--beta", "-1"),
        ("--learning-rate", "0"),
        ("--epochs", "0"),
        ("--evidence-weight", "inf"),
        ("--max-steps", "0"),
        ("--max-steps", "-2"),
        ("--batch-size", "0"),
        ("--gradient-accumulation", "0"),
        ("--max-length", "0"),
        ("--max-completion-length", "0"),
        ("--save-steps", "0"),
        ("--save-total-limit", "0"),
    ],
)
def test_invalid_args_fail_before_runtime_imports(tmp_path, options):
    with pytest.raises(SystemExit):
        train.parse_args(argv(tmp_path, "dpo", *options))


@pytest.mark.parametrize(
    "options",
    [("--num-generations", "1"), ("--num-generations", "3"), ("--max-length", "1536")],
)
def test_grpo_batch_and_length_args_fail_early(tmp_path, options):
    with pytest.raises(SystemExit):
        train.parse_args(argv(tmp_path, "grpo", *options))


def test_rank_unsafe_multi_process_launch_is_explicitly_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit):
        train.parse_args(argv(tmp_path))


def test_checkpoint_budget_and_nondefault_adapter_paths(tmp_path):
    args = train.parse_args(argv(tmp_path, "dpo", "--max-steps", "2"))
    assert train.checkpoint_kwargs(args) == {
        "save_strategy": "steps",
        "save_steps": 2,
        "save_total_limit": 3,
    }
    args = train.parse_args(
        argv(
            tmp_path,
            "dpo",
            "--max-steps",
            "64",
            "--save-steps",
            "16",
            "--save-total-limit",
            "4",
        )
    )
    assert train.checkpoint_kwargs(args)["save_steps"] == 16
    assert (
        train.checkpoint_kwargs(train.parse_args(argv(tmp_path)))["save_strategy"]
        == "epoch"
    )
    assert (
        train.adapter_path(tmp_path / "checkpoint-16", "dpo")
        == tmp_path / "checkpoint-16" / "policy"
    )
    assert train.adapter_path(tmp_path / "final", "grpo") == tmp_path / "final"


@pytest.mark.parametrize("split", ["tune", "test", "regression", None])
def test_callable_builder_cannot_launder_non_train_rows(split):
    row = silver_row()
    row["split"] = split
    with pytest.raises(ValueError, match="train-tagged"):
        list(preferences.omission_preferences([row]))


def test_invalid_target_and_duplicate_id_fail_closed():
    row = silver_row()
    with pytest.raises(ValueError, match="Duplicate"):
        train.validate_training_rows([row, row], "grpo")
    with pytest.raises(ValueError, match="Completion/target mismatch"):
        list(preferences.omission_preferences([{**row, "target": {"decisions": []}}]))
    with pytest.raises(ValueError, match="Invalid completion"):
        list(
            preferences.omission_preferences(
                [{**row, "completion": '{"decisions":[]}'}]
            )
        )


def test_preferences_are_valid_distinct_auditable_and_not_human_gold():
    row = silver_row()
    original = copy.deepcopy(row)
    pair = list(preferences.omission_preferences([row]))[0]
    assert row == original
    assert pair["source_row_id"] == row["id"] and not pair["human_adjudicated"]
    assert pair["provenance"].startswith("teacher_silver")
    assert (
        "source_row_id" not in pair["prompt"]
        and "human_adjudicated" not in pair["prompt"]
    )
    for key in ("prompt", "chosen", "rejected"):
        assert pair[key + "_sha256"] == hashlib.sha256(pair[key].encode()).hexdigest()
    train.validate_training_rows([pair], "dpo")
    synthetic = list(preferences.synthetic_preferences([0, 1]))
    assert len(synthetic) == 8 and len({p["id"] for p in synthetic}) == 8
    assert len({p["reason"] for p in synthetic}) == 4
    assert all(not p["human_adjudicated"] and p["hard_semantic"] for p in synthetic)


def test_dpo_rejects_missing_malformed_and_formatting_only_pairs():
    pair = next(preferences.omission_preferences([silver_row()]))
    for rejected in (
        None,
        "",
        "not JSON",
        '{"decisions":[]}',
        json.dumps(json.loads(pair["chosen"]), indent=2),
    ):
        with pytest.raises(ValueError):
            train.validate_training_rows([{**pair, "rejected": rejected}], "dpo")
    bad = json.loads(pair["chosen"])
    next(d for d in bad["decisions"] if d["relations"])["relations"][0][
        "evidence_span_ids"
    ] = ["INVENTED"]
    with pytest.raises(ValueError, match="Invalid chosen"):
        train.validate_training_rows([{**pair, "chosen": dumps(bad)}], "dpo")


def test_prompt_payload_rejects_quality_metadata_and_double_rendering():
    row = silver_row()
    lines = row["prompt"].strip().splitlines()
    payload = json.loads(lines[-1])
    payload["entities"][0]["is_noise"] = True
    poisoned = {**row, "prompt": "\n".join(lines[:-1] + [dumps(payload)])}
    with pytest.raises(ValueError, match="metadata leakage"):
        train.validate_training_rows([poisoned], "grpo")
    with pytest.raises(ValueError, match="already chat-rendered"):
        train.validate_training_rows(
            [{**row, "prompt": "<|im_start|>" + row["prompt"]}], "grpo"
        )


def test_grpo_reward_columns_stay_out_of_prompt_and_batch_metadata():
    row = silver_row()
    dataset = train.grpo_dataset_rows(
        [{**row, "reward_batch": train.prompt_batch(row), "is_noise": True}]
    )
    assert set(dataset[0]) == {"prompt", "batch_json", "target_json"}
    assert dataset[0]["prompt"] == row["prompt"]
    batch = json.loads(dataset[0]["batch_json"])
    assert set(batch) == {"id", "pairs", "spans"}
    assert (
        "target" not in batch and "completion" not in batch and "is_noise" not in batch
    )


def test_reward_alignment_eos_tokens_and_exact_rollout_counts(tmp_path):
    row = silver_row()
    counters = Counter()
    reward = train.make_relation_reward(tmp_path / "rollouts.jsonl", 0.2, 999, counters)
    batch = dumps(train.prompt_batch(row))
    with pytest.raises(ValueError, match="length mismatch"):
        reward([row["completion"]], [], [dumps(row["target"])], [[1, 999]])
    assert not (tmp_path / "rollouts.jsonl").exists()
    scores = reward(
        [row["completion"], "{"],
        [batch] * 2,
        [dumps(row["target"])] * 2,
        [[1, 2, 999], [1, 2]],
        trainer_state=SimpleNamespace(global_step=3),
    )
    assert scores[0] > scores[1]
    logs = train.read_rows(tmp_path / "rollouts.jsonl")
    assert logs[0]["eos_observed"] and not logs[0]["truncated"]
    assert logs[1]["truncated"] and logs[1]["invalid"] and logs[1]["step"] == 3
    assert counters == {
        "online_rollouts": 2,
        "rollout_completion_tokens": 5,
        "truncated_rollouts": 1,
    }


def test_reward_nonfinite_or_conversational_outputs_are_rejected(tmp_path, monkeypatch):
    row = silver_row()
    reward = train.make_relation_reward(
        tmp_path / "rollouts.jsonl", 0.2, 999, Counter()
    )
    batch = [dumps(train.prompt_batch(row))]
    target = [dumps(row["target"])]
    with pytest.raises(ValueError, match="plain string"):
        reward(
            [[{"role": "assistant", "content": row["completion"]}]],
            batch,
            target,
            [[1]],
        )
    monkeypatch.setattr(
        train, "reward_components", lambda *a, **k: {"reward": float("nan")}
    )
    with pytest.raises(ValueError, match="Non-finite"):
        reward([row["completion"]], batch, target, [[1]])


@pytest.mark.parametrize("mode", ["dpo", "grpo"])
def test_reload_constructs_independent_base_and_correct_adapter_chain(tmp_path, mode):
    args = train.parse_args(argv(tmp_path, mode))
    loader, peft = Mock(), Mock()
    base, merged, result = object(), object(), object()
    loader.from_pretrained.return_value = base
    sft = Mock()
    sft.merge_and_unload.return_value = merged
    peft.from_pretrained.side_effect = [sft, result] if mode == "grpo" else [result]
    saved = train.adapter_path(tmp_path / "final", mode)
    assert (
        train.load_saved_policy(args, saved, loader, peft, {"local_files_only": True})
        is result
    )
    loader.from_pretrained.assert_called_once_with(
        args.base_model, local_files_only=True
    )
    if mode == "grpo":
        assert peft.from_pretrained.call_args_list[0].args == (base, args.sft_adapter)
        sft.merge_and_unload.assert_called_once_with()
    assert peft.from_pretrained.call_args.args == (
        merged if mode == "grpo" else base,
        str(saved),
    )
    assert peft.from_pretrained.call_args.kwargs == {
        "is_trainable": False,
        "local_files_only": True,
    }


def test_terminal_verifier_removes_nested_autocast_and_matches_eval_cache_mode():
    trainer, wrapped, plain = Mock(), Mock(), Mock()
    trainer.accelerator.unwrap_model.return_value = plain
    assert train.terminal_inference_model(trainer, wrapped, "policy") is plain
    trainer.accelerator.unwrap_model.assert_called_once_with(wrapped, keep_fp32_wrapper=False)
    plain.eval.assert_called_once_with()
    plain.set_adapter.assert_called_once_with("policy")
    assert plain.config.use_cache is False


def test_saved_weight_gate_is_exact_and_never_replaced_by_logit_tolerance(monkeypatch):
    monkeypatch.setattr(train, "adapter_fingerprint", lambda model, adapter: "actual")
    assert train.require_saved_weight_equality(object(), "actual") == "actual"
    with pytest.raises(RuntimeError, match="recorded trained weights"):
        train.require_saved_weight_equality(object(), "rounded")


class Parameter:
    """Tiny mock tensor, not torch; hashing is tested without any model runtime."""

    shape = (1,)

    def __init__(self, value, trainable=False):
        self.value, self.requires_grad = value, trainable

    def numel(self):
        return 1

    def detach(self):
        return self

    float = cpu = contiguous = numpy = detach

    def tobytes(self):
        return struct.pack("f", self.value)


def test_adapter_fingerprint_ignores_name_but_detects_weight_change():
    reference = Parameter(1)
    model = SimpleNamespace(
        named_parameters=lambda: [
            ("layer.lora_A.policy.weight", Parameter(1)),
            ("layer.lora_A.reference.weight", reference),
        ]
    )
    assert train.adapter_fingerprint(model, "policy") == train.adapter_fingerprint(
        model, "reference"
    )
    reference.value = 2
    assert train.adapter_fingerprint(model, "policy") != train.adapter_fingerprint(
        model, "reference"
    )
    with pytest.raises(RuntimeError, match="No LoRA"):
        train.adapter_fingerprint(model, "missing")


@pytest.mark.parametrize("rounded_first", [False, True])
def test_second_adapter_precision_reload_is_bounded_and_exact(rounded_first):
    class Model:
        def __init__(self):
            self.calls = 0
            self.reference = Parameter(0)
        def named_parameters(self):
            return [("layer.lora_A.policy.weight", Parameter(1)), ("layer.lora_A.reference.weight", self.reference)]
        def load_adapter(self, checkpoint, **kwargs):
            assert checkpoint == "fixed-sft" and kwargs["is_trainable"] is False
            self.calls += 1
            self.reference.value = .999 if rounded_first and self.calls == 1 else 1
        def set_adapter(self, name):
            assert name == "policy"
    model = Model()
    result = train.initialize_dpo_reference(model, "fixed-sft")
    assert result["initial_adapters_bitwise_equal"]
    assert result["reference_load_attempts"] == 1 + rounded_first
    assert model.calls == 1 + rounded_first


def test_reference_reload_never_disables_equality_gate():
    calls = []
    model = SimpleNamespace(
        named_parameters=lambda: [("layer.lora_A.policy.weight", Parameter(1)), ("layer.lora_A.reference.weight", Parameter(2))],
        load_adapter=lambda *a, **kw: calls.append(kw), set_adapter=lambda name: None,
    )
    with pytest.raises(RuntimeError, match="Exact SFT reference"):
        train.initialize_dpo_reference(model, "fixed-sft")
    assert len(calls) == 2


class Mask:
    def __init__(self, count):
        self.count = count

    def sum(self):
        return SimpleNamespace(item=lambda: self.count)

    def __ne__(self, ignored):
        return self


def test_sft_weighted_index_counts_do_not_reach_collator_or_model():
    items = [{"input_ids": [1], "labels": [1]}, {"input_ids": [2], "labels": [2]}]
    original = copy.deepcopy(items)
    dataset = train.PairCountDataset(items, [{"pairs": [0] * 16}, {"pairs": [0] * 4}])
    # Repeated/shuffled sampler indices must repeat the corresponding counts.
    sampled = [dataset[i] for i in [1, 0, 1]]
    delegate = Mock(
        return_value={
            "input_ids": [[2], [1], [2]],
            "attention_mask": Mask(20),
            "labels": Mask(8),
        }
    )
    batch = train.PairCountCollator(delegate)(sampled)
    assert all("_pair_count" not in item for item in delegate.call_args.args[0])
    assert batch["_pair_counts"] == [4, 16, 4] and items == original
    counters = Counter()
    counts = batch.pop("_pair_counts")
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    train.record_sft_batch(counters, batch, counts)
    assert counters == {
        "sft_sequence_loss_exposures": 3,
        "sft_pair_loss_exposures": 24,
        "sft_input_token_loss_exposures": 20,
        "sft_completion_token_loss_exposures": 8,
        "sft_prompt_token_loss_exposures": 12,
    }
    with pytest.raises(ValueError, match="alignment mismatch"):
        train.record_sft_batch(counters, batch, [4])
    with pytest.raises(ValueError, match="alignment mismatch"):
        train.PairCountDataset(items, [{"pairs": []}])


def test_dpo_counts_pairs_not_two_independent_prompts_or_reference_passes():
    counters = Counter()
    train.record_dpo_batch(
        counters,
        {
            "prompt_input_ids": [[1, 2], [3]],
            "prompt_attention_mask": Mask(3),
            "chosen_attention_mask": Mask(7),
            "rejected_attention_mask": Mask(5),
        },
    )
    assert counters == {
        "dpo_pair_loss_exposures": 2,
        "dpo_prompt_token_loss_exposures": 6,
        "dpo_completion_token_loss_exposures": 12,
    }


def test_preference_cli_counts_hashes_and_no_overwrite(tmp_path, monkeypatch):
    rows = [silver_row(str(i), group=0) for i in range(8)]
    source, out = tmp_path / "train.jsonl", tmp_path / "preferences"
    source.write_text("".join(dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_preferences",
            "--data",
            str(source),
            "--output-dir",
            str(out),
            "--synthetic-groups",
            "1",
        ],
    )
    preferences.main()
    manifest = json.loads((out / "manifest.json").read_text())
    assert (
        manifest["balanced_training_rows"] == manifest["size_matched_no_hard_rows"] == 8
    )
    assert manifest["rows"] == 12 and manifest["human_adjudicated_rows"] == 0
    assert (
        manifest["source_data_sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    for name, audit in manifest["subsets"].items():
        path = out / (name + ".jsonl")
        assert len(train.read_rows(path)) == audit["rows"] == audit["unique_ids"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["sha256"]
    before = (out / "full.jsonl").read_bytes()
    with pytest.raises(ValueError, match="never overwrite"):
        preferences.main()
    assert (out / "full.jsonl").read_bytes() == before


@pytest.mark.parametrize("mode", ["sft", "continued_sft", "dpo", "grpo"])
def test_entrypoint_wiring_with_mock_runtime_only(tmp_path, monkeypatch, mode):
    """Exercise real main/config wiring with fake trainers; never import torch."""
    continued = mode == "continued_sft"
    mode = "sft" if continued else mode
    base_parameters = {"base.weight": Parameter(1)}
    saved_weights, peft_calls = {}, []

    class Logits:
        def __getitem__(self, index):
            return self
        def float(self):
            return self
        cpu = abs = float
        def __sub__(self, other):
            return self
        def max(self):
            return 0.0

    class Model:
        training = True
        config = SimpleNamespace(max_position_embeddings=8192)
        device = SimpleNamespace(type="cuda")

        def __init__(self):
            self.params = dict(base_parameters)

        def load_adapter(self, path, adapter_name, is_trainable=False, **kwargs):
            self.params[f"layer.lora_A.{adapter_name}.weight"] = Parameter(
                1, is_trainable
            )

        def set_adapter(self, name):
            self.active = name
            for key, value in self.params.items():
                value.requires_grad = f".{name}." in key

        def merge_and_unload(self):
            self.params = dict(base_parameters)
            return self

        def named_parameters(self):
            return list(self.params.items())

        def parameters(self):
            return list(self.params.values())

        def enable_input_require_grads(self):
            self.input_gradients_enabled = True

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=Logits())

        def to(self, device):
            return self

        def eval(self):
            self.training = False
            return self

    def from_peft(model, path, adapter_name="default", is_trainable=False, **kwargs):
        peft_calls.append((str(path), adapter_name, is_trainable))
        model.load_adapter(path, adapter_name, is_trainable)
        if str(path) in saved_weights:
            model.params[f"layer.lora_A.{adapter_name}.weight"].value = saved_weights[str(path)]
        return model

    def get_peft(model, config):
        assert not continued, "Continued SFT must not initialize fresh LoRA weights"
        model.load_adapter("new", "default", True)
        return model

    trainers = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.state = SimpleNamespace(
                global_step=0,
                save_to_json=lambda path: Path(path).write_text('{"mock":true}'),
            )
            trainers.append(self)
            self.accelerator = SimpleNamespace(unwrap_model=lambda model, **kw: model)

        def compute_loss(self, model, inputs, **kwargs):
            assert "_pair_counts" not in inputs
            return 0.5

        def train(self):
            self.state.global_step = self.args.max_steps
            if mode == "sft":
                self.compute_loss(
                    self.model,
                    {
                        "input_ids": [[1, 2, 3]],
                        "attention_mask": Mask(3),
                        "labels": Mask(2),
                        "_pair_counts": [16],
                    },
                )
            elif mode == "dpo":
                self.compute_loss(
                    self.model,
                    {
                        "prompt_input_ids": [[1, 2]],
                        "prompt_attention_mask": Mask(2),
                        "chosen_attention_mask": Mask(3),
                        "rejected_attention_mask": Mask(2),
                    },
                )
            else:
                row = self.train_dataset[0]
                self.reward_funcs(
                    [row["target_json"]] * 4,
                    [row["batch_json"]] * 4,
                    [row["target_json"]] * 4,
                    [[1, 999]] * 4,
                    trainer_state=self.state,
                )
            name = "policy" if mode == "dpo" else "default"
            self.model.params[f"layer.lora_A.{name}.weight"].value += 0.25
            return SimpleNamespace(metrics={"mock_train_loss": 0.5})

        def save_model(self, path):
            train.adapter_path(path, mode).mkdir(parents=True)
            name = "policy" if mode == "dpo" else "default"
            saved_weights[str(train.adapter_path(path, mode))] = self.model.params[f"layer.lora_A.{name}.weight"].value

    model_loader = Mock()
    model_loader.from_pretrained.side_effect = lambda *a, **k: Model()
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,
        max_memory_allocated=lambda: 0,
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=fake_cuda, bfloat16="mock_bf16",
            inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext())
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            LoraConfig=lambda **k: k,
            PeftModel=SimpleNamespace(from_pretrained=from_peft),
            get_peft_model=get_peft,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=model_loader,
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            Trainer=FakeTrainer,
            TrainingArguments=lambda **k: SimpleNamespace(**k),
            set_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(
            DPOConfig=lambda **k: SimpleNamespace(**k),
            GRPOConfig=lambda **k: SimpleNamespace(**k),
            DPOTrainer=FakeTrainer,
            GRPOTrainer=FakeTrainer,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset=SimpleNamespace(from_list=lambda rows: rows)),
    )
    monkeypatch.setattr(
        train.importlib.metadata,
        "version",
        lambda module: "0.23.1" if module == "trl" else "MOCK",
    )
    row = silver_row()
    if mode == "dpo":
        row = next(preferences.omission_preferences([row]))
    (tmp_path / "data.jsonl").write_text(dumps(row) + "\n")
    continuation_args = []
    if continued:
        parent_model = Model()
        parent_model.load_adapter("parent", "default")
        fingerprint = train.adapter_fingerprint(parent_model, "default")
        parent_path = tmp_path / "parent.json"
        parent_path.write_text(json.dumps({"status": "training_complete_evaluation_pending",
            "arguments": {"mode": "sft", "base_model": str(tmp_path / "base")}, "policy_adapter_sha256_after": fingerprint,
            "global_step": 64, "data_sha256": "parent-data"}))
        continuation_args = ["--sft-adapter", str(tmp_path / "grounded64" / "final"),
                             "--parent-receipt", str(parent_path)]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            *argv(tmp_path, mode, "--max-steps", "2", "--gradient-accumulation", "4"),
            *continuation_args,
        ],
    )
    train.main()
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["global_step"] == 2 and manifest["policy_weights_changed"]
    assert manifest["final_adapter"] == str(
        train.adapter_path(tmp_path / "out" / "final", mode)
    )
    config = trainers[0].args
    assert config.remove_unused_columns == (mode == "dpo")
    assert (
        config.max_steps == 2
        and config.save_strategy == "steps"
        and config.save_steps == 2
    )
    if mode == "dpo":
        assert config.max_prompt_length is None
        assert (
            manifest["reference_adapter_sha256_before"]
            == manifest["reference_adapter_sha256_after"]
        )
        assert set(trainers[0].train_dataset[0]) == {"prompt", "chosen", "rejected"}
    elif mode == "grpo":
        assert config.gradient_checkpointing_kwargs == {"use_reentrant": True}
        assert trainers[0].model.input_gradients_enabled
        assert "TRL0.23.1" in manifest["gradient_checkpointing_compatibility"]
        assert config.max_prompt_length + config.max_completion_length == 6144
        assert config.steps_per_generation == 4 and config.num_generations == 4
        assert set(trainers[0].train_dataset[0]) == {
            "prompt",
            "batch_json",
            "target_json",
        }
        assert manifest["observed_exposure"]["online_rollouts"] == 4
    else:
        assert manifest["observed_exposure"]["sft_pair_loss_exposures"] == 16
        assert manifest["observed_exposure"]["sft_sequence_loss_exposures"] == 1
        assert manifest["observed_exposure"]["sft_completion_token_loss_exposures"] == 2
    assert model_loader.from_pretrained.call_count == (2 if continued else 1)
    if continued:
        assert peft_calls[0] == (str(tmp_path / "grounded64" / "final"), "default", True)
        assert manifest["parent_training_receipt"]["initialization_verified"]
        assert manifest["policy_adapter_sha256_before"] == fingerprint
        assert manifest["saved_adapter_sha256_verified"] == manifest["policy_adapter_sha256_after"]
        assert manifest["arguments"]["verify_save"]
        assert not manifest["parent_training_receipt"]["optimizer_state_resumed"]


def test_main_does_not_overwrite_preexisting_output(tmp_path, monkeypatch):
    pair = next(preferences.omission_preferences([silver_row()]))
    (tmp_path / "data.jsonl").write_text(dumps(pair) + "\n")
    (tmp_path / "out").mkdir()
    existing = tmp_path / "out" / "run_manifest.json"
    existing.write_text("existing result")
    monkeypatch.setattr(sys, "argv", ["train", *argv(tmp_path)])
    with pytest.raises(ValueError, match="never overwrite"):
        train.main()
    assert existing.read_text() == "existing result"


def test_runtime_failure_is_persisted_as_failure_not_preflight_success(
    tmp_path, monkeypatch
):
    pair = next(preferences.omission_preferences([silver_row()]))
    (tmp_path / "data.jsonl").write_text(dumps(pair) + "\n")
    monkeypatch.setattr(sys, "argv", ["train", *argv(tmp_path)])
    monkeypatch.setattr(
        train, "execute", Mock(side_effect=RuntimeError("mock runtime failure"))
    )
    with pytest.raises(RuntimeError, match="mock runtime failure"):
        train.main()
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error_type"] == "RuntimeError"

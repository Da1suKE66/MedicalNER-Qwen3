"""Explicit zero-step evaluator selection; mocked loading, never GPU execution."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from generative.evaluate import load_evaluation_model, main


@pytest.mark.parametrize("arguments", [[], ["--adapter", "adapter", "--base-only"],
                                        ["--base-only", "--grpo-sft-adapter", "sft"]])
def test_cli_rejects_missing_conflicting_or_merged_base_controls_before_io(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", [
        "generative.evaluate", "--gold", "must-not-read", "--base-model", "base",
        "--variant", "unused", "--output-dir", "must-not-create", *arguments,
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize("base_only, adapter, sft, expected_chain", [
    (True, None, None, []),
    (False, "trained", None, ["trained"]),
    (False, "residual", "sft", ["sft", "residual"]),
])
def test_loading_paths_keep_base_dtype_attention_and_explicit_adapter_chain(
    monkeypatch, base_only, adapter, sft, expected_chain,
):
    base, merged, wrapped = Mock(), Mock(), Mock()
    base.to.return_value = base
    base.eval.return_value = base
    wrapped.to.return_value = wrapped
    wrapped.eval.return_value = wrapped
    base_loader = Mock(return_value=base)
    peft_loader = Mock()
    if sft:
        initial = Mock()
        initial.merge_and_unload.return_value = merged
        peft_loader.side_effect = [initial, wrapped]
    else:
        peft_loader.return_value = wrapped
    dtype = object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16=dtype))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=base_loader)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=peft_loader)))
    result = load_evaluation_model("base", adapter, base_only=base_only, grpo_sft_adapter=sft)
    base_loader.assert_called_once_with(
        "base", torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True)
    assert [call.args[1] for call in peft_loader.call_args_list] == expected_chain
    if sft:
        assert peft_loader.call_args_list[1].args[0] is merged
    assert result is (base if base_only else wrapped)
    result.to.assert_called_once_with("cuda")
    result.eval.assert_called_once_with()


@pytest.mark.parametrize("base_only, adapter, sft", [
    (False, None, None), (True, "adapter", None), (True, None, "sft"),
])
def test_direct_loader_also_rejects_ambiguous_conditions(base_only, adapter, sft):
    with pytest.raises(ValueError, match="--adapter"):
        load_evaluation_model("must-not-load", adapter, base_only=base_only, grpo_sft_adapter=sft)

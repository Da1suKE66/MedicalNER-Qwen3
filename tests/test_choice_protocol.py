"""No GPU/model: exact finite-choice token paths and partial-output accounting."""

from collections import Counter
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from kg_agent.choice_protocol import ChoiceProtocol, ChoiceProtocolError


class Tokenizer:
    eos_token_id = 999
    pad_token_id = 998
    bos_token_id = 997
    all_special_ids = [999, 998, 997]

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.calls = Counter()

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls[text] += 1
        # Deliberately nonadditive tokenization of concatenated actions. Training
        # must call this for each complete action independently.
        if text.count("\n") > 1:
            return [777]
        return self.overrides.get(text, list(map(ord, text)))


def task(actions):
    if not actions:
        return [0, 1, 12]
    if actions[0] == 0 or len(actions) == 2:
        return None
    return [2, 21]


def codec(tokenizer=None, choices_fn=task, prompt_length=2):
    return ChoiceProtocol(tokenizer or Tokenizer(), prompt_length, choices_fn)


def test_hf_callback_restricts_each_token_and_allows_eos_only_at_terminal():
    protocol = codec()
    prompt = [71, 72]
    assert protocol.prefix_allowed_tokens_fn(0, prompt) == [ord("0"), ord("1")]
    assert protocol.prefix_allowed_tokens_fn(0, prompt + [ord("1")]) == [10, ord("2")]
    assert protocol.prefix_allowed_tokens_fn(0, prompt + list(map(ord, "12"))) == [10]
    assert protocol.prefix_allowed_tokens_fn(0, prompt + list(map(ord, "12\n"))) == [ord("2")]
    complete = list(map(ord, "12\n21\n"))
    assert protocol.prefix_allowed_tokens_fn(0, prompt + complete) == [999]
    result = protocol.read_result(complete)
    assert result.actions == (12, 21) and result.complete and not result.eos_observed


@pytest.mark.parametrize("actions", [(0,), (1, 2), (1, 21), (12, 2), (12, 21)])
def test_training_and_inference_paths_are_identical(actions):
    tokenizer = Tokenizer()
    protocol = codec(tokenizer)
    tokens = protocol.encode_actions(actions)
    expected = [token for action in actions for token in map(ord, f"{action}\n")] + [999]
    assert tokens == expected
    prefix = [90, 91]
    for token in tokens:
        assert token in protocol.prefix_allowed_tokens_fn(0, prefix)
        prefix.append(token)
    result = protocol.read_result(tokens, max_new_tokens=len(tokens))
    assert result.actions == actions and result.complete and result.eos_observed
    assert not result.budget_exhausted and not result.partial_action and result.error is None
    assert all(text.count("\n") == 1 for text in tokenizer.calls)


@pytest.mark.parametrize("suffix,budget,actions,partial", [
    ([], 0, (), False),
    ([ord("1")], 1, (), True),
    (list(map(ord, "12\n")), 3, (12,), False),
    (list(map(ord, "12\n2")), 4, (12,), True),
])
def test_generation_budget_never_marks_half_actions_or_nonterminal_tasks_complete(suffix, budget, actions, partial):
    result = codec().read_result(suffix, max_new_tokens=budget)
    assert result.actions == actions and result.partial_action is partial
    assert not result.complete and result.budget_exhausted and result.error is None


def test_terminal_at_last_budget_token_is_complete_without_eos():
    result = codec().read_result(list(map(ord, "0\n")), max_new_tokens=2)
    assert result.complete and not result.eos_observed and not result.budget_exhausted


def test_early_stop_below_budget_remains_incomplete_without_claiming_budget_exhaustion():
    result = codec().read_result(list(map(ord, "12\n")), max_new_tokens=100)
    assert not result.complete and not result.budget_exhausted and not result.partial_action


@pytest.mark.parametrize("suffix,error,actions,eos", [
    ([999], "eos_before_terminal", (), False),
    ([ord("1"), 999], "eos_before_terminal", (), False),
    ([ord("9")], "token_outside_current_choices", (), False),
    (list(map(ord, "12\n9")), "token_outside_current_choices", (12,), False),
    (list(map(ord, "0\n1")), "tokens_after_terminal", (0,), False),
    (list(map(ord, "0\n")) + [999, 999], "tokens_after_eos", (0,), True),
    (list(map(ord, "0\n")) + [999, 998], "tokens_after_eos", (0,), True),
])
def test_invalid_outputs_are_explicit_and_never_repaired(suffix, error, actions, eos):
    result = codec().read_result(suffix, max_new_tokens=len(suffix))
    assert error in result.error and result.actions == actions
    assert not result.complete and result.eos_observed is eos
    assert not result.budget_exhausted
    with pytest.raises(ChoiceProtocolError):
        codec().prefix_allowed_tokens_fn(0, [55, 66] + suffix)


@pytest.mark.parametrize("actions", [[], [1], [9], [1, 0], [0, 1], [-1], [True], [1.5]])
def test_training_rejects_incomplete_unavailable_and_invalid_actions(actions):
    with pytest.raises(ChoiceProtocolError):
        codec().encode_actions(actions)


def test_partial_training_prefix_never_teaches_an_illegal_early_eos():
    protocol = codec()
    assert protocol.encode_actions([12], require_complete=False) == list(map(ord, "12\n"))
    assert protocol.encode_actions([], require_complete=False) == []
    assert protocol.encode_actions([0], require_complete=False) == list(map(ord, "0\n")) + [999]


def test_empty_task_has_only_eos_and_zero_actions():
    protocol = codec(choices_fn=lambda actions: None)
    assert protocol.prefix_allowed_tokens_fn(0, [6, 7]) == [999]
    assert protocol.encode_actions([]) == [999]
    assert protocol.read_result([]).complete
    assert protocol.read_result([999]).eos_observed


@pytest.mark.parametrize("options", [[], [1, 1], [True], [-1], [1.2], "12"])
def test_invalid_task_choices_fail_closed(options):
    with pytest.raises(ChoiceProtocolError):
        codec(choices_fn=lambda actions: options)


@pytest.mark.parametrize("overrides,error", [
    ({"1\n": []}, "empty"),
    ({"1\n": [998]}, "special"),
    ({"1\n": [999]}, "special"),
    ({"1\n": [-2]}, "nonnegative"),
    ({"1\n": [True]}, "nonnegative"),
    ({"1\n": [41], "12\n": [41]}, "collision"),
    ({"1\n": [41], "12\n": [41, 42]}, "prefix-free"),
    ({"1\n": [41, 42], "12\n": [41]}, "prefix-free"),
])
def test_invalid_tokenizer_encodings_are_rejected(overrides, error):
    with pytest.raises(ChoiceProtocolError, match=error):
        codec(Tokenizer(overrides))


def test_special_inventory_and_explicit_eos_are_both_enforced():
    tokenizer = Tokenizer({"1\n": [1234]})
    tokenizer.all_special_ids = [1234]
    with pytest.raises(ChoiceProtocolError, match="special"):
        codec(tokenizer)
    tokenizer = Tokenizer({"1\n": [555]})
    with pytest.raises(ChoiceProtocolError, match="special"):
        ChoiceProtocol(tokenizer, 0, task, eos_token_id=555)
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ChoiceProtocolError, match="eos_token_id"):
        codec(tokenizer)


def test_lazy_state_codec_collisions_also_fail_closed():
    tokenizer = Tokenizer({"1\n": [11], "2\n": [11]})
    protocol = codec(tokenizer, choices_fn=lambda actions: [1] if not actions else [2])
    result = protocol.read_result([11])
    assert not result.complete and "collision" in result.error


def test_single_sample_boundary_and_tensor_shape_are_explicit():
    protocol = codec()
    with pytest.raises(ChoiceProtocolError, match="batch_id=0"):
        protocol.prefix_allowed_tokens_fn(1, [1, 2])
    with pytest.raises(ChoiceProtocolError, match="prompt_length"):
        protocol.prefix_allowed_tokens_fn(0, [1])
    with pytest.raises(ChoiceProtocolError):
        protocol.prefix_allowed_tokens_fn(0, [[1, 2]])
    class Tensor:
        def tolist(self):
            return [1, 2, ord("0")]
    assert protocol.prefix_allowed_tokens_fn(0, Tensor()) == [10]


def test_cached_states_do_not_repeat_task_callbacks_and_are_branch_safe():
    calls = Counter()
    def choices(actions):
        calls[actions] += 1
        return task(actions)
    tokenizer = Tokenizer()
    protocol = codec(tokenizer, choices_fn=choices)
    for _ in range(3):
        assert protocol.read_result(list(map(ord, "12\n21\n"))).actions == (12, 21)
        assert protocol.read_result(list(map(ord, "0\n"))).actions == (0,)
        protocol.prefix_allowed_tokens_fn(0, [11, 22] + list(map(ord, "1\n")))
    assert all(count == 1 for count in calls.values())
    assert all(count == 1 for count in tokenizer.calls.values())


def test_prompt_tokens_are_never_parsed_as_actions():
    protocol = codec(prompt_length=3)
    assert protocol.prefix_allowed_tokens_fn(0, [999, 998, 997]) == [ord("0"), ord("1")]
    result = protocol.read_result([True])
    assert not result.complete and result.error is not None

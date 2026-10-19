"""Finite integer choices for an existing causal LM, without generated JSON.

The task owns the state machine: ``choices_fn(completed_actions)`` returns the
available nonnegative integers, or ``None`` when the task is complete. Each
integer is independently tokenized as ASCII decimal plus a newline. Training
must use ``encode_actions`` rather than tokenize a concatenated output string:
tokenization across action boundaries is not assumed to be additive.

One protocol instance belongs to one sample and one fixed prompt length. Pass
``prefix_allowed_tokens_fn`` to Hugging Face ``generate`` and explicitly use the
same ``eos_token_id``. This constrains syntax and permitted actions, not the
semantic correctness of the selected actions. No model or torch import occurs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any


ChoicesFn = Callable[[tuple[int, ...]], Sequence[int] | None]


class ChoiceProtocolError(ValueError):
    """An invalid codec, task state, or generated token sequence."""


@dataclass(frozen=True)
class ChoiceResult:
    actions: tuple[int, ...]
    complete: bool
    eos_observed: bool
    budget_exhausted: bool
    partial_action: bool
    error: str | None = None


@dataclass
class _Trie:
    children: dict[int, "_Trie"] = field(default_factory=dict)
    action: int | None = None


@dataclass(frozen=True)
class _State:
    actions: tuple[int, ...]
    node: _Trie | None  # None is a task-terminal boundary, not a dead end.
    pending: tuple[int, ...] = ()
    eos: bool = False


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ChoiceProtocolError(f"{description} must be a nonnegative integer, not bool")
    return int(value)


def _integers(values: Any, description: str) -> tuple[int, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ChoiceProtocolError(f"{description} must be a one-dimensional integer sequence")
    return tuple(_integer(value, description) for value in values)


class ChoiceProtocol:
    def __init__(self, tokenizer, prompt_length: int, choices_fn: ChoicesFn,
                 eos_token_id: int | None = None):
        self.tokenizer = tokenizer
        self.prompt_length = _integer(prompt_length, "prompt_length")
        if not callable(choices_fn):
            raise ChoiceProtocolError("choices_fn must be callable")
        self.choices_fn = choices_fn
        eos = getattr(tokenizer, "eos_token_id", None) if eos_token_id is None else eos_token_id
        self.eos_token_id = _integer(eos, "eos_token_id")
        special = getattr(tokenizer, "all_special_ids", ()) or ()
        self._special_ids = set(_integers(special, "special token IDs"))
        # Include explicit EOS and common special IDs even for minimal tokenizers
        # whose all_special_ids inventory is incomplete.
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
                     "sep_token_id", "cls_token_id", "mask_token_id"):
            value = getattr(tokenizer, name, None)
            if value is not None:
                self._special_ids.add(_integer(value, name))
        self._special_ids.add(self.eos_token_id)
        self._encodings: dict[int, tuple[int, ...]] = {}
        self._encoding_owners: dict[tuple[int, ...], int] = {}
        self._boundaries: dict[tuple[int, ...], _State] = {}
        self._states: dict[tuple[int, ...], _State] = {(): self._boundary(())}

    def _encode_choice(self, action: int) -> tuple[int, ...]:
        if action not in self._encodings:
            ids = _integers(
                self.tokenizer.encode(f"{action}\n", add_special_tokens=False),
                "choice encoding",
            )
            if not ids:
                raise ChoiceProtocolError("Choice encoding cannot be empty")
            if self._special_ids.intersection(ids):
                raise ChoiceProtocolError("Choice encoding contains a special token")
            owner = self._encoding_owners.get(ids)
            if owner is not None and owner != action:
                raise ChoiceProtocolError("Distinct integer choices have a token encoding collision")
            self._encodings[action] = ids
            self._encoding_owners[ids] = action
        return self._encodings[action]

    def _boundary(self, actions: tuple[int, ...]) -> _State:
        if actions in self._boundaries:
            return self._boundaries[actions]
        options = self.choices_fn(actions)
        if options is None:
            state = _State(actions, None)
        else:
            choices = _integers(options, "choices")
            if not choices:
                raise ChoiceProtocolError("Empty choices are a dead end; use None for task completion")
            if len(set(choices)) != len(choices):
                raise ChoiceProtocolError("Duplicate integer choices are not allowed")
            root = _Trie()
            for action in choices:
                node = root
                for token in self._encode_choice(action):
                    if node.action is not None:
                        raise ChoiceProtocolError("Choice encodings must be prefix-free")
                    node = node.children.setdefault(token, _Trie())
                if node.children or node.action is not None:
                    raise ChoiceProtocolError("Choice encodings must be prefix-free and unique")
                node.action = action
            state = _State(actions, root)
        self._boundaries[actions] = state
        return state

    def _advance(self, state: _State, token: int) -> _State:
        if state.eos:
            raise ChoiceProtocolError("tokens_after_eos")
        if state.node is None:
            if token != self.eos_token_id:
                raise ChoiceProtocolError("tokens_after_terminal")
            return _State(state.actions, None, eos=True)
        if token == self.eos_token_id:
            raise ChoiceProtocolError("eos_before_terminal")
        node = state.node.children.get(token)
        if node is None:
            raise ChoiceProtocolError("token_outside_current_choices")
        if node.action is not None:
            return self._boundary(state.actions + (node.action,))
        return _State(state.actions, node, state.pending + (token,))

    def _state_for(self, tokens: tuple[int, ...]) -> _State:
        if tokens in self._states:
            return self._states[tokens]
        # HF normally appends one token; the immediate parent is then cached.
        # Branching calls are safe too: every cache key includes the full suffix.
        length = len(tokens) - 1
        while tokens[:length] not in self._states:
            length -= 1
        state = self._states[tokens[:length]]
        for index in range(length, len(tokens)):
            state = self._advance(state, tokens[index])
            self._states[tokens[:index + 1]] = state
        return state

    def prefix_allowed_tokens_fn(self, batch_id: int, input_ids) -> list[int]:
        """HF callback for a single sample; input_ids includes the fixed prompt."""
        if _integer(batch_id, "batch_id") != 0:
            raise ChoiceProtocolError("Only batch_id=0 is supported; use a separate codec per sample")
        ids = _integers(input_ids, "input_ids")
        if len(ids) < self.prompt_length:
            raise ChoiceProtocolError("input_ids is shorter than the fixed prompt_length")
        state = self._state_for(ids[self.prompt_length:])
        if state.eos:
            raise ChoiceProtocolError("Generation requested more tokens after EOS")
        if state.node is None:
            return [self.eos_token_id]
        return sorted(state.node.children)

    def encode_actions(self, actions: Sequence[int], require_complete: bool = True) -> list[int]:
        """Encode exactly the inference path and append one EOS at task completion.

        With require_complete=False a nonterminal action prefix is allowed, but
        receives no EOS: an early EOS would be illegal under the same protocol.
        """
        if not isinstance(require_complete, bool):
            raise ChoiceProtocolError("require_complete must be bool")
        chosen = _integers(actions, "actions")
        tokens: list[int] = []
        state = self._states[()]
        for action in chosen:
            if state.node is None:
                raise ChoiceProtocolError("Action supplied after task completion")
            previous_actions = state.actions
            tokens.extend(self._encode_choice(action))
            state = self._state_for(tuple(tokens))
            if state.pending or state.actions != previous_actions + (action,):
                # An unavailable action may happen to be a partial trie prefix;
                # never silently accept it as a legal completed training action.
                raise ChoiceProtocolError("Action is not available in the current task state")
        if require_complete and state.node is not None:
            raise ChoiceProtocolError("Training actions do not complete the task")
        if state.node is None:
            tokens.append(self.eos_token_id)
        return tokens

    def read_result(self, generated_ids, max_new_tokens: int | None = None) -> ChoiceResult:
        """Read the generated suffix only, retaining valid actions on failure.

        Completed actions are sufficient for complete=True; EOS is reported
        separately. No trailing token, including padding or a second EOS, is
        silently discarded. A valid partial prefix is incomplete, not repaired.
        """
        budget = None if max_new_tokens is None else _integer(max_new_tokens, "max_new_tokens")
        try:
            ids = _integers(generated_ids, "generated_ids")
        except ChoiceProtocolError as exc:
            return ChoiceResult((), False, False, False, False, str(exc))
        state, error = self._states[()], None
        for index in range(len(ids)):
            try:
                state = self._state_for(ids[:index + 1])
            except ChoiceProtocolError as exc:
                error = f"{exc} at generated token {index}"
                break
        complete = state.node is None and error is None
        return ChoiceResult(
            state.actions, complete, state.eos,
            budget is not None and len(ids) >= budget and not complete and error is None,
            bool(state.pending), error,
        )

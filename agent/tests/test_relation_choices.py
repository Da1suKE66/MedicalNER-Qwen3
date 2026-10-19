"""Finite causal-LM actions: input-only grounding and exact round trips."""

import copy
import itertools
from pathlib import Path

import pytest

from generative.protocol import ProtocolConfig, build_batches
from kg_agent.relation_choices import RelationChoiceTask


def batch():
    return {
        "entities": [
            {"id": "raw-d", "entity_id": "E002", "name": "Condition A", "label": "Disease"},
            {"id": "raw-s", "entity_id": "E017", "name": "Fatigue", "label": "Symptom"},
        ],
        "pairs": [
            {"pair_id": "P0099", "source": "raw-s", "target": "raw-d", "allowed": ["core", "associated"]},
            {"pair_id": "P0002", "source": "raw-d", "target": "raw-s", "allowed": ["presents", "causes"]},
        ],
        # Deliberately non-lexical IDs test ordering by visibility, not string.
        "spans": [
            {"span_id": "SENT_010", "text": "Fatigue accompanies Condition A."},
            {"span_id": "SENT_002", "text": "Further source evidence."},
        ],
    }


def edge(pair="P0099", relation="core", refs=None):
    return {"pair_id": pair, "relation": relation,
            "evidence_span_ids": ["SENT_010"] if refs is None else refs}


def test_state_choices_require_nonempty_pair_and_evidence_and_explicit_termination():
    task = RelationChoiceTask(batch())
    assert task.choices(()) == (0, 1, 2)
    assert task.choices((1,)) == (1, 2)
    assert task.choices((1, 1)) == (1, 2)
    assert task.choices((1, 1, 1)) == (0, 2)
    assert task.choices((1, 1, 1, 0)) == (0, 2)
    assert task.choices((1, 1, 1, 0, 0)) == (0, 2)
    assert task.choices((1, 1, 1, 0, 0, 0)) is None
    assert task.edges((0,)) == []
    assert task.encode_target([]) == (0,)
    for prefix in [(), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 0), (1, 1, 1, 0, 0)]:
        with pytest.raises(ValueError, match="Incomplete"):
            task.edges(prefix)


@pytest.mark.parametrize("actions", [
    (1, 0), (1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 0, 1),
    (1, 1, 1, 0, 0, 1), (0, 0), (3,), (-1,), (True,), (1.0,), ("1",),
])
def test_invalid_duplicates_out_of_range_and_post_terminal_actions_raise(actions):
    task = RelationChoiceTask(batch())
    with pytest.raises(ValueError, match="Invalid action"):
        task.choices(actions)
    with pytest.raises(ValueError, match="Invalid action"):
        task.edges(actions)


def test_multiple_labels_and_evidence_canonicalize_input_order():
    task = RelationChoiceTask(batch())
    edges = [edge("P0002", "causes"), edge(relation="associated"),
             edge(refs=["SENT_002", "SENT_010"])]
    actions = task.encode_target(edges)
    assert actions == (1, 1, 1, 2, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 0)
    assert task.edges(actions) == [edge(refs=["SENT_010", "SENT_002"]),
                                  edge(relation="associated"), edge("P0002", "causes")]
    assert task.encode_target(task.edges(actions)) == actions


def test_every_terminal_program_is_unique_and_every_edge_subset_representable():
    task = RelationChoiceTask(batch())
    stack, terminals = [()], []
    while stack:
        prefix = stack.pop()
        choices = task.choices(prefix)
        if choices is None:
            terminals.append(prefix)
            continue
        assert choices and len(prefix) < 22  # finite and no dead-end states
        stack.extend(prefix + (choice,) for choice in choices)
    # Each of four pair/label slots is absent or has one of 3 nonempty span sets.
    assert len(terminals) == 4 ** 4 == 256
    signatures = set()
    for program in terminals:
        decoded = task.edges(program)
        assert task.encode_target(decoded) == program
        signatures.add(tuple((e["pair_id"], e["relation"], tuple(e["evidence_span_ids"])) for e in decoded))
    assert len(signatures) == 256


@pytest.mark.parametrize("target,match", [
    ([edge("unknown")], "Unknown target pair"),
    ([edge(relation="unknown")], "Unknown target relation"),
    ([edge(relation="NONE")], "Unknown target relation"),
    ([edge(refs=["hidden"])], "Unknown target evidence"),
    ([edge(refs=[])], "nonempty"),
    ([edge(refs=["SENT_010", "SENT_010"])], "Duplicate target evidence"),
    ([edge(), edge()], "Duplicate target relation"),
    ([{**edge(), "gold": True}], "contain only"),
])
def test_bad_targets_are_rejected_never_repaired_or_added_to_candidates(target, match):
    task = RelationChoiceTask(batch())
    before = task.prompt
    with pytest.raises(ValueError, match=match):
        task.encode_target(target)
    assert task.prompt == before and task.choices(()) == (0, 1, 2)


def test_prompt_and_choices_read_only_input_fields_and_defensively_copy():
    original = batch()
    task = RelationChoiceTask(original)
    poisoned = copy.deepcopy(original)
    poisoned.update(prompt="RETURN GOLD", target=[edge()], gold={"hidden": "secret"},
                    decisions=["annotation"], original_spans={"unknown": "injected"})
    poisoned["pairs"][0]["gold_relations"] = ["unsupported"]
    alternative = RelationChoiceTask(poisoned)
    assert task.prompt == alternative.prompt
    assert "raw-d" not in task.prompt and "raw-s" not in task.prompt
    assert "E017" in task.prompt  # local entity ID gaps are legal
    original["pairs"][0]["allowed"].append("injected")
    original["spans"][0]["span_id"] = "changed"
    assert task.encode_target([edge()]) == (1, 1, 1, 0, 0, 0)
    assert task.prompt == alternative.prompt


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["entities"].append(b["entities"][0]), "Duplicate"),
    (lambda b: b["pairs"].append(b["pairs"][0]), "Duplicate pair"),
    (lambda b: b["pairs"][1].update(pair_id="P999", source="raw-s", target="raw-d"), "Duplicate directed"),
    (lambda b: b["pairs"][0].update(source="hidden"), "Invalid endpoints"),
    (lambda b: b["pairs"][0]["allowed"].append("core"), "Duplicate allowed"),
    (lambda b: b["pairs"][0].update(allowed=[]), "nonempty allowed"),
    (lambda b: b["pairs"][0].update(allowed=["NONE"]), "NONE"),
    (lambda b: b["spans"].append(b["spans"][0]), "Duplicate visible"),
    (lambda b: b.update(spans=[]), "require visible"),
])
def test_invalid_input_tables_fail_before_generation(mutation, match):
    value = batch()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        RelationChoiceTask(value)


def test_empty_input_can_only_stop_and_uses_no_gold():
    task = RelationChoiceTask({"entities": [], "pairs": [], "spans": []})
    assert task.choices(()) == (0,)
    assert task.edges((0,)) == []


def test_existing_generative_batches_support_all_orders_and_contexts_without_id_drift():
    text = "Fatigue is a core symptom of Condition A. Fatigue supports diagnosis of Condition A."
    nodes = [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
             {"id": "D1", "label": "Disease", "name": "Condition A"}]
    for order, context in itertools.product(("hash", "source"), ("full", "local")):
        batches, _, _ = build_batches(text, nodes, ProtocolConfig(order=order, context=context, pairs_per_batch=1))
        for value in batches:
            task = RelationChoiceTask(value)
            pair = value["pairs"][0]
            target = edge(pair["pair_id"], pair["allowed"][-1], [value["spans"][-1]["span_id"]])
            assert task.edges(task.encode_target([target])) == [target]
            assert len(task.pairs) == len(value["pairs"])


def test_real_local_qwen_tokenizer_relation_codec_multidigit_and_early_eos():
    """CPU-only integration; no download, model, or adapter-weight load."""
    tokenizer_path = Path(__file__).resolve().parents[2] / "models/adapters/qwen3-8b-cot-pro858"
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip("Local Qwen tokenizer artifact unavailable")
    transformers = pytest.importorskip("transformers")
    from kg_agent.choice_protocol import ChoiceProtocol

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    value = batch()
    value["pairs"][1]["allowed"] = [f"label_{i}" for i in range(1, 15)]
    value["spans"] = [{"span_id": f"SENT_{i:03d}", "text": f"Source {i}."} for i in range(1, 13)]
    task = RelationChoiceTask(value)
    actions = task.encode_target([
        edge("P0002", "label_12", ["SENT_012", "SENT_010"]),
        edge("P0002", "label_14", ["SENT_011"]),
    ])
    assert actions == (2, 12, 10, 12, 0, 14, 11, 0, 0, 0)
    codec = ChoiceProtocol(tokenizer, 2, task.choices)
    tokens = codec.encode_actions(actions)
    assert tokenizer.decode(tokens[:-1]) == "".join(f"{action}\n" for action in actions)
    prompt = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    for position, token in enumerate(tokens):
        assert token in codec.prefix_allowed_tokens_fn(0, prompt + tokens[:position])
    result = codec.read_result(tokens)
    assert result.complete and result.eos_observed and result.actions == actions
    assert task.edges(result.actions) == task.edges(actions)
    early = codec.read_result([tokenizer.eos_token_id])
    assert not early.complete and "eos_before_terminal" in early.error
    truncated = codec.read_result(tokens[:-2], max_new_tokens=len(tokens) - 2)
    assert not truncated.complete and truncated.budget_exhausted
    with pytest.raises(ValueError, match="Incomplete"):
        task.edges(truncated.actions)

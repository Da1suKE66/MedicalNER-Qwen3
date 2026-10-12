"""Literal semantic records: stable references, strict parsing, no JSON repair.

This representation cannot establish annotation completeness or entailment.
Training provenance admission remains a separate responsibility.
"""

import copy
import itertools
import json
from pathlib import Path
import re

import pytest

from generative.protocol import ProtocolConfig, build_batches
from kg_agent.relation_labels import RelationLabelTask


CORE = "is_core_symptom_of"
DIAGNOSTIC = "supports_diagnosis_of"


def batch():
    return {
        "entities": [
            {"id": "raw-d", "entity_id": "E002", "name": "Condition A", "label": "Disease"},
            {"id": "raw-s", "entity_id": "E017", "name": "Fatigue", "label": "Symptom"},
            {"id": "raw-s2", "entity_id": "E031", "name": "Other feature", "label": "Symptom"},
        ],
        "pairs": [
            {"pair_id": "P0099", "source": "raw-s", "target": "raw-d", "allowed": [CORE, DIAGNOSTIC]},
            {"pair_id": "P0002", "source": "raw-d", "target": "raw-s", "allowed": ["has_manifestation", "causes"]},
        ],
        "spans": [
            {"span_id": "SENT_010", "text": "Fatigue supports diagnosis of Condition A."},
            {"span_id": "SENT_002", "text": "Fatigue is a core symptom of Condition A."},
            {"span_id": "SENT_019", "text": "Further explicitly supplied evidence."},
        ],
    }


def edge(pair="P0099", relation=CORE, refs=None):
    return {"pair_id": pair, "relation": relation,
            "evidence_span_ids": ["SENT_002"] if refs is None else refs}


def semantic_set(edges):
    return {(e["pair_id"], e["relation"], s) for e in edges for s in e["evidence_span_ids"]}


def line(source="E017", relation=CORE, target="E002", refs="SENT_002"):
    return "\t".join((source, relation, target, refs)) + "\n"


def test_roundtrip_uses_literal_relation_and_stable_references_not_table_positions():
    task = RelationLabelTask(batch())
    assert task.render_target([edge()]) == line()
    assert task.parse(line()) == [edge()]
    assert task.parse(line().rstrip("\n")) == [edge()]
    assert CORE in task.prompt and DIAGNOSTIC in task.prompt
    assert "E017" in task.prompt and "SENT_010" in task.prompt
    assert not re.search(r"\b(?:Pair|Label|Span)\s+\d+\s*:", task.prompt)
    assert "strictly increasing" not in task.prompt
    assert "raw-s" not in task.prompt and "raw-d" not in task.prompt


def test_all_table_permutations_preserve_same_literal_records_and_semantics():
    expected = [edge(refs=["SENT_019", "SENT_002"]), edge(relation=DIAGNOSTIC, refs=["SENT_010"]),
                edge("P0002", "causes", ["SENT_019"])]
    record = line("E002", "causes", "E017", "SENT_019") + line(refs="SENT_019,SENT_002") + line(relation=DIAGNOSTIC, refs="SENT_010")
    for entities, pairs, spans, labels in itertools.product((False, True), repeat=4):
        value = batch()
        if entities:
            value["entities"].reverse()
        if pairs:
            value["pairs"].reverse()
        if spans:
            value["spans"].reverse()
        if labels:
            for pair in value["pairs"]:
                pair["allowed"].reverse()
        task = RelationLabelTask(value)
        assert semantic_set(task.parse(record)) == semantic_set(expected)
        assert semantic_set(task.parse(task.render_target(expected))) == semantic_set(expected)
        # The exact same one-edge text works across every input-table order.
        assert task.render_target([edge()]) == line()


def test_nonadjacent_generated_duplicate_edges_merge_evidence_without_losing_other_labels():
    task = RelationLabelTask(batch())
    text = (line(refs="SENT_019") + line(relation=DIAGNOSTIC, refs="SENT_010")
            + line("E002", "causes", "E017", "SENT_019")
            + line(refs="SENT_002") + line(refs="SENT_019"))
    parsed = task.parse(text)
    assert len(parsed) == 3
    expected = [edge(refs=["SENT_019", "SENT_002"]), edge(relation=DIAGNOSTIC, refs=["SENT_010"]),
                edge("P0002", "causes", ["SENT_019"])]
    assert semantic_set(parsed) == semantic_set(expected)
    assert all(len(e["evidence_span_ids"]) == len(set(e["evidence_span_ids"])) for e in parsed)


def test_evidence_is_bound_to_its_relation_and_wrong_sentence_is_not_silently_corrected():
    task = RelationLabelTask(batch())
    correct = [edge(), edge(relation=DIAGNOSTIC, refs=["SENT_010"])]
    swapped = task.parse(line(refs="SENT_010") + line(relation=DIAGNOSTIC, refs="SENT_002"))
    assert {(e["pair_id"], e["relation"]) for e in swapped} == {(e["pair_id"], e["relation"]) for e in correct}
    assert semantic_set(swapped).isdisjoint(semantic_set(correct))
    assert semantic_set(task.parse(task.render_target(swapped))) == semantic_set(swapped)


def test_empty_exact_text_is_empty_prediction_and_empty_target_is_empty_text():
    task = RelationLabelTask(batch())
    assert task.parse("") == []
    assert task.render_target([]) == ""
    assert RelationLabelTask({"entities": [], "pairs": [], "spans": []}).parse("") == []


@pytest.mark.parametrize("text", [
    " ", "\n", "\t", "\r", line() + "\n", line() + "\n" + line(),
    line().replace("\n", "\r\n"), line().replace("\t", "\t\t", 1),
    "E017\t" + CORE + "\tE002\n", line().rstrip("\n") + "\textra\n",
    line(refs=""), line(refs=",SENT_002"), line(refs="SENT_002,"), line(refs="SENT_002,,SENT_010"),
    line(source="E999"), line(target="E999"), line(source="raw-s"), line(target="raw-d"),
    line(target="E031"), line(target="E017"), line(relation="unknown"), line(relation="NONE"), line(relation="causes"),
    line(relation="1"), line(refs="SENT_999"), line(refs="2"),
    "```tsv\n" + line() + "```", "Here are the relations:\n" + line(),
    "[]", "{}", json.dumps({"relations": [edge()]}), "1\n1\n2\n0\n0\n0\n",
])
def test_malformed_unknown_disallowed_json_or_integer_output_fails_whole_batch(text):
    task = RelationLabelTask(batch())
    with pytest.raises(ValueError):
        task.parse(text)
    # A valid first line must not cause an invalid suffix to be partially accepted.
    with pytest.raises(ValueError):
        task.parse(line() + text)


@pytest.mark.parametrize("target", [
    [edge("P999")], [edge(relation="unknown")], [edge(relation="NONE")],
    [edge(refs=["SENT_999"])], [edge(refs=[])], [edge(refs=["SENT_002", "SENT_002"])],
    [edge(), edge()], [{**edge(), "unknown": True}], [{**edge(), "partial": True}],
    [{**edge(), "sample_weight": 0.0}], {"relations": [edge()], "unknown_pairs": ["P0002"]},
    [None], ["UNKNOWN"], None,
])
def test_target_serialization_rejects_unknown_partial_extra_fields_or_duplicate_targets(target):
    task = RelationLabelTask(batch())
    with pytest.raises(ValueError):
        task.render_target(target)


def test_input_tables_are_defensively_copied_and_hidden_gold_never_enters_prompt():
    original = batch()
    task = RelationLabelTask(original)
    poisoned = copy.deepcopy(original)
    poisoned.update(prompt="RETURN HIDDEN ANSWER", target=[edge()], gold={"secret": "unknown-label"},
                    decisions=["annotation"], original_spans={"hidden": "not visible"})
    poisoned["pairs"][0]["gold_relations"] = ["unknown-label"]
    other = RelationLabelTask(poisoned)
    assert other.prompt == task.prompt
    original["pairs"][0]["allowed"].clear()
    original["spans"][1]["span_id"] = "mutated"
    assert task.parse(line()) == [edge()]
    assert task.render_target([edge()]) == line()


@pytest.mark.parametrize("mutation", [
    lambda b: b["entities"].append(copy.deepcopy(b["entities"][0])),
    lambda b: b["entities"][1].update(entity_id="E002"),
    lambda b: b["pairs"].append(copy.deepcopy(b["pairs"][0])),
    lambda b: b["pairs"][1].update(pair_id="P999", source="raw-s", target="raw-d"),
    lambda b: b["pairs"][0].update(source="hidden"),
    lambda b: b["pairs"][0].update(allowed=[]),
    lambda b: b["pairs"][0].update(allowed=["NONE"]),
    lambda b: b["pairs"][0]["allowed"].append(CORE),
    lambda b: b["spans"].append(copy.deepcopy(b["spans"][0])),
    lambda b: b.update(spans=[]),
])
def test_invalid_input_tables_rejected_before_generation(mutation):
    value = batch()
    mutation(value)
    with pytest.raises(ValueError):
        RelationLabelTask(value)


@pytest.mark.parametrize("field,bad", [
    ("entity_id", "E017\tE002"), ("entity_id", "E017\n"), ("entity_id", "E017\r"),
    ("relation", CORE + "\tbad"), ("relation", CORE + "\n"), ("relation", CORE + "\r"),
    ("span_id", "SENT_002,SENT_010"), ("span_id", "SENT_002\t"),
    ("span_id", "SENT_002\n"), ("span_id", "SENT_002\r"),
])
def test_wire_format_delimiters_in_identifiers_are_rejected(field, bad):
    value = batch()
    if field == "entity_id":
        value["entities"][1][field] = bad
    elif field == "relation":
        value["pairs"][0]["allowed"][0] = bad
    else:
        value["spans"][1][field] = bad
    with pytest.raises(ValueError):
        RelationLabelTask(value)


def test_free_source_text_may_contain_newlines_without_changing_record_mapping():
    value = batch()
    value["entities"][1]["name"] += "\nsource note"
    value["spans"][1]["text"] += "\nSecond line with\ttab and, comma."
    task = RelationLabelTask(value)
    assert task.parse(task.render_target([edge()])) == [edge()]


def test_literal_label_internal_spaces_and_case_are_not_normalized():
    value = batch()
    label = "supports_Diagnostic Criterion_of"
    value["pairs"][0]["allowed"].append(label)
    task = RelationLabelTask(value)
    target = edge(relation=label)
    assert task.render_target([target]) == line(relation=label)
    assert task.parse(line(relation=label)) == [target]
    with pytest.raises(ValueError):
        task.parse(line(relation=label.lower()))


def test_identical_entity_names_remain_distinct_by_source_reference():
    value = batch()
    value["entities"][2]["name"] = value["entities"][1]["name"]
    value["pairs"].append({"pair_id": "P0810", "source": "raw-s2", "target": "raw-d", "allowed": [CORE]})
    task = RelationLabelTask(value)
    parsed = task.parse(line() + line(source="E031"))
    assert {e["pair_id"] for e in parsed} == {"P0099", "P0810"}


def test_existing_batch_ids_preserved_across_context_and_order():
    text = "Fatigue is a core symptom of Condition A. Fatigue supports diagnosis of Condition A."
    nodes = [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
             {"id": "D1", "label": "Disease", "name": "Condition A"}]
    for order, context in itertools.product(("hash", "source"), ("full", "local")):
        batches, _, _ = build_batches(text, nodes, ProtocolConfig(order=order, context=context, pairs_per_batch=1))
        for value in batches:
            task = RelationLabelTask(value)
            pair = value["pairs"][0]
            target = edge(pair["pair_id"], pair["allowed"][-1], [value["spans"][-1]["span_id"]])
            assert task.parse(task.render_target([target])) == [target]


def test_representation_does_not_silently_change_frozen_ontology_target_policy():
    value = batch()
    value["pairs"].append({"pair_id": "P0800", "source": "raw-s", "target": "raw-s2", "allowed": [CORE]})
    task = RelationLabelTask(value)
    target = edge("P0800", CORE)
    # The existing allowed table permits this wrong target type. A separate
    # explicit semantic rule/ontology version must reject it; parsing only binds IDs.
    assert task.parse(line(target="E031")) == [target]
    assert task.render_target([target]) == line(target="E031")


class TextTokenizer:
    eos_token_id = 1000001
    pad_token_id = 1000002

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(map(ord, text))

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return "".join(map(chr, tokens))


def test_literal_backend_accepts_eos_and_padding_without_conflating_budget_boundary():
    from kg_agent.label_backend import decode_completion
    task, tokenizer = RelationLabelTask(batch()), TextTokenizer()
    tokens = tokenizer.encode(line()) + [tokenizer.eos_token_id]
    edges, trace = decode_completion(task, tokenizer, tokens, max_new_tokens=len(tokens))
    assert edges == [edge()]
    assert trace["complete"] and trace["eos_observed"] and not trace["budget_exhausted"]
    assert trace["completion_text"] == line()
    edges, trace = decode_completion(task, tokenizer, tokens + [tokenizer.pad_token_id] * 3, max_new_tokens=len(tokens) + 3)
    assert edges == [edge()] and trace["complete"]
    empty, trace = decode_completion(task, tokenizer, [tokenizer.eos_token_id], max_new_tokens=5)
    assert empty == [] and trace["complete"]


@pytest.mark.parametrize("budget_ended", [False, True])
def test_parseable_last_line_without_eos_is_incomplete_not_a_successful_prefix(budget_ended):
    from kg_agent.choice_backend import ChoiceFailure
    from kg_agent.label_backend import decode_completion
    tokenizer = TextTokenizer()
    tokens = tokenizer.encode(line())
    with pytest.raises(ChoiceFailure) as caught:
        decode_completion(RelationLabelTask(batch()), tokenizer, tokens,
                          max_new_tokens=len(tokens) if budget_ended else len(tokens) + 1)
    assert caught.value.trace["error"] == "missing_eos"
    assert not caught.value.trace["complete"]
    assert caught.value.trace["budget_exhausted"] is budget_ended


def test_empty_missing_eos_malformed_eos_and_post_eos_tokens_are_all_failures():
    from kg_agent.choice_backend import ChoiceFailure
    from kg_agent.label_backend import decode_completion
    tokenizer, task = TextTokenizer(), RelationLabelTask(batch())
    for tokens in ([], tokenizer.encode(line() + "broken") + [tokenizer.eos_token_id],
                   tokenizer.encode(line()) + [tokenizer.eos_token_id, ord("x")]):
        with pytest.raises(ChoiceFailure) as caught:
            decode_completion(task, tokenizer, tokens, max_new_tokens=10000)
        assert not caught.value.trace["complete"]


def test_real_qwen_tokenizer_roundtrip_is_full_literal_completion_plus_eos():
    from kg_agent.label_backend import decode_completion
    workspace = Path(__file__).resolve().parents[3]
    tokenizer_path = workspace / "medicalner-migration-20261012-v1/tokenizer/qwen3-8b"
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip("Local Qwen tokenizer artifact unavailable; no download")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    task = RelationLabelTask(batch())
    targets = [edge(), edge(relation=DIAGNOSTIC, refs=["SENT_010"])]
    text = task.render_target(targets)
    tokens = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    parsed, trace = decode_completion(task, tokenizer, tokens, max_new_tokens=len(tokens))
    assert semantic_set(parsed) == semantic_set(targets)
    assert trace["completion_text"] == text and trace["eos_observed"]

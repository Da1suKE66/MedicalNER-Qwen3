from __future__ import annotations

import pytest

from kg_agent.contracts import EntityCandidate, SourceDocument, Span
from kg_agent.entity_choices import CONTRACT, IMPORTANCES, LABELS, EntityChoiceTask
from kg_agent.source import resolve_span_ref


def task(text="alpha beta, gamma", **kwargs):
    return EntityChoiceTask([{"span_id": "SENT_001", "text": text, "section": "body"}], **kwargs)


def test_prompt_and_boundaries_keep_exact_unicode_and_whitespace():
    source = "  Cafe\u0301\tβ2，中文\r\nnext-word🙂  "
    value = task(source)
    assert [token.text for token in value.tokens] == ["Cafe\u0301", "β2", "，", "中", "文", "next", "-", "word", "🙂"]
    assert all(source[token.start:token.end] == token.text for token in value.tokens)
    assert CONTRACT in value.prompt and "LABELS:" in value.prompt
    assert "0 is legal only at a start step" in value.prompt
    assert '"text":"  Café\\tβ2，中文\\r\\nnext-word🙂  "' in value.prompt
    # The exact endpoint for this multi-token selection is derived from source.
    candidate = EntityCandidate(f"SENT_001:2-{value.tokens[5].end}", LABELS[0], name=source[2:value.tokens[5].end])
    restored = value.candidates(value.encode_target([candidate]))[0]
    assert restored.name.encode("utf-8") == candidate.name.encode("utf-8")
    assert "\t" in restored.name and "\r\n" in restored.name


def test_basic_fsm_and_explicit_stop():
    value = task()
    assert value.choices([]) == [0, 1, 2, 3, 4]
    assert value.choices([1]) == [1, 2, 3, 4]
    assert value.choices([1, 2]) == list(range(1, len(LABELS) + 1))
    assert value.choices([1, 2, 1]) == list(range(1, len(IMPORTANCES) + 1))
    assert value.choices([1, 2, 1, 1])[0] == 0
    assert value.choices([1, 2, 1, 1, 0]) is None
    assert value.candidates([1, 2, 1, 1, 0]) == [EntityCandidate("SENT_001:0-10", LABELS[0], IMPORTANCES[0], "alpha beta")]


@pytest.mark.parametrize("actions", [[1], [1, 1], [1, 1, 1], [1, 1, 1, 1]])
def test_partial_output_is_not_silently_accepted(actions):
    value = task()
    assert value.choices(actions) is not None
    assert not value.trace(actions)["complete"]
    with pytest.raises(ValueError, match="incomplete"):
        value.candidates(actions)


@pytest.mark.parametrize("actions", [[-1], [5], [True], [1.0], ["1"], [0, 1], [1, 0], [2, 1], [1, 1, 0], [1, 1, 1, 0], "1", None])
def test_illegal_actions_fail_closed(actions):
    with pytest.raises(ValueError):
        task().choices(actions)


def test_spans_cannot_be_crossed_and_width_is_enforced():
    value = EntityChoiceTask([
        {"span_id": "SENT_001", "text": "alpha beta gamma"},
        {"span_id": "SENT_002", "text": "delta epsilon"},
    ], max_span_tokens=2)
    assert value.choices([1]) == [1, 2]
    assert value.choices([3]) == [3]
    assert value.choices([4]) == [4, 5]
    with pytest.raises(ValueError):
        value.choices([3, 4])
    with pytest.raises(ValueError, match="max_span_tokens"):
        value.encode_target([EntityCandidate("SENT_001", LABELS[0])])


def test_nested_overlapping_and_same_start_multilabel_are_legal():
    value = task("alpha beta gamma")
    candidates = [
        EntityCandidate("SENT_001:6-16", LABELS[0], name="beta gamma"),
        EntityCandidate("SENT_001:0-10", LABELS[1], name="alpha beta"),
        EntityCandidate("SENT_001", LABELS[0], name="alpha beta gamma"),
        EntityCandidate("SENT_001:0-5", LABELS[0], name="alpha"),
        EntityCandidate("SENT_001:0-10", LABELS[0], name="alpha beta"),
    ]
    actions = value.encode_target(candidates)
    assert actions[:-1:4] == [1, 1, 1, 1, 2]
    recovered = value.candidates(actions)
    assert {(candidate.span_id, candidate.label, candidate.name) for candidate in recovered} == {
        (candidate.span_id, candidate.label, candidate.name) for candidate in candidates
    }


def test_strict_identity_order_blocks_duplicate_and_reverse_order():
    value = task("alpha beta")
    previous = [1, 1, len(LABELS), 1]
    assert value.choices(previous + [1]) == [2]
    with pytest.raises(ValueError):
        value.choices(previous + [1, 1])
    assert value.choices([2, 2, len(LABELS), 1]) == [0]
    with pytest.raises(ValueError):
        value.choices([2, 2, 1, 1, 1])


def test_cap_requires_stop_and_emits_possible_truncation_trace():
    value = task(max_candidates=1)
    prefix = [1, 1, 1, 1]
    assert value.choices(prefix) == [0]
    trace = value.trace(prefix + [0])
    assert trace["complete"] and trace["candidate_cap_reached"] and trace["possible_truncation"]
    assert trace["candidate_count"] == 1
    assert not value.trace([0])["candidate_cap_reached"]
    with pytest.raises(ValueError, match="max_candidates"):
        value.encode_target([EntityCandidate("SENT_001:0-5", LABELS[0]), EntityCandidate("SENT_001:6-10", LABELS[0])])


def test_empty_source_and_empty_target():
    value = EntityChoiceTask([])
    assert value.choices([]) == [0]
    assert value.encode_target([]) == [0]
    assert value.candidates([0]) == []
    assert task("\t \r\n").choices([]) == [0]


@pytest.mark.parametrize("candidate", [
    EntityCandidate("missing", LABELS[0]),
    EntityCandidate("SENT_001:0-999", LABELS[0]),
    EntityCandidate("SENT_001:2-4", LABELS[0], name="ph"),
    EntityCandidate("SENT_001:0-5", LABELS[0], name="ALPHA"),
    EntityCandidate("SENT_001", "UNKNOWN"),
    EntityCandidate("SENT_001", LABELS[0], "nonsense"),
    EntityCandidate("SENT_001", LABELS[0], name="invented"),
    {"span_id": "SENT_001", "label": LABELS[0], "status": "UNKNOWN"},
    {"span_id": "SENT_001", "label": LABELS[0], "mention": "alpha", "name": "beta"},
])
def test_bad_or_unknown_targets_are_not_repaired_or_converted_to_empty(candidate):
    with pytest.raises(ValueError):
        task().encode_target([candidate])


def test_ambiguous_mentions_require_offsets_but_unique_copy_is_accepted():
    value = task("alpha beta alpha")
    with pytest.raises(ValueError, match="ambiguous"):
        value.encode_target([EntityCandidate("SENT_001", LABELS[0], name="alpha")])
    actions = value.encode_target([EntityCandidate("SENT_001:11-16", LABELS[0], name="alpha")])
    assert value.candidates(actions)[0].span_id == "SENT_001:11-16"
    actions = value.encode_target([{"span_id": "SENT_001", "label": LABELS[0], "mention": "beta"}])
    assert value.candidates(actions)[0].name == "beta"


def test_duplicate_identity_is_rejected_even_if_importance_differs():
    value = task()
    with pytest.raises(ValueError, match="duplicate"):
        value.encode_target([EntityCandidate("SENT_001:0-5", LABELS[0], IMPORTANCES[0]),
                             EntityCandidate("SENT_001:0-5", LABELS[0], IMPORTANCES[1])])


def test_original_source_reference_resolves_against_existing_contract():
    text = "  alpha\t beta."
    value = task(text)
    candidate = value.candidates(value.encode_target([EntityCandidate("SENT_001:2-13", LABELS[0], name="alpha\t beta")]))[0]
    document = SourceDocument("free_text", text, spans=[Span("SENT_001", text, 0, len(text))])
    resolved = resolve_span_ref(document, candidate.span_id)
    assert resolved is not None and resolved.text == candidate.name


def test_source_copy_hashes_and_target_context_are_deterministic():
    source = [{"span_id": "SENT_001", "text": "alpha", "section": "body"}]
    value = EntityChoiceTask(source, target_name="context-only")
    same = EntityChoiceTask(source, target_name="context-only")
    assert (value.source_sha256, value.protocol_sha256, value.prompt) == (same.source_sha256, same.protocol_sha256, same.prompt)
    source[0]["text"] = "changed"
    assert value.tokens[0].text == "alpha"
    assert "context-only" not in [token.text for token in value.tokens]
    assert value.source_sha256 != EntityChoiceTask(source).source_sha256
    assert value.protocol_sha256 != task(max_span_tokens=2).protocol_sha256


@pytest.mark.parametrize("kwargs", [{"max_candidates": 0}, {"max_candidates": True}, {"max_span_tokens": 0}, {"max_span_tokens": 1.5}, {"target_name": 5}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        task(**kwargs)


@pytest.mark.parametrize("spans", [
    [{"span_id": "x", "text": "a"}, {"span_id": "x", "text": "b"}],
    [{"span_id": "x:1", "text": "a"}],
    [{"span_id": "", "text": "a"}],
    [{"span_id": "x", "text": 1}],
])
def test_invalid_source_is_rejected(spans):
    with pytest.raises(ValueError):
        EntityChoiceTask(spans)


def test_all_legal_single_entities_round_trip_without_gold_candidate_injection():
    value = task("甲，alpha β", max_candidates=2, max_span_tokens=3)
    for start in value.choices([])[1:]:
        for end in value.choices([start]):
            for label in value.choices([start, end]):
                actions = [start, end, label, 1, 0]
                candidates = value.candidates(actions)
                assert value.encode_target(candidates) == actions
                assert candidates[0].name in "甲，alpha β"


def test_next_record_choices_are_exactly_all_later_source_identities():
    value = task("甲乙", max_candidates=2)
    identities = [(start, end, label) for start in (1, 2) for end in range(start, 3)
                  for label in range(1, len(LABELS) + 1)]
    for previous in identities:
        prefix = [*previous, 1]
        actual = []
        for start in value.choices(prefix)[1:]:
            ends = value.choices(prefix + [start])
            assert ends  # No allowed start leads to a grammar dead end.
            for end in ends:
                labels = value.choices(prefix + [start, end])
                assert labels
                for label in labels:
                    actual.append((start, end, label))
                    actions = prefix + [start, end, label, 1, 0]
                    assert value.choices(actions) is None
                    assert value.encode_target(value.candidates(actions)) == actions
        assert actual == [identity for identity in identities if identity > previous]

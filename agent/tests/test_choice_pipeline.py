"""CPU integration: both causal choice tasks to a program-assembled graph.

The fake generator supplies declared source answers through the real task FSM
and token-prefix codec. These tests establish plumbing and failure accounting,
not trained-model extraction accuracy.
"""

from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest

from generative.protocol import ProtocolConfig
from kg_agent import pipeline, prompts
from kg_agent.assemble import AssemblerConfig
from kg_agent.choice_backend import ChoiceEntitySelector, ChoiceFailure, ChoiceRelationSelector
from kg_agent.choice_protocol import ChoiceProtocol
from kg_agent.contracts import EntityCandidate
from kg_agent.entity_choices import EntityChoiceTask
from kg_agent.pipeline import PipelineConfig, TwoStageKGAgent
from kg_agent.relation_choices import RelationChoiceTask
from kg_agent.source import SourceRouter, resolve_span_ref


TEXT = ("Fatigue is a core symptom of Condition Alpha and supports its diagnosis. "
        "Fatigue is a core symptom of Condition Alpha and also supports its diagnosis.")
RELATIONS = ("is_core_symptom_of", "supports_diagnosis_of")


class NumericTokenizer:
    eos_token_id = 257
    pad_token_id = 258
    all_special_ids = [257, 258]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode("ascii"))


@pytest.fixture(autouse=True)
def forbid_legacy_json_path(monkeypatch):
    """Patch both definition and imported aliases; any fallback fails the test."""
    forbidden = []
    for module in (pipeline, prompts):
        for name in ("build_entity_prompt", "parse_entity_candidates", "build_relation_prompt",
                     "parse_relation_decisions", "extract_json_object"):
            call = Mock(side_effect=AssertionError(f"Legacy JSON path called: {module.__name__}.{name}"))
            monkeypatch.setattr(module, name, call)
            forbidden.append(call)
    yield
    for call in forbidden:
        call.assert_not_called()


class FixedAnswerGenerator:
    """Encode known test answers; never emit JSON or bypass the token codec."""

    def __init__(self, raw=TEXT, *, names=("Fatigue", "Condition Alpha"), failure=None,
                 failure_stage=None, unsupported_entity=False):
        self.document = SourceRouter().route(raw)
        self.names = names
        self.failure = failure
        self.failure_stage = failure_stage
        self.unsupported_entity = unsupported_entity
        self.calls = []
        self.results = []

    def _entity_answer(self, task):
        visible = {token.span_id for token in task.tokens}
        answer = []
        for span in self.document.spans:
            if span.id not in visible:
                continue
            for name in self.names:
                start = span.text.find(name)
                if start < 0:
                    continue
                answer.append(EntityCandidate(
                    f"{span.id}:{start}-{start + len(name)}",
                    "Symptom" if name == "Fatigue" else "Disease",
                    "core" if name == "Fatigue" else "required",
                    name,
                ))
        if self.unsupported_entity:
            # A named hallucination cannot pass the action target serializer.
            answer = [EntityCandidate(self.document.spans[0].id, "Disease", name="Invented Disease")]
        return answer

    def _relation_answer(self, task):
        names = {row["id"]: row["name"] for row in task.entities}
        answer = []
        for pair in task.pairs:
            if (names[pair["source"]], names[pair["target"]]) != ("Fatigue", "Condition Alpha"):
                continue
            assert set(RELATIONS) <= set(pair["allowed"])
            # Both chosen source sentences support the test's two relations.
            evidence = [row["span_id"] for row in task.spans]
            answer.extend({"pair_id": pair["pair_id"], "relation": label,
                           "evidence_span_ids": evidence} for label in RELATIONS)
        return answer

    def run(self, task):
        self.calls.append(task)
        assert isinstance(task, (EntityChoiceTask, RelationChoiceTask))
        stage = "entity" if isinstance(task, EntityChoiceTask) else "relation"
        if stage == self.failure_stage:
            raise self.failure
        answer = self._entity_answer(task) if stage == "entity" else self._relation_answer(task)
        actions = task.encode_target(answer)
        codec = ChoiceProtocol(NumericTokenizer(), 2, task.choices)
        tokens = codec.encode_actions(actions)
        prompt = [700, 701]  # prompt tokens are intentionally not action tokens
        for index, token in enumerate(tokens):
            assert token in codec.prefix_allowed_tokens_fn(0, prompt + tokens[:index])
        result = codec.read_result(tokens, max_new_tokens=len(tokens))
        assert result.complete and result.eos_observed and result.error is None
        assert list(result.actions) == list(actions)
        trace = {**asdict(result), "generated_token_ids": tokens, "test_fake_generator": True}
        self.results.append((stage, trace))
        return list(result.actions), trace


def make_agent(generator, *, max_spans=16, pairs_per_batch=4):
    backend = Mock()
    backend.generate.side_effect = AssertionError("Free text generation is forbidden")
    config = PipelineConfig(
        max_spans_per_chunk=max_spans,
        assembler=AssemblerConfig(include_provenance=True, evidence_as_list=True),
    )
    agent = TwoStageKGAgent(
        backend, config=config,
        entity_selector=ChoiceEntitySelector(generator),
        relation_classifier=ChoiceRelationSelector(
            generator, config=ProtocolConfig(order="source", pairs_per_batch=pairs_per_batch)),
    )
    return agent, backend


@pytest.mark.parametrize("max_spans,pairs_per_batch", [(16, 4), (1, 1)])
def test_raw_text_both_stages_preserve_multilabel_multievidence_and_exact_source(max_spans, pairs_per_batch):
    generator = FixedAnswerGenerator()
    agent, legacy = make_agent(generator, max_spans=max_spans, pairs_per_batch=pairs_per_batch)
    result = agent.run(TEXT)
    legacy.generate.assert_not_called()
    assert result.validation.valid
    assert result.trace["mode"] == "two_stage_bounded_choices"
    assert result.trace["entity_output_contract"] == result.trace["relation_output_contract"] == "bounded-choices-v1"
    assert result.trace["dropped_candidates"] == []
    assert {(row["id"], row["label"], row["name"]) for row in result.output["entities"]} == {
        ("D1", "Disease", "Condition Alpha"), ("S1", "Symptom", "Fatigue")}
    assert len(result.output["relations"]) == 2
    assert {(row["source"], row["target"], row["relation"]) for row in result.output["relations"]} == {
        ("S1", "D1", label) for label in RELATIONS}
    for entity in result.graph.entities:
        assert entity.name in TEXT and len(entity.span_ids) == 2
        for reference in entity.span_ids:
            span = resolve_span_ref(generator.document, reference)
            assert span is not None and span.text == entity.name
    evidence_ids = [span.id for span in generator.document.spans]
    evidence_text = [span.text for span in generator.document.spans]
    for edge in result.output["relations"]:
        assert edge["evidence_span_ids"] == evidence_ids
        assert edge["evidence"] == evidence_text
        assert "pair_id" not in edge
    assert json.loads(json.dumps(result.output)) == result.output
    entity_runs = [task for task in generator.calls if isinstance(task, EntityChoiceTask)]
    relation_runs = [task for task in generator.calls if isinstance(task, RelationChoiceTask)]
    assert len(entity_runs) == (2 if max_spans == 1 else 1)
    assert len(relation_runs) == (2 if pairs_per_batch == 1 else 1)
    assert sum(len(task.pairs) for task in relation_runs) == 2
    assert result.trace["coverage"] == {"entities": 2, "relations": 2, "grounded_mentions": 4}
    assert all(trace["complete"] and trace["eos_observed"] for _, trace in generator.results)


@pytest.mark.parametrize("names,expected", [((), set()), (("Fatigue",), {"Fatigue"})])
def test_empty_or_single_entity_results_cannot_invent_missing_endpoint(names, expected):
    generator = FixedAnswerGenerator(names=names)
    agent, legacy = make_agent(generator)
    result = agent.run(TEXT)
    assert result.validation.valid
    assert {entity.name for entity in result.graph.entities} == expected
    assert result.output["relations"] == []
    assert result.trace["relation_classifier"]["candidate_pairs"] == 0
    assert all(isinstance(task, EntityChoiceTask) for task in generator.calls)
    legacy.generate.assert_not_called()


def test_source_absent_entity_answer_fails_before_any_relation_call():
    generator = FixedAnswerGenerator(unsupported_entity=True)
    agent, legacy = make_agent(generator)
    with pytest.raises(ValueError, match="absent or ambiguous"):
        agent.run(TEXT)
    assert len(generator.calls) == 1 and isinstance(generator.calls[0], EntityChoiceTask)
    legacy.generate.assert_not_called()


@pytest.mark.parametrize("stage", ["entity", "relation"])
def test_choice_failure_is_propagated_unchanged_and_never_returned_as_empty_success(stage):
    trace = {"complete": False, "budget_exhausted": True, "generated_token_ids": [49]}
    failure = ChoiceFailure("Deliberate incomplete generation", trace)
    generator = FixedAnswerGenerator(failure=failure, failure_stage=stage)
    agent, legacy = make_agent(generator)
    with pytest.raises(ChoiceFailure) as raised:
        agent.run(TEXT)
    assert raised.value is failure and raised.value.trace is trace
    assert raised.value.trace["complete"] is False
    assert len(generator.calls) == (1 if stage == "entity" else 2)
    legacy.generate.assert_not_called()


def test_structured_icd_routes_around_both_choice_tasks_and_every_model_call():
    generator = Mock()
    generator.run.side_effect = AssertionError("ICD must not call a model")
    agent, legacy = make_agent(generator)
    result = agent.run({
        "URI": "urn:icd:A01", "code": "A01", "title": "Example disease",
        "synonyms": ["Example syndrome"],
        "descendants": [{"code": "A01.1", "title": "Example subtype"}],
        "exclusions": [{"code": "B02", "title": "Excluded condition"}],
    })
    assert result.validation.valid
    assert result.trace["mode"] == "deterministic_icd_parser"
    assert {row["name"] for row in result.output["entities"]} == {
        "Example disease", "Example subtype", "Excluded condition"}
    assert {row["relation"] for row in result.output["relations"]} == {"subtype_of", "rules_out"}
    generator.run.assert_not_called()
    legacy.generate.assert_not_called()

"""Independent end-to-end checks for legacy candidate and prompt compatibility."""

import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from kg_agent.pipeline import PipelineConfig, TwoStageKGAgent
from kg_agent.prompts import build_relation_prompt


class EntityReply:
    def __init__(self, mentions):
        self.mentions = mentions

    def generate(self, prompt, *, max_new_tokens):
        return json.dumps({"entity_candidates": [
            {"span_id": "SENT_001", "mention": name, "label": label, "importance": "core"}
            for name, label in self.mentions
        ]})


class RecordingRelations:
    def __init__(self, emit_multiple=False):
        self.prompts = []
        self.pair_batches = []
        self.emit_multiple = emit_multiple

    def generate(self, prompt, *, max_new_tokens):
        self.prompts.append(prompt)
        pair_block = prompt.split("\nCANDIDATE PAIRS\n", 1)[1].split("\n\nSOURCE SPANS\n", 1)[0]
        pairs = re.findall(r"^(P\d+): (\S+) -> (\S+) \| allowed=(.*)$", pair_block, re.MULTILINE)
        self.pair_batches.append(pairs)
        output = []
        if self.emit_multiple:
            for pair_id, source, target, allowed in pairs:
                if source.startswith("S") and target.startswith("D"):
                    assert {"is_core_symptom_of", "supports_diagnosis_of"} <= set(allowed.split(","))
                    output.extend({"pair_id": pair_id, "relation": label, "evidence_span_id": "SENT_001"}
                                  for label in ("is_core_symptom_of", "supports_diagnosis_of"))
        return json.dumps({"relations": output})


def run_fourteen_diseases(config):
    mentions = [(f"Condition{index:02d}", "Disease") for index in range(14)]
    raw = "The synthetic conditions are " + ", ".join(name for name, _ in mentions) + "."
    backend = RecordingRelations()
    result = TwoStageKGAgent(EntityReply(mentions), backend, config=config).run(raw)
    assert len(result.graph.entities) == 14
    assert len(result.trace["chunks"]) == 1
    assert result.validation.valid
    return result, backend


def test_default_candidate_coverage_submits_all_182_pairs_in_two_batches():
    result, backend = run_fourteen_diseases(PipelineConfig(entity_mention_copy=True))
    assert [len(batch) for batch in backend.pair_batches] == [128, 54]
    observed = [pair for batch in backend.pair_batches for pair in batch]
    assert len({pair[0] for pair in observed}) == 182
    assert {pair[0] for pair in observed} == {f"P{index:04d}" for index in range(1, 183)}
    ids = {entity.id for entity in result.graph.entities}
    assert {(pair[1], pair[2]) for pair in observed} == {(a, b) for a in ids for b in ids if a != b}
    trace = result.trace["chunks"][0]
    assert trace["candidate_pairs"] == trace["submitted_pairs"] == 182
    assert trace["omitted_pairs_by_explicit_cap"] == 0
    assert trace["relation_call_count"] == 2


def test_explicit_128_candidate_cap_reports_all_54_omissions():
    result, backend = run_fourteen_diseases(PipelineConfig(
        entity_mention_copy=True, max_pairs_per_chunk=128, relation_batch_size=16,
    ))
    assert [len(batch) for batch in backend.pair_batches] == [16] * 8
    assert len({pair[0] for batch in backend.pair_batches for pair in batch}) == 128
    trace = result.trace["chunks"][0]
    assert trace["candidate_pairs"] == 182
    assert trace["submitted_pairs"] == 128
    assert trace["omitted_pairs_by_explicit_cap"] == 54


def test_multilabel_opt_in_preserves_both_edges_on_one_pair():
    backend = RecordingRelations(emit_multiple=True)
    result = TwoStageKGAgent(
        EntityReply([("Feature", "Symptom"), ("Condition", "Disease")]), backend,
        config=PipelineConfig(entity_mention_copy=True, relation_multilabel=True),
    ).run("Feature is a core symptom of Condition and independently supports its diagnosis.")
    assert result.validation.valid
    assert len(result.graph.relations) == 2
    assert len({(edge.source, edge.target) for edge in result.graph.relations}) == 1
    assert {edge.relation for edge in result.graph.relations} == {"is_core_symptom_of", "supports_diagnosis_of"}
    assert all(edge.evidence_ids == ["SENT_001"] for edge in result.graph.relations)
    assert "do not choose only one" in backend.prompts[0]
    assert "do not output NONE entries" in backend.prompts[0]


def prompt_inputs():
    return (
        [{"id": "S1", "label": "Symptom", "name": "Feature"}],
        [{"pair_id": "P0001", "source": "S1", "target": "D1", "allowed_relations": ["is_core_symptom_of"]}],
        [{"span_id": "SENT_001", "text": "Synthetic evidence."}],
    )


def test_legacy_default_prompt_remains_byte_for_byte_unchanged():
    expected = """You are the relation classifier in a medical knowledge-graph pipeline.
All entity IDs and pair IDs below are fixed. Classify only explicitly supported relations.
For each pair, choose one relation from its allowed list or NONE. Do not invent endpoints or relation names.
Evidence must be one source span ID that supports the chosen relation. Omit NONE pairs from the output.
Output JSON only with this exact shape:
{"relations":[{"pair_id":"P0001","relation":"is_core_symptom_of","evidence_span_id":"SENT_001"}]}

ENTITIES
S1 (Symptom; importance=associated): Feature

CANDIDATE PAIRS
P0001: S1 -> D1 | allowed=is_core_symptom_of

SOURCE SPANS
SENT_001: Synthetic evidence."""
    assert build_relation_prompt(*prompt_inputs()) == expected
    assert build_relation_prompt(*prompt_inputs(), multilabel=False) == expected
    assert PipelineConfig().relation_multilabel is False


def test_legacy_explicit_none_prompt_preserves_single_decision_contract():
    prompt = build_relation_prompt(*prompt_inputs(), emit_none=True)
    assert "For every pair, return exactly one decision." in prompt
    assert "Do not omit pairs. The order and pair IDs must be preserved." in prompt
    assert "multiple relations" not in prompt


def test_multilabel_explicit_none_is_only_for_unsupported_pairs():
    prompt = build_relation_prompt(*prompt_inputs(), emit_none=True, multilabel=True)
    assert "A pair can have multiple relations; do not choose only one." in prompt
    assert "If no allowed relation is supported, return exactly one NONE entry" in prompt
    assert "For every pair, return exactly one decision." not in prompt


@pytest.mark.parametrize("base_model", [False, True])
@pytest.mark.parametrize("opt_in", [False, True])
def test_pipeline_cli_propagates_default_and_opt_in_contracts(tmp_path, monkeypatch, base_model, opt_in):
    import run_pipeline

    configurations = []

    class CapturingAgent:
        def __init__(self, *args, config, **kwargs):
            configurations.append(config)

        def run(self, *args, **kwargs):
            return SimpleNamespace(output={"entities": [], "relations": []}, trace={})

    monkeypatch.setattr(run_pipeline, "TwoStageKGAgent", CapturingAgent)
    monkeypatch.setattr(run_pipeline, "TransformersBackend", lambda *a, **kw: object())
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"text": "Synthetic source."}) + "\n")
    args = ["run_pipeline.py", "--input", str(source), "--output", str(tmp_path / "output.jsonl")]
    if base_model:
        args += ["--base-model", "mock-only"]
    if opt_in:
        args += ["--max-pairs-per-chunk", "128", "--relation-multilabel", "--relation-batch-size", "16"]
    monkeypatch.setattr(sys, "argv", args)
    run_pipeline.main()
    assert len(configurations) == 1
    config = configurations[0]
    assert config.max_pairs_per_chunk == (128 if opt_in else 0)
    assert config.relation_multilabel is opt_in
    assert config.relation_batch_size == (16 if opt_in else 128)


def test_closed_world_builder_requires_explicit_flag_before_read_or_write(tmp_path):
    output = tmp_path / "must-not-exist"
    completed = subprocess.run([
        sys.executable, "-B", "-m", "generative.build_data",
        "--train", str(tmp_path / "also-does-not-exist.json"),
        "--output-dir", str(output),
    ], cwd=AGENT_ROOT, text=True, capture_output=True)
    assert completed.returncode == 2
    assert "--assume-closed-world-silver" in completed.stderr
    assert "unannotated silver pairs" in completed.stderr
    assert "FileNotFoundError" not in completed.stderr
    assert not output.exists()

"""Synthetic-only tests. No private medical records are stored in the repository."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_repair.pipeline import repair_record, run_pipeline, text_repairs
from data_repair.verify import verify_release
from kg_agent.contracts import ONTOLOGY


TEXT = "Fatigue is a core symptom of Condition A and supports diagnosis of Condition A."


def record(text=TEXT, graph=None):
    graph = graph or {
        "entities": [{"id": "S1", "label": "Symptom", "name": "Fatigue"},
                     {"id": "D1", "label": "Disease", "name": "Condition A"}],
        "relations": [{"source": "S1", "target": "D1", "relation": label, "evidence": text}
                      for label in ("is_core_symptom_of", "supports_diagnosis_of")],
    }
    return {"messages": [{"role": "system", "content": ONTOLOGY["version"]},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(graph)}]}


def graph_of(value):
    return json.loads(value["messages"][2]["content"])


def change_graph(value, graph):
    value["messages"][2]["content"] = json.dumps(graph)
    return value


def fix(value):
    return repair_record(value, 1, 0, "train")


def test_multilabel_and_original_are_preserved():
    value = record()
    before = copy.deepcopy(value)
    result = fix(value)
    assert value == before == result["original_record"]
    assert len(result["pairs"]) == 1
    assert len(result["pairs"][0]["relations"]) == 2
    assert result["pairs"][0]["status"] == "mechanically_supported_silver"
    assert result["unannotated_pair_policy"] == "UNKNOWN_NOT_VERIFIED_NONE"


def test_whitelist_repairs_source_entity_evidence_and_offsets():
    text = "Pregnullcy is mentioned with Condition A."
    graph = graph_of(record())
    graph["entities"][0]["name"] = "Pregnullcy"
    for r in graph["relations"]:
        r["evidence"] = text
    result = fix(record(text, graph))
    assert result["source"]["text"] == "Pregnancy is mentioned with Condition A."
    hit = result["entities"][0]["mentions"][0]
    assert result["source"]["original_text"][hit["original_start"]:hit["original_end"]] == "Pregnullcy"
    assert result["source"]["text"][hit["start"]:hit["end"]] == "Pregnancy"
    assert all(a["evidence"] for a in result["annotations"])


def test_no_global_null_replacement_or_id_repair():
    value = {"id": "pregnullcy", "name": "pregnullcy", "uri": "https://x/pregnullcy", "description": "null hypothesis nullify"}
    repaired, _ = text_repairs(value)
    assert repaired["id"] == value["id"] and repaired["uri"] == value["uri"]
    assert repaired["name"] == "pregnancy"
    assert repaired["description"] == value["description"]


def test_repair_is_idempotent():
    value = {"name": "pregnullcy", "properties": {"Name": "finullces"}}
    first, changes = text_repairs(value)
    second, again = text_repairs(first)
    assert changes and not again and second == first


def test_missing_evidence_never_falls_back_or_becomes_none():
    value = record()
    graph = graph_of(value)
    graph["relations"][0]["evidence"] = "A sentence that does not occur."
    result = fix(change_graph(value, graph))
    assert "evidence_not_found" in result["annotations"][0]["issues"]
    assert result["annotations"][0]["evidence"] == []
    assert result["pairs"][0]["status"] == "review"
    assert len(result["pairs"][0]["relations"]) == 2


def test_all_evidence_fragments_must_resolve():
    value = record()
    graph = graph_of(value)
    graph["relations"][0]["evidence"] = [TEXT, "Missing second evidence."]
    result = fix(change_graph(value, graph))
    assert result["pairs"][0]["status"] == "review"


def test_repeated_evidence_is_not_arbitrarily_bound_to_first_occurrence():
    value = record(TEXT + " " + TEXT)
    graph = graph_of(value)
    for edge in graph["relations"]:
        edge["evidence"] = TEXT
    result = fix(change_graph(value, graph))
    assert "evidence_occurrence_ambiguous" in result["annotations"][0]["issues"]
    assert len(result["annotations"][0]["evidence"]) == 2


def test_duplicate_ids_quarantine_pair():
    value = record()
    graph = graph_of(value)
    graph["entities"].append(copy.deepcopy(graph["entities"][0]))
    result = fix(change_graph(value, graph))
    assert "duplicate_entity_ids" in result["issues"]
    assert all(p["status"] == "review" for p in result["pairs"])


def test_label_case_is_safe_but_unknown_semantic_alias_is_not_guessed():
    value = record()
    graph = graph_of(value)
    graph["entities"][0]["label"] = "symptom"
    result = fix(change_graph(value, graph))
    assert result["entities"][0]["label"] == "Symptom"
    graph["entities"][0]["label"] = "ClinicalProblem"
    assert "unknown_entity_label" in fix(change_graph(value, graph))["entities"][0]["issues"]


def test_unknown_endpoint_and_label_do_not_crash():
    value = record()
    graph = graph_of(value)
    graph["relations"][0].update(source=None, relation=[])
    result = fix(change_graph(value, graph))
    assert "unknown_or_ambiguous_endpoint" in result["annotations"][0]["issues"]
    assert "invalid_relation_label" in result["annotations"][0]["issues"]


def test_negation_requires_review_not_automatic_label_flip():
    value = record("Fatigue is not a core symptom of Condition A.")
    result = fix(value)
    assert "negation_scope_needs_review" in result["annotations"][0]["issues"]
    assert result["pairs"][0]["relations"][0]["relation"] == "is_core_symptom_of"


def test_cross_sentence_reference_is_review_not_declared_false():
    text = "Condition A is described here. Fatigue is a core symptom."
    value = record(text)
    graph = graph_of(value)
    for edge in graph["relations"]:
        edge["evidence"] = "Fatigue is a core symptom."
    result = fix(change_graph(value, graph))
    assert "evidence_endpoint_binding_needs_review" in result["annotations"][0]["issues"]
    assert result["pairs"][0]["status"] == "review"
    assert result["semantic_adjudication"] == "not_performed"


def test_no_speculative_english_alias_or_entity_merge():
    value = record("Fatigue occurs with Generalised anxiety disorder.")
    graph = graph_of(value)
    graph["entities"][1]["name"] = "Generalized anxiety disorder"
    result = fix(change_graph(value, graph))
    assert "entity_not_grounded" in result["entities"][1]["issues"]
    assert result["entities"][1]["name"] == "Generalized anxiety disorder"


def test_duplicate_labels_preserve_all_annotation_provenance():
    value = record()
    graph = graph_of(value)
    graph["relations"].append(copy.deepcopy(graph["relations"][0]))
    result = fix(change_graph(value, graph))
    assert len(result["annotations"]) == 3
    assert len(result["pairs"][0]["relations"]) == 2
    assert result["pairs"][0]["relations"][0]["original_annotation_indices"] == [0, 2]


def test_span_offsets_and_abbreviations_survive():
    result = fix(record("Condition A, e.g. fatigue at 1.5 units, is described."))
    assert all(result["source"]["text"][s["start"]:s["end"]] == s["text"] for s in result["spans"])
    assert not any(s["text"] in {"e", "g", "1", "5"} for s in result["spans"])


def fixture_files(tmp_path, *, overlap=False):
    icd = record(json.dumps({"uri": "https://id.who.int/icd/entity/1", "code": "6A00", "title": "Condition A"}))
    other = record(TEXT if overlap else TEXT.replace("Fatigue", "Tiredness"))
    records = [icd, record(), copy.deepcopy(icd), other]
    path = tmp_path / "source.json"
    path.write_text(json.dumps(records))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"source_train_sha256": digest, "train_groups": [0], "tune_groups": [1]}))
    return path, split, digest


def test_full_run_conserves_records_freezes_split_and_guards_release(tmp_path):
    path, split, digest = fixture_files(tmp_path)
    out = tmp_path / "release"
    result = run_pipeline(path, split, out, expected_sha256=digest)
    assert result["raw_record_conservation"] and result["input_unchanged"]
    assert result["summary"]["records"] == 4
    assert result["train_groups"] == result["tune_groups"] == 1
    assert not any(result["release_gates"].values())
    assert len((out / "records.jsonl").read_text().splitlines()) == 4
    assert verify_release(out)["status"] == "integrity_checks_passed"
    for line in (out / "pair_candidates.jsonl").read_text().splitlines():
        assert not json.loads(line)["ready_for_automatic_training"]
    with pytest.raises(ValueError, match="refusing overwrite"):
        run_pipeline(path, split, out, expected_sha256=digest)


def test_hash_and_schema_guard(tmp_path):
    path, split, digest = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_pipeline(path, split, tmp_path / "release", expected_sha256="wrong")
    records = json.loads(path.read_text())
    records[1]["messages"][0]["content"] = "schema-v2"
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="schema provenance"):
        run_pipeline(path, split, tmp_path / "release", expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_cross_split_source_duplicates_are_quarantined(tmp_path):
    path, split, digest = fixture_files(tmp_path, overlap=True)
    result = run_pipeline(path, split, tmp_path / "release", expected_sha256=digest)
    assert result["cross_split_source_hashes"] == 1
    assert (tmp_path / "release/pair_candidates.jsonl").read_text() == ""


def test_invalid_teacher_kept_for_review():
    value = record()
    value["messages"][2]["content"] = "not json"
    result = fix(value)
    assert result["status"] == "review" and result["original_record"] == value
    assert "invalid_teacher_graph" in result["issues"]


def test_integrity_verifier_detects_output_modification(tmp_path):
    path, split, digest = fixture_files(tmp_path)
    out = tmp_path / "release"
    run_pipeline(path, split, out, expected_sha256=digest)
    (out / "pair_candidates.jsonl").write_text("")
    with pytest.raises(ValueError, match="Output hash mismatch"):
        verify_release(out)


def test_frozen_split_source_mismatch_rejected(tmp_path):
    path, split, digest = fixture_files(tmp_path)
    value = json.loads(split.read_text())
    value["source_train_sha256"] = "different"
    split.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different source"):
        run_pipeline(path, split, tmp_path / "release", expected_sha256=digest)

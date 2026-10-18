"""Shared non-JSON protocol: literal semantics, full provenance and raw scoring."""

from copy import deepcopy
import json

import pytest

from semantic.protocol import (
    CONTRACT, ENTITY_COLUMNS, RELATION_COLUMNS, TARGET_TYPE_RULES,
    assemble_graph, build_entity_prompt, build_relation_prompt, decode_records,
    entity_id, escape_field, normalize_name, parse_entities, parse_relations,
    render_entities, render_relations, unescape_field,
)


TEXT = "Severe fatigue is a core symptom of Condition Alpha. Severe fatigue also supports its diagnosis."


def title_metadata(title="Condition Alpha, severe recurrent episode"):
    return {"title_context": {"text": title, "source_group_id": "same-input-group",
        "source_record_sha256": "a" * 64, "source_start": 31, "source_end": 31 + len(title)}}


def entities(text=TEXT):
    return parse_entities(text, render_entities([
        {"label": "Symptom", "mention": "Severe fatigue", "name": "Severe fatigue"},
        {"label": "Disease", "mention": "Condition Alpha", "name": "Condition Alpha"},
    ]))


def relation(result, label="is_core_symptom_of", evidence=None):
    source = next(e for e in result["entities"] if e["label"] == "Symptom")
    target = next(e for e in result["entities"] if e["label"] == "Disease")
    return {"source": source["id"], "target": target["id"], "relation": label,
            "evidence": evidence or TEXT.split(". ")[0] + "."}


@pytest.mark.parametrize("value", ["plain", "诊断标准", "literal \\n and \\t", "tab\tnewline\nCR\rslash\\", "\\\\\n\t"])
def test_escaped_fields_are_fully_reversible(value):
    assert unescape_field(escape_field(value)) == value
    assert "\n" not in escape_field(value) and "\t" not in escape_field(value)


@pytest.mark.parametrize("value", ["\\", "bad\\q", "bad\\u00FF"])
def test_undeclared_escapes_are_not_repaired(value):
    with pytest.raises(ValueError, match="escape"):
        unescape_field(value)


def test_literal_relation_full_label_names_and_program_owned_json():
    er = entities()
    assert er["valid"] and len(er["entities"]) == len(er["filtered_entities"]) == 2
    assert all(e["id"].startswith("ent_") and e["grounding_verified"] for e in er["entities"])
    completion = render_relations([relation(er)], er["entities"])
    assert completion.split("\t")[:3] == ["Symptom", "Severe fatigue", "is_core_symptom_of"]
    assert all(e["id"] not in completion for e in er["entities"])
    rr = parse_relations(TEXT, er["entities"], completion)
    assert rr["valid"] and rr["relations"] == rr["filtered_relations"]
    graph = assemble_graph(er, rr)
    assert len(graph["relations"]) == 1
    assert json.loads(json.dumps(graph)) == graph
    assert set(graph["entities"][0]) == {"id", "label", "name"}
    assert set(graph["relations"][0]) == {"source", "target", "relation", "evidence"}
    assert rr["trace"]["eos_checked"] is False  # parser never pretends to check generation EOS


def test_canonical_title_missing_from_body_is_retained_with_true_input_provenance():
    text = "Fatigue may be persistent during an episode."
    metadata = title_metadata()
    title = metadata["title_context"]["text"]
    result = parse_entities(text, render_entities([
        {"label": "Disease", "mention": title, "name": title},
        {"label": "Symptom", "mention": "Fatigue", "name": "Fatigue"},
    ]), metadata)
    assert len(result["filtered_entities"]) == 2
    disease = next(e for e in result["entities"] if e["label"] == "Disease")
    assert disease["canonical_name_source"] == "input_title"
    assert disease["name"] == title and "severe recurrent" in disease["name"]
    anchor = disease["mentions"][0]
    assert anchor["source_start"] == 31 and anchor["source_end"] == 31 + len(title)
    assert anchor["source_record_sha256"] == "a" * 64
    assert anchor["source_group_id"] == "same-input-group"
    assert anchor["context"] == "title_context"
    prompt = build_entity_prompt(text, metadata)
    assert title in prompt and "SOURCE TITLE" in prompt


@pytest.mark.parametrize("change", ["hash", "offset", "bool_offset", "missing", "extra"])
def test_title_requires_the_exact_explicit_provenance_contract(change):
    metadata = title_metadata()
    title = metadata["title_context"]
    if change == "hash": title["source_record_sha256"] = "invented"
    if change == "offset": title["source_end"] += 1
    if change == "bool_offset": title["source_start"] = True
    if change == "missing": del title["source_group_id"]
    if change == "extra": title["gold_aliases"] = ["fake"]
    with pytest.raises(ValueError):
        build_entity_prompt(TEXT, metadata)


def test_metadata_gold_aliases_never_modify_input_or_admission():
    row = {"label": "Disease", "mention": "Condition Alpha", "name": "Invented Chronic Syndrome"}
    poisoned = {"gold": {"name": row["name"]}, "aliases": {row["mention"]: row["name"]},
                "title": row["name"], "entities": [row], "relation_labels": ["fake"]}
    assert build_entity_prompt(TEXT) == build_entity_prompt(TEXT, poisoned)
    plain = parse_entities(TEXT, render_entities([row]))
    assert plain == parse_entities(TEXT, render_entities([row]), poisoned)
    assert plain["valid"] and len(plain["entities"]) == 1
    assert not plain["filtered_entities"]
    assert plain["entities"][0]["canonical_name_source"] == "model_inferred"
    assert plain["trace"]["unverified_canonical_names"] == 1
    assert plain["rejected_entities"][0]["reasons"][0]["reason"] == "canonical_name_not_in_input"


def test_unsupported_names_mentions_are_raw_fp_candidates_and_not_stage_errors():
    completion = render_entities([
        {"label": "Disease", "mention": "Condition Alpha", "name": "Invented Syndrome"},
        {"label": "Symptom", "mention": "Imaginary weakness", "name": "Imaginary weakness"},
        {"label": "Symptom", "mention": "Severe fatigue", "name": "Severe fatigue"},
    ])
    result = parse_entities(TEXT, completion)
    assert result["valid"] and not result["errors"]
    assert len(result["entities"]) == 3 and len(result["filtered_entities"]) == 1
    assert len(result["rejected_entities"]) == 2
    assert assemble_graph(result, {"valid": True, "relations": [], "filtered_relations": []}, filtered=False)["entities"]


def test_name_keys_preserve_qualifiers_negation_numbers_punctuation_and_offsets():
    text = "Not severe disorder, type 2; severe disorder, type 1. Ａlpha\n syndrome."
    rows = [{"label": "Disease", "mention": name, "name": name} for name in (
        "Not severe disorder, type 2", "severe disorder, type 1", "Alpha syndrome")]
    result = parse_entities(text, render_entities(rows))
    assert len(result["filtered_entities"]) == 3
    assert len({e["id"] for e in result["entities"]}) == 3
    assert normalize_name("not A, type 2") != normalize_name("A type 2")
    alpha = next(e for e in result["entities"] if e["name"] == "Alpha syndrome")
    anchor = alpha["mentions"][0]
    assert text[anchor["start"]:anchor["end"]] == "Ａlpha\n syndrome"
    assert entity_id("Disease", "Ａlpha\n syndrome") == alpha["id"]


def test_source_order_and_duplicate_entity_rows_do_not_change_stable_ids():
    rows = [
        {"label": "Disease", "mention": "Condition Alpha", "name": "Condition Alpha"},
        {"label": "Symptom", "mention": "Severe fatigue", "name": "Severe fatigue"}]
    forward = parse_entities(TEXT, render_entities(rows))
    reverse = parse_entities(TEXT, render_entities(rows[::-1] + rows))
    assert [(e["id"], e["name"]) for e in forward["entities"]] == [(e["id"], e["name"]) for e in reverse["entities"]]
    symptom = next(e for e in reverse["entities"] if e["label"] == "Symptom")
    assert len(symptom["mentions"]) == 2 and len(symptom["source_lines"]) == 2


def test_multiple_labels_multiple_evidence_records_and_reversed_order():
    er = entities()
    first = relation(er)
    second = relation(er, "supports_diagnosis_of", "Severe fatigue also supports its diagnosis.")
    third = {**first, "evidence": TEXT}
    completion = render_relations([second, third, first, first], er["entities"])
    result = parse_relations(TEXT, er["entities"], completion)
    assert result["valid"] and len(result["relations"]) == len(result["filtered_relations"]) == 2
    core = next(r for r in result["relations"] if r["relation"] == "is_core_symptom_of")
    assert len(core["evidence_texts"]) == 2
    assert core["evidence_texts"] == [TEXT, first["evidence"]]
    assert len(core["source_lines"]) == 3
    assert parse_relations(TEXT, er["entities"], render_relations(result["relations"], er["entities"]))["valid"]


def test_multiline_literal_evidence_roundtrip_without_json_repair():
    text = "Severe fatigue\n\tis a core symptom of Condition Alpha.\\source"
    er = entities(text)
    target = relation(er, evidence=text)
    completion = render_relations([target], er["entities"])
    assert len(completion.splitlines()) == 1
    result = parse_relations(text, er["entities"], completion)
    assert result["valid"] and result["filtered_relations"][0]["evidence"] == text


def test_target_type_mismatch_keeps_raw_prediction_but_filters_only_bad_edge():
    er = entities()
    symptom = next(e for e in er["entities"] if e["label"] == "Symptom")
    bad = {**relation(er), "target": symptom["id"]}
    result = parse_relations(TEXT, er["entities"], render_relations([relation(er), bad], er["entities"]))
    assert result["valid"] and len(result["relations"]) == 2 and len(result["filtered_relations"]) == 1
    reasons = result["rejected_relations"][0]["reasons"]
    assert {reason["reason"] for reason in reasons} == {"target_type_mismatch", "self_relation"}
    assert len(assemble_graph(er, result)["relations"]) == 1
    assert len(assemble_graph(er, result, filtered=False)["relations"]) == 2


def test_unsupported_evidence_is_kept_raw_and_filtered_separately():
    er = entities()
    result = parse_relations(TEXT, er["entities"], render_relations([relation(er, evidence="Fabricated evidence.")], er["entities"]))
    assert result["valid"] and len(result["relations"]) == 1
    assert not result["filtered_relations"]
    assert result["rejected_relations"][0]["reasons"] == [{"reason": "evidence_not_in_body"}]


def test_short_relation_window_keeps_verified_full_document_entity_table():
    er = entities()
    window = "Severe fatigue also supports its diagnosis."
    target = relation(er, "supports_diagnosis_of", window)
    result = parse_relations(window, er["entities"], render_relations([target], er["entities"]))
    assert "Condition Alpha" not in window
    assert result["valid"] and len(result["filtered_relations"]) == 1
    ungrounded = deepcopy(er["entities"])
    next(e for e in ungrounded if e["label"] == "Disease")["grounding_verified"] = False
    rejected = parse_relations(window, ungrounded, render_relations([target], ungrounded))
    assert rejected["valid"] and len(rejected["relations"]) == 1 and not rejected["filtered_relations"]


@pytest.mark.parametrize("bad", ["{\"entities\":[]}", "1\n", "Disease\tCondition Alpha\n", "Disease\tCondition Alpha\tBad\\q\n", "Unknown\tSevere fatigue\tSevere fatigue\n"])
def test_syntax_errors_empty_entire_stage_but_preserve_all_raw_and_partial_trace(bad):
    correct = render_entities([{"label": "Symptom", "mention": "Severe fatigue", "name": "Severe fatigue"}])
    result = parse_entities(TEXT, correct + bad)
    assert result["valid"] is False and result["entities"] == result["filtered_entities"] == []
    assert result["raw_completion"] == correct + bad
    assert result["trace"]["parsed_entities"] and result["errors"]
    assert assemble_graph(result, {"valid": True, "relations": []}) == {"entities": [], "relations": []}


@pytest.mark.parametrize("change", ["unknown_endpoint", "unknown_relation", "wrong_source", "blank_record"])
def test_invalid_relation_rows_cannot_materialize_partial_graph_edges(change):
    er = entities()
    valid = render_relations([relation(er)], er["entities"])
    if change == "unknown_endpoint": bad = valid.replace("Condition Alpha", "New Disease")
    elif change == "unknown_relation": bad = valid.replace("is_core_symptom_of", "123")
    elif change == "wrong_source": bad = valid.replace("is_core_symptom_of", "subtype_of")
    else: bad = "\n"
    result = parse_relations(TEXT, er["entities"], valid + bad)
    assert not result["valid"] and result["relations"] == result["filtered_relations"] == []
    assert result["trace"]["parsed_relations"] and result["raw_completion"] == valid + bad
    assert assemble_graph(er, result)["relations"] == []
    assert len(assemble_graph(er, result)["entities"]) == 2


def test_empty_valid_completion_has_no_none_label_and_no_inferred_negative_supervision():
    result = parse_entities(TEXT, "")
    assert result["valid"] and result["entities"] == []
    er = entities()
    rr = parse_relations(TEXT, er["entities"], "")
    assert rr["valid"] and rr["relations"] == []
    assert render_entities([]) == render_relations([], er["entities"]) == ""
    assert "negative" not in rr


def test_relation_prompt_is_linear_entity_table_not_all_pairs_and_has_no_ids():
    nodes = [{"id": f"private-{index}", "label": "Disease", "name": f"Disease-{index}"} for index in range(150)]
    prompt = build_relation_prompt("Clinical source", nodes)
    for node in nodes:
        assert prompt.count("Disease\t" + node["name"] + "\n") == 1
        assert node["id"] not in prompt
    assert "pair_id" not in prompt and "P0001" not in prompt
    assert len(prompt) < 20000
    assert "source_name<TAB>relation_label<TAB>target_type<TAB>target_name" in prompt


def test_ambiguous_normalized_type_name_reference_is_rejected_before_generation():
    nodes = entities()["entities"]
    duplicate = deepcopy(nodes[0])
    duplicate.update(id="different", name=duplicate["name"].upper())
    with pytest.raises(ValueError, match="unique"):
        build_relation_prompt(TEXT, nodes + [duplicate])
    with pytest.raises(ValueError, match="unique"):
        parse_relations(TEXT, nodes + [duplicate], "")


def test_all_guards_are_explicit_ontology_source_labels_and_existing_target_types():
    from semantic.protocol import SOURCE_RELATIONS
    from kg_agent.contracts import SCHEMA_LABELS
    for (source, label), targets in TARGET_TYPE_RULES.items():
        assert label in SOURCE_RELATIONS[source]
        assert targets and set(targets) <= set(SCHEMA_LABELS)


def test_decode_utility_retains_line_errors_without_dropping_bad_rows_silently():
    rows, errors = decode_records("Disease\tA\tA\ninvalid\n", ENTITY_COLUMNS)
    assert rows == [(1, {"label": "Disease", "mention": "A", "name": "A"})]
    assert errors[0]["line"] == 2 and errors[0]["raw_line"] == "invalid"
    assert len(RELATION_COLUMNS) == 6 and CONTRACT == "semantic-text-v1"

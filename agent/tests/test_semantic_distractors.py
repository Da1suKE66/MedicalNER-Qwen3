"""Matched input corruption proof tests with invented source names only."""
from copy import deepcopy
import json

import pytest

from semantic.data import compile_source, sha, validate_row
from semantic.distractors import (build_donor_pool, choose_donors, compile_row,
                                  validate_distractor_row, _lengths, position_summary,
                                  _verify_previous_release)
from semantic import protocol as p
from test_semantic_data import prepared, FakeTokenizer


def document(name, group, *, text=None, split="train"):
    item = prepared(text or f"{name} has beta.", title=name, split=split)
    item["group_id"], item["source_index"] = group, group * 10
    return item


def setup(recipient=None):
    items = [recipient or document("Alpha", 3), document("Gamma", 4), document("Omega", 5)]
    rows = [compile_source(item)[0] for item in items]
    entities = [next(r for r in group if r["stage"] == "entities") for group in rows]
    contexts, donors = build_donor_pool(entities, {})
    base = next(r for r in rows[0] if r["stage"] == "relations")
    return base, contexts[base["metadata"]["source_record_sha256"]], donors


def test_input_only_corruption_preserves_every_positive_and_unknown_policy():
    base, context, donors = setup()
    original = deepcopy(base)
    row = compile_row(base, context, donors)
    assert base == original
    assert row["variant"] == "full_source_distractors"
    assert row["data_version"] == base["data_version"]
    for key in ("completion", "target", "target_sha256", "target_count", "split", "stage", "group_id"):
        assert row[key] == base[key]
    assert row["metadata"]["supervised_atom_ids"] == base["metadata"]["supervised_atom_ids"]
    assert row["metadata"]["unannotated_pair_policy"] == base["metadata"]["unannotated_pair_policy"]
    trace = row["metadata"]["input_corruption"]
    assert 1 <= trace["added_count"] <= 2
    assert len(row["task_input"]["entities"]) == len(base["task_input"]["entities"]) + trace["added_count"]
    assert row["task_input"]["entities"][:len(base["task_input"]["entities"])] == base["task_input"]["entities"]
    assert validate_row(row)
    assert validate_distractor_row(row, base, context, donors)
    assert "input_corruption" not in row["prompt"]
    for donor in trace["added_entities"]:
        assert donor["donor_split"] == "train" and donor["donor_source_group_id"] != row["group_id"]
        assert donor["full_body_matches"] == donor["trusted_title_matches"] == 0
        assert donor["clinical_unrelatedness_adjudicated"] is False


def test_corruption_does_not_expand_relation_vocabulary():
    base, context, donors = setup()
    row = compile_row(base, context, donors)
    marker = "RELATION VOCABULARY AND OPTIONAL TARGET-TYPE GUARDS:"
    assert row["prompt"].split(marker, 1)[1] == base["prompt"].split(marker, 1)[1]


def test_absence_checks_full_source_outside_short_recipient_window():
    item = document("Alpha", 3, text="Alpha has beta. " + "Background words. " * 30 + " Gamma.")
    quote = "Alpha has beta."
    item["annotations"][0]["annotation"]["evidence"] = quote
    item["annotations"][0]["evidence_fragments"] = [[{"original_start": 0, "original_end": len(quote), "original_text": quote}]]
    base_full, context, donors = setup(item)
    base_short = next(r for r in compile_source(item, variant="short_source", short_chars=64)[0] if r["stage"] == "relations")
    assert "Gamma" not in base_short["task_input"]["text"]
    row = compile_row(base_short, context, donors)
    assert "Gamma" not in [d["name"] for d in row["metadata"]["input_corruption"]["added_entities"]]
    assert row["metadata"]["input_corruption"]["recipient_full_body_sha256"] == sha(item["source"]["original_text"])


def test_trusted_title_blocks_same_name_even_when_absent_from_body():
    base, context, donors = setup()
    title_raw = '{"title":"Gamma","code":"QX"}'
    original = {"messages": [{"role": "user", "content": title_raw}]}
    context_title = {"text": "Gamma", "source_group_id": "3", "source_record_sha256": sha(original),
                     "source_start": title_raw.index("Gamma"), "source_end": title_raw.index("Gamma") + 5}
    item = document("Alpha", 3)
    rows = compile_source(item, context_title)[0]
    erow = next(r for r in rows if r["stage"] == "entities")
    context_map, _ = build_donor_pool([erow], {sha(original): {"source_group_id": "3", "original_record": original}})
    base = next(r for r in rows if r["stage"] == "relations")
    row = compile_row(base, context_map[base["metadata"]["source_record_sha256"]], donors)
    assert "Gamma" not in [d["name"] for d in row["metadata"]["input_corruption"]["added_entities"]]


def test_same_donors_for_matched_full_and_short_views():
    base, context, donors = setup()
    short = next(r for r in compile_source(document("Alpha", 3), variant="short_source")[0] if r["stage"] == "relations")
    full_row, short_row = compile_row(base, context, donors), compile_row(short, context, donors)
    left = full_row["metadata"]["input_corruption"]["added_entities"]
    right = short_row["metadata"]["input_corruption"]["added_entities"]
    assert [d["donor_identity_sha256"] for d in left] == [d["donor_identity_sha256"] for d in right]


def test_ghosts_have_no_fabricated_recipient_spans_and_position_is_disclosed():
    base, context, donors = setup()
    row = compile_row(base, context, donors)
    trace = row["metadata"]["input_corruption"]
    by_id = {e["id"]: e for e in row["task_input"]["entities"]}
    ordered = p._ordered(row["task_input"]["entities"], row["metadata"]["order"])
    for d in trace["added_entities"]:
        entity = by_id[d["injected_entity_id"]]
        assert entity["mentions"] == [] and entity["grounding_verified"] is False
        assert ordered[d["display_entity_table_index"]]["id"] == entity["id"]
    assert "position" in trace["position_caveat"]


def test_no_donor_preserves_row_and_reports_unperturbed():
    base, context, _ = setup()
    row = compile_row(base, context, [])
    assert row["prompt"] == base["prompt"] and row["completion"] == base["completion"]
    trace = row["metadata"]["input_corruption"]
    assert trace["added_count"] == 0 and not trace["model_input_changed"]
    assert trace["unperturbed_reason"] == "no_eligible_donor_within_finite_search"


def test_finite_search_never_drops_a_row():
    base, context, donors = setup()
    same_group = next(d for d in donors if d["donor_source_group_id"] == "3")
    row = compile_row(base, context, [same_group], max_candidates=1)
    assert row["completion"] == base["completion"]
    trace = row["metadata"]["input_corruption"]
    assert trace["recipe"]["candidates_examined"] == 1
    assert trace["recipe"]["search_status"] == "finite_search_exhausted"
    assert trace["added_count"] == 0


def test_token_budget_can_remove_corruption_but_never_target_tokens():
    base, context, donors = setup()
    tokenizer = FakeTokenizer()
    budget = _lengths(base, tokenizer)["total_tokens"]
    row = compile_row(base, context, donors, tokenizer=tokenizer, max_length=budget)
    assert row["prompt"] == base["prompt"] and row["completion"] == base["completion"]
    trace = row["metadata"]["input_corruption"]
    assert trace["added_count"] == 0 and trace["token_budget_removed_donors"] > 0
    assert trace["unperturbed_reason"] == "token_budget"
    assert validate_distractor_row(row, base, context, donors, tokenizer=tokenizer, max_length=budget)
    with pytest.raises(ValueError, match="Unmodified base row"):
        compile_row(base, context, donors, tokenizer=tokenizer, max_length=1)


def test_tune_donors_and_tune_recipients_rejected():
    tune = next(r for r in compile_source(document("Alpha", 3, split="tune"))[0] if r["stage"] == "entities")
    with pytest.raises(ValueError, match="Held-out"):
        build_donor_pool([tune], {})
    base, context, donors = setup()
    bad = deepcopy(donors[0]); bad["donor_split"] = "tune"
    with pytest.raises(ValueError, match="non-train"):
        compile_row(base, context, [bad])
    base["split"] = "tune"
    with pytest.raises(ValueError, match="Held-out"):
        compile_row(base, context, donors)


@pytest.mark.parametrize("field", ["completion", "target_sha256", "prompt"])
def test_deterministic_reconstruction_rejects_mutation(field):
    base, context, donors = setup()
    row = compile_row(base, context, donors)
    row[field] += "changed"
    with pytest.raises(ValueError, match="reconstruction"):
        validate_distractor_row(row, base, context, donors)


def test_context_hash_and_cross_group_tampering_rejected():
    base, context, donors = setup()
    bad = deepcopy(context); bad["body"] += "changed"
    with pytest.raises(ValueError, match="hash drift"):
        compile_row(base, bad, donors)
    bad = deepcopy(context); bad["source_group_id"] = "other"
    with pytest.raises(ValueError, match="identity mismatch"):
        compile_row(base, bad, donors)


def test_source_absence_is_not_a_clinical_negative_claim():
    base, context, donors = setup()
    row = compile_row(base, context, donors)
    added_ids = {d["injected_entity_id"] for d in row["metadata"]["input_corruption"]["added_entities"]}
    assert all(t["source"] not in added_ids and t["target"] not in added_ids for t in row["target"])
    assert "NONE" not in row["completion"]
    assert "not_clinical_negative" in row["metadata"]["input_corruption"]["interpretation"]


def test_canonical_control_preserves_targets_and_same_donor_recipe():
    base, context, donors = setup()
    canonical_base = next(r for r in compile_source(document("Alpha", 3), variant="short_canonical")[0]
                          if r["stage"] == "relations")
    before = deepcopy(canonical_base)
    source_row = compile_row(base, context, donors)
    row = compile_row(canonical_base, context, donors)
    assert canonical_base == before
    assert row["variant"] == "short_canonical_distractors"
    assert row["metadata"]["order"] == canonical_base["metadata"]["order"] == "canonical"
    assert row["completion"] == canonical_base["completion"]
    assert row["target"] == canonical_base["target"]
    assert row["metadata"]["supervised_atom_ids"] == canonical_base["metadata"]["supervised_atom_ids"]
    source_trace = source_row["metadata"]["input_corruption"]
    trace = row["metadata"]["input_corruption"]
    assert trace["recipe"] == source_trace["recipe"]
    assert [d["donor_identity_sha256"] for d in trace["added_entities"]] == [
        d["donor_identity_sha256"] for d in source_trace["added_entities"]]
    ordered = p._ordered(row["task_input"]["entities"], "canonical")
    keys = [(e["label"], p.normalize_name(e["name"])) for e in ordered]
    assert keys == sorted(keys)
    assert all(d["display_entity_table_index"] > 0 for d in trace["added_entities"])
    assert "naturally interleaves" in trace["position_caveat"]
    assert validate_distractor_row(row, canonical_base, context, donors)


def test_actual_position_statistics_use_display_indices_and_condition_on_type():
    rows = [{"task_input": {"entities": [{}] * 6}, "metadata": {"input_corruption": {
        "added_entities": [{"label": label, "display_entity_table_index": index}
                           for label, index in [("Disease", 0), ("Disease", 2), ("Symptom", 5)]]}}}]
    result = position_summary(rows)
    assert result["overall"]["thirds"] == {"head": 1, "middle": 1, "tail": 1}
    assert result["overall"]["exact_index"] == {0: 1, 2: 1, 5: 1}
    assert result["overall"]["index_and_table_size"] == {"0/6": 1, "2/6": 1, "5/6": 1}
    assert result["overall"]["literal_first"] == result["overall"]["literal_last"] == 1
    assert result["by_entity_type"]["Disease"]["thirds"] == {"head": 1, "middle": 1}
    assert result["by_entity_type"]["Symptom"]["thirds"] == {"tail": 1}


def test_previous_release_guard_checks_base_recipe_and_every_file(tmp_path):
    contents = b"frozen data\n"
    (tmp_path / "old.jsonl").write_bytes(contents)
    parameters = {"seed": 42, "max_candidates": 128, "max_length": 8192}
    manifest = {"base_release_manifest_sha256": "base", "parameters": parameters,
                "output_sha256": {"old.jsonl": sha(contents)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert _verify_previous_release(tmp_path, "base", parameters) == manifest
    with pytest.raises(ValueError, match="different frozen base"):
        _verify_previous_release(tmp_path, "changed", parameters)
    with pytest.raises(ValueError, match="donor recipe"):
        _verify_previous_release(tmp_path, "base", {**parameters, "seed": 1})
    (tmp_path / "old.jsonl").write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA drift"):
        _verify_previous_release(tmp_path, "base", parameters)

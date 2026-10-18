"""Input-only title provenance and complete historical regression denominators."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from semantic import historical as h
from semantic import evaluate as e
from semantic.data import canonical, sha


URI = "http://id.who.int/icd/release/11/2025-01/mms/123456"
TITLE = "Condition Alpha"
TEXT = "Severe fatigue is a core symptom of Condition Alpha."


def record(text, graph):
    return {"messages": [{"role": "user", "content": "Extract graph.\nMedical text:\n" + text},
                         {"role": "assistant", "content": json.dumps(graph)}]}


def fixture():
    graph = {"entities": [{"id": "s", "label": "Symptom", "name": "Severe fatigue"},
                           {"id": "d", "label": "Disease", "name": TITLE}],
             "relations": [{"source": "s", "target": "d", "relation": "is_core_symptom_of", "evidence": TEXT}]}
    # The title is deliberately repeated OUTSIDE the title field.
    originals = [record(URI + "\n6A01\n" + TITLE + "\nAdditional mention of " + TITLE, {"entities": [graph["entities"][1]], "relations": []}),
                 record(TEXT, graph), record("An unmatched original paragraph.", graph)]
    raw = {"entities": [{"id": URI, "title": TITLE, "code": "6A01", "definition": "Prefix: " + TEXT + " End."}]}
    matching = {"counts": {"exact:1": 1, "normalized:1": 1, "exact:0": 1, "normalized:0": 1},
        "matches": [{"sample_index": 1, "exact_source_ids": [URI], "nfkc_whitespace_source_ids": [URI], "input_only_anchor_sample_index": 0},
                    {"sample_index": 2, "exact_source_ids": [], "nfkc_whitespace_source_ids": []}],
        "historical_anchor_counts": {"unique_exact_match_and_historical_anchor": 1}}
    return originals, raw, matching


def compile_fixture():
    return h.compile_records(*fixture(), expected_samples=3)


def test_exact_source_identity_title_field_and_unmatched_gold_retained():
    cases, receipts, audit = compile_fixture()
    title = cases[1]["metadata"]["title_context"]
    user = cases[0]["original_record"]["messages"][0]["content"]
    assert user.count(TITLE) == 2
    assert user[title["source_start"]:title["source_end"]] == TITLE
    assert title["source_start"] == user.index(TITLE)
    assert cases[0]["source_group_id"] == cases[1]["source_group_id"] == "raw858:" + URI
    assert cases[2]["source_group_id"] == "unmatched_input_sha256:" + sha(cases[2]["text"])
    assert "title_context" not in cases[2]["metadata"]
    assert len(receipts) == 1
    assert audit["denominators"]["overall"] == {"samples": 3, "gold_entities": 5, "gold_relations": 2}
    assert audit["title_context_coverage"] == {"free_text_with_verified_title": 1, "free_text_without_title": 1, "structured_anchors": 1}
    assert all(c["gold"] == json.loads(c["original_record"]["messages"][1]["content"]) for c in cases)
    assert all(c["training_allowed"] is False and c["blind_test"] is False for c in cases)


def test_gold_name_cannot_create_title_or_source_association():
    original, raw, matching = fixture()
    before, _, _ = h.compile_records(original, raw, matching, expected_samples=3)
    changed = deepcopy(original)
    graph = json.loads(changed[2]["messages"][1]["content"])
    graph["entities"][1]["name"] = "A different hallucinated canonical disease"
    changed[2]["messages"][1]["content"] = json.dumps(graph)
    after, _, _ = h.compile_records(changed, raw, matching, expected_samples=3)
    assert after[2]["input_provenance"] == before[2]["input_provenance"]
    assert after[2]["source_group_id"] == before[2]["source_group_id"]
    assert "title_context" not in after[2]["metadata"]
    assert after[2]["gold_sha256"] != before[2]["gold_sha256"]


@pytest.mark.parametrize("mutation", ["anchor", "source_id", "missing", "duplicate", "title", "count"])
def test_unproven_matching_claims_rejected(mutation):
    original, raw, matching = fixture()
    if mutation == "anchor": matching["matches"][0]["input_only_anchor_sample_index"] = 2
    elif mutation == "source_id": matching["matches"][0]["exact_source_ids"] = ["another-source"]
    elif mutation == "missing": matching["matches"].pop()
    elif mutation == "duplicate": matching["matches"].append(deepcopy(matching["matches"][0]))
    elif mutation == "title": raw["entities"][0]["title"] += " invented"
    else: matching["counts"]["exact:1"] = 2
    with pytest.raises(ValueError):
        h.compile_records(original, raw, matching, expected_samples=3)


def test_normalized_only_source_match_never_grants_title():
    original, raw, matching = fixture()
    original[1]["messages"][0]["content"] = original[1]["messages"][0]["content"].replace("Severe", "Ｓｅｖｅｒｅ")
    matching["matches"][0].update(exact_source_ids=[])
    matching["matches"][0].pop("input_only_anchor_sample_index")
    matching["counts"] = {"exact:0": 2, "normalized:1": 1, "normalized:0": 1}
    matching["historical_anchor_counts"]["unique_exact_match_and_historical_anchor"] = 0
    cases, _, audit = h.compile_records(original, raw, matching, expected_samples=3)
    assert "title_context" not in cases[1]["metadata"]
    assert audit["title_context_coverage"]["free_text_without_title"] == 2


def release_fixture(tmp_path, monkeypatch):
    paths, expected = [], {}
    for key, value in zip(("gold", "raw_sources", "source_match"), fixture()):
        path = tmp_path / (key + ".json")
        path.write_text(canonical(value))
        paths.append(path)
        expected[key] = sha(path.read_bytes())
    monkeypatch.setattr(h, "AUDITED_INPUT_SHA256", expected)
    out = tmp_path / "release"
    h.build_release(*paths, out, expected_samples=3)
    return out


def test_release_rebuilds_identities_and_gold_even_if_manifest_hash_rewritten(tmp_path, monkeypatch):
    out = release_fixture(tmp_path, monkeypatch)
    cases, report = h.load_release(out, expected_samples=3)
    assert report["blind_test"] is False and report["training_allowed"] is False
    cases[2]["gold"]["relations"] = []
    name = "evaluation/historical_records.jsonl"
    (out / name).write_text("".join(canonical(c) + "\n" for c in cases))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["output_sha256"][name] = sha((out / name).read_bytes())
    (out / "manifest.json").write_text(canonical(manifest))
    with pytest.raises(ValueError, match="original gold/input/provenance"):
        h.load_release(out, expected_samples=3)


def test_nonblind_nontraining_contract_cannot_be_relabelled(tmp_path, monkeypatch):
    out = release_fixture(tmp_path, monkeypatch)
    manifest = json.loads((out / "manifest.json").read_text())
    for key in ("training_allowed", "blind_test"):
        changed = deepcopy(manifest)
        changed[key] = True
        (out / "manifest.json").write_text(canonical(changed))
        with pytest.raises(ValueError, match="nonblind"):
            h.load_release(out, expected_samples=3)


class EmptyGenerator:
    def generate_many(self, prompts, stage, **kwargs):
        return [{"completion_text": "", "trace": {"complete": False, "eos_observed": False,
            "budget_exhausted": False, "error": "test_transport_failure"}} for _ in prompts]


def test_failed_generation_has_both_coverage_strata_and_all_original_gold_fn():
    cases, _, _ = compile_fixture()
    metrics, predictions, _ = e.evaluate_cases(cases, EmptyGenerator(), modes=("live", "oracle"))
    assert metrics["evaluation_set"] == "historical_regression" and metrics["blind_test"] is False
    for mode in ("live", "oracle"):
        report = metrics["modes"][mode]
        assert report["strict_raw"]["free_text"]["relation"]["fn"] == 2
        for name in ("free_text_with_verified_title", "free_text_without_title"):
            stratum = report["historical_context_strata"][name]
            assert stratum["samples"] == 1 and stratum["strict_raw"]["free_text"]["relation"]["fn"] == 1
        assert [p["input_identity"]["sample_index"] for p in predictions[mode]["raw"]] == [0, 1, 2]


def test_oracle_equivalent_ids_merge_only_prompt_table_keep_directed_gold_edges():
    originals, raw, matching = fixture()
    graph = json.loads(originals[1]["messages"][1]["content"])
    graph["entities"].append({"id": "d-alias", "label": "Disease", "name": "CONDITION  ALPHA"})
    graph["relations"].extend([
        {"source": "s", "target": "d-alias", "relation": "is_core_symptom_of", "evidence": TEXT},
        {"source": "d-alias", "target": "s", "relation": "is_core_symptom_of", "evidence": TEXT}])
    originals[1]["messages"][1]["content"] = json.dumps(graph)
    cases, _, _ = h.compile_records(originals, raw, matching, expected_samples=3)
    assert len(e._oracle_entities(cases[1])) == 2
    assert len(cases[1]["gold"]["entities"]) == 3 and len(cases[1]["gold"]["relations"]) == 3
    assert len(e._relation_set(cases[1]["gold"])) == 2  # Duplicate edge collapses, reverse stays.
    before = deepcopy(cases)
    metrics, _, results = e.evaluate_cases(cases, EmptyGenerator(), modes=("oracle",))
    assert results["oracle"][1]["trace"]["oracle_equivalent_gold_ids_merged"] == 1
    assert metrics["modes"]["oracle"]["strict_raw"]["free_text"]["relation"]["fn"] == 3
    assert cases == before


def test_real_148_release_and_empty_predictions_preserve_all_gold():
    path = Path(__file__).resolve().parents[3] / "audit_work/full_semantic_20261019/historical_v1"
    if not path.is_dir(): pytest.skip("Private historical release absent")
    cases, receipt = e.load_release(path, evaluation_set="historical_regression")
    assert receipt["denominators"] == {
        "overall": {"samples": 148, "gold_entities": 979, "gold_relations": 704},
        "free_text": {"samples": 81, "gold_entities": 829, "gold_relations": 638},
        "structured_icd": {"samples": 67, "gold_entities": 150, "gold_relations": 66}}
    metrics, predictions, _ = e.evaluate_cases(cases, EmptyGenerator(), modes=("live", "oracle"))
    for mode in ("live", "oracle"):
        report = metrics["modes"][mode]
        assert len(predictions[mode]["raw"]) == 148
        assert report["strict_raw"]["free_text"]["relation"]["fn"] == 638
        strata = report["historical_context_strata"]
        assert strata["free_text_with_verified_title"]["samples"] == 57
        assert strata["free_text_without_title"]["samples"] == 24
        assert sum(s["strict_raw"]["free_text"]["relation"]["fn"] for s in strata.values()) == 638


def test_cli_historical_requires_full_148_and_dispatches_explicit_population(tmp_path, monkeypatch):
    argv = ["--release", "release", "--base-model", "base", "--entity-adapter", "entity", "--relation-adapter", "relation",
            "--output-dir", str(tmp_path / "out"), "--evaluation-set", "historical_regression", "--preflight-only"]
    with pytest.raises(SystemExit): e.main(argv + ["--expected-samples", "24"])
    calls = []
    def loader(path, **kwargs):
        calls.append(kwargs)
        return compile_fixture()[0], {"manifest_sha256": "data", "short_chars": 2400}
    monkeypatch.setattr(e, "load_release", loader)
    monkeypatch.setattr(e, "check_adapter_contract", lambda path, stage, **kwargs: {
        "stage": stage, "variant": "full_source", "adapter_files_sha256": "a", "base_input_sha256": "b", "data_sha256": "c"})
    result = e.main(argv)
    assert calls == [{"expected_samples": 148, "evaluation_set": "historical_regression"}]
    assert result["evaluation_set"] == "historical_regression" and result["training_allowed"] is False

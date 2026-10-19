"""Small synthetic end-to-end release, immutability and split checks."""

import json
from pathlib import Path

import pytest

from data_repair.release_v2 import LEGACY_VERSION, VERSION, canonical, prepare_record, run, sha
from data_repair.supervision_v2 import LEGACY_EVIDENCE_POLICY, UNIQUE_EVIDENCE_POLICY
from data_repair.verify_v2 import verify
from kg_agent.contracts import ONTOLOGY


def message(user, graph):
    return {"messages": [{"role": "system", "content": ONTOLOGY["version"]},
                         {"role": "user", "content": "Medical text:\n" + user},
                         {"role": "assistant", "content": "<output>" + canonical(graph) + "</output>"}]}


def disease(index):
    title = "Condition " + index
    node = {"id": "D", "name": title, "label": "Disease"}
    return message(canonical({"code": "XY" + index, "title": title}), {"entities": [node], "relations": []})


def free(index):
    text = f"Feature {index} is a core symptom of Condition {index} during pregnullcy."
    return message(text, {"entities": [{"id": "S", "name": "Feature " + index, "label": "Symptom"},
                                     {"id": "D", "name": "Condition " + index, "label": "Disease",
                                      "properties": {"Definition": "Mentioned during pregnullcy.", "optional": None}}],
                          "relations": [{"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": text}]})


def fixture(tmp_path):
    records = [disease("A"), free("A"), disease("B"), free("B")]
    source = tmp_path / "source.json"
    source.write_text(canonical(records))
    frozen = tmp_path / "split.json"
    frozen.write_text(canonical({"source_train_sha256": sha(source.read_bytes()), "train_groups": [0], "tune_groups": [1]}))
    return source, frozen


def test_full_release_conserves_records_split_and_supervision(tmp_path):
    source, split = fixture(tmp_path)
    old = source.read_bytes()
    out = tmp_path / "release"
    result = run(source, split, out, sha(old))
    assert source.read_bytes() == old
    assert result["records_conserved"] and result["input_unchanged"]
    assert result["summary"]["records"] == 4
    for name in ("short", "source_order"):
        assert all(v["added_vs_base"] == v["missing_vs_base"] == 0 for v in result["supervision_comparison"][name].values())
    assert result["gates"]["eligible_for_existing_rl_objective"] is False
    checked = verify(out)
    assert checked["passed"] and checked["frozen_source_groups_disjoint"]
    assert checked["counts"]["repaired_graph_known_bad_words"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        run(source, split, out, sha(old))


def test_wrong_source_hash_refused_before_any_output(tmp_path):
    source, split = fixture(tmp_path)
    out = tmp_path / "not-created"
    with pytest.raises(ValueError, match="SHA256"):
        run(source, split, out, "wrong")
    assert not out.exists()


def test_graph_prose_is_repaired_but_null_and_original_preserved():
    record = free("A")
    before = canonical(record)
    item = prepare_record(record, 1, 0, "train")
    assert canonical(record) == before
    properties = item["repaired_graph"]["entities"][1]["properties"]
    assert properties["Definition"] == "Mentioned during pregnancy."
    assert properties["optional"] is None
    assert item["source"]["original_text"].endswith("pregnullcy.")
    assert item["source"]["text"].endswith("pregnancy.")


def test_output_corruption_fails_verifier(tmp_path):
    source, split = fixture(tmp_path)
    out = tmp_path / "release"
    run(source, split, out, sha(source.read_bytes()))
    with (out / "base/train.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Release file changed"):
        verify(out)


def test_new_release_reviews_ambiguous_citations_with_explicit_legacy_replay(tmp_path):
    from test_partial_supervision_v2 import ambiguous_example
    _, graph, doc = ambiguous_example()
    source, split = fixture(tmp_path)
    records = json.loads(source.read_text())
    records[1] = message(doc.raw, graph)
    source.write_text(canonical(records))
    old_source = source.read_bytes()
    frozen = json.loads(split.read_text())
    frozen["source_train_sha256"] = sha(old_source)
    split.write_text(canonical(frozen))
    strict_dir, legacy_dir = tmp_path / "unique", tmp_path / "legacy"
    strict = run(source, split, strict_dir, sha(old_source))
    legacy = run(source, split, legacy_dir, sha(old_source), evidence_policy=LEGACY_EVIDENCE_POLICY)
    assert source.read_bytes() == old_source
    assert strict["version"] == VERSION and legacy["version"] == LEGACY_VERSION
    assert strict["evidence_grounding_policy"] == UNIQUE_EVIDENCE_POLICY
    assert "evidence_grounding_policy" not in legacy
    assert strict["summary"]["ambiguous_evidence_annotations"] == 1
    reviews = [json.loads(line) for line in (strict_dir / "review_queue.jsonl").read_text().splitlines()]
    assert any(r.get("reason") == "evidence_occurrence_ambiguous" and r["not_a_negative"] for r in reviews)
    strict_rows = [json.loads(line) for line in (strict_dir / "base/train.jsonl").read_text().splitlines()]
    legacy_rows = [json.loads(line) for line in (legacy_dir / "base/train.jsonl").read_text().splitlines()]
    assert sum(v == "SUPPORTED_SILVER" for r in strict_rows for v in r["pair_supervision"].values()) == 1
    assert sum(v == "SUPPORTED_SILVER" for r in legacy_rows for v in r["pair_supervision"].values()) == 2
    assert all(r["data_version"] == VERSION for r in strict_rows)
    assert all(r["data_version"] == LEGACY_VERSION for r in legacy_rows)
    assert verify(strict_dir)["passed"] and verify(legacy_dir)["passed"]

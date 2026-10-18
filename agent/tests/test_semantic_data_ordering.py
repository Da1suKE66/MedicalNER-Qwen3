"""Frozen v2 ordering must use semantic names/positions, never opaque IDs."""
from copy import deepcopy
import json

from semantic.data import compile_source, sha
from test_semantic_data import prepared


def source():
    item = prepared("Omega and Alpha have beta.")
    item["entities"].append({"id": "z9", "label": "Disease", "canonical_name": "Omega",
        "mentions": [{"original_start": 0, "original_end": 5, "original_text": "Omega"}]})
    extra = deepcopy(item["annotations"][0])
    extra["index"] = 1
    extra["annotation"].update(target="z9", relation="is_core_symptom_of")
    item["annotations"].append(extra)
    graph = {"entities": [{"id": e["id"], "label": e["label"], "name": e["canonical_name"]} for e in item["entities"]],
             "relations": [a["annotation"] for a in item["annotations"]]}
    item["original_record"]["messages"][1]["content"] = json.dumps(graph)
    item["original_record_sha256"] = sha(item["original_record"])
    return item


def relation_row(item, variant):
    return next(r for r in compile_source(item, variant=variant)[0] if r["stage"] == "relations")


def test_canonical_prompt_completion_invariant_under_opaque_id_permutation():
    original, renamed = source(), source()
    mapping = {"d": "zz_last", "s": "middle", "z9": "aa_first"}
    for entity in renamed["entities"]:
        entity["id"] = mapping[entity["id"]]
    for annotation in renamed["annotations"]:
        edge = annotation["annotation"]
        edge["source"], edge["target"] = mapping[edge["source"]], mapping[edge["target"]]
    # IDs are internal; neither inputs shown to the LM nor semantic targets
    # change when names/types/source positions and annotations are unchanged.
    before, after = relation_row(original, "short_canonical"), relation_row(renamed, "short_canonical")
    assert before["prompt"] == after["prompt"]
    assert before["completion"] == after["completion"]
    fields = [line.split("\t") for line in before["completion"].splitlines()]
    assert [row[4] for row in fields] == ["Alpha", "Omega"]


def test_source_relation_targets_follow_endpoint_positions_then_relation_evidence():
    row = relation_row(source(), "full_source")
    fields = [line.split("\t") for line in row["completion"].splitlines()]
    assert [value[4] for value in fields] == ["Omega", "Alpha"]
    assert row["target"][0]["relation"] == "is_core_symptom_of"

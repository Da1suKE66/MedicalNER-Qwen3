from collections import Counter
from copy import deepcopy
import json

import pytest

from data_repair.abstention_repair import make_case
from data_repair.rapid_variants import (
    atomic_counter, build, input_pair_order, make_variant, merge_sources,
    project_row, select_rows,
)
from data_repair.supervision_v2 import KNOWN, serialize_partial, validate_partial_row
from generative.protocol import dumps
from robustness.entity_noise import prompt_payload, protocol_prefix
from data_repair.release_v2 import sha


def fixture(serial=0):
    def anchor(source, target, suffix):
        return dict(source=source, target=target, relation="is_core_symptom_of", group_id=1,
                    sample_id=10, parent_id=f"parent-{suffix}", parent_sha256="a" * 64,
                    evidence_span_ids=["SENT_001"])
    return make_case(anchor("Feature Alpha", "Condition Alpha", 1),
                     anchor("Feature Beta", "Condition Beta", 2), serial=serial)


def plain(row):
    result = deepcopy(row)
    result.pop("abstention_provenance")
    result["id"] = "plain:" + result["id"]
    return result


def test_source_order_is_input_only_and_absent_entities_sort_last():
    row = fixture()
    order = [(p["source"], p["target"]) for p in input_pair_order(row, "source")]
    changed = deepcopy(row)
    changed["target"]["decisions"].reverse()
    changed["pair_supervision"] = {p["pair_id"]: "UNKNOWN_TEST" for p in row["pairs"]}
    changed["abstention_provenance"] = {"secret_gold": "must not be read"}
    assert [(p["source"], p["target"]) for p in input_pair_order(changed, "source")] == order
    entity = changed["entities"][0]
    entity["name"] = "Certainly absent fictional entity"
    ordered = input_pair_order(changed, "source")
    assert ordered[-1]["source"] == entity["id"]
    with pytest.raises(ValueError, match="Unsupported"):
        input_pair_order(row, "gold")


@pytest.mark.parametrize("order", ["hash", "source"])
@pytest.mark.parametrize("size", [1, 4, 16])
def test_chunking_preserves_all_effective_atoms_and_accounts_for_unknown(order, size):
    parents = [fixture()]
    before = deepcopy(parents)
    rows, skipped = make_variant(parents, variant="test", order=order, pairs_per_batch=size)
    assert parents == before
    assert atomic_counter(rows, known_only=True) == atomic_counter(parents, known_only=True)
    assert atomic_counter(rows) + atomic_counter(skipped) == atomic_counter(parents)
    for row in rows:
        assert len(row["pairs"]) <= size
        assert row["spans"] == parents[0]["spans"]
        assert row["original_spans"] == parents[0]["original_spans"]
        assert validate_partial_row(row)
        assert row["sample_weight"] == 1
        assert "rapid_variant_provenance" not in row["prompt"]
        for d in row["target"]["decisions"]:
            origin = row["rapid_variant_provenance"]["pair_sources"][d["pair_id"]]
            assert origin["parent_row_id"] == parents[0]["id"]
            assert origin["parent_abstention_provenance"] == parents[0]["abstention_provenance"]
    assert all(all(s not in KNOWN for s in row["pair_supervision"].values()) for row in skipped)
    if size == 1:
        assert len(skipped) == 4
    assert atomic_counter(rows + rows, known_only=True) != atomic_counter(parents, known_only=True)


def test_global_merge_restores_source_order_across_parent_boundaries():
    source = plain(fixture())
    parents = []
    # All chunks contain known decisions in this artificial graph.
    for index in range(0, 12, 4):
        row = project_row(source, source["pairs"][index:index + 4], variant="test", chunk_index=index)
        row.pop("rapid_variant_provenance")
        row["id"] = f"source:{index}"
        parents.append(row)
    merged = merge_sources(parents)
    assert len(merged) == 1 and len(merged[0]["pairs"]) == 12
    children, skipped = make_variant(parents, variant="source", order="source", pairs_per_batch=4)
    assert [(p["source"], p["target"]) for row in children + skipped for p in row["pairs"]] == [
        (p["source"], p["target"]) for p in input_pair_order(merged[0], "source")]
    assert atomic_counter(children, known_only=True) == atomic_counter(parents, known_only=True)


def test_multilabel_and_unknown_reason_are_preserved_and_mutations_detected():
    source = plain(fixture())
    d = next(d for d in source["target"]["decisions"] if d["relations"])
    pair = next(p for p in source["pairs"] if p["pair_id"] == d["pair_id"])
    second = next(label for label in pair["allowed"] if label != d["relations"][0]["relation"])
    d["relations"].append({"relation": second, "evidence_span_ids": deepcopy(d["relations"][0]["evidence_span_ids"])})
    source["completion"], source["loss_mask_spans"] = serialize_partial(source["target"], source["pair_supervision"])
    rows, _ = make_variant([source], variant="short", order="source", pairs_per_batch=4)
    assert atomic_counter(rows, known_only=True) == atomic_counter([source], known_only=True)
    assert max(len(d["relations"]) for row in rows for d in row["target"]["decisions"]) == 2
    broken = deepcopy(rows)
    target = next(d for row in broken for d in row["target"]["decisions"] if len(d["relations"]) == 2)
    target["relations"].pop()
    assert atomic_counter(broken, known_only=True) != atomic_counter([source], known_only=True)


def test_selection_uses_metadata_strata_and_is_deterministic():
    rows = [plain(fixture(i)) for i in range(5)] + [fixture(i) for i in range(5)]
    chosen = select_rows(rows, base_rows=2, synthetic_rows=3)
    assert len(chosen) == 5
    assert sum("abstention_provenance" in row for row in chosen) == 3
    shuffled = list(reversed(rows))
    assert {row["id"] for row in select_rows(shuffled, base_rows=2, synthetic_rows=3)} == {row["id"] for row in chosen}
    changed = deepcopy(rows)
    for row in changed:
        row["target"] = "test outcome must not affect selection"
    assert [row["id"] for row in select_rows(changed, base_rows=2, synthetic_rows=3)] == [row["id"] for row in chosen]
    with pytest.raises(ValueError, match="Insufficient"):
        select_rows(rows, base_rows=6, synthetic_rows=1)


def test_conflicting_entities_duplicate_candidates_and_tune_are_rejected():
    row = plain(fixture())
    duplicate = deepcopy(row)
    duplicate["id"] += ":other"
    with pytest.raises(ValueError, match="Duplicated"):
        merge_sources([row, duplicate])
    duplicate["entities"][0]["name"] += " changed"
    duplicate["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(duplicate)) + "\n"
    with pytest.raises(ValueError, match="Conflicting"):
        merge_sources([row, duplicate])
    duplicate["split"] = "tune"
    with pytest.raises(ValueError, match="training"):
        merge_sources([duplicate])


def test_build_is_immutable_deterministic_private_and_aggregate_only(tmp_path):
    rows = [plain(fixture()), fixture(1)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(dumps(row) + "\n" for row in rows))
    before = source.read_bytes()
    outputs = [tmp_path / "first", tmp_path / "second"]
    reports = [build(source, output, expected_sha256=sha(before), base_rows=1, synthetic_rows=1) for output in outputs]
    assert reports[0] == reports[1]
    assert source.read_bytes() == before
    report = reports[0]
    assert report["source_rows"] == 2 and report["source_rows_before_selection"] == 2
    assert not report["tokenizer_checked"]
    assert "Feature Alpha" not in json.dumps(report)
    assert str(tmp_path) not in json.dumps(report)
    assert len({a["effective_atomic_counter_sha256"] for a in report["audits"].values()}) == 1
    for name, digest in report["output_sha256"].items():
        assert sha((outputs[0] / name).read_bytes()) == digest
    with pytest.raises(ValueError, match="overwrite"):
        build(source, outputs[0], expected_sha256=sha(before), base_rows=1, synthetic_rows=1)
    with pytest.raises(ValueError, match="SHA256"):
        build(source, tmp_path / "bad", expected_sha256="bad", base_rows=1, synthetic_rows=1)
    assert not (tmp_path / "bad").exists()

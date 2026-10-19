"""Resident reconstruction tests use only invented source strings."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from data_repair import text_normalization as matcher
from semantic import protocol as p
from semantic import relation_windows as shared
from semantic.data import canonical, compile_source, compile_eval_record, sha
from semantic.relation_windows_resident import build_release, load_receipt, reconstruct_anchors
from test_semantic_data import FakeTokenizer, prepared
from test_semantic_relation_windows import fixture


def anchor_receipt(item, full):
    ledger = []
    for atom, target in zip(full["metadata"]["supervised_atom_ids"], full["target"]):
        annotation = item["annotations"][int(atom.split(":")[1])]
        hit = min(annotation["evidence_fragments"][int(atom.split(":")[-1])],
                  key=lambda h: (h["original_start"], h["original_end"]))
        ledger.append({"source_record_sha256": full["metadata"]["source_record_sha256"], "atom_id": atom,
                       "evidence_sha256": sha(target["evidence"]), "start": hit["original_start"], "end": hit["original_end"]})
    ledger.sort(key=lambda r: (r["source_record_sha256"], r["atom_id"]))
    return {"receipt_version": "resident-chosen-anchor-equivalence-v1",
        "matcher_implementation_sha256": sha(Path(matcher.__file__).read_bytes()),
        "expected_unique_atom_count": len(ledger), "all_original_chosen_spans_equal": True,
        "prepared_chosen_anchor_ledger_sha256": sha(ledger), "resident_chosen_anchor_ledger_sha256": sha(ledger),
        "prior_prepared_equivalence_audit_sha256": "a" * 64, "prior_prepared_records_sha256": "b" * 64}


def build_fixture(tmp_path):
    prefix = "Alpha has beta. "
    text = prefix + "x" * (2099 - len(prefix)) + " clue. " + "z" * 3000 + " clue."
    item = prepared(text)
    positions = [text.index("clue."), text.rindex("clue.")]
    item["annotations"][0]["annotation"]["evidence"] = "clue."
    item["annotations"][0]["evidence_fragments"] = [[
        {"original_start": i, "original_end": i + 5, "original_text": "clue."} for i in positions]]
    full = next(r for r in compile_source(item)[0] if r["stage"] == "relations")
    base = tmp_path / "resident_base"
    contents = {"full_source/train_relations.jsonl": [full],
                "evaluation/train_records.jsonl": [compile_eval_record(item)]}
    for variant in shared.VARIANTS:
        contents[f"{variant}/train_relations.jsonl"] = [
            r for r in compile_source(item, variant=variant)[0] if r["stage"] == "relations"]
    for name, rows in contents.items():
        path = base / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(canonical(row) + "\n" for row in rows))
    (base / "title_source_receipts.json").write_text("{}\n")
    manifest = {"output_sha256": {name: sha((base / name).read_bytes()) for name in
        [*contents, "title_source_receipts.json"]}, "protocol_sha256": sha(Path(p.__file__).read_bytes()),
        "train_tune_group_overlap": 0}
    (base / "manifest.json").write_text(canonical(manifest) + "\n")
    receipt = {**anchor_receipt(item, full), "base_release_manifest_sha256": sha((base / "manifest.json").read_bytes()),
        "full_relation_file_sha256": sha((base / "full_source/train_relations.jsonl").read_bytes()),
        "expected_overlap_counts": {"missing_frozen_chosen_anchor_overlap": 1,
            "projected_chosen_anchor_fix_exposures_including_unchanged_fallback": 2,
            "projected_chosen_anchor_fix_rows_including_unchanged_fallback": 2,
            "missing_other_evidence_occurrence_scope_unverified": 1}}
    receipt_path = tmp_path / "aggregate.json"
    receipt_path.write_text(canonical(receipt) + "\n")
    return base, receipt_path, receipt


def test_ledger_recovers_prepared_first_hit_without_reading_prepared_file():
    item, full, _ = fixture()
    receipt = anchor_receipt(item, full)
    anchors, ledger = reconstruct_anchors([full], receipt)
    anchor = next(iter(anchors[full["metadata"]["source_record_sha256"]].values()))
    assert anchor["start"] == 52 and anchor["resident_evidence_hit_count"] == 2
    assert "prepared_evidence_hit_count" not in anchor
    assert sha(ledger) == receipt["prepared_chosen_anchor_ledger_sha256"]


def test_ledger_sha_rejects_tampering_even_when_atom_count_matches():
    item, full, _ = fixture()
    receipt = anchor_receipt(item, full)
    receipt["resident_chosen_anchor_ledger_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ledger SHA mismatch"):
        reconstruct_anchors([full], receipt)


def test_matcher_implementation_drift_rejected():
    item, full, _ = fixture()
    receipt = anchor_receipt(item, full)
    receipt["matcher_implementation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="matcher implementation drift"):
        reconstruct_anchors([full], receipt)


def test_duplicate_source_or_heldout_rows_rejected():
    item, full, _ = fixture()
    receipt = anchor_receipt(item, full)
    with pytest.raises(ValueError, match="duplicate source"):
        reconstruct_anchors([full, deepcopy(full)], receipt)
    bad = deepcopy(full); bad["split"] = "tune"
    with pytest.raises(ValueError, match="Held-out"):
        reconstruct_anchors([bad], receipt)


def test_receipt_bytes_are_pinned_before_any_data_admission(tmp_path):
    _, path, _ = build_fixture(tmp_path)
    expected = sha(path.read_bytes())
    assert load_receipt(path, expected)["all_original_chosen_spans_equal"] is True
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="receipt SHA mismatch"):
        load_receipt(path, expected)


def test_build_is_resident_only_with_paired_tokens_and_no_fabricated_prepared_provenance(tmp_path, monkeypatch):
    base, path, receipt = build_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("prepared-source path must never be used")
    monkeypatch.setattr(shared, "chosen_anchors", forbidden)
    original_read_bytes, original_read_text = Path.read_bytes, Path.read_text
    def guard_bytes(path_, *args, **kwargs):
        assert path_.name not in ("records.jsonl", "frozen_split.jsonl")
        return original_read_bytes(path_, *args, **kwargs)
    def guard_text(path_, *args, **kwargs):
        assert path_.name not in ("records.jsonl", "frozen_split.jsonl")
        return original_read_text(path_, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", guard_bytes)
    monkeypatch.setattr(Path, "read_text", guard_text)
    output = tmp_path / "new_release"
    summary = build_release(base, path, output, FakeTokenizer(), receipt_sha256=sha(path.read_bytes()), max_length=50000)
    assert summary["release_version"] == "relation_complete_overlap_v2"
    assert summary["original_unique_source_atom_count"] == 1
    assert summary["annotation_exposures_per_epoch"] == 2
    assert summary["resident_reconstruction"]["external_prepared_records_read"] is False
    assert "prepared_source_sha256" not in summary
    assert summary["variants"]["short_source"]["token_counts"] == summary["variants"]["short_canonical"]["token_counts"]
    for variant in shared.VARIANTS:
        rows = shared._read(output / variant / "train_relations.jsonl")
        assert len(rows) == 2
        for row in rows:
            policy = row["metadata"]["relation_window_supervision"]
            assert policy["anchor_reconstruction"] == "resident_full_source_minimum_match"
            assert policy["external_prepared_records_read"] is False
            assert "prepared_record_sha256" not in policy
    with pytest.raises(ValueError, match="overwrite"):
        build_release(base, path, output, FakeTokenizer(), receipt_sha256=sha(path.read_bytes()))


@pytest.mark.parametrize("field,match", [("base_release_manifest_sha256", "base manifest"),
                                        ("full_relation_file_sha256", "full relation file")])
def test_independent_base_and_full_file_receipt_gates(tmp_path, field, match):
    base, path, receipt = build_fixture(tmp_path)
    receipt[field] = "0" * 64
    path.write_text(canonical(receipt) + "\n")
    with pytest.raises(ValueError, match=match):
        build_release(base, path, tmp_path / "new_release", FakeTokenizer(), receipt_sha256=sha(path.read_bytes()))

"""Mechanical synthetic-contract tests; no assertion of clinical effectiveness."""
from copy import deepcopy
import json

import pytest

from generative.recovery_preferences import (
    CONTRACT, KINDS, STRUCTURAL, as_sft, build, make_case, summarize, verify_case,
)
from generative.protocol import digest
from generative.train import validate_training_rows


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("size", [2, 4, 8])
def test_every_error_family_has_recipe_verified_preference_at_every_size(kind, size):
    for template in range(6):
        row = make_case(template, kind, pairs_per_batch=size, template=template)
        detail = verify_case(row)
        assert detail["reward_margin"] > 0
        assert detail["pairs_per_batch"] == size
        assert bool(detail["expected_rejected_errors"]) == (kind in STRUCTURAL)
        assert row["provenance"] == "fully_artificial_exhaustive_assertions_no_medical_source"
        assert row["human_adjudicated"] is False
        assert row["medical_source_records"] == 0
        assert not row["prompt"].startswith("<|im_start|>")
        assert "construction_recipe" not in row["prompt"]


@pytest.mark.parametrize("field,value", [
    ("oracle_source_text", "actual clinical source inserted"),
    ("prompt", "changed input"), ("chosen", "{}"), ("rejected", "{}"),
    ("human_adjudicated", True), ("medical_source_records", 1),
    ("provenance", "clinical_gold"), ("eligible_for_rl", False),
    ("semantic_claim", "medical_correctness"),
])
def test_tampering_even_with_resigned_visible_hash_is_rejected(field, value):
    row = make_case(0, "terminal_bracket_reversal")
    row[field] = value
    if field in {"prompt", "chosen", "rejected"}:
        row[field + "_sha256"] = digest(value)
    with pytest.raises(ValueError, match="reconstruction"):
        verify_case(row)


@pytest.mark.parametrize("field,value", [
    ("pair_supervision", {"P0001": "UNKNOWN_UNANNOTATED"}),
    ("loss_mask_spans", []), ("loss_mask_contract", "any"),
    ("clinical_source", "anything"), ("unknown_metadata", True),
])
def test_extra_metadata_cannot_smuggle_partial_or_external_supervision(field, value):
    row = make_case(0, "omit_one_edge")
    row[field] = value
    with pytest.raises(ValueError, match="reconstruction"):
        verify_case(row)
    if field in {"pair_supervision", "loss_mask_spans", "loss_mask_contract"}:
        with pytest.raises(ValueError, match="Partial/UNKNOWN"):
            validate_training_rows([row], "dpo")


def test_source_text_cannot_be_supplied_through_recipe():
    row = make_case(0, "wrong_target")
    row["construction_recipe"]["source"] = "external content"
    with pytest.raises(ValueError, match="external source"):
        verify_case(row)


def test_sft_view_preserves_receipt_and_exact_chosen_completion():
    preference = make_case(2, "fabricated_pair_id", pairs_per_batch=8, template=5)
    row = as_sft(preference)
    assert row["chosen"] == row["completion"]
    assert row["target"] == json.loads(row["chosen"])
    assert row["construction_recipe"] == preference["construction_recipe"]
    assert row["group_id"] == preference["group_id"]
    assert verify_case(row)["reward_margin"] > 0
    row["completion"] = row["rejected"]
    with pytest.raises(ValueError, match="reconstruction"):
        verify_case(row)


def test_mechanical_holdout_is_disjoint_and_cannot_be_admitted_as_training():
    a = make_case(0, "omit_one_edge")
    b = make_case(0, "omit_one_edge", partition="mechanical_holdout")
    assert a["group_id"] != b["group_id"] and a["prompt_sha256"] != b["prompt_sha256"]
    assert verify_case(b)["reward_margin"] > 0
    with pytest.raises(ValueError, match="train-tagged"):
        validate_training_rows([b], "dpo")


def test_release_has_exact_balanced_budget_and_immutable_hashes(tmp_path):
    one, two = build(tmp_path / "one"), build(tmp_path / "two")
    assert one == two
    audit = one["audits"]["train"]
    assert audit["rows"] == 256
    assert audit["edge_count_direction"] == {"chosen_more": 80, "chosen_fewer": 80, "equal": 96}
    assert audit["counts"]["structural_rejected_rows"] == 64
    assert audit["counts"]["all_negative_chosen"] == 40
    assert audit["counts"]["multilabel_rows"] > 0
    assert set(audit["candidate_sizes"]) == {2, 4, 8}
    for name, metadata in one["files"].items():
        import hashlib
        assert hashlib.sha256((tmp_path / "one" / name).read_bytes()).hexdigest() == metadata["sha256"]
    with pytest.raises(ValueError, match="overwrite"):
        build(tmp_path / "one")


def test_target_type_error_is_mechanically_false_not_global_ontology_redefinition():
    row = make_case(1, "wrong_target_type", pairs_per_batch=2)
    detail = verify_case(row)
    assert not detail["expected_rejected_errors"]
    assert detail["chosen_edges"] == detail["rejected_edges"]
    assert detail["reward_margin"] > 0


def test_bracket_error_is_exact_observed_terminal_reversal():
    row = make_case(1, "terminal_bracket_reversal")
    assert row["chosen"].endswith("]}")
    assert row["rejected"] == row["chosen"][:-2] + "}]"
    assert verify_case(row)["expected_rejected_errors"] == ["invalid_json"]

"""Formal runs may not silently become sampled, leaky, or partial-epoch runs."""
from argparse import Namespace
from collections import Counter
import json

import pytest

from semantic.data import compile_source
from semantic.train import exposure_summary, read_training_rows, validate_arguments
from test_semantic_data import prepared


def args(**overrides):
    values = dict(epochs=2, batch_size=1, gradient_accumulation=8, max_steps=-1,
                  smoke_only=False, learning_rate=5e-5, lora_rank=16, max_length=8192,
                  dropout=.05, data_sha256="a" * 64)
    return Namespace(**(values | overrides))


def test_step_cap_is_never_a_formal_full_data_run():
    with pytest.raises(ValueError, match="smoke-only"):
        validate_arguments(args(max_steps=64))
    validate_arguments(args(max_steps=2, smoke_only=True))


def test_fractional_formal_epoch_refused():
    with pytest.raises(ValueError, match="whole number"):
        validate_arguments(args(epochs=.5))


def test_formal_run_requires_audited_release_digest():
    with pytest.raises(ValueError, match="audited release"):
        validate_arguments(args(data_sha256=None))


def test_training_reader_rejects_tune_rows(tmp_path):
    rows, _ = compile_source(prepared(split="tune"))
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="Held-out"):
        read_training_rows(path, "entities")


def test_exposure_counts_actual_visits_not_expected_epochs():
    rows = [{"group_id": "a"}, {"group_id": "a"}, {"group_id": "b"}]
    measured = exposure_summary(rows, Counter({0: 2, 1: 2}))
    assert measured["unique_rows_seen"] == 2
    assert measured["row_visits"] == 4
    assert measured["min_visits"] == 0
    assert not measured["all_rows_seen"]
    assert measured["unique_source_groups_seen"] == 1

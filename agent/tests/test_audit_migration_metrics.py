"""Independent audit math and read-only output guarantees, using synthetic data."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit_migration_metrics.py"
SPEC = importlib.util.spec_from_file_location("audit_migration_metrics", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_graph(relations):
    return {"entities": [{"id": "a", "label": "Disease", "name": "Alpha"},
                         {"id": "b", "label": "Disease", "name": "Beta"}],
            "relations": [{"source": "a", "target": "b", "relation": r} for r in relations]}


def record(graph):
    return {"messages": [{"role": "assistant", "content": graph}]}


def test_independent_scorer_uses_unique_multilabel_triples_and_micro_counts():
    gold = make_graph(["subtype_of", "differentiates_from"])
    predicted = make_graph(["subtype_of", "subtype_of", "co_occurs_with"])
    result = audit.score([predicted], [record(gold)])
    assert result == {"tp": 1, "fp": 1, "fn": 1, "precision": .5, "recall": .5, "f1": .5}


def test_independent_scorer_ignores_graph_ids_order_case_and_width():
    gold = make_graph(["subtype_of"])
    predicted = {"entities": [{"id": "new-b", "label": "Disease", "name": "BETA"},
                               {"id": "new-a", "label": "Disease", "name": " Ａｌｐｈａ "}],
                 "relations": [{"source": "new-a", "target": "new-b", "relation": "subtype_of"}]}
    assert audit.score([predicted], [record(gold)])["f1"] == 1
    predicted["relations"][0].update(source="new-b", target="new-a")
    assert audit.score([predicted], [record(gold)])["f1"] == 0


def test_independent_scorer_does_not_reward_empty_rows_or_drop_invalid_edges():
    gold = make_graph([])
    assert audit.score([gold], [record(gold)])["f1"] == 0
    predicted = make_graph([])
    predicted["relations"].append("malformed")
    assert audit.score([predicted], [record(gold)])["fp"] == 1
    with pytest.raises(ValueError, match="length mismatch"):
        audit.score([], [record(gold)])


def test_audit_metric_arithmetic_validation_cannot_be_disabled_with_python_optimization():
    with pytest.raises(ValueError, match="mismatch"):
        audit.compare({"f1": .5}, {"f1": .9}, "synthetic")
    with pytest.raises(ValueError, match="nonnegative integers"):
        audit.metric(-1, 0, 0)


def test_audit_refuses_migration_writes_and_existing_output(tmp_path):
    root = tmp_path / "migration"
    root.mkdir()
    with pytest.raises(ValueError, match="read-only"):
        audit.check_output(root, root / "history/metrics.json")
    existing = tmp_path / "existing.json"
    existing.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        audit.check_output(root, existing)
    assert existing.read_text() == "preserve"
    audit.check_output(root, tmp_path / "new.json")

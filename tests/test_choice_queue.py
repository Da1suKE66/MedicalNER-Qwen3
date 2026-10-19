"""A finished choice run must retain every frozen evaluation identity."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_rapid_queue import validate_outputs


def artifacts(tmp_path, *, status="complete", identities=("a", "b"), samples=2):
    report = {"status": status, "samples": samples, "relation": {}, "evidence": {},
              "scope": "artificial_semantic_only"}
    (tmp_path / "metrics.json").write_text(json.dumps(report))
    (tmp_path / "predictions.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in identities))
    return {"kind": "choice_evaluation", "output": str(tmp_path), "expected_samples": 2}


def test_complete_choice_run_preserves_denominator(tmp_path):
    assert validate_outputs(artifacts(tmp_path))["samples"] == 2


@pytest.mark.parametrize("changes", [{"status": "running"}, {"identities": ("a",)}, {"samples": 1}])
def test_incomplete_choice_run_cannot_be_promoted(tmp_path, changes):
    with pytest.raises(ValueError, match="denominator"):
        validate_outputs(artifacts(tmp_path, **changes))


@pytest.mark.parametrize("identities", [("a", "a"), (None, "b")])
def test_duplicate_or_missing_choice_identity_is_rejected(tmp_path, identities):
    with pytest.raises(ValueError, match="identities"):
        validate_outputs(artifacts(tmp_path, identities=identities))

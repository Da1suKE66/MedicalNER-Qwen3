"""Package validation runs against extracted code, never checkout import caches."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from data_repair.abstention_repair import make_case
from data_repair.rapid_variants import build as build_training
from data_repair.release_v2 import sha
from generative.protocol import dumps
from generative.rapid_preferences import build as build_preferences
from generative.rapid_screen import prepare_screen

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_rapid_snapshot", REPO / "scripts/validate_rapid_snapshot.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    code = root / "code"
    code.mkdir(parents=True)
    for directory in ("agent", "src"):
        for source in (REPO / directory).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".json"} and not {"tests", "__pycache__"}.intersection(source.parts):
                target = code / source.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for relative in ("schemas/current.json", "schemas/v2.0.0/schema.json"):
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    def anchor(a, b, n):
        return {"source": a, "target": b, "relation": "is_core_symptom_of", "group_id": 1,
                "sample_id": 10, "parent_id": str(n), "parent_sha256": "a" * 64,
                "evidence_span_ids": ["SENT_001"]}
    row = make_case(anchor("Feature Alpha", "Condition Alpha", 1),
                    anchor("Feature Beta", "Condition Beta", 2), serial=0)
    plain = json.loads(json.dumps(row))
    plain.pop("abstention_provenance")
    plain["id"] = "plain:" + plain["id"]
    source = tmp_path / "source.jsonl"
    source.write_text(dumps(plain) + "\n" + dumps(row) + "\n")
    build_training(source, root / "data", expected_sha256=sha(source.read_bytes()), base_rows=1, synthetic_rows=1)
    records = []
    for i in range(2):
        text = f"Feature {i} is a core symptom of Condition {i}."
        graph = {"entities": [{"id": "S", "name": f"Feature {i}", "label": "Symptom"},
                              {"id": "D", "name": f"Condition {i}", "label": "Disease"}],
                 "relations": [{"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": text}]}
        records.append({"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": dumps(graph)}]})
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(records))
    prepare_screen(gold, root / "data/experiments.json", "V4_control", root / "evaluation", count=2)
    build_preferences(root / "rl", repeats=1, holdout_repeats=1)
    return root


def test_extracted_snapshot_exercises_all_cpu_paths_without_model(snapshot):
    result = module.validate_snapshot(snapshot)
    assert result["status"] == "passed" and not result["model_loaded"]
    assert len(result["medical_conditions"]) == 4
    assert len(result["training_variants"]) == 4
    assert result["dpo_preferences_validated"] == 8
    assert result["mechanical"]["samples"] == 8
    assert all(r["gold_denominator_in_empty_prediction_rehearsal"] == 2 for r in result["medical_conditions"].values())


def test_missing_schema_reproduces_deployment_failure_despite_checkout_schema(snapshot):
    (snapshot / "code/schemas/v2.0.0/schema.json").unlink()
    assert (REPO / "schemas/v2.0.0/schema.json").is_file()
    with pytest.raises(RuntimeError, match="Missing or external snapshot resource.*schema.json"):
        module.validate_snapshot(snapshot)


def test_ontology_is_a_required_packaged_resource(snapshot):
    (snapshot / "code/agent/kg_agent/ontology.json").unlink()
    with pytest.raises(RuntimeError, match="Missing or external snapshot resource.*ontology.json"):
        module.validate_snapshot(snapshot)


def test_external_schema_symlink_cannot_hide_an_incomplete_package(snapshot):
    path = snapshot / "code/schemas/v2.0.0/schema.json"
    path.unlink()
    path.symlink_to(REPO / "schemas/v2.0.0/schema.json")
    with pytest.raises(RuntimeError, match="Missing or external snapshot resource"):
        module.validate_snapshot(snapshot)

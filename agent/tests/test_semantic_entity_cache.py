"""CPU fixed-identity cache replay: duplicate prompts, failures and tampering."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic import entity_cache as cache
from semantic import evaluate as e
from semantic import runtime as r
from semantic import protocol as p
from semantic.data import canonical, sha
from generative.chat import render_prompt


TEXT = "Alpha has beta."
ENTITIES = "Disease\tAlpha\tAlpha\nSymptom\tbeta\tbeta\n"
RELATIONS = "Symptom\tbeta\tis_associated_symptom_of\tDisease\tAlpha\tAlpha has beta.\n"


class Tokenizer:
    eos_token_id = 999999
    pad_token_id = 0
    def encode(self, text, **kwargs): return [ord(c) for c in text]
    def decode(self, ids, **kwargs): return "".join(chr(token) for token in ids)
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"] + "\n</think>\n"


def receipt(variant="full_source"):
    return {"stage": "entities", "variant": variant, "contract": p.CONTRACT,
        "adapter_final_sha256": "a" * 64, "adapter_files_sha256": "b" * 64,
        "base_input_sha256": "c" * 64, "data_sha256": "d" * 64,
        "completed": True, "smoke_only": False, "protocol_sha256": r.protocol_digest()}


class Backend:
    def __init__(self, batch_size=2, fail_first=True, forbid_entities=False, training_variant="full_source", fail_kind="missing_eos"):
        self.batch_size, self.fail_first, self.forbid_entities = batch_size, fail_first, forbid_entities
        self.fail_kind = fail_kind
        self.max_input_tokens, self.max_new_tokens = 8192, 4096
        self.tokenizer, self.model = Tokenizer(), SimpleNamespace(fingerprint="a" * 64)
        self.receipts = {"entities": receipt(training_variant)}
        self.weight_verification = {"entities": {"expected_sha256": "a" * 64, "loaded_sha256": "a" * 64, "cuda_sha256": "a" * 64, "load_attempts": 1}}
        self.entity_calls, self.relation_calls = 0, 0
    def generate_many(self, prompts, stage, **kwargs):
        if stage == "entities": assert not self.forbid_entities, "Cache secretly called entity model"
        outputs = []
        for prompt in prompts:
            fail = stage == "entities" and self.fail_first and self.entity_calls == 0
            if stage == "entities": self.entity_calls += 1
            else: self.relation_calls += 1
            text = ENTITIES if stage == "entities" else RELATIONS
            rendered = render_prompt(self.tokenizer, prompt)
            if fail and self.fail_kind == "input_budget":
                outputs.append(r.SemanticGenerator._failure(self, prompt, stage, "input_budget_exceeded_no_truncation",
                                                           prompt_tokens=len(self.tokenizer.encode(rendered))))
                continue
            if fail and self.fail_kind == "invalid_records": text = "bad\n"
            ids = [ord("x")] * 4096 if fail and self.fail_kind == "missing_eos" else self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            output = r.decode_completion(self.tokenizer, ids, max_new_tokens=4096)
            output["trace"].update(stage=stage, prompt_sha256=sha(prompt), rendered_prompt_sha256=sha(rendered),
                prompt_tokens=len(self.tokenizer.encode(rendered)), max_new_tokens=4096,
                elapsed_seconds=10.0, batch_size=len(prompts), oom_retries=0)
            outputs.append(output)
        return outputs


def population():
    result = []
    for index in range(130):
        free = index < 69
        text = TEXT if free else json.dumps({"title": "Alpha", "code": "QA"})
        gold = {"entities": [{"id": "a", "label": "Disease", "name": "Alpha"}], "relations": []}
        if free:
            gold["entities"].append({"id": "b", "label": "Symptom", "name": "beta"})
            gold["relations"].append({"source": "b", "target": "a", "relation": "is_associated_symptom_of", "evidence": TEXT})
        metadata = {"source_type": "free_text" if free else "structured_icd"}
        original = {"messages": [{"role": "user", "content": text}, {"role": "assistant", "content": json.dumps(gold)}]}
        result.append({"id": f"tune:{index}", "index": index, "split": "tune", "source_type": metadata["source_type"],
            "source_group_id": str(index), "text": text, "metadata": metadata, "original_record": original,
            "gold": gold, "input_identity": {"sha256": sha({"text": text, "metadata": metadata})}})
    return result


def make_origin(tmp_path, monkeypatch, *, entity_variant="full_source", short_chars=2400, fail_kind="missing_eos"):
    monkeypatch.setattr(cache, "adapter_fingerprint", lambda model, stage: model.fingerprint)
    training_variant = "full_source" if entity_variant == "full_source" else "short_source_complete_windows"
    cases, events, backend = population(), [], Backend(training_variant=training_variant, fail_kind=fail_kind)
    if entity_variant == "short_source":
        for case in cases[:69]:
            case["text"] += " " + "Unannotated background context. " * 6
            case["original_record"]["messages"][0]["content"] = case["text"]
            case["input_identity"] = {"sha256": sha({"text": case["text"], "metadata": case["metadata"]})}
    metrics, predictions, _ = e.evaluate_cases(cases, backend, modes=("live",), entity_variant=entity_variant,
        short_chars=short_chars, on_generation=events.append)
    data = {"manifest_sha256": "e" * 64, "short_chars": short_chars, "record_counts": {"tune:free_text": 69, "tune:structured_icd": 61}}
    directory = tmp_path / "origin"
    directory.mkdir()
    for view in ("raw", "filtered"):
        (directory / f"predictions.{view}.jsonl").write_text("".join(canonical(row) + "\n" for row in predictions["live"][view]))
    (directory / "generations.jsonl").write_text("".join(canonical(row) + "\n" for row in events))
    metrics.update(input_release_sha256=data["manifest_sha256"], protocol_sha256=r.protocol_digest(),
        artifacts_sha256={name: sha((directory / name).read_bytes()) for name in ("generations.jsonl", "predictions.raw.jsonl", "predictions.filtered.jsonl")})
    (directory / "metrics.json").write_text(canonical(metrics))
    adapter = receipt(training_variant)
    manifest = {"status": "complete", "complete": True, "completed": True, "failed": False, "samples": 130,
        "metrics_sha256": sha((directory / "metrics.json").read_bytes()), "input_release": data,
        "output_contract": p.CONTRACT, "sample_indices": list(range(130)),
        "runtime_sha256": sha(Path(r.__file__).read_bytes()), "protocol_sha256": r.protocol_digest(),
        "evaluator_sha256": sha(Path(e.__file__).read_bytes()), "entity_variant": entity_variant,
        "entity_strategy": r.entity_strategy_contract(entity_variant, adapter, short_chars=short_chars),
        "loaded_adapter_verification": backend.weight_verification,
        "adapters": {"entities": {key: adapter[key] for key in ("stage", "variant", "adapter_files_sha256", "base_input_sha256", "data_sha256")}},
        "args": {"mode": "live", "entity_variant": entity_variant, "base_model": "base", "max_input_tokens": 8192, "max_new_tokens": 4096, "batch_size": 2}}
    for name in ("manifest.json", "run_manifest.json"): (directory / name).write_text(canonical(manifest))
    return SimpleNamespace(path=directory, cases=cases, data=data, metrics=metrics, predictions=predictions, events=events,
                           entity_variant=entity_variant, entity_receipt=adapter)


@pytest.fixture
def origin(tmp_path, monkeypatch):
    return make_origin(tmp_path, monkeypatch)


def plan(origin, **kwargs):
    return cache.EntityCachePlan.load(origin.path, origin.cases, origin.data, origin.entity_receipt,
        **{"entity_variant": origin.entity_variant, "base_model": "base", "max_input_tokens": 8192, "max_new_tokens": 4096, "mode": "live", **kwargs})


def rewrite(origin, *, transform_manifest=None, transform_rows=None):
    manifest = json.loads((origin.path / "manifest.json").read_text())
    if transform_rows:
        rows = [json.loads(line) for line in (origin.path / "generations.jsonl").read_text().splitlines()]
        transform_rows(rows)
        (origin.path / "generations.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
        metrics = json.loads((origin.path / "metrics.json").read_text())
        metrics["artifacts_sha256"]["generations.jsonl"] = sha((origin.path / "generations.jsonl").read_bytes())
        (origin.path / "metrics.json").write_text(canonical(metrics))
        manifest["metrics_sha256"] = sha((origin.path / "metrics.json").read_bytes())
    if transform_manifest: transform_manifest(manifest)
    for name in ("manifest.json", "run_manifest.json"): (origin.path / name).write_text(canonical(manifest))


def test_full_population_cached_failure_and_duplicate_prompts_match_uncached_metrics_with_batch8(origin):
    backend = Backend(batch_size=8, forbid_entities=True)
    replay = plan(origin).bind(backend)
    metrics, predictions, results = e.evaluate_cases(origin.cases, replay, modes=("live",))
    assert metrics["modes"] == origin.metrics["modes"]
    assert metrics["generation_complete"] is False and metrics["modes"]["live"]["filtered"]["free_text"]["relation"]["fn"] == 1
    for view in ("raw", "filtered"):
        assert [row["output"] for row in predictions["live"][view]] == [row["output"] for row in origin.predictions["live"][view]]
    info = metrics["entity_cache"]
    assert info["all_cached_calls_consumed"] and info["cached_entity_calls"] == 69
    assert info["new_gpu_entity_calls"] == backend.entity_calls == 0
    assert info["new_gpu_relation_calls"] == backend.relation_calls == 68
    assert info["new_gpu_relation_calls_unit"] == "relation_backend_attempted_jobs"
    assert "not GPU decode successes" in info["new_gpu_relation_calls_definition"]
    assert info["origin_entity_batch_size"] == 2 and info["relation_batch_size"] == 8
    assert len({row["prompt"] for row in replay.plan.rows}) == 1  # Same prompt, 69 distinct identities.
    first = results["live"][0]["generations"][0]
    assert first["status"] == "failed" and first["generation_trace"]["budget_exhausted"]
    assert first["generation_trace"]["entity_cache"]["new_gpu_entity_generation"] is False


def test_live_plus_oracle_replays_only_live_entities_and_generates_both_relation_components(origin):
    backend = Backend(batch_size=8, forbid_entities=True)
    replay = plan(origin, mode="both").bind(backend)
    metrics, _, _ = e.evaluate_cases(origin.cases, replay, modes=("live", "oracle"))
    assert metrics["modes"]["live"] == origin.metrics["modes"]["live"]
    assert metrics["modes"]["oracle"]["filtered"]["free_text"]["relation"]["tp"] == 69
    assert metrics["entity_cache"]["cached_entity_calls"] == 69
    assert metrics["entity_cache"]["new_gpu_relation_calls"] == backend.relation_calls == 137


def test_relation_attempt_count_includes_backend_failure_not_success(origin, monkeypatch):
    backend = Backend(batch_size=8, forbid_entities=True)
    replay = plan(origin).bind(backend)
    def fail(prompts, stage, **kwargs):
        assert stage == "relations"
        raise RuntimeError("backend refused before GPU decoding")
    monkeypatch.setattr(backend, "generate_many", fail)
    with pytest.raises(RuntimeError, match="before GPU decoding"):
        replay.generate_many(["first", "second"], "relations")
    assert replay.actual_relation_calls == 2
    assert backend.relation_calls == 0
    assert replay.provenance["new_gpu_relation_calls_unit"] == "relation_backend_attempted_jobs"


@pytest.mark.parametrize("fault", ["runtime", "completed", "adapter", "window", "metric_hash"])
def test_manifest_version_completion_adapter_and_hash_guards(origin, fault):
    def change(m):
        if fault == "runtime": m["runtime_sha256"] = "old_v1_runtime"
        elif fault == "completed": m["completed"] = False
        elif fault == "adapter": m["adapters"]["entities"]["adapter_files_sha256"] = "f" * 64
        elif fault == "window": m["entity_strategy"]["entity_window_char_budget"] = 300
        else: m["metrics_sha256"] = "f" * 64
    rewrite(origin, transform_manifest=change)
    with pytest.raises(ValueError): plan(origin)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "oracle", "identity", "prompt", "window", "cached"])
def test_complete_coverage_and_exact_inputs_even_if_artifact_hashes_recomputed(origin, fault):
    def change(rows):
        first = next(row for row in rows if row["stage"] == "entities")
        if fault == "missing": rows.remove(first)
        elif fault == "duplicate": rows.append(deepcopy(first))
        elif fault == "oracle": first["mode"] = "oracle"
        elif fault == "identity": first["input_identity"]["sample_index"] += 1
        elif fault == "prompt": first["prompt"] += "injected"
        elif fault == "window": first["window"]["start"] += 1
        else: first["generation_trace"]["entity_cache"] = {"cache_replay": True}
    rewrite(origin, transform_rows=change)
    with pytest.raises(ValueError): plan(origin)


def test_missing_artifact_and_oracle_only_and_different_decode_budget_are_rejected(origin):
    for kwargs in ({"mode": "oracle"}, {"max_new_tokens": 2048}, {"max_input_tokens": 4096}):
        with pytest.raises(ValueError): plan(origin, **kwargs)
    (origin.path / "predictions.raw.jsonl").unlink()
    with pytest.raises(FileNotFoundError): plan(origin)


def test_actual_loaded_weights_and_token_ids_are_verified(origin):
    admitted = plan(origin)
    backend = Backend()
    backend.model.fingerprint = "wrong"
    with pytest.raises(ValueError, match="actual loaded"): admitted.bind(backend)
    admitted.rows[0]["generation_trace"]["generated_token_ids"][0] += 1
    with pytest.raises(ValueError, match="decode"): admitted.bind(Backend())


def test_duplicate_prompt_cannot_substitute_another_case_identity(origin):
    replay = plan(origin).bind(Backend(forbid_entities=True))
    outputs = replay.generate_many([row["prompt"] for row in replay.plan.rows[:2]], "entities")
    event = deepcopy(replay.plan.rows[1])
    event["generation_trace"] = outputs[0]["trace"]
    assert replay.plan.rows[0]["prompt"] == event["prompt"]
    with pytest.raises(ValueError, match="identity"): replay.validate_generation_event(event)
    with pytest.raises(ValueError, match="not every"): replay.finalize_cache()


def test_no_unplanned_prompt_fallback_and_no_success_before_full_consumption(origin):
    backend = Backend(forbid_entities=True)
    replay = plan(origin).bind(backend)
    with pytest.raises(ValueError, match="prompt order"): replay.generate_many(["not a planned input"], "entities")
    assert backend.entity_calls == backend.relation_calls == replay.cursor == 0
    with pytest.raises(ValueError, match="not every"): replay.finalize_cache()


def test_origin_mutation_after_loading_prevents_completed_run(origin):
    admitted = plan(origin)
    replay = admitted.bind(Backend(batch_size=8, forbid_entities=True))
    (origin.path / "run_manifest.json").write_text("changed after admission")
    with pytest.raises(ValueError, match="origin file changed"):
        e.evaluate_cases(origin.cases, replay, modes=("live",))


def test_complete_window_short_cache_preserves_partial_window_failures_and_global_merges(tmp_path, monkeypatch):
    origin = make_origin(tmp_path, monkeypatch, entity_variant="short_source", short_chars=64)
    replay = plan(origin).bind(Backend(batch_size=8, forbid_entities=True, training_variant="short_source_complete_windows"))
    metrics, predictions, _ = e.evaluate_cases(origin.cases, replay, entity_variant="short_source", short_chars=64, modes=("live",))
    assert metrics["modes"] == origin.metrics["modes"]
    assert metrics["generation_complete"] is False
    assert metrics["entity_cache"]["expected_entity_calls"] > 69
    assert metrics["entity_cache"]["expected_entity_calls"] == metrics["entity_cache"]["cached_entity_calls"]
    assert metrics["entity_cache"]["entity_strategy"]["entity_training_variant"] == "short_source_complete_windows"
    for view in ("raw", "filtered"):
        assert [row["output"] for row in predictions["live"][view]] == [row["output"] for row in origin.predictions["live"][view]]


@pytest.mark.parametrize("failure", ["input_budget", "invalid_records"])
def test_empty_pre_generation_refusal_and_eos_complete_parser_failure_are_replayed_unchanged(tmp_path, monkeypatch, failure):
    origin = make_origin(tmp_path, monkeypatch, fail_kind=failure)
    replay = plan(origin).bind(Backend(batch_size=8, forbid_entities=True))
    metrics, _, results = e.evaluate_cases(origin.cases, replay, modes=("live",))
    assert metrics["modes"] == origin.metrics["modes"]
    first = results["live"][0]["generations"][0]
    assert first["status"] == "failed"
    assert first["error"] == ("input_budget_exceeded_no_truncation" if failure == "input_budget" else "invalid_semantic_records")
    assert metrics["modes"]["live"]["filtered"]["free_text"]["relation"]["fn"] == 1


def test_evaluation_cli_cache_manifest_progress_and_real_relation_calls(origin, tmp_path, monkeypatch):
    monkeypatch.setattr(e, "load_release", lambda *args, **kwargs: (origin.cases, origin.data))
    monkeypatch.setattr(e, "check_adapter_contract", lambda path, stage, **kwargs: {**receipt(), "stage": stage})
    backend = Backend(batch_size=8, forbid_entities=True)
    monkeypatch.setattr(e, "SemanticGenerator", lambda *args, **kwargs: backend)
    directory = tmp_path / "cache_eval"
    args = ["--release", "release", "--base-model", "base", "--entity-adapter", "entities", "--relation-adapter", "new_relations",
        "--output-dir", str(directory), "--entity-cache-run", str(origin.path), "--mode", "live", "--batch-size", "8"]
    metrics = e.main(args)
    manifest = json.loads((directory / "manifest.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    assert manifest["completed"] is True and manifest["entity_cache"]["all_cached_calls_consumed"] is True
    assert metrics["modes"] == origin.metrics["modes"]
    assert progress["cached_entity_jobs_replayed"] == 69 and progress["new_gpu_entity_jobs"] == 0
    assert progress["new_gpu_relation_jobs"] == backend.relation_calls == 68
    events = [json.loads(line) for line in (directory / "generations.jsonl").read_text().splitlines()]
    assert all(bool(event["generation_trace"].get("entity_cache")) == (event["stage"] == "entities") for event in events)


def test_cache_preflight_reports_deferred_in_memory_verification_without_loading_model(origin, tmp_path, monkeypatch):
    monkeypatch.setattr(e, "load_release", lambda *args, **kwargs: (origin.cases, origin.data))
    monkeypatch.setattr(e, "check_adapter_contract", lambda path, stage, **kwargs: {**receipt(), "stage": stage})
    monkeypatch.setattr(e, "SemanticGenerator", lambda *args, **kwargs: pytest.fail("Preflight loaded GPU model"))
    result = e.main(["--release", "release", "--base-model", "base", "--entity-adapter", "entities", "--relation-adapter", "relations",
        "--output-dir", str(tmp_path / "preflight"), "--entity-cache-run", str(origin.path), "--mode", "live", "--preflight-only"])
    assert result["completed"] is False and result["entity_cache"]["token_and_in_memory_checks"] == "pending_generator_binding"

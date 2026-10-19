"""Strict, evaluation-only replay of complete-run live entity generations.

This is not a prompt-keyed memoizer. A fixed identity/window plan is consumed in
the frozen runtime's stable scheduling order, then checked against each actual
runtime event before it can be written. Relations always use the real backend.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from evaluation_identity import make_input_identity
from generative.chat import render_prompt
from generative.train import adapter_fingerprint
from semantic import protocol as p
from semantic import runtime as r
from semantic.data import sha, source_windows


VERSION = "semantic-live-entity-replay-v1"


def require(condition, message):
    if not condition:
        raise ValueError("Entity cache: " + message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class EntityCachePlan:
    def __init__(self, rows, provenance, origin_files, entity_receipt):
        self.rows = rows
        self.provenance = provenance
        self.origin_files = origin_files
        self.entity_receipt = deepcopy(entity_receipt)

    @classmethod
    def load(cls, directory, cases, data_manifest, entity_receipt, *, entity_variant, base_model,
             max_input_tokens, max_new_tokens, mode="both"):
        require(mode in ("live", "both"), "replay requires live evaluation, never oracle-only")
        require(len(cases) == 130 and all(case.get("split") == "tune" for case in cases), "only the complete frozen 130-case tune population is supported")
        require(len({case["id"] for case in cases}) == 130 and len({case["index"] for case in cases}) == 130, "duplicate case identities")
        require(sum(case["source_type"] == "free_text" for case in cases) == 69, "complete 69-free/61-structured population required")
        root = Path(directory)
        files = {}
        def read(name):
            path = root / name
            content = path.read_bytes()
            files[str(path.resolve())] = sha(content)
            return content
        manifest_bytes, run_bytes, metrics_bytes = read("manifest.json"), read("run_manifest.json"), read("metrics.json")
        origin, run, metrics = json.loads(manifest_bytes), json.loads(run_bytes), json.loads(metrics_bytes)
        require(origin == run, "origin manifest and run manifest disagree")
        require(origin.get("status") == "complete" and origin.get("complete") is True and origin.get("completed") is True
                and origin.get("failed") is not True, "origin is not a completed successful evaluation run")
        require(not origin.get("entity_cache"), "cache-of-cache origin runs are not admitted")
        require(metrics.get("status") == "complete" and metrics.get("complete") is True, "origin metrics are incomplete")
        require(origin.get("metrics_sha256") == sha(metrics_bytes), "origin metrics artifact SHA mismatch")
        require(origin.get("samples") == metrics.get("samples") == 130 and metrics.get("modes", {}).get("live", {}).get("samples") == 130,
                "origin does not cover every live tune case")
        require(origin.get("output_contract") == metrics.get("output_contract") == p.CONTRACT, "origin output contract mismatch")
        require(origin.get("sample_indices") == [case["index"] for case in cases], "origin sample order/identity mismatch")
        require(origin.get("input_release") == data_manifest and metrics.get("input_release_sha256") == data_manifest["manifest_sha256"],
                "different frozen release or input provenance")
        runtime_sha = sha(Path(r.__file__).read_bytes())
        require(origin.get("runtime_sha256") == runtime_sha, "origin runtime SHA differs; no compatibility fallback is allowed")
        require(origin.get("protocol_sha256") == metrics.get("protocol_sha256") == r.protocol_digest(), "protocol SHA mismatch")
        require(_digest(origin.get("evaluator_sha256")), "origin evaluator SHA missing")
        require(origin.get("entity_variant") == entity_variant, "different entity input view")
        expected_strategy = r.entity_strategy_contract(entity_variant, entity_receipt, short_chars=data_manifest["short_chars"])
        require(origin.get("entity_strategy") == expected_strategy, "entity view/order/window/training receipt mismatch")
        args = origin.get("args", {})
        require(args.get("mode") in ("live", "both") and args.get("entity_variant") == entity_variant, "origin was not the declared live entity strategy")
        require(args.get("base_model") == str(base_model), "base model path differs")
        require(args.get("max_input_tokens") == max_input_tokens and args.get("max_new_tokens") == max_new_tokens,
                "entity input/output token budgets differ; only batch size may change")
        expected_adapter = {key: entity_receipt[key] for key in ("stage", "variant", "adapter_files_sha256", "base_input_sha256", "data_sha256")}
        require(origin.get("adapters", {}).get("entities") == expected_adapter, "different entity adapter files/receipt/data/base")
        fingerprint = entity_receipt.get("adapter_final_sha256")
        require(_digest(fingerprint), "entity receipt fingerprint missing")
        verification = origin.get("loaded_adapter_verification", {}).get("entities", {})
        require(all(verification.get(key) == fingerprint for key in ("expected_sha256", "loaded_sha256", "cuda_sha256")),
                "origin in-memory entity weights were not verified exactly")
        artifacts = metrics.get("artifacts_sha256", {})
        artifact_bytes = {}
        for name in ("generations.jsonl", "predictions.raw.jsonl", "predictions.filtered.jsonl"):
            content = read(name)
            artifact_bytes[name] = content
            require(artifacts.get(name) == sha(content), "origin artifact SHA mismatch: " + name)
            if name.startswith("predictions."):
                predictions = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
                modes = ("live", "oracle") if args["mode"] == "both" else ("live",)
                require(len(predictions) == len(cases) * len(modes) and len({row["id"] for row in predictions}) == len(predictions),
                        "origin predictions do not cover every unique case/mode")
                for mode_name in modes:
                    selected = [row for row in predictions if row.get("mode") == mode_name]
                    require([row["id"] for row in selected] == [mode_name + ":" + case["id"] for case in cases], "origin prediction mode/order differs")
                    for row, case in zip(selected, cases):
                        require(row.get("input_identity") == make_input_identity(case["original_record"], case["index"])
                                and row.get("semantic_input_identity") == case["input_identity"], "origin prediction identity differs")
        all_rows = [json.loads(line) for line in artifact_bytes["generations.jsonl"].decode().splitlines() if line.strip()]
        require(len({row["id"] for row in all_rows}) == len(all_rows), "duplicate origin generation IDs")
        require(all(row.get("mode") in ("live", "oracle") and row.get("stage") in ("entities", "relations") for row in all_rows),
                "unrecognized origin generation mode/stage")
        by_case = {case["id"]: case for case in cases if case["source_type"] == "free_text"}
        origin_modes = ("live", "oracle") if args["mode"] == "both" else ("live",)
        for row in all_rows:
            case = by_case.get(row.get("document_id"))
            require(case is not None and row["mode"] in origin_modes, "origin generation belongs to an unknown case or mode")
            require(row.get("input_identity") == make_input_identity(case["original_record"], case["index"])
                    and row.get("semantic_input_identity") == case["input_identity"], "origin generation input identity differs")
            require(row["id"].startswith(row["mode"] + ":" + case["id"] + ":" + row["stage"]), "origin generation ID/mode association differs")
        candidates = [row for row in all_rows if row["stage"] == "entities"]
        require(all(row["mode"] == "live" for row in candidates), "oracle/non-live entity records are forbidden")
        by_id = {row["id"]: row for row in candidates}
        rows, job_number = [], 0
        free = [case for case in cases if case["source_type"] == "free_text"]
        for document_index, case in enumerate(free):
            text, metadata = case["text"], case["metadata"]
            windows = [(0, len(text))] if entity_variant == "full_source" else source_windows(text, data_manifest["short_chars"], min(400, data_manifest["short_chars"] // 4))
            for window_index, (a, b) in enumerate(windows):
                suffix = "" if entity_variant == "full_source" else f":{window_index}"
                key = f"live:{case['id']}:entities{suffix}"
                require(key in by_id, "missing entity source/window: " + key)
                row = by_id[key]
                window = {"start": a, "end": b, "index": window_index}
                local_metadata = (metadata if entity_variant == "full_source" else
                    {**deepcopy(metadata), "variant": entity_variant, "order": "source",
                     "source_window": {**window, "full_source_fallback": False}, "relation_entity_table_scope": None})
                prompt = p.build_entity_prompt(text[a:b], local_metadata, "source")
                require(row.get("document_id") == case["id"] and row.get("document_index") == document_index
                        and row.get("window") == window and row.get("entity_variant") == entity_variant,
                        "entity source/window/order identity differs")
                require(row.get("input_identity") == make_input_identity(case["original_record"], case["index"])
                        and row.get("semantic_input_identity") == case["input_identity"], "original/augmented input identity differs")
                require(row.get("text") == text[a:b] and row.get("metadata") == local_metadata and row.get("prompt") == prompt,
                        "entity source or literal prompt differs")
                require(row.get("output_contract") == p.CONTRACT and row["generation_trace"].get("output_contract") == p.CONTRACT,
                        "entity record output contract differs")
                require(row["generation_trace"].get("stage") == "entities" and row["generation_trace"].get("prompt_sha256") == sha(prompt),
                        "entity trace stage/prompt hash differs")
                trace = row["generation_trace"]
                refusal = (trace.get("generated_token_ids") == [] and trace.get("generated_tokens") == 0
                           and trace.get("complete") is False and trace.get("eos_observed") is False and trace.get("error"))
                require(trace.get("max_new_tokens") == max_new_tokens or (refusal and "max_new_tokens" not in trace),
                        "saved entity decoding budget differs")
                require(not row["generation_trace"].get("entity_cache"), "cache-of-cache origins are not admitted")
                error = trace.get("error")
                if trace.get("complete") is not True or trace.get("eos_observed") is not True or trace.get("budget_exhausted") or trace.get("deadline_exhausted"):
                    error = error or "incomplete_generation"
                parsed = None if error else p.parse_entities(row["text"], row["completion_text"], row["metadata"])
                if parsed is not None and not parsed["valid"]: error = "invalid_semantic_records"
                require(row.get("parse") == parsed and row.get("error") == error and row.get("status") == ("failed" if error else "complete"),
                        "saved parse/failure state differs from exact frozen replay")
                require(row.get("invalid_attempt_records") == (sum(bool(line.strip()) for line in row["completion_text"].splitlines()) if error else 0),
                        "saved invalid-attempt diagnostic differs")
                rows.append((len(prompt), job_number, deepcopy(row)))
                job_number += 1
        require(len(rows) == len(candidates), "unexpected extra entity case/window")
        ordered = [entry[2] for entry in sorted(rows, key=lambda entry: (entry[0], entry[1]))]
        provenance = {"version": VERSION, "origin_run": str(root.resolve()), "origin_manifest_sha256": sha(manifest_bytes),
            "origin_run_manifest_sha256": sha(run_bytes), "origin_metrics_sha256": sha(metrics_bytes),
            "origin_generations_sha256": artifacts["generations.jsonl"], "origin_evaluator_sha256": origin["evaluator_sha256"],
            "runtime_sha256": runtime_sha, "protocol_sha256": r.protocol_digest(),
            "cache_code_sha256": sha(Path(__file__).read_bytes()), "release_manifest_sha256": data_manifest["manifest_sha256"],
            "entity_adapter_fingerprint": fingerprint, "entity_adapter_files_sha256": entity_receipt["adapter_files_sha256"],
            "entity_strategy": expected_strategy, "expected_entity_calls": len(ordered), "samples": 130,
            "origin_failed_entity_calls": sum(row["status"] == "failed" for row in ordered),
            "origin_entity_batch_size": args.get("batch_size"), "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
            "policy": "Strict identity/window consumption and event acknowledgement; failures replay unchanged; no prompt lookup fallback, oracle entities, cache chaining or GPU entity recomputation",
            "code_match_scope": "Runtime and protocol bytes must match exactly. Origin/current evaluator hashes are separately recorded; literal prompts and all entity job identities are reconstructed independently.",
            "token_and_in_memory_checks": "pending_generator_binding"}
        return cls(ordered, provenance, files, entity_receipt)

    def verify_origin_unchanged(self):
        for filename, digest in self.origin_files.items():
            require(sha(Path(filename).read_bytes()) == digest, "origin file changed after admission")

    def bind(self, backend):
        return EntityCacheGenerator(self, backend)


class EntityCacheGenerator:
    def __init__(self, plan, backend):
        self.plan, self.backend = plan, backend
        self.batch_size = backend.batch_size
        self.receipts = backend.receipts
        self.weight_verification = backend.weight_verification
        self.cursor, self.validated, self.pending, self.actual_relation_calls = 0, 0, [], 0
        require(self.receipts.get("entities") == plan.entity_receipt, "loaded entity receipt differs from admitted receipt")
        require(backend.max_input_tokens == plan.provenance["max_input_tokens"] and backend.max_new_tokens == plan.provenance["max_new_tokens"], "loaded decoding budgets differ")
        self._verify_weights()
        for row in plan.rows:
            trace = row["generation_trace"]
            rendered = render_prompt(backend.tokenizer, row["prompt"])
            ids = trace.get("generated_token_ids")
            require(isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids), "saved token IDs are malformed")
            refusal = not ids and trace.get("complete") is False and trace.get("eos_observed") is False and trace.get("error")
            require((trace.get("rendered_prompt_sha256") == sha(rendered) or (refusal and "rendered_prompt_sha256" not in trace))
                    and trace.get("prompt_tokens") == len(backend.tokenizer.encode(rendered, add_special_tokens=False)),
                    "tokenizer/prompt token identity differs")
            decoded = r.decode_completion(backend.tokenizer, ids, max_new_tokens=backend.max_new_tokens,
                                          deadline_exhausted=trace.get("deadline_exhausted", False))
            require(decoded["completion_text"] == row["completion_text"] == trace.get("completion_text"), "saved token IDs do not decode to the exact cached text")
            # Pre-generation refusals have no token sequence and carry a more
            # specific error than missing_eos. Preserve those failures exactly.
            for key in ("generated_tokens", "eos_observed", "budget_exhausted", "deadline_exhausted", "complete"):
                require(trace.get(key, False if key == "deadline_exhausted" else None) == decoded["trace"][key], "cached stopping telemetry differs from actual token IDs")
            require(refusal or trace.get("error") == decoded["trace"]["error"], "cached stop reason differs")
        self.provenance = {**deepcopy(plan.provenance), "token_and_in_memory_checks": "verified",
            "relation_batch_size": backend.batch_size, "relation_generation": "Fresh real backend attempts with greedy non-thinking decoding; refusals and failures remain explicit",
            "new_gpu_relation_calls_unit": "relation_backend_attempted_jobs",
            "new_gpu_relation_calls_definition": "Number of relation prompts submitted to the real backend, counted before invocation. Includes input-budget/deadline refusals and backend exceptions; not GPU decode successes or successful relation outputs."}

    def _verify_weights(self):
        require(adapter_fingerprint(self.backend.model, "entities") == self.plan.provenance["entity_adapter_fingerprint"],
                "actual loaded entity adapter fingerprint differs")

    def generate_many(self, prompts, stage, **kwargs):
        require(not self.pending, "previous cached batch has unacknowledged identity events")
        if stage == "relations":
            self.actual_relation_calls += len(prompts)
            return self.backend.generate_many(prompts, stage, **kwargs)
        require(stage == "entities", "unknown stage")
        selected = self.plan.rows[self.cursor:self.cursor + len(prompts)]
        require(len(selected) == len(prompts) and [row["prompt"] for row in selected] == list(prompts),
                "entity prompt order/consumption differs from frozen runtime plan")
        outputs = []
        for row in selected:
            trace = deepcopy(row["generation_trace"])
            trace["entity_cache"] = {"version": VERSION, "cache_replay": True, "new_gpu_entity_generation": False,
                "origin_generation_id": row["id"], "origin_record_sha256": sha(row),
                "origin_manifest_sha256": self.provenance["origin_manifest_sha256"],
                "origin_generations_sha256": self.provenance["origin_generations_sha256"],
                "entity_adapter_fingerprint": self.provenance["entity_adapter_fingerprint"],
                "original_generation_elapsed_seconds": trace.get("elapsed_seconds")}
            trace["elapsed_seconds"] = 0.0
            outputs.append({"completion_text": row["completion_text"], "trace": trace})
        self.pending = selected[:]
        self.cursor += len(selected)
        return outputs

    def validate_generation_event(self, event):
        if event["stage"] == "relations":
            require(not self.pending and not event["generation_trace"].get("entity_cache"), "relations were improperly cached")
            return
        require(event["stage"] == "entities" and self.pending, "unexpected/unplanned entity event")
        expected = self.pending[0]
        for key in ("id", "document_id", "document_index", "mode", "input_identity", "semantic_input_identity", "window", "entity_variant", "text", "metadata", "prompt", "completion_text", "parse", "status", "error", "invalid_attempt_records"):
            require(event.get(key) == expected.get(key), "actual entity event identity or failure state mismatch: " + key)
        provenance = event["generation_trace"].get("entity_cache", {})
        require(provenance.get("origin_generation_id") == expected["id"] and provenance.get("origin_record_sha256") == sha(expected), "cache provenance lost during runtime replay")
        original_trace = {**expected["generation_trace"], "elapsed_seconds": 0.0}
        actual_trace = {key: value for key, value in event["generation_trace"].items() if key != "entity_cache"}
        require(actual_trace == original_trace, "cached token/stop telemetry changed during replay")
        self.pending.pop(0)
        self.validated += 1

    def finalize_cache(self):
        require(not self.pending and self.cursor == self.validated == len(self.plan.rows), "not every cached source/window was consumed and identity-checked")
        self._verify_weights()
        self.plan.verify_origin_unchanged()
        return {**deepcopy(self.provenance), "all_cached_calls_consumed": True, "cached_entity_calls": self.validated,
                "new_gpu_entity_calls": 0, "new_gpu_relation_calls": self.actual_relation_calls}

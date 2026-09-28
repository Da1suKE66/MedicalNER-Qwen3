"""CPU-only alignment regressions; never loads a model or modifies real data."""

import copy
import hashlib
import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import _graph, _prf, _relation_set
from evaluate_graphs import evaluate_records, filter_records
from evaluation_identity import (
    IDENTITY_KEY,
    make_input_identity,
    resolve_sample_indices,
    user_content_sha256,
    validate_alignment,
)
from generative.protocol import ProtocolConfig, dumps


TEXT = "Fatigue is a core symptom of Condition A."
GRAPH = {
    "entities": [
        {"id": "S", "label": "Symptom", "name": "Fatigue"},
        {"id": "D", "label": "Disease", "name": "Condition A"},
    ],
    "relations": [{
        "source": "S", "target": "D", "relation": "is_core_symptom_of",
        "evidence": TEXT,
    }],
}


def record(text=TEXT, graph=None):
    return {"messages": [
        {"role": "user", "content": text},
        {"role": "assistant", "content": dumps(GRAPH if graph is None else graph)},
    ]}


def prediction(item, index):
    return {
        "output": copy.deepcopy(_graph(item["messages"][1]["content"])),
        IDENTITY_KEY: make_input_identity(item, index),
    }


def generate(batch):
    return {"text": dumps({"decisions": [
        {"pair_id": pair["pair_id"], "relations": [
            {"relation": "is_core_symptom_of", "evidence_span_ids": ["SENT_001"]}
        ] if pair["source"] == "S" and pair["target"] == "D" else []}
        for pair in batch["pairs"]
    ]}), "tokens": 13}


class EvaluationIdentityTests(unittest.TestCase):
    def test_hash_uses_literal_user_message_not_routed_or_normalized_text(self):
        raw = "Instruction\r\nMedical text:\r\nＦatigue at 5㎎.  \n"
        item = record(raw)
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self.assertEqual(make_input_identity(item, 7)["user_content_sha256"], expected)
        self.assertNotEqual(expected, user_content_sha256(record(raw.strip())))
        self.assertNotEqual(expected, user_content_sha256(record(raw.replace("㎎", "mg"))))
        changed_labels = copy.deepcopy(item)
        changed_labels["messages"][1]["content"] = "different labels"
        self.assertEqual(expected, user_content_sha256(changed_labels))

    def test_missing_ambiguous_or_nonstring_user_is_not_fabricated(self):
        for messages in ([], [{"role": "user", "content": None}],
                         [{"role": "user", "content": []}],
                         [{"role": "user", "content": "a"}] * 2):
            with self.subTest(messages=messages), self.assertRaisesRegex(ValueError, "user message"):
                make_input_identity({"messages": messages}, 0)

    def test_same_length_reordering_and_repeated_row_rejected(self):
        records = [record(), record(TEXT + " Other input.")]
        rows = [prediction(item, i) for i, item in enumerate(records)]
        for bad in (rows[::-1], [rows[0], rows[0]]):
            for strict in (False, True):
                with self.subTest(strict=strict), self.assertRaisesRegex(ValueError, "sample_index mismatch"):
                    evaluate_records(records, bad, strict_identity=strict)

    def test_reindexed_wrong_source_and_changed_wrapper_rejected(self):
        records = [record(), record(TEXT + " Other input.")]
        rows = [prediction(item, i) for i, item in enumerate(records)][::-1]
        for i, row in enumerate(rows):
            row[IDENTITY_KEY]["sample_index"] = i
        with self.assertRaisesRegex(ValueError, "user_content_sha256 mismatch"):
            evaluate_records(records, rows)
        with self.assertRaisesRegex(ValueError, "user_content_sha256 mismatch"):
            evaluate_records([record("Instruction\nMedical text:\n" + TEXT)], [prediction(records[0], 0)])

    def test_identical_inputs_still_require_correct_indices(self):
        records = [record(), record()]
        rows = [prediction(item, i) for i, item in enumerate(records)]
        self.assertEqual(evaluate_records(records, rows, strict_identity=True)["alignment"]["status"], "verified")
        with self.assertRaisesRegex(ValueError, "sample_index mismatch"):
            evaluate_records(records, rows[::-1])

    def test_legacy_mixed_and_invalid_predictions_are_visibly_unverified(self):
        records = [record(), record(), record()]
        rows = [prediction(records[0], 0), {"output": copy.deepcopy(GRAPH)}, '{"entities": [']
        before = copy.deepcopy(rows)
        metrics = evaluate_records(records, rows)
        self.assertEqual(metrics["alignment"], {
            "status": "unverified", "verified_samples": 1, "unverified_samples": 2,
        })
        self.assertEqual(metrics["overall"]["alignment"], metrics["alignment"])
        self.assertEqual(metrics["samples"][1]["alignment"]["reason"], "missing_input_identity")
        self.assertEqual(metrics["overall"]["relation"]["fn"], 1)
        self.assertEqual(metrics["overall"]["invalid_predictions"], 1)
        self.assertEqual(rows, before)
        with self.assertRaisesRegex(ValueError, "strict identity.*unverified"):
            evaluate_records(records, rows, strict_identity=True)

    def test_declared_legacy_index_conflicts_are_not_ignored(self):
        for key in ("sample_index", "sample_id"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                evaluate_records([record()], [{"output": GRAPH, key: 8}])
        metrics = evaluate_records([record()], [{"output": GRAPH, "sample_index": 0}])
        self.assertEqual(metrics["alignment"]["status"], "unverified")

    def test_malformed_identity_rejected_in_compatibility_mode(self):
        item = record()
        valid = make_input_identity(item, 0)
        malformed = [None, {}, "wrong"] + [
            {**valid, key: value} for key, value in (
                ("schema_version", True), ("schema_version", 999),
                ("sample_index", False), ("sample_index", -1),
                ("user_content_sha256", "not-a-digest"), ("status", "maybe"),
            )
        ]
        for identity in malformed:
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                evaluate_records([item], [{"output": GRAPH, IDENTITY_KEY: identity}])

    def test_unverified_provenance_is_sticky_but_still_checked(self):
        item = record()
        row = prediction(item, 0)
        row[IDENTITY_KEY]["status"] = "unverified"
        self.assertEqual(evaluate_records([item], [row])["alignment"]["status"], "unverified")
        with self.assertRaisesRegex(ValueError, "unverified"):
            evaluate_records([item], [row], strict_identity=True)
        row[IDENTITY_KEY]["user_content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            evaluate_records([item], [row])

    def test_subset_indices_are_explicit_unique_and_not_guessed_from_predictions(self):
        records = [record(), record(TEXT + " Another source.")]
        rows = [prediction(item, i) for item, i in zip(records, [7, 3])]
        metrics = evaluate_records(records, rows, sample_indices=[7, 3], strict_identity=True)
        self.assertEqual([r["sample_index"] for r in metrics["samples"]], [7, 3])
        with self.assertRaisesRegex(ValueError, "sample_index mismatch"):
            evaluate_records(records, rows)
        for indices in ([1], [1, 1], [False, 1], [-1, 2], ["0", 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                resolve_sample_indices(records, indices)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            validate_alignment(records, rows[:1])

    def test_filter_and_rule_ablation_keep_identity_trace_and_legacy_status(self):
        from generative.evaluate import rule_ablation
        item = record()
        row = prediction(item, 0)
        row["trace"] = {"chunks": [{"relation_errors": ["invalid_json"]}]}
        row["experiment"] = "keep-me"
        before = copy.deepcopy(row)
        filtered, _ = filter_records([row])
        self.assertEqual(row, before)
        self.assertEqual(filtered[0][IDENTITY_KEY], row[IDENTITY_KEY])
        self.assertEqual(filtered[0]["experiment"], "keep-me")
        self.assertEqual(evaluate_records([item], filtered, strict_identity=True)["overall"]["relation_stage_errors"], 1)
        for report in rule_ablation([item], [row], strict_identity=True).values():
            self.assertEqual(report["metrics"]["alignment"]["status"], "verified")
        legacy, _ = filter_records([{"output": GRAPH}])
        self.assertNotIn(IDENTITY_KEY, legacy[0])
        self.assertEqual(evaluate_records([item], legacy)["alignment"]["status"], "unverified")
        with self.assertRaisesRegex(ValueError, "strict identity"):
            rule_ablation([item], legacy, strict_identity=True)

    def test_original_unique_triple_scoring_is_unchanged(self):
        item = record()
        row = prediction(item, 0)
        row["output"]["relations"][0]["evidence"] = "wrong evidence"
        row["output"]["relations"].append(copy.deepcopy(row["output"]["relations"][0]))
        row["output"]["relations"].append({"source": "MISSING", "target": "D", "relation": "causes"})
        metrics = evaluate_records([item], [row], strict_identity=True)["overall"]
        expected = _prf(_relation_set(row["output"]), _relation_set(GRAPH))
        for key, value in expected.items():
            self.assertEqual(metrics["relation"][key], value)
        self.assertEqual((metrics["relation"]["tp"], metrics["relation"]["fp"]), (1, 1))
        self.assertEqual(metrics["evidence_exact"]["tp"], 0)

    def test_invalid_graph_keeps_identity_and_false_negatives_after_filtering(self):
        item = record()
        row = {"output": "not JSON", IDENTITY_KEY: make_input_identity(item, 0)}
        filtered, removed = filter_records([row])
        self.assertEqual(filtered, [row])
        self.assertFalse(removed)
        metrics = evaluate_records([item], filtered, strict_identity=True)
        self.assertEqual(metrics["alignment"]["status"], "verified")
        self.assertEqual(metrics["overall"]["invalid_predictions"], 1)
        self.assertEqual(metrics["overall"]["relation"]["fn"], 1)


class GenerationIdentityTests(unittest.TestCase):
    def run_records(self, *args, **kwargs):
        from generative.evaluate import run_records
        with redirect_stdout(io.StringIO()):
            return run_records(*args, **kwargs)

    def test_new_oracle_predictions_and_raw_batches_have_bound_identity(self):
        records = [record(), record(TEXT + " Another source.")]
        rows, raw, audit = self.run_records(records, None, ProtocolConfig(), generate, oracle=True, strict_identity=True)
        self.assertEqual(audit["identity_verified_samples"], 2)
        self.assertEqual(evaluate_records(records, rows, strict_identity=True)["alignment"]["status"], "verified")
        for generated in raw:
            index = generated["sample_id"]
            self.assertEqual(generated[IDENTITY_KEY], rows[index][IDENTITY_KEY])
            self.assertTrue(generated["id"].startswith(f"{index}:"))
        with self.assertRaisesRegex(ValueError, "sample_index mismatch"):
            evaluate_records(records, rows[::-1])

    def test_reordered_cache_rejected_before_any_generation(self):
        records = [record(), record(TEXT + " Another source.")]
        cache = [prediction(item, i) for i, item in enumerate(records)]
        callback = Mock(side_effect=AssertionError("must validate the whole cache first"))
        with self.assertRaisesRegex(ValueError, "sample_index mismatch"):
            self.run_records(records, cache[::-1], ProtocolConfig(), callback)
        callback.assert_not_called()
        cache[1][IDENTITY_KEY]["user_content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            self.run_records(records, cache, ProtocolConfig(), callback)
        callback.assert_not_called()

    def test_old_cache_is_not_upgraded_in_new_predictions_or_raw_output(self):
        records = [record()]
        cache = [{"output": copy.deepcopy(GRAPH)}]
        before = copy.deepcopy(cache)
        rows, raw, audit = self.run_records(records, cache, ProtocolConfig(), generate)
        self.assertEqual(cache, before)
        self.assertEqual(audit["identity_unverified_samples"], 1)
        for row in [*rows, *raw]:
            self.assertEqual(row[IDENTITY_KEY]["status"], "unverified")
        self.assertEqual(evaluate_records(records, rows)["alignment"]["status"], "unverified")
        callback = Mock()
        for cached in (cache, rows):
            with self.subTest(cached=cached), self.assertRaisesRegex(ValueError, "strict identity"):
                self.run_records(records, cached, ProtocolConfig(), callback, strict_identity=True)
        callback.assert_not_called()

    def test_verified_cache_remains_verified_and_is_not_mutated(self):
        records = [record()]
        cache = [prediction(records[0], 0)]
        before = copy.deepcopy(cache)
        rows, _, _ = self.run_records(records, cache, ProtocolConfig(), generate, strict_identity=True)
        self.assertEqual(cache, before)
        self.assertEqual(evaluate_records(records, rows, strict_identity=True)["alignment"]["status"], "verified")

    def test_oracle_free_text_does_not_inherit_unused_cache_provenance(self):
        records = [record()]
        rows, _, _ = self.run_records(records, [{"output": GRAPH}], ProtocolConfig(), generate, oracle=True)
        self.assertEqual(rows[0][IDENTITY_KEY]["status"], "verified")

    def test_zero_batches_and_structured_icd_keep_original_indices(self):
        from kg_agent.assemble import assemble_graph
        from kg_agent.source import SourceRouter, parse_icd_graph
        icd_text = '{"icd_code":"6A00","title":"Condition A"}'
        icd_graph = assemble_graph(parse_icd_graph(SourceRouter().route(icd_text)))
        records = [record("No candidate pairs.", {"entities": [], "relations": []}), record(icd_text, icd_graph)]
        callback = Mock(side_effect=AssertionError("no neural generation expected"))
        rows, raw, _ = self.run_records(records, None, ProtocolConfig(), callback, oracle=True, sample_indices=[4, 9])
        callback.assert_not_called()
        self.assertFalse(raw)
        self.assertEqual([r[IDENTITY_KEY]["sample_index"] for r in rows], [4, 9])
        self.assertEqual(evaluate_records(records, rows, strict_identity=True, sample_indices=[4, 9])["alignment"]["status"], "verified")
        cached = [{"output": {"entities": [], "relations": []}}, {"output": icd_graph}]
        before = copy.deepcopy(cached)
        rows, _, _ = self.run_records(records, cached, ProtocolConfig(), callback, oracle=True, sample_indices=[4, 9])
        self.assertEqual([r[IDENTITY_KEY]["status"] for r in rows], ["verified", "unverified"])
        self.assertEqual(cached, before)

    def test_subset_merge_matches_full_run_including_zero_candidate_rows(self):
        from generative.evaluate import rule_ablation
        records = [record(TEXT + f" Observation {i}.") for i in range(6)]
        # Duplicate inputs still have distinct positions, and one cache row has
        # no candidate batch from which an index could be inferred.
        records[3] = copy.deepcopy(records[1])
        cached = [prediction(item, i) for i, item in enumerate(records)]
        cached[2]["output"]["entities"] = []
        for oracle in (False, True):
            with self.subTest(oracle=oracle):
                expected, expected_raw, audit = self.run_records(records, cached, ProtocolConfig(), generate, oracle=oracle)
                merged, raw, counts = [None] * len(records), [], Counter()
                for indices in ([0, 2, 4], [1, 3, 5]):
                    subset = [records[i] for i in indices]
                    rows, generated, partial = self.run_records(
                        subset, [cached[i] for i in indices], ProtocolConfig(), generate,
                        oracle=oracle, strict_identity=True, sample_indices=indices,
                    )
                    evaluate_records(subset, rows, strict_identity=True, sample_indices=indices)
                    rule_ablation(subset, rows, strict_identity=True, sample_indices=indices)
                    for index, row in zip(indices, rows):
                        merged[index] = row
                    raw.extend(generated)
                    counts.update(partial)
                self.assertEqual(merged, expected)
                self.assertEqual(sorted(raw, key=lambda r: r["sample_id"]), expected_raw)
                self.assertEqual(dict(counts), audit)
                self.assertEqual(evaluate_records(records, merged), evaluate_records(records, expected))

    def test_batch_sample_id_cannot_silently_reindex_subset(self):
        import generative.evaluate as module
        original = module.build_batches
        def remapped(*args, **kwargs):
            batches, doc, audit = original(*args, **kwargs)
            for batch in batches:
                batch["sample_id"] = 4
            return batches, doc, audit
        callback = Mock()
        with patch.object(module, "build_batches", remapped), self.assertRaisesRegex(ValueError, "sample_indices"):
            self.run_records([record()], None, ProtocolConfig(), callback, oracle=True)
        callback.assert_not_called()

    def test_same_length_reordered_tagged_generation_batches_rejected(self):
        def reordered(batches):
            return [{**generate(b), "id": b["id"], "sample_id": b["sample_id"], IDENTITY_KEY: b[IDENTITY_KEY]}
                    for b in reversed(batches)]
        with self.assertRaisesRegex(ValueError, "Generation batch alignment mismatch"):
            self.run_records([record()], None, ProtocolConfig(pairs_per_batch=1), None, oracle=True, generate_many=reordered)

    def test_cache_and_generation_length_guards_remain(self):
        callback = Mock()
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            self.run_records([{}], [], ProtocolConfig(), callback)
        callback.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Generation batch alignment"):
            self.run_records([record()], None, ProtocolConfig(), None, oracle=True, generate_many=lambda batches: [])

    def test_v2_identity_is_from_raw_user_not_normalized_generation_payload(self):
        raw = "Instruction\r\nMedical text:\r\n" + TEXT + " Dose 5㎎.\r\n"
        item = record(raw)
        rows, generated, _ = self.run_records([item], None, ProtocolConfig(matching="nfkc_v2"), generate, oracle=True)
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        self.assertEqual(rows[0][IDENTITY_KEY]["user_content_sha256"], expected)
        self.assertTrue(all(row[IDENTITY_KEY]["user_content_sha256"] == expected for row in generated))
        self.assertEqual(evaluate_records([item], rows, strict_identity=True)["overall"]["relation"]["tp"], 1)


class IdentityCLITests(unittest.TestCase):
    def test_graph_cli_reports_legacy_status_and_accepts_explicit_subset_jsonl(self):
        from evaluate_graphs import main
        item = record()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, predictions = root / "gold.json", root / "predictions.jsonl"
            output, filtered, indices = root / "metrics.json", root / "filtered.jsonl", root / "indices.json"
            gold.write_text(dumps([item]))
            predictions.write_text(dumps({"output": GRAPH}) + "\n")
            argv = ["evaluate_graphs", "--gold", str(gold), "--predictions", str(predictions), "--output", str(output)]
            printed = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(printed):
                main()
            self.assertIn("unverified", printed.getvalue())
            self.assertEqual(json.loads(output.read_text())["alignment"]["status"], "unverified")
            with patch.object(sys, "argv", argv + ["--strict-identity"]), self.assertRaisesRegex(ValueError, "strict identity"):
                main()
            predictions.write_text(dumps(prediction(item, 9)) + "\n")
            indices.write_text("[9]")
            argv += ["--strict-identity", "--sample-indices", str(indices), "--filter-output", str(filtered)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report["filtered"]["alignment"]["status"], "verified")
            self.assertEqual(json.loads(filtered.read_text())[IDENTITY_KEY]["sample_index"], 9)

    def test_graph_cli_preserves_cases_metadata_and_order(self):
        from evaluate_graphs import main
        records = [record(), record(TEXT + " Another source.")]
        rows = [{**prediction(item, i), "id": i} for i, item in enumerate(records)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, predictions, output = root / "gold.json", root / "pred.json", root / "metrics.json"
            gold.write_text(dumps(records))
            predictions.write_text(dumps({"cases": rows}))
            argv = ["evaluate_graphs", "--gold", str(gold), "--predictions", str(predictions), "--output", str(output), "--strict-identity"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(json.loads(output.read_text())["alignment"]["status"], "verified")
            predictions.write_text(dumps({"cases": rows[::-1]}))
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "sample_index mismatch"):
                main()

    def test_strict_generation_cli_rejects_legacy_cache_before_model_loading(self):
        from generative.evaluate import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold, cache, output = root / "gold.json", root / "cache.jsonl", root / "not-created"
            gold.write_text(dumps([record()]))
            cache.write_text(dumps({"output": GRAPH}) + "\n")
            argv = ["generative.evaluate", "--gold", str(gold), "--entity-predictions", str(cache),
                    "--base-model", "must-not-load", "--adapter", "must-not-load", "--variant", "unused",
                    "--output-dir", str(output), "--strict-identity"]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "strict identity"):
                main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

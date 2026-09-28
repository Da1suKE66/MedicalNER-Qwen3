"""Offline CPU safety tests; model loaders and training are never executed."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_repair.supervision_v2 import partial_supervision, serialize_partial
from generative import train
from generative.protocol import ProtocolConfig, build_batches
from generative.supervision_guard import (
    NEGATIVE, POSITIVE, SupervisionGuardError, audit_partial_supervision,
)


def partial_row(*, negative=False):
    text = "Feature A is a core symptom of Condition B. Condition C is also mentioned."
    entities = [
        {"id": "S", "name": "Feature A", "label": "Symptom"},
        {"id": "D", "name": "Condition B", "label": "Disease"},
        {"id": "X", "name": "Condition C", "label": "Disease"},
    ]
    batches, doc, _ = build_batches(text, entities, ProtocolConfig(), sample_id=1)
    graph = {"entities": entities, "relations": [{
        "source": "S", "target": "D", "relation": "is_core_symptom_of",
        "evidence": text.split(". ")[0] + ".",
    }]}
    row = {**batches[0], "split": "train", "group_id": 1, "sample_weight": 1.0}
    supervision, _ = partial_supervision(row, graph, doc)
    row.update(supervision)
    if negative:
        pid = next(p for p, status in row["pair_supervision"].items()
                   if status.startswith("UNKNOWN_"))
        row["pair_supervision"][pid] = NEGATIVE
        row["unknown_reasons"].pop(pid, None)
        # New builders may attach reliable-negative provenance separately,
        # without changing the status vocabulary or leaking it into the prompt.
        row["negative_provenance"] = {pid: {"source": "test-only-reviewed-negative"}}
        reserialize(row)
    return row


def reserialize(row):
    row["completion"], row["loss_mask_spans"] = serialize_partial(
        row["target"], row["pair_supervision"]
    )


def legacy_row():
    row = partial_row()
    for key in ("pair_supervision", "loss_mask_contract", "loss_mask_spans", "unknown_reasons"):
        row.pop(key)
    return row


def argv(root, *extra):
    return [
        "--mode", "sft", "--data", str(root / "data.jsonl"),
        "--base-model", str(root / "never-load-base"),
        "--output-dir", str(root / "out"), *extra,
    ]


class SupervisionGuardTests(unittest.TestCase):
    def test_positive_only_rejected_and_unknown_excluded(self):
        with self.assertRaises(SupervisionGuardError) as caught:
            audit_partial_supervision([partial_row()])
        audit = caught.exception.audit
        self.assertEqual(audit["decision"], "rejected")
        self.assertEqual(audit["positive_pairs"], 1)
        self.assertEqual(audit["negative_pairs"], 0)
        self.assertEqual(audit["unknown_pairs"], 5)
        self.assertEqual(audit["unknown_loss_pairs"], 0)
        self.assertEqual(audit["negative_ratio"], 0.0)
        self.assertFalse(audit["positive_only_ablation"])
        self.assertIn("--allow-positive-only-ablation", str(caught.exception))
        json.dumps(audit)

    def test_explicit_bounded_positive_only_ablation(self):
        audit = audit_partial_supervision(
            [partial_row()], allow_positive_only_ablation=True, max_steps=2
        )
        self.assertEqual(audit["decision"], "allowed_positive_only_ablation")
        self.assertTrue(audit["positive_only_ablation"])
        self.assertEqual(audit["ablation_max_steps"], 2)
        self.assertTrue(audit["warnings"])
        with self.assertRaises(SupervisionGuardError):
            audit_partial_supervision([partial_row()], max_steps=2)

    def test_override_requires_a_positive_integer_step_budget(self):
        for budget in (-1, 0, None, True, 1.5, float("inf"), "2"):
            with self.subTest(budget=budget), self.assertRaisesRegex(
                SupervisionGuardError, "requires positive --max-steps"
            ):
                audit_partial_supervision(
                    [partial_row()], allow_positive_only_ablation=True, max_steps=budget
                )

    def test_valid_negatives_and_separate_provenance_pass_without_mutation(self):
        row = partial_row(negative=True)
        before = copy.deepcopy(row)
        audit = audit_partial_supervision([row])
        self.assertEqual(row, before)
        self.assertEqual(audit["decision"], "allowed")
        self.assertEqual(audit["positive_pairs"], 1)
        self.assertEqual(audit["negative_pairs"], 1)
        self.assertEqual(audit["unknown_pairs"], 4)
        self.assertEqual(audit["supervised_pairs"], 2)
        self.assertEqual(audit["candidate_pairs"], 6)
        self.assertEqual(audit["negative_ratio"], 0.5)
        self.assertEqual(audit["status_counts"][NEGATIVE], 1)
        self.assertFalse(audit["positive_only_ablation"])

    def test_sparse_negatives_warn_without_a_ratio_cutoff(self):
        # Approximately 0.1% negatives still pass; the ratio is descriptive.
        base = partial_row()
        rows = [{**base, "id": f"positive:{i}"} for i in range(999)]
        rows.append(partial_row(negative=True))
        audit = audit_partial_supervision(rows)
        self.assertEqual(audit["positive_pairs"], 1000)
        self.assertEqual(audit["negative_pairs"], 1)
        self.assertAlmostEqual(audit["negative_ratio"], 1 / 1001)
        self.assertEqual(audit["decision"], "allowed")
        self.assertIsNone(audit["minimum_negative_ratio"])
        self.assertTrue(audit["warnings"])

    def test_all_unknown_status_variants_stay_masked(self):
        row = partial_row(negative=True)
        unknown = [p for p, s in row["pair_supervision"].items() if s.startswith("UNKNOWN_")]
        for pid, status in zip(unknown, (
            "UNKNOWN_UNANNOTATED", "UNKNOWN_REVIEW", "UNKNOWN_EVIDENCE", "UNKNOWN_FUTURE_REASON"
        )):
            row["pair_supervision"][pid] = status
        reserialize(row)
        audit = audit_partial_supervision([row])
        self.assertEqual(audit["unknown_pairs"], 4)
        self.assertEqual(audit["unknown_loss_pairs"], 0)
        self.assertEqual(audit["negative_pairs"], 1)
        self.assertEqual(audit["negative_ratio"], 0.5)

    def test_count_pairs_not_relation_edges(self):
        row = partial_row(negative=True)
        decision = next(d for d in row["target"]["decisions"] if d["relations"])
        decision["relations"].append({
            **decision["relations"][0], "relation": "supports_diagnosis_of",
        })
        reserialize(row)
        audit = audit_partial_supervision([row])
        self.assertEqual(audit["positive_pairs"], 1)
        self.assertEqual(audit["negative_ratio"], 0.5)

    def test_negative_targets_must_be_actual_empty_lists(self):
        for relations in ({}, "", None, 0, [{"relation": "is_core_symptom_of", "evidence_span_ids": []}]):
            row = partial_row(negative=True)
            decision = next(d for d in row["target"]["decisions"]
                            if row["pair_supervision"][d["pair_id"]] == NEGATIVE)
            decision["relations"] = relations
            with self.subTest(relations=relations), self.assertRaises(SupervisionGuardError):
                audit_partial_supervision([row], allow_positive_only_ablation=True, max_steps=2)

    def test_positive_and_unknown_targets_must_match_status(self):
        for status in (POSITIVE, "UNKNOWN_UNANNOTATED"):
            row = partial_row(negative=True)
            decision = next(d for d in row["target"]["decisions"]
                            if row["pair_supervision"][d["pair_id"]] == status)
            decision["relations"] = [] if status == POSITIVE else [{
                "relation": "is_core_symptom_of", "evidence_span_ids": [row["spans"][0]["span_id"]],
            }]
            with self.subTest(status=status), self.assertRaises(SupervisionGuardError):
                audit_partial_supervision([row])

    def test_corrupt_masks_and_contract_cannot_be_overridden(self):
        base = partial_row(negative=True)
        variants = [
            {**base, "loss_mask_spans": []},
            {**base, "loss_mask_spans": [[0, len(base["completion"])]]},
            {**base, "loss_mask_contract": "unsupported"},
            {**base, "completion": base["completion"] + " "},
            {**base, "pair_supervision": {}},
        ]
        for row in variants:
            with self.subTest(keys=row.keys()), self.assertRaises(SupervisionGuardError) as caught:
                audit_partial_supervision([row], allow_positive_only_ablation=True, max_steps=2)
            self.assertEqual(caught.exception.audit["invalid_rows"], 1)
            self.assertFalse(caught.exception.audit["positive_only_ablation"])

    def test_lost_mask_markers_cannot_downgrade_v2_to_legacy(self):
        row = legacy_row()
        row["data_version"] = "teacher-silver-repair-v2-partial-supervision"
        with self.assertRaises(SupervisionGuardError):
            audit_partial_supervision([row])

    def test_unknown_or_malformed_status_is_not_a_negative(self):
        for status in ("NEGATIVE", "UNSUPPORTED", None, [], 1):
            row = partial_row(negative=True)
            pid = next(p for p, value in row["pair_supervision"].items() if value == NEGATIVE)
            row["pair_supervision"][pid] = status
            with self.subTest(status=status), self.assertRaises(SupervisionGuardError):
                audit_partial_supervision([row], allow_positive_only_ablation=True, max_steps=2)

    def test_all_unknown_row_is_rejected_even_with_override(self):
        row = partial_row()
        row["pair_supervision"] = {pid: "UNKNOWN_REVIEW" for pid in row["pair_supervision"]}
        for decision in row["target"]["decisions"]:
            decision["relations"] = []
        reserialize(row)
        with self.assertRaisesRegex(SupervisionGuardError, "all-UNKNOWN"):
            audit_partial_supervision([row], allow_positive_only_ablation=True, max_steps=2)

    def test_zero_or_invalid_weights_cannot_supply_nominal_negatives(self):
        for weight in (0, -1, float("nan"), float("inf"), True, "1"):
            row = partial_row(negative=True)
            row["sample_weight"] = weight
            with self.subTest(weight=weight), self.assertRaisesRegex(
                SupervisionGuardError, "sample_weight must be finite and positive"
            ):
                audit_partial_supervision([row])
        row = partial_row(negative=True)
        row["sample_weight"] = 0.5
        self.assertEqual(audit_partial_supervision([row])["negative_pairs"], 1)

    def test_mixed_formats_fail_but_legacy_only_is_unchanged(self):
        row = legacy_row()
        self.assertEqual(audit_partial_supervision([row])["decision"], "not_applicable_legacy")
        with self.assertRaisesRegex(SupervisionGuardError, "Do not mix"):
            audit_partial_supervision([row, partial_row(negative=True)])

    def test_library_execute_cannot_skip_the_gate(self):
        manifest = {}
        # Deliberately no runtime/model options: admission must reject first.
        with self.assertRaises(SupervisionGuardError):
            train.execute(SimpleNamespace(mode="sft", max_steps=-1), [partial_row()], Path("unused"), manifest)
        self.assertEqual(manifest["supervision_audit"]["decision"], "rejected")


class OffsetTokenizer:
    eos_token_id = 99999
    eos_token = "EOS"
    pad_token = "PAD"

    def encode(self, text, add_special_tokens=False):
        return ([99998] if add_special_tokens else []) + [ord(c) + 1 for c in text]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        result = {"input_ids": self.encode(text, add_special_tokens)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def apply_chat_template(self, messages, **kwargs):
        return ("<|im_start|>user\n" + messages[0]["content"]
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


class MainAdmissionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="supervision-guard-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def run_main(self, row, *extra):
        (self.root / "data.jsonl").write_text(json.dumps(row) + "\n")
        with patch.object(sys, "argv", ["train", *argv(self.root, *extra)]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            train.main()

    def manifest(self):
        return json.loads((self.root / "out" / "run_manifest.json").read_text())

    def fake_runtime(self, tokenizer=None):
        tokenizer = tokenizer or OffsetTokenizer()
        forbidden = Mock(side_effect=AssertionError("Model loading / training / GPU access is forbidden in this test"))
        loader = SimpleNamespace(from_pretrained=forbidden)
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=forbidden)),
            "peft": SimpleNamespace(LoraConfig=forbidden, PeftModel=loader, get_peft_model=forbidden),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=loader,
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                Trainer=forbidden, TrainingArguments=forbidden, set_seed=lambda seed: None,
            ),
        }
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(train.importlib.metadata, "version", return_value="CPU_TEST"))
        self.addCleanup(forbidden.assert_not_called)
        return modules

    def test_rejection_is_persisted_before_any_ml_runtime_import(self):
        original_import = __import__

        def no_runtime(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "peft", "transformers"}:
                self.fail(f"Runtime imported before rejection: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_runtime), self.assertRaises(SupervisionGuardError):
            self.run_main(partial_row())
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["supervision_audit"]["negative_pairs"], 0)
        self.assertEqual(manifest["supervision_audit"]["unknown_loss_pairs"], 0)
        self.assertEqual(manifest["data_sha256"], hashlib.sha256((self.root / "data.jsonl").read_bytes()).hexdigest())

    def test_override_without_budget_is_rejected_with_manifest(self):
        with self.assertRaisesRegex(SupervisionGuardError, "requires positive --max-steps"):
            self.run_main(partial_row(), "--allow-positive-only-ablation")
        self.assertEqual(self.manifest()["supervision_audit"]["decision"], "rejected")

    def test_invalid_mask_failure_keeps_audit_in_manifest(self):
        row = partial_row(negative=True)
        row["loss_mask_spans"] = [[0, len(row["completion"])]]
        with self.assertRaises(SupervisionGuardError):
            self.run_main(row)
        audit = self.manifest()["supervision_audit"]
        self.assertEqual(audit["invalid_rows"], 1)
        self.assertEqual(audit["decision"], "rejected")

    def test_valid_partial_preflight_uses_existing_dataset_without_loading_model(self):
        modules = self.fake_runtime()
        self.run_main(partial_row(negative=True), "--preflight-only")
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "cpu_preflight_passed_no_model_loaded")
        audit = manifest["supervision_audit"]
        self.assertEqual(audit["decision"], "allowed")
        self.assertEqual(audit["tokenizer_validation"], "passed_JsonlSFTDataset")
        modules["transformers"].AutoTokenizer.from_pretrained.assert_called_once_with(
            str(self.root / "never-load-base"), local_files_only=True
        )

    def test_positive_only_preflight_records_explicit_ablation_and_budget(self):
        self.fake_runtime()
        self.run_main(partial_row(), "--preflight-only", "--allow-positive-only-ablation", "--max-steps", "2")
        manifest = self.manifest()
        self.assertTrue(manifest["arguments"]["allow_positive_only_ablation"])
        audit = manifest["supervision_audit"]
        self.assertTrue(audit["positive_only_ablation"])
        self.assertEqual(audit["ablation_max_steps"], 2)
        self.assertEqual(audit["tokenizer_validation"], "passed_JsonlSFTDataset")

    def test_tokenizer_mask_failure_precedes_model_loading(self):
        class NoOffsets(OffsetTokenizer):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                result.pop("offset_mapping", None)
                return result

        self.fake_runtime(NoOffsets())
        with self.assertRaisesRegex(ValueError, "valid completion offsets"):
            self.run_main(partial_row(negative=True))
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["supervision_audit"]["decision"], "rejected")
        self.assertEqual(manifest["supervision_audit"]["tokenizer_validation"], "failed_JsonlSFTDataset")

    def test_preflight_only_does_not_implicitly_allow_positive_only(self):
        modules = self.fake_runtime()
        with self.assertRaises(SupervisionGuardError):
            self.run_main(partial_row(), "--preflight-only")
        modules["transformers"].AutoTokenizer.from_pretrained.assert_not_called()

    def test_ablation_switch_cannot_enable_rl(self):
        for mode in ("dpo", "grpo"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(argv(self.root, "--mode", mode, "--allow-positive-only-ablation"))

    def test_legacy_cli_and_preflight_require_no_new_options(self):
        self.fake_runtime()
        args = train.parse_args(argv(self.root))
        self.assertFalse(args.allow_positive_only_ablation)
        self.assertEqual(args.max_steps, -1)
        self.assertEqual(args.epochs, 3)
        self.run_main(legacy_row(), "--preflight-only")
        self.assertEqual(self.manifest()["supervision_audit"]["decision"], "not_applicable_legacy")

    def test_legacy_and_partial_heldout_rejection_creates_no_output_directory(self):
        for factory in (legacy_row, partial_row):
            for split in ("tune", "test", None):
                row = factory()
                row["split"] = split
                with self.subTest(format=factory.__name__, split=split), self.assertRaisesRegex(ValueError, "train-tagged"):
                    self.run_main(row)
                self.assertFalse((self.root / "out").exists())

    def test_invalid_legacy_train_row_creates_no_output_directory(self):
        row = legacy_row()
        row.pop("id")
        with self.assertRaisesRegex(ValueError, "Missing/invalid id"):
            self.run_main(row)
        self.assertFalse((self.root / "out").exists())

    def test_partial_rl_rejection_creates_no_output_directory(self):
        for mode in ("dpo", "grpo"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "not eligible"):
                self.run_main(partial_row(), "--mode", mode, "--sft-adapter", "must-not-load")
            self.assertFalse((self.root / "out").exists())

    def test_existing_output_is_not_overwritten_even_on_rejection(self):
        (self.root / "out").mkdir()
        manifest_path = self.root / "out" / "run_manifest.json"
        manifest_path.write_text("existing result")
        with self.assertRaisesRegex(ValueError, "never overwrite"):
            self.run_main(partial_row())
        self.assertEqual(manifest_path.read_text(), "existing result")


if __name__ == "__main__":
    unittest.main()

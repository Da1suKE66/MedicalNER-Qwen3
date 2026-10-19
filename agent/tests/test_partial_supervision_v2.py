"""No private source data or GPU needed for these safety contracts."""

import copy
import json
import sys
from collections import Counter
from types import SimpleNamespace

import pytest

from data_repair.supervision_v2 import (
    LEGACY_EVIDENCE_POLICY, UNIQUE_EVIDENCE_POLICY,
    inject_contract_negative, masked_token_labels, partial_supervision,
    serialize_partial, validate_partial_row,
)
from generative.protocol import ProtocolConfig, build_batches
from robustness.entity_noise import prompt_payload, protocol_prefix
from generative.protocol import dumps
from generative.train import validate_training_rows


def example(context="full"):
    text = "Feature A is a core symptom of Condition B. Condition C is also mentioned."
    nodes = [{"id": "S", "name": "Feature A", "label": "Symptom"},
             {"id": "D", "name": "Condition B", "label": "Disease"},
             {"id": "X", "name": "Condition C", "label": "Disease"}]
    graph = {"entities": nodes, "relations": [{"source": "S", "target": "D",
             "relation": "is_core_symptom_of", "evidence": text.split('. ')[0] + '.'}]}
    batches, doc, _ = build_batches(text, nodes, ProtocolConfig(context=context), sample_id=1)
    row = {**batches[0], "split": "train", "group_id": 1, "sample_weight": 1.0}
    partial, _ = partial_supervision(row, graph, doc)
    return {**row, **partial}, graph, doc


class Tokenizer:
    eos_token_id = 99999
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kwargs):
        value = {"input_ids": [ord(c) + 1 for c in text]}
        if add_special_tokens:
            value["input_ids"].insert(0, 99998)
        if kwargs.get("return_offsets_mapping"):
            value["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return value


def test_unknown_not_negative_input_unchanged_and_known_pair_survives():
    row, _, _ = example()
    assert validate_partial_row(row)
    assert sum(v == "SUPPORTED_SILVER" for v in row["pair_supervision"].values()) == 1
    assert sum(v == "UNKNOWN_UNANNOTATED" for v in row["pair_supervision"].values()) == 5
    assert len(json.loads(row["prompt"].strip().splitlines()[-1])["pairs"]) == 6
    ids, labels = masked_token_labels(row, Tokenizer())
    for start, end in row["loss_mask_spans"]:
        assert all(v == -100 for v in labels[start:end])
    assert any(v != -100 for v in labels)
    assert len(ids) == len(labels)


def test_review_blocks_only_its_pair_without_turning_it_into_none():
    row, graph, doc = example()
    graph["relations"].append({"source": "D", "target": "X", "relation": "subtype_of", "evidence": "Absent."})
    part, stats = partial_supervision(row, graph, doc)
    assert stats["SUPPORTED_SILVER"] == 1 and stats["UNKNOWN_EVIDENCE"] == 1
    assert validate_partial_row({**row, **part})
    blocked, stats = partial_supervision(row, graph, doc, blocked_pairs=[("S", "D")])
    assert stats.get("SUPPORTED_SILVER", 0) == 0
    with pytest.raises(ValueError, match="all-UNKNOWN"):
        validate_partial_row({**row, **blocked})


def test_multilabel_and_duplicate_evidence_preserved():
    row, graph, doc = example()
    edge = graph["relations"][0]
    graph["relations"] += [copy.deepcopy(edge), {**edge, "relation": "supports_diagnosis_of"}]
    part, stats = partial_supervision(row, graph, doc)
    assert stats["positive_edges"] == 2 and stats["multilabel_pairs"] == 1
    assert validate_partial_row({**row, **part})


def test_any_missing_fragment_masks_whole_pair_not_other_pairs():
    row, graph, doc = example()
    graph["relations"][0]["evidence"] = [graph["relations"][0]["evidence"], "Absent fragment."]
    part, stats = partial_supervision(row, graph, doc)
    assert stats["UNKNOWN_EVIDENCE"] == 1 and stats["positive_edges"] == 0


def test_tampered_mask_and_rl_are_rejected():
    row, _, _ = example()
    validate_training_rows([row], "sft")
    for mode in ("dpo", "grpo"):
        with pytest.raises(ValueError, match="not eligible"):
            validate_training_rows([row], mode)
    broken = {**row, "loss_mask_spans": []}
    with pytest.raises(ValueError, match="tampered"):
        validate_partial_row(broken)


def test_token_straddling_unknown_boundary_is_fully_masked():
    row, _, _ = example()
    start, end = row["loss_mask_spans"][0]
    class Merged(Tokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [100, 101], "offset_mapping": [(0, 1), (max(0, start-1), end)]}
    _, labels = masked_token_labels(row, Merged())
    assert labels == [100, -100]


def test_dataset_masks_prompt_unknown_and_padding_and_keeps_eos(monkeypatch):
    from train_sft import JsonlSFTDataset, CausalCollator
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda x, **kw: x, long="long"))
    row, _, _ = example()
    tokenizer = Tokenizer()
    ds = JsonlSFTDataset([row], tokenizer, 20000)
    item = ds[0]
    prompt_len = len(tokenizer(row["prompt"], add_special_tokens=True)["input_ids"])
    assert all(v == -100 for v in item["labels"][:prompt_len])
    for a, b in row["loss_mask_spans"]:
        assert all(v == -100 for v in item["labels"][prompt_len+a:prompt_len+b])
    assert item["labels"][-1] == tokenizer.eos_token_id
    short = {k: v[:-1] for k,v in item.items() if k in {"input_ids", "labels"}}
    batch = CausalCollator(tokenizer)([item, short])
    assert batch["labels"][1][-1] == -100 and batch["attention_mask"][1][-1] == 0


def test_synthetic_negative_preserves_unknown_mask_and_positive():
    row, _, doc = example()
    augmented = inject_contract_negative(row, doc.raw)
    assert validate_partial_row(augmented)
    assert sum(v == "SYNTHETIC_NEGATIVE" for v in augmented["pair_supervision"].values()) == 2
    after = {(p["source"], p["target"]): augmented["pair_supervision"][p["pair_id"]]
             for p in augmented["pairs"]}
    for pair in row["pairs"]:
        assert after[(pair["source"], pair["target"])] == row["pair_supervision"][pair["pair_id"]]
    assert augmented["eligible_for_rl"] is False


def test_partial_exposure_counts_do_not_misreport_masked_completion_as_prompt():
    from generative.train import PairCountCollator, record_sft_batch
    class Mask:
        def __init__(self, count): self.count = count
        def sum(self): return SimpleNamespace(item=lambda: self.count)
        def __ne__(self, other): return self
    meta = {"prompt_tokens": 10, "completion_tokens": 8, "masked_completion_tokens": 3,
            "supervised_pairs": 1, "unknown_pairs": 2}
    def delegate(items):
        assert all(set(item) == {"input_ids", "labels"} for item in items)
        return {"input_ids": [[1]], "attention_mask": Mask(18), "labels": Mask(5)}
    batch = PairCountCollator(delegate)([{"input_ids": [1], "labels": [1],
                                         "_pair_count": 3, "_supervision_counts": meta}])
    counts, metadata = batch.pop("_pair_counts"), batch.pop("_supervision_counts_batch")
    counter = Counter()
    record_sft_batch(counter, batch, counts, metadata)
    assert counter["sft_pair_loss_exposures"] == 1
    assert counter["sft_candidate_pair_context_exposures"] == 3
    assert counter["sft_prompt_token_loss_exposures"] == 10
    assert counter["sft_masked_completion_token_context_exposures"] == 3
    with pytest.raises(ValueError, match="does not match"):
        record_sft_batch(Counter(), batch, counts, [{**meta, "masked_completion_tokens": 2}])


def test_partial_data_cannot_become_omission_preferences_or_drop_its_mask():
    from generative.build_preferences import omission_preferences
    row, _, _ = example()
    with pytest.raises(ValueError, match="not eligible"):
        list(omission_preferences([row]))
    row["data_version"] = "teacher-silver-repair-v2-partial-supervision"
    for key in ("loss_mask_contract", "loss_mask_spans", "pair_supervision"):
        row.pop(key)
    with pytest.raises(ValueError, match="loss-mask contract"):
        validate_training_rows([row], "sft")


def test_gold_metadata_in_prompt_is_rejected_by_dataset_and_training_gate(monkeypatch):
    from train_sft import JsonlSFTDataset
    row, _, _ = example()
    before = row["prompt"].splitlines()
    payload = json.loads(before[-1])
    payload["pair_supervision"] = row["pair_supervision"]
    row["prompt"] = before[0] + "\n" + json.dumps(payload, separators=(",", ":")) + "\n"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_training_rows([row], "sft")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(ValueError, match="unexpected fields"):
        JsonlSFTDataset([row], Tokenizer(), 20000)


def test_exact_chat_rendering_accepted_but_metadata_in_prose_is_not():
    row, _, _ = example()
    raw = row["prompt"]
    row["prompt"] = "<|im_start|>user\n" + raw + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert validate_partial_row(row)
    row["prompt"] = "gold labels are ...\n" + raw
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_partial_row(row)


@pytest.mark.parametrize("mutation, expected", [
    (lambda r: next(e for e in r["entities"] if e["id"] == "S").update(label="Medication"),
     "Stale allowed_relations"),
    (lambda r: r["entities"][0].update(label="not-an-entity-type"), "Noncanonical"),
    (lambda r: r["entities"][0].update(name=""), "Empty entity name"),
    (lambda r: r["entities"][1].update(entity_id=r["entities"][0]["entity_id"]), "local entity ID"),
    (lambda r: r["pairs"][0]["allowed"].append("invented_relation"), "Stale allowed_relations"),
    (lambda r: r["pairs"][0].update(target=r["pairs"][0]["source"]), "endpoints"),
    (lambda r: r["pairs"][1].update(source=r["pairs"][0]["source"], target=r["pairs"][0]["target"]),
     "Duplicate directed pair"),
    (lambda r: r["spans"].append(copy.deepcopy(r["spans"][0])), "Duplicate source span"),
])
def test_input_tables_rejected_even_when_prompt_is_synchronized(mutation, expected):
    row, _, _ = example()
    mutation(row)
    row["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(row)) + "\n"
    with pytest.raises(ValueError, match=expected):
        validate_partial_row(row)


def ambiguous_example():
    text = ("Feature A is described. Condition B is described. Feature A is described. "
            "Condition C is a subtype of Condition B.")
    nodes = [{"id": "S", "name": "Feature A", "label": "Symptom"},
             {"id": "D", "name": "Condition B", "label": "Disease"},
             {"id": "X", "name": "Condition C", "label": "Disease"}]
    graph = {"entities": nodes, "relations": [
        {"source": "S", "target": "D", "relation": "is_core_symptom_of", "evidence": "Feature A is described."},
        {"source": "X", "target": "D", "relation": "subtype_of", "evidence": "Condition C is a subtype of Condition B."}]}
    batches, doc, _ = build_batches(text, nodes, ProtocolConfig(), sample_id=1)
    return batches[0], graph, doc


def test_ambiguous_occurrence_masks_whole_multilabel_pair_and_preserves_other_pair():
    batch, graph, doc = ambiguous_example()
    graph["relations"].insert(0, {"source": "S", "target": "D", "relation": "supports_diagnosis_of",
                                  "evidence": doc.spans[0].id})
    before = copy.deepcopy((batch, graph, doc))
    partial, stats = partial_supervision(batch, graph, doc)
    pid = next(p["pair_id"] for p in batch["pairs"] if (p["source"], p["target"]) == ("S", "D"))
    assert partial["pair_supervision"][pid] == "UNKNOWN_EVIDENCE"
    assert partial["unknown_reasons"][pid] == "evidence_occurrence_ambiguous"
    assert next(d for d in partial["target"]["decisions"] if d["pair_id"] == pid)["relations"] == []
    assert stats["positive_edges"] == stats["SUPPORTED_SILVER"] == 1
    assert partial["evidence_grounding_policy"] == UNIQUE_EVIDENCE_POLICY
    row = {**batch, **partial}
    assert validate_partial_row(row)
    _, labels = masked_token_labels(row, Tokenizer())
    for a, b in row["loss_mask_spans"]:
        assert all(label == -100 for label in labels[a:b])
    assert (batch, graph, doc) == before
    legacy, legacy_stats = partial_supervision(batch, graph, doc, evidence_policy=LEGACY_EVIDENCE_POLICY)
    assert legacy_stats["positive_edges"] == 3
    assert "evidence_grounding_policy" not in legacy
    # Reading old rows neither rejects nor silently rewrites their old targets.
    assert validate_partial_row({**batch, **legacy})


def test_local_context_cannot_make_globally_ambiguous_evidence_unique():
    batch, graph, doc = ambiguous_example()
    batch["spans"] = [s for s in batch["spans"] if s["span_id"] != doc.spans[2].id]
    partial, _ = partial_supervision(batch, graph, doc)
    pid = next(p["pair_id"] for p in batch["pairs"] if (p["source"], p["target"]) == ("S", "D"))
    assert partial["unknown_reasons"][pid] == "evidence_occurrence_ambiguous"


def test_explicit_span_id_disambiguates_repeated_text():
    batch, graph, doc = ambiguous_example()
    graph["relations"][0]["evidence"] = doc.spans[0].id
    partial, stats = partial_supervision(batch, graph, doc)
    assert stats["SUPPORTED_SILVER"] == 2
    assert validate_partial_row({**batch, **partial})


def test_unknown_evidence_policy_is_rejected():
    batch, graph, doc = ambiguous_example()
    with pytest.raises(ValueError, match="evidence grounding policy"):
        partial_supervision(batch, graph, doc, evidence_policy="typo")

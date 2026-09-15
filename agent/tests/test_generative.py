"""CPU contract/reward tests; these are not model accuracy experiments."""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generative.protocol import (
    ProtocolConfig,
    build_batches,
    dumps,
    original_offset,
    parse_decisions,
    repair_text,
    reward_components,
    supervision,
)


TEXT = "Fatigue is a core symptom of Condition A and supports diagnosis of Condition A."
NODES = [
    {"id": "S1", "label": "Symptom", "name": "Fatigue"},
    {"id": "D1", "label": "Disease", "name": "Condition A"},
]
GOLD = {
    "entities": NODES,
    "relations": [
        {"source": "S1", "target": "D1", "relation": r, "evidence": TEXT}
        for r in ("is_core_symptom_of", "supports_diagnosis_of")
    ],
}


def example(config=None):
    config = config or ProtocolConfig(strict_evidence=True)
    batches, doc, _ = build_batches(TEXT, NODES, config)
    target, stats = supervision(batches[0], GOLD, doc, config)
    return batches[0], doc, target, stats


def test_generation_retains_multiple_relations():
    batch, _, target, stats = example()
    parsed, errors = parse_decisions(dumps(target), batch)
    assert not errors and len(parsed) == 2 and stats["multi_relation_pairs"] == 1


def test_no_gold_in_prompt_or_id_semantics():
    c = ProtocolConfig()
    a, _, _ = build_batches(TEXT, NODES, c)
    renamed = [{**n, "id": f"ARBITRARY_{i}"} for i, n in enumerate(NODES)]
    b, _, _ = build_batches(TEXT, renamed, c)
    assert a[0]["prompt"] == b[0]["prompt"]
    assert "Fatigue" in a[0]["prompt"] and "Condition A" in a[0]["prompt"]


def test_output_shortening_does_not_drop_candidates():
    nodes = [
        {"id": str(i), "name": f"Disease {i}", "label": "Disease"} for i in range(14)
    ]
    text = ". ".join(n["name"] for n in nodes)
    for size in (4, 16):
        batches, _, audit = build_batches(
            text, nodes, ProtocolConfig(pairs_per_batch=size)
        )
        assert sum(len(b["pairs"]) for b in batches) == 182
        assert audit["candidate_pairs"] == 182  # not the old 128 prefix


def test_cross_sentence_endpoint_not_dropped():
    text = "Problems are present. They persist daily."
    nodes = [
        {"id": "S", "name": "present. They persist", "label": "Symptom"},
        {"id": "D", "name": "Problems", "label": "Disease"},
    ]
    batches, _, audit = build_batches(text, nodes, ProtocolConfig(context="local"))
    assert not audit["ungrounded_ids"] and len(batches[0]["pairs"]) == 2


def test_unknown_annotation_never_becomes_none():
    batch, doc, _, _ = example()
    bad = copy.deepcopy(GOLD)
    bad["relations"][0]["evidence"] = "This sentence never existed."
    target, stats = supervision(batch, bad, doc, ProtocolConfig(strict_evidence=True))
    assert target is None and stats["unknown_batches"] == 1


def test_invalid_json_and_missing_decisions_are_visible():
    batch, _, target, _ = example()
    assert parse_decisions("<think>guess</think>" + dumps(target), batch)[1] == [
        "invalid_json"
    ]
    assert "missing_pair_decisions" in parse_decisions('{"decisions":[]}', batch)[1]


def test_unknown_evidence_is_not_silently_accepted():
    batch, _, target, _ = example()
    bad = copy.deepcopy(target)
    next(d for d in bad["decisions"] if d["relations"])["relations"][0][
        "evidence_span_ids"
    ] = ["SENT_999999"]
    assert "invalid_evidence" in parse_decisions(dumps(bad), batch)[1]


def test_duplicate_pair_and_label_are_errors():
    batch, _, target, _ = example()
    bad = copy.deepcopy(target)
    bad["decisions"].append(bad["decisions"][0])
    assert "duplicate_pair" in parse_decisions(dumps(bad), batch)[1]
    bad = copy.deepcopy(target)
    d = next(d for d in bad["decisions"] if d["relations"])
    d["relations"].append(d["relations"][0])
    assert "duplicate_relation" in parse_decisions(dumps(bad), batch)[1]
    from generative.evaluate import assemble_decisions
    from evaluate import _relation_set

    edges, _ = assemble_decisions(batch, dumps(bad))
    assert _relation_set({"entities": NODES, "relations": edges}) == _relation_set(GOLD)


def test_nonhashable_values_do_not_crash_parser():
    batch, _, target, _ = example()
    bad = copy.deepcopy(target)
    bad["decisions"][0]["pair_id"] = []
    assert parse_decisions(dumps(bad), batch)[1]


def test_empty_reward_cannot_beat_positive_answer():
    batch, _, target, _ = example()
    empty = {
        "decisions": [
            {"pair_id": p["pair_id"], "relations": []} for p in batch["pairs"]
        ]
    }
    good = reward_components(dumps(target), batch, target)
    abstain = reward_components(dumps(empty), batch, target)
    assert good["reward"] == 1.0
    assert abstain["reward"] < 0 and abstain["empty_on_positive"]


def test_false_positive_reward_penalty():
    batch, _, target, _ = example()
    bad = copy.deepcopy(target)
    d = next(d for d in bad["decisions"] if d["relations"])
    d["relations"].append(
        {"relation": "is_associated_symptom_of", "evidence_span_ids": ["SENT_001"]}
    )
    score = reward_components(dumps(bad), batch, target)
    assert score["fp"] == 1 and score["reward"] < 1


def test_evidence_reward_is_ablatable_without_changing_graph_targets():
    batch, _, target, _ = example()
    a = reward_components(dumps(target), batch, target)
    b = reward_components(dumps(target), batch, target, evidence_weight=0)
    assert a["tp"] == b["tp"] == 2 and abs(a["reward"] - b["reward"] - 0.2) < 1e-9


def test_text_repair_is_whitelisted_and_reversible_for_evidence():
    raw = "Pregnullcy and finullces are noted. Null and nullable stay unchanged."
    clean, changes = repair_text(raw)
    assert (
        clean == "Pregnancy and finances are noted. Null and nullable stay unchanged."
    )
    assert original_offset(len(clean), changes, end=True) == len(raw)
    nodes = [
        {"id": "S", "name": "Pregnullcy", "label": "Symptom"},
        {"id": "D", "name": "finullces", "label": "Disease"},
    ]
    batches, _, _ = build_batches(raw, nodes, ProtocolConfig(repair=True))
    assert (
        batches[0]["original_spans"]["SENT_001"]
        == "Pregnullcy and finullces are noted."
    )
    assert "Pregnancy" in batches[0]["prompt"]


def test_single_label_is_explicit_diagnostic_not_silent_dict_overwrite():
    _, _, _, stats = example(ProtocolConfig(multilabel=False))
    assert (
        stats["positive_edges"] == 1
        and stats["intentionally_dropped_extra_labels"] == 1
    )


def test_synthetic_dpo_pairs_have_different_answers_and_reason_codes():
    from generative.build_preferences import synthetic_preferences

    rows = list(synthetic_preferences([2, 7]))
    assert len(rows) == 8
    assert all(r["chosen"] != r["rejected"] and r["split"] == "train" for r in rows)
    assert len({r["reason"] for r in rows}) == 4
    assert all(r["hard_semantic"] for r in rows)


def test_invalid_generated_edge_is_counted_as_fp():
    from generative.evaluate import assemble_decisions
    from evaluate import _relation_set

    batch, _, target, _ = example()
    bad = copy.deepcopy(target)
    next(d for d in bad["decisions"] if d["relations"])["relations"][0][
        "evidence_span_ids"
    ] = ["MADE_UP"]
    edges, errors = assemble_decisions(batch, dumps(bad))
    keys = _relation_set({"entities": NODES, "relations": edges})
    assert "invalid_evidence" in errors
    assert len(keys - _relation_set(GOLD)) == 1


def test_full_document_eval_keeps_missed_candidates_as_fn():
    from generative.evaluate import run_records
    from evaluate_graphs import evaluate_records

    record = {
        "messages": [
            {"role": "user", "content": TEXT},
            {"role": "assistant", "content": dumps(GOLD)},
        ]
    }
    cached = [{"output": {"entities": NODES[:1], "relations": []}}]
    predictions, _, audit = run_records(
        [record], cached, ProtocolConfig(), lambda b: {"text": ""}
    )
    metric = evaluate_records([record], predictions)["free_text"]["relation"]
    assert (
        metric["fn"] == 2
        and metric["tp"] == 0
        and audit["candidate_covered_triples"] == 0
    )


def test_full_document_eval_does_not_accept_missing_cache_rows():
    import pytest
    from generative.evaluate import run_records

    with pytest.raises(ValueError, match="length mismatch"):
        run_records([{}], [], ProtocolConfig(), lambda b: {})


def test_local_evidence_gate_counts_unknown_batch():
    text = (
        TEXT
        + " "
        + " ".join(f"Unrelated observation number {i}." for i in range(12))
        + " This distant sentence supplies a diagnostic condition."
    )
    config = ProtocolConfig(context="local", strict_evidence=True)
    batches, doc, _ = build_batches(text, NODES, config)
    gold = copy.deepcopy(GOLD)
    gold["relations"][0][
        "evidence"
    ] = "This distant sentence supplies a diagnostic condition."
    target, stats = supervision(batches[0], gold, doc, config)
    assert target is None and stats["unknown_batches"] == 1


def test_batched_generation_alignment_is_required():
    import pytest
    from generative.evaluate import run_records

    record = {
        "messages": [
            {"role": "user", "content": TEXT},
            {"role": "assistant", "content": dumps(GOLD)},
        ]
    }
    with pytest.raises(ValueError, match="Generation batch alignment"):
        run_records(
            [record],
            None,
            ProtocolConfig(),
            None,
            oracle=True,
            generate_many=lambda batches: [],
        )


def test_checkpoint_selection_uses_tune_f05_not_loss_or_empty_precision():
    from generative.run_experiment import select_checkpoint
    from pair_metrics import counts_metric

    candidates = [
        {"step": 1, "checkpoint": "a", "relation": counts_metric(0, 0, 100)},
        {"step": 2, "checkpoint": "b", "relation": counts_metric(30, 10, 70)},
        {"step": 3, "checkpoint": "c", "relation": counts_metric(60, 20, 40)},
    ]
    result = select_checkpoint(candidates)
    assert result["checkpoint"] == "c" and result["eligibility_met"]


def test_chat_renderer_disables_thinking_and_prevents_double_wrapping():
    import pytest
    from generative.chat import render_prompt

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            assert messages == [{"role": "user", "content": "payload"}]
            return "<|im_start|>user\npayload<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    text = render_prompt(Tokenizer(), "payload")
    with pytest.raises(ValueError, match="already chat-rendered"):
        render_prompt(Tokenizer(), text)

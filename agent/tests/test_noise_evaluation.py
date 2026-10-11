import copy

import pytest

from generative.protocol import ProtocolConfig, build_batches, dumps, supervision
from robustness.entity_noise import NoiseConfig, augment_heldout_row
from robustness.noise_evaluation import summarize


def paired():
    text = "Fatigue is a core symptom of Condition Alpha. Sweating is a core symptom of Condition Alpha."
    nodes = [{"id": "D", "name": "Condition Alpha", "label": "Disease"}, {"id": "S", "name": "Fatigue", "label": "Symptom"}, {"id": "T", "name": "Sweating", "label": "Symptom"}]
    graph = {"entities": nodes, "relations": [{"source": s, "target": "D", "relation": "is_core_symptom_of", "evidence": text} for s in ("S", "T")]}
    config = ProtocolConfig(strict_evidence=True)
    batches, doc, _ = build_batches(text, nodes, config, sample_id=7)
    target, _ = supervision(batches[0], graph, doc, config)
    clean = {**batches[0], "target": target, "completion": dumps(target), "split": "tune", "group_id": 77, "sample_weight": 1.0}
    return {"clean": clean, "noisy": augment_heldout_row(clean, split="tune", config=NoiseConfig(seed=17))}


def test_gold_has_perfect_score_and_zero_noise_edges():
    pair = paired()
    results = summarize([pair], [{"id": pair[c]["id"], "text": pair[c]["completion"]} for c in ("clean", "noisy")])
    assert results["clean"]["f1"] == results["noisy"]["f1"] == 1
    assert results["noise_incident"]["false_relation_pair_rate"] == 0
    assert results["nonincident"]["noisy_recall"] == 1


def test_empty_output_cannot_pass_the_positive_retention_guard():
    pair = paired()
    generations = []
    for condition in ("clean", "noisy"):
        target = copy.deepcopy(pair[condition]["target"])
        for d in target["decisions"]:
            d["relations"] = []
        generations.append({"id": pair[condition]["id"], "text": dumps(target)})
    result = summarize([pair], generations)
    assert result["noise_incident"]["false_relation_pair_rate"] == 0
    assert result["noisy"]["recall"] == 0 and result["noisy"]["empty_on_positive"] == 1


def test_partial_generation_is_not_full_stress_result():
    with pytest.raises(ValueError, match="Incomplete"):
        summarize([paired()], [])


@pytest.mark.parametrize("corruption", ["reordered", "missing_id", "wrong_prompt"])
def test_same_length_stress_output_requires_declared_identity(corruption):
    from robustness.noise_evaluation import digest

    pair = paired()
    generations = [{"id": pair[c]["id"], "text": pair[c]["completion"],
                    "prompt_sha256": digest(pair[c]["prompt"].encode("utf-8"))}
                   for c in ("clean", "noisy")]
    assert summarize([pair], generations)["clean"]["f1"] == 1
    if corruption == "reordered":
        generations.reverse()
    elif corruption == "missing_id":
        generations[0].pop("id")
    else:
        generations[0]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="alignment mismatch"):
        summarize([pair], generations)

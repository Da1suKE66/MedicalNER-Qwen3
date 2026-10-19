"""Recipe-verified artificial preferences for semantic and grammar recovery.

No clinical, silver, tune or historical data is read. Absence is fully defined
inside each authored artificial exercise. Structural rejects are intentionally
invalid and require the explicit versioned training admission gate. This module
never relaxes UNKNOWN supervision or claims medical adjudication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .protocol import ProtocolConfig, build_batches, digest, dumps, parse_decisions, reward_components, supervision

CONTRACT = "fully-artificial-recovery-preferences-v1"
MORE = ("omit_one_edge", "omit_second_relation", "omit_all_supported", "empty_on_positive")
FEWER = ("add_unsupported_edge", "add_multiple_unsupported", "contradict_explicit_negation", "all_negative_hallucination")
EQUAL = ("wrong_target", "wrong_target_type", "reverse_direction", "wrong_evidence")
STRUCTURAL = ("terminal_bracket_reversal", "premature_outer_close", "duplicate_pair_id", "fabricated_pair_id")
KINDS = MORE + FEWER + EQUAL + STRUCTURAL
CORE, DIAGNOSTIC = "is_core_symptom_of", "supports_diagnosis_of"
PROVENANCE = "fully_artificial_exhaustive_assertions_no_medical_source"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recipe(index, kind, partition, pairs_per_batch, template):
    if (type(index) is not int or index < 0 or kind not in KINDS
            or partition not in {"train", "mechanical_holdout"}
            or type(pairs_per_batch) is not int or pairs_per_batch not in {2, 4, 8}
            or type(template) is not int or template not in range(6)):
        raise ValueError("Invalid artificial construction recipe")
    return dict(version=CONTRACT, index=index, kind=kind, partition=partition,
                pairs_per_batch=pairs_per_batch, template=template)


def _sentence(source, target, label, template):
    alternatives = {
        CORE: [f"{source} is a core symptom of {target}.",
               f"For {target}, the explicitly listed core symptom is {source}.",
               f"The artificial description of {target} identifies {source} as a core symptom."],
        DIAGNOSTIC: [f"{source} independently supports diagnosis of {target}.",
                     f"Diagnosis of {target} is explicitly supported by {source}.",
                     f"The exercise states that {source} provides diagnostic support for {target}."],
        "subtype_of": [f"{source} is a subtype of {target}.",
                       f"The subtype listed under {target} is {source}.",
                       f"In this exercise, {source} has the parent category {target} and is its subtype."],
    }
    return alternatives[label][template % 3]


def _construct(recipe):
    """Author every fact from a small recipe; arbitrary external text is impossible."""
    if set(recipe) != {"version", "index", "kind", "partition", "pairs_per_batch", "template"}:
        raise ValueError("Unexpected recipe fields; external source text is forbidden")
    if recipe["version"] != CONTRACT:
        raise ValueError("Unsupported recipe version")
    expected = _recipe(*(recipe[k] for k in ("index", "kind", "partition", "pairs_per_batch", "template")))
    if expected != recipe:
        raise ValueError("Noncanonical construction recipe")
    # JSON object key order is not semantic. Preserve the original recipe's
    # canonical construction order before deriving identities from its bytes.
    recipe = expected
    index, kind, partition, size, template = (recipe[k] for k in ("index", "kind", "partition", "pairs_per_batch", "template"))
    key = digest(dumps(recipe))[:12]
    names = {"S1": f"Feature Cedar {key}", "S2": f"Feature Birch {key}",
             "D1": f"Condition Amber {key}", "D2": f"Condition Indigo {key}"}
    nodes = [{"id": eid, "name": name, "label": "Symptom" if eid.startswith("S") else "Disease"}
             for eid, name in names.items()]
    if kind == "reverse_direction":
        candidates = [("D1", "D2"), ("D2", "D1")]
        primary, label = candidates[0], "subtype_of"
    else:
        candidates = [("S1", "D1"), ("S1", "S2") if kind == "wrong_target_type" else ("S1", "D2")]
        primary, label = candidates[0], CORE
    for pair in (("S2", "D2"), ("S2", "D1"), ("S1", "D2"), ("S1", "S2"),
                 ("D1", "D2"), ("D2", "D1"), ("D1", "S1"), ("D2", "S2")):
        if pair not in candidates:
            candidates.append(pair)
    candidates = candidates[:size]
    facts = []
    if kind not in {"contradict_explicit_negation", "all_negative_hallucination"}:
        facts.append((*primary, label))
        if label == CORE and (kind == "omit_second_relation" or index % 3 == 0):
            facts.append((*primary, DIAGNOSTIC))
        if size >= 4 and kind in {"omit_all_supported", "empty_on_positive", "omit_one_edge", "add_multiple_unsupported"}:
            facts.append(("S2", "D2", CORE))
    sentences = [_sentence(names[s], names[t], rel, template) for s, t, rel in facts]
    intro = [
        "This is an artificial exercise. The following assertion list is exhaustive for this fictional world.",
        "Read this fictional case. Every positive relationship is stated below; all other directed relationships in the candidate table are absent.",
        "Synthetic extraction task: the listed statements completely define the relationships, including the absence of any unlisted relationship.",
    ][template % 3]
    inventory = "The fictional entities are " + ", ".join(names.values()) + "."
    negative = (f"{names['S1']} is explicitly not a core symptom of {names['D1']}."
                if kind == "contradict_explicit_negation" else
                "No positive relation is asserted between any of the listed entities."
                if kind == "all_negative_hallucination" else
                "Shared appearance in the inventory does not assert any additional relationship.")
    body = sentences if template < 3 else list(reversed(sentences))
    if template % 2:
        text = intro + "\n" + inventory + "\n" + "\n".join(body) + "\n" + negative
    else:
        text = intro + " " + " ".join(body) + " " + inventory + " " + negative
    text += " No other directed relationships or additional labels are asserted in this exercise."
    graph = {"entities": nodes, "relations": [
        {"source": s, "target": t, "relation": rel, "evidence": sentence}
        for (s, t, rel), sentence in zip(facts, sentences)]}
    config = ProtocolConfig(strict_evidence=True, pairs_per_batch=size,
                            order="hash" if (index + template) % 2 else "source")
    batches, doc, audit = build_batches(text, nodes, config, sample_id=f"{partition}:{key}",
                                      selected_pairs=set(candidates))
    if len(batches) != 1 or len(batches[0]["pairs"]) != size or audit["ungrounded_ids"]:
        raise ValueError("Artificial source/candidate coverage drift")
    batch = batches[0]
    chosen, _ = supervision(batch, graph, doc, config)
    if chosen is None:
        raise ValueError("Constructed evidence does not ground")
    rejected = deepcopy(chosen)
    by_pair = {(p["source"], p["target"]): p for p in batch["pairs"]}
    by_id = {d["pair_id"]: d["relations"] for d in rejected["decisions"]}
    primary_id = by_pair[primary]["pair_id"]
    positive = by_id[primary_id]
    def add(pair, rel):
        pid = by_pair[pair]["pair_id"]
        evidence = next((e["evidence_span_ids"] for d in chosen["decisions"] for e in d["relations"]),
                        [batch["spans"][-1]["span_id"]])
        by_id[pid].append({"relation": rel, "evidence_span_ids": list(evidence)})
    if kind == "omit_one_edge":
        positive.pop(0)
    elif kind == "omit_second_relation":
        positive[:] = [e for e in positive if e["relation"] != DIAGNOSTIC]
    elif kind in {"omit_all_supported", "empty_on_positive"}:
        for value in by_id.values(): value.clear()
    elif kind in {"add_unsupported_edge", "contradict_explicit_negation", "all_negative_hallucination"}:
        add(primary if kind != "add_unsupported_edge" else candidates[1], CORE)
    elif kind == "add_multiple_unsupported":
        for pair in candidates:
            if not by_id[by_pair[pair]["pair_id"]]:
                add(pair, by_pair[pair]["allowed"][0])
    elif kind in {"wrong_target", "wrong_target_type", "reverse_direction"}:
        moved = positive.pop(0)
        by_id[by_pair[candidates[1]]["pair_id"]].append(moved)
    elif kind == "wrong_evidence":
        refs = positive[0]["evidence_span_ids"]
        positive[0]["evidence_span_ids"] = [next(s["span_id"] for s in batch["spans"] if s["span_id"] not in refs)]
    elif kind == "duplicate_pair_id":
        decision = next(d for d in rejected["decisions"] if d["pair_id"] != primary_id)
        decision["pair_id"] = primary_id
    elif kind == "fabricated_pair_id":
        rejected["decisions"][0]["pair_id"] = "P9999"
    chosen_text, rejected_text = dumps(chosen), dumps(rejected)
    if kind == "terminal_bracket_reversal":
        rejected_text = chosen_text[:-2] + "}]"
    elif kind == "premature_outer_close":
        rejected_text = chosen_text[:-1]
    row = {
        "id": f"{CONTRACT}:{partition}:{key}", "group_id": f"artificial:{partition}:{key}",
        "split": partition, "training_view": "preference", "preference_contract": CONTRACT,
        "construction_recipe": deepcopy(recipe), "prompt": batch["prompt"],
        "chosen": chosen_text, "rejected": rejected_text, "reason": kind,
        "human_adjudicated": False, "medical_source_records": 0,
        "provenance": PROVENANCE, "eligible_for_rl": True,
        "semantic_claim": "mechanical_extraction_only", "oracle": graph,
        "oracle_source_text": text, "candidate_pairs": [list(pair) for pair in candidates],
        "protocol_config": asdict(config),
    }
    row.update({f"{field}_sha256": digest(row[field]) for field in ("prompt", "chosen", "rejected")})
    return row, batch, chosen


def as_sft(row):
    """Retain the complete preference receipt, adding only its chosen SFT view."""
    from .train import prompt_batch
    result = deepcopy(row)
    payload = json.loads(row["prompt"].strip().splitlines()[-1])
    result.update(training_view="chosen_sft", completion=row["chosen"], target=json.loads(row["chosen"]),
                  pairs=prompt_batch(row)["pairs"], entities=payload["entities"], spans=payload["spans"], sample_weight=1.0)
    return result


def make_case(index, kind, *, partition="train", pairs_per_batch=4, template=0):
    return _construct(_recipe(index, kind, partition, pairs_per_batch, template))[0]


def verify_case(row):
    """Rebuild the entire row from a closed recipe, not caller-authored evidence.

    Unknown extra metadata fails exact equality, including all UNKNOWN/partial
    supervision fields. The legacy RL guard must run before this opt-in branch.
    """
    if not isinstance(row, dict) or row.get("preference_contract") != CONTRACT:
        raise ValueError("Unsupported artificial preference contract")
    expected, batch, target = _construct(row.get("construction_recipe", {}))
    if row.get("training_view") == "chosen_sft":
        expected = as_sft(expected)
    if row != expected:
        raise ValueError("Preference differs from deterministic artificial recipe reconstruction")
    chosen, errors = parse_decisions(row["chosen"], batch)
    rejected, bad = parse_decisions(row["rejected"], batch)
    if errors or row["chosen"] == row["rejected"]:
        raise ValueError("Invalid chosen or identical preference")
    if row["reason"] not in STRUCTURAL and (bad or chosen == rejected):
        raise ValueError("Semantic preference must contain distinct valid decisions")
    if row["reason"] in STRUCTURAL and not bad:
        raise ValueError("Structural rejected target must demonstrate an actual contract failure")
    a = reward_components(row["chosen"], batch, target)["reward"]
    b = reward_components(row["rejected"], batch, target)["reward"]
    if not a > b:
        raise ValueError("Preference direction fails the existing mechanical reward")
    # Structural-invalid texts are not counted as zero edges. These lexical
    # edge slots are exact generated fields, not a permissive repaired score.
    rejected_edges = len(re.findall(r'"relation":', row["rejected"]))
    return {"chosen_edges": len(chosen), "rejected_edges": rejected_edges,
            "rejected_edge_count_scope": "authored relation-field occurrences, including invalid-ID/JSON text",
            "expected_rejected_errors": bad, "reward_margin": a - b,
            "pairs_per_batch": len(batch["pairs"]), "multilabel_pairs": sum(len(d["relations"]) > 1 for d in target["decisions"]),
            "chosen_all_negative": not chosen}


def summarize(rows, tokenizer=None, max_length=3072):
    counts, lengths, errors, sizes, templates, kinds = (Counter() for _ in range(6))
    rendered = []
    for row in rows:
        detail = verify_case(row)
        delta = detail["chosen_edges"] - detail["rejected_edges"]
        counts["chosen_more" if delta > 0 else "chosen_fewer" if delta < 0 else "equal"] += 1
        counts["all_negative_chosen"] += detail["chosen_all_negative"]
        counts["multilabel_rows"] += bool(detail["multilabel_pairs"])
        counts["structural_rejected_rows"] += bool(detail["expected_rejected_errors"])
        errors.update(detail["expected_rejected_errors"])
        sizes[detail["pairs_per_batch"]] += 1
        templates[row["construction_recipe"]["template"]] += 1
        kinds[row["reason"]] += 1
        if tokenizer:
            from .chat import render_prompt
            a, b = (len(tokenizer.encode(row[key], add_special_tokens=False)) + 1 for key in ("chosen", "rejected"))
            lengths["chosen_longer" if a > b else "chosen_shorter" if a < b else "equal"] += 1
            rendered.append({**row, "prompt": render_prompt(tokenizer, row["prompt"])})
    result = {"rows": len(rows), "edge_count_direction": {k: counts[k] for k in ("chosen_more", "chosen_fewer", "equal")},
              "counts": dict(counts), "rejected_error_counts": dict(errors), "candidate_sizes": dict(sizes),
              "templates": dict(templates), "kinds": dict(kinds), "token_length_direction": dict(lengths),
              "unique_prompts": len({r["prompt_sha256"] for r in rows}),
              "source_medical_records": 0, "human_adjudicated_records": 0}
    if tokenizer:
        from .train import length_audit
        result["length_audit"] = length_audit(rendered, tokenizer, max_length, "dpo")
    return result


def build(output, *, tokenizer=None, max_length=3072):
    out = Path(output).resolve()
    if out.exists():
        raise ValueError("Refusing to overwrite an existing preference release")
    repo = Path(__file__).resolve().parents[2]
    if out.is_relative_to(repo):
        raise ValueError("Keep data releases outside the source repository")
    train, held = [], []
    for kind in KINDS:
        count = 20 if kind in MORE + FEWER else 8 if kind in EQUAL else 16
        for index in range(count):
            train.append(make_case(index, kind, pairs_per_batch=(2, 4, 8)[index % 3], template=index % 6))
        for index in range(2):
            held.append(make_case(index, kind, partition="mechanical_holdout", pairs_per_batch=(2, 8)[index], template=(index + 4) % 6))
    assert len(train) == 256 and len(held) == 32
    if {r["group_id"] for r in train} & {r["group_id"] for r in held}:
        raise ValueError("Train/holdout construction groups overlap")
    sft = [as_sft(row) for row in train]
    for row in sft:
        verify_case(row)
    audits = {"train": summarize(train, tokenizer, max_length), "mechanical_holdout": summarize(held, tokenizer, max_length)}
    if audits["train"]["edge_count_direction"] != {"chosen_more": 80, "chosen_fewer": 80, "equal": 96}:
        raise ValueError("Preference-count balance drift")
    out.mkdir(parents=True, mode=0o700)
    for name, rows in (("train", train), ("sft_chosen", sft), ("mechanical_holdout_preferences", held)):
        path = out / (name + ".jsonl")
        path.write_text("".join(dumps(row) + "\n" for row in rows))
        path.chmod(0o600)
    # Direct prompt-level evaluation must use the authored candidate table;
    # generating all possible world pairs would be a different task.
    manifest = {
        "version": CONTRACT, "audits": audits,
        "training_requires_explicit_versioned_rejected_structure_gate": True,
        "sft_rows_match_preference_chosen_exactly": True,
        "train_holdout_group_overlap": 0, "tokenizer_checked": tokenizer is not None,
        "max_length": max_length, "code_sha256": _sha(__file__),
        "files": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())},
        "recommended_training": {"initial_policy": "verified grounded64 SFT adapter, same policy and frozen reference",
            "epochs": 1, "max_steps": -1, "batch_size": 1, "gradient_accumulation": 4,
            "expected_updates": 64, "expected_preference_exposures": 256,
            "dpo_beta": 0.1, "learning_rate": 5e-6, "seed": 42,
            "control": "same-parent continued SFT on all 256 sft_chosen rows, 1 epoch / 64 updates",
            "selection": "compare with same-parent zero-step; medical quality requires separate full frozen evaluation"},
        "limitations": [
            "Entirely artificial authored extraction tasks; no clinical source or human medical adjudication.",
            "Mechanical holdout names/groups are disjoint but templates overlap; this is not medical generalization.",
            "All absences are asserted by complete artificial worlds; silver UNKNOWN rows are neither imported nor relabelled.",
            "Structural reject edge counts refer to authored relation fields, not valid graph edges or permissively repaired metrics.",
            "The structure treatments target previously observed error classes and are an adaptive follow-up, not original preregistered ranking.",
            "Equal updates do not equalize DPO and continued-SFT token loss exposures or compute costs.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "manifest.json").chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(dumps(build(args.output, tokenizer=tokenizer, max_length=args.max_length)))


if __name__ == "__main__":
    main()

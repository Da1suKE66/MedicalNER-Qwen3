"""Recipe-verified semantic preferences encoded as finite integer actions.

Only complete fictional worlds from recovery_preferences are admitted. The
old structural rejects are excluded, never repaired. Clinical/silver UNKNOWN
and held-out sources cannot enter training through this compiler.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from kg_agent.choice_protocol import ChoiceProtocol
from kg_agent.relation_choices import RelationChoiceTask
from .chat import render_prompt
from .choice_data import _canonical, _sha, _verified_source, compile_recovery_row
from .recovery_preferences import STRUCTURAL, verify_case


DATA_VERSION = "artificial-bounded-choice-preferences-v1"
EXCLUDED_KINDS = tuple(STRUCTURAL)


def _edges(value):
    return [{"pair_id": decision["pair_id"], **deepcopy(edge)}
            for decision in value["decisions"] for edge in decision["relations"]]


def compile_preference(source_row):
    """Reconstruct a complete artificial source and both legal semantic targets."""
    batch, chosen, _ = _verified_source(source_row)
    if source_row["training_view"] != "preference":
        raise ValueError("Compile the original preference source, not an SFT view")
    if source_row["reason"] in EXCLUDED_KINDS:
        raise ValueError("Structural/identity errors are excluded from semantic choice DPO")
    sft = compile_recovery_row(source_row, "relations")
    task = RelationChoiceTask(sft["task_input"]["batch"])
    chosen_actions = list(task.encode_target(_edges(chosen)))
    rejected_actions = list(task.encode_target(_edges(json.loads(source_row["rejected"]))))
    chosen_edges, rejected_edges = task.edges(chosen_actions), task.edges(rejected_actions)
    if chosen_actions == rejected_actions or chosen_edges == rejected_edges:
        raise ValueError("Preference is equivalent after canonical finite-choice encoding")
    if chosen_actions != sft["actions"]:
        raise ValueError("Preference chosen and matched SFT actions differ")
    pair_map = {pair["pair_id"]: pair for pair in batch["pairs"]}
    actual = Counter((pair_map[e["pair_id"]]["source"], pair_map[e["pair_id"]]["target"], e["relation"])
                     for e in chosen_edges)
    oracle = Counter((e["source"], e["target"], e["relation"]) for e in source_row["oracle"]["relations"])
    if actual != oracle:
        raise ValueError("Chosen actions do not recover the complete authored oracle")
    return {
        "data_version": DATA_VERSION, "output_contract": sft["output_contract"],
        "id": source_row["id"] + ":relation_preference", "group_id": source_row["group_id"],
        "split": source_row["split"], "stage": "relations", "reason": source_row["reason"],
        "task_input": sft["task_input"], "prompt": task.prompt,
        "chosen_actions": chosen_actions, "rejected_actions": rejected_actions,
        "source_row": deepcopy(source_row), "source_row_sha256": _sha(source_row),
        "admission": {"source_recipe_verified": True, "supervision": "fully_artificial_exhaustive_world",
                      "unknown_supervision_allowed": False, "medical_source_records": 0,
                      "human_adjudicated": False, "complete_oracle_recovered": True,
                      "completion_encoding": "independent_integer_newline_tokens_then_eos"},
    }


def validate_preference_row(row, *, require_train=True):
    if type(require_train) is not bool:
        raise ValueError("require_train must be bool")
    if not isinstance(row, dict) or row.get("data_version") != DATA_VERSION:
        raise ValueError("Unsupported finite-choice preference data version")
    if require_train and row.get("split") != "train":
        raise ValueError("Only train rows may enter choice preference training")
    expected = compile_preference(row.get("source_row"))
    if _canonical(row) != _canonical(expected):
        raise ValueError("Preference differs from complete artificial source/action reconstruction")
    return RelationChoiceTask(expected["task_input"]["batch"])


def encode_preference_row(row, tokenizer, max_length, require_train=True):
    """Encode the exact inference path, never a concatenated integer string."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    task = validate_preference_row(row, require_train=require_train)
    prompt_ids = list(tokenizer.encode(render_prompt(tokenizer, task.prompt), add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("Preference prompt encoding cannot be empty")
    codec = ChoiceProtocol(tokenizer, len(prompt_ids), task.choices)
    chosen_ids = codec.encode_actions(row["chosen_actions"])
    rejected_ids = codec.encode_actions(row["rejected_actions"])
    if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_length:
        raise ValueError("Choice preference exceeds max_length; no silent truncation")
    return {"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids,
            "stage": "relations", "_choice_counts": {"prompt_tokens": len(prompt_ids),
            "chosen_tokens": len(chosen_ids), "rejected_tokens": len(rejected_ids)}}


def _direction(a, b):
    return "chosen_more" if a > b else "chosen_fewer" if a < b else "equal"


def summarize(rows, tokenizer=None, max_length=3072):
    kinds, sizes, templates, edges, actions, tokens = (Counter() for _ in range(6))
    maximum = 0
    for row in rows:
        task = validate_preference_row(row, require_train=False)
        kinds[row["reason"]] += 1
        sizes[len(task.pairs)] += 1
        templates[row["source_row"]["construction_recipe"]["template"]] += 1
        edges[_direction(len(task.edges(row["chosen_actions"])), len(task.edges(row["rejected_actions"])))] += 1
        actions[_direction(len(row["chosen_actions"]), len(row["rejected_actions"]))] += 1
        if tokenizer is not None:
            encoded = encode_preference_row(row, tokenizer, max_length, require_train=False)
            tokens[_direction(len(encoded["chosen_ids"]), len(encoded["rejected_ids"]))] += 1
            maximum = max(maximum, len(encoded["prompt_ids"]) + max(len(encoded["chosen_ids"]), len(encoded["rejected_ids"])))
    return {"rows": len(rows), "kinds": dict(kinds), "candidate_sizes": dict(sizes),
            "templates": dict(templates), "edge_count_direction": dict(edges),
            "action_count_direction": dict(actions), "token_count_direction": dict(tokens),
            "max_prompt_plus_completion_tokens": maximum if tokenizer is not None else None}


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source_dir, output, *, tokenizer=None, max_length=3072):
    source_dir, out = Path(source_dir), Path(output).resolve()
    if out.exists():
        raise ValueError("Refusing to overwrite an existing preference release")
    if out.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("Keep private releases outside the source repository")
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    partitions, excluded, inputs = {}, {}, {}
    for split, name in (("train", "train.jsonl"), ("mechanical_holdout", "mechanical_holdout_preferences.jsonl")):
        path = source_dir / name
        inputs[name] = _file_sha(path)
        if inputs[name] != source_manifest["files"][name]["sha256"]:
            raise ValueError("Frozen original recovery source hash mismatch")
        selected, omitted = [], Counter()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            verify_case(row)
            if row["split"] != split:
                raise ValueError("Source partition mismatch")
            if row["reason"] in EXCLUDED_KINDS:
                omitted[row["reason"]] += 1
            else:
                selected.append(compile_preference(row))
        partitions[split], excluded[split] = selected, dict(omitted)
    all_train, held = partitions["train"], partitions["mechanical_holdout"]
    if len(all_train) != 192 or len(held) != 24:
        raise ValueError("Frozen source semantic row counts drift")
    # Deterministic metadata-only selection: preserve all equal-edge examples;
    # sample each add/omit family equally to retain 48/48/32 edge balance.
    by_kind = {}
    for row in all_train:
        by_kind.setdefault(row["reason"], []).append(row)
    selected = []
    for kind, rows in sorted(by_kind.items()):
        quota = 12 if len(rows) == 20 else 8
        selected.extend(sorted(rows, key=lambda row: _sha(row["source_row"]["construction_recipe"]))[:quota])
    selected.sort(key=lambda row: _sha(row["source_row"]["construction_recipe"]))
    if len(selected) != 128:
        raise ValueError("Matched training subset count drift")
    for field in ("group_id", "prompt", "source_row_sha256"):
        if {row[field] for row in all_train} & {row[field] for row in held}:
            raise ValueError("Train/holdout identity intersection")
    sft = [compile_recovery_row(row["source_row"], "relations") for row in selected]
    if any(row["chosen_actions"] != chosen["actions"] or row["prompt"] != chosen["prompt"] for row, chosen in zip(selected, sft)):
        raise ValueError("Matched SFT/preference exposure mismatch")
    audits = {"train": summarize(selected, tokenizer, max_length),
              "train_all_semantic": summarize(all_train, tokenizer, max_length),
              "mechanical_holdout": summarize(held, tokenizer, max_length)}
    if audits["train"]["edge_count_direction"] != {"chosen_more": 48, "chosen_fewer": 48, "equal": 32}:
        raise ValueError("Matched edge-count balance drift")
    out.mkdir(parents=True, mode=0o700)
    for name, rows in (("train", selected), ("train_all_semantic", all_train), ("sft_chosen", sft),
                       ("mechanical_holdout_preferences", held)):
        path = out / (name + ".jsonl")
        path.write_text("".join(_canonical(row) + "\n" for row in rows))
        path.chmod(0o600)
    manifest = {"version": DATA_VERSION, "source_files": inputs, "excluded": excluded,
                "exclusion_policy": "No structural repairs: invalid JSON, duplicate/fabricated pair IDs excluded",
                "audits": audits, "matched_source_order_sha256": _sha([r["source_row_sha256"] for r in selected]),
                "split_group_sha256": {key: _sha(sorted(r["group_id"] for r in rows)) for key, rows in partitions.items()},
                "train_holdout_intersection": 0, "chosen_sft_actions_and_order_match": True,
                "tokenizer_checked": tokenizer is not None, "max_length": max_length,
                "code_sha256": _file_sha(__file__),
                "files": {p.name: {"sha256": _file_sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())},
                "training": {"rows": 128, "epochs": 1, "gradient_accumulation": 4, "batch_size": 1,
                             "expected_updates": 32, "seed": 42, "parent": "same verified bounded-choice parent for DPO/SFT/reference"},
                "limitations": ["Adaptive artificial semantic screen, not medical effectiveness or generalization.",
                                "Train/holdout identities differ but authored templates overlap.",
                                "No clinical or silver UNKNOWN rows are admitted; every world is explicitly exhaustive.",
                                "Equal example exposure does not equalize DPO/SFT token losses or compute."]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "manifest.json").chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-length", type=int, default=3072)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(build(args.source_dir, args.output, tokenizer=tokenizer, max_length=args.max_length)))


if __name__ == "__main__":
    main()

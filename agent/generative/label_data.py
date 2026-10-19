"""Verified artificial SFT targets containing literal semantic relation labels.

The disk container remains JSON, but the model completion is ordinary text:
local entity ID, literal relation label, local entity ID, visible evidence IDs,
separated by TABs.  Complete fictional worlds alone justify empty completions.
This compiler is not a clinical/silver supervision migration or a DPO builder.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from .chat import render_prompt
from .choice_data import _canonical, _relation_input, _sha, _verified_source
from .recovery_preferences import CONTRACT as SOURCE_CONTRACT


DATA_VERSION = "artificial-literal-relation-data-v1"
COMPLETION_ENCODING = "literal_relation_lines_then_eos"
STAGES = ("relations",)


def _task(task_input):
    from kg_agent.relation_labels import RelationLabelTask
    return RelationLabelTask(task_input["batch"])


def _edge_signatures(edges):
    return Counter((edge["pair_id"], edge["relation"], tuple(sorted(edge["evidence_span_ids"])))
                   for edge in edges)


def compile_recovery_row(source_row, stage="relations"):
    """Compile only recipe-reconstructed, exhaustive artificial relation targets.

    A source preference's rejected text is retained solely as provenance; even
    an intentionally malformed rejected JSON is never sent to this SFT model.
    Evaluation may compile held-out recipes, but encode_row checks split again.
    """
    if stage not in STAGES:
        raise ValueError("Literal relation data supports only the relations stage")
    batch, chosen, _ = _verified_source(source_row)
    from kg_agent.label_backend import CONTRACT as OUTPUT_CONTRACT
    task_input = _relation_input(batch)
    task = _task(task_input)
    source_targets = [{"pair_id": decision["pair_id"], **deepcopy(edge)}
                      for decision in chosen["decisions"] for edge in decision["relations"]]
    completion = task.render_target(source_targets)
    if not isinstance(completion, str):
        raise ValueError("Literal relation renderer must return text")
    target = task.parse(completion)
    if _edge_signatures(target) != _edge_signatures(source_targets):
        raise ValueError("Literal relation render/parse changed a target or its evidence")
    pair_map = {pair["pair_id"]: pair for pair in batch["pairs"]}
    actual = Counter((pair_map[edge["pair_id"]]["source"], pair_map[edge["pair_id"]]["target"], edge["relation"])
                     for edge in target)
    oracle = Counter((edge["source"], edge["target"], edge["relation"])
                     for edge in source_row["oracle"]["relations"])
    if actual != oracle:
        raise ValueError("Literal relation targets do not recover the complete artificial oracle")
    if not target and completion != "":
        raise ValueError("A verified no-relation target must be empty text, not NONE")
    return {
        "data_version": DATA_VERSION, "output_contract": OUTPUT_CONTRACT,
        "id": source_row["id"] + ":relations:literal_labels",
        "group_id": source_row["group_id"], "split": source_row["split"], "stage": stage,
        "task_input": task_input, "prompt": task.prompt, "completion": completion, "target": target,
        "source_row": deepcopy(source_row), "source_row_sha256": _sha(source_row),
        "admission": {
            "source_contract": SOURCE_CONTRACT, "source_recipe_verified": True,
            "supervision": "fully_artificial_exhaustive_world", "unknown_supervision_allowed": False,
            "medical_source_records": 0, "human_adjudicated": False,
            "complete_oracle_recovered": True, "target_source": "recipe_chosen_relations",
            "target_count": len(target), "completion_encoding": COMPLETION_ENCODING,
            "action_count_semantics": "relation_records_not_numeric_actions",
            "empty_completion_policy": "EOS_only_after_verified_exhaustive_no_relation_world",
        },
    }


def validate_row(row, *, require_train=True):
    """Reconstruct source, input tables and literal targets; accept no extra fields."""
    if type(require_train) is not bool:
        raise ValueError("require_train must be bool")
    if not isinstance(row, dict) or row.get("data_version") != DATA_VERSION:
        raise ValueError("Unsupported literal relation data version")
    if require_train and row.get("split") != "train":
        raise ValueError("Only train rows may enter literal relation training; held-out remains evaluation-only")
    expected = compile_recovery_row(row.get("source_row"), row.get("stage"))
    if _canonical(row) != _canonical(expected):
        raise ValueError("Literal relation row differs from complete source/task/target reconstruction")
    return _task(expected["task_input"])


def encode_row(row, tokenizer, max_length, require_train=True):
    """Encode the full literal completion once, then add exactly one EOS.

    Prompt labels are -100.  Nothing is truncated or independently retokenized
    by line or field. The existing exposure mixin may consume _choice_counts,
    whose action_count now denotes relation records, not integer decisions.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    task = validate_row(row, require_train=require_train)
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("Literal relation completion requires a valid EOS token ID")
    prompt_ids = list(tokenizer.encode(render_prompt(tokenizer, task.prompt), add_special_tokens=False))
    completion_ids = list(tokenizer.encode(row["completion"], add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("Literal relation prompt encoding cannot be empty")
    if any(type(token) is not int or token < 0 for token in prompt_ids + completion_ids):
        raise ValueError("Tokenizer returned invalid token IDs")
    if eos in completion_ids:
        raise ValueError("Literal completion unexpectedly contains EOS before the appended terminator")
    if bool(row["completion"]) != bool(completion_ids):
        raise ValueError("Empty and nonempty literal completions must retain their tokenization meaning")
    completion_ids.append(eos)
    if len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError(
            f"Literal relation row exceeds max_length: {len(prompt_ids)} prompt + {len(completion_ids)} completion > {max_length}; no silent truncation"
        )
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "_choice_counts": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
                           "action_count": len(row["target"]), "stage": "relations",
                           "action_count_unit": "relation_records", "completion_encoding": COMPLETION_ENCODING},
    }

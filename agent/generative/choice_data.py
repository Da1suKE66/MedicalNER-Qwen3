"""Compile verified artificial worlds into non-JSON causal-LM choice targets.

JSON is only the on-disk container. Model completions are independently encoded
integer/newline actions followed by EOS. This first release admits only the
closed recipes from recovery_preferences; it never infers negative supervision
from absent silver annotations or converts UNKNOWN into a stop action.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from kg_agent.choice_backend import CONTRACT as OUTPUT_CONTRACT
from kg_agent.choice_protocol import ChoiceProtocol
from kg_agent.contracts import EntityCandidate
from kg_agent.entity_choices import EntityChoiceTask
from kg_agent.relation_choices import RelationChoiceTask
from kg_agent.source import SourceRouter

from .chat import render_prompt
from .protocol import exact_hits
from .recovery_preferences import CONTRACT as SOURCE_CONTRACT, _construct, as_sft, verify_case


DATA_VERSION = "artificial-bounded-choice-data-v1"
STAGES = ("entities", "relations")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _verified_source(source_row):
    # verify_case rejects UNKNOWN metadata, unsupported provenance and all
    # changes not reconstructible from its closed recipe, including split.
    audit = verify_case(source_row)
    expected, batch, chosen = _construct(source_row["construction_recipe"])
    if source_row["training_view"] == "chosen_sft":
        expected = as_sft(expected)
    # Canonical JSON also distinguishes bool/int and int/float substitutions
    # that ordinary Python mapping equality could otherwise consider equal.
    if _canonical(source_row) != _canonical(expected):
        raise ValueError("Source row differs from its typed artificial recipe reconstruction")
    return batch, chosen, audit


def _entity_input_and_targets(source_row):
    raw = source_row["oracle_source_text"]
    document = SourceRouter().route(raw)
    if document.source_type != "free_text" or document.raw != raw:
        raise ValueError("Artificial entity source must preserve exact free-text coordinates")
    spans = [{"span_id": span.id, "text": span.text, "section": span.section}
             for span in document.spans]
    seen, candidates = set(), []
    for entity in source_row["oracle"]["entities"]:
        found = False
        for span in spans:
            for start, end in exact_hits(entity["name"], span["text"]):
                found = True
                key = (span["span_id"], start, end, entity["label"])
                if key in seen:
                    continue
                seen.add(key)
                # Every occurrence has explicit offsets: repeated strings never
                # collapse into a first-hit target. The mention is source-owned.
                candidates.append(EntityCandidate(
                    f"{span['span_id']}:{start}-{end}", entity["label"],
                    "associated", span["text"][start:end],
                ))
        if not found:
            raise ValueError("Artificial oracle entity has no selectable source mention")
    inputs = {"spans": spans, "target_name": None, "max_candidates": 64,
              "max_span_tokens": None}
    return inputs, candidates


def _relation_input(batch):
    # Preserve the actual construction batch's original endpoint IDs and table
    # order. Re-parsing the old prompt would lose original IDs and is forbidden.
    return {"batch": {
        "entities": [{key: entity[key] for key in ("id", "entity_id", "name", "label")}
                     for entity in batch["entities"]],
        "pairs": [{key: deepcopy(pair[key]) for key in ("pair_id", "source", "target", "allowed")}
                  for pair in batch["pairs"]],
        "spans": [{key: span[key] for key in ("span_id", "text")} for span in batch["spans"]],
    }}


def _task(stage, task_input):
    if stage == "entities":
        return EntityChoiceTask(**task_input)
    if stage == "relations":
        return RelationChoiceTask(task_input["batch"])
    raise ValueError("stage must be entities or relations")


def compile_recovery_row(source_row, stage):
    """Return a JSON-writable task, action list and complete source provenance.

    Held-out recipes may be compiled for evaluation, retaining their split.
    Training admission happens again in encode_row. No medical/silver source or
    arbitrary task dictionary can enter this compiler.
    """
    if stage not in STAGES:
        raise ValueError("stage must be entities or relations")
    batch, chosen, _ = _verified_source(source_row)
    if stage == "entities":
        task_input, targets = _entity_input_and_targets(source_row)
    else:
        task_input = _relation_input(batch)
        targets = [{"pair_id": decision["pair_id"], **deepcopy(edge)}
                   for decision in chosen["decisions"] for edge in decision["relations"]]
    task = _task(stage, task_input)
    actions = list(task.encode_target(targets))
    if task.choices(tuple(actions)) is not None:
        raise ValueError("Compiled choice target is not terminal")
    return {
        "data_version": DATA_VERSION, "output_contract": OUTPUT_CONTRACT,
        "id": f"{source_row['id']}:{stage}", "group_id": source_row["group_id"],
        "split": source_row["split"], "stage": stage,
        "task_input": task_input, "prompt": task.prompt, "actions": actions,
        "source_row": deepcopy(source_row), "source_row_sha256": _sha(source_row),
        "admission": {
            "source_contract": SOURCE_CONTRACT,
            "supervision": "fully_artificial_exhaustive_world",
            "source_recipe_verified": True, "unknown_supervision_allowed": False,
            "medical_source_records": 0, "human_adjudicated": False,
            "target_source": "all_oracle_entity_mentions" if stage == "entities" else "recipe_chosen_relations",
            "target_count": len(targets),
            "entity_importance_policy": "constant_associated_not_clinical_annotation" if stage == "entities" else None,
            "completion_encoding": "independent_integer_newline_tokens_then_eos",
        },
    }


def validate_row(row, *, require_train=True):
    """Rebuild the complete row and return its verified finite-choice task."""
    if not isinstance(require_train, bool):
        raise ValueError("require_train must be bool")
    if not isinstance(row, dict) or row.get("data_version") != DATA_VERSION:
        raise ValueError("Unsupported bounded-choice data version")
    if require_train and row.get("split") != "train":
        raise ValueError("Only train rows may enter choice training; held-out remains evaluation-only")
    expected = compile_recovery_row(row.get("source_row"), row.get("stage"))
    if _canonical(row) != _canonical(expected):
        raise ValueError("Choice row differs from its complete source/task/action reconstruction")
    return _task(expected["stage"], expected["task_input"])


def encode_row(row, tokenizer, max_length, require_train=True):
    """Build Python lists for LM SFT, matching constrained inference exactly.

    The collator/trainer must remove _choice_counts before forwarding tensors.
    Rejected preference text is retained only in the disk provenance container;
    neither that text nor its tokens are supplied to the model.
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    task = validate_row(row, require_train=require_train)
    rendered = render_prompt(tokenizer, task.prompt)
    prompt_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("Choice prompt encoding cannot be empty")
    protocol = ChoiceProtocol(tokenizer, len(prompt_ids), task.choices)
    completion_ids = protocol.encode_actions(row["actions"], require_complete=True)
    if len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError(
            f"Choice row exceeds max_length: {len(prompt_ids)} prompt + {len(completion_ids)} completion > {max_length}; no silent truncation"
        )
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "_choice_counts": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
                           "action_count": len(row["actions"]), "stage": row["stage"]},
    }

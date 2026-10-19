"""Reconstruct data from resident TRAIN rows plus a source-free reviewed recipe."""
from __future__ import annotations
import argparse
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    raw = value if isinstance(value, bytes) else (value if isinstance(value, str) else canonical(value)).encode()
    return hashlib.sha256(raw).hexdigest()


def verified_rows(path, expected_sha):
    path = Path(path)
    if sha(path.read_bytes()) != expected_sha:
        raise ValueError("Resident TRAIN file checksum differs from recipe")
    rows = [json.loads(line) for line in path.open()]
    if any(r["split"] != "train" for r in rows):
        raise ValueError("TRAIN rows only")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate base identity")
    return rows


def reconstruct_preferences(rows, recipe):
    from semantic import protocol as p
    from semantic.data import validate_row
    by_id = {r["id"]: r for r in rows}
    output = []
    for item in recipe["items"]:
        row = by_id[item["source_row_id"]]
        validate_row(row)
        if sha(row) != item["source_row_sha256"]:
            raise ValueError("Source row hash mismatch")
        target = deepcopy(row["target"])
        mutation = item["mutation"]
        index = mutation["line_index"]
        if len(row["completion"].splitlines()) != len(target):
            raise ValueError("Expected one evidence record per target")
        if target[index]["relation"] != mutation["original_label"]:
            raise ValueError("Changed source relation")
        if mutation["kind"] == "explicit_negation_label_flip":
            if mutation["original_label"] != "argues_against" or mutation["rejected_label"] != "is_core_symptom_of":
                raise ValueError("Unreviewed label mutation")
            target[index]["relation"] = mutation["rejected_label"]
        elif mutation["kind"] == "subtype_direction_reversal":
            if mutation["original_label"] != "subtype_of":
                raise ValueError("Unreviewed direction mutation")
            target[index]["source"], target[index]["target"] = target[index]["target"], target[index]["source"]
        else:
            raise ValueError("Unknown mutation")
        rejected = p.render_relations(target, row["task_input"]["entities"])
        original_lines, rejected_lines = row["completion"].splitlines(), rejected.splitlines()
        if len(original_lines) != len(rejected_lines) or sum(x != y for x, y in zip(original_lines, rejected_lines)) != 1:
            raise ValueError("Preference must alter exactly one record")
        parsed = p.parse_relations(row["task_input"]["text"], row["task_input"]["entities"], rejected, row["metadata"])
        if not parsed["valid"] or parsed["rejected_relations"]:
            raise ValueError("Hard negative must pass syntax, entity-reference, ontology and quote filters")
        if sha(rejected) != item["rejected_sha256"]:
            raise ValueError("Reconstructed rejected SHA differs")
        review = item["review"]
        if not all(review.get(k) is True for k in ("accepted", "source_supported", "negative_explicitly_contradicted")):
            raise ValueError("Unreviewed preference")
        exported_mutation = deepcopy(mutation)
        exported_mutation["semantic_change_kind"] = mutation["kind"]
        exported_mutation["kind"] = "reviewed_semantic_contradiction"
        output.append({"source_row_id": row["id"], "source_row_sha256": sha(row),
                       "chosen": row["completion"], "rejected": rejected,
                       "mutation": exported_mutation, "review": review})
    return output


def reconstruct_replay(rows, recipe):
    from semantic.data import validate_row
    by_id = {r["id"]: r for r in rows}
    result = deepcopy(rows)
    for item in recipe["items"]:
        row = by_id[item["source_row_id"]]
        validate_row(row)
        if sha(row) != item["source_row_sha256"]:
            raise ValueError("Source row hash mismatch")
        if not any(e["label"] == "Diagnostic Criterion" and len(e["name"]) >= 160 for e in row["target"]):
            raise ValueError("Replay selection policy mismatch")
        duplicate = deepcopy(row)
        duplicate["id"] += ":criterion_replay:1"
        validate_row(duplicate)
        result.append(duplicate)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("Replay has duplicate identity")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-path", required=True)
    parser.add_argument("--resident-train", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.agent_path)
    recipe = json.loads(Path(args.recipe).read_text())
    rows = verified_rows(args.resident_train, recipe["input_file_sha256"])
    if recipe["kind"] == "semantic_preference_single_mutation_v1":
        output = reconstruct_preferences(rows, recipe)
    elif recipe["kind"] == "criterion_complete_row_replay_v1":
        output = reconstruct_replay(rows, recipe)
    else:
        raise ValueError("Unknown recipe")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(canonical(r) + "\n" for r in output))
    destination.chmod(0o600)
    receipt = {"kind": recipe["kind"], "split": "train", "rows": len(output),
               "input_file_sha256": recipe["input_file_sha256"], "recipe_sha256": sha(recipe),
               "output_file_sha256": sha(destination.read_bytes()),
               "source_text_uploaded": False, "evaluation_inputs_read": False,
               "original_chosen_and_prompts_preserved": True, "missing_annotations_used_as_negative": False}
    destination.with_suffix(destination.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(canonical(receipt))


if __name__ == "__main__":
    main()

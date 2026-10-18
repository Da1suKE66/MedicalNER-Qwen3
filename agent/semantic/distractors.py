"""Matched relation-input corruption with source-absent TRAIN entity names.

This is an input robustness experiment, not clinical negative adjudication.
Targets, UNKNOWN policy, rows and annotation-atom exposure remain unchanged.
Only the supplied entity table gains entries; no NONE target is introduced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from data_repair.text_normalization import find_mentions
from generative.chat import render_prompt
from semantic.data import DATA_VERSION, canonical, sha, validate_row
from semantic import protocol as p
from semantic.runtime import verify_title_context

EXPERIMENT = "train-source-absence-entity-input-corruption-v1"
BASE_VARIANTS = ("full_source", "short_source", "short_canonical")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _identity(row):
    return row["metadata"]["source_record_sha256"]


def build_donor_pool(entity_rows, title_receipts):
    """Validate full-source TRAIN donors and retain original source receipts."""
    contexts, candidates = {}, []
    seen_rows = set()
    for row in entity_rows:
        validate_row(row, require_train=True)
        if row["stage"] != "entities" or row["variant"] != "full_source":
            raise ValueError("Donors require frozen full-source train entity rows")
        if row["id"] in seen_rows:
            raise ValueError("Duplicate donor row identity")
        seen_rows.add(row["id"])
        metadata, text = row["metadata"], row["task_input"]["text"]
        window = metadata["source_window"]
        if window["start"] != 0 or window["end"] != len(text) or sha(text) != metadata["source_text_sha256"]:
            raise ValueError("Donor/recipient context must be the exact complete source")
        title = metadata.get("title_context")
        if title:
            verify_title_context(metadata, title_receipts, source_group_id=row["group_id"])
        context = {"split": "train", "source_group_id": row["group_id"],
            "source_record_sha256": _identity(row), "source_index": metadata["source_index"],
            "body": text, "body_sha256": sha(text), "title": title["text"] if title else "",
            "title_sha256": sha(title["text"] if title else ""),
            "title_context_sha256": sha(title) if title else None,
            "full_entity_row_id": row["id"], "full_entity_row_sha256": sha(row)}
        key = _identity(row)
        if key in contexts:
            raise ValueError("More than one full entity row for a source identity")
        contexts[key] = context
        for entity in row["target"]:
            if not (find_mentions(entity["name"], text) or find_mentions(entity["name"], context["title"])):
                raise ValueError("Donor name lacks support in its own original body/title")
            candidate = {"label": entity["label"], "name": entity["name"],
                "donor_entity_id": entity["id"], "donor_split": "train",
                "donor_source_group_id": row["group_id"], "donor_source_record_sha256": key,
                "donor_entity_row_id": row["id"], "donor_entity_row_sha256": sha(row),
                "donor_body_sha256": sha(text), "donor_title_sha256": context["title_sha256"]}
            candidate["donor_identity_sha256"] = sha(candidate)
            candidates.append(candidate)
    # Avoid weighting common repeated names by the number of source occurrences.
    unique = {}
    for candidate in sorted(candidates, key=lambda x: x["donor_identity_sha256"]):
        unique.setdefault((candidate["label"], p.normalize_name(candidate["name"])), candidate)
    return contexts, sorted(unique.values(), key=lambda x: x["donor_identity_sha256"])


def choose_donors(base_row, context, donor_pool, *, seed=42, max_candidates=128):
    """A finite deterministic scan; all windows of one source share a recipe."""
    if type(seed) is not int or type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("Seed and finite candidate budget must be integers")
    if context["split"] != "train" or base_row["split"] != "train":
        raise ValueError("Tune/test cannot participate in corruption recipes")
    if context["source_record_sha256"] != _identity(base_row) or context["source_group_id"] != base_row["group_id"]:
        raise ValueError("Recipient context identity mismatch")
    if sha(context["body"]) != context["body_sha256"] or sha(context["title"]) != context["title_sha256"]:
        raise ValueError("Recipient full-source hash drift")
    digest = sha({"experiment": EXPERIMENT, "seed": seed, "recipient": _identity(base_row)})
    wanted = 1 + int(digest[:8], 16) % 2
    labels = {e["label"] for e in base_row["task_input"]["entities"]}
    existing = {(e["label"], p.normalize_name(e["name"])) for e in base_row["task_input"]["entities"]}
    selected, reasons = [], Counter()
    if donor_pool:
        start = int(digest[8:24], 16) % len(donor_pool)
        for offset in range(min(max_candidates, len(donor_pool))):
            donor = donor_pool[(start + offset) % len(donor_pool)]
            if donor["donor_split"] != "train":
                raise ValueError("Donor pool contains a non-train source")
            key = donor["label"], p.normalize_name(donor["name"])
            if donor["donor_source_group_id"] == base_row["group_id"]:
                reasons["same_source_group"] += 1
            elif donor["label"] not in labels:
                reasons["label_not_already_in_entity_table"] += 1
            elif key in existing:
                reasons["duplicate_entity_key"] += 1
            elif find_mentions(donor["name"], context["body"]):
                reasons["full_body_match"] += 1
            elif find_mentions(donor["name"], context["title"]):
                reasons["trusted_title_match"] += 1
            else:
                selected.append(deepcopy(donor))
                existing.add(key)
                if len(selected) == wanted:
                    break
        examined = offset + 1
    else:
        examined = 0
    return selected, {"seed": seed, "max_candidates": max_candidates,
        "requested_count": wanted, "candidates_examined": examined,
        "candidate_rejections": dict(reasons), "selected_before_token_budget": len(selected),
        "search_status": "requested_count_found" if len(selected) == wanted else "finite_search_exhausted",
        "selection_scope": "same_recipient_source_recipe_across_full_and_short_views",
        "donor_labels_restricted_to_existing_recipient_types": True}


def _row(base_row, context, selected, recipe, *, budget_removed=0):
    row = deepcopy(base_row)
    row["id"] += ":input_distractors_v1"
    row["variant"] += "_distractors"
    metadata = deepcopy(row["metadata"])
    metadata["variant"] = row["variant"]
    table = row["task_input"]["entities"]
    original_count = len(table)
    used_ids, added = {e["id"] for e in table}, []
    for donor in selected:
        identity = "distractor_" + donor["donor_identity_sha256"][:24]
        while identity in used_ids:
            identity += "_x"
        used_ids.add(identity)
        # These have no recipient source spans. Do not invent offsets to affect
        # prompt position, and do not copy a donor's grounding into this source.
        table.append({"id": identity, "label": donor["label"], "name": donor["name"],
                      "mentions": [], "grounding_verified": False})
        added.append({**deepcopy(donor), "injected_entity_id": identity,
                      "full_body_matches": 0, "trusted_title_matches": 0,
                      "clinical_unrelatedness_adjudicated": False})
    ordered = p._ordered(table, metadata["order"])
    for donor in added:
        donor["display_entity_table_index"] = next(i for i, e in enumerate(ordered) if e["id"] == donor["injected_entity_id"])
    metadata["input_corruption"] = {"experiment": EXPERIMENT, "base_row_id": base_row["id"],
        "base_row_sha256": sha(base_row), "base_variant": base_row["variant"],
        "recipient_source_record_sha256": context["source_record_sha256"],
        "recipient_source_group_id": context["source_group_id"],
        "recipient_full_body_sha256": context["body_sha256"],
        "recipient_trusted_title_sha256": context["title_sha256"],
        "recipient_title_context_sha256": context["title_context_sha256"],
        "recipient_full_entity_row_sha256": context["full_entity_row_sha256"],
        "recipe": deepcopy(recipe), "added_entities": added,
        "added_count": len(added), "original_entity_count": original_count,
        "token_budget_removed_donors": budget_removed,
        "model_input_changed": bool(added), "unperturbed_reason": None if added else
            "token_budget" if budget_removed else "no_eligible_donor_within_finite_search",
        "absence_match_policy": "NFKC_casefold_whitespace_fold_word_boundaries_no_alias_or_fuzzy_matching",
        "interpretation": "input_corruption_source_absence_robustness_not_clinical_negative_labels",
        "position_caveat": ("Existing canonical type/name order naturally interleaves donors; positional or type shortcuts are not ruled out"
                            if base_row["variant"] == "short_canonical" else
                            "Unlocated entities use frozen source-order default position; may appear near table start"),
        "no_new_relation_targets": True, "unknown_policy_unchanged": True}
    row["metadata"] = metadata
    row["task_input"]["metadata"] = deepcopy(metadata)
    row["prompt"] = p.build_relation_prompt(row["task_input"]["text"], table, metadata, metadata["order"])
    row["input_sha256"] = sha(row["task_input"])
    return row


def _lengths(row, tokenizer):
    prompt = len(tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False))
    completion = len(tokenizer.encode(row["completion"], add_special_tokens=False)) + 1
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def compile_row(base_row, context, donor_pool, *, seed=42, max_candidates=128, tokenizer=None, max_length=8192):
    validate_row(base_row, require_train=True)
    if base_row["stage"] != "relations" or base_row["variant"] not in BASE_VARIANTS:
        raise ValueError("Corruption requires a frozen full_source/short_source/short_canonical relation row")
    metadata = base_row["metadata"]
    if metadata["source_text_sha256"] != context["body_sha256"]:
        raise ValueError("Recipient is not linked to the exact complete body")
    if sha((metadata.get("title_context") or {}).get("text", "")) != context["title_sha256"]:
        raise ValueError("Recipient trusted title drift")
    window = metadata["source_window"]
    if context["body"][window["start"]:window["end"]] != base_row["task_input"]["text"]:
        raise ValueError("Recipient local text does not match full source coordinates")
    selected, recipe = choose_donors(base_row, context, donor_pool, seed=seed, max_candidates=max_candidates)
    initial = len(selected)
    while True:
        row = _row(base_row, context, selected, recipe, budget_removed=initial - len(selected))
        if tokenizer is None or _lengths(row, tokenizer)["total_tokens"] <= max_length:
            break
        if not selected:
            raise ValueError("Unmodified base row exceeds token budget; no truncation or row dropping")
        selected.pop()
    validate_row(row, require_train=True)
    for field in ("data_version", "split", "stage", "group_id", "completion", "target", "target_count", "target_sha256"):
        if row[field] != base_row[field]:
            raise ValueError("Input-only corruption changed supervised targets or source identity")
    if row["metadata"]["supervised_atom_ids"] != base_row["metadata"]["supervised_atom_ids"]:
        raise ValueError("Input-only corruption changed positive annotation atoms")
    return row


def validate_distractor_row(row, base_row, context, donor_pool, **kwargs):
    """Independently replay the deterministic donor recipe and every byte."""
    expected = compile_row(base_row, context, donor_pool, **kwargs)
    if canonical(row) != canonical(expected):
        raise ValueError("Corruption row differs from deterministic source/donor reconstruction")
    return validate_row(row, require_train=True)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def position_summary(rows):
    """Count actual displayed positions, including conditional type distributions.

    Thirds use floor(3 * zero_based_index / table_size), not semantic judgements.
    Also retain literal first/last and every exact (index, table_size) count.
    """
    def empty():
        return {"count": 0, "thirds": Counter(), "exact_index": Counter(),
                "index_and_table_size": Counter(), "literal_first": 0, "literal_last": 0}
    overall, by_type = empty(), {}
    for row in rows:
        trace = row["metadata"]["input_corruption"]
        size = len(row["task_input"]["entities"])
        for donor in trace["added_entities"]:
            index = donor["display_entity_table_index"]
            if not 0 <= index < size:
                raise ValueError("Donor display position outside entity table")
            bucket = ("head", "middle", "tail")[min(2, 3 * index // size)]
            for stats in (overall, by_type.setdefault(donor["label"], empty())):
                stats["count"] += 1
                stats["thirds"][bucket] += 1
                stats["exact_index"][index] += 1
                stats["index_and_table_size"][f"{index}/{size}"] += 1
                stats["literal_first"] += int(index == 0)
                stats["literal_last"] += int(index == size - 1)
    return {"index_origin": 0, "thirds_definition": "floor(3*index/table_size): 0=head,1=middle,2=tail",
            "overall": overall, "by_entity_type": by_type,
            "interpretation": "Measured display positions only; no claim that all position/type shortcuts are eliminated"}


def _verify_previous_release(previous_release, base_manifest_sha, parameters):
    if previous_release is None:
        return None
    previous = Path(previous_release)
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest["base_release_manifest_sha256"] != base_manifest_sha or manifest["parameters"] != parameters:
        raise ValueError("Previous release uses a different frozen base or donor recipe")
    for name, digest in manifest["output_sha256"].items():
        if sha((previous / name).read_bytes()) != digest:
            raise ValueError("Previous release file SHA drift: " + name)
    return manifest


def build_release(base_release, output_dir, tokenizer, *, seed=42, max_candidates=128, max_length=8192,
                  previous_release=None):
    base, output = Path(base_release), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing corruption release")
    manifest = json.loads((base / "manifest.json").read_text())
    names = ["full_source/train_entities.jsonl", "title_source_receipts.json"] + [f"{v}/train_relations.jsonl" for v in BASE_VARIANTS]
    for name in names:
        if sha((base / name).read_bytes()) != manifest["output_sha256"][name]:
            raise ValueError("Frozen base release SHA drift: " + name)
    if sha(Path(p.__file__).read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Frozen semantic protocol drift")
    parameters = {"seed": seed, "max_candidates": max_candidates, "max_length": max_length}
    previous = _verify_previous_release(previous_release, sha((base / "manifest.json").read_bytes()), parameters)
    contexts, donors = build_donor_pool(_read(base / names[0]), json.loads((base / "title_source_receipts.json").read_text()))
    summary = {"experiment": EXPERIMENT, "status": "compiled_not_trained", "data_version": DATA_VERSION,
        "base_release_manifest_sha256": sha((base / "manifest.json").read_bytes()),
        "base_files_sha256": {name: manifest["output_sha256"][name] for name in names},
        "protocol_sha256": manifest["protocol_sha256"], "compiler_sha256": sha(Path(__file__).read_bytes()),
        "parameters": parameters,
        "donor_pool": {"unique_type_name_entries": len(donors), "source_records": len(contexts),
                       "source_groups": len({c["source_group_id"] for c in contexts.values()}),
                       "eligible_context_source_records": len(contexts),
                       "eligible_context_source_groups": len({c["source_group_id"] for c in contexts.values()}),
                       "retained_pool_source_records": len({d["donor_source_record_sha256"] for d in donors}),
                       "retained_pool_source_groups": len({d["donor_source_group_id"] for d in donors}),
                       "splits": ["train"], "tune_or_test_donors": 0},
        "release_version": "distractors_v2",
        "comparison_design": {"recommended_primary_arm": "short_canonical_distractors",
                              "matched_control": "short_canonical",
                              "stress_only_arms": ["full_source_distractors", "short_source_distractors"],
                              "control_input_and_target_ordering_unchanged": True,
                              "same_source_donor_recipe_across_all_variants": True},
        "arms": {}, "output_sha256": {},
        "caveats": ["Source surface absence is not clinical unrelatedness or reliable negative relation adjudication.",
                    "Current live filtering already rejects source-absent entities; this arm alone cannot establish live F1 improvement.",
                    "Frozen source-order display places unlocated entities near table start; position shortcuts remain a confound.",
                    "Same annotation atoms, rows and epochs preserve update counts; extra input tokens increase compute.",
                    "No NONE, empty target, missing-pair negative or preference data is created."]}
    for variant in BASE_VARIANTS:
        rows = _read(base / f"{variant}/train_relations.jsonl")
        emitted, histogram, positions, rejection_counts = [], Counter(), Counter(), Counter()
        requested, searches, unperturbed = Counter(), Counter(), Counter()
        budget_removed = 0
        before_tokens, after_tokens, after_lengths = Counter(), Counter(), []
        base_atoms, new_atoms = Counter(), Counter()
        for original in rows:
            row = compile_row(original, contexts[_identity(original)], donors, seed=seed,
                              max_candidates=max_candidates, tokenizer=tokenizer, max_length=max_length)
            emitted.append(row)
            trace = row["metadata"]["input_corruption"]
            histogram[trace["added_count"]] += 1
            requested[trace["recipe"]["requested_count"]] += 1
            searches[trace["recipe"]["search_status"]] += 1
            if trace["unperturbed_reason"]:
                unperturbed[trace["unperturbed_reason"]] += 1
            budget_removed += trace["token_budget_removed_donors"]
            rejection_counts.update(trace["recipe"]["candidate_rejections"])
            positions.update(e["display_entity_table_index"] for e in trace["added_entities"])
            before_tokens.update(_lengths(original, tokenizer))
            after = _lengths(row, tokenizer)
            after_tokens.update(after)
            after_lengths.append(after["total_tokens"])
            base_atoms.update((_identity(original), a) for a in original["metadata"]["supervised_atom_ids"])
            new_atoms.update((_identity(row), a) for a in row["metadata"]["supervised_atom_ids"])
        if base_atoms != new_atoms or len(rows) != len(emitted):
            raise ValueError("Matched source-row/annotation exposure invariant failed")
        if before_tokens["completion_tokens"] != after_tokens["completion_tokens"]:
            raise ValueError("Input corruption changed target token exposure")
        relative = f"{variant}_distractors/train_relations.jsonl"
        _write_rows(output / relative, emitted)
        summary["output_sha256"][relative] = sha((output / relative).read_bytes())
        summary["arms"][variant + "_distractors"] = {"control_variant": variant,
            "control_sha256": manifest["output_sha256"][f"{variant}/train_relations.jsonl"],
            "rows": len(rows), "positive_annotation_atoms": sum(base_atoms.values()),
            "source_records": len({_identity(r) for r in rows}),
            "rows_by_added_entity_count": dict(sorted(histogram.items())),
            "rows_by_requested_entity_count": dict(sorted(requested.items())),
            "search_status_counts": dict(searches), "unperturbed_reason_counts": dict(unperturbed),
            "token_budget_removed_entities": budget_removed,
            "added_entity_display_index_histogram": dict(sorted(positions.items())),
            "display_positions": position_summary(emitted),
            "analysis_role": "recommended_primary_matched_ablation" if variant == "short_canonical" else "stress_only_not_primary_effect_comparison",
            "used_donor_source_records": len({d["donor_source_record_sha256"] for r in emitted for d in r["metadata"]["input_corruption"]["added_entities"]}),
            "used_donor_source_groups": len({d["donor_source_group_id"] for r in emitted for d in r["metadata"]["input_corruption"]["added_entities"]}),
            "candidate_rejection_counts": dict(rejection_counts),
            "base_token_counts": dict(before_tokens), "corrupted_token_counts": dict(after_tokens),
            "added_prompt_tokens": after_tokens["prompt_tokens"] - before_tokens["prompt_tokens"],
            "max_total_tokens": max(after_lengths), "all_rows_fit_without_truncation": True,
            "completion_bytes_unchanged_per_row": True, "targets_unchanged_per_row": True,
            "annotation_counter_unchanged": True, "row_count_unchanged": True,
            "updates_per_epoch_at_batch1_accum8": math.ceil(len(rows) / 8)}
    _write_rows(output / "donor_pool.jsonl", donors)
    summary["output_sha256"]["donor_pool.jsonl"] = sha((output / "donor_pool.jsonl").read_bytes())
    if previous:
        for name, digest in previous["output_sha256"].items():
            if summary["output_sha256"].get(name) != digest:
                raise ValueError("New release altered a previously frozen data file: " + name)
        summary["previous_release"] = {
            "manifest_sha256": sha((Path(previous_release) / "manifest.json").read_bytes()),
            "existing_data_files_byte_identical": True,
            "preserved_output_sha256": previous["output_sha256"],
            "new_data_files": sorted(set(summary["output_sha256"]) - set(previous["output_sha256"])),
            "new_arm": "short_canonical_distractors"}
    (output / "manifest.json").write_text(canonical(summary) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--public-aggregate")
    parser.add_argument("--previous-release")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    summary = build_release(args.base_release, args.output_dir, tokenizer, seed=args.seed,
                            max_candidates=args.max_candidates, max_length=args.max_length,
                            previous_release=args.previous_release)
    if args.public_aggregate:
        public = Path(args.public_aggregate)
        if public.exists():
            raise ValueError("Refusing to overwrite an existing public aggregate")
        public.parent.mkdir(parents=True, exist_ok=True)
        public.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(canonical(summary))


if __name__ == "__main__":
    main()

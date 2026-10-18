"""Repeat frozen chosen evidence spans in every containing source-only window.

Equal text at a different source location is not migrated. Existing full-source
fallback examples remain explicitly marked auxiliary supervision; inference
does not use their gold-dependent fallback. Ordinary missing labels stay UNKNOWN.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re

from semantic.data import canonical, encode_row, sha, source_windows, validate_row
from semantic import protocol as p
from semantic.runtime import verify_title_context

RELEASE_VERSION = "relation_complete_overlap_v1"
VARIANTS = ("short_source", "short_canonical")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path, values, *, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        if jsonl:
            for value in values:
                handle.write(canonical(value) + "\n")
        else:
            handle.write(canonical(values) + "\n")


def chosen_anchors(full_row, prepared):
    """Recover only the original compiler's minimum prepared evidence hit."""
    validate_row(full_row, require_train=True)
    if full_row["stage"] != "relations" or full_row["variant"] != "full_source":
        raise ValueError("Expected frozen full_source TRAIN relations")
    metadata, text = full_row["metadata"], full_row["task_input"]["text"]
    if (prepared["split"] != "train" or str(prepared["group_id"]) != full_row["group_id"] or
            prepared["original_record_sha256"] != metadata["source_record_sha256"] or
            sha(prepared["original_record"]) != prepared["original_record_sha256"] or
            prepared["source"]["original_text"] != text or sha(text) != metadata["source_text_sha256"]):
        raise ValueError("Prepared TRAIN source identity/body/group mismatch")
    if prepared.get("record_blocked") or prepared["source_type"] != "free_text":
        raise ValueError("Blocked/structured sources cannot supply relation supervision")
    annotations = {a["index"]: a for a in prepared["annotations"]}
    if len(annotations) != len(prepared["annotations"]):
        raise ValueError("Prepared annotation indices are not unique")
    atoms = metadata["supervised_atom_ids"]
    if len(set(atoms)) != len(atoms) or len(atoms) != len(full_row["target"]):
        raise ValueError("Frozen target/atom identities are not one-to-one")
    anchors = {}
    for atom, target in zip(atoms, full_row["target"]):
        match = re.fullmatch(r"relation:(\d+):evidence:(\d+)", atom)
        if not match:
            raise ValueError("Unsupported frozen relation atom identity")
        annotation_index, fragment_index = map(int, match.groups())
        annotation = annotations[annotation_index]
        if annotation.get("pair_blocked") or not annotation["all_evidence_located"]:
            raise ValueError("Frozen atom unexpectedly refers to blocked/unlocated evidence")
        if annotation["annotation"]["relation"] != target["relation"]:
            raise ValueError("Frozen relation label differs from prepared annotation")
        hits = annotation["evidence_fragments"][fragment_index]
        if not hits:
            raise ValueError("Prepared evidence has no source anchor")
        hit = min(hits, key=lambda h: (h["original_start"], h["original_end"]))
        start, end = hit["original_start"], hit["original_end"]
        if not 0 <= start < end <= len(text) or text[start:end] != target["evidence"] or hit["original_text"] != target["evidence"]:
            raise ValueError("Chosen prepared evidence span differs from frozen target bytes")
        anchors[atom] = {"start": start, "end": end, "evidence_sha256": sha(target["evidence"]),
                         "prepared_evidence_hit_count": len(hits)}
    return anchors


def compile_source_rows(full_row, prepared, old_views, title_receipts, *, short_chars=2400, overlap=400):
    """Reconstruct old assignments before expanding only chosen-span overlap."""
    anchors = chosen_anchors(full_row, prepared)
    return compile_from_anchors(full_row, anchors, old_views, title_receipts,
        provenance={"version": RELEASE_VERSION, "prepared_record_sha256": sha(prepared)},
        short_chars=short_chars, overlap=overlap)


def compile_from_anchors(full_row, anchors, old_views, title_receipts, *, provenance,
                         anchor_scope="chosen_prepared_hit_only", short_chars=2400, overlap=400):
    """Shared window expansion; caller must establish anchor provenance first."""
    validate_row(full_row, require_train=True)
    if full_row["stage"] != "relations" or full_row["variant"] != "full_source":
        raise ValueError("Shared expansion requires frozen full_source TRAIN relations")
    metadata, text = full_row["metadata"], full_row["task_input"]["text"]
    if sha(text) != metadata["source_text_sha256"]:
        raise ValueError("Full-source body hash drift")
    if metadata.get("title_context"):
        verify_title_context(metadata, title_receipts, source_group_id=full_row["group_id"])
    source_key = metadata["source_record_sha256"]
    table = full_row["task_input"]["entities"]
    by_id = {e["id"]: e for e in table}
    targets = dict(zip(metadata["supervised_atom_ids"], full_row["target"]))
    if set(anchors) != set(targets):
        raise ValueError("Chosen anchor identities do not cover exact frozen targets")
    for atom, anchor in anchors.items():
        a, b = anchor["start"], anchor["end"]
        if not 0 <= a < b <= len(text) or text[a:b] != targets[atom]["evidence"] or sha(text[a:b]) != anchor["evidence_sha256"]:
            raise ValueError("Chosen source anchor does not match exact frozen evidence")
    full_rank = {atom: i for i, atom in enumerate(targets)}
    windows = source_windows(text, short_chars, overlap)
    first_assignment, complete_assignment = defaultdict(list), defaultdict(list)
    atom_audits = []
    for atom, anchor in anchors.items():
        containing = [i for i, (a, b) in enumerate(windows) if a <= anchor["start"] and anchor["end"] <= b]
        first = containing[0] if containing else -1
        first_assignment[first].append(atom)
        for index in containing or [-1]:
            complete_assignment[index].append(atom)
        atom_audits.append({"source_record_sha256": source_key, "source_index": metadata["source_index"],
            "atom_id": atom, "chosen_anchor": deepcopy(anchor), "old_assigned_window": first,
            "new_assigned_windows": containing or [-1], "new_overlap_occurrences": max(0, len(containing) - 1),
            "scope": anchor_scope, "auxiliary_full_source_only": not containing})
    results = {}
    for variant in VARIANTS:
        old_rows = old_views[variant]
        old_by_window = {}
        for old in old_rows:
            validate_row(old, require_train=True)
            index = old["metadata"]["source_window"]["index"]
            if index in old_by_window:
                raise ValueError("Duplicate frozen source-window row")
            old_by_window[index] = old
            if (old["variant"] != variant or old["stage"] != "relations" or
                    old["metadata"]["source_record_sha256"] != source_key or
                    old["group_id"] != full_row["group_id"] or old["task_input"]["entities"] != table or
                    old["metadata"].get("title_context") != metadata.get("title_context")):
                raise ValueError("Frozen view source/complete entity-table drift")
            a, b = (0, len(text)) if index == -1 else windows[index]
            if old["task_input"]["text"] != text[a:b] or old["metadata"]["source_window"] != {
                    "start": a, "end": b, "index": index, "full_source_fallback": index == -1}:
                raise ValueError("Frozen source-only window schedule differs")
            actual = dict(zip(old["metadata"]["supervised_atom_ids"], old["target"]))
            expected = {atom: targets[atom] for atom in first_assignment[index]}
            if len(actual) != old["target_count"] or actual != expected:
                raise ValueError("Frozen earliest-window target assignment differs from chosen anchors")
        if set(old_by_window) != set(first_assignment):
            raise ValueError("Frozen earliest-window row coverage differs")
        order = "canonical" if variant == "short_canonical" else "source"
        output = []
        for index, selected_atoms in sorted(complete_assignment.items()):
            selected = list(selected_atoms)
            if order == "source":
                selected.sort(key=full_rank.__getitem__)
            else:
                def semantic_key(atom):
                    t = targets[atom]
                    source, target = by_id[t["source"]], by_id[t["target"]]
                    return (source["label"], p.normalize_name(source["name"]), target["label"],
                            p.normalize_name(target["name"]), t["relation"], anchors[atom]["start"])
                selected.sort(key=semantic_key)
            a, b = (0, len(text)) if index == -1 else windows[index]
            clean = [deepcopy(targets[atom]) for atom in selected]
            local_metadata = {**deepcopy(metadata), "variant": variant, "order": order,
                "source_window": {"start": a, "end": b, "index": index, "full_source_fallback": index == -1},
                "relation_entity_table_scope": "complete_source_entities", "supervised_atom_ids": selected,
                "relation_window_supervision": {**deepcopy(provenance),
                    "scope": "frozen_chosen_evidence_span_in_all_containing_source_windows",
                    "base_full_row_sha256": sha(full_row),
                    "other_equal_text_occurrences_migrated": False,
                    "auxiliary_full_source_training_only": index == -1,
                    "runtime_gold_fallback_allowed": False,
                    "unknown_windows_are_not_negative_examples": True}}
            inputs = {"text": text[a:b], "entities": deepcopy(table), "metadata": local_metadata}
            row = {"data_version": full_row["data_version"],
                "id": f"train:{metadata['source_index']}:{variant}:relations:{index}",
                "split": "train", "stage": "relations", "variant": variant, "group_id": full_row["group_id"],
                "metadata": local_metadata, "task_input": inputs,
                "prompt": p.build_relation_prompt(inputs["text"], table, local_metadata, order),
                "completion": p.render_relations(clean, table), "target": clean, "target_count": len(clean),
                "input_sha256": sha(inputs), "target_sha256": sha(clean)}
            validate_row(row, require_train=True)
            if index == -1:
                old = old_by_window[index]
                if any(row[key] != old[key] for key in ("prompt", "completion", "target", "target_sha256", "target_count")):
                    raise ValueError("Auxiliary full-source fallback target/prompt changed")
            output.append(row)
        results[variant] = output
    left, right = results[VARIANTS[0]], results[VARIANTS[1]]
    for source_row, canonical_row in zip(left, right):
        if source_row["metadata"]["source_window"] != canonical_row["metadata"]["source_window"] or dict(zip(
                source_row["metadata"]["supervised_atom_ids"], source_row["target"])) != dict(zip(
                canonical_row["metadata"]["supervised_atom_ids"], canonical_row["target"])):
            raise ValueError("Order variants changed atom/window exposure")
    return results, atom_audits, {"source_record_sha256": source_key, "source_index": metadata["source_index"],
        "source_only_windows": len(windows), "positive_regular_windows": sum(i >= 0 for i in complete_assignment),
        "unknown_regular_window_indices": [i for i in range(len(windows)) if i not in complete_assignment]}


def _quantiles(values):
    values = sorted(values)
    return {"min": values[0], "p50": values[math.ceil(len(values) * .5) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1], "max": values[-1]}


def build_release(base_release, prepared_source, audit_path, output_dir, tokenizer, *, max_length=8192):
    base, source_path, output = Path(base_release), Path(prepared_source), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite relation overlap release")
    manifest = json.loads((base / "manifest.json").read_text())
    audit = json.loads(Path(audit_path).read_text())
    if audit["base_release_manifest_sha256"] != sha((base / "manifest.json").read_bytes()):
        raise ValueError("Independent audit is for another frozen release")
    names = ["full_source/train_relations.jsonl", "title_source_receipts.json", "evaluation/train_records.jsonl"]
    names += [f"{v}/train_relations.jsonl" for v in VARIANTS]
    for name in names:
        if sha((base / name).read_bytes()) != manifest["output_sha256"][name]:
            raise ValueError("Frozen data file SHA drift: " + name)
    if sha(source_path.read_bytes()) != manifest["source_artifacts"]["records.jsonl"]:
        raise ValueError("Frozen prepared source SHA drift")
    if sha(Path(p.__file__).read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol SHA drift")
    prepared_rows = _read(source_path)
    train_groups = {str(r["group_id"]) for r in prepared_rows if r["split"] == "train"}
    held_groups = {str(r["group_id"]) for r in prepared_rows if r["split"] != "train"}
    if train_groups & held_groups:
        raise ValueError("Frozen source group leakage")
    prepared = {r["original_record_sha256"]: r for r in prepared_rows if r["split"] == "train"}
    full = _read(base / names[0])
    old_views = {v: defaultdict(list) for v in VARIANTS}
    for variant in VARIANTS:
        for row in _read(base / f"{variant}/train_relations.jsonl"):
            old_views[variant][row["metadata"]["source_record_sha256"]].append(row)
    titles = json.loads((base / "title_source_receipts.json").read_text())
    rows, atom_audits, source_audits = {v: [] for v in VARIANTS}, [], []
    original_atoms, original_labels = Counter(), Counter()
    for full_row in full:
        key = full_row["metadata"]["source_record_sha256"]
        compiled, atom_trace, source_trace = compile_source_rows(full_row, prepared[key],
            {v: old_views[v][key] for v in VARIANTS}, titles)
        for variant in VARIANTS:
            rows[variant].extend(compiled[variant])
        atom_audits.extend(atom_trace)
        source_audits.append(source_trace)
        original_atoms.update((key, atom) for atom in full_row["metadata"]["supervised_atom_ids"])
        original_labels.update(t["relation"] for t in full_row["target"])
    return publish_release(base, output, tokenizer, manifest, names, full, rows, atom_audits, source_audits,
        original_atoms, original_labels, audit, max_length=max_length,
        provenance={"release_version": RELEASE_VERSION, "prepared_source_sha256": sha(source_path.read_bytes()),
                    "independent_input_audit_sha256": sha(Path(audit_path).read_bytes()),
                    "compiler_sha256": sha(Path(__file__).read_bytes())})


def publish_release(base, output, tokenizer, manifest, names, full, rows, atom_audits, source_audits,
                    original_atoms, original_labels, audit, *, max_length, provenance):
    """Shared paired-view checks, tokenization and immutable artifact writer."""
    counters = {v: Counter((r["metadata"]["source_record_sha256"], atom) for r in rows[v]
                          for atom in r["metadata"]["supervised_atom_ids"]) for v in VARIANTS}
    if counters[VARIANTS[0]] != counters[VARIANTS[1]] or set(counters[VARIANTS[0]]) != set(original_atoms):
        raise ValueError("Order-paired atom Counter or unique coverage mismatch")
    if any(n != 1 for n in original_atoms.values()):
        raise ValueError("Original annotation atom identities are not unique")
    exposures = sum(counters[VARIANTS[0]].values())
    extra = exposures - sum(original_atoms.values())
    expected = audit["counts"]
    if (extra != expected["missing_frozen_chosen_anchor_overlap"] or exposures != expected[
            "projected_chosen_anchor_fix_exposures_including_unchanged_fallback"] or
            len(rows[VARIANTS[0]]) != expected["projected_chosen_anchor_fix_rows_including_unchanged_fallback"]):
        raise ValueError("Independent chosen-span audit counts do not match release")
    outputs, variants = {}, {}
    for variant in VARIANTS:
        token_counts, lengths, target_lengths, label_exposure = Counter(), [], [], Counter()
        for row in rows[variant]:
            encoded = encode_row(row, tokenizer, max_length)
            token_counts.update({k: encoded["_choice_counts"][k] for k in ("prompt_tokens", "completion_tokens")})
            lengths.append(len(encoded["input_ids"]))
            target_lengths.append(encoded["_choice_counts"]["completion_tokens"])
            label_exposure.update(t["relation"] for t in row["target"])
        relative = f"{variant}/train_relations.jsonl"
        _write(output / relative, rows[variant], jsonl=True)
        outputs[relative] = sha((output / relative).read_bytes())
        variants[variant] = {"rows": len(rows[variant]), "supervised_eos_per_epoch": len(rows[variant]),
            "unique_source_atom_count": len(counters[variant]), "annotation_exposures": exposures,
            "token_counts": {**dict(token_counts), "total_tokens": sum(lengths)},
            "total_token_quantiles": _quantiles(lengths), "completion_token_quantiles": _quantiles(target_lengths),
            "updates_per_epoch_at_batch1_accum8": math.ceil(len(rows[variant]) / 8),
            "label_exposure_counts": dict(sorted(label_exposure.items()))}
    for name, contents in (("chosen_anchor_audit.jsonl", atom_audits), ("source_window_audit.jsonl", source_audits)):
        _write(output / name, contents, jsonl=True)
        outputs[name] = sha((output / name).read_bytes())
    train_free = [r for r in _read(base / "evaluation/train_records.jsonl") if r["source_type"] == "free_text"]
    all_windows = sum(len(source_windows(r["text"], 2400, 400)) for r in train_free)
    positive_regular = sum(a["positive_regular_windows"] for a in source_audits)
    fallback_atoms = sum(a["auxiliary_full_source_only"] for a in atom_audits)
    fallback_rows = sum(r["metadata"]["source_window"]["index"] == -1 for r in rows[VARIANTS[0]])
    summary = {**deepcopy(provenance), "status": "compiled_not_trained",
        "base_release_manifest_sha256": sha((base / "manifest.json").read_bytes()),
        "base_files_sha256": {name: manifest["output_sha256"][name] for name in names},
        "protocol_sha256": manifest["protocol_sha256"],
        "source_records": len(full), "source_groups": len({r["group_id"] for r in full}), "split": "train",
        "train_heldout_group_overlap": 0, "parameters": {"short_chars": 2400, "overlap": 400, "max_length": max_length},
        "original_unique_source_atom_count": len(original_atoms), "unique_source_atom_coverage_unchanged": True,
        "annotation_exposures_per_epoch": exposures, "added_chosen_span_overlap_exposures": extra,
        "other_equal_text_occurrence_omissions_not_migrated": expected["missing_other_evidence_occurrence_scope_unverified"],
        "per_atom_exposure_histogram": dict(sorted(Counter(counters[VARIANTS[0]].values()).items())),
        "paired_views_same_atom_counter_windows_eos_updates": True,
        "regular_positive_windows": positive_regular, "all_train_free_text_source_only_windows": all_windows,
        "unknown_windows_not_trained": all_windows - positive_regular,
        "unknown_windows_in_relation_admitted_sources": sum(len(a["unknown_regular_window_indices"]) for a in source_audits),
        "auxiliary_full_source_rows": fallback_rows, "auxiliary_full_source_atoms": fallback_atoms,
        "runtime_uses_gold_fallback": False, "all_target_text_names_labels_unchanged": True,
        "original_unique_label_counts": dict(sorted(original_labels.items())),
        "rare_labels_original_count_at_most_9": {k: {"unique_atoms": n,
            "window_exposures": variants[VARIANTS[0]]["label_exposure_counts"][k]}
            for k, n in sorted(original_labels.items()) if n <= 9},
        "variants": variants, "output_sha256": outputs,
        "limitations": ["Chosen prepared spans are inherited algorithmic silver matches, not clinical adjudication.",
                        "Other equal-text occurrences remain scope-unverified and are not migrated.",
                        "Empty/unannotated source windows stay UNKNOWN, not negative examples.",
                        "Three original long atoms remain in two full-context auxiliary training rows; short runtime has no gold-dependent fallback.",
                        "Compared with old first-window-only training, this changes repeated exposure, EOS count and update count; only the new two order views are matched."]}
    _write(output / "manifest.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--prepared-source", required=True)
    parser.add_argument("--input-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(canonical(build_release(args.base_release, args.prepared_source, args.input_audit,
        args.output_dir, tokenizer, max_length=args.max_length)))


if __name__ == "__main__":
    main()

"""Repeat every locally grounded admitted silver entity in every source window.

Window boundaries depend only on source text. This repairs first-window-only
supervision, not the underlying teacher annotations' incompleteness. Windows
without admitted positives remain UNKNOWN and never become empty negatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from semantic.data import DATA_VERSION, canonical, encode_row, sha, source_windows, validate_row
from semantic import protocol as p
from semantic.runtime import verify_title_context

RELEASE_VERSION = "entity_complete_windows_v1"
VARIANT = "short_source_complete_windows"


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def compile_source_row(base_row, title_receipts, *, short_chars=2400, overlap=400):
    """Return training rows plus every source-window's no-text admission trace."""
    validate_row(base_row, require_train=True)
    if base_row["stage"] != "entities" or base_row["variant"] != "full_source":
        raise ValueError("Requires frozen full_source TRAIN entity rows")
    text, original_metadata = base_row["task_input"]["text"], base_row["metadata"]
    window = original_metadata["source_window"]
    if (window["start"], window["end"]) != (0, len(text)) or sha(text) != original_metadata["source_text_sha256"]:
        raise ValueError("Full source identity/window mismatch")
    if original_metadata.get("title_context"):
        verify_title_context(original_metadata, title_receipts, source_group_id=base_row["group_id"])
    targets, atoms = base_row["target"], original_metadata["supervised_atom_ids"]
    if len(targets) != len(atoms) or len(set(atoms)) != len(atoms):
        raise ValueError("Frozen target/annotation identity is not one-to-one")
    if len({(e["label"], p.normalize_name(e["name"])) for e in targets}) != len(targets):
        raise ValueError("Frozen entity identities are not unique")
    rows, audits, covered = [], [], set()
    for index, (start, end) in enumerate(source_windows(text, short_chars, overlap)):
        local_text = text[start:end]
        metadata = {**deepcopy(original_metadata), "variant": VARIANT, "order": "source",
            "source_window": {"start": start, "end": end, "index": index, "full_source_fallback": False}}
        selected, reasons, mention_choices = [], Counter(), Counter()
        for target, atom in zip(targets, atoms):
            # Only the frozen canonical name is eligible; no alias/name is invented.
            if not p._anchors(target["name"], local_text, metadata):
                reasons["canonical_name_not_locally_grounded"] += 1
                continue
            anchors = p._anchors(target["mention"], local_text, metadata)
            choice = "original_mention_surface"
            if not anchors:
                anchors = p._anchors(target["name"], local_text, metadata)
                choice = "exact_local_canonical_name_surface"
            anchor = min(anchors, key=lambda a: (a["context"] != "body", a["start"], a["end"]))
            candidate = {"id": target["id"], "label": target["label"], "name": target["name"],
                         "mention": anchor["text"]}
            parsed = p.parse_entities(local_text, p.render_entities([candidate]), metadata)
            supported = parsed.get("filtered_entities", [])
            if not parsed.get("valid") or len(supported) != 1:
                raise ValueError("Locally grounded frozen entity failed protocol; refusing silent omission")
            candidate["mentions"] = deepcopy(supported[0]["mentions"])
            candidate["grounding_verified"] = True
            # Keep trusted-title-only mentions before body positions, as source
            # ordering in the frozen compiler does; internal IDs never choose order.
            position = anchor["start"] if anchor["context"] == "body" else -1
            selected.append((position, anchor["end"], atom, candidate))
            mention_choices[choice] += 1
        selected.sort(key=lambda item: (item[0], item[1], item[2]))
        local_atoms = [item[2] for item in selected]
        audit = {"source_record_sha256": original_metadata["source_record_sha256"],
                 "source_group_id": base_row["group_id"], "source_window": metadata["source_window"],
                 "target_count": len(selected), "unavailable_target_reasons": dict(reasons),
                 "mention_selection_counts": dict(mention_choices),
                 "status": "supervised_positive_annotations" if selected else "unknown_no_admitted_annotations_not_trained"}
        audits.append(audit)
        if not selected:
            continue
        metadata["supervised_atom_ids"] = local_atoms
        metadata["entity_window_supervision"] = {
            "version": RELEASE_VERSION, "base_row_id": base_row["id"], "base_row_sha256": sha(base_row),
            "scope": "all_locally_groundable_frozen_admitted_annotations",
            "window_selection": "source_only_all_windows_no_gold_fallback",
            "canonical_names_and_labels_unchanged": True,
            "repeat_exposure_across_windows": True,
            "not_exhaustive_clinical_annotation": True,
            "mention_selection_counts": dict(mention_choices)}
        clean = [item[3] for item in selected]
        task_input = {"text": local_text, "entities": [], "metadata": metadata}
        row = {"data_version": DATA_VERSION, "id": base_row["id"] + f":{RELEASE_VERSION}:{index}",
               "split": "train", "stage": "entities", "variant": VARIANT, "group_id": base_row["group_id"],
               "metadata": metadata, "task_input": task_input,
               "prompt": p.build_entity_prompt(local_text, metadata, "source"),
               "completion": p.render_entities(clean), "target": clean, "target_count": len(clean),
               "input_sha256": sha(task_input), "target_sha256": sha(clean)}
        validate_row(row, require_train=True)
        covered.update(local_atoms)
        rows.append(row)
    if covered != set(atoms):
        raise ValueError(f"Source-only windows cannot cover {len(set(atoms) - covered)} admitted entities; no gold fallback permitted")
    return rows, audits


def build_release(base_release, output_dir, tokenizer, *, short_chars=2400, overlap=400, max_length=8192):
    base, output = Path(base_release), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite an existing entity-window release")
    manifest = json.loads((base / "manifest.json").read_text())
    names = ["full_source/train_entities.jsonl", "short_source/train_entities.jsonl",
             "title_source_receipts.json", "evaluation/train_records.jsonl"]
    for name in names:
        if sha((base / name).read_bytes()) != manifest["output_sha256"][name]:
            raise ValueError("Frozen source data SHA drift: " + name)
    if sha(Path(p.__file__).read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol drift")
    title_receipts = json.loads((base / "title_source_receipts.json").read_text())
    full = _read(base / names[0])
    old_short = _read(base / names[1])
    originals = Counter((r["metadata"]["source_record_sha256"], atom) for r in full for atom in r["metadata"]["supervised_atom_ids"])
    emitted, audits, exposure, token_counts, lengths, targets = [], [], Counter(), Counter(), [], []
    for original in full:
        rows, traces = compile_source_row(original, title_receipts, short_chars=short_chars, overlap=overlap)
        audits.extend(traces)
        for row in rows:
            encoded = encode_row(row, tokenizer, max_length)
            token_counts.update({k:encoded["_choice_counts"][k] for k in ("prompt_tokens", "completion_tokens")})
            lengths.append(len(encoded["input_ids"]))
            targets.append(row["target_count"])
            exposure.update((row["metadata"]["source_record_sha256"], atom) for atom in row["metadata"]["supervised_atom_ids"])
        emitted.extend(rows)
    if set(exposure) != set(originals) or any(v != 1 for v in originals.values()):
        raise ValueError("Unique document-level annotation identity coverage mismatch")
    records = [r for r in _read(base / names[3]) if r["source_type"] == "free_text"]
    represented = {r["metadata"]["source_record_sha256"] for r in full}
    all_windows = sum(len(source_windows(r["text"], short_chars, overlap)) for r in records)
    excluded_windows = sum(len(source_windows(r["text"], short_chars, overlap)) for r in records if r["original_record_sha256"] not in represented)
    relative = f"{VARIANT}/train_entities.jsonl"
    _write_rows(output / relative, emitted)
    _write_rows(output / "window_audit.jsonl", audits)
    _write_rows(output / "unknown_windows.jsonl", [r for r in audits if not r["target_count"]])
    summary = {"release_version": RELEASE_VERSION, "status": "compiled_not_trained", "data_version": DATA_VERSION,
        "variant": VARIANT, "base_release_manifest_sha256": sha((base / "manifest.json").read_bytes()),
        "protocol_sha256": manifest["protocol_sha256"], "compiler_sha256": sha(Path(__file__).read_bytes()),
        "base_files_sha256": {name:manifest["output_sha256"][name] for name in names},
        "parameters": {"short_chars": short_chars, "overlap": overlap, "max_length": max_length},
        "source_records": len(full), "source_groups": len({r["group_id"] for r in full}), "split": "train",
        "rows": len(emitted), "old_first_window_only_rows": len(old_short),
        "unique_document_entity_atoms": len(originals), "unique_atom_coverage_unchanged": True,
        "total_supervised_entity_window_exposures": sum(exposure.values()),
        "extra_entity_exposures_vs_original": sum(exposure.values()) - sum(originals.values()),
        "per_atom_window_exposure_histogram": dict(sorted(Counter(exposure.values()).items())),
        "annotation_counter_equal_to_old": exposure == originals,
        "source_only_windows_represented_train_sources": len(audits),
        "unknown_no_annotation_windows_not_trained": sum(not a["target_count"] for a in audits),
        "all_train_free_text_sources": len(records), "all_train_free_text_source_only_windows": all_windows,
        "excluded_source_windows_no_admitted_train_source": excluded_windows,
        "inference_minus_supervised_window_count": all_windows - len(emitted),
        "source_only_full_fallbacks": 0,
        "token_counts": {**dict(token_counts), "total_tokens": sum(lengths)},
        "max_total_tokens": max(lengths), "max_entities_per_window": max(targets),
        "updates_per_epoch_at_batch1_accum8": math.ceil(len(emitted) / 8),
        "all_targets_protocol_verified_and_untruncated": True,
        "output_sha256": {name:sha((output / name).read_bytes()) for name in
                          [relative, "window_audit.jsonl", "unknown_windows.jsonl"]},
        "limitations": ["Teacher silver remains incomplete; all locally grounded admitted labels is not all true entities.",
                        "Empty/unannotated windows remain UNKNOWN; inference includes them but training does not supervise empty negatives.",
                        "Overlapping/repeated mentions and visible trusted titles intentionally repeat the same document entity supervision.",
                        "This changes exposure and update counts; it is not an atom-Counter-matched length ablation."]}
    (output / "manifest.json").write_text(canonical(summary) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--short-chars", type=int, default=2400)
    parser.add_argument("--overlap", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(canonical(build_release(args.base_release, args.output_dir, tokenizer,
        short_chars=args.short_chars, overlap=args.overlap, max_length=args.max_length)))


if __name__ == "__main__":
    main()

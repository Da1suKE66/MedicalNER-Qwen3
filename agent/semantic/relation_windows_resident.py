"""Rebuild relation overlap supervision exclusively from resident frozen data.

A separately approved aggregate receipt pins the exact 5551-anchor ledger for
this release. No prepared-source records or per-atom external ledger is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from data_repair import text_normalization as matcher
from semantic import protocol as p
from semantic.data import canonical, sha, validate_row
from semantic import relation_windows as shared

RELEASE_VERSION = "relation_complete_overlap_v2"
RECEIPT_VERSION = "resident-chosen-anchor-equivalence-v1"


def load_receipt(path, expected_sha256):
    raw = Path(path).read_bytes()
    if len(expected_sha256) != 64 or sha(raw) != expected_sha256:
        raise ValueError("Approved aggregate receipt SHA mismatch")
    receipt = json.loads(raw)
    if (receipt.get("receipt_version") != RECEIPT_VERSION or receipt.get("all_original_chosen_spans_equal") is not True or
            receipt.get("prepared_chosen_anchor_ledger_sha256") != receipt.get("resident_chosen_anchor_ledger_sha256")):
        raise ValueError("Aggregate receipt does not certify exact original/resident anchor equivalence")
    return receipt


def reconstruct_anchors(full_rows, receipt):
    """Require exact source slices and a complete independently pinned ledger."""
    if sha(Path(matcher.__file__).read_bytes()) != receipt["matcher_implementation_sha256"]:
        raise ValueError("Frozen source matcher implementation drift")
    anchors_by_source, ledger, identities = {}, [], set()
    for row in full_rows:
        validate_row(row, require_train=True)
        if row["stage"] != "relations" or row["variant"] != "full_source":
            raise ValueError("Resident reconstruction requires full_source TRAIN relations")
        metadata, text = row["metadata"], row["task_input"]["text"]
        key = metadata["source_record_sha256"]
        if key in anchors_by_source or sha(text) != metadata["source_text_sha256"]:
            raise ValueError("Resident source identity/body drift or duplicate source row")
        window = metadata["source_window"]
        if window["start"] != 0 or window["end"] != len(text):
            raise ValueError("Resident reconstruction requires the complete source text")
        atoms = metadata["supervised_atom_ids"]
        if len(atoms) != len(row["target"]) or len(set(atoms)) != len(atoms):
            raise ValueError("Resident target/atom identity is not one-to-one")
        anchors = {}
        for atom, target in zip(atoms, row["target"]):
            hits = matcher.find_mentions(target["evidence"], text)
            if not hits:
                raise ValueError("Frozen evidence has no resident source match")
            start, end = min(hits)
            if text[start:end] != target["evidence"]:
                raise ValueError("Earliest resident evidence match differs in original source bytes")
            identity = (key, atom)
            if identity in identities:
                raise ValueError("Duplicate resident source/atom identity")
            identities.add(identity)
            anchors[atom] = {"start": start, "end": end, "evidence_sha256": sha(target["evidence"]),
                             "resident_evidence_hit_count": len(hits)}
            ledger.append({"source_record_sha256": key, "atom_id": atom,
                           "evidence_sha256": sha(target["evidence"]), "start": start, "end": end})
        anchors_by_source[key] = anchors
    ledger.sort(key=lambda r: (r["source_record_sha256"], r["atom_id"]))
    if len(ledger) != receipt["expected_unique_atom_count"]:
        raise ValueError("Resident anchor ledger atom count mismatch")
    if sha(ledger) != receipt["resident_chosen_anchor_ledger_sha256"]:
        raise ValueError("Resident chosen-anchor ledger SHA mismatch; counts alone do not admit data")
    return anchors_by_source, ledger


def build_release(base_release, receipt_path, output_dir, tokenizer, *, receipt_sha256, max_length=8192):
    base, output = Path(base_release), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite resident relation overlap release")
    receipt = load_receipt(receipt_path, receipt_sha256)
    base_raw = (base / "manifest.json").read_bytes()
    if sha(base_raw) != receipt["base_release_manifest_sha256"]:
        raise ValueError("Resident base manifest differs from independently verified release")
    manifest = json.loads(base_raw)
    names = ["full_source/train_relations.jsonl", "title_source_receipts.json", "evaluation/train_records.jsonl"]
    names += [f"{v}/train_relations.jsonl" for v in shared.VARIANTS]
    for name in names:
        if sha((base / name).read_bytes()) != manifest["output_sha256"][name]:
            raise ValueError("Resident frozen file SHA drift: " + name)
    if sha((base / names[0]).read_bytes()) != receipt["full_relation_file_sha256"]:
        raise ValueError("Resident full relation file differs from anchor equivalence receipt")
    if sha(Path(p.__file__).read_bytes()) != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol implementation drift")
    if manifest["train_tune_group_overlap"] != 0:
        raise ValueError("Frozen base does not certify source-group separation")
    full = shared._read(base / names[0])
    anchors, ledger = reconstruct_anchors(full, receipt)
    train_records = shared._read(base / "evaluation/train_records.jsonl")
    by_source = {r["original_record_sha256"]: r for r in train_records}
    for row in full:
        source = by_source[row["metadata"]["source_record_sha256"]]
        if (source["split"] != "train" or str(source["source_group_id"]) != row["group_id"] or
                source["text"] != row["task_input"]["text"]):
            raise ValueError("Resident training source identity differs from frozen evaluation input")
    old_views = {v: defaultdict(list) for v in shared.VARIANTS}
    for variant in shared.VARIANTS:
        for row in shared._read(base / f"{variant}/train_relations.jsonl"):
            old_views[variant][row["metadata"]["source_record_sha256"]].append(row)
    titles = json.loads((base / "title_source_receipts.json").read_text())
    rows, atom_audits, source_audits = {v: [] for v in shared.VARIANTS}, [], []
    original_atoms, original_labels = Counter(), Counter()
    row_provenance = {"version": RELEASE_VERSION, "anchor_reconstruction": "resident_full_source_minimum_match",
        "anchor_ledger_sha256": sha(ledger), "approved_equivalence_receipt_sha256": receipt_sha256,
        "prior_prepared_equivalence_audit_sha256": receipt["prior_prepared_equivalence_audit_sha256"],
        "original_prepared_source_independently_verified_equal": True,
        "external_prepared_records_read": False}
    for full_row in full:
        key = full_row["metadata"]["source_record_sha256"]
        compiled, atom_trace, source_trace = shared.compile_from_anchors(full_row, anchors[key],
            {v: old_views[v][key] for v in shared.VARIANTS}, titles, provenance=row_provenance,
            anchor_scope="resident_minimum_hit_verified_equal_to_original_prepared_chosen_span")
        for variant in shared.VARIANTS:
            rows[variant].extend(compiled[variant])
        atom_audits.extend(atom_trace)
        source_audits.append(source_trace)
        original_atoms.update((key, atom) for atom in full_row["metadata"]["supervised_atom_ids"])
        original_labels.update(t["relation"] for t in full_row["target"])
    audit = {"counts": receipt["expected_overlap_counts"]}
    provenance = {"release_version": RELEASE_VERSION, "compiler_sha256": sha(Path(__file__).read_bytes()),
        "shared_window_compiler_sha256": sha(Path(shared.__file__).read_bytes()),
        "independent_input_audit_sha256": receipt_sha256,
        "resident_reconstruction": {"method": "min(find_mentions(frozen evidence, complete resident source))",
            "anchor_ledger_sha256": sha(ledger), "anchor_count": len(ledger),
            "approved_aggregate_receipt_sha256": receipt_sha256,
            "prior_prepared_equivalence_audit_sha256": receipt["prior_prepared_equivalence_audit_sha256"],
            "prior_prepared_records_sha256": receipt["prior_prepared_records_sha256"],
            "matcher_implementation_sha256": receipt["matcher_implementation_sha256"],
            "external_prepared_records_read": False, "external_per_atom_ledger_read": False,
            "old_original_anchor_equivalence_verified_for_this_frozen_release": True}}
    return shared.publish_release(base, output, tokenizer, manifest, names, full, rows, atom_audits,
        source_audits, original_atoms, original_labels, audit, max_length=max_length, provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--equivalence-receipt", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(canonical(build_release(args.base_release, args.equivalence_receipt, args.output_dir, tokenizer,
        receipt_sha256=args.receipt_sha256, max_length=args.max_length)))


if __name__ == "__main__":
    main()

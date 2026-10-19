"""Deterministic input-order / short-output views of frozen partial SFT rows.

These are dense-pair diagnostics, not sparse-output or local-context training.
Every supervised directed decision (including all its labels and evidence) is
preserved exactly once per dataset pass. Candidate chunks are chosen from input
order alone. Entirely UNKNOWN chunks are omitted because they have no admissible
SFT supervision; their counters are audited separately. No source text is cut.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from generative.protocol import ProtocolConfig, digest, dumps
from generative.supervision_guard import audit_partial_supervision
from robustness.entity_noise import fingerprint, prompt_payload, protocol_prefix
from .abstention_repair import validate_case
from .controlled_experiments import audit_rows, read_rows
from .release_v2 import canonical, sha
from .supervision_v2 import KNOWN, serialize_partial, validate_partial_row
from .text_normalization import find_mentions


VERSION = "rapid-frozen-decisions-20261012-v1"
VARIANTS = {
    "V4_control": {"order": "hash", "pairs_per_batch": 16},
    "V4_source_order": {"order": "source", "pairs_per_batch": 16},
    "V4_hash_short4": {"order": "hash", "pairs_per_batch": 4},
    "V4_source_short4": {"order": "source", "pairs_per_batch": 4},
}


def input_pair_order(row, order):
    """Read only entity names/types, endpoint IDs and numbered input text.

    Hash order is the existing frozen order. Source order mirrors the shared
    inference protocol's entity-first traversal, with absent synthetic entities
    placed last. No labels, evidence, or supervision status enters this key.
    """
    if order == "hash":
        by_id = {entity["id"]: entity for entity in row["entities"]}
        return sorted(row["pairs"], key=lambda pair: digest(dumps([
            (by_id[pair[key]]["name"], by_id[pair[key]]["label"]) for key in ("source", "target")])))
    if order != "source":
        raise ValueError("Unsupported pair order")
    text = "\n".join(span["text"] for span in row["spans"])
    def key(entity):
        hits = find_mentions(entity["name"], text)
        return (hits[0][0] if hits else len(text) + 1,
                entity["label"], entity["name"].casefold(), entity["id"])
    ranks = {entity["id"]: index for index, entity in enumerate(sorted(row["entities"], key=key))}
    return sorted(row["pairs"], key=lambda pair: (ranks[pair["source"]], ranks[pair["target"]]))


def _origin(row, pid):
    for field in ("rapid_variant_provenance", "rapid_merge_provenance"):
        if field in row:
            return row[field]["pair_sources"][pid]
    return {"parent_row_id": row["id"], "parent_pair_id": pid,
            "parent_row_sha256": fingerprint(row),
            "parent_abstention_provenance": row.get("abstention_provenance")}


def select_rows(rows, *, base_rows=192, synthetic_rows=64):
    """Freeze metadata-stratified rows before inspecting any model outcome."""
    if base_rows is None and synthetic_rows is None:
        return list(rows)
    if any(type(value) is not int or value < 1 for value in (base_rows, synthetic_rows)):
        raise ValueError("Both selection strata require positive integer budgets")
    strata = {False: [], True: []}
    for row in rows:
        strata["abstention_provenance" in row].append(row)
    selected = []
    for synthetic, count in ((False, base_rows), (True, synthetic_rows)):
        if len(strata[synthetic]) < count:
            raise ValueError("Insufficient rows in requested selection stratum")
        ordered = sorted(strata[synthetic], key=lambda row: sha(canonical(
            [VERSION, 42, row["group_id"], row["sample_id"], row["id"]])))
        selected.extend(ordered[:count])
    # Keep the same parent order in the control as in the frozen source.
    wanted = {row["id"] for row in selected}
    return [row for row in rows if row["id"] in wanted]


def merge_sources(rows):
    """Merge selected candidate blocks using source/provenance identity only."""
    groups = {}
    for row in rows:
        validate_partial_row(row)
        if row.get("rapid_variant_provenance") or row.get("rapid_merge_provenance"):
            raise ValueError("Variants must start from original frozen rows")
        if row.get("split") != "train":
            raise ValueError("Only frozen training rows are eligible")
        if "abstention_provenance" in row:
            validate_case(row)
        key = canonical([row["group_id"], row["sample_id"], row["split"], row["spans"],
                         row.get("original_spans"), row.get("sample_weight", 1.0),
                         row.get("annotation_source"), row.get("data_version"),
                         row.get("abstention_provenance")])
        groups.setdefault(key, []).append(row)
    merged = []
    for key, parents in groups.items():
        row = deepcopy(parents[0])
        entities, pairs, decisions, statuses, reasons, origins, seen = {}, [], [], {}, {}, {}, set()
        for parent in parents:
            for entity in parent["entities"]:
                value = {field: entity[field] for field in ("id", "name", "label")}
                if entity["id"] in entities and entities[entity["id"]] != value:
                    raise ValueError("Conflicting source entity identity")
                entities[entity["id"]] = value
            by_id = {decision["pair_id"]: decision for decision in parent["target"]["decisions"]}
            for pair in parent["pairs"]:
                endpoint = pair["source"], pair["target"]
                if endpoint in seen:
                    raise ValueError("Duplicated directed candidate across source blocks")
                seen.add(endpoint)
                old, pid = pair["pair_id"], f"P{len(pairs) + 1:04d}"
                pairs.append({**deepcopy(pair), "pair_id": pid})
                decisions.append({**deepcopy(by_id[old]), "pair_id": pid})
                statuses[pid] = parent["pair_supervision"][old]
                if old in parent.get("unknown_reasons", {}):
                    reasons[pid] = parent["unknown_reasons"][old]
                origins[pid] = _origin(parent, old)
        text = "\n".join(span["text"] for span in row["spans"])
        def entity_key(entity):
            hits = find_mentions(entity["name"], text)
            return (hits[0][0] if hits else len(text) + 1, entity["label"], entity["name"].casefold(), entity["id"])
        row["entities"] = [{**entity, "entity_id": f"E{index + 1:03d}"}
                           for index, entity in enumerate(sorted(entities.values(), key=entity_key))]
        row.update(id="merged:" + sha(key), pairs=pairs, target={"decisions": decisions},
                   pair_supervision=statuses, unknown_reasons=reasons,
                   rapid_merge_provenance={"pair_sources": origins})
        row.pop("abstention_provenance", None)
        row["completion"], row["loss_mask_spans"] = serialize_partial(row["target"], statuses)
        row["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(row)) + "\n"
        validate_partial_row(row)
        merged.append(row)
    if atomic_counter(rows) != atomic_counter(merged):
        raise ValueError("Source merging changed atomic decisions")
    return merged


def atomic_counter(rows, *, known_only=False):
    """Per-decision multiset including source identity, exact evidence and weight.

    Unlike set comparison, duplicated or dropped decisions cannot pass. Parent
    pair IDs disambiguate provenance even after chunk-local IDs are reassigned.
    UNKNOWN reasons and synthetic-negative proofs are included in the identity.
    """
    result = Counter()
    for row in rows:
        by_id = {pair["pair_id"]: pair for pair in row["pairs"]}
        for decision in row["target"]["decisions"]:
            pid = decision["pair_id"]
            status = row["pair_supervision"][pid]
            if known_only and status not in KNOWN:
                continue
            pair = by_id[pid]
            origin = _origin(row, pid)
            parent_id, original_pid = origin["parent_row_id"], origin["parent_pair_id"]
            proof = (origin["parent_abstention_provenance"] or {}).get("negative_proofs", {}).get(original_pid)
            identity = [parent_id, row["sample_id"], row["group_id"], row["split"],
                        original_pid, pair["source"], pair["target"], pair["allowed"],
                        status, decision["relations"], row.get("unknown_reasons", {}).get(pid),
                        row.get("sample_weight", 1.0), row["spans"], row.get("original_spans"), proof]
            result[sha(canonical(identity))] += 1
    return result


def counter_digest(counter):
    return sha(canonical(sorted(counter.items())))


def project_row(parent, pairs, *, variant, chunk_index, validate=True):
    row = deepcopy(parent)
    old_to_new = {pair["pair_id"]: f"P{index + 1:04d}" for index, pair in enumerate(pairs)}
    decisions = {decision["pair_id"]: decision for decision in parent["target"]["decisions"]}
    row["pairs"] = [{**deepcopy(pair), "pair_id": old_to_new[pair["pair_id"]]} for pair in pairs]
    used = {pair[key] for pair in pairs for key in ("source", "target")}
    # Preserve the input entity-table order and local IDs; filter only endpoints
    # unused by this chunk, as required by the shared prompt input contract.
    row["entities"] = [deepcopy(entity) for entity in parent["entities"] if entity["id"] in used]
    row["target"] = {"decisions": [
        {**deepcopy(decisions[pair["pair_id"]]), "pair_id": old_to_new[pair["pair_id"]]}
        for pair in pairs]}
    row["pair_supervision"] = {new: parent["pair_supervision"][old] for old, new in old_to_new.items()}
    row["unknown_reasons"] = {old_to_new[old]: reason for old, reason in parent.get("unknown_reasons", {}).items()
                              if old in old_to_new}
    row["completion"], row["loss_mask_spans"] = serialize_partial(row["target"], row["pair_supervision"])
    row.pop("abstention_provenance", None)
    row.pop("rapid_merge_provenance", None)
    row["rapid_variant_provenance"] = {
        "version": VERSION, "variant": variant, "chunk_index": chunk_index,
        "pair_sources": {new: deepcopy(_origin(parent, old)) for old, new in old_to_new.items()},
        "source_context_unchanged": True, "selection_uses_gold": False,
    }
    row["id"] = f"{parent['id']}:{variant}:{chunk_index}"
    row["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(row)) + "\n"
    if validate:
        validate_partial_row(row)
    return row


def make_variant(rows, *, variant, order, pairs_per_batch):
    if type(pairs_per_batch) is not int or pairs_per_batch < 1:
        raise ValueError("Positive integer chunk size required")
    result, skipped = [], []
    for parent in merge_sources(rows):
        ordered = input_pair_order(parent, order)
        for index, offset in enumerate(range(0, len(ordered), pairs_per_batch)):
            pairs = ordered[offset:offset + pairs_per_batch]
            supervised = any(parent["pair_supervision"][p["pair_id"]] in KNOWN for p in pairs)
            row = project_row(parent, pairs, variant=variant, chunk_index=index, validate=supervised)
            (result if supervised else skipped).append(row)
    if atomic_counter(rows, known_only=True) != atomic_counter(result, known_only=True):
        raise ValueError("Supervised atomic decision multiset drift")
    if atomic_counter(rows) != atomic_counter(result) + atomic_counter(skipped):
        raise ValueError("Candidate atomic decision multiset drift")
    represented = {origin["parent_row_id"] for row in result
                   for origin in row["rapid_variant_provenance"]["pair_sources"].values()}
    if {r["id"] for r in rows} != represented:
        raise ValueError("A parent source row lost all effective supervision")
    return result, skipped


def build(source, output, *, expected_sha256, tokenizer_dir=None, max_length=6144,
          base_rows=192, synthetic_rows=64):
    source, output = Path(source).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.exists():
        raise ValueError("Refusing to overwrite an existing rapid release")
    if output.is_relative_to(repo) or source.is_relative_to(output):
        raise ValueError("Private output must be outside repository and source")
    source_bytes = source.read_bytes()
    if sha(source_bytes) != expected_sha256:
        raise ValueError("Frozen source SHA256 mismatch")
    all_rows = read_rows(source)
    rows = select_rows(all_rows, base_rows=base_rows, synthetic_rows=synthetic_rows)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Only frozen training rows are eligible")
    if any(len(row["pairs"]) > 16 for row in rows):
        raise ValueError("Expected at most 16 original candidate pairs per row")
    # Control preserves the selected original rows; all transformed arms are
    # derived independently from the same metadata-frozen source selection.
    arms, skipped = {"V4_control": rows}, {"V4_control": []}
    for name, config in VARIANTS.items():
        if name == "V4_control":
            continue
        arms[name], skipped[name] = make_variant(rows, variant=name, **config)
    tokenizer = None
    if tokenizer_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError("A local fast tokenizer with EOS is required")
    audits = {}
    baseline = atomic_counter(rows, known_only=True)
    for name, variant_rows in arms.items():
        effective = atomic_counter(variant_rows, known_only=True)
        if effective != baseline:
            raise ValueError("Published arms differ in effective supervision")
        audits[name] = {
            "supervision": audit_rows(variant_rows, tokenizer, max_length),
            "admission": audit_partial_supervision(variant_rows),
            "parent_rows": len(rows), "all_unknown_chunks_skipped": len(skipped[name]),
            "unknown_decisions_skipped": sum(len(row["pairs"]) for row in skipped[name]),
            "effective_atomic_counter_sha256": counter_digest(effective),
            "retained_atomic_counter_sha256": counter_digest(atomic_counter(variant_rows)),
            "skipped_atomic_counter_sha256": counter_digest(atomic_counter(skipped[name])),
            "supervised_decisions_preserved_once_per_pass": effective == baseline,
            "all_source_rows_represented": True,
        }
    if source.read_bytes() != source_bytes:
        raise ValueError("Frozen source changed during construction")
    matrix = {"version": VERSION, "defaults": asdict(ProtocolConfig(repair=True, strict_evidence=True, matching="nfkc_v2")),
              "variants": VARIANTS}
    report = {
        "version": VERSION, "source_sha256": expected_sha256,
        "source_rows_before_selection": len(all_rows), "source_rows": len(rows),
        "selection": {"base_rows": base_rows, "synthetic_rows": synthetic_rows, "seed": 42,
                      "rule": "sha256_of_version_seed_group_sample_row_id_within_metadata_strata",
                      "selected_parent_ids_sha256": sha(canonical([row["id"] for row in rows]))},
        "source_groups": len({r["group_id"] for r in rows}),
        "audits": audits, "tokenizer_checked": tokenizer is not None, "max_length": max_length,
        "pair_selection_and_order_are_input_only": True,
        "source_context_unchanged": True, "evaluation_gold_modified": False,
        "effective_atomic_counter_sha256": counter_digest(baseline),
        "all_atomic_counter_sha256": counter_digest(atomic_counter(rows)),
        "eligible_for_existing_rl": False,
        "code_sha256": sha(Path(__file__).read_bytes()),
        "caveats": [
            "This is dense fixed-pair SFT, not sparse-output or shortened-input SFT.",
            "Selected source blocks are merged then globally input-sorted; training retains only the frozen candidate union, while inference considers its full candidate set.",
            "All-UNKNOWN chunks are skipped, without removing any positive or explicit negative decision.",
            "Short chunks repeat full contexts and alter syntax/EOS exposure, sequence count, and per-sequence loss weighting.",
            "Equal optimizer steps do not match effective atomic, token, epoch, or compute budgets across chunk sizes.",
            "Synthetic extraction negatives retain construction provenance but are not expert clinical judgments.",
        ],
    }
    if tokenizer:
        report["tokenizer_sha256"] = {name: sha((Path(tokenizer_dir) / name).read_bytes())
                                      for name in ("tokenizer.json", "tokenizer_config.json")}
    output.mkdir(parents=True, mode=0o700)
    hashes = {}
    for name, variant_rows in arms.items():
        path = output / f"{name}.jsonl"
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write("".join(dumps(row) + "\n" for row in variant_rows).encode())
        hashes[path.name] = sha(path.read_bytes())
    path = output / "experiments.json"
    with path.open("x") as handle:
        path.chmod(0o600)
        handle.write(json.dumps(matrix, indent=2) + "\n")
    hashes[path.name] = sha(path.read_bytes())
    report["output_sha256"] = hashes
    path = output / "manifest.json"
    with path.open("x") as handle:
        path.chmod(0o600)
        handle.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--base-rows", type=int, default=192)
    parser.add_argument("--synthetic-rows", type=int, default=64)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, expected_sha256=args.expected_sha256,
                           tokenizer_dir=args.tokenizer_dir, max_length=args.max_length,
                           base_rows=None if args.full_data else args.base_rows,
                           synthetic_rows=None if args.full_data else args.synthetic_rows)))


if __name__ == "__main__":
    main()

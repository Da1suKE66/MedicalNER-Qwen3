"""Rebuild train-only semantic DPO preferences from resident admitted SFT rows.

Chosen completions remain byte-identical to canonical SFT. A rejected sequence
contains one explicit defect; unannotated relations are never negative labels.
No evaluation text, predictions, gold, or external medical knowledge is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from generative.chat import render_prompt
from semantic import protocol as p
from semantic.data import canonical, encode_row, sha, validate_row

VERSION = "semantic-targeted-dpo-round2-v1"
SEED = "round2-train-only-semantic-preferences-20261019"
LOCKS = {
    "base_manifest": "c5e1b8bbb443ac51e49458ae7368e41b6fef8c2c6136edce31da5c6c7aaefb3d",
    "overlap_manifest": "6ec898ca4e04860bb592e1a0348dc2aa542aeb6026925d4ebe3be1f97c136b74",
    "protocol": "c7bdde2df59637f62964119a190984043ea7e9de4d5732b47fcbfc38e7f76606",
}
LIMITATIONS = [
    "Chosen is frozen non-exhaustive silver, not human-adjudicated clinical truth.",
    "Omission preferences remove a known positive, never label an unannotated edge negative.",
    "Outside-table and ontology-invalid direction negatives teach explicit input/protocol constraints.",
    "Exact-repetition preferences preserve the graph; they target wasted generation and truncation.",
    "Variable completion lengths can produce length preference; lengths are reported by mutation.",
    "Mechanical holdout is from original TRAIN and may already have been seen by SFT.",
    "Clinical improvement requires matched actual pipeline precision/recall/F1 evaluation.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    require(rows, "Empty rows")
    return rows


def locked(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, "Frozen SHA mismatch: " + str(path))
    return raw


def role(group):
    return "mechanical_holdout" if int(sha(SEED + ":group:" + str(group))[:16], 16) % 8 == 0 else "train"


def load_sources(base_root, overlap_root):
    base_root, overlap_root = Path(base_root), Path(overlap_root)
    locked(Path(p.__file__), LOCKS["protocol"])
    bm = json.loads(locked(base_root / "manifest.json", LOCKS["base_manifest"]))
    om = json.loads(locked(overlap_root / "manifest.json", LOCKS["overlap_manifest"]))
    specs = {
        "entities": (base_root, bm, "full_source/train_entities.jsonl"),
        "full_relations": (base_root, bm, "full_source/train_relations.jsonl"),
        "relations": (overlap_root, om, "short_canonical/train_relations.jsonl"),
        "split_metadata": (base_root, bm, "source_audit.jsonl"),
    }
    data, files = {}, {}
    for key, (root, manifest, rel) in specs.items():
        path = root / rel
        digest = manifest["output_sha256"][rel]
        data[key] = [json.loads(s) for s in locked(path, digest).decode().splitlines() if s.strip()]
        files[str(path.resolve())] = digest
    groups = defaultdict(set)
    for row in data.pop("split_metadata"):
        groups[row["split"]].add(str(row["source_group_id"]))
    require(not groups["train"] & groups["tune"], "Train/tune source leakage")
    for rows in data.values():
        for row in rows:
            require(row["split"] == "train" and str(row["group_id"]) in groups["train"]
                    and str(row["group_id"]) not in groups["tune"], "Non-training source or donor")
    full = {r["metadata"]["source_record_sha256"]: r for r in data["full_relations"]}
    require(len(full) == len(data["full_relations"]), "Duplicate full source")
    sources = {r["id"]: r for r in data["relations"]}
    require(len(sources) == len(data["relations"]), "Duplicate source row ID")
    for row in sources.values():
        validate_row(row)
        require(row["variant"] == "short_canonical", "Wrong parent-compatible view")
        original = full[row["metadata"]["source_record_sha256"]]
        atom_map = dict(zip(original["metadata"]["supervised_atom_ids"], original["target"]))
        require(len(row["metadata"]["supervised_atom_ids"]) == row["target_count"], "Atom count mismatch")
        require(all(atom_map.get(atom) == target for atom, target in zip(
            row["metadata"]["supervised_atom_ids"], row["target"])), "Chosen atoms differ from full source")
    return sources, data["entities"], full, files


def decode_lines(row):
    lines = row["completion"].splitlines(keepends=True)
    require(all(line.endswith("\n") for line in lines), "Missing completion newline")
    return lines, [line.removesuffix("\n").split("\t") for line in lines]


def mutate(row, mutation):
    lines, fields = decode_lines(row)
    index = mutation["line_index"]
    require(type(index) is int and 0 <= index < len(lines), "Invalid mutation line")
    kind = mutation["kind"]
    if kind == "outside_entity_table":
        require(mutation["endpoint"] in ("source", "target"), "Invalid endpoint")
        fields[index][1 if mutation["endpoint"] == "source" else 4] = p.escape_field(mutation["donor"]["name"])
        lines[index] = "\t".join(fields[index]) + "\n"
    elif kind == "ontology_invalid_direction":
        f = fields[index]
        f[0], f[1], f[3], f[4] = f[3], f[4], f[0], f[1]
        lines[index] = "\t".join(f) + "\n"
    elif kind == "omit_known_positive":
        lines.pop(index)
    elif kind == "repeat_exact_record":
        # Preserve all positives and add two copies, never a new relation claim.
        lines.extend([lines[index], lines[index]])
    else:
        raise ValueError("Unknown mechanical mutation")
    return "".join(lines)


def encode_pair(pair, tokenizer, max_length, *, require_train=True):
    require(not require_train or pair["split"] == "train", "Mechanical holdout cannot train")
    row = pair["source_row"]
    require(pair["prompt"] == row["prompt"] and pair["chosen"] == row["completion"], "Chosen drift")
    item = encode_row(row, tokenizer, max_length)
    prompt = list(tokenizer.encode(render_prompt(tokenizer, pair["prompt"]), add_special_tokens=False))
    chosen = list(tokenizer.encode(pair["chosen"], add_special_tokens=False))
    rejected = list(tokenizer.encode(pair["rejected"], add_special_tokens=False))
    eos = tokenizer.eos_token_id
    require(type(eos) is int and prompt and chosen and rejected and eos not in chosen and eos not in rejected,
            "Invalid or embedded EOS")
    chosen.append(eos); rejected.append(eos)
    require(chosen != rejected, "Identical preference")
    require(max(len(prompt) + len(chosen), len(prompt) + len(rejected)) <= max_length,
            "No truncation: pair exceeds max_length")
    require(item["input_ids"] == prompt + chosen and item["labels"] == [-100] * len(prompt) + chosen,
            "SFT token/mask mismatch")
    if pair["mutation"]["kind"] == "outside_entity_table":
        require(len(chosen) == len(rejected), "Outside-table pair must match token lengths")
    return {"id": pair["id"], "group_id": row["group_id"], "prompt_ids": prompt,
            "chosen_ids": chosen, "rejected_ids": rejected, "stage": "relations"}


def verify_pair(pair, sources, entity_rows, full, *, require_train=True):
    require(pair["data_version"] == VERSION and pair["unknown_as_negative"] is False, "Wrong preference semantics")
    row = sources[pair["source_row"]["id"]]
    require(pair["source_row"] == row and pair["source_row_sha256"] == sha(row), "Source row changed")
    require(pair["split"] == role(row["group_id"]), "Group partition changed")
    require(not require_train or pair["split"] == "train", "Holdout cannot train")
    require(pair["prompt"] == row["prompt"] and pair["chosen"] == row["completion"], "Frozen chosen changed")
    task = row["task_input"]
    chosen = p.parse_relations(task["text"], task["entities"], pair["chosen"], metadata=row["metadata"])
    require(chosen["valid"] and not chosen["rejected_relations"], "Chosen has invalid reference/evidence/type")
    mutation = pair["mutation"]
    kind, index = mutation["kind"], mutation["line_index"]
    lines, fields = decode_lines(row)
    require(type(index) is int and 0 <= index < len(lines), "Invalid line")
    if kind != "reviewed_semantic_contradiction":
        require(pair["rejected"] == mutate(row, mutation), "Rejected mutation drift")
    rejected = p.parse_relations(task["text"], task["entities"], pair["rejected"], metadata=row["metadata"])
    if kind == "outside_entity_table":
        donor = mutation["donor"]
        donor_row = next(r for r in entity_rows if r["id"] == donor["row_id"])
        entity = next(e for e in donor_row["target"] if e["id"] == donor["entity_id"])
        require(donor["name"] == entity["name"] and donor["label"] == entity["label"]
                and donor["source_row_sha256"] == sha(donor_row), "Unverified donor")
        require(role(donor_row["group_id"]) == pair["split"]
                and donor_row["group_id"] != row["group_id"], "Donor partition leakage")
        offset = 0 if mutation["endpoint"] == "source" else 3
        require(p.unescape_field(fields[index][offset]) == donor["label"], "Donor type mismatch")
        require(p.normalize_name(donor["name"]) not in {p.normalize_name(e["name"]) for e in task["entities"]},
                "Donor is in entity table")
        original = full[row["metadata"]["source_record_sha256"]]
        title = (original["metadata"].get("title_context") or {}).get("text", "")
        require(not p.find_mentions(donor["name"], original["task_input"]["text"])
                and (not title or not p.find_mentions(donor["name"], title)), "Donor is source-present")
        require(not rejected["valid"] and len(rejected["errors"]) == 1
                and rejected["errors"][0]["reason"] == "unknown_entity_reference", "Unproven outside-table defect")
    elif kind == "ontology_invalid_direction":
        require(not rejected["valid"] and len(rejected["errors"]) == 1
                and rejected["errors"][0]["reason"] == "unknown_or_wrong_source_relation", "Direction not ontology-invalid")
    elif kind == "omit_known_positive":
        require(len(lines) >= 2 and rejected["valid"] and not rejected["rejected_relations"], "Invalid omission")
        triple = tuple(fields[index][:5])
        require(sum(tuple(f[:5]) == triple for f in fields) == 1, "Omission must remove a whole known positive edge")
        require(len(chosen["relations"]) == len(rejected["relations"]) + 1, "Omission removed no known edge")
    elif kind == "repeat_exact_record":
        require(rejected["valid"] and not rejected["rejected_relations"], "Repetition changed semantic validity")
        require(len(rejected["relations"]) == len(chosen["relations"]), "Repetition changed graph size")
        require(len(pair["rejected"].splitlines()) == len(lines) + 2, "Bad repetition count")
    elif kind == "reviewed_semantic_contradiction":
        review = pair["review"]
        require(pair["split"] == "train" and all(review.get(k) is True for k in
                ("accepted", "source_supported", "negative_explicitly_contradicted")), "Missing semantic review")
        require(review.get("reviewer_model") and review.get("review_id") and mutation.get("explanation"), "Missing review provenance")
        require(rejected["valid"] and not rejected["rejected_relations"], "Semantic hard pair must remain structurally valid")
        other = pair["rejected"].splitlines(keepends=True)
        require(len(other) == len(lines) and [i for i, (a, b) in enumerate(zip(lines, other)) if a != b] == [index],
                "Semantic review changed multiple records")
        require(fields[index][5] == other[index].removesuffix("\n").split("\t")[5], "Semantic negative changed evidence")
        # This validates provenance/shape, NOT the clinical truth of API review.
    else:
        raise ValueError("Unknown preference kind")
    return row


def make_pair(row, mutation, *, rejected=None, review=None):
    rejected = mutate(row, mutation) if rejected is None else rejected
    result = {"data_version": VERSION, "id": sha({"row": row["id"], "mutation": mutation, "rejected": rejected}),
        "split": role(row["group_id"]), "source_row": row, "source_row_sha256": sha(row),
        "prompt": row["prompt"], "chosen": row["completion"], "rejected": rejected,
        "mutation": mutation, "unknown_as_negative": False}
    if review is not None:
        result["review"] = review
    return result


def build(base_root, overlap_root, tokenizer, output, *, max_length=4096, external=None,
          train_pairs=192, holdout_pairs=32):
    output = Path(output)
    require(not output.exists(), "Refuse existing release")
    require(train_pairs % 6 == 0 and holdout_pairs % 4 == 0, "Use explicit balanced quotas")
    sources, entities, full, files = load_sources(base_root, overlap_root)
    donors = defaultdict(list)
    for row in entities:
        for entity in row["target"]:
            donors[(role(row["group_id"]), entity["label"])].append({
                "row_id": row["id"], "entity_id": entity["id"], "name": entity["name"], "label": entity["label"],
                "group_id": row["group_id"], "source_row_sha256": sha(row)})
    for values in donors.values():
        values.sort(key=lambda v: sha(SEED + canonical(v)))
    rows = sorted(sources.values(), key=lambda r: sha(SEED + ":row:" + r["id"]))
    kinds = ("outside_entity_table", "omit_known_positive", "ontology_invalid_direction", "repeat_exact_record")
    pairs, encoded, excluded = [], {}, Counter()
    for partition in ("train", "mechanical_holdout"):
        quotas = (train_pairs // 3, train_pairs // 3, train_pairs // 6, train_pairs // 6) if partition == "train" else (holdout_pairs // 4,) * 4
        used_groups, used_rows = Counter(), set()
        for kind, quota in zip(kinds, quotas):
            accepted = 0
            for row in rows:
                if accepted == quota:
                    break
                if role(row["group_id"]) != partition or row["id"] in used_rows or used_groups[row["group_id"]] >= 3:
                    continue
                lines, fields = decode_lines(row)
                if len(lines) < 2:
                    continue
                mutations = []
                for index, f in enumerate(fields):
                    if kind == "outside_entity_table":
                        endpoint = "source" if accepted % 2 == 0 else "target"
                        typ = p.unescape_field(f[0 if endpoint == "source" else 3])
                        old = f[1 if endpoint == "source" else 4]
                        token_count = len(tokenizer.encode("\t" + old + "\t", add_special_tokens=False))
                        for donor in donors[(partition, typ)]:
                            if donor["group_id"] == row["group_id"]:
                                continue
                            if len(tokenizer.encode("\t" + p.escape_field(donor["name"]) + "\t", add_special_tokens=False)) != token_count:
                                continue
                            mutations.append({"kind": kind, "line_index": index, "endpoint": endpoint, "donor": donor})
                    else:
                        mutations.append({"kind": kind, "line_index": index})
                found = None
                for mutation in mutations:
                    pair = make_pair(row, mutation)
                    try:
                        verify_pair(pair, sources, entities, full, require_train=False)
                        item = encode_pair(pair, tokenizer, max_length, require_train=False)
                    except ValueError as exc:
                        excluded[str(exc)] += 1
                        continue
                    found = pair, item
                    break
                if found:
                    pair, item = found
                    pairs.append(pair); encoded[pair["id"]] = item
                    used_groups[row["group_id"]] += 1; used_rows.add(row["id"]); accepted += 1
            require(accepted == quota, f"Insufficient {partition}/{kind}: {accepted}/{quota}")
    external_count = 0
    if external:
        for item in read_rows(external):
            row = sources[item["source_row_id"]]
            require(item["source_row_sha256"] == sha(row), "External source hash mismatch")
            pair = make_pair(row, item["mutation"], rejected=item["rejected"], review=item["review"])
            verify_pair(pair, sources, entities, full)
            tokenized = encode_pair(pair, tokenizer, max_length)
            require(pair["id"] not in encoded, "Duplicate external pair")
            pairs.append(pair); encoded[pair["id"]] = tokenized; external_count += 1
        require(external_count % 4 == 0, "Reviewed external count must be multiple of accumulation 4; do not silently drop")
        files[str(Path(external).resolve())] = sha(Path(external).read_bytes())
    train = [r for r in pairs if r["split"] == "train"]
    held = [r for r in pairs if r["split"] == "mechanical_holdout"]
    train_groups = {str(r["source_row"]["group_id"]) for r in train}
    held_groups = {str(r["source_row"]["group_id"]) for r in held}
    require(not train_groups & held_groups, "Holdout overlap")
    output.mkdir(parents=True, mode=0o700)
    hashes = {}
    for name, values in (("candidate_train.jsonl", train), ("mechanical_holdout.jsonl", held)):
        path = output / name
        path.write_text("".join(canonical(row) + "\n" for row in values)); path.chmod(0o600)
        hashes[name] = sha(path.read_bytes())
    lengths = {}
    for kind in sorted({r["mutation"]["kind"] for r in pairs}):
        values = [encoded[r["id"]] for r in pairs if r["mutation"]["kind"] == kind]
        lengths[kind] = {"pairs": len(values), "prompt_tokens": sum(len(v["prompt_ids"]) for v in values),
            "chosen_tokens": sum(len(v["chosen_ids"]) for v in values),
            "rejected_tokens": sum(len(v["rejected_ids"]) for v in values),
            "completion_length_delta_counts": dict(Counter(len(v["rejected_ids"]) - len(v["chosen_ids"]) for v in values))}
    manifest = {"data_version": VERSION, "parent_view": "short_canonical", "protocol_sha256": LOCKS["protocol"],
        "train_pairs": len(train), "mechanical_holdout_pairs": len(held), "train_source_groups": len(train_groups),
        "mechanical_holdout_groups": len(held_groups), "group_overlap": 0, "external_reviewed_pairs": external_count,
        "output_sha256": hashes, "input_file_locks": files, "no_truncation": True, "max_length": max_length,
        "max_prompt_plus_completion_tokens": max(len(v["prompt_ids"]) + max(len(v["chosen_ids"]), len(v["rejected_ids"])) for v in encoded.values()),
        "mutation_counts": dict(Counter(r["mutation"]["kind"] for r in train)), "lengths": lengths,
        "excluded_attempts": dict(excluded), "limitations": LIMITATIONS,
        "builder_sha256": sha(Path(__file__).read_bytes())}
    path = output / "manifest.json"
    path.write_text(canonical(manifest) + "\n"); path.chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--overlap-release", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--external-reviewed")
    parser.add_argument("--train-pairs", type=int, default=192)
    parser.add_argument("--holdout-pairs", type=int, default=32)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    result = build(args.base_release, args.overlap_release, tokenizer, args.output_dir, max_length=args.max_length,
        external=args.external_reviewed, train_pairs=args.train_pairs, holdout_pairs=args.holdout_pairs)
    print(canonical({k: result[k] for k in ("train_pairs", "mechanical_holdout_pairs", "mutation_counts", "max_prompt_plus_completion_tokens")}))


if __name__ == "__main__":
    main()

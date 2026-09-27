"""Read-only release verification; optional real LOCAL Qwen tokenizer checks."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

from generative.chat import render_prompt
from generative.protocol import repair_text
from .release_v2 import canonical, sha, strict_json
from .supervision_v2 import masked_token_labels, validate_partial_row


def strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)
    elif isinstance(value, str):
        yield value


def verify(directory, tokenizer_dir=None, *, max_length=8192):
    from .text_normalization import normalize_with_offsets, repair_graph_text
    root = Path(directory)
    manifest = strict_json((root / "manifest.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    runtime_hashes = {name:sha((repo / name).read_bytes()) for name in manifest["code_sha256"]}
    if sha(Path(manifest["source_path"]).read_bytes()) != manifest["source_sha256"]:
        raise ValueError("Original source changed")
    if manifest.get("upstream_source"):
        upstream = manifest["upstream_source"]
        if sha(Path(upstream["path"]).read_bytes()) != upstream["sha256"]:
            raise ValueError("Original upstream source changed")
    for name, digest in manifest["output_sha256"].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f"Release file changed: {name}")
    tokenizer = None
    if tokenizer_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError("Require the local fast Qwen tokenizer and EOS")
    counts, by_file, cases = Counter(), {}, {}
    records = [strict_json(line) for line in (root / "records.jsonl").open()]
    groups = {split: set() for split in ("train", "tune")}
    for item in records:
        if sha(canonical(item["original_record"])) != item["original_record_sha256"]:
            raise ValueError("Original record conservation failed")
        groups[item["split"]].add(item["group_id"])
        graph = item["repaired_graph"]
        fixed, edits = repair_graph_text(graph)
        if fixed != graph or any(edit.get("applied", True) for edit in edits):
            raise ValueError("Graph repair is not idempotent")
        bad_words = sum(len(repair_text(text)[1]) for text in strings(graph))
        counts["repaired_graph_known_bad_words"] += bad_words
        if bad_words:
            raise ValueError("Known bad words remain in repaired graph")
        source = item["source"]
        if normalize_with_offsets(source["text"], repair=True)["text"] != source["text"]:
            raise ValueError("Source normalization is not idempotent")
        teacher_user = next(m["content"] for m in item["original_record"]["messages"] if m["role"] == "user")
        full_user = item["source_view"]["raw_user"]
        restoration = item.get("source_restoration")
        if restoration:
            if sha(teacher_user) != restoration["original_user_sha256"] or sha(full_user) != restoration["effective_user_sha256"]:
                raise ValueError("Source restoration provenance mismatch")
            counts["verified_upstream_restorations"] += 1
        elif full_user != teacher_user:
            raise ValueError("Unregistered teacher source modification")
        view = item["source_view"]
        if full_user[view["payload_start"]:view["payload_end"]] != view["raw_payload"]:
            raise ValueError("Payload does not map to original user")
        for node in item["entities"]:
            for hit in node["mentions"]:
                if source["text"][hit["start"]:hit["end"]] != hit["text"]:
                    raise ValueError("Normalized mention mismatch")
                if full_user[hit["user_start"]:hit["user_end"]] != hit["original_text"]:
                    raise ValueError("Original user mention mismatch")
                counts["verified_original_mentions"] += 1
        if item["source_index"] in (14, 39, 47, 499, 504, 511, 658, 904, 909, 355):
            cases[str(item["source_index"])] = {
                "record_blocked": item["record_blocked"], "blocked_pairs": item["blocked_pairs"],
                "issues": [issue["code"] for issue in item["issues"]],
                "ungrounded_entities": [e["id"] for e in item["entities"] if not e["mentions"]],
                "missing_evidence_indices": [a["index"] for a in item["annotations"] if not a["all_evidence_located"]],
            }
    if groups["train"] & groups["tune"]:
        raise ValueError("Frozen source groups overlap")
    for name in manifest["output_sha256"]:
        if name.count("/") != 1 or not name.endswith(("/train.jsonl", "/tune.jsonl")):
            continue
        c = Counter()
        seen = set()
        for line in (root / name).open():
            row = strict_json(line)
            validate_partial_row(row)
            if row["id"] in seen:
                raise ValueError("Duplicate training row id")
            seen.add(row["id"])
            if row["split"] != name.split("/")[1].split(".")[0]:
                raise ValueError("Train/tune tag mismatch")
            if records[row["sample_id"]]["record_blocked"]:
                raise ValueError("Quarantined record entered training")
            if records[row["sample_id"]]["split"] != row["split"]:
                raise ValueError("Frozen sample split changed")
            if repair_text(row["prompt"])[1] or repair_text(row["completion"])[1]:
                raise ValueError("Bad words entered model view")
            if any(d["relations"] for d in row["target"]["decisions"]
                   if row["pair_supervision"][d["pair_id"]].startswith("UNKNOWN_")):
                raise ValueError("UNKNOWN has a positive placeholder")
            c["rows"] += 1
            c["unknown_decisions_masked"] += sum(v.startswith("UNKNOWN_") for v in row["pair_supervision"].values())
            c["supported_silver_pairs"] += sum(v == "SUPPORTED_SILVER" for v in row["pair_supervision"].values())
            c["synthetic_negative_pairs"] += sum(v == "SYNTHETIC_NEGATIVE" for v in row["pair_supervision"].values())
            if tokenizer is not None:
                ids, labels = masked_token_labels(row, tokenizer)
                encoded = tokenizer(row["completion"], add_special_tokens=False, return_offsets_mapping=True)
                for label, (start,end) in zip(labels, encoded["offset_mapping"]):
                    if any(start < b and end > a for a,b in row["loss_mask_spans"]) and label != -100:
                        raise ValueError("UNKNOWN token receives loss")
                rendered = render_prompt(tokenizer, row["prompt"])
                length = len(tokenizer(rendered, add_special_tokens=True)["input_ids"]) + len(ids) + 1
                c["max_sequence_tokens"] = max(c["max_sequence_tokens"], length)
                c["overlength_rows"] += length > max_length
                c["completion_tokens"] += len(ids) + 1
                c["supervised_completion_tokens"] += sum(label != -100 for label in labels) + 1
                c["masked_completion_tokens"] += sum(label == -100 for label in labels)
        by_file[name] = dict(c)
        counts.update({k:v for k,v in c.items() if k != "max_sequence_tokens"})
        counts["max_sequence_tokens"] = max(counts["max_sequence_tokens"], c["max_sequence_tokens"])
    if counts["overlength_rows"]:
        raise ValueError(f"{counts['overlength_rows']} rows exceed the token budget; do not silently truncate")
    return {"passed": True, "manifest_sha256": sha((root / "manifest.json").read_bytes()),
            "runtime_code_sha256": runtime_hashes,
            "runtime_files_changed_since_build": [name for name, digest in runtime_hashes.items()
                                                  if manifest["code_sha256"][name] != digest],
            "original_records": len(records), "all_output_hashes_verified": True,
            "original_source_unchanged": True, "frozen_source_groups_disjoint": True,
            "counts": dict(counts), "by_file": by_file, "known_issue_cases": cases,
            "real_tokenizer_checked": tokenizer is not None,
            "tokenizer": ({"path": str(Path(tokenizer_dir).resolve()),
                           "tokenizer_json_sha256": sha((Path(tokenizer_dir) / "tokenizer.json").read_bytes()),
                           "tokenizer_config_sha256": sha((Path(tokenizer_dir) / "tokenizer_config.json").read_bytes()),
                           "transformers": importlib.metadata.version("transformers"),
                           "tokenizers": importlib.metadata.version("tokenizers"), "max_sequence_length": max_length}
                          if tokenizer is not None else None),
            "model_training_or_accuracy_evaluated": False,
            "human_semantic_or_real_negative_adjudication_completed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise ValueError("Choose a new validation report, refusing overwrite")
    report = verify(args.directory, args.tokenizer_dir, max_length=args.max_length)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as handle:
        report_path.chmod(0o600)
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(canonical({"passed": report["passed"], "real_tokenizer_checked": report["real_tokenizer_checked"],
                     "counts": report["counts"], "report": str(report_path)}))


if __name__ == "__main__":
    main()

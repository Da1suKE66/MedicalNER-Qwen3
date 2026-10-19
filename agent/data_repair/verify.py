"""Read-only integrity checks for a completed private candidate release."""

import argparse
from collections import Counter
import json
from pathlib import Path

from build_stage_data import _message
from data_repair.pipeline import dumps, sha, text_repairs
from evaluate import _graph
from generative.protocol import repair_text


def verify_release(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if sha(Path(manifest["source_path"]).read_bytes()) != manifest["source_sha256"]:
        raise ValueError("Original source changed")
    for filename, digest in manifest["output_sha256"].items():
        if sha((root / filename).read_bytes()) != digest:
            raise ValueError(f"Output hash mismatch: {filename}")
    counts = Counter()
    by_record = {}
    with (root / "records.jsonl").open() as handle:
        for line in handle:
            r = json.loads(line)
            counts["records"] += 1
            if r["record_id"] in by_record:
                raise ValueError("Duplicate record ID")
            by_record[r["record_id"]] = (r["group_id"], r["split"])
            if sha(dumps(r["original_record"])) != r["original_record_sha256"]:
                raise ValueError("Original record hash mismatch")
            source = r["source"]
            if sha(source["original_text"]) != source["original_sha256"] or sha(source["text"]) != source["repaired_sha256"]:
                raise ValueError("Source text hash mismatch")
            if repair_text(source["text"])[1]:
                raise ValueError("Text repair is not idempotent")
            if "repaired_graph" not in r:
                continue
            graph = r["repaired_graph"]
            original_graph = _graph(_message(r["original_record"], "assistant"))
            if len(graph["entities"]) != len(original_graph["entities"]) or len(graph["relations"]) != len(original_graph["relations"]):
                raise ValueError("Entities or relations silently removed")
            if text_repairs(graph)[1]:
                raise ValueError("Graph text repair is not idempotent")
            for before, after in zip(original_graph["entities"], graph["entities"]):
                if isinstance(before, dict) and before.get("id") != after.get("id"):
                    raise ValueError("Entity ID changed")
            for before, after in zip(original_graph["relations"], graph["relations"]):
                if isinstance(before, dict) and any(before.get(k) != after.get(k) for k in ("source", "target", "relation")):
                    raise ValueError("Relation semantics silently rewritten")
            counts["annotations_preserved"] += len(graph["relations"])
            for span in r["spans"]:
                if source["text"][span["start"]:span["end"]] != span["text"]:
                    raise ValueError("Span offset mismatch")
            for item in r["entities"]:
                for hit in item["mentions"]:
                    if source["text"][hit["start"]:hit["end"]] != hit["text"]:
                        raise ValueError("Mention offset mismatch")
                    restored = repair_text(source["original_text"][hit["original_start"]:hit["original_end"]])[0]
                    if restored != hit["text"]:
                        raise ValueError("Original mention offset mapping mismatch")
                    counts["mention_offsets_checked"] += 1
            for annotation in r["annotations"]:
                for hit in annotation["evidence"]:
                    if source["text"][hit["start"]:hit["end"]] != hit["text"]:
                        raise ValueError("Evidence offset mismatch")
                    restored = repair_text(source["original_text"][hit["original_start"]:hit["original_end"]])[0]
                    if restored != hit["text"]:
                        raise ValueError("Original evidence mapping mismatch")
                    counts["evidence_offsets_checked"] += 1
            for pair in r["pairs"]:
                values = [a["annotation"].get("relation") for a in r["annotations"] if a["index"] in pair["annotation_indices"]]
                labels = {v if isinstance(v, str) else None for v in values}
                if labels != {a["relation"] for a in pair["relations"]}:
                    raise ValueError("Multilabel set changed")
                counts["multilabel_pairs_checked"] += len(labels) > 1
    if counts["records"] != manifest["summary"]["records"]:
        raise ValueError("Record count mismatch")
    groups = {}
    for group, split in by_record.values():
        if group in groups and groups[group] != split:
            raise ValueError("Source group split leakage")
        groups[group] = split
    return {"status": "integrity_checks_passed", **dict(counts),
            "clinical_accuracy_verified": False, "release_ready": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    print(json.dumps(verify_release(args.directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

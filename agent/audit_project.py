"""Read-only audit of the historical data and pipeline; no model is modified."""

from __future__ import annotations
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

# Audit a frozen baseline checkout without copying or changing its source.
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--project-root", default=str(Path(__file__).parent))
_project = Path(_bootstrap.parse_known_args()[0].project_root).resolve()
sys.path.insert(0, str(_project))

from build_stage_data import (
    _align,
    _canonical_gold_entities,
    _evidence_span,
    _graph_from_teacher,
    _message,
    make_records,
)
from kg_agent.contracts import allowed_relations
from kg_agent.normalize import normalize_text
from kg_agent.source import SourceRouter, chunk_spans


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_raw(records):
    c = collections.Counter()
    labels = collections.Counter()
    relations = collections.Counter()
    skipped = collections.Counter()
    corruptions = collections.Counter()
    examples = []
    for i, record in enumerate(records):
        doc = SourceRouter().route(_message(record, "user"))
        c["records"] += 1
        c[doc.source_type] += 1
        suspect = [
            word.casefold()
            for word in re.findall(r"\b[A-Za-z]+null[A-Za-z]*\b", doc.raw)
            if word.casefold()
            not in {"annul", "annuls", "annulled", "annulling", "annulment"}
        ]
        corruptions.update(suspect)
        c["records_with_suspected_null_word_corruption"] += bool(suspect)
        if doc.source_type == "structured_icd":
            continue
        gold = _graph_from_teacher(_message(record, "assistant"))
        rows, mapping, missing = _canonical_gold_entities(gold["entities"], doc.spans)
        stage = {r["stage_id"]: r for r in rows}
        c["entities"] += len(gold["entities"])
        c["unaligned_entities"] += len(missing)
        labels.update(n["label"] for n in gold["entities"])
        c["single_character_spans"] += sum(len(s.text.strip()) == 1 for s in doc.spans)
        c["spans"] += len(doc.spans)
        endpoint_counts = collections.Counter(
            (r["source"], r["target"]) for r in gold["relations"]
        )
        labels_per_pair = collections.defaultdict(set)
        for rel in gold["relations"]:
            labels_per_pair[(rel["source"], rel["target"])].add(rel["relation"])
        c["repeated_endpoint_pairs"] += sum(n > 1 for n in endpoint_counts.values())
        c["multi_relation_endpoint_pairs"] += sum(
            len(labels) > 1 for labels in labels_per_pair.values()
        )
        c["collapsed_extra_relation_labels"] += sum(
            max(0, n - 1) for n in endpoint_counts.values()
        )
        for rel in gold["relations"]:
            c["relations"] += 1
            relations[rel["relation"]] += 1
            s = stage.get(mapping.get(rel["source"]))
            t = stage.get(mapping.get(rel["target"]))
            if not s or not t:
                c["relations_lost_unaligned_endpoint"] += 1
                continue
            if rel["relation"] not in allowed_relations(s["label"], t["label"]):
                c["relations_lost_signature"] += 1
                skipped[f"{s['label']} / {rel['relation']} / {t['label']}"] += 1
                continue
            ev = rel.get("evidence") or rel.get("evidence_sentence") or ""
            if isinstance(ev, list):
                ev = ev[0] if ev else ""
            aligned = _evidence_span(ev, doc.spans, s["name"], t["name"])
            exact = (
                any(
                    normalize_text(str(ev)) in normalize_text(span.text)
                    for span in doc.spans
                )
                if ev
                else False
            )
            both = any(
                normalize_text(s["name"]) in normalize_text(span.text)
                and normalize_text(t["name"]) in normalize_text(span.text)
                for span in doc.spans
            )
            if not exact and not both:
                c["evidence_fallback_first_span"] += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "sample_id": i,
                            "relation": rel["relation"],
                            "source": s["name"],
                            "target": t["name"],
                            "fallback": aligned,
                            "fallback_text": doc.spans[0].text,
                        }
                    )
            chunks = chunk_spans(doc.spans)
            evchunk = next((ch for ch in chunks if aligned in {x.id for x in ch}), [])
            active = {x.id for x in evchunk}
            if (
                s["span_id"].split(":")[0] not in active
                or t["span_id"].split(":")[0] not in active
            ):
                c["positive_requires_oracle_endpoint_injection"] += 1
    return {
        "counts": dict(c),
        "entity_labels": dict(labels),
        "relation_labels": dict(relations),
        "skipped_signatures": dict(skipped),
        "fallback_examples": examples,
        "suspected_null_word_corruption": dict(corruptions),
    }


def audit_stage(path, tokenizer=None):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    c = collections.Counter(
        records=len(rows), source_records=len({r["sample_id"] for r in rows})
    )
    labels = collections.Counter()
    by_position = collections.defaultdict(collections.Counter)
    lengths = []
    for row in rows:
        target = json.loads(row["completion"])
        ds = target.get("relations", [])
        if ds:
            flags = [d["relation"] != "NONE" for d in ds]
            c["decisions"] += len(ds)
            c["pure_none_batches"] += not any(flags)
            if any(flags) and not all(flags):
                c["mixed_batches"] += 1
                c["positive_prefix_mixed_batches"] += flags == sorted(
                    flags, reverse=True
                )
            labels.update(d["relation"] for d in ds)
            for j, d in enumerate(ds):
                by_position[j][d["relation"] == "NONE"] += 1
        if tokenizer:
            p = tokenizer(row["prompt"])["input_ids"]
            d = tokenizer(row["completion"], add_special_tokens=False)["input_ids"] + [
                tokenizer.eos_token_id
            ]
            lengths.append(len(p) + len(d))
            c["over4096"] += len(p) + len(d) > 4096
            remaining = tokenizer.decode(p[-max(1, 4096 - len(d)) :])
            c["entity_table_header_lost"] += (
                "ENTITIES\n" in row["prompt"] and "ENTITIES\n" not in remaining
            )
    result = {
        "sha256": sha(path),
        "counts": dict(c),
        "labels": dict(labels),
        "positive_by_position": {
            j: {"positive": x[False], "none": x[True]} for j, x in by_position.items()
        },
    }
    if lengths:
        result["tokens"] = {
            "min": min(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
            "max": max(lengths),
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        default=str(_project),
        help="Agent source tree to audit; use baseline checkout for historical numbers",
    )
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--stage", action="append", default=[])
    p.add_argument("--tokenizer")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    train = json.loads(Path(args.train).read_text())
    dev = json.loads(Path(args.dev).read_text())
    fingerprints = lambda rows: {
        hashlib.sha256(
            normalize_text(SourceRouter().route(_message(r, "user")).raw).encode()
        ).hexdigest()
        for r in rows
    }
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = {
        "audited_code_sha256": {
            str(f.relative_to(_project)): sha(f) for f in sorted(_project.rglob("*.py"))
        },
        "files": {"train_sha256": sha(args.train), "dev_sha256": sha(args.dev)},
        "train": audit_raw(train),
        "dev": audit_raw(dev),
        "exact_source_overlap": len(fingerprints(train) & fingerprints(dev)),
        "stage": {Path(f).name: audit_stage(f, tokenizer) for f in args.stage},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

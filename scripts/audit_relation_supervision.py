#!/usr/bin/env python3
"""Read-only CPU audit of the private migration's supervision contracts.

The JSON report contains aggregate counts, hashes and migration-relative paths
only. It never exports source text, prompts, completions, entity names, record
IDs, or individual review items. No network, model or tokenizer is required.

Usage:
  python scripts/audit_relation_supervision.py --migration-root /private/bundle \
      --output /private/new-audit.json [--compare-evidence-policy]

The optional comparison replays the current repository's CPU data builder. It
does not write a release, edit historical data or change evaluation labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import re
import sys


POSITIVE = "SUPPORTED_SILVER"
NEGATIVE = "SYNTHETIC_NEGATIVE"
REPORT_VERSION = "relation-supervision-audit-v1"
SOURCE_FILES = {
    "records": "data/repaired_v2/records.jsonl",
    "review": "data/repaired_v2/review_queue.jsonl",
    "split": "data/repaired_v2/frozen_split.jsonl",
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object in each data row")
                yield value


def code(value):
    """Allow categorical codes in aggregates, never arbitrary source strings."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_:-]{0,120}", value):
        raise ValueError("Noncategorical value in an audit metadata field")
    return value


def supervision_counts(records):
    counts, distribution = Counter(), Counter()
    sample_ids, groups = set(), set()
    for row in records:
        statuses = row["pair_supervision"]
        pairs = row["pairs"]
        decisions = row["target"]["decisions"]
        if (not statuses or not pairs or len(statuses) != len(pairs) or set(statuses) != {p["pair_id"] for p in pairs}
                or [p["pair_id"] for p in pairs] != [d["pair_id"] for d in decisions]):
            raise ValueError("Pair/status/target alignment failed")
        local = Counter(code(value) for value in statuses.values())
        if any(value not in (POSITIVE, NEGATIVE) and not value.startswith("UNKNOWN_") for value in local):
            raise ValueError("Unexpected supervision status")
        for decision in decisions:
            status = statuses[decision["pair_id"]]
            if not isinstance(decision["relations"], list) or (status == POSITIVE) != bool(decision["relations"]):
                raise ValueError("Positive/empty target contradicts its supervision status")
            counts["positive_relation_edges"] += len(decision["relations"])
            counts["multi_relation_pairs"] += len(decision["relations"]) > 1
        counts.update(local)
        counts["rows"] += 1
        counts["candidate_pairs"] += len(pairs)
        p, n = local[POSITIVE], local[NEGATIVE]
        u = sum(amount for value, amount in local.items() if value.startswith("UNKNOWN_"))
        counts["unknown_pairs"] += u
        if p and n:
            distribution["mixed_positive_negative_supervision"] += 1
        elif p:
            distribution["positive_only_supervision"] += 1
        elif n:
            distribution["negative_only_supervision"] += 1
        else:
            distribution["no_supervised_decisions"] += 1
        distribution["all_decisions_positive"] += p == len(pairs)
        distribution["all_decisions_negative"] += n == len(pairs)
        distribution["rows_with_unknown"] += u > 0
        sample_ids.add(row["sample_id"])
        groups.add(row["group_id"])
    # Zero-valued fields remain explicit, especially the abstention gap.
    for key in ("rows", "candidate_pairs", POSITIVE, NEGATIVE, "unknown_pairs", "positive_relation_edges", "multi_relation_pairs"):
        counts.setdefault(key, 0)
    for key in ("mixed_positive_negative_supervision", "positive_only_supervision", "negative_only_supervision",
                "no_supervised_decisions", "all_decisions_positive", "all_decisions_negative", "rows_with_unknown"):
        distribution.setdefault(key, 0)
    return {"counts": dict(sorted(counts.items())), "row_distribution": dict(sorted(distribution.items())),
            "distinct_sample_ids": len(sample_ids), "source_groups": len(groups)}


def positive_keys(records):
    result = set()
    for row in records:
        pairs = {p["pair_id"]: p for p in row["pairs"]}
        for decision in row["target"]["decisions"]:
            pair = pairs[decision["pair_id"]]
            result.update((row["sample_id"], pair["source"], pair["target"], edge["relation"])
                          for edge in decision["relations"])
    return result


def source_audit(root):
    counts, ambiguous = Counter(), set()
    for record in rows(root / SOURCE_FILES["records"]):
        counts["records"] += 1
        counts[code(record["source_type"])] += 1
        if record["source_type"] != "free_text":
            continue
        missing = {str(e["id"]) for e in record["entities"] if not e["mentions"]}
        counts["free_entities"] += len(record["entities"])
        counts["ungrounded_entities"] += len(missing)
        counts["free_relation_annotations"] += len(record["annotations"])
        multilabel = defaultdict(set)
        for annotation in record["annotations"]:
            edge = annotation["annotation"]
            source, target = str(edge["source"]), str(edge["target"])
            counts["relations_incident_ungrounded"] += source in missing or target in missing
            counts["missing_evidence_annotations"] += not annotation["all_evidence_located"]
            multilabel[(source, target)].add(edge["relation"])
            if any(len(fragment) > 1 for fragment in annotation["evidence_fragments"]):
                counts["ambiguous_evidence_annotations"] += 1
                ambiguous.add((record["source_index"], source, target, edge["relation"]))
        counts["multi_relation_pairs"] += sum(len(labels) > 1 for labels in multilabel.values())
    retained = {}
    for split in ("train", "tune"):
        retained[split] = len(ambiguous & positive_keys(rows(root / f"data/repaired_v2/base/{split}.jsonl")))
    return {"counts": dict(sorted(counts.items())), "ambiguous_positive_labels_in_base": retained,
            "ambiguous_positive_labels_in_base_total": sum(retained.values()),
            "caveat": "Ambiguous citations and absent surface matches are review findings, not adjudicated medical errors."}


def review_audit(root):
    count, kind, reason, split, source_ids = 0, Counter(), Counter(), Counter(), set()
    for item in rows(root / SOURCE_FILES["review"]):
        count += 1
        kind[code(item["kind"])] += 1
        reason[code(item.get("reason", item.get("details", {}).get("code", "unspecified")))] += 1
        split[code(item["split"])] += 1
        source_ids.add(item["source_index"])
    return {"items": count, "source_records": len(source_ids), "by_kind": dict(sorted(kind.items())),
            "by_reason": dict(sorted(reason.items())), "by_split": dict(sorted(split.items()))}


def legacy_stage_audit(root):
    split = {r["source_index"]: r["split"] for r in rows(root / SOURCE_FILES["split"])}
    result = {}
    for stage in ("entities", "relations"):
        path = f"data/teacher_original/train_{stage}_v4.jsonl"
        counts, sources = Counter(), defaultdict(set)
        none_targets = none_rows = 0
        for row in rows(root / path):
            role = split[row["sample_id"]]
            counts[code(role)] += 1
            sources[role].add(row["sample_id"])
            target = json.loads(row["completion"])
            n = sum(edge.get("relation") == "NONE" for edge in target.get("relations", []))
            none_targets += n
            none_rows += n > 0
        result[path] = {"rows": sum(counts.values()), "rows_by_internal_split": dict(sorted(counts.items())),
                        "source_records_by_internal_split": {k: len(v) for k, v in sorted(sources.items())},
                        "literal_NONE_targets": none_targets, "rows_with_literal_NONE": none_rows}
    return result


def compare_evidence_policy(root):
    # Import only the local repository's CPU builder; no remote or model load.
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "agent"))
    from data_repair.release_v2 import build_rows, variants
    from data_repair.supervision_v2 import LEGACY_EVIDENCE_POLICY, UNIQUE_EVIDENCE_POLICY

    result = {"policy": UNIQUE_EVIDENCE_POLICY, "splits": {}, "labels_rewritten_for_evaluation": False,
              "new_release_written": False, "builder_code_sha256": {
                  name: sha(repo / name) for name in ("agent/data_repair/release_v2.py",
                      "agent/data_repair/supervision_v2.py", "agent/data_repair/text_normalization.py",
                      "agent/generative/protocol.py", "agent/kg_agent/contracts.py",
                      "agent/kg_agent/ontology.json", "agent/kg_agent/source.py")}}
    prepared = list(rows(root / SOURCE_FILES["records"]))
    for split in ("train", "tune"):
        stored = list(rows(root / f"data/repaired_v2/base/{split}.jsonl"))
        replay, unique = [], []
        for item in prepared:
            if item["split"] != split:
                continue
            legacy_item = deepcopy(item)
            legacy_item["evidence_grounding_policy"] = LEGACY_EVIDENCE_POLICY
            replay.extend(build_rows(legacy_item, variants()["base"], sampled=split == "train")[0])
            unique_item = deepcopy(item)
            unique_item["evidence_grounding_policy"] = UNIQUE_EVIDENCE_POLICY
            unique.extend(build_rows(unique_item, variants()["base"], sampled=split == "train")[0])
        before, after = positive_keys(stored), positive_keys(unique)
        result["splits"][split] = {
            "legacy_replay_equals_stored_rows": replay == stored,
            "stored": supervision_counts(stored), "unique": supervision_counts(unique),
            "stored_unique_positive_labels": len(before), "unique_positive_labels": len(after),
            "labels_quarantined": len(before - after), "labels_added": len(after - before),
            "retained_unknown_reasons": dict(sorted(Counter(
                code(reason) for row in unique for reason in row.get("unknown_reasons", {}).values()).items())),
        }
    result["legacy_replay_all_equal"] = all(r["legacy_replay_equals_stored_rows"] for r in result["splits"].values())
    return result


def audit(root, *, compare=False):
    roots = ("data/repaired_v2", "data/v2_controlled")
    files = sorted({p for name in roots for split in ("train", "tune")
                    for p in (root / name).glob(f"*/{split}.jsonl")}
                   | set((root / "data/v3").glob("V3_*.jsonl")))
    if not files:
        raise ValueError("Migration has no relation supervision datasets")
    input_paths = set(files) | {root / name for name in SOURCE_FILES.values()}
    input_paths.update(root / f"data/teacher_original/train_{stage}_v4.jsonl" for stage in ("entities", "relations"))
    hashes = {str(path.relative_to(root)): sha(path) for path in sorted(input_paths)}
    report = {"version": REPORT_VERSION, "python_version": platform.python_version(),
              "audit_script_sha256": sha(Path(__file__)), "input_sha256": hashes,
              "count_scope": "One unweighted pass per file; derived variants are not independent medical samples.",
              "row_distribution_definition": "positive_only/negative_only refer to supervised decisions and may contain masked UNKNOWN; all_decisions_* exclude UNKNOWN.",
              "datasets": {str(path.relative_to(root)): supervision_counts(rows(path)) for path in files},
              "reviews": review_audit(root), "source_audit": source_audit(root),
              "legacy_stage_internal_tune_exposure": legacy_stage_audit(root),
              "evidence_policy_comparison": compare_evidence_policy(root) if compare else {"performed": False},
              "raw_text_exported": False, "historical_data_modified": False}
    if any(sha(root / name) != digest for name, digest in hashes.items()):
        raise ValueError("An input changed during the audit; report refused")
    report["input_hashes_unchanged_after_audit"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-evidence-policy", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit output")
    root = args.migration_root.resolve()
    if args.output.resolve().is_relative_to(root):
        parser.error("Audit output must be outside the read-only migration bundle")
    try:
        report = audit(root, compare=args.compare_evidence_policy)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        # Do not print exception values, which can contain private payloads/paths.
        parser.exit(2, f"Audit failed ({type(exc).__name__}); no report was written.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"audit_complete": True, "datasets": len(report["datasets"]),
                      "review_items": report["reviews"]["items"],
                      "evidence_policy_compared": args.compare_evidence_policy}, sort_keys=True))


if __name__ == "__main__":
    main()

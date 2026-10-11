#!/usr/bin/env python3
"""Independently audit archived extraction metrics without running inference.

Uses only the Python standard library, never the production graph scorer or
relation parser. Output contains aggregate counts, relative artifact paths and
hashes; medical source text, prompts and generated text are never exported.

Example:
    python scripts/audit_migration_metrics.py \\
        --migration-root /path/to/migration --output reports/migration_metrics.json

The output must be a new file outside the migration directory. Historical
metrics and predictions are read-only. Missing full-run predictions cannot be
replaced by a claim of independent raw-output verification.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


GOLD = Path("data/evaluation/dev_gold_full_148.json")
CACHE = Path("data/evaluation/entity_cache_148.jsonl")
FRESH = Path("history/20260928/retest/fresh_results.json")
PRIOR = Path("history/20260928/retest/independent_audit.json")
ORIGINAL_CACHE = Path(
    "history/initial_agent/remote_reports/"
    "dev_predictions_v4_e1r1_full_bigchunk_jsonstop_512.jsonl"
)
COUNTS = ("tp", "fp", "fn")
METRICS = (*COUNTS, "precision", "recall", "f1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def graph(value):
    if isinstance(value, dict) and "output" in value:
        value = value["output"]
    if isinstance(value, str):
        if "<output>" in value:
            value = value.split("<output>", 1)[1].split("</output>", 1)[0]
        value = json.loads(value)
    require(
        isinstance(value, dict)
        and isinstance(value.get("entities"), list)
        and isinstance(value.get("relations"), list),
        "Archived graph has invalid containers",
    )
    return value


def triples(value):
    """Unique directed typed triples, independent of IDs and array order.

    Archived malformed entities fail explicitly rather than being silently
    ignored. Invalid relation emissions remain false-positive sentinels, matching
    the frozen historical scoring contract. This is a concept metric, not a
    span-level, fuzzy-synonym or clinical-entailment metric.
    """
    value = graph(value)
    nodes = {}
    for node in value["entities"]:
        require(isinstance(node, dict), "Archived graph contains a malformed entity")
        properties = node.get("properties")
        name = node.get("name") or (
            properties.get("Name") if isinstance(properties, dict) else None
        )
        label, node_id = node.get("label"), node.get("id")
        require(
            type(node_id) in (int, str) and str(node_id).strip()
            and isinstance(name, str) and name.strip()
            and isinstance(label, str) and label.strip(),
            "Archived graph contains a malformed entity",
        )
        node_id = str(node_id)
        require(node_id not in nodes, "Archived graph contains duplicate entity IDs")
        nodes[node_id] = (label, normalize(name))
    result = set()
    for index, edge in enumerate(value["relations"]):
        if not isinstance(edge, dict):
            result.add((("__INVALID_ENDPOINT__", str(index)), "__INVALID_RELATION__",
                        ("__INVALID_ENDPOINT__", repr(edge))))
            continue
        endpoints = [
            nodes.get(str(edge.get(key)), ("__INVALID_ENDPOINT__", str(edge.get(key))))
            for key in ("source", "target")
        ]
        label = str(edge.get("relation") or edge.get("type") or "").strip()
        result.add((endpoints[0], label or "__INVALID_RELATION__", endpoints[1]))
    return result


def metric(tp, fp, fn):
    require(all(type(v) is int and v >= 0 for v in (tp, fp, fn)),
            "Metric counts must be nonnegative integers")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def compare(calculated, saved, context):
    require(all(
        isinstance(saved.get(k), (int, float)) and math.isfinite(saved[k])
        and abs(value - saved[k]) < 1e-12
        for k, value in calculated.items()
    ), "Archived metric mismatch: " + context)


def score(predictions, records):
    require(len(predictions) == len(records), "Prediction/gold length mismatch")
    totals = Counter(tp=0, fp=0, fn=0)
    for predicted, record in zip(predictions, records):
        assistants = [m["content"] for m in record["messages"] if m["role"] == "assistant"]
        require(len(assistants) == 1, "Expected exactly one gold assistant message")
        p, g = triples(predicted), triples(assistants[0])
        require(not any(t[0][0] == "__INVALID_ENDPOINT__" or t[2][0] == "__INVALID_ENDPOINT__"
                        for t in g), "Malformed gold relation endpoint")
        totals.update(tp=len(p & g), fp=len(p - g), fn=len(g - p))
    return metric(**totals)


def verify_report(report, relative_path):
    samples = report["metrics"]["samples"]
    require(isinstance(samples, list), "Missing per-sample archived counts")
    indices = [r["sample_index"] for r in samples]
    require(all(type(i) is int for i in indices) and set(indices) == set(range(len(samples)))
            and len(set(indices)) == len(indices), "Incomplete archived sample indices")
    groups_checked = 0
    for scope in ("overall", "free_text", "structured_icd"):
        rows = [r for r in samples if scope == "overall" or r["source_type"] == scope]
        saved = report["metrics"][scope]
        require(len(rows) == saved["samples"], "Archived sample count mismatch")
        for name in ("entity", "relation", "evidence_exact", "evidence_compatible"):
            counts = {k: sum(r[name][k] for r in rows) for k in COUNTS}
            calculated = metric(**counts)
            compare(calculated, saved[name], relative_path + "/" + scope + "/" + name)
            tp, fp, fn = (counts[k] for k in COUNTS)
            f05 = 1.25 * tp / (1.25 * tp + fp + .25 * fn) if tp + fp + fn else 0.0
            compare({"f0_5": f05}, saved[name], "F0.5")
            groups_checked += 1
        per = {}
        for row in rows:
            for label, counts in row["per_relation"].items():
                per.setdefault(label, Counter()).update(counts)
        require(set(per) == set(saved["per_relation"]), "Archived relation type mismatch")
        active = []
        for label, counts in per.items():
            calculated = metric(*(counts[k] for k in COUNTS))
            compare(calculated, saved["per_relation"][label], "per-relation counts")
            if counts["tp"] + counts["fn"]:
                active.append(calculated["f1"])
        compare({"macro_relation_f1": sum(active) / len(active) if active else 0.0},
                saved, "gold-supported macro relation F1")
    return groups_checked


def audit(root):
    gold = read_json(root / GOLD)
    gold_hash = sha256((root / GOLD).read_bytes())
    prior, fresh = read_json(root / PRIOR), read_json(root / FRESH)
    require(prior["gold_sha256"] == gold_hash == fresh["manifest"]["plan"]["gold_sha256"],
            "Archived gold digest does not match migrated gold")
    result = {
        "schema_version": 1,
        "scope": "Read-only archived metrics audit; no new inference or training",
        "gold_sha256": gold_hash,
        "gold_records": len(gold),
        "archived_metric_arithmetic_checks": [],
    }
    group_count = 0
    for path in sorted((root / "history/20260928").glob("eval*/*/*/metrics.json")):
        report = read_json(path)
        if not isinstance(report.get("metrics", {}).get("samples"), list):
            continue  # Pair-level stress metrics have a different scoring universe.
        relative = path.relative_to(root).as_posix()
        group_count += verify_report(report, relative)
        result["archived_metric_arithmetic_checks"].append({
            "path": relative, "sha256": sha256(path.read_bytes()),
            "samples": len(report["metrics"]["samples"]),
            "free_text_relation": {k: report["metrics"]["free_text"]["relation"][k] for k in METRICS},
            "raw_predictions_available": (path.parent / "predictions.jsonl").is_file(),
            "raw_generations_available": (path.parent / "generations.jsonl").is_file(),
            "verification_level": "per_sample_count_aggregation_and_metric_arithmetic",
        })
    require(group_count > 0, "No archived per-sample metric reports found")
    result["arithmetic_metric_groups_checked"] = group_count
    result["archived_five_document_inference_recomputed"] = {}
    for name in ("V2_base", "historical_N0_same_V2_prompts"):
        condition = fresh["conditions"][name]
        ids = [r["sample_id"] for r in condition["documents"]]
        require(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < len(gold) for i in ids),
                "Invalid archived inference sample IDs")
        require(ids == fresh["manifest"]["plan"]["sample_ids"], "Archived inference sample order mismatch")
        calculated = score([r["output"] for r in condition["documents"]], [gold[i] for i in ids])
        compare(calculated, condition["metrics"], "archived five-document inference")
        raw = Counter(batches=0, truncated=0, decisions=0, empty_decisions=0, raw_edges=0)
        generations = condition["generations"]
        require([r["id"] for r in generations] == fresh["manifest"]["plan"]["batch_ids"],
                "Archived generation batch order mismatch")
        for row in generations:
            require(row["sample_id"] in ids, "Unexpected generation sample ID")
            decisions = json.loads(row["text"])["decisions"]
            raw.update(batches=1, truncated=int(row["truncated"]), decisions=len(decisions),
                       empty_decisions=sum(not d["relations"] for d in decisions),
                       raw_edges=sum(len(d["relations"]) for d in decisions))
        result["archived_five_document_inference_recomputed"][name] = {
            "samples": len(ids), "relation": calculated, "raw_generation_counts": dict(raw),
            "verification_level": "independent_set_scoring_of_archived_graph_outputs",
        }
    cache_bytes = (root / CACHE).read_bytes()
    require(sha256(cache_bytes) == prior["cache_sha256"], "Archived entity cache digest mismatch")
    cache = [json.loads(line) for line in cache_bytes.splitlines() if line.strip()]
    free = prior["free_text_ids"]
    require(len(free) == len(set(free)) and all(type(i) is int and 0 <= i < len(gold) for i in free),
            "Invalid archived free-text strata")
    result["old_entity_cache_graph_recomputed"] = {
        "overall": score(cache, gold),
        "free_text": score([cache[i] for i in free], [gold[i] for i in free]),
        "free_text_rows": len(free),
        "rows_with_identity_metadata": sum("input_identity" in r for r in cache),
        "alignment_status": "not_independently_verified",
        "strata_source": PRIOR.as_posix(),
        "note": "Cache includes old relation outputs; these are not N0 relation-model scores.",
    }
    result["old_cache_matches_original_complete_predictions"] = cache_bytes == (root / ORIGINAL_CACHE).read_bytes()
    result["historical_N0_saved_evidence_only"] = {}
    for scope in ("predicted", "oracle"):
        saved = prior["historical_N0"][scope]
        compare(metric(*(saved[k] for k in COUNTS)), saved, "historical N0 saved summary")
        result["historical_N0_saved_evidence_only"][scope] = {k: saved[k] for k in METRICS}
    result["limitations"] = [
        "Full-run V2/N0 predictions and generations are not re-scored by this script; arithmetic checks are not raw-output verification.",
        "The five-document artifact is archived inference, not a newly executed inference experiment.",
        "Historical entity-cache alignment remains unverified; newly computing hashes cannot authenticate its original input order.",
        "The repeatedly used regression corpus is not an untouched final holdout.",
        "Silver-gold false positives are annotation disagreements, not automatically medically unsupported hallucinations.",
    ]
    return result


def check_output(root, output):
    require(not output.is_relative_to(root), "Output must be outside the read-only migration directory")
    require(not output.exists(), "Output already exists; choose a new aggregate report filename")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--migration-root", required=True, type=Path, help="Directory containing data/ and history/")
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON outside the migration directory")
    args = parser.parse_args(argv)
    root, output = args.migration_root.resolve(), args.output.resolve()
    try:
        check_output(root, output)
        result = audit(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        # Do not expose source payloads or machine-specific paths in diagnostics.
        if isinstance(exc, OSError):
            parser.error("Cannot read required artifacts or create the new output file")
        if isinstance(exc, (KeyError, TypeError, json.JSONDecodeError)):
            parser.error("Unexpected archived artifact structure")
        parser.error(str(exc))
    print(json.dumps({"status": "passed", "archived_reports_checked": len(result["archived_metric_arithmetic_checks"]),
                      "arithmetic_metric_groups_checked": result["arithmetic_metric_groups_checked"],
                      "archived_inference_conditions_recomputed": len(result["archived_five_document_inference_recomputed"]),
                      "new_inference_performed": False}))


if __name__ == "__main__":
    main()

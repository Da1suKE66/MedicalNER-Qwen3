"""Build a NEW private, partially supervised silver-data repair candidate.

No input overwrite, medical auto-adjudication, network call, training or metric
rewrite. The frozen split and the complete original annotation remain intact.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from build_stage_data import _message
from evaluate import _graph
from generative.protocol import ProtocolConfig, build_batches, dumps
from kg_agent.source import SentenceSpanSplitter
from .pipeline import split_assignment
from .supervision_v2 import (
    EVIDENCE_POLICIES, LEGACY_EVIDENCE_POLICY, UNIQUE_EVIDENCE_POLICY,
    inject_contract_negative, partial_supervision, validate_partial_row,
)


LEGACY_VERSION = "teacher-silver-repair-v2-partial-supervision"
VERSION = LEGACY_VERSION + "-unique-evidence-v1"
ROOT = Path(__file__).resolve().parents[2]


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON number")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def prepare_record(record, index, group, split, conflict=None, restoration=None, *,
                   evidence_policy=UNIQUE_EVIDENCE_POLICY):
    from .review_policy import review_record
    from .source_provenance import source_view
    from .text_normalization import (
        find_mentions, map_range, normalize_with_offsets, repair_graph_text,
    )

    if evidence_policy not in EVIDENCE_POLICIES:
        raise ValueError("Unsupported evidence grounding policy")
    original_user = _message(record, "user")
    user = original_user
    if restoration:
        if restoration["original_user_sha256"] != sha(original_user):
            raise ValueError("Source restoration belongs to another input")
        user = restoration["effective_user"]
        if restoration["effective_user_sha256"] != sha(user):
            raise ValueError("Restored source digest mismatch")
    view = source_view(user)
    mapping = normalize_with_offsets(view["raw_payload"], repair=True)
    graph = _graph(_message(record, "assistant"))
    if graph is None:
        raise ValueError(f"Invalid source graph {index}")
    # Verify inner JSON independently of the permissive historical graph reader.
    raw_assistant = _message(record, "assistant").strip()
    if raw_assistant.startswith("<output>") and raw_assistant.endswith("</output>"):
        raw_assistant = raw_assistant[len("<output>"):-len("</output>")].strip()
    if strict_json(raw_assistant) != graph:
        raise ValueError("Strict and historical graph parsers disagree")
    fixed, changes = repair_graph_text(graph)
    ids = [str(node.get("id", "")) for node in fixed["entities"]]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("Invalid or duplicate source entity IDs")
    if any(str(edge.get(k, "")) not in ids for edge in fixed["relations"] for k in ("source", "target")):
        raise ValueError("Dangling relation endpoint")
    decision = review_record(fixed, mapping["text"], source_index=index)
    issues = list(decision["issues"])
    source_issues = view.get("issues", [])
    blocked = bool(decision["record_blocked"])
    if conflict:
        issues.append({"code": "conflicting_identical_input", "details": conflict})
        blocked = True
    if any(item.get("code") == "suspected_start_truncation" for item in source_issues):
        issues.append({"code": "upstream_source_completeness_unresolved"})
        blocked = True
    result = {"source_index": index, "group_id": group, "split": split,
              "original_record": record, "original_record_sha256": sha(canonical(record)),
              "source_type": view["source_type"], "source_view": view,
              "source_view_origin": "verified_upstream_restoration" if restoration else "original_teacher_user",
              "source_restoration": restoration,
              "source": {"original_text": view["raw_payload"], "text": mapping["text"],
                         "normalization_changes": mapping["changes"],
                         "offset_basis": "source_view.raw_user payload, Python character offsets; original_record always retains the old teacher input",
                         "original_sha256": sha(view["raw_payload"]), "sha256": sha(mapping["text"])},
              "repaired_graph": fixed, "graph_changes": changes,
              "issues": issues, "blocked_pairs": decision["blocked_pairs"],
              "record_blocked": blocked, "entities": [], "annotations": [],
              "semantic_adjudication": "not_human_adjudicated",
              "unannotated_pair_policy": "UNKNOWN_MASKED_NOT_NONE"}
    if evidence_policy != LEGACY_EVIDENCE_POLICY:
        result["evidence_grounding_policy"] = evidence_policy
    if view["source_type"] != "free_text":
        result["status"] = "deterministic_source_preserved_review_untyped_fields"
        return result
    def locate(value):
        matches = []
        for start, end in find_mentions(value, mapping["text"]):
            a, b = map_range(mapping, start, end)
            matches.append({"start": start, "end": end, "text": mapping["text"][start:end],
                            "original_start": a, "original_end": b,
                            "original_text": view["raw_payload"][a:b],
                            "user_start": view["payload_start"] + a,
                            "user_end": view["payload_start"] + b})
        return matches
    for node in fixed["entities"]:
        mentions = locate(node.get("name") or node.get("properties", {}).get("Name", ""))
        result["entities"].append({"id": str(node["id"]), "canonical_name": node.get("name"),
                                   "label": node.get("label"), "mentions": mentions,
                                   "status": "exact_surface" if mentions else "alias_or_context_review"})
    blocked_pairs = {tuple(map(str, p)) for p in decision["blocked_pairs"]}
    for i, edge in enumerate(fixed["relations"]):
        evidence = edge.get("evidence") or edge.get("evidence_sentence") or ""
        values = evidence if isinstance(evidence, list) else [evidence]
        locations = [locate(v) if isinstance(v, str) and v.strip() else [] for v in values]
        annotation = {"index": i, "annotation": edge,
                      "evidence_fragments": locations,
                      "all_evidence_located": bool(locations) and all(locations),
                      "pair_blocked": blocked or (str(edge["source"]), str(edge["target"])) in blocked_pairs}
        if evidence_policy != LEGACY_EVIDENCE_POLICY:
            annotation["ambiguous_evidence_occurrence"] = any(len(hits) > 1 for hits in locations)
        result["annotations"].append(annotation)
    result["spans"] = []
    for span in SentenceSpanSplitter().split(mapping["text"]):
        a, b = map_range(mapping, span.start, span.end)
        result["spans"].append({**asdict(span), "original_start": a, "original_end": b})
    result["status"] = "quarantined_record" if blocked else "partially_supervised_silver_candidate"
    return result


def build_rows(item, config, *, sampled=True, seed=42):
    if item["source_type"] != "free_text":
        return [], {"structured_icd": 1}, set()
    graph, source = item["repaired_graph"], item["source"]["original_text"]
    index = item["source_index"]
    batches, doc, audit = build_batches(source, graph["entities"], config, sample_id=index)
    stats = Counter(records=1, all_candidate_pairs=audit["candidate_pairs"],
                    ungrounded_entities=len(audit["ungrounded_ids"]))
    positive_pairs = {(str(edge["source"]), str(edge["target"])) for edge in graph["relations"]}
    if sampled:
        all_pairs = {(pair["source"], pair["target"]) for batch in batches for pair in batch["pairs"]}
        unannotated = sorted(all_pairs - positive_pairs)
        random.Random(seed + index).shuffle(unannotated)
        selected = (all_pairs & positive_pairs) | set(unannotated[:max(8, config.negative_ratio * len(all_pairs & positive_pairs))])
        batches, doc, _ = build_batches(source, graph["entities"], config,
                                       sample_id=index, selected_pairs=selected)
    rows, supervised = [], set()
    for batch in batches:
        target, counts = partial_supervision(batch, graph, doc,
                                            blocked_pairs=item["blocked_pairs"],
                                            record_blocked=item["record_blocked"],
                                            evidence_policy=item.get("evidence_grounding_policy", LEGACY_EVIDENCE_POLICY))
        stats.update(counts)
        if not counts.get("SUPPORTED_SILVER", 0):
            stats["all_unknown_rows_excluded"] += 1
            continue
        row = {**batch, **target, "group_id": item["group_id"], "split": item["split"],
               "sample_weight": 1.0, "annotation_source": "mechanically_repaired_teacher_silver",
               "data_version": (LEGACY_VERSION if item.get("evidence_grounding_policy", LEGACY_EVIDENCE_POLICY)
                                == LEGACY_EVIDENCE_POLICY else VERSION)}
        validate_partial_row(row)
        for pair, decision in zip(row["pairs"], row["target"]["decisions"]):
            supervised.update((index, pair["source"], pair["target"], edge["relation"])
                              for edge in decision["relations"])
        rows.append(row)
    stats["rows"] = len(rows)
    stats["retained_pair_decisions"] = sum(len(row["pairs"]) for row in rows)
    stats["retained_unknown_decisions"] = sum(value.startswith("UNKNOWN_") for row in rows for value in row["pair_supervision"].values())
    return rows, dict(stats), supervised


def variants():
    base = dict(repair=True, strict_evidence=True, matching="nfkc_v2")
    return {"base": ProtocolConfig(**base),
            "short": ProtocolConfig(**base, pairs_per_batch=4),
            "source_order": ProtocolConfig(**base, order="source"),
            "local": ProtocolConfig(**base, context="local")}


def run(input_path, frozen_split, output_dir, expected_sha256, *, historical_dev=None,
        upstream_raw=None, expected_upstream_sha256=None, evidence_policy=UNIQUE_EVIDENCE_POLICY):
    from .review_policy import conflicting_inputs

    source, split_path, out = map(Path, (input_path, frozen_split, output_dir))
    if evidence_policy not in EVIDENCE_POLICIES:
        raise ValueError("Unsupported evidence grounding policy")
    if out.exists():
        raise ValueError("Refusing to overwrite any existing data version")
    before = sha(source.read_bytes())
    if before != expected_sha256:
        raise ValueError("Unexpected source SHA256")
    records = strict_json(source.read_text())
    frozen = strict_json(split_path.read_text())
    groups, splits = split_assignment(records, frozen, before)
    restorations, upstream_provenance = {}, None
    if upstream_raw:
        from .upstream_restore import source_restorations

        upstream_path = Path(upstream_raw)
        upstream_hash = sha(upstream_path.read_bytes())
        if not expected_upstream_sha256 or expected_upstream_sha256 != upstream_hash:
            raise ValueError("Unexpected upstream source SHA256")
        restorations = source_restorations(records, strict_json(upstream_path.read_text()), groups)
        upstream_provenance = {"path": str(upstream_path.resolve()), "sha256": upstream_hash,
                               "restored_source_indices": sorted(restorations)}
    conflicts = conflicting_inputs(records)
    prepared = [prepare_record(record, i, group, split, conflicts.get(i), restorations.get(i),
                               evidence_policy=evidence_policy)
                for i, (record, group, split) in enumerate(zip(records, groups, splits))]
    by_text = defaultdict(list)
    for item in prepared:
        if item["source_type"] == "free_text":
            by_text[sha(" ".join(item["source"]["text"].casefold().split()))].append(item)
    overlaps = 0
    for items in by_text.values():
        if len({item["split"] for item in items}) > 1:
            overlaps += 1
            for item in items:
                item["record_blocked"] = True
                item["status"] = "quarantined_record"
                item["issues"].append({"code": "cross_split_normalized_text_overlap"})
    summary, issue_counts, review, changes = Counter(), Counter(), [], []
    for item in prepared:
        summary["records"] += 1
        summary[item["source_type"]] += 1
        summary["quarantined_records"] += item["record_blocked"]
        summary["upstream_restorations"] += bool(item.get("source_restoration"))
        summary["free_entities"] += len(item["entities"])
        summary["ungrounded_entities"] += sum(not node["mentions"] for node in item["entities"])
        summary["free_relations"] += len(item["annotations"])
        summary["missing_evidence_annotations"] += sum(not a["all_evidence_located"] for a in item["annotations"])
        if evidence_policy != LEGACY_EVIDENCE_POLICY:
            summary["ambiguous_evidence_annotations"] += sum(a["ambiguous_evidence_occurrence"] for a in item["annotations"])
        summary["graph_changes"] += len(item["graph_changes"])
        summary["source_changes"] += len(item["source"]["normalization_changes"])
        for issue in item["issues"] + item["source_view"].get("issues", []):
            issue_counts[issue["code"]] += 1
            review.append({"source_index": item["source_index"], "split": item["split"],
                           "kind": "record_or_semantics", "details": issue,
                           "decision": "PENDING", "not_a_negative": True})
        for node in item["entities"]:
            if not node["mentions"]:
                review.append({"source_index": item["source_index"], "split": item["split"],
                               "kind": "entity", "entity_id": node["id"],
                               "reason": "alias_or_missing_context_not_confirmed_false_entity",
                               "decision": "PENDING", "not_a_negative": True})
        for annotation in item["annotations"]:
            if not annotation["all_evidence_located"]:
                review.append({"source_index": item["source_index"], "split": item["split"],
                               "kind": "relation", "relation_index": annotation["index"],
                               "reason": "evidence_scope_or_binding_unresolved", "decision": "PENDING",
                               "not_a_negative": True})
            if annotation.get("ambiguous_evidence_occurrence"):
                review.append({"source_index": item["source_index"], "split": item["split"],
                               "kind": "relation", "relation_index": annotation["index"],
                               "reason": "evidence_occurrence_ambiguous", "decision": "PENDING",
                               "not_a_negative": True})
        for field, edits in (("graph", item["graph_changes"]), ("source", item["source"]["normalization_changes"])):
            changes.extend({"source_index": item["source_index"], "field": field, "change": edit} for edit in edits)
        if item.get("source_restoration"):
            changes.append({"source_index": item["source_index"], "field": "source_prefix",
                            "change": {k:v for k,v in item["source_restoration"].items() if k != "effective_user"}})
    out.mkdir(parents=True, mode=0o700)
    def write_jsonl(name, rows):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("x") as handle:
            path.chmod(0o600)
            for row in rows:
                handle.write(canonical(row) + "\n")
        return sha(path.read_bytes())
    outputs = {}
    def write(name, rows):
        outputs[name] = write_jsonl(name, rows)
    write("records.jsonl", prepared)
    write("changes.jsonl", changes)
    write("review_queue.jsonl", review)
    write("upstream_restorations.jsonl", [restorations[i] for i in sorted(restorations)])
    write("frozen_split.jsonl", [{"source_index": i, "group_id": g, "split": s}
                                for i, (g, s) in enumerate(zip(groups, splits))])
    config_stats, sets, base_train = {}, {}, []
    for name, config in variants().items():
        subsets, stats = {"train": [], "tune": []}, {"train": Counter(), "tune": Counter()}
        label_sets = {"train": set(), "tune": set()}
        for item in prepared:
            split = item["split"]
            rows, counts, triples = build_rows(item, config, sampled=split == "train")
            subsets[split].extend(rows)
            stats[split].update(counts)
            label_sets[split].update(triples)
        for split, rows in subsets.items():
            write(f"{name}/{split}.jsonl", rows)
        sets[name] = label_sets
        config_stats[name] = {"config": asdict(config), "stats": {s: dict(c) for s,c in stats.items()}}
        if name == "base":
            base_train = subsets["train"]
        print(canonical({"variant": name, "stats": config_stats[name]["stats"]}), flush=True)
    # Keep the easier synthetic negative distribution separate and explicitly
    # labelled. It is not a substitute for clinician-reviewed real false entities.
    auxiliary = []
    for row in base_train:
        if int(sha(row["id"])[:8], 16) % 10 == 0:
            auxiliary.append(inject_contract_negative(row, prepared[row["sample_id"]]["source"]["text"]))
    write("synthetic_negative_auxiliary/train.jsonl", auxiliary)
    comparison = {}
    for name in sets:
        comparison[name] = {split: {"supervised_labels": len(sets[name][split]),
                                   "added_vs_base": len(sets[name][split] - sets["base"][split]),
                                   "missing_vs_base": len(sets["base"][split] - sets[name][split])}
                            for split in ("train", "tune")}
    # Full-context sorting and batching MUST retain exactly the same supervision.
    for name in ("short", "source_order"):
        if sets[name] != sets["base"]:
            raise RuntimeError(f"Uncontrolled supervision change in {name}")
    common = {split: set.intersection(*(value[split] for value in sets.values())) for split in ("train", "tune")}
    write("ablation_common_supervision.jsonl", [{"split": split, "source_index": i, "source": src, "target": dst, "relation": rel}
                                               for split in common for i,src,dst,rel in sorted(common[split])])
    dev_audit = None
    if historical_dev:
        path = Path(historical_dev)
        old_sha = sha(path.read_bytes())
        dev_records = strict_json(path.read_text())
        dev_conflicts = conflicting_inputs(dev_records)
        audits = [prepare_record(r, i, i, "historical_regression_audit_only", dev_conflicts.get(i),
                                  evidence_policy=evidence_policy)
                  for i,r in enumerate(dev_records)]
        write("historical_regression_audit_only.jsonl", audits)
        if sha(path.read_bytes()) != old_sha:
            raise RuntimeError("Historical regression data changed")
        dev_audit = {"records": len(dev_records), "original_sha256": old_sha, "unchanged": True,
                     "labels_rewritten_for_scoring": False}
    code_files = [p for p in (ROOT / "agent/data_repair").glob("*.py")]
    code_files += [ROOT / name for name in ("agent/generative/protocol.py", "agent/generative/train.py", "agent/train_sft.py",
                                           "agent/kg_agent/source.py", "agent/kg_agent/ontology.json", "schemas/v2.0.0/schema.json")]
    manifest = {"version": LEGACY_VERSION if evidence_policy == LEGACY_EVIDENCE_POLICY else VERSION,
                "status": "mechanically_repaired_partial_silver_candidate",
                "source_path": str(source.resolve()), "source_sha256": before,
                "frozen_split_path": str(split_path.resolve()), "frozen_split_sha256": sha(split_path.read_bytes()),
                "records_conserved": len(prepared) == len(records), "input_unchanged": sha(source.read_bytes()) == before,
                "cross_split_normalized_source_groups": overlaps,
                "summary": dict(summary), "issue_counts": dict(issue_counts), "review_items": len(review),
                "variants": config_stats, "supervision_comparison": comparison,
                "common_supervision_counts": {s:len(v) for s,v in common.items()},
                "synthetic_auxiliary_rows": len(auxiliary), "synthetic_negatives_are_clinical_gold": False,
                "historical_regression": dev_audit,
                "upstream_source": upstream_provenance,
                "code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in code_files},
                "output_sha256": outputs,
                "gates": {"mechanical_repair_executed": True, "originals_preserved": True,
                          "real_tokenizer_loss_mask_checked": False, "human_semantic_adjudication": False,
                          "real_negative_pool_adjudicated": False, "eligible_for_existing_rl_objective": False,
                          "new_model_evaluation_completed": False, "automatic_deployment_allowed": False}}
    if evidence_policy != LEGACY_EVIDENCE_POLICY:
        manifest["evidence_grounding_policy"] = evidence_policy
    if not manifest["records_conserved"] or not manifest["input_unchanged"]:
        raise RuntimeError("Source conservation failed")
    path = out / "manifest.json"
    with path.open("x") as handle:
        path.chmod(0o600)
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frozen-split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--historical-dev")
    parser.add_argument("--upstream-raw")
    parser.add_argument("--expected-upstream-sha256")
    parser.add_argument("--evidence-policy", choices=EVIDENCE_POLICIES, default=UNIQUE_EVIDENCE_POLICY,
                        help="Unique citation locations by default; legacy policy is explicit historical replay only")
    args = parser.parse_args()
    result = run(args.input, args.frozen_split, args.output_dir, args.expected_sha256,
                 historical_dev=args.historical_dev, upstream_raw=args.upstream_raw,
                 expected_upstream_sha256=args.expected_upstream_sha256,
                 evidence_policy=args.evidence_policy)
    print(canonical({"output": args.output_dir, "summary": result["summary"], "gates": result["gates"]}), flush=True)


if __name__ == "__main__":
    main()

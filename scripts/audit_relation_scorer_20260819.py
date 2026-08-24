#!/usr/bin/env python3
"""Audit relation scoring under raw-ID and semantic endpoint identities.

The DeepSeek graph is a teacher-relative reference, not a medically reviewed
gold annotation.  The audit deliberately keeps five views separate:

1. raw-ID strict: exact (source id, relation type, target id)
2. entity-aligned strict: endpoint identity includes name, label, and code/span
3. core triple only: endpoint identity includes only name and label
4. inverse-normalized: core triples plus explicitly declared inverse aliases
5. component decomposition: source, target, type, reversal, and wrong-type rates

No relation evidence or entity property ordering participates in relation
identity in any view.  Raw target/prediction strings are exported unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


CASE_FIELDS = {
    301: "indexTerms",
    251: "indexTerms",
    402: "diagnosticCriteria",
    28: "diagnosticCriteria",
    675: "exclusions",
    943: "exclusions",
    1424: "definition",
    1434: "definition",
}

# The active v3.1.1 workbook-derived table does not define inverse relations.
# This single pair is therefore an explicit *audit-policy candidate*, not an
# assertion that the source workbook specifies target types or directions.
# It is kept separate in output metadata and can be disabled with
# --no-audit-inverse-candidates.
AUDIT_INVERSE_CANDIDATES = {
    "required_for": {
        "canonical_relation": "has_diagnostic_criterion",
        "swap_endpoints": True,
        "rationale": (
            "audit candidate: Diagnostic Criterion --required_for--> Disease "
            "versus Disease --has_diagnostic_criterion--> Diagnostic Criterion"
        ),
    }
}

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I | re.S)
OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.I | re.S)
SPECIAL_TOKEN_RE = re.compile(
    r"<\|(?:im_end|endoftext|eot_id|end_of_text)\|>", re.I
)

EndpointKey = tuple[str, ...]
RelationKey = tuple[EndpointKey, str, EndpointKey]
RawRelationKey = tuple[str, str, str]


def normalize(value: Any) -> str:
    """Case-fold and collapse whitespace without changing punctuation."""
    return " ".join(str(value or "").casefold().split())


def canonical_relation_type(value: Any) -> str:
    """Keep the closed-vocabulary spelling strict; trim surrounding space."""
    return str(value or "").strip()


def parse_graph(item: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(item, dict):
        return None, "missing"
    text = str(item.get("output") or item.get("raw") or "").strip()
    match = OUTPUT_RE.search(text)
    if match:
        text = match.group(1).strip()
    text = FENCE_RE.sub("", text)
    text = SPECIAL_TOKEN_RE.sub("", text).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "non_object"
    if not isinstance(parsed.get("entities"), list) or not isinstance(
        parsed.get("relations"), list
    ):
        return None, "invalid_graph_shape"
    return parsed, "ok"


def code_or_span(entity: dict[str, Any]) -> str:
    properties = entity.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    for key in (
        "ICD-11 Code",
        "icdcode",
        "Source Span",
        "source_span",
        "Span",
        "span",
        "Evidence Span",
        "evidence_span",
    ):
        value = properties.get(key)
        if value not in (None, "", []):
            if isinstance(value, (dict, list)):
                return normalize(json.dumps(value, ensure_ascii=False, sort_keys=True))
            return normalize(value)
    return ""


def strict_entity_key(entity: dict[str, Any]) -> EndpointKey:
    """User-requested semantic key: normalized name, type, code/span."""
    return (
        normalize(entity.get("name")),
        normalize(entity.get("label")),
        code_or_span(entity),
    )


def core_entity_key(entity: dict[str, Any]) -> EndpointKey:
    """Content endpoint key that cannot cascade from a metadata-code error."""
    return (
        normalize(entity.get("name")),
        normalize(entity.get("label")),
    )


def entity_map(
    graph: dict[str, Any] | None, key_fn: Callable[[dict[str, Any]], EndpointKey]
) -> dict[str, EndpointKey]:
    if not isinstance(graph, dict):
        return {}
    result: dict[str, EndpointKey] = {}
    for entity in graph.get("entities", []):
        if isinstance(entity, dict):
            result[str(entity.get("id") or "")] = key_fn(entity)
    return result


def raw_id_relations(graph: dict[str, Any] | None) -> Counter[RawRelationKey]:
    result: Counter[RawRelationKey] = Counter()
    if not isinstance(graph, dict):
        return result
    for relation in graph.get("relations", []):
        if isinstance(relation, dict):
            result[
                (
                    str(relation.get("source") or ""),
                    canonical_relation_type(relation.get("relation")),
                    str(relation.get("target") or ""),
                )
            ] += 1
    return result


def semantic_relations(
    graph: dict[str, Any] | None,
    key_fn: Callable[[dict[str, Any]], EndpointKey],
) -> Counter[RelationKey]:
    result: Counter[RelationKey] = Counter()
    if not isinstance(graph, dict):
        return result
    by_id = entity_map(graph, key_fn)
    for relation in graph.get("relations", []):
        if not isinstance(relation, dict):
            continue
        source_id = str(relation.get("source") or "")
        target_id = str(relation.get("target") or "")
        source = by_id.get(source_id, ("<missing-id>", source_id))
        target = by_id.get(target_id, ("<missing-id>", target_id))
        result[(source, canonical_relation_type(relation.get("relation")), target)] += 1
    return result


def inverse_normalize(
    relations: Counter[RelationKey],
    inverse_rules: dict[str, dict[str, Any]],
) -> Counter[RelationKey]:
    result: Counter[RelationKey] = Counter()
    for (source, relation_type, target), count in relations.items():
        rule = inverse_rules.get(relation_type)
        if not rule:
            result[(source, relation_type, target)] += count
            continue
        canonical = str(rule["canonical_relation"])
        if bool(rule.get("swap_endpoints")):
            result[(target, canonical, source)] += count
        else:
            result[(source, canonical, target)] += count
    return result


def overlap_count(left: Counter[Any], right: Counter[Any]) -> int:
    return sum((left & right).values())


def score_counters(target: Counter[Any], prediction: Counter[Any]) -> dict[str, Any]:
    tp = overlap_count(target, prediction)
    target_n = sum(target.values())
    prediction_n = sum(prediction.values())
    fp = prediction_n - tp
    fn = target_n - tp
    precision = tp / prediction_n if prediction_n else (1.0 if not target_n else 0.0)
    recall = tp / target_n if target_n else (1.0 if not prediction_n else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "target_count": target_n,
        "prediction_count": prediction_n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def jsonable_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return [jsonable_key(item) for item in key]
    return key


def counter_rows(counter: Counter[Any]) -> list[dict[str, Any]]:
    return [
        {"key": jsonable_key(key), "count": count}
        for key, count in sorted(counter.items(), key=lambda item: repr(item[0]))
    ]


def diff_rows(
    target: Counter[Any], prediction: Counter[Any]
) -> dict[str, list[dict[str, Any]]]:
    return {
        "matched": counter_rows(target & prediction),
        "false_negative": counter_rows(target - prediction),
        "false_positive": counter_rows(prediction - target),
    }


def id_endpoint_audit(
    target_graph: dict[str, Any] | None,
    prediction_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    """Explain where local ID agreement disagrees with semantic identity."""
    target_core = entity_map(target_graph, core_entity_key)
    pred_core = entity_map(prediction_graph, core_entity_key)
    target_strict = entity_map(target_graph, strict_entity_key)
    pred_strict = entity_map(prediction_graph, strict_entity_key)

    same_id_core_conflicts = []
    same_id_metadata_conflicts = []
    for entity_id in sorted(set(target_core) & set(pred_core)):
        if target_core[entity_id] != pred_core[entity_id]:
            same_id_core_conflicts.append(
                {
                    "entity_id": entity_id,
                    "target_core_key": jsonable_key(target_core[entity_id]),
                    "prediction_core_key": jsonable_key(pred_core[entity_id]),
                }
            )
        elif target_strict[entity_id] != pred_strict[entity_id]:
            same_id_metadata_conflicts.append(
                {
                    "entity_id": entity_id,
                    "shared_core_key": jsonable_key(target_core[entity_id]),
                    "target_code_or_span": target_strict[entity_id][2],
                    "prediction_code_or_span": pred_strict[entity_id][2],
                }
            )

    target_raw = raw_id_relations(target_graph)
    pred_raw = raw_id_relations(prediction_graph)
    matched_raw = target_raw & pred_raw
    semantic_consistent = 0
    semantic_conflicting = 0
    conflict_rows = []
    for (source_id, relation_type, target_id), count in matched_raw.items():
        target_semantic = (
            target_core.get(source_id, ("<missing-id>", source_id)),
            relation_type,
            target_core.get(target_id, ("<missing-id>", target_id)),
        )
        pred_semantic = (
            pred_core.get(source_id, ("<missing-id>", source_id)),
            relation_type,
            pred_core.get(target_id, ("<missing-id>", target_id)),
        )
        if target_semantic == pred_semantic:
            semantic_consistent += count
        else:
            semantic_conflicting += count
            conflict_rows.append(
                {
                    "raw_id_key": [source_id, relation_type, target_id],
                    "count": count,
                    "target_core_key": jsonable_key(target_semantic),
                    "prediction_core_key": jsonable_key(pred_semantic),
                }
            )

    return {
        "same_id_core_entity_conflicts": same_id_core_conflicts,
        "same_id_only_code_or_span_conflicts": same_id_metadata_conflicts,
        "matched_raw_id_relation_count": sum(matched_raw.values()),
        "matched_raw_id_relations_semantically_consistent": semantic_consistent,
        "matched_raw_id_relations_semantically_conflicting": semantic_conflicting,
        "raw_id_semantic_conflict_details": conflict_rows,
    }


def component_score(target: Counter[Any], prediction: Counter[Any]) -> dict[str, Any]:
    return score_counters(target, prediction)


def decompose(
    target: Counter[RelationKey],
    prediction: Counter[RelationKey],
    inverse_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_sources: Counter[EndpointKey] = Counter()
    target_targets: Counter[EndpointKey] = Counter()
    target_types: Counter[str] = Counter()
    pred_sources: Counter[EndpointKey] = Counter()
    pred_targets: Counter[EndpointKey] = Counter()
    pred_types: Counter[str] = Counter()

    for (source, relation_type, endpoint), count in target.items():
        target_sources[source] += count
        target_targets[endpoint] += count
        target_types[relation_type] += count
    for (source, relation_type, endpoint), count in prediction.items():
        pred_sources[source] += count
        pred_targets[endpoint] += count
        pred_types[relation_type] += count

    unmatched_target = target - prediction
    unmatched_prediction = prediction - target
    same_type_reversed = 0
    endpoints_right_type_wrong = 0

    # These diagnostic categories are evaluated only after exact core triples
    # have been removed, and consume prediction counts to avoid double-counting.
    remaining_prediction = unmatched_prediction.copy()
    for (source, relation_type, endpoint), count in unmatched_target.items():
        reversed_key = (endpoint, relation_type, source)
        rescued = min(count, remaining_prediction.get(reversed_key, 0))
        if rescued:
            same_type_reversed += rescued
            remaining_prediction[reversed_key] -= rescued
            if remaining_prediction[reversed_key] <= 0:
                del remaining_prediction[reversed_key]

    for (source, relation_type, endpoint), count in unmatched_target.items():
        candidates = [
            key
            for key in remaining_prediction
            if key[0] == source and key[2] == endpoint and key[1] != relation_type
        ]
        outstanding = count
        for key in sorted(candidates, key=repr):
            rescued = min(outstanding, remaining_prediction.get(key, 0))
            endpoints_right_type_wrong += rescued
            outstanding -= rescued
            remaining_prediction[key] -= rescued
            if remaining_prediction[key] <= 0:
                del remaining_prediction[key]
            if not outstanding:
                break

    raw_core = score_counters(target, prediction)
    normalized_target = inverse_normalize(target, inverse_rules)
    normalized_prediction = inverse_normalize(prediction, inverse_rules)
    inverse_score = score_counters(normalized_target, normalized_prediction)
    inverse_rescued = inverse_score["tp"] - raw_core["tp"]
    target_n = sum(target.values())

    return {
        "denominator_target_relations": target_n,
        "source_endpoint": component_score(target_sources, pred_sources),
        "target_endpoint": component_score(target_targets, pred_targets),
        "relation_type": component_score(target_types, pred_types),
        "exact_core_triple": raw_core,
        "same_relation_type_but_direction_reversed": {
            "count": same_type_reversed,
            "rate_over_target": same_type_reversed / target_n if target_n else 0.0,
        },
        "both_endpoints_correct_but_type_wrong": {
            "count": endpoints_right_type_wrong,
            "rate_over_target": endpoints_right_type_wrong / target_n if target_n else 0.0,
        },
        "audit_inverse_candidate_rescued": {
            "count": inverse_rescued,
            "rate_over_target": inverse_rescued / target_n if target_n else 0.0,
        },
    }


def add_counter(total: Counter[Any], value: Counter[Any]) -> None:
    total.update(value)


def aggregate_cases(
    rows: Iterable[dict[str, Any]], inverse_rules: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    regimes = ("raw_id_strict", "entity_aligned_strict", "core_triple_only")
    target_totals: dict[str, Counter[Any]] = {name: Counter() for name in regimes}
    pred_totals: dict[str, Counter[Any]] = {name: Counter() for name in regimes}

    # Prefix both endpoints by case ID. Entity IDs and semantic endpoints are
    # local to a graph and must not collide across examples in a micro aggregate.
    for row in rows:
        case_id = row["id"]
        for name in regimes:
            for key, count in row["_counters"][name]["target"].items():
                source, relation_type, endpoint = key
                aggregate_key = (
                    (str(case_id), source),
                    relation_type,
                    (str(case_id), endpoint),
                )
                target_totals[name][aggregate_key] += count
            for key, count in row["_counters"][name]["prediction"].items():
                source, relation_type, endpoint = key
                aggregate_key = (
                    (str(case_id), source),
                    relation_type,
                    (str(case_id), endpoint),
                )
                pred_totals[name][aggregate_key] += count

    core_target = target_totals["core_triple_only"]
    core_pred = pred_totals["core_triple_only"]
    inverse_target = inverse_normalize(core_target, inverse_rules)
    inverse_pred = inverse_normalize(core_pred, inverse_rules)
    return {
        "raw_id_strict": score_counters(
            target_totals["raw_id_strict"], pred_totals["raw_id_strict"]
        ),
        "entity_aligned_strict": score_counters(
            target_totals["entity_aligned_strict"],
            pred_totals["entity_aligned_strict"],
        ),
        "core_triple_only": score_counters(core_target, core_pred),
        "inverse_normalized": score_counters(inverse_target, inverse_pred),
        "decomposition": decompose(core_target, core_pred, inverse_rules),
    }


def raw_export_row(case: dict[str, Any], model_field: str) -> dict[str, Any]:
    target = case.get("deepseek_original")
    prediction = case.get(model_field)
    target = target if isinstance(target, dict) else {}
    prediction = prediction if isinstance(prediction, dict) else {}
    return {
        "id": case.get("id"),
        "split": case.get("split"),
        "source_field": CASE_FIELDS[int(case["id"])],
        "target": {
            "raw": target.get("raw"),
            "output": target.get("output"),
        },
        "prediction": {
            "model_field": model_field,
            "raw": prediction.get("raw"),
            "output": prediction.get("output"),
        },
        "generation_meta": case.get("generation_meta", {}).get(model_field),
    }


def audit_case(
    case: dict[str, Any],
    model_field: str,
    inverse_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    target_graph, target_status = parse_graph(case.get("deepseek_original"))
    pred_graph, pred_status = parse_graph(case.get(model_field))

    target_raw = raw_id_relations(target_graph)
    pred_raw = raw_id_relations(pred_graph)
    target_strict = semantic_relations(target_graph, strict_entity_key)
    pred_strict = semantic_relations(pred_graph, strict_entity_key)
    target_core = semantic_relations(target_graph, core_entity_key)
    pred_core = semantic_relations(pred_graph, core_entity_key)
    target_inverse = inverse_normalize(target_core, inverse_rules)
    pred_inverse = inverse_normalize(pred_core, inverse_rules)

    counters = {
        "raw_id_strict": {"target": target_raw, "prediction": pred_raw},
        "entity_aligned_strict": {
            "target": target_strict,
            "prediction": pred_strict,
        },
        "core_triple_only": {"target": target_core, "prediction": pred_core},
    }

    return {
        "id": int(case["id"]),
        "split": case.get("split"),
        "source_field": CASE_FIELDS[int(case["id"])],
        "parse_status": {"target": target_status, "prediction": pred_status},
        "entity_counts": {
            "target": len(target_graph.get("entities", [])) if target_graph else 0,
            "prediction": len(pred_graph.get("entities", [])) if pred_graph else 0,
        },
        "id_endpoint_audit": id_endpoint_audit(target_graph, pred_graph),
        "scores": {
            "raw_id_strict": score_counters(target_raw, pred_raw),
            "entity_aligned_strict": score_counters(target_strict, pred_strict),
            "core_triple_only": score_counters(target_core, pred_core),
            "inverse_normalized": score_counters(target_inverse, pred_inverse),
            "decomposition": decompose(target_core, pred_core, inverse_rules),
        },
        "differences": {
            "raw_id_strict": diff_rows(target_raw, pred_raw),
            "entity_aligned_strict": diff_rows(target_strict, pred_strict),
            "core_triple_only": diff_rows(target_core, pred_core),
            "inverse_normalized": diff_rows(target_inverse, pred_inverse),
        },
        "_counters": counters,
    }


def strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_private(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [strip_private(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model-field", default="output_only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument(
        "--no-audit-inverse-candidates",
        action="store_true",
        help="disable the explicit audit-only required_for inverse candidate",
    )
    args = parser.parse_args()

    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise SystemExit("artifact must contain a cases list")

    by_id = {
        int(case["id"]): case
        for case in cases
        if isinstance(case, dict) and str(case.get("id", "")).isdigit()
    }
    missing = sorted(set(CASE_FIELDS) - set(by_id))
    if missing:
        raise SystemExit(f"artifact is missing requested cases: {missing}")

    inverse_rules = (
        {} if args.no_audit_inverse_candidates else AUDIT_INVERSE_CANDIDATES
    )
    ordered_cases = [by_id[case_id] for case_id in CASE_FIELDS]
    audited = [audit_case(case, args.model_field, inverse_rules) for case in ordered_cases]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audited:
        groups[f"split:{row['split']}"] .append(row)
        groups[f"field:{row['source_field']}"] .append(row)

    report = {
        "metadata": {
            "artifact": str(args.artifact),
            "model_field": args.model_field,
            "case_ids": list(CASE_FIELDS),
            "teacher_reference_warning": (
                "DeepSeek target is a noisy teacher-relative reference, not medically "
                "reviewed gold. Scores measure agreement, not clinical correctness."
            ),
            "relation_identity": {
                "raw_id_strict": ["source_id", "exact_relation_type", "target_id"],
                "entity_aligned_strict": [
                    ["normalize(entity_name)", "normalize(entity_type)", "code_or_span"],
                    "exact_relation_type",
                    ["normalize(entity_name)", "normalize(entity_type)", "code_or_span"],
                ],
                "core_triple_only": [
                    ["normalize(entity_name)", "normalize(entity_type)"],
                    "exact_relation_type",
                    ["normalize(entity_name)", "normalize(entity_type)"],
                ],
                "excluded_from_all_relation_identities": [
                    "evidence",
                    "descriptions",
                    "non-endpoint properties",
                    "property ordering",
                    "entity/relation list ordering",
                ],
            },
            "schema_declared_inverse_pairs": [],
            "audit_inverse_candidates_enabled": bool(inverse_rules),
            "audit_inverse_candidates": inverse_rules,
            "inverse_policy_warning": (
                "The active workbook-derived schema does not declare inverse pairs. "
                "Enabled candidates are diagnostic hypotheses and are reported separately."
            ),
        },
        "aggregate": aggregate_cases(audited, inverse_rules),
        "grouped": {
            name: aggregate_cases(rows, inverse_rules)
            for name, rows in sorted(groups.items())
        },
        "cases": audited,
    }
    report = strip_private(report)

    raw_report = {
        "metadata": {
            "artifact": str(args.artifact),
            "model_field": args.model_field,
            "case_ids": list(CASE_FIELDS),
            "note": "raw strings are copied unchanged from the generation artifact",
        },
        "cases": [raw_export_row(case, args.model_field) for case in ordered_cases],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.raw_output.write_text(
        json.dumps(raw_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

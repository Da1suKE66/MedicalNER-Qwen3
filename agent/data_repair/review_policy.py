"""Conservative REVIEW gates, not medical adjudication or an ontology migration.

``review_record`` never edits a graph or invents an empty relation. A blocked
ordered pair means quarantine its supervision as UNKNOWN pending review, not
teach NONE. Mere context/negation hints do not block pairs. Evidence grounding,
NFKC repair and offsets belong to the caller; this module has no grounding gate.

``conflicting_inputs`` returns a sparse source_index -> review-result mapping.
Each value additionally exposes ``code`` and ``related_indices`` (other group
members, excluding itself) directly, for the caller's whole-record gate.
It compares COMPLETE, byte-for-byte system/user strings. Graph comparison ignores
arbitrary node IDs, entity/edge order and superficial Unicode/whitespace changes,
but retains names, types, properties (including ICD codes), evidence and graph
attributes. Different evidence is inconsistent supervision, not proof of a false
medical fact. No synonym, disease subtype or ICD-code merging is attempted.

Dependencies: standard library and the existing kg_agent.contracts only.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from types import MappingProxyType
import unicodedata

from kg_agent.contracts import allowed_relations, normalize_label


POLICY_VERSION = "review-policy-v2"

# Candidate domain/range checks for ADJUDICATION, NOT replacement signatures.
# In particular, the production ontology constrains source modules only. These
# references never change allowed_relations and do not assert clinical falsehood.
TYPE_REVIEW_REFERENCES = MappingProxyType({
    "is_core_symptom_of": (("Symptom", "Disease"),),
    "is_associated_symptom_of": (("Symptom", "Disease"),),
    "differentiates_from": (("Disease", "Disease"),),
    "subtype_of": (("Disease", "Disease"),),
    "rules_out": (("Disease", "Disease"),),
    "has_diagnostic_criterion": (("Disease", "Diagnostic Criterion"),),
    "required_for": (("Diagnostic Criterion", "Disease"),),
    "supports_diagnosis_of": (("Symptom", "Disease"), ("Examination", "Disease")),
    "supports_Diagnostic Criterion_of": (("Symptom", "Diagnostic Criterion"),),
    "argues_against": (("Symptom", "Disease"),),
    "associated_with_risk": (("Disease", "Risk"),),
    "increases_risk_of": (("Risk", "Disease"),),
})


def _text(value):
    """Comparison-only normalization; never return a rewritten source/graph."""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _fold(value):
    return _text(value).casefold()


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_dump(value).encode("utf-8")).hexdigest()


def _id(value):
    if type(value) not in (str, int) or not str(value).strip():
        raise ValueError("invalid_entity_or_endpoint_id")
    return str(value)


def _name(node):
    properties = node.get("properties")
    value = node.get("name") or (properties.get("Name") if isinstance(properties, dict) else None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_entity_name")
    return value


def _inspect_graph(graph):
    """Structural ambiguity must not turn into an apparently safe empty result."""
    if not isinstance(graph, dict) or not all(isinstance(graph.get(k), list) for k in ("entities", "relations")):
        raise ValueError("invalid_graph_shape")
    by_id = {}
    for node in graph["entities"]:
        if not isinstance(node, dict):
            raise ValueError("invalid_entity_shape")
        key = _id(node.get("id"))
        if key in by_id:
            raise ValueError("duplicate_entity_id")
        _name(node)
        if not isinstance(node.get("label"), str) or not node["label"].strip():
            raise ValueError("invalid_entity_type")
        by_id[key] = node
    for edge in graph["relations"]:
        if not isinstance(edge, dict):
            raise ValueError("invalid_relation_shape")
        source, target = _id(edge.get("source")), _id(edge.get("target"))
        if source not in by_id or target not in by_id:
            raise ValueError("missing_relation_endpoint")
        if not isinstance(edge.get("relation"), str) or not edge["relation"].strip():
            raise ValueError("invalid_relation_label")
    return by_id


_POPULATION = (
    r"(?:(?:typically|normally)[\s-]+developing|healthy|normal)\s+"
    r"(?:children|child|adults|adolescents|people|individuals|peers|controls)\b"
    r"|(?:正常(?:发育的?)?|健康|典型发育)(?:儿童|孩子|成人|青少年|人群|个体|对照)"
)
_NORMAL_POPULATION = re.compile(_POPULATION, re.I)
_NORMAL_ASSERTION = re.compile(
    r"(?:" + _POPULATION + r")[^.;。；!?！？]{0,70}?"
    r"(?:\b(?:show|shows|exhibit|exhibits|experience|experiences|have|has|display|demonstrate)\b|出现|表现|具有)"
    r"|\b(?:common|typical|expected|normal)\s+(?:in|among|for)\s+(?:" + _POPULATION + r")",
    re.I,
)
_THRESHOLD_OR_CONTRAST = re.compile(
    r"\b(?:diagnos\w*|criteria|criterion|threshold|require\w*|persist\w*|sustained|"
    r"impair\w*|sever\w*|marked\w*|exceed\w*|beyond|below|above|"
    r"but|whereas|however|although|unlike|compared|instead|except|"
    r"disorder\w*|disease\w*|indicat\w*|suggest\w*|signal\w*)\b"
    r"|\b(?:more|less)\s+than\b|\bat\s+least\b|\brather\s+than\b"
    r"|\b\d+(?:\.\d+)?\s+(?:weeks?|months?|years?)\b"
    r"|诊断|标准|阈值|持续|损害|受损|严重|明显|超过|低于|高于|但是|然而|相比|除外",
    re.I,
)
_NEGATIVE_SCOPE = re.compile(
    r"\b(?:not|never|no|cannot|without|absen\w*|exclud\w*|neither|nor)\b"
    r"|\b(?:do|does|did|is|are|was|were|ca|could|would|should|wo)n['’]t\b"
    r"|并非|不是|不能|未见|没有|无此",
    re.I,
)
# Multiple propositions need scope adjudication, even when one clause clearly
# describes a healthy population. Prefer a nonblocking hint to overgeneralizing.
_MIXED_CLAUSES = re.compile(
    r"[.;!?。；！？]\s*\S|\b(?:and|or|while|yet|unless|if|when|because|therefore|"
    r"despite|which|whose)\b|以及|但是|然而|如果|尽管|而且|同时|但|且|则",
    re.I,
)
_NORMALITY_HEADING = re.compile(r"boundary\s+with\s+normality|normal\s+variation|正常性边界|正常变异", re.I)


def _evidence_fragments(edge):
    value = edge.get("evidence")
    if value is None:
        value = edge.get("evidence_sentence")
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        return []
    return [_fold(v) for v in values]


def _normality_review(target, edge):
    """Local target + citation semantics only; no whole-document keyword veto."""
    name = _fold(_name(target))
    evidence = _evidence_fragments(edge)
    if not evidence:
        return None  # The caller, not this policy, handles missing evidence.
    if not any(_NORMAL_POPULATION.search(v) for v in [name, *evidence]):
        return None
    normal_citations = [v for v in evidence if _NORMAL_ASSERTION.search(v)]
    # Require the target itself to make the normal-population assertion. A short
    # symptom phrase merely occurring in such a citation is not enough to block.
    linked = any(name in v or v in name for v in normal_citations)
    target_asserts_normality = bool(_NORMAL_ASSERTION.search(name))
    mixed = any(_THRESHOLD_OR_CONTRAST.search(v) or _NEGATIVE_SCOPE.search(v) or _MIXED_CLAUSES.search(v)
                for v in [name, *evidence])
    if target_asserts_normality and linked and not mixed and len(normal_citations) == len(evidence):
        return ("normal_population_as_diagnostic_criterion", "quarantine_pair", {
            "reason": "Target and cited evidence describe an ordinary/healthy population, not a disease-specific diagnostic requirement; adjudication required",
            "target_has_population_assertion": target_asserts_normality,
            "citation_fragments": len(evidence),
        })
    if mixed:
        return ("normality_mixed_or_threshold_scope_hint", "hint", {
            "reason": "Normality and diagnostic threshold/contrast/negative scope coexist; do not infer a contradiction or delete a criterion",
        })
    return ("normality_context_or_binding_hint", "hint", {
        "reason": "Population wording alone or unlinked target/citation is insufficient for quarantine",
    })


def review_record(graph, text, *, source_index=None) -> dict:
    """Return issues, ordered-pair quarantine IDs and a whole-record block flag.

    A type-reference mismatch is an explicit REVIEW quarantine, not enforcement
    of a new ontology. No issue means only that these few rules did not fire.
    Missing/unannotated pairs are never enumerated, classified or made negative.
    Pair IDs use strings, as in the existing repair pipeline. Quarantine is a
    caller-owned UNKNOWN loss mask, not deletion of the containing batch.
    All returned containers are independent of the input graph.
    """
    result = {"issues": [], "blocked_pairs": [], "record_blocked": False}
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    graph = deepcopy(graph)

    def issue(code, action, *, index=None, pair=None, details=None):
        item = {"code": code, "action": action,
                "severity": "hint" if action == "hint" else "review",
                "source_index": source_index, "relation_index": index,
                "policy_version": POLICY_VERSION}
        if pair is not None:
            item.update(source=pair[0], target=pair[1])
        if details:
            item.update(deepcopy(details))
        result["issues"].append(item)
        if action == "quarantine_record":
            result["record_blocked"] = True
        elif action == "quarantine_pair" and list(pair) not in result["blocked_pairs"]:
            result["blocked_pairs"].append(list(pair))

    try:
        nodes = _inspect_graph(graph)
    except ValueError as exc:
        issue(str(exc), "quarantine_record")
        return result
    for i, edge in enumerate(graph["relations"]):
        pair = (_id(edge["source"]), _id(edge["target"]))
        source, target = (nodes[key] for key in pair)
        observed = (normalize_label(source["label"]), normalize_label(target["label"]))
        relation = edge["relation"]
        if None in observed or relation not in allowed_relations(*observed):
            issue("existing_source_module_requires_review", "quarantine_pair", index=i, pair=pair,
                  details={"observed_types": list(observed), "relation": relation,
                           "reason": "Unrecognized type or relation outside the existing source module; do not invent a replacement label"})
        reference = TYPE_REVIEW_REFERENCES.get(relation)
        if reference and observed not in reference:
            issue("relation_type_reference_requires_review", "quarantine_pair", index=i, pair=pair,
                  details={"observed_types": list(observed), "relation": relation,
                           "reference_type_pairs": [list(p) for p in reference],
                           "reference_authority": "adjudication_reference_not_production_ontology",
                           "reason": "Source/target range or concept granularity needs review; current ontology is not narrowed and no medical falsity is asserted"})
        if relation == "has_diagnostic_criterion":
            finding = _normality_review(target, edge)
            if finding:
                code, action, details = finding
                issue(code, action, index=i, pair=pair, details=details)
            elif _NORMALITY_HEADING.search(text) and any(_NORMALITY_HEADING.search(v) for v in _evidence_fragments(edge)):
                issue("normality_section_only_hint", "hint", index=i, pair=pair,
                      details={"reason": "A normality section can contain valid diagnostic thresholds; heading alone is not a blocking rule"})
    return result


def _canonical_value(value):
    if isinstance(value, str):
        return _text(value)
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_canonical_value(v) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _canonical_value(v) for k, v in value.items()}
    raise ValueError("invalid_non_json_graph_value")


def _canonical_graph(graph):
    nodes = _inspect_graph(graph)
    semantic_ids = {}
    for key, node in nodes.items():
        # Keep ALL attributes, including ICD codes and subtype descriptions.
        attributes = {k: _canonical_value(v) for k, v in node.items()
                      if k not in {"id", "entity_id", "name", "label"}}
        semantic_ids[key] = _dump({"label": normalize_label(node["label"]) or _text(node["label"]),
                                   "name": _fold(_name(node)), "attributes": attributes})
    edges = []
    for edge in graph["relations"]:
        item = {k: _canonical_value(v) for k, v in edge.items() if k not in {"source", "target"}}
        for key in ("evidence", "evidence_sentence"):
            if isinstance(item.get(key), list):
                item[key] = sorted(item[key], key=_dump)
        item.update(source=semantic_ids[_id(edge["source"])], target=semantic_ids[_id(edge["target"])])
        edges.append(_dump(item))
    return {"entities": sorted(semantic_ids.values()), "relations": sorted(set(edges)),
            "attributes": {k: _canonical_value(v) for k, v in graph.items() if k not in {"entities", "relations"}}}


def _strict_graph(answer):
    # Known whole-answer wrappers only; never guess a JSON substring or coerce a
    # malformed assistant response to an empty graph.
    wrapped = re.fullmatch(r"\s*<output>\s*(.*?)\s*</output>\s*", answer, re.S | re.I)
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", answer, re.S | re.I)
    value = (wrapped or fenced).group(1) if wrapped or fenced else answer

    def object_hook(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("duplicate_json_key")
        return dict(pairs)

    def invalid_constant(_):
        raise ValueError("nonfinite_json")

    try:
        return json.loads(value, object_pairs_hook=object_hook, parse_constant=invalid_constant)
    except (ValueError, TypeError):
        raise ValueError("invalid_assistant_graph_json") from None


def conflicting_inputs(records) -> dict:
    """Sparse source_index -> whole-record review mapping for identical inputs.

    Accept a sequence of raw {messages: [system, user, assistant]} records; an
    explicit nonnegative integer source_index overrides the sequence position.
    No stripping, NFKC, substring or body-only comparison of input messages is
    used. Every member of a conflicting input group is quarantined, even members
    sharing one of its graphs. Malformed annotations are also blocked and cannot
    make an otherwise conflicting group appear consistent.

    Values have top-level ``code`` and ``related_indices`` (the other source
    indices), plus ``conflict_group_source_indices`` including the current one,
    ``issues``, ``blocked_pairs`` and ``record_blocked``. An empty blocked_pairs
    list does not override record_blocked=True: the whole record is UNKNOWN.
    """
    groups = defaultdict(list)
    seen_indices = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Record must be a mapping")
        index = record.get("source_index", position)
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ValueError("source_index must be a unique nonnegative integer")
        seen_indices.add(index)
        messages = record.get("messages")
        if (not isinstance(messages, list) or len(messages) != 3
                or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages)
                or [m.get("role") for m in messages] != ["system", "user", "assistant"]):
            raise ValueError("Require complete ordered system/user/assistant messages")
        # Group by exact strings, not a possibly lossy hash alone.
        key = (messages[0]["content"], messages[1]["content"])
        try:
            graph = _canonical_graph(_strict_graph(messages[2]["content"]))
            canonical, invalid = _dump(graph), False
        except (ValueError, TypeError, KeyError):
            canonical, invalid = None, True
        groups[key].append((index, canonical, invalid))
    result = {}
    for key, members in groups.items():
        variants = {value for _, value, invalid in members if not invalid}
        invalid = any(flag for _, _, flag in members)
        if not invalid and len(variants) <= 1:
            continue
        indices = sorted(index for index, _, _ in members)
        for index, canonical, bad in members:
            code = "invalid_graph_in_identical_input_group" if invalid else "conflicting_identical_inputs"
            issue = {"code": code, "severity": "review", "action": "quarantine_record",
                     "source_index": index, "policy_version": POLICY_VERSION,
                     "input_sha256": _hash(list(key)),
                     "canonical_graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest() if canonical is not None else None,
                     "conflict_group_source_indices": list(indices),
                     "related_indices": [j for j in indices if j != index],
                     "related_source_indices": [j for j in indices if j != index],
                     "distinct_valid_canonical_graphs": len(variants),
                     "this_graph_invalid": bad,
                     "reason": "Identical complete input cannot disambiguate the differing graphs/annotations; retain subtypes and all originals for review, never merge or train NONE"}
            result[index] = {"code": code,
                             "related_indices": [j for j in indices if j != index],
                             "conflict_group_source_indices": list(indices),
                             "issues": [issue], "blocked_pairs": [], "record_blocked": True}
    return dict(sorted(result.items()))

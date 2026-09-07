"""Stable intermediate contracts for the two-stage medical KG pipeline.

The public graph format intentionally stays small.  All fields that require
grounding (IDs, names, evidence text, and metadata) are owned by Python.  The
LLM stages only exchange span IDs, labels, pair IDs, and relation names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import json
from pathlib import Path


LABEL_ALIASES = {
    "disease": "Disease",
    "disorder": "Disease",
    "condition": "Disease",
    "symptom": "Symptom",
    "sign": "Symptom",
    "diagnostic criterion": "Diagnostic Criterion",
    "diagnostic_criterion": "Diagnostic Criterion",
    "criterion": "Diagnostic Criterion",
    "etiology": "Etiology",
    "cause": "Etiology",
    "risk factor": "Risk Factor",
    "risk_factor": "Risk Factor",
    "risk": "Risk",
    "assessment scale": "Assessment Scale",
    "assessment_scale": "Assessment Scale",
    "assessment": "Assessment",
    "examination": "Examination",
    "treatment": "Treatment",
    "therapy": "Treatment",
    "prognostic factor": "Prognostic Factor",
    "functional impact": "Functional Impact",
    "test": "Test",
    "investigation": "Test",
    "medication": "Medication",
    "procedure": "Procedure",
    "anatomy": "Anatomy",
    "phenotype": "Phenotype",
}

LABEL_PREFIXES = {
    "Disease": "D",
    "Symptom": "S",
    "Diagnostic Criterion": "DC",
    "Etiology": "E",
    "Risk Factor": "RF",
    "Treatment": "T",
    "Prognostic Factor": "PF",
    "Functional Impact": "FI",
    "Test": "X",
    "Medication": "M",
    "Procedure": "P",
    "Anatomy": "A",
    "Phenotype": "PH",
    "Risk": "R",
    "Assessment Scale": "AS",
    "Assessment": "AS",
    "Examination": "EX",
}

IMPORTANCE_ORDER = {
    "required": 0,
    "core": 1,
    "differential": 2,
    "treatment": 3,
    "risk": 4,
    "associated": 5,
    "incidental": 6,
    "alias": 7,
}

# The classifier is only allowed to choose from this closed relation space.
# Add project-specific signatures here instead of letting the model invent
# relation names at inference time.
RELATION_SIGNATURES: dict[tuple[str, str], tuple[str, ...]] = {
    ("Symptom", "Disease"): (
        "is_core_symptom_of",
        "is_associated_symptom_of",
        "supports_diagnosis_of",
        "argues_against",
        "suggests",
    ),
    ("Disease", "Symptom"): ("presents_with", "causes"),
    ("Disease", "Disease"): (
        "subtype_of",
        "differentiates_from",
        "rules_out",
        "associated_with",
        "progresses_to",
        "causes",
        "has_specifier",
        "predisposes_to",
    ),
    ("Disease", "Diagnostic Criterion"): ("has_diagnostic_criterion",),
    ("Diagnostic Criterion", "Disease"): (
        "required_for",
        "supports_diagnosis_of",
        "co_occurs_with",
    ),
    ("Etiology", "Disease"): ("causes", "is_associated_with"),
    ("Etiology", "Symptom"): ("causes", "contributes_to"),
    ("Symptom", "Symptom"): (
        "has_manifestation",
        "part_of",
        "precedes",
        "relieved_by",
        "worsens",
    ),
    ("Symptom", "Diagnostic Criterion"): ("supports_Diagnostic Criterion_of",),
    ("Disease", "Risk"): ("associated_with_risk",),
    ("Risk", "Disease"): ("increases_risk_of",),
    ("Disease", "Assessment Scale"): ("recommended_assessment",),
    ("Assessment Scale", "Disease"): ("evaluates",),
    ("Disease", "Assessment"): ("recommended_assessment",),
    ("Disease", "Examination"): ("recommended_examination",),
    ("Examination", "Disease"): ("supports_diagnosis_of",),
    ("Medication", "Symptom"): ("causes_side_effect",),
    ("Risk Factor", "Disease"): ("is_risk_factor_for", "increases_risk_of"),
    ("Disease", "Treatment"): ("treated_by",),
    ("Treatment", "Disease"): ("treats",),
    ("Disease", "Test"): ("evaluated_by",),
    ("Test", "Disease"): ("supports_diagnosis_of",),
    ("Disease", "Functional Impact"): ("has_functional_impact",),
    ("Functional Impact", "Disease"): ("is_functional_impact_of",),
}


ONTOLOGY = json.loads(Path(__file__).with_name("ontology.json").read_text(encoding="utf-8"))
SCHEMA_LABELS = tuple(ONTOLOGY["node_labels"])
LABEL_PREFIXES.update({"Patient": "PT", "Treatment Plan": "TP", "Communication Strategy": "CS", "Guideline": "G", "Evidence": "EV"})
LABEL_PREFIXES = {label: LABEL_PREFIXES[label] for label in SCHEMA_LABELS}
LABEL_ALIASES = {alias: label for alias, label in LABEL_ALIASES.items() if label in SCHEMA_LABELS}
LABEL_ALIASES.update({label.casefold(): label for label in SCHEMA_LABELS})
# The original ontology constrains source modules, not target types.
RELATION_SIGNATURES = {
    (source, target): tuple(dict.fromkeys(r["relation"] for r in ONTOLOGY["source_relations"] if r["source_label"] == source))
    for source in SCHEMA_LABELS for target in SCHEMA_LABELS
}


def normalize_label(value: Any) -> str | None:
    """Return the canonical label or ``None`` for an unknown label."""

    if value is None:
        return None
    raw = " ".join(str(value).strip().casefold().replace("-", " ").split())
    return LABEL_ALIASES.get(raw)


def allowed_relations(source_label: str, target_label: str) -> tuple[str, ...]:
    return RELATION_SIGNATURES.get((source_label, target_label), ())


def normalize_importance(value: Any) -> str:
    raw = str(value or "associated").strip().casefold().replace(" ", "_")
    if raw in {"core_symptom", "core_feature"}:
        return "core"
    if raw in {"risk_factor", "riskfactor"}:
        return "risk"
    return raw if raw in IMPORTANCE_ORDER else "associated"


@dataclass(frozen=True)
class Span:
    id: str
    text: str
    start: int
    end: int
    section: str = "body"


@dataclass
class SourceDocument:
    source_type: str
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)

    def span_map(self) -> dict[str, Span]:
        return {span.id: span for span in self.spans}


@dataclass(frozen=True)
class EntityCandidate:
    span_id: str
    label: str
    importance: str = "associated"
    name: str | None = None


@dataclass
class Entity:
    id: str
    label: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    span_ids: list[str] = field(default_factory=list)
    importance: str = "associated"


@dataclass
class Relation:
    source: str
    target: str
    relation: str
    evidence_ids: list[str] = field(default_factory=list)
    evidence_text: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class ValidationIssue:
    kind: str
    message: str
    item: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, kind: str, message: str, item: dict[str, Any] | None = None) -> None:
        self.issues.append(ValidationIssue(kind, message, item or {}))
        self.valid = False


def natural_id_key(value: str) -> tuple[str, int]:
    """Sort IDs such as D1, DC2, and RF10 deterministically."""

    prefix = "".join(char for char in value if not char.isdigit())
    digits = "".join(char for char in value if char.isdigit())
    return prefix, int(digits or 0)


def iter_relation_signatures() -> Iterable[tuple[str, str, str]]:
    for (source, target), relations in RELATION_SIGNATURES.items():
        for relation in relations:
            yield source, target, relation

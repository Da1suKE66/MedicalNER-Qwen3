"""Input-only generative batches, strict decisions, and auditable rewards.

IDs are assigned by the program; the causal LM generates relation/evidence choices.
No gold labels/evidence are accepted by build_batches. Train and inference share it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

from kg_agent.contracts import allowed_relations, normalize_label
from kg_agent.source import SourceRouter


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def corrections():
    path = Path(__file__).resolve().parents[2] / "schemas/v2.0.0/schema.json"
    return json.loads(path.read_text())["text_corrections"]


@lru_cache(maxsize=1)
def correction_pattern():
    return re.compile(
        r"(?<!\w)(?:"
        + "|".join(map(re.escape, sorted(corrections(), key=len, reverse=True)))
        + r")(?!\w)",
        re.I,
    )


def repair_text(text):
    """Only known whole-word corruptions, never global null->nan replacement."""
    fixes = corrections()
    pattern = correction_pattern()
    changes = []

    def replace(match):
        before = match.group()
        after = fixes[before.lower()]
        if before.isupper():
            after = after.upper()
        elif before[0].isupper():
            after = after[0].upper() + after[1:]
        changes.append(
            {
                "start": match.start(),
                "end": match.end(),
                "before": before,
                "after": after,
            }
        )
        return after

    return pattern.sub(replace, text), changes


def exact_hits(name, text):
    words = str(name or "").strip().split()
    if not words:
        return []
    pattern = re.compile(
        r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", re.I
    )
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def span_hits(text, document):
    hits = exact_hits(text, document.raw)
    return {s.id for s in document.spans for a, b in hits if s.start < b and s.end > a}


def original_offset(position, changes, *, end=False):
    """Map repaired-text offsets back to immutable original source offsets."""
    delta = 0
    for change in changes:
        start = change["start"] + delta
        stop = start + len(change["after"])
        if position < start:
            break
        if position < stop:
            return change["end"] if end else change["start"]
        delta += len(change["after"]) - (change["end"] - change["start"])
    return position - delta


@dataclass(frozen=True)
class ProtocolConfig:
    repair: bool = False
    strict_evidence: bool = False
    pairs_per_batch: int = 16
    order: str = "hash"
    context: str = "full"
    multilabel: bool = True
    rebalance: bool = False
    negative_ratio: int = 3

    def __post_init__(self):
        if (
            self.pairs_per_batch < 1
            or self.order not in {"hash", "source"}
            or self.context not in {"full", "local"}
        ):
            raise ValueError("Invalid generative protocol")
        if self.negative_ratio < 0:
            raise ValueError("negative_ratio must be nonnegative")


def build_batches(raw, nodes, config, *, sample_id="", selected_pairs=None):
    """No pair prefix cap; missing grounding is diagnosed, not silently invented.

    selected_pairs is a TRAIN-ONLY sampling set; never pass it in evaluation.
    Local context retains at least one mention of both endpoints of every pair.
    It can still miss relation evidence: this is measured during supervision.
    Original node names and graph IDs are retained for strict external scoring.
    """
    text, changes = repair_text(raw) if config.repair else (raw, [])
    doc = SourceRouter().route(text)
    if doc.source_type != "free_text":
        raise ValueError("Relation model must not process structured ICD records")
    entities, missing = [], []
    known = set()
    for node in nodes:
        old_id = str(node.get("id", ""))
        if not old_id or old_id in known:
            raise ValueError("Input entity IDs must be nonempty and unique")
        known.add(old_id)
        original = str(node.get("name") or node.get("properties", {}).get("Name", ""))
        name = repair_text(original)[0] if config.repair else original
        label = normalize_label(node.get("label"))
        hits = exact_hits(name, doc.raw)
        if not hits or not label:
            missing.append(old_id)
            continue
        positions = sorted(
            {
                i
                for i, s in enumerate(doc.spans)
                for a, b in hits
                if s.start < b and s.end > a
            }
        )
        entities.append(
            {
                "id": old_id,
                "name": name,
                "label": label,
                "positions": positions,
                "start": hits[0][0],
            }
        )
    entities.sort(key=lambda n: (n["start"], n["label"], n["name"].casefold(), n["id"]))
    candidates = []
    for source in entities:
        for target in entities:
            allowed = list(allowed_relations(source["label"], target["label"]))
            if source["id"] == target["id"] or not allowed:
                continue
            if (
                selected_pairs is not None
                and (source["id"], target["id"]) not in selected_pairs
            ):
                continue
            candidates.append(
                {"source": source["id"], "target": target["id"], "allowed": allowed}
            )
    if config.order == "hash":
        by_id = {e["id"]: e for e in entities}
        # Excludes labels/graph gold, graph IDs, and sample index from the order.
        candidates.sort(
            key=lambda p: digest(
                dumps(
                    [
                        (by_id[p[k]]["name"], by_id[p[k]]["label"])
                        for k in ("source", "target")
                    ]
                )
            )
        )
    by_id = {e["id"]: e for e in entities}
    batches = []
    for offset in range(0, len(candidates), config.pairs_per_batch):
        pairs = [
            {**p, "pair_id": f"P{i+1:04d}"}
            for i, p in enumerate(candidates[offset : offset + config.pairs_per_batch])
        ]
        if config.context == "full":
            spans = doc.spans
        else:
            indices = set()
            for pair in pairs:
                a, b = min(
                    (
                        (i, j)
                        for i in by_id[pair["source"]]["positions"]
                        for j in by_id[pair["target"]]["positions"]
                    ),
                    key=lambda ij: (abs(ij[0] - ij[1]), ij),
                )
                # Two short mention windows, not the entire distant interval.
                for pos in (a, b):
                    indices.update(range(max(0, pos - 1), min(len(doc.spans), pos + 2)))
                # A single clinical concept may cross a splitter boundary.
                # Keep the full exact occurrence, including all crossed spans.
                for key in ("source", "target"):
                    hit = exact_hits(by_id[pair[key]]["name"], doc.raw)[0]
                    indices.update(
                        i
                        for i, s in enumerate(doc.spans)
                        if s.start < hit[1] and s.end > hit[0]
                    )
            spans = [doc.spans[i] for i in sorted(indices)]
        visible_ids = {p[k] for p in pairs for k in ("source", "target")}
        # Fresh local IDs remove arbitrary upstream graph-ID semantics.
        table = [
            {
                "entity_id": f"E{i+1:03d}",
                "id": e["id"],
                "name": e["name"],
                "label": e["label"],
            }
            for i, e in enumerate(entities)
            if e["id"] in visible_ids
        ]
        mapping = {e["id"]: e["entity_id"] for e in table}
        payload = {
            "entities": [{k: v for k, v in e.items() if k != "id"} for e in table],
            "pairs": [
                {
                    "pair_id": p["pair_id"],
                    "source": mapping[p["source"]],
                    "target": mapping[p["target"]],
                    "allowed_relations": p["allowed"],
                }
                for p in pairs
            ],
            "spans": [{"span_id": s.id, "text": s.text} for s in spans],
        }
        prompt = (
            "Extract only explicitly supported medical relations for the fixed ordered pairs. "
            "Entity names/types and numbered source text are provided; IDs carry no medical meaning. "
            "Use only each pair's allowed relations. A pair can have multiple distinct relations. "
            "Cite the source span(s) supporting each relation. Co-mention does not establish a relation. "
            "Preserve pair order. Return one decision per pair; use an empty relations list when none is supported. "
            'Return JSON only: {"decisions":[{"pair_id":"P0001","relations":[{"relation":"RELATION","evidence_span_ids":["SENT_001"]}]}]}.\n'
            + dumps(payload)
            + "\n"
        )
        original_spans = {
            s.id: raw[
                original_offset(s.start, changes) : original_offset(
                    s.end, changes, end=True
                )
            ]
            for s in spans
        }
        batches.append(
            {
                "id": f"{sample_id}:{offset}",
                "sample_id": sample_id,
                "prompt": prompt,
                "pairs": pairs,
                "spans": payload["spans"],
                "original_spans": original_spans,
                "entities": table,
            }
        )
    return (
        batches,
        doc,
        {
            "candidate_pairs": len(candidates),
            "ungrounded_ids": missing,
            "repairs": changes,
        },
    )


def supervision(batch, graph, doc, config):
    """Return None for UNKNOWN, never teach NONE from a failed annotation."""
    annotated = {}
    for edge in graph["relations"]:
        annotated.setdefault((str(edge["source"]), str(edge["target"])), []).append(
            edge
        )
    visible = {s["span_id"] for s in batch["spans"]}
    decisions, stats = [], Counter()
    for pair in batch["pairs"]:
        relations = {}
        for edge in annotated.get((pair["source"], pair["target"]), []):
            label = edge.get("relation")
            ev = edge.get("evidence") or edge.get("evidence_sentence") or ""
            if isinstance(ev, list):
                texts = ev
            else:
                texts = [ev]
            evidence = set()
            for value in texts:
                value = repair_text(str(value))[0] if config.repair else str(value)
                if value in doc.span_map():
                    evidence.add(value)
                else:
                    evidence.update(span_hits(value, doc))
            covered = (
                evidence <= visible
                if config.strict_evidence
                else bool(evidence & visible)
            )
            if label not in pair["allowed"] or not evidence or not covered:
                return None, {
                    "unknown_batches": 1,
                    "reason": "unavailable_annotation_or_evidence",
                }
            relations.setdefault(label, set()).update(evidence & visible)
        labels = sorted(relations)
        stats["original_positive_edges"] += len(labels)
        stats["multi_relation_pairs"] += int(len(labels) > 1)
        if not config.multilabel:
            stats["intentionally_dropped_extra_labels"] += max(0, len(labels) - 1)
            labels = labels[:1]
        stats["positive_edges"] += len(labels)
        stats["none_pairs"] += not labels
        decisions.append(
            {
                "pair_id": pair["pair_id"],
                "relations": [
                    {"relation": label, "evidence_span_ids": sorted(relations[label])}
                    for label in labels
                ],
            }
        )
    return {"decisions": decisions}, dict(stats)


def parse_decisions(text, batch):
    """Strict JSON, all pairs exactly once, no silently swallowed invalid edges."""
    errors, valid = [], []
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return [], ["invalid_json"]
    if (
        not isinstance(data, dict)
        or set(data) != {"decisions"}
        or not isinstance(data["decisions"], list)
    ):
        return [], ["invalid_shape"]
    expected = {p["pair_id"]: p for p in batch["pairs"]}
    visible = {s["span_id"] for s in batch["spans"]}
    seen = set()
    order = []
    for decision in data["decisions"]:
        if not isinstance(decision, dict) or set(decision) != {"pair_id", "relations"}:
            errors.append("invalid_decision")
            continue
        pid = decision["pair_id"]
        if not isinstance(pid, str) or pid not in expected:
            errors.append("unknown_pair")
            continue
        if pid in seen:
            errors.append("duplicate_pair")
            continue
        seen.add(pid)
        order.append(pid)
        if not isinstance(decision["relations"], list):
            errors.append("invalid_relations")
            continue
        labels = set()
        for edge in decision["relations"]:
            if not isinstance(edge, dict) or set(edge) != {
                "relation",
                "evidence_span_ids",
            }:
                errors.append("invalid_edge")
                continue
            label = edge["relation"]
            refs = edge["evidence_span_ids"]
            if not isinstance(label, str) or label not in expected[pid]["allowed"]:
                errors.append("unknown_relation")
                continue
            if label in labels:
                errors.append("duplicate_relation")
                continue
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or ref not in visible for ref in refs)
            ):
                errors.append("invalid_evidence")
                continue
            labels.add(label)
            valid.append(
                {
                    "pair_id": pid,
                    "relation": label,
                    "evidence_span_ids": sorted(set(refs)),
                }
            )
    if seen != set(expected):
        errors.append("missing_pair_decisions")
    if order != list(expected):
        errors.append("pair_order_mismatch")
    return valid, errors


def reward_components(completion, batch, target, *, evidence_weight=0.2):
    predicted, errors = parse_decisions(completion, batch)
    gold, target_errors = parse_decisions(dumps(target), batch)
    if target_errors:
        raise ValueError(f"Invalid reward target: {target_errors}")
    p = {(r["pair_id"], r["relation"]) for r in predicted}
    g = {(r["pair_id"], r["relation"]) for r in gold}
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f05 = (
        1.25 * precision * recall / (0.25 * precision + recall)
        if precision + recall
        else 0.0
    )
    # Strongly discourage the degenerate all-NONE solution on positive prompts.
    empty_correct = float(not g and not p and not errors)
    gold_evidence = {
        (r["pair_id"], r["relation"]): set(r["evidence_span_ids"]) for r in gold
    }
    overlap = sum(
        len(
            set(r["evidence_span_ids"])
            & gold_evidence.get((r["pair_id"], r["relation"]), set())
        )
        / len(
            set(r["evidence_span_ids"])
            | gold_evidence.get((r["pair_id"], r["relation"]), set())
        )
        for r in predicted
    )
    evidence = overlap / max(1, len(p), len(g))
    reward = (
        0.65 * f05
        + 0.15 * recall
        + evidence_weight * evidence
        + 0.8 * empty_correct
        - 0.3 * bool(errors)
        - 0.2 * fp / max(1, len(p))
        - 0.2 * bool(g and not p)
    )
    return {
        "reward": reward,
        "f0_5": f05,
        "recall": recall,
        "evidence": evidence,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "invalid": bool(errors),
        "empty_on_positive": bool(g and not p),
    }


def load_variant(path, variant):
    spec = json.loads(Path(path).read_text())
    return ProtocolConfig(**{**spec["defaults"], **spec["variants"][variant]})

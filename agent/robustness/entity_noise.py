"""Post-grounding synthetic entity errors with strictly separate train/stress APIs.

This is an independent augmentation layer, not a change to inference grounding.
Only manufactured, absent names receive new empty incident decisions. Real
prediction differences, synonyms, type-only disagreements and silver omissions
are NOT adjudicated here. Harder medical negatives require separate review.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
import unicodedata

from generative.protocol import ProtocolConfig, build_batches, dumps, parse_decisions
from kg_agent.contracts import SCHEMA_LABELS, allowed_relations, normalize_label


VERSION = "entity-noise-v1"


class NoiseError(ValueError):
    """Contract failure. Messages deliberately exclude source text and names."""


class NoEligibleMutation(NoiseError):
    """Cannot modify this batch while preserving the positive-survivor guard."""


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def training_fingerprint(row):
    """Ignore upstream IDs/metadata when detecting identical training views."""
    return fingerprint([row["prompt"], row["completion"]])


def _require(condition, message):
    if not condition:
        raise NoiseError(message)


def _fold(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


@dataclass(frozen=True)
class NoiseConfig:
    seed: int = 42
    operation: str = "replace"
    type_mode: str = "mixed"
    injected_pairs: int = 2
    noisy_weight: float = 1.0

    def __post_init__(self):
        _require(type(self.seed) is int, "Seed must be an integer")
        _require(self.operation in {"replace", "inject"}, "Unsupported operation")
        _require(self.type_mode in {"keep", "change", "mixed"},
                 "Type-only or realistic negatives require review, not empty targets")
        _require(type(self.injected_pairs) is int and self.injected_pairs >= 1,
                 "injected_pairs must be positive")
        _require(isinstance(self.noisy_weight, (int, float))
                 and math.isfinite(self.noisy_weight) and 0 < self.noisy_weight <= 1,
                 "Noisy weights must be finite and in (0, 1]")


@lru_cache(maxsize=1)
def protocol_prefix():
    """Obtain the shared header from a synthetic input, without copying protocol."""
    batches, _, _ = build_batches(
        "Concept Alpha and Concept Beta.",
        [{"id": "a", "name": "Concept Alpha", "label": "Disease"},
         {"id": "b", "name": "Concept Beta", "label": "Disease"}],
        ProtocolConfig(),
    )
    return batches[0]["prompt"].rsplit("\n", 2)[0]


def prompt_payload(row):
    """Explicit input-only projection; never interpolate row/provenance metadata."""
    mapping = {e["id"]: e["entity_id"] for e in row["entities"]}
    return {
        "entities": [{k: e[k] for k in ("entity_id", "name", "label")}
                     for e in row["entities"]],
        "pairs": [{"pair_id": p["pair_id"], "source": mapping[p["source"]],
                   "target": mapping[p["target"]], "allowed_relations": p["allowed"]}
                  for p in row["pairs"]],
        "spans": [{"span_id": s["span_id"], "text": s["text"]}
                  for s in row["spans"]],
    }


def validate_row(row, *, split="train"):
    """Fail closed on UNKNOWN, stale prompts, ID/type/evidence drift or split drift."""
    try:
        _require(isinstance(row, dict) and row.get("split") == split,
                 "Explicit matching split required")
        _require(not (split == "train" and "stress_evaluation" in row),
                 "Held-out stress artifacts cannot be used for training")
        _require(isinstance(row.get("id"), str) and bool(row["id"]), "Missing row ID")
        for key in ("group_id", "sample_id"):
            _require(type(row.get(key)) in (int, str) and str(row[key]) != "",
                     "Explicit source group/sample identity required")
        weight = row.get("sample_weight", 1.0)
        _require(isinstance(weight, (int, float)) and math.isfinite(weight) and weight > 0,
                 "Invalid sampling weight")
        entities, pairs, spans = row["entities"], row["pairs"], row["spans"]
        _require(all(isinstance(v, list) and v for v in (entities, pairs, spans)),
                 "Nonempty entity/pair/span tables required")
        ids, local_ids, by_id = set(), set(), {}
        for entity in entities:
            eid, local = entity["id"], entity["entity_id"]
            _require(isinstance(eid, str) and eid and eid not in ids,
                     "Duplicate or invalid upstream entity ID")
            _require(isinstance(local, str) and re.fullmatch(r"E\d{3,}", local)
                     and local not in local_ids, "Duplicate or invalid local entity ID")
            _require(isinstance(entity["name"], str) and entity["name"].strip(),
                     "Empty entity name")
            _require(entity["label"] in SCHEMA_LABELS
                     and normalize_label(entity["label"]) == entity["label"],
                     "Noncanonical entity type")
            ids.add(eid)
            local_ids.add(local)
            by_id[eid] = entity
        seen_pairs, endpoints, pair_ids = set(), set(), set()
        for pair in pairs:
            source, target, pid = pair["source"], pair["target"], pair["pair_id"]
            _require(source in ids and target in ids and source != target,
                     "Invalid directed pair endpoints")
            _require((source, target) not in seen_pairs, "Duplicate directed pair")
            _require(isinstance(pid, str) and re.fullmatch(r"P\d{4,}", pid)
                     and pid not in pair_ids, "Duplicate or invalid pair ID")
            expected = list(allowed_relations(by_id[source]["label"], by_id[target]["label"]))
            _require(expected and pair["allowed"] == expected, "Stale allowed_relations")
            seen_pairs.add((source, target))
            endpoints.update((source, target))
            pair_ids.add(pid)
        _require(endpoints == ids, "Unused or missing prompt entities")
        span_ids = [s["span_id"] for s in spans]
        _require(all(isinstance(s, str) and s for s in span_ids)
                 and len(set(span_ids)) == len(span_ids), "Duplicate or invalid span ID")
        _require(all(isinstance(s["text"], str) for s in spans), "Invalid source spans")
        if "original_spans" in row:
            _require(isinstance(row["original_spans"], dict)
                     and set(row["original_spans"]) == set(span_ids)
                     and all(isinstance(v, str) for v in row["original_spans"].values()),
                     "Original/source span mapping mismatch")
        prefix, payload, suffix = row["prompt"].rsplit("\n", 2)
        _require(prefix == protocol_prefix() and suffix == "", "Unrecognized prompt header")
        _require(json.loads(payload) == prompt_payload(row),
                 "Prompt/table mismatch or metadata leakage")
        _require(isinstance(row.get("target"), dict), "UNKNOWN is not an empty target")
        _require(json.loads(row["completion"]) == row["target"], "Target/completion mismatch")
        _, errors = parse_decisions(dumps(row["target"]), row)
        _require(not errors, "Invalid decisions or evidence contract")
    except NoiseError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError):
        raise NoiseError("Malformed generative row; no source text included") from None
    return row


def _rank(config, row, purpose, value):
    key = [VERSION, config.seed, row["id"], purpose, value]
    # Preserve v1 training determinism while domain-separating held-out nonces.
    if row["split"] != "train":
        key = ["heldout_stress", row["split"], *key]
    return fingerprint(key)


def _fresh_id(prefix, occupied, key):
    for attempt in range(len(occupied) + 1):
        candidate = prefix + fingerprint([key, attempt])[:24]
        if candidate not in occupied:
            return candidate
    raise NoiseError("Could not allocate unique entity ID")


def _nonce_name(config, row, source_texts):
    # Pronounceable synthetic nonces, not clinical terms borrowed from another
    # patient/record. Absence is a mechanical construction check, not adjudication.
    syllables = ("ba", "de", "fi", "go", "hu", "ja", "ke", "lo",
                 "mi", "nu", "pa", "re", "si", "to", "va", "zu")
    texts = [_fold(s["text"]) for s in row["spans"]]
    texts += [_fold(v) for v in row.get("original_spans", {}).values()]
    texts += [_fold(v) for v in source_texts]
    names = {_fold(e["name"]) for e in row["entities"]}
    for attempt in range(128):
        code = _rank(config, row, "absent_name", attempt)
        name = " ".join("".join(syllables[int(h, 16)] for h in code[i:i + 6]).title()
                        for i in (0, 6))
        if _fold(name) not in names and all(_fold(name) not in t for t in texts):
            return name
    raise NoEligibleMutation("Unable to construct an absent nonce name")


def _select_entity(row, config):
    positives = [p for p, d in zip(row["pairs"], row["target"]["decisions"])
                 if d["relations"]]
    candidates = row["entities"]
    if config.operation == "replace" and positives:
        candidates = [e for e in candidates
                      if any(e["id"] not in (p["source"], p["target"]) for p in positives)]
    if not candidates:
        raise NoEligibleMutation("Replacement would erase every positive decision")
    return min(candidates, key=lambda e: _rank(config, row, "entity", e["entity_id"]))


def augment_row(row, config=None, *, source_texts=()):
    """Training-only entry point. A held-out row can never enter this wrapper."""
    return _augment_row(row, config, source_texts=source_texts, expected_split="train")


def augment_heldout_row(row, *, split, config=None, source_texts=()):
    """Return a tune/test stress row, never temporarily or persistently relabel train.

    Targets are used read-only to preserve clean positives and construct a stress
    reference. The output is not training supervision and default train validation
    rejects it. ``split`` must explicitly match the frozen source row's split.
    """
    _require(split in {"tune", "test"}, "Held-out split must explicitly be tune or test")
    return _augment_row(row, config, source_texts=source_texts, expected_split=split)


def _augment_row(row, config=None, *, source_texts=(), expected_split):
    """Return one deep-copied, auditable noisy row; never mutate the clean input.

    Replacement changes one upstream ID and updates BOTH directed endpoints;
    local IDs and pair order remain stable. Injection adds a neutral local ID and
    a bounded subset of incident ordered pairs. Nonincident supervision is copied
    exactly, including evidence and inherited silver-empty decisions.
    """
    config = config or NoiseConfig()
    validate_row(row, split=expected_split)
    _require("entity_noise" not in row, "Recursive augmentation is forbidden")
    source = _select_entity(row, config)
    result = deepcopy(row)
    key = [fingerprint(row), asdict(config)]
    new_id = _fresh_id("C", {e["id"] for e in row["entities"]}, key)
    new_name = _nonce_name(config, row, source_texts)
    change_type = (config.type_mode == "change" or
                   config.type_mode == "mixed" and int(_rank(config, row, "type", 0), 16) % 2 == 0)
    label = source["label"]
    if change_type:
        candidates = [t for t in SCHEMA_LABELS if t != label
                      and all(allowed_relations(t if p["source"] == source["id"]
                                                else next(e["label"] for e in row["entities"] if e["id"] == p["source"]),
                                                t if p["target"] == source["id"]
                                                else next(e["label"] for e in row["entities"] if e["id"] == p["target"]))
                              for p in row["pairs"])]
        if not candidates:
            raise NoEligibleMutation("No schema-compatible type corruption")
        label = min(candidates, key=lambda t: _rank(config, row, "new_type", t))
    if config.operation == "replace":
        entity = next(e for e in result["entities"] if e["id"] == source["id"])
        entity.update(id=new_id, name=new_name, label=label)
        for pair in result["pairs"]:
            for endpoint in ("source", "target"):
                if pair[endpoint] == source["id"]:
                    pair[endpoint] = new_id
    else:
        local = f"E{1 + max(int(e['entity_id'][1:]) for e in row['entities']):03d}"
        entity = {"id": new_id, "entity_id": local, "name": new_name, "label": label}
        result["entities"].append(entity)
        # Both orientations are considered; no inferred clean-clean negatives.
        pairs = [(a, b) for old in row["entities"] for a, b in ((entity, old), (old, entity))
                 if allowed_relations(a["label"], b["label"])]
        pairs.sort(key=lambda ab: _rank(config, row, "injected_pair",
                                       [ab[0]["entity_id"], ab[1]["entity_id"]]))
        if not pairs:
            raise NoEligibleMutation("Injected entity has no allowed candidate pairs")
        next_pid = 1 + max(int(p["pair_id"][1:]) for p in row["pairs"])
        for i, (a, b) in enumerate(pairs[:config.injected_pairs]):
            pid = f"P{next_pid + i:04d}"
            result["pairs"].append({"pair_id": pid, "source": a["id"], "target": b["id"],
                                    "allowed": list(allowed_relations(a["label"], b["label"]))})
            result["target"]["decisions"].append({"pair_id": pid, "relations": []})
    by_id = {e["id"]: e for e in result["entities"]}
    incident = []
    for pair, decision in zip(result["pairs"], result["target"]["decisions"]):
        pair["allowed"] = list(allowed_relations(by_id[pair["source"]]["label"],
                                                  by_id[pair["target"]]["label"]))
        if new_id in (pair["source"], pair["target"]):
            decision["relations"] = []
            incident.append(pair["pair_id"])
    result["id"] = row["id"] + ":entity-" + fingerprint(key)[:24]
    result["sample_weight"] = min(row.get("sample_weight", 1.0), config.noisy_weight)
    result["completion"] = dumps(result["target"])
    result["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(result)) + "\n"
    result["entity_noise"] = {
        "version": VERSION, "config": asdict(config),
        "source_row_id": row["id"], "source_row_sha256": fingerprint(row),
        "source_training_view_sha256": training_fingerprint(row),
        "id_mapping": ({source["id"]: new_id} if config.operation == "replace" else {}),
        "changed_entity_ids": [new_id], "changed_local_ids": [entity["entity_id"]],
        "incident_pair_ids": incident, "parent_entity_id": source["id"],
        "old_label": source["label"], "new_label": label,
        "old_name_sha256": fingerprint(source["name"]), "new_name_sha256": fingerprint(new_name),
        "construction": "absent_synthetic_name_and_type" if change_type else "absent_synthetic_name",
        "absence_scope": "provided_source_spans_NFKC_casefold_whitespace",
        "clinical_quality": "synthetic_contract_negative_not_validated_medical_negative",
        "harder_realistic_noise": "PENDING_REVIEW_NOT_TRAINED",
        "inherited_nonincident_empty": "unchanged_not_reclassified_as_verified_negative",
    }
    if expected_split != "train":
        result["stress_evaluation"] = {
            "split": expected_split, "usage": "heldout_stress_only_not_training",
            "selection_must_be_model_result_independent": True,
        }
    validate_row(result, split=expected_split)
    audit_augmentation(row, result, expected_split=expected_split)
    return result


def audit_augmentation(clean, noisy, *, expected_split="train"):
    """Check conservation using upstream IDs, not positional source/target guesses."""
    _require(expected_split in {"train", "tune", "test"}, "Invalid expected audit split")
    validate_row(clean, split=expected_split)
    validate_row(noisy, split=expected_split)
    meta = noisy["entity_noise"]
    _require(meta["source_row_sha256"] == fingerprint(clean), "Source provenance mismatch")
    for key in ("split", "group_id", "sample_id", "spans", "original_spans"):
        _require(clean.get(key) == noisy.get(key), "Source/split/evidence was modified")
    changed = set(meta["changed_entity_ids"])
    mapping = meta["id_mapping"]
    decisions = {d["pair_id"]: d for d in noisy["target"]["decisions"]}
    pairs = {p["pair_id"]: p for p in noisy["pairs"]}
    removed_edges = retained_edges = incident_pairs = 0
    for pair, target in zip(clean["pairs"], clean["target"]["decisions"]):
        after = pairs.get(pair["pair_id"])
        _require(after is not None, "Original pair was dropped")
        for endpoint in ("source", "target"):
            _require(after[endpoint] == mapping.get(pair[endpoint], pair[endpoint]),
                     "Directed endpoint mapping mismatch")
        if changed.intersection((after["source"], after["target"])):
            removed_edges += len(target["relations"])
        else:
            _require(after == pair and decisions[pair["pair_id"]] == target,
                     "Nonincident pair/positive/evidence changed")
            retained_edges += len(target["relations"])
    for pair in noisy["pairs"]:
        if changed.intersection((pair["source"], pair["target"])):
            _require(decisions[pair["pair_id"]]["relations"] == [], "Noise incident is not empty")
            incident_pairs += 1
    _require(incident_pairs > 0, "No error entity exposed in a candidate pair")
    visible = {e["id"] for e in noisy["entities"]}
    _require(changed <= visible, "Error entity was lost before prompt rendering")
    _require(not any(d["relations"] for d in clean["target"]["decisions"])
             or any(d["relations"] for d in noisy["target"]["decisions"]),
             "A positive batch became all-empty")
    return {"removed_incident_positive_edges": removed_edges,
            "retained_nonincident_positive_edges": retained_edges,
            "noise_incident_pairs": incident_pairs, "noise_entities": len(changed)}


def augment_batch(batch, target, *, split, group_id, config=None, source_texts=()):
    """Adapter for protocol.build_batches + supervision; UNKNOWN stays an error."""
    row = {**batch, "target": target, "completion": dumps(target),
           "split": split, "group_id": group_id, "sample_weight": 1.0}
    return augment_row(row, config, source_texts=source_texts)

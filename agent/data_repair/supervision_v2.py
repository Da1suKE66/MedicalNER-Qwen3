"""Partial generation supervision. UNKNOWN is masked, never a negative label.

The input pair table is left intact. Empty UNKNOWN placeholders exist only to
serialize a valid fixed-order completion; their entire token spans have no loss.
This does not make incomplete silver annotation suitable for an RL reward.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re

from generative.protocol import dumps, parse_decisions
from kg_agent.contracts import SCHEMA_LABELS, allowed_relations, normalize_label


CONTRACT = "pair-unknown-loss-mask-v1"
KNOWN = {"SUPPORTED_SILVER", "SYNTHETIC_NEGATIVE"}
LEGACY_EVIDENCE_POLICY = "legacy-all-occurrences-v1"
UNIQUE_EVIDENCE_POLICY = "unique-evidence-v1"
EVIDENCE_POLICIES = (UNIQUE_EVIDENCE_POLICY, LEGACY_EVIDENCE_POLICY)


def serialize_partial(target, statuses):
    pieces, ignored, position = ['{"decisions":['], [], len('{"decisions":[')
    decisions = target["decisions"]
    for index, decision in enumerate(decisions):
        # Explicit field order survives JSONL writers that sort object keys.
        value = dumps({"pair_id": decision["pair_id"], "relations": [
            {"relation": edge["relation"], "evidence_span_ids": edge["evidence_span_ids"]}
            for edge in decision["relations"]]})
        if statuses[decision["pair_id"]] not in KNOWN:
            # Include adjoining separators so a fused BPE token cannot teach
            # the masked empty list through a neighbouring syntax target.
            ignored.append([max(len('{"decisions":['), position - int(index > 0)),
                            position + len(value) + int(index + 1 < len(decisions))])
        pieces.append(value)
        position += len(value)
        if index + 1 < len(decisions):
            pieces.append(",")
            position += 1
    pieces.append("]}")
    merged = []
    for start, end in ignored:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return "".join(pieces), merged


def partial_supervision(batch, graph, doc, *, blocked_pairs=(), record_blocked=False,
                        evidence_policy=UNIQUE_EVIDENCE_POLICY):
    """Preserve every valid label of one pair; quarantine an incomplete pair.

    No annotation means unknown. Finding one valid label never proves the
    other labels on the pair absent. Source/target IDs and evidence stay local.
    New targets require each citation to locate uniquely in the full document,
    even if a local window hides another occurrence. Explicit span references
    are already disambiguated. The legacy policy is for historical replay only;
    existing serialized rows are never changed by validation.
    """
    from .text_normalization import find_mentions

    if evidence_policy not in EVIDENCE_POLICIES:
        raise ValueError("Unsupported evidence grounding policy")
    blocked = {tuple(map(str, pair)) for pair in blocked_pairs}
    annotated = defaultdict(list)
    for edge in graph["relations"]:
        annotated[(str(edge["source"]), str(edge["target"]))].append(edge)
    visible = {span["span_id"] for span in batch["spans"]}
    decisions, statuses, reasons, stats = [], {}, {}, Counter()
    for pair in batch["pairs"]:
        key = pair["source"], pair["target"]
        pid = pair["pair_id"]
        labels = defaultdict(set)
        reason = ""
        if record_blocked or key in blocked:
            status, reason = "UNKNOWN_REVIEW", "semantic_or_source_review"
        elif not annotated[key]:
            status, reason = "UNKNOWN_UNANNOTATED", "silver_absence_is_not_negative"
        else:
            for edge in annotated[key]:
                values = edge.get("evidence") or edge.get("evidence_sentence") or ""
                values = values if isinstance(values, list) else [values]
                evidence = set()
                if not values or any(not isinstance(v, str) or not v.strip() for v in values):
                    reason = "empty_or_invalid_evidence"
                    break
                for value in values:
                    if value in doc.span_map():
                        spans = {value}
                    else:
                        hits = find_mentions(value, doc.raw)
                        if len(hits) > 1 and evidence_policy == UNIQUE_EVIDENCE_POLICY:
                            reason = "evidence_occurrence_ambiguous"
                            break
                        spans = {s.id for s in doc.spans for a, b in hits
                                 if s.start < b and a < s.end}
                    if not spans:
                        reason = "evidence_not_grounded"
                        break
                    if not spans <= visible:
                        reason = "evidence_outside_context"
                        break
                    evidence.update(spans)
                if reason:
                    break
                if edge.get("relation") not in pair["allowed"]:
                    reason = "relation_outside_contract"
                    break
                labels[edge["relation"]].update(evidence)
            status = "UNKNOWN_EVIDENCE" if reason else "SUPPORTED_SILVER"
        statuses[pid] = status
        if reason:
            reasons[pid] = reason
        relations = ([{"relation": label, "evidence_span_ids": sorted(labels[label])}
                      for label in sorted(labels)] if status == "SUPPORTED_SILVER" else [])
        decisions.append({"pair_id": pid, "relations": relations})
        stats[status] += 1
        stats["positive_edges"] += len(relations)
        stats["multilabel_pairs"] += len(relations) > 1
    target = {"decisions": decisions}
    completion, ignored = serialize_partial(target, statuses)
    result = {"target": target, "completion": completion, "pair_supervision": statuses,
              "unknown_reasons": reasons, "loss_mask_spans": ignored,
              "loss_mask_contract": CONTRACT, "eligible_for_rl": False}
    if evidence_policy != LEGACY_EVIDENCE_POLICY:
        result["evidence_grounding_policy"] = evidence_policy
    return result, dict(stats)


def _validate_input_tables(row):
    """Check the actual source ontology without silently tightening its range.

    Prompt/row equality alone cannot detect stale allowed lists after an entity
    type changes. Keep this independent of augmentation's train-only validator
    because partial rows may be tune data or already chat-rendered.
    """
    entities, pairs, spans = (row.get(key) for key in ("entities", "pairs", "spans"))
    if any(not isinstance(value, list) or not value for value in (entities, pairs, spans)):
        raise ValueError("Nonempty entity/pair/span tables required")
    by_id, local_ids = {}, set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Invalid entity table")
        eid, local = entity.get("id"), entity.get("entity_id")
        if not isinstance(eid, str) or not eid or eid in by_id:
            raise ValueError("Duplicate or invalid upstream entity ID")
        if (not isinstance(local, str) or not re.fullmatch(r"E\d{3,}", local)
                or local in local_ids):
            raise ValueError("Duplicate or invalid local entity ID")
        if not isinstance(entity.get("name"), str) or not entity["name"].strip():
            raise ValueError("Empty entity name")
        label = entity.get("label")
        if not isinstance(label, str) or label not in SCHEMA_LABELS or normalize_label(label) != label:
            raise ValueError("Noncanonical entity type")
        by_id[eid] = entity
        local_ids.add(local)
    endpoints, directed, pids = set(), set(), set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Invalid pair table")
        source, target, pid = (pair.get(key) for key in ("source", "target", "pair_id"))
        if (not isinstance(source, str) or not isinstance(target, str)
                or source not in by_id or target not in by_id or source == target):
            raise ValueError("Invalid directed pair endpoints")
        if (source, target) in directed:
            raise ValueError("Duplicate directed pair")
        if not isinstance(pid, str) or not re.fullmatch(r"P\d{4,}", pid) or pid in pids:
            raise ValueError("Duplicate or invalid pair ID")
        expected = list(allowed_relations(by_id[source]["label"], by_id[target]["label"]))
        if not expected or pair.get("allowed") != expected:
            raise ValueError("Stale allowed_relations for entity types")
        endpoints.update((source, target))
        directed.add((source, target))
        pids.add(pid)
    if endpoints != set(by_id):
        raise ValueError("Unused or missing prompt entities")
    span_ids = []
    for span in spans:
        if (not isinstance(span, dict) or not isinstance(span.get("span_id"), str)
                or not span["span_id"] or not isinstance(span.get("text"), str)):
            raise ValueError("Invalid source span")
        span_ids.append(span["span_id"])
    if len(set(span_ids)) != len(span_ids):
        raise ValueError("Duplicate source span ID")
    if "original_spans" in row:
        originals = row["original_spans"]
        if (not isinstance(originals, dict) or set(originals) != set(span_ids)
                or any(not isinstance(text, str) for text in originals.values())):
            raise ValueError("Original/source span mapping mismatch")


def validate_partial_row(row):
    from robustness.entity_noise import prompt_payload, protocol_prefix

    _validate_input_tables(row)
    if row.get("evidence_grounding_policy", LEGACY_EVIDENCE_POLICY) not in EVIDENCE_POLICIES:
        raise ValueError("Unsupported evidence grounding policy")
    # Validate the input too: SFT must not accept hidden gold/provenance columns
    # inserted into its JSON payload or prose. Only the shared deterministic
    # input projection, raw or exactly chat-rendered, is admissible.
    expected_prompt = protocol_prefix() + "\n" + dumps(prompt_payload(row)) + "\n"
    rendered_prompt = ("<|im_start|>user\n" + expected_prompt
                       + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    if row.get("prompt") not in (expected_prompt, rendered_prompt):
        raise ValueError("Partial SFT prompt contains unexpected fields, metadata, or template drift")
    if row.get("loss_mask_contract") != CONTRACT:
        raise ValueError("Unsupported or missing UNKNOWN loss-mask contract")
    statuses = row.get("pair_supervision")
    if not isinstance(statuses, dict) or not statuses:
        raise ValueError("Missing per-pair supervision")
    pids = [p["pair_id"] for p in row["pairs"]]
    if set(statuses) != set(pids) or len(set(pids)) != len(pids):
        raise ValueError("Pair supervision does not match the input table")
    if not any(value in KNOWN for value in statuses.values()):
        raise ValueError("An all-UNKNOWN row must not enter SFT")
    if any(value not in KNOWN and not value.startswith("UNKNOWN_") for value in statuses.values()):
        raise ValueError("Unknown supervision status")
    target = row["target"]
    if [d["pair_id"] for d in target["decisions"]] != pids:
        raise ValueError("Partial target order differs from input order")
    for decision in target["decisions"]:
        status = statuses[decision["pair_id"]]
        if (status == "SUPPORTED_SILVER") != bool(decision["relations"]):
            raise ValueError("UNKNOWN/negative placeholders cannot contain positive targets")
    expected, spans = serialize_partial(target, statuses)
    if row.get("completion") != expected or row.get("loss_mask_spans") != spans:
        raise ValueError("Missing, stale, or tampered UNKNOWN loss mask")
    _, errors = parse_decisions(expected, row)
    if errors:
        raise ValueError(f"Invalid partial target: {errors}")
    return True


def masked_token_labels(row, tokenizer):
    """Use real character offsets; tokens touching UNKNOWN get -100 entirely."""
    validate_partial_row(row)
    try:
        encoded = tokenizer(row["completion"], add_special_tokens=False,
                            truncation=False, return_offsets_mapping=True)
    except (TypeError, NotImplementedError) as exc:
        raise ValueError("Partial supervision requires a fast tokenizer with offsets") from exc
    ids, offsets = encoded["input_ids"], encoded.get("offset_mapping")
    if offsets is None or len(ids) != len(offsets):
        raise ValueError("Tokenizer did not supply valid completion offsets")
    ignored, labels = row["loss_mask_spans"], []
    for token, (start, end) in zip(ids, offsets):
        if not (0 <= start <= end <= len(row["completion"])):
            raise ValueError("Out-of-range token offsets")
        unknown = start == end or any(start < b and end > a for a, b in ignored)
        labels.append(-100 if unknown else token)
    if not any(label != -100 for label in labels):
        raise ValueError("No supervised completion tokens remain")
    return ids, labels


def inject_contract_negative(row, source_text):
    """A synthetic-noise auxiliary set, never a claim about clinical negatives."""
    from robustness.entity_noise import NoiseConfig, augment_row, prompt_payload, protocol_prefix

    result = augment_row(row, NoiseConfig(operation="inject", injected_pairs=2),
                         source_texts=(source_text,))
    statuses = deepcopy(row["pair_supervision"])
    for pid in result["entity_noise"]["incident_pair_ids"]:
        statuses[pid] = "SYNTHETIC_NEGATIVE"
    # Appending every manufactured negative at the end recreates a label-order
    # shortcut. Reorder all pairs by input names/types, not annotation status.
    entities = {entity["id"]: entity for entity in result["entities"]}
    def rank(pair):
        values = [(entities[pair[k]]["name"], entities[pair[k]]["label"])
                  for k in ("source", "target")]
        return hashlib.sha256(dumps(values).encode()).hexdigest()
    decisions = {d["pair_id"]: d for d in result["target"]["decisions"]}
    result["pairs"].sort(key=rank)
    remapping, new_statuses, ordered = {}, {}, []
    for index, pair in enumerate(result["pairs"]):
        old, new = pair["pair_id"], f"P{index + 1:04d}"
        remapping[old] = new
        pair["pair_id"] = new
        new_statuses[new] = statuses[old]
        ordered.append({**decisions[old], "pair_id": new})
    statuses = new_statuses
    result["target"] = {"decisions": ordered}
    result["unknown_reasons"] = {remapping[p]: r for p,r in row.get("unknown_reasons", {}).items()}
    result["entity_noise"]["incident_pair_ids"] = [remapping[p] for p in result["entity_noise"]["incident_pair_ids"]]
    result["entity_noise"]["pair_order"] = "input_name_type_hash_not_label"
    result["prompt"] = protocol_prefix() + "\n" + dumps(prompt_payload(result)) + "\n"
    result["pair_supervision"] = statuses
    result["completion"], result["loss_mask_spans"] = serialize_partial(result["target"], statuses)
    result["loss_mask_contract"] = CONTRACT
    result["eligible_for_rl"] = False
    validate_partial_row(result)
    return result

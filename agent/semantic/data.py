"""Source-grounded, non-exhaustive medical silver SFT compilation.

The original records and frozen source-group split remain immutable.  EOS ends
the supplied annotation sequence; it does not label missing relations negative.
No empty silver targets, NONE labels or preference negatives are manufactured.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from generative.chat import render_prompt

DATA_VERSION = "source-grounded-semantic-silver-v1"
VARIANTS = ("full_source", "short_source", "short_canonical")
SUPERVISION = "teacher_silver_positive_annotations_not_exhaustive"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    raw = value if isinstance(value, bytes) else (value if isinstance(value, str) else canonical(value)).encode()
    return hashlib.sha256(raw).hexdigest()


def _norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _user(record):
    messages = [m["content"] for m in record["messages"] if m["role"] == "user"]
    if len(messages) != 1 or not isinstance(messages[0], str):
        raise ValueError("Expected one immutable user source")
    return messages[0]


def build_title_context(structured):
    """Receipt for an exact title in the original same-group structured input.

    The routed title only proposes a string. Admission requires an exact
    match in the ORIGINAL user source, never the assistant graph or gold name.
    Offsets are half-open Python character offsets in that original user text.
    """
    if structured["source_type"] != "structured_icd":
        raise ValueError("Title context requires a structured source record")
    original = structured["original_record"]
    if sha(original) != structured["original_record_sha256"]:
        raise ValueError("Original source record SHA drift")
    title = structured["source_view"]["routed"]["metadata"].get("title")
    raw = _user(original)
    if not isinstance(title, str) or not title or title not in raw:
        raise ValueError("Title must occur exactly in original source")
    view = structured["source_view"]
    payload = view.get("raw_payload", raw)
    payload_start = view.get("payload_start", 0)
    if title not in payload or raw[payload_start:payload_start + len(payload)] != payload:
        raise ValueError("Title source payload does not reconstruct original user")
    start = payload_start + payload.index(title)
    return {"text": title, "source_group_id": str(structured["group_id"]),
            "source_record_sha256": sha(original), "source_start": start,
            "source_end": start + len(title)}


def _protocol():
    from semantic import protocol
    return protocol


def source_windows(text, max_chars=2400, overlap=400):
    """Source-only contiguous windows; no target/gold chooses their boundaries."""
    if type(max_chars) is not int or max_chars < 32 or not 0 <= overlap < max_chars:
        raise ValueError("Invalid source window size/overlap")
    if not text:
        return [(0, 0)]
    windows, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # Prefer a sentence/paragraph end in the last third; otherwise use
            # the fixed character boundary. The input bytes are never changed.
            matches = list(re.finditer(r"(?:[.!?。！？]\s+|\n+)", text[start + max_chars * 2 // 3:end]))
            if matches:
                end = start + max_chars * 2 // 3 + matches[-1].end()
        windows.append((start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return windows


def _metadata(item, title_context):
    if sha(item["original_record"]) != item["original_record_sha256"]:
        raise ValueError("Original source record SHA drift")
    if title_context and str(title_context["source_group_id"]) != str(item["group_id"]):
        raise ValueError("Cross-group title context forbidden")
    view = item["source_view"]
    text = item["source"]["original_text"]
    if view["raw_payload"] != text:
        raise ValueError("Training text differs from immutable source payload")
    effective_user = view.get("raw_user", _user(item["original_record"]))
    a, b = view.get("payload_start", 0), view.get("payload_end", len(effective_user))
    if effective_user[a:b] != text:
        raise ValueError("Source payload coordinates do not recover exact text")
    if item.get("source_restoration"):
        restoration = item["source_restoration"]
        if (restoration["effective_user"] != effective_user or restoration["effective_user_sha256"] != sha(effective_user)
                or restoration["original_user_sha256"] != sha(_user(item["original_record"]))):
            raise ValueError("Verified source restoration receipt mismatch")
    elif effective_user != _user(item["original_record"]):
        raise ValueError("Unrestored effective input differs from original user")
    return {"source_group_id": str(item["group_id"]),
            "source_record_sha256": item["original_record_sha256"],
            "source_index": item["source_index"], "source_type": item["source_type"],
            "source_view_origin": item["source_view_origin"],
            "source_payload_start": a, "source_payload_end": b,
            "effective_user_sha256": sha(effective_user), "source_text_sha256": sha(text),
            "supervision": SUPERVISION, "human_adjudicated": False,
            "unannotated_pair_policy": "UNKNOWN_NOT_EXPLICITLY_SUPERVISED",
            "eos_semantics": "end_of_supplied_annotations_not_exhaustive_negative_claim",
            **({"title_context": deepcopy(title_context)} if title_context else {})}


def _targets(item, metadata):
    """Admit only source-supported entities and unblocked located positive edges."""
    p = _protocol()
    text, entities, audit = item["source"]["original_text"], [], Counter()
    title = (metadata.get("title_context") or {}).get("text", "")
    aliases, canonical_ids = {}, {}
    for entity in item["entities"]:
        name = entity.get("canonical_name") or ""
        if not name or not any(_norm(name) in _norm(source) for source in (text, title) if source):
            audit["entity_canonical_name_not_source_supported"] += 1
            continue
        locations = [m for m in entity["mentions"]
                     if text[m["original_start"]:m["original_end"]] == m["original_text"]]
        if locations:
            loc = min(locations, key=lambda m: (m["original_start"], m["original_end"]))
            mention, start, end, context = loc["original_text"], loc["original_start"], loc["original_end"], "body"
        elif _norm(name) == _norm(title):
            mention, start, end, context = title, -1, -1, "title_context"
            audit["entity_grounded_by_trusted_title"] += 1
        else:
            audit["entity_exact_mention_unavailable"] += 1
            continue
        key = entity["label"], _norm(name)
        if key in canonical_ids:
            aliases[str(entity["id"])] = canonical_ids[key]
            audit["entity_duplicate_type_name_merged"] += 1
            continue
        candidate = {"id": str(entity["id"]), "label": entity["label"], "name": name,
                     "mention": mention, "_start": start, "_end": end, "_context": context,
                     "_atom": f"entity:{entity['id']}"}
        check = p.parse_entities(text, p.render_entities([candidate]), metadata=metadata)
        supported = check.get("filtered_entities", check.get("validated_entities", check.get("entities", [])))
        if not check.get("valid") or len(supported) != 1:
            audit["entity_protocol_rejected"] += 1
            continue
        candidate["mentions"] = deepcopy(supported[0]["mentions"])
        candidate["grounding_verified"] = True
        canonical_ids[key] = candidate["id"]
        aliases[candidate["id"]] = candidate["id"]
        entities.append(candidate)
    admitted = {e["id"] for e in entities}
    relations = []
    for ann in item["annotations"]:
        edge = ann["annotation"]
        if ann.get("pair_blocked"):
            audit["relation_pair_blocked"] += 1
            continue
        source_id, target_id = aliases.get(str(edge["source"])), aliases.get(str(edge["target"]))
        if source_id not in admitted or target_id not in admitted:
            audit["relation_endpoint_unavailable"] += 1
            continue
        if not ann["all_evidence_located"]:
            audit["relation_evidence_unlocated"] += 1
            continue
        for fragment_index, hits in enumerate(ann["evidence_fragments"]):
            loc = min(hits, key=lambda m: (m["original_start"], m["original_end"]))
            start, end = loc["original_start"], loc["original_end"]
            evidence = text[start:end]
            if evidence != loc["original_text"] or not evidence:
                raise ValueError("Evidence source offsets drifted")
            candidate = {"source": source_id, "target": target_id,
                         "relation": edge["relation"], "evidence": evidence,
                         "_start": start, "_end": end,
                         "_atom": f"relation:{ann['index']}:evidence:{fragment_index}"}
            check = p.parse_relations(text, entities, p.render_relations([candidate], entities), metadata=metadata)
            supported = check.get("filtered_relations", check.get("relations", []))
            if not check.get("valid") or len(supported) != 1:
                audit["relation_protocol_rejected"] += 1
                for error in check.get("errors", []):
                    audit["relation_rejection:" + error["reason"]] += 1
                for rejection in check.get("rejected_relations", []):
                    for reason in rejection["reasons"]:
                        audit["relation_rejection:" + reason["reason"]] += 1
                continue
            relations.append(candidate)
    return entities, relations, audit


def _clean(target):
    return {key: deepcopy(value) for key, value in target.items() if not key.startswith("_")}


def compile_source(item, title_context=None, variant="full_source", short_chars=2400):
    """Compile one frozen source to entity/relation rows and a no-text audit.

    Each admitted annotation atom appears exactly once per variant. Short
    relation prompts retain the complete supplied entity table, matching a
    full-document entity pass followed by windowed relation extraction.
    Oversized/cross-boundary targets receive an explicit full-source fallback.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown source variant")
    metadata = _metadata(item, title_context)
    audit = Counter({"source_records_processed": 1})
    if item["source_type"] != "free_text":
        audit["structured_separate_deterministic"] += 1
        return [], dict(audit)
    if item["record_blocked"]:
        audit["quarantined_record_no_training_supervision"] += 1
        return [], dict(audit)
    entities, relations, counts = _targets(item, metadata)
    audit.update(counts)
    text, p = item["source"]["original_text"], _protocol()
    windows = [(0, len(text))] if variant == "full_source" else source_windows(text, short_chars, min(400, short_chars // 4))
    order = "canonical" if variant == "short_canonical" else "source"
    rows = []
    entity_by_id = {e["id"]: e for e in entities}
    for stage, targets in (("entities", entities), ("relations", relations)):
        assigned = defaultdict(list)
        for target in targets:
            eligible = [i for i, (a, b) in enumerate(windows)
                        if target.get("_context") == "title_context" or a <= target["_start"] and target["_end"] <= b]
            # Relation endpoints belong to the verified complete-source entity
            # table. Only evidence must fit this local relation window, exactly
            # matching runtime after the complete-source entity stage.
            assigned[eligible[0] if eligible else -1].append(target)
        for window_index, selected in sorted(assigned.items()):
            a, b = windows[window_index] if window_index >= 0 else (0, len(text))
            source = text[a:b]
            if order == "canonical":
                selected.sort(key=lambda t: ((t["label"], _norm(t["name"])) if stage == "entities" else
                    (entity_by_id[t["source"]]["label"], _norm(entity_by_id[t["source"]]["name"]),
                     entity_by_id[t["target"]]["label"], _norm(entity_by_id[t["target"]]["name"]), t["relation"], t["_start"])))
            else:
                selected.sort(key=lambda t: ((t["_start"], t["_end"], t["_atom"]) if stage == "entities" else
                    (entity_by_id[t["source"]]["_start"], entity_by_id[t["target"]]["_start"], t["relation"], t["_start"])))
            local_metadata = {**deepcopy(metadata), "variant": variant, "order": order,
                              "source_window": {"start": a, "end": b, "index": window_index,
                                                "full_source_fallback": window_index == -1},
                              "relation_entity_table_scope": "complete_source_entities" if stage == "relations" else None,
                              "supervised_atom_ids": [t["_atom"] for t in selected]}
            clean = [_clean(t) for t in selected]
            table = [_clean(e) for e in entities]
            prompt = (p.build_entity_prompt(source, metadata=local_metadata, order=order) if stage == "entities"
                      else p.build_relation_prompt(source, table, metadata=local_metadata, order=order))
            completion = p.render_entities(clean) if stage == "entities" else p.render_relations(clean, table)
            if not completion.strip():
                raise ValueError("Ordinary silver cannot supervise an empty annotation sequence")
            task_input = {"text": source, "entities": table if stage == "relations" else [], "metadata": local_metadata}
            row = {"data_version": DATA_VERSION,
                   "id": f"{item['split']}:{item['source_index']}:{variant}:{stage}:{window_index}",
                   "split": item["split"], "stage": stage, "variant": variant,
                   "group_id": str(item["group_id"]), "metadata": local_metadata,
                   "task_input": task_input, "prompt": prompt, "completion": completion,
                   "target": clean, "target_count": len(clean),
                   "input_sha256": sha(task_input), "target_sha256": sha(clean)}
            validate_row(row, require_train=False)
            rows.append(row)
            audit[stage + "_rows"] += 1
            audit[stage + "_atoms"] += len(selected)
            if window_index == -1:
                audit[stage + "_fallback_atoms"] += len(selected)
        if not targets:
            audit[stage + "_no_admitted_positive_targets_no_empty_supervision"] += 1
    return rows, dict(audit)


def validate_row(row, *, require_train=True):
    p = _protocol()
    if row.get("data_version") != DATA_VERSION or row.get("stage") not in ("entities", "relations"):
        raise ValueError("Unsupported semantic silver data contract")
    if require_train and row.get("split") != "train":
        raise ValueError("Held-out/tune rows cannot enter training")
    stage, inputs = row["stage"], row["task_input"]
    metadata = inputs["metadata"]
    if row["metadata"] != metadata or metadata["supervision"] != SUPERVISION:
        raise ValueError("Supervision metadata drift")
    if row["group_id"] != metadata["source_group_id"] or row["input_sha256"] != sha(inputs):
        raise ValueError("Input identity drift")
    if row["target_sha256"] != sha(row["target"]) or row["target_count"] != len(row["target"]):
        raise ValueError("Target identity drift")
    if not row["target"] or not row["completion"].strip():
        raise ValueError("Incomplete silver does not admit empty supervision")
    prompt = (p.build_entity_prompt(inputs["text"], metadata=metadata, order=metadata["order"]) if stage == "entities"
              else p.build_relation_prompt(inputs["text"], inputs["entities"], metadata=metadata, order=metadata["order"]))
    target = (p.render_entities(row["target"]) if stage == "entities"
              else p.render_relations(row["target"], inputs["entities"]))
    if row["prompt"] != prompt or row["completion"] != target:
        raise ValueError("Prompt or completion differs from protocol reconstruction")
    parsed = (p.parse_entities(inputs["text"], target, metadata=metadata) if stage == "entities"
              else p.parse_relations(inputs["text"], inputs["entities"], target, metadata=metadata))
    filtered = parsed.get("filtered_" + stage, parsed.get("validated_" + stage, parsed.get(stage, [])))
    if not parsed.get("valid") or not filtered:
        raise ValueError("Compiled completion fails grounded protocol validation")
    if stage == "entities":
        expected = {(e["label"], _norm(e["name"])) for e in row["target"]}
        actual = {(e["label"], _norm(e["name"])) for e in filtered}
    else:
        expected = {(e["source"], e["target"], e["relation"], e["evidence"]) for e in row["target"]}
        actual = {(e["source"], e["target"], e["relation"], evidence) for e in filtered for evidence in e["evidence_texts"]}
    if expected != actual:
        raise ValueError("Grounded protocol did not recover every supervised semantic target")
    return parsed


def encode_row(row, tokenizer, max_length, require_train=True):
    """One full-text completion encoding + EOS; no target/prompt truncation."""
    validate_row(row, require_train=require_train)
    prompt_ids = list(tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False))
    completion_ids = list(tokenizer.encode(row["completion"], add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0 or eos in completion_ids or not completion_ids:
        raise ValueError("Invalid EOS/completion tokenization")
    completion_ids.append(eos)
    if len(prompt_ids) + len(completion_ids) > max_length:
        raise ValueError(f"Untruncated semantic row exceeds max_length: {len(prompt_ids)} + {len(completion_ids)} > {max_length}")
    return {"input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
            "_choice_counts": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
                               "action_count": row["target_count"], "stage": row["stage"],
                               "action_count_unit": "semantic_annotation_records",
                               "completion_encoding": "literal_semantic_lines_then_eos"}}


def compile_eval_record(item, title_context=None):
    from evaluate import _graph
    original = item["original_record"]
    assistant = [m["content"] for m in original["messages"] if m["role"] == "assistant"]
    gold = _graph(assistant[0])
    if gold is None:
        raise ValueError("Invalid original gold")
    metadata = _metadata(item, title_context)
    text = item["source"]["original_text"]
    inputs = {"text": text, "metadata": metadata}
    return {"id": f"{item['split']}:{item['source_index']}", "index": item["source_index"],
            "split": item["split"], "source_group_id": str(item["group_id"]),
            "source_type": item["source_type"], "text": text, "metadata": metadata,
            "gold": deepcopy(gold), "original_record": deepcopy(original),
            "original_record_sha256": sha(original),
            "input_identity": {"version": DATA_VERSION + "-trusted-title-input-v1",
                               "sha256": sha(inputs), "source_text_sha256": sha(text),
                               "source_view_origin": item["source_view_origin"]},
            "source_restoration": deepcopy(item.get("source_restoration"))}


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")


def _quantiles(values):
    values = sorted(values)
    return {str(q): values[int((len(values) - 1) * q)] for q in (0, .5, .9, .95, 1)} if values else {}


def build_release(source_root, output_dir, tokenizer=None, max_length=8192, short_chars=2400):
    source_root, output_dir = Path(source_root), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Refusing to overwrite an existing release")
    manifest = json.loads((source_root / "manifest.json").read_text())
    for filename in ("records.jsonl", "frozen_split.jsonl"):
        if sha((source_root / filename).read_bytes()) != manifest["output_sha256"][filename]:
            raise ValueError("Frozen source artifact SHA mismatch: " + filename)
    items = [json.loads(line) for line in (source_root / "records.jsonl").read_text().splitlines()]
    split = [json.loads(line) for line in (source_root / "frozen_split.jsonl").read_text().splitlines()]
    assignments = {r["source_index"]: (r["group_id"], r["split"]) for r in split}
    if len(assignments) != len(items) or len({r["source_index"] for r in items}) != len(items):
        raise ValueError("Frozen source identity is not one-to-one")
    groups = defaultdict(set)
    titles, title_sources = {}, {}
    for item in items:
        if assignments[item["source_index"]] != (item["group_id"], item["split"]):
            raise ValueError("Frozen group/split identity drift")
        groups[item["split"]].add(str(item["group_id"]))
        if item["source_type"] == "structured_icd":
            group = str(item["group_id"])
            if group in titles:
                raise ValueError("Ambiguous same-group structured title")
            titles[group] = build_title_context(item)
            title_sources[item["original_record_sha256"]] = {
                "source_group_id": group, "original_record": item["original_record"]}
    if groups["train"] & groups["tune"]:
        raise ValueError("Train/tune source-group leakage")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, stats, audits, counters = {}, {}, [], {}
    for variant in VARIANTS:
        stage_rows = defaultdict(list)
        atoms = Counter()
        counts = Counter()
        for item in items:
            rows, audit = compile_source(item, titles.get(str(item["group_id"])), variant, short_chars)
            counts.update(audit)
            audits.append({"variant": variant, "split": item["split"], "source_index": item["source_index"],
                           "source_group_id": str(item["group_id"]), **audit})
            for row in rows:
                stage_rows[(row["split"], row["stage"])].append(row)
                atoms.update((row["split"], row["metadata"]["source_index"], atom) for atom in row["metadata"]["supervised_atom_ids"])
        counters[variant] = atoms
        stats[variant] = {"audit_counts": dict(counts), "files": {}}
        for (split_name, stage), rows in sorted(stage_rows.items()):
            relative = f"{variant}/{split_name}_{stage}.jsonl"
            _write_jsonl(output_dir / relative, rows)
            outputs[relative] = sha((output_dir / relative).read_bytes())
            lengths, over = [], []
            if tokenizer is not None:
                for row in rows:
                    encoded = encode_row(row, tokenizer, 10**9, require_train=False)
                    length = len(encoded["input_ids"])
                    lengths.append(length)
                    if length > max_length:
                        over.append({"id": row["id"], "tokens": length})
            stats[variant]["files"][relative] = {"rows": len(rows), "target_atoms": sum(r["target_count"] for r in rows),
                "token_length_quantiles": _quantiles(lengths), "total_tokens": sum(lengths),
                "over_max_length": over, "trainable_without_truncation": not over}
    if not all(counters[v] == counters["full_source"] for v in VARIANTS):
        raise ValueError("Variant annotation supervision Counter mismatch")
    for split_name in ("train", "tune"):
        records = [compile_eval_record(item, titles.get(str(item["group_id"]))) for item in items if item["split"] == split_name]
        relative = f"evaluation/{split_name}_records.jsonl"
        _write_jsonl(output_dir / relative, records)
        outputs[relative] = sha((output_dir / relative).read_bytes())
    _write_jsonl(output_dir / "source_audit.jsonl", audits)
    (output_dir / "title_source_receipts.json").write_text(canonical(title_sources) + "\n")
    for relative in ("source_audit.jsonl", "title_source_receipts.json"):
        outputs[relative] = sha((output_dir / relative).read_bytes())
    release = {"data_version": DATA_VERSION, "source_artifacts": {k: manifest["output_sha256"][k] for k in ("records.jsonl", "frozen_split.jsonl")},
               "record_counts": dict(Counter(f"{r['split']}:{r['source_type']}" for r in items)),
               "group_counts": {k: len(v) for k, v in groups.items()}, "train_tune_group_overlap": 0,
               "title_contexts": len(titles), "variants": stats, "output_sha256": outputs,
               "supervision_counter_equal_across_variants": True,
               "supervision": SUPERVISION, "ordinary_silver_empty_targets": 0,
               "explicit_negative_examples": 0, "structured_model_training_rows": 0,
               "eos_semantics": "annotation_sequence_termination_not_exhaustive_negative_claim",
               "gold_unchanged": True, "max_length_checked": max_length, "short_chars": short_chars,
               "all_train_sources_processed": True, "historical_test_training_allowed": False,
               "protocol_sha256": sha(Path(_protocol().__file__).read_bytes()),
               "compiler_sha256": sha(Path(__file__).read_bytes())}
    (output_dir / "manifest.json").write_text(canonical(release) + "\n")
    return release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--short-chars", type=int, default=2400)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = build_release(args.source_root, args.output_dir, tokenizer, args.max_length, args.short_chars)
    print(canonical({k: result[k] for k in ("data_version", "record_counts", "group_counts", "variants", "supervision_counter_equal_across_variants")}))


if __name__ == "__main__":
    main()

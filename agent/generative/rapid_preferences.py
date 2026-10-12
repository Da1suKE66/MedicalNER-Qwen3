"""Finite, fully artificial DPO engineering screen and exact initial-loader control.

This module never imports medical records or infers a negative from missing silver
annotations. Its complete artificial facts establish a mechanical extraction
oracle. Success here establishes runtime/contract behavior, not medical quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from .protocol import (
    ProtocolConfig, build_batches, digest, dumps, parse_decisions,
    reward_components, supervision,
)
from .train import (
    adapter_fingerprint, initialize_dpo_reference, length_audit, prompt_batch,
    read_rows, require_saved_weight_equality, validate_training_rows,
)


VERSION = "rapid-artificial-preferences-v1"
KINDS = (
    "omit_supported_edge", "omit_second_relation", "add_unsupported_edge",
    "contradict_explicit_negation", "wrong_target", "reversed_subtype",
    "wrong_label", "wrong_evidence",
)
CORE = "is_core_symptom_of"
DIAGNOSTIC = "supports_diagnosis_of"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_output(path):
    out = Path(path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Output must be new/empty; refuse to overwrite an earlier run")
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_case(index, kind, *, partition="train"):
    """Build one complete artificial world, then change exactly one fact.

    Each partition has disjoint artificial names. Evaluation still shares
    templates with training and must not be called medical generalization.
    """
    if kind not in KINDS or partition not in {"train", "mechanical_holdout"}:
        raise ValueError("Unsupported artificial case kind/partition")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Case index must be a nonnegative integer")
    key = digest(f"{VERSION}:{partition}:{index}:{kind}")[:12]
    s, a, b = f"Feature {key}", f"Condition {key}A", f"Condition {key}B"
    nodes = [
        {"id": "S", "name": s, "label": "Symptom"},
        {"id": "A", "name": a, "label": "Disease"},
        {"id": "B", "name": b, "label": "Disease"},
    ]
    # These facts are authored, exhaustive within the exercise, and intentionally
    # artificial. No missing annotation in a clinical document is consulted.
    intro = "This is a fully artificial extraction exercise with an exhaustive list of asserted relations. "
    if kind == "reversed_subtype":
        fact = f"{a} is a subtype of {b}; {b} is not a subtype of {a}."
        extra = f"{s} has no relation to either condition."
        facts = [("A", "B", "subtype_of", fact)]
    elif kind == "contradict_explicit_negation":
        fact = f"{s} is explicitly not a core symptom of {a}."
        extra = f"{b} is unrelated to both {s} and {a}; no positive relations are asserted."
        facts = []
    else:
        fact = f"{s} is a core symptom of {a}."
        facts = [("S", "A", CORE, fact)]
        if kind == "omit_second_relation":
            second = f"{s} independently supports diagnosis of {a}."
            facts.append(("S", "A", DIAGNOSTIC, second))
            extra = f"{second} {b} is unrelated to both {s} and {a}."
        else:
            extra = f"{b} is unrelated to both {s} and {a}; there is no separately asserted diagnostic-support relation."
    text = intro + fact + " " + extra + " No other relations between the listed entities are asserted."
    graph = {
        "entities": nodes,
        "relations": [
            {"source": src, "target": dst, "relation": label, "evidence": evidence}
            for src, dst, label, evidence in facts
        ],
    }
    config = ProtocolConfig(strict_evidence=True, pairs_per_batch=16, order="source")
    batches, doc, _ = build_batches(text, nodes, config, sample_id=f"{partition}:{key}")
    if len(batches) != 1:
        raise ValueError("Artificial exercise must be one complete candidate batch")
    batch = batches[0]
    chosen, _ = supervision(batch, graph, doc, config)
    if chosen is None:
        raise ValueError("Artificial oracle has unavailable evidence")
    pair_id = {(p["source"], p["target"]): p["pair_id"] for p in batch["pairs"]}
    rejected = copy.deepcopy(chosen)
    decisions = {d["pair_id"]: d["relations"] for d in rejected["decisions"]}

    def relations(src, dst):
        return decisions[pair_id[src, dst]]

    def add(src, dst, label, evidence=None):
        relations(src, dst).append({
            "relation": label,
            "evidence_span_ids": evidence or [batch["spans"][0]["span_id"]],
        })

    if kind == "omit_supported_edge":
        relations("S", "A").clear()
    elif kind == "omit_second_relation":
        relations("S", "A")[:] = [r for r in relations("S", "A") if r["relation"] != DIAGNOSTIC]
    elif kind == "add_unsupported_edge":
        add("S", "B", CORE)
    elif kind == "contradict_explicit_negation":
        add("S", "A", CORE)
    elif kind == "wrong_target":
        edge = relations("S", "A").pop()
        relations("S", "B").append(edge)
    elif kind == "reversed_subtype":
        edge = relations("A", "B").pop()
        relations("B", "A").append(edge)
    elif kind == "wrong_label":
        relations("S", "A")[0]["relation"] = DIAGNOSTIC
    else:
        edge = relations("S", "A")[0]
        unused = [sp["span_id"] for sp in batch["spans"] if sp["span_id"] not in edge["evidence_span_ids"]]
        if not unused:
            raise ValueError("Wrong-evidence control needs a distinct available span")
        edge["evidence_span_ids"] = [unused[-1]]
    row = {
        "id": f"{VERSION}:{partition}:{key}",
        "group_id": f"artificial:{partition}:{key}",
        "split": "train" if partition == "train" else "mechanical_holdout",
        "prompt": batch["prompt"], "chosen": dumps(chosen), "rejected": dumps(rejected),
        "reason": kind, "human_adjudicated": False,
        "provenance": "fully_artificial_exhaustive_assertions_no_medical_source",
        "eligible_for_rl": True, "semantic_claim": "mechanical_extraction_only",
        "oracle": graph, "oracle_source_text": text,
    }
    row.update({f"{field}_sha256": digest(row[field]) for field in ("prompt", "chosen", "rejected")})
    verify_case(row)
    record = {"messages": [
        {"role": "user", "content": text},
        {"role": "assistant", "content": dumps(graph)},
    ]}
    return row, record


def verify_case(row):
    """Verify the emitted protocol and directed evidence-aware preference."""
    if row.get("provenance") != "fully_artificial_exhaustive_assertions_no_medical_source":
        raise ValueError("Only the fully artificial oracle is eligible")
    if row.get("human_adjudicated") is not False or row.get("semantic_claim") != "mechanical_extraction_only":
        raise ValueError("Do not relabel artificial cases as medically validated")
    config = ProtocolConfig(strict_evidence=True, pairs_per_batch=16, order="source")
    oracle = row["oracle"]
    batches, doc, _ = build_batches(row["oracle_source_text"], oracle["entities"], config)
    if len(batches) != 1 or batches[0]["prompt"] != row["prompt"]:
        raise ValueError("Artificial prompt differs from the complete source oracle")
    target, _ = supervision(batches[0], oracle, doc, config)
    if target is None or dumps(target) != row["chosen"]:
        raise ValueError("Chosen target differs from the authored exhaustive oracle")
    # Keep held-out tags intact. Callers cannot train these without the existing
    # training admission gate explicitly rejecting the non-train partition.
    batch = prompt_batch(row)
    chosen, errors = parse_decisions(row["chosen"], batch)
    rejected, bad = parse_decisions(row["rejected"], batch)
    if errors or bad or chosen == rejected:
        raise ValueError("Invalid/identical artificial preference")
    for field in ("prompt", "chosen", "rejected"):
        if row[f"{field}_sha256"] != digest(row[field]):
            raise ValueError("Artificial preference hash mismatch")
    a = reward_components(row["chosen"], batch, target)["reward"]
    b = reward_components(row["rejected"], batch, target)["reward"]
    if not a > b:
        raise ValueError("Artificial preference/reward direction mismatch")
    return {"chosen_edges": len(chosen), "rejected_edges": len(rejected), "reward_margin": a - b}


def summarize(rows, tokenizer=None):
    edges, chars, tokens, reasons = Counter(), Counter(), Counter(), Counter()
    prompt_sizes, completion_sizes = [], []
    for row in rows:
        detail = verify_case(row)
        delta = detail["chosen_edges"] - detail["rejected_edges"]
        edges["chosen_more" if delta > 0 else "chosen_fewer" if delta < 0 else "equal"] += 1
        delta = len(row["chosen"]) - len(row["rejected"])
        chars["chosen_longer" if delta > 0 else "chosen_shorter" if delta < 0 else "equal"] += 1
        reasons[row["reason"]] += 1
        if tokenizer is not None:
            from .chat import render_prompt
            prompt_sizes.append(len(tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False)))
            a, b = [len(tokenizer.encode(row[k], add_special_tokens=False)) + 1 for k in ("chosen", "rejected")]
            completion_sizes.extend([a, b])
            tokens["chosen_longer" if a > b else "chosen_shorter" if a < b else "equal"] += 1
    result = {
        "rows": len(rows), "unique_prompts": len({r["prompt_sha256"] for r in rows}),
        "reasons": dict(reasons), "edge_count_direction": dict(edges),
        "character_length_direction": dict(chars),
        "token_length_direction": dict(tokens) if tokenizer is not None else None,
        "human_adjudicated_rows": 0, "medical_source_records": 0,
    }
    if tokenizer is not None:
        rendered = [{**r, "prompt": render_prompt(tokenizer, r["prompt"])} for r in rows]
        result["length_audit"] = length_audit(rendered, tokenizer, 2048, "dpo")
    return result


def build(output_dir, *, repeats=8, holdout_repeats=2, tokenizer=None):
    if repeats < 1 or holdout_repeats < 1:
        raise ValueError("Positive train and held-out repeat counts are required")
    # Construct/validate first; admission errors must not create a release.
    train = [make_case(i, kind)[0] for i in range(repeats) for kind in KINDS]
    held = [make_case(i, kind, partition="mechanical_holdout") for i in range(holdout_repeats) for kind in KINDS]
    validate_training_rows(train, "dpo")
    holdout = [r for r, _ in held]
    if {r["prompt_sha256"] for r in train} & {r["prompt_sha256"] for r in holdout}:
        raise ValueError("Train/mechanical-holdout prompt overlap")
    summary = summarize(train, tokenizer)
    if any(not summary["edge_count_direction"].get(k) for k in ("chosen_more", "chosen_fewer", "equal")):
        raise ValueError("Preference release is one-directional in edge count")
    if any(not summary["character_length_direction"].get(k) for k in ("chosen_longer", "chosen_shorter", "equal")):
        raise ValueError("Preference release is one-directional in character length")
    out = _new_output(output_dir)
    for name, rows in (("train", train), ("mechanical_holdout_preferences", holdout)):
        (out / f"{name}.jsonl").write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "mechanical_holdout_records.json").write_text(dumps([record for _, record in held]) + "\n", encoding="utf-8")
    (out / "mechanical_matrix.json").write_text(dumps({
        "version": VERSION,
        "defaults": asdict(ProtocolConfig(strict_evidence=True, pairs_per_batch=16, order="source")),
        "variants": {"mechanical": {}},
    }) + "\n", encoding="utf-8")
    manifest = {
        "version": VERSION, "train": summary, "mechanical_holdout": summarize(holdout, tokenizer),
        "files": {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())},
        "limitations": [
            "Entire corpus is artificial; no medical facts or silver absences are used as negative labels.",
            "Mechanical holdout has unseen artificial names but shares templates and cannot measure medical generalization.",
            "Preference directions cover more/fewer/equal edges and longer/shorter/equal text; token directions are reported when tokenizer is supplied.",
            "A 16-step DPO screen establishes engineering feasibility only; assess medical tune separately against the same SFT starting checkpoint.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def export_initial(base_model, sft_adapter, data, output_dir):
    """Export zero optimizer steps through exactly the DPO policy/ref loader.

    The independent evaluator reload must reproduce the initial policy weights
    and first-prompt last-token logits. No Trainer, optimizer, or training call
    is made. Compare the later DPO manifest's initial fingerprint with this one.
    """
    from .chat import render_prompt
    from .evaluate import load_evaluation_model
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_rows(data)
    validate_training_rows(rows, "dpo")
    for row in rows:
        verify_case(row)
    if importlib.metadata.version("peft") != "0.17.1" or importlib.metadata.version("transformers") != "4.56.2":
        raise RuntimeError("Use the pinned medicalner runtime: PEFT 0.17.1, Transformers 4.56.2")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("An authorized idle BF16 GPU is required")
    out = _new_output(output_dir)
    manifest = {
        "status": "initializing", "global_step": 0, "optimizer_created": False,
        "base_model": str(base_model), "sft_adapter": str(sft_adapter), "data_sha256": _sha(data),
        "scope": "DPO initial policy/reference loader and evaluator reload; no optimization",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            model, sft_adapter, adapter_name="policy", is_trainable=True, local_files_only=True,
        )
        manifest["dpo_initialization"] = initialize_dpo_reference(model, sft_adapter)
        fingerprint = adapter_fingerprint(model, "policy")
        model.set_adapter("policy")
        model.to("cuda").eval()
        inputs = tokenizer(render_prompt(tokenizer, rows[0]["prompt"]), add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            expected = model(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        model.save_pretrained(out / "final", selected_adapters=["policy"])
        tokenizer.save_pretrained(out / "final")
        saved = out / "final" / "policy"
        manifest.update(initial_policy_sha256=fingerprint, final_adapter=str(saved))
        model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = load_evaluation_model(base_model, str(saved))
        manifest["saved_adapter_sha256_verified"] = require_saved_weight_equality(reloaded, fingerprint)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            actual = reloaded(**inputs, use_cache=False).logits[:, -1, :].float().cpu()
        delta = float((expected - actual).abs().max())
        manifest["save_reload_max_logit_difference"] = delta
        if not math.isfinite(delta) or delta > 0.02:
            raise RuntimeError(f"Initial policy/evaluator logits differ: {delta}")
        manifest["status"] = "zero_step_initial_policy_exported_verified"
        manifest["verification_scope"] = "Exact LoRA fingerprint and first prompt last-token plain-inference logits; not a universal numerical-equivalence proof"
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--holdout-repeats", type=int, default=2)
    p.add_argument("--tokenizer", help="Optional local Qwen tokenizer for actual token-length audits")
    p = sub.add_parser("export-initial")
    p.add_argument("--base-model", required=True)
    p.add_argument("--sft-adapter", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "build":
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        result = build(args.output_dir, repeats=args.repeats, holdout_repeats=args.holdout_repeats, tokenizer=tokenizer)
    else:
        result = export_initial(args.base_model, args.sft_adapter, args.data, args.output_dir)
    print(dumps(result))


if __name__ == "__main__":
    main()

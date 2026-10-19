"""Held-out paired entity-noise stress test, separate from full-graph F1.

Its synthetic absent-name negatives test input-contract robustness, not clinical
truth. Selection is hash-fixed, not mined from model errors; train accepts neither
the clean nor the corrupted tune rows. No stress score is used for checkpoint
selection. All models in this test must use the same stored prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from generative.chat import render_prompt
from generative.protocol import dumps, parse_decisions
from pair_metrics import counts_metric
from .entity_noise import NoiseConfig, NoEligibleMutation, augment_heldout_row, validate_row


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build(source, output, limit=64, seed=170915):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Choose a new held-out challenge directory")
    if limit < 1:
        raise ValueError("Positive limit required")
    raw = source.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(r.get("split") != "tune" for r in rows):
        raise ValueError("Challenge construction requires explicitly held-out tune rows")
    rows.sort(key=lambda r: digest(f"{seed}:{r['id']}".encode()))
    chosen, per_doc, skipped = [], Counter(), Counter()
    for row in rows:
        validate_row(row, split="tune")
        if per_doc[row["sample_id"]] >= 2:
            skipped["per_document_cap"] += 1
            continue
        try:
            noisy = augment_heldout_row(row, split="tune", config=NoiseConfig(seed=seed))
        except NoEligibleMutation:
            skipped["cannot_preserve_positive"] += 1
            continue
        chosen.append({"clean": row, "noisy": noisy})
        per_doc[row["sample_id"]] += 1
        if len(chosen) >= limit:
            break
    if not chosen:
        raise ValueError("No eligible paired challenge rows")
    output.mkdir(parents=True, mode=0o700)
    content = "".join(dumps(r) + "\n" for r in chosen).encode()
    (output / "paired.jsonl").write_bytes(content)
    manifest = {"source_sha256": digest(raw), "challenge_sha256": digest(content), "pairs_of_prompts": len(chosen),
                "groups": sorted({r["clean"]["group_id"] for r in chosen}), "source_documents": len(per_doc), "seed": seed,
                "skipped": dict(skipped), "split": "tune_stress_not_training", "max_rows_per_document": 2,
                "selection": "hash-fixed before model inference; no checkpoint selection from this test",
                "limitation": "Manufactured absent-name corruption, not adjudicated real prediction errors; fixed G1 16-pair prompt style"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(paired, generations):
    if len(generations) != 2 * len(paired):
        raise ValueError("Incomplete stress generation coverage")
    totals = {"clean": Counter(), "noisy": Counter(), "nonincident": Counter()}
    incident = Counter()
    for i, pair in enumerate(paired):
        predictions = {}
        for j, condition in enumerate(("clean", "noisy")):
            row, generated = pair[condition], generations[2 * i + j]
            if generated.get("id") != row["id"]:
                raise ValueError("Stress generation ID alignment mismatch")
            expected_digest = digest(row["prompt"].encode("utf-8"))
            if (
                "prompt_sha256" in generated
                and generated["prompt_sha256"] != expected_digest
            ):
                raise ValueError("Stress generation prompt_sha256 alignment mismatch")
            expected, gold_errors = parse_decisions(row["completion"], row)
            found, errors = parse_decisions(generated["text"], row)
            if gold_errors:
                raise ValueError("Invalid frozen challenge target")
            gold = {(e["pair_id"], e["relation"]) for e in expected}
            pred = {(e["pair_id"], e["relation"]) for e in found}
            predictions[condition] = (pred, gold)
            invalid_edges = sum(e in {"invalid_decision", "unknown_pair", "invalid_relations", "invalid_edge", "unknown_relation", "invalid_evidence"} for e in errors)
            totals[condition].update(tp=len(pred & gold), fp=len(pred - gold) + invalid_edges, fn=len(gold - pred),
                                     invalid_rows=bool(errors), positive_rows=bool(gold), empty_on_positive=bool(gold and not pred),
                                     truncated_rows=generated.get("truncated", False), rows=1)
        noisy = pair["noisy"]
        changed = set(noisy["entity_noise"]["changed_entity_ids"])
        bad_pairs = {p["pair_id"] for p in noisy["pairs"] if changed.intersection((p["source"], p["target"]))}
        pred, gold = predictions["noisy"]
        incident["candidate_pairs"] += len(bad_pairs)
        incident["pairs_with_emitted_relation"] += len({p for p, r in pred if p in bad_pairs})
        incident["emitted_edges"] += sum(p in bad_pairs for p, r in pred)
        # Both views keep these targets byte-identical, so their recall change
        # isolates collateral loss when a different entity is corrupted.
        survivors = {(p, r) for p, r in gold if p not in bad_pairs}
        totals["nonincident"]["gold_edges"] += len(survivors)
        totals["nonincident"]["clean_tp"] += len(predictions["clean"][0] & survivors)
        totals["nonincident"]["noisy_tp"] += len(pred & survivors)
    result = {condition: {**dict(totals[condition]), **counts_metric(*(totals[condition][k] for k in ("tp", "fp", "fn")))} for condition in ("clean", "noisy")}
    result["noise_incident"] = {**dict(incident), "false_relation_pair_rate": incident["pairs_with_emitted_relation"] / incident["candidate_pairs"] if incident["candidate_pairs"] else 0.0}
    survivor = totals["nonincident"]
    result["nonincident"] = {**dict(survivor), "clean_recall": survivor["clean_tp"] / survivor["gold_edges"] if survivor["gold_edges"] else 0,
                             "noisy_recall": survivor["noisy_tp"] / survivor["gold_edges"] if survivor["gold_edges"] else 0}
    result["definition"] = "Pair-ID + relation micro match within fixed prompts, not entity-name graph F1. Invalid emitted edges count as FP as in the graph evaluator; pure JSON failures contribute missed targets. Positive-empty rows reported separately. Silver clean targets are not human gold."
    return result


def evaluate(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from kg_agent.prompts import extract_json_object

    source, output = Path(args.data), Path(args.output_dir)
    if output.exists():
        raise ValueError("Choose a new stress evaluation output directory")
    paired = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = [row[k] for row in paired for k in ("clean", "noisy")]
    if not rows or any(r.get("split") != "tune" for r in rows):
        raise ValueError("Only frozen held-out challenge rows are allowed")
    for row in rows:
        validate_row(row, split="tune")
    output.mkdir(parents=True, mode=0o700)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    if args.grpo_sft_adapter:
        model = PeftModel.from_pretrained(model, args.grpo_sft_adapter).merge_and_unload()
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()
    generations = []
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        encoded = tokenizer([render_prompt(tokenizer, r["prompt"]) for r in batch], padding=True, return_tensors="pt").to(model.device)
        n = encoded["input_ids"].shape[1]
        if n + 16000 > model.config.max_position_embeddings:
            raise ValueError("Stress prompt exceeds context budget; never silently truncate")
        class CompleteJSON(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = []
                for row in input_ids[:, n:]:
                    try:
                        extract_json_object(tokenizer.decode(row, skip_special_tokens=True))
                        done.append(True)
                    except ValueError:
                        done.append(False)
                return torch.tensor(done, dtype=torch.bool, device=input_ids.device)
        with torch.inference_mode():
            tokens = model.generate(**encoded, do_sample=False, max_new_tokens=16000, use_cache=True, pad_token_id=tokenizer.pad_token_id, stopping_criteria=StoppingCriteriaList([CompleteJSON()]))[:, n:]
        for row, tokens in zip(batch, tokens):
            ids = tokens.tolist()
            if tokenizer.pad_token_id in ids:
                ids = ids[:ids.index(tokenizer.pad_token_id)]
            generated = {"id": row["id"], "prompt_sha256": digest(row["prompt"].encode("utf-8")), "text": tokenizer.decode(ids, skip_special_tokens=True).strip(), "tokens": len(ids), "truncated": len(ids) >= 16000}
            generations.append(generated)
            with (output / "generations.jsonl").open("a") as stream:
                stream.write(dumps(generated) + "\n")
        print(dumps({"completed_prompts": len(generations), "total_prompts": len(rows)}), flush=True)
    metrics = summarize(paired, generations)
    metrics.update(data_sha256=digest(source.read_bytes()), arguments=vars(args), complete_pairs=len(paired), not_full_graph_evaluation=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(dumps(metrics))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "evaluate"])
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--base-model")
    p.add_argument("--adapter")
    p.add_argument("--grpo-sft-adapter")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.mode == "build":
        print(dumps(build(args.data, args.output_dir, args.limit)))
    else:
        if not args.base_model or not args.adapter or args.batch_size < 1:
            p.error("Model, adapter and positive batch size required")
        evaluate(args)


if __name__ == "__main__":
    main()

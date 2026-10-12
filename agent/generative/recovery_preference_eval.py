"""No-gradient held-out preference scores on recipe-verified artificial tasks.

These teacher-forced rankings are neither generated graph accuracy nor medical
generalization. No optimizer, training admission, truncation or JSON repair occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from .chat import render_prompt
from .protocol import dumps
from .recovery_preferences import verify_case
from .train import adapter_fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_holdout(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty held-out preference file")
    ids, details = set(), []
    for row in rows:
        detail = verify_case(row)
        if row["split"] != "mechanical_holdout" or row["training_view"] != "preference":
            raise ValueError("Only recipe-verified mechanical_holdout preference rows are eligible")
        if row["id"] in ids:
            raise ValueError("Duplicate held-out preference ID")
        ids.add(row["id"])
        details.append(detail)
    return rows, details


def encode_preference(row, tokenizer, max_length):
    """Match pinned DPO tokenize_row: separately encode, append exactly one EOS."""
    if type(max_length) is not int or max_length < 2 or tokenizer.eos_token_id is None:
        raise ValueError("Positive sequence budget and EOS tokenizer required")
    prompt = tokenizer.encode(render_prompt(tokenizer, row["prompt"]), add_special_tokens=False)
    chosen = tokenizer.encode(row["chosen"], add_special_tokens=False) + [tokenizer.eos_token_id]
    rejected = tokenizer.encode(row["rejected"], add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prompt or max(len(prompt) + len(chosen), len(prompt) + len(rejected)) > max_length:
        raise ValueError("Held-out sequence exceeds budget; truncation is forbidden")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def classify(margin, tolerance):
    if not math.isfinite(margin) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Finite margin and nonnegative tie tolerance required")
    return 1 if margin > tolerance else -1 if margin < -tolerance else 0


def score_row(row_id, lengths, policy, reference, tolerance):
    values = [*policy, *reference]
    if len(policy) != 2 or len(reference) != 2 or any(not math.isfinite(v) for v in values):
        raise ValueError("Four finite sequence log probabilities required")
    chosen, rejected = lengths
    if min(chosen, rejected) < 1:
        raise ValueError("Completion lengths must include at least EOS")
    pc, pr = policy
    rc, rr = reference
    result = {"id": row_id, "chosen_tokens_including_eos": chosen, "rejected_tokens_including_eos": rejected,
              "policy_chosen_logp": pc, "policy_rejected_logp": pr,
              "reference_chosen_logp": rc, "reference_rejected_logp": rr,
              "raw_margin": pc - pr, "reference_raw_margin": rc - rr,
              "adjusted_margin": (pc - pr) - (rc - rr),
              "policy_chosen_mean_logp": pc / chosen, "policy_rejected_mean_logp": pr / rejected,
              "reference_chosen_mean_logp": rc / chosen, "reference_rejected_mean_logp": rr / rejected,
              "length_normalized_margin": pc / chosen - pr / rejected,
              "reference_length_normalized_margin": rc / chosen - rr / rejected,
              "length_normalized_adjusted_margin": (pc - rc) / chosen - (pr - rr) / rejected}
    for name in ("raw", "reference_raw", "adjusted", "length_normalized", "length_normalized_adjusted"):
        result[name + "_outcome"] = classify(result[name + "_margin"], tolerance)
    if any(not math.isfinite(value) for key, value in result.items() if key != "id"):
        raise ValueError("Nonfinite preference statistic")
    return result


def summarize(scores):
    report = {"samples": len(scores)}
    for name in ("raw", "reference_raw", "adjusted", "length_normalized", "length_normalized_adjusted"):
        counts = Counter(row[name + "_outcome"] for row in scores)
        wins, ties, losses = counts[1], counts[0], counts[-1]
        report[name] = {"wins": wins, "ties": ties, "losses": losses,
                        "decisive_win_rate": wins / (wins + losses) if wins + losses else None,
                        "tie_fraction": ties / len(scores) if scores else None,
                        "mean_margin": sum(row[name + "_margin"] for row in scores) / len(scores) if scores else None}
    return report


def stratify(rows, details, scores):
    groups = defaultdict(lambda: defaultdict(list))
    for row, detail, score in zip(rows, details, scores):
        delta = detail["chosen_edges"] - detail["rejected_edges"]
        length_delta = score["chosen_tokens_including_eos"] - score["rejected_tokens_including_eos"]
        labels = {"reason": row["reason"],
                  "edge_count_direction": "chosen_more" if delta > 0 else "chosen_fewer" if delta < 0 else "equal",
                  "token_length_direction": "chosen_longer" if length_delta > 0 else "chosen_shorter" if length_delta < 0 else "equal",
                  "candidate_pairs": str(detail["pairs_per_batch"])}
        for key, label in labels.items():
            groups[key][label].append(score)
    return {key: {label: summarize(values) for label, values in sorted(bins.items())}
            for key, bins in sorted(groups.items())}


def checkpoint_equal(model, adapter, directory):
    """Confirm loading preserved checkpoint values, including PEFT upcast edges."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(directory) / "adapter_model.safetensors"
    if not path.is_file():
        raise ValueError("A local safetensors adapter checkpoint is required")
    expected = load_file(str(path), device="cpu")
    actual = get_peft_model_state_dict(model, adapter_name=adapter)
    return set(actual) == set(expected) and all(
        actual[name].shape == expected[name].shape
        and torch.equal(actual[name].detach().float().cpu(), expected[name].float()) for name in expected)


def load_shared_model(base_model, policy_path, reference_path):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(base, policy_path, adapter_name="policy", is_trainable=False, local_files_only=True)
    if not checkpoint_equal(model, "policy", policy_path):
        raise RuntimeError("Policy adapter differs from checkpoint immediately after loading")
    policy_before = adapter_fingerprint(model, "policy")
    for attempt in (1, 2):
        model.load_adapter(reference_path, adapter_name="reference", is_trainable=False, local_files_only=True)
        if checkpoint_equal(model, "reference", reference_path):
            break
    else:
        raise RuntimeError("Reference checkpoint values could not be restored exactly")
    if adapter_fingerprint(model, "policy") != policy_before:
        raise RuntimeError("Reference loading changed policy parameters")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to("cuda").eval(), {"reference_load_attempts": attempt,
                                    "policy_checkpoint_verified": True, "reference_checkpoint_verified": True}


def completion_logp(model, prompt_ids, completion_ids):
    """One unpadded forward; slice logits predicting completion including EOS."""
    import torch
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode(), torch.autocast(model.device.type, enabled=False):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        # Token at index len(prompt) is predicted by logits at len(prompt)-1.
        logits = output.logits[0, len(prompt_ids) - 1: -1, :].float()
        labels = ids[0, len(prompt_ids):]
        if logits.shape[0] != len(completion_ids):
            raise RuntimeError("Completion probability alignment mismatch")
        logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum().item()
    if not math.isfinite(logp):
        raise RuntimeError("Nonfinite completion log probability")
    return float(logp)


def evaluate_scores(model, encoded, rows, details, *, tie_tolerance=1e-6, scorer=completion_logp,
                    fingerprint=adapter_fingerprint):
    if not (len(encoded) == len(rows) == len(details)):
        raise ValueError("Preference alignment mismatch")
    before = {name: fingerprint(model, name) for name in ("policy", "reference")}
    per_adapter = {}
    for adapter in ("policy", "reference"):
        model.set_adapter(adapter)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        per_adapter[adapter] = [(scorer(model, row["prompt"], row["chosen"]),
                                 scorer(model, row["prompt"], row["rejected"])) for row in encoded]
    after = {name: fingerprint(model, name) for name in ("policy", "reference")}
    if before != after:
        raise RuntimeError("No-gradient preference evaluation changed adapter parameters")
    scores = [score_row(row["id"], (len(tokens["chosen"]), len(tokens["rejected"])), policy, reference, tie_tolerance)
              for row, tokens, policy, reference in zip(rows, encoded, per_adapter["policy"], per_adapter["reference"])]
    return scores, {"policy_adapter_sha256": before["policy"],
                    "policy_adapter_sha256_after": after["policy"],
                    "reference_adapter_sha256_before": before["reference"],
                    "reference_adapter_sha256_after": after["reference"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("gold", "base-model", "policy-adapter", "reference-sft-adapter", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not math.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("tie-tolerance must be finite and nonnegative")
    rows, details = load_holdout(args.gold)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new preference evaluation output directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    encoded = [encode_preference(row, tokenizer, args.max_length) for row in rows]
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "loading", "samples": len(rows), "data_sha256": sha(args.gold),
              "arguments": vars(args), "scope": "synthetic_heldout_teacher_forced_preferences_not_medical",
              "tie_tolerance": args.tie_tolerance, "no_truncation": True,
              "score_definition": "Completion-only summed natural log probability including one EOS; prompt tokens excluded.",
              "adjusted_margin_definition": "(policy_chosen-policy_rejected)-(reference_chosen-reference_rejected); unscaled by beta.",
              "outcome_definition": "win: margin>tolerance; loss: margin<-tolerance; tie otherwise. Numeric outcome 1/-1/0.",
              "decisive_win_rate_definition": "wins/(wins+losses); null when all ties, including an unchanged zero-step policy.",
              "length_normalized_scope": "Mean log probability per completion token, including EOS; diagnostic, not the DPO objective.",
              "limitations": ["Artificial held-out names/groups share construction templates; not medical generalization.",
                              "Teacher-forced preference ranking does not measure free-generation validity or graph F1.",
                              "Structural-reject edge-count strata count authored relation fields, not repaired valid edges."]}
    started = time.monotonic()
    try:
        model, loading = load_shared_model(args.base_model, args.policy_adapter, args.reference_sft_adapter)
        if max(len(row["prompt"]) + max(len(row["chosen"]), len(row["rejected"])) for row in encoded) > model.config.max_position_embeddings:
            raise ValueError("Model context capacity exceeded; truncation is forbidden")
        scores, identity = evaluate_scores(model, encoded, rows, details, tie_tolerance=args.tie_tolerance)
        (out / "scores.jsonl").write_text("".join(dumps(row) + "\n" for row in scores))
        report.update(status="complete", **identity, loading=loading,
                      aggregate=summarize(scores), strata=stratify(rows, details, scores),
                      scores_sha256=sha(out / "scores.jsonl"),
                      max_sequence_tokens=max(len(row["prompt"]) + max(len(row["chosen"]), len(row["rejected"])) for row in encoded))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(dumps({"status": report["status"], "samples": len(rows), "aggregate": report["aggregate"]}))


if __name__ == "__main__":
    main()

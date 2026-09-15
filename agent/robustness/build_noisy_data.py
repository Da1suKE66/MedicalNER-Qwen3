"""Build new PRIVATE train-only versions from an existing generative dataset.

From the repository, with agent on PYTHONPATH:
  python -m robustness.build_noisy_data --source-dir /private/G1_data \
      --output-dir /private/G1_noise_v1 --rates 0 0.10 0.25 --seed 42

Rates mean noisy ROWS / final ROWS, not corrupted entities / all entities.
Always append: all clean rows and tune bytes survive. Replacement refers only
to an entity inside an added copy. No raw records may be written in the repo.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

from generative.protocol import dumps
from .entity_noise import (
    VERSION, NoiseConfig, NoiseError, NoEligibleMutation, augment_row, augment_heldout_row,
    audit_augmentation, fingerprint, training_fingerprint, validate_row,
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def noisy_count(clean_rows, rate):
    try:
        fraction = Fraction(str(rate))
    except (ValueError, ZeroDivisionError):
        raise NoiseError("Noise rate must be finite") from None
    if not 0 <= fraction <= Fraction(1, 2):
        raise NoiseError("Append noise rate must be in [0, 0.5]; no repeated parent rows")
    return math.floor(clean_rows * fraction / (1 - fraction) + Fraction(1, 2))


def validate_splits(train, tune):
    if not train or not tune:
        raise NoiseError("Nonempty frozen train and tune splits are required")
    for split, rows in (("train", train), ("tune", tune)):
        for row in rows:
            validate_row(row, split=split)
            if "entity_noise" in row:
                raise NoiseError("Input must be an unaugmented frozen dataset")
        if len({r["id"] for r in rows}) != len(rows):
            raise NoiseError("Duplicate row IDs in input split")
    for key in ("group_id", "sample_id", "id"):
        if {str(r[key]) for r in train} & {str(r[key]) for r in tune}:
            raise NoiseError("Train/tune source identity overlap")
    if {fingerprint(r["prompt"]) for r in train} & {fingerprint(r["prompt"]) for r in tune}:
        raise NoiseError("Identical model inputs cross train/tune boundary")


def plan_noise(rows, count, config):
    """Nested seeded ranking, without replacement and without consulting tune labels."""
    if count > len(rows) or count < 0:
        raise NoiseError("Requested more noisy copies than unique clean rows")
    contexts = {}
    for row in rows:
        texts = contexts.setdefault(str(row["sample_id"]), set())
        texts.update(s["text"] for s in row["spans"])
        texts.update(row.get("original_spans", {}).values())
    ranking = sorted(rows, key=lambda r: fingerprint([VERSION, config.seed, "select", r["id"]]))
    pool, views, output_views = [], set(), set()
    counts = Counter()
    for row in ranking:
        if len(pool) == count:
            break
        counts["examined_clean_rows"] += 1
        view = training_fingerprint(row)
        if view in views:
            counts["skipped_duplicate_training_views"] += 1
            continue
        views.add(view)
        try:
            noisy = augment_row(row, config, source_texts=sorted(contexts[str(row["sample_id"])]))
        except NoEligibleMutation:
            counts["skipped_positive_survivor_or_contract_guard"] += 1
            continue
        new_view = training_fingerprint(noisy)
        if new_view in output_views:
            raise NoiseError("Duplicate generated training view")
        output_views.add(new_view)
        pool.append(noisy)
    if len(pool) != count:
        raise NoiseError("Insufficient eligible unique rows; lower rate or use injection")
    counts["generated_unique_noisy_rows"] = len(pool)
    return pool, dict(counts)


def exposure(rows):
    """Actual dataset counts plus weight sums; characters/bytes are NOT tokens."""
    totals, weighted = Counter(), Counter()
    labels = Counter()
    for row in rows:
        decisions = row["target"]["decisions"]
        meta = row.get("entity_noise", {})
        counts = {
            "rows": 1,
            "noisy_rows": int(bool(meta)),
            "entity_appearances": len(row["entities"]),
            "noise_entity_appearances": len(meta.get("changed_entity_ids", [])),
            "pairs": len(row["pairs"]),
            "noise_incident_pairs": len(meta.get("incident_pair_ids", [])),
            "positive_edges": sum(len(d["relations"]) for d in decisions),
            "positive_pairs": sum(bool(d["relations"]) for d in decisions),
            "empty_pairs": sum(not d["relations"] for d in decisions),
            "all_empty_rows": int(not any(d["relations"] for d in decisions)),
            "prompt_characters": len(row["prompt"]),
            "completion_characters": len(row["completion"]),
            "prompt_utf8_bytes": len(row["prompt"].encode("utf-8")),
            "completion_utf8_bytes": len(row["completion"].encode("utf-8")),
        }
        weight = row.get("sample_weight", 1.0)
        totals.update(counts)
        weighted.update({k: weight * v for k, v in counts.items()})
        labels.update(e["relation"] for d in decisions for e in d["relations"])
    weight_sum = weighted["rows"]
    return {
        "counts": dict(totals), "sampling_weight_sums": dict(weighted),
        "groups": len({r["group_id"] for r in rows}),
        "samples": len({r["sample_id"] for r in rows}),
        "positive_relation_label_counts": dict(sorted(labels.items())),
        "weighted_expected_per_draw": {k: v / weight_sum for k, v in weighted.items()} if weight_sum else {},
        "weighted_positive_pair_fraction": weighted["positive_pairs"] / max(1, weighted["pairs"]),
        "noisy_row_fraction": totals["noisy_rows"] / max(1, len(rows)),
        "noise_entity_fraction": totals["noise_entity_appearances"] / max(1, totals["entity_appearances"]),
        "noise_incident_pair_fraction": totals["noise_incident_pairs"] / max(1, totals["pairs"]),
        "weighted_noisy_row_fraction": weighted["noisy_rows"] / max(1, weight_sum),
        "duplicate_training_views": len(rows) - len({training_fingerprint(r) for r in rows}),
    }


def _private_write(path, data):
    """Exclusive create, private permissions, and no overwriting of old versions."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_rows(data):
    try:
        return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeError, ValueError):
        raise NoiseError("Invalid UTF-8 training JSONL; source content suppressed") from None


def _row_hashes(data):
    return "".join(f"{sha256(line)}  line-{i}\n" for i, line in enumerate(data.splitlines(keepends=True), 1)
                   if line.strip()).encode("ascii")


def _code_hashes():
    agent = Path(__file__).resolve().parents[1]
    names = ("robustness/entity_noise.py", "robustness/build_noisy_data.py",
             "generative/protocol.py", "generative/train.py", "generative/chat.py",
             "kg_agent/contracts.py", "kg_agent/ontology.json", "kg_agent/source.py")
    return {name: sha256((agent / name).read_bytes()) for name in names}


def _token_audit(rows, tokenizer, max_length):
    if tokenizer is None:
        return {"status": "NOT_RUN", "reason": "Local model tokenizer not supplied; bytes/characters are not tokens",
                "required_before_training": "Run shared tokenizer/chat-template length preflight; never truncate endpoints"}
    from generative.chat import render_prompt
    from generative.train import length_audit
    rendered = [{**r, "prompt": render_prompt(tokenizer, r["prompt"])} for r in rows]
    audit = length_audit(rendered, tokenizer, max_length, "sft")
    return {"status": "PASSED_SHARED_TRAIN_LENGTH_AUDIT", **audit,
            "max_length": max_length, "total_audited_tokens": round(audit["mean_tokens"] * audit["sequences"])}


def build_versions(source_dir, output_dir, *, rates=(0, 0.10, 0.25), config=None,
                   min_positive_density_retention=0.75, tokenizer=None, max_length=6144):
    """Produce immutable clean/noisy train JSONL, unchanged tune, aggregate reports/hashes."""
    config = config or NoiseConfig()
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        raise NoiseError("Private medical training data must remain outside the repository")
    if output.exists():
        raise NoiseError("Output must be a new path; existing data will not be overwritten")
    if source == output or source in output.parents or output in source.parents:
        raise NoiseError("Input and output dataset directories must not overlap")
    if not 0 < min_positive_density_retention <= 1:
        raise NoiseError("Positive-density retention must be in (0, 1]")
    rates = sorted({Fraction(str(r)) for r in rates})
    if not rates:
        raise NoiseError("At least one noise rate is required")
    for rate in rates:
        noisy_count(1, rate)
    code_hashes = _code_hashes()
    source_bytes = {name: (source / name).read_bytes()
                    for name in ("train.jsonl", "tune.jsonl", "manifest.json")}
    source_hashes = {k: sha256(v) for k, v in source_bytes.items()}
    try:
        source_manifest = json.loads(source_bytes["manifest.json"])
        for split in ("train", "tune"):
            if source_manifest["dataset_sha256"][split] != source_hashes[split + ".jsonl"]:
                raise NoiseError("Source manifest/data SHA256 mismatch")
    except (ValueError, KeyError, TypeError):
        raise NoiseError("Source manifest must contain verified train/tune hashes") from None
    train, tune = (_read_rows(source_bytes[s + ".jsonl"]) for s in ("train", "tune"))
    validate_splits(train, tune)
    train_groups, tune_groups = set(source_manifest["train_groups"]), set(source_manifest["tune_groups"])
    if (train_groups & tune_groups or not {r["group_id"] for r in train} <= train_groups
            or not {r["group_id"] for r in tune} <= tune_groups):
        raise NoiseError("Rows violate frozen source group manifest")
    base_exposure = exposure(train)
    if base_exposure["counts"]["positive_edges"] == 0:
        raise NoiseError("All-empty clean training data cannot establish a safe control")
    pool, selection = plan_noise(train, noisy_count(len(train), max(rates)), config)
    parents = {r["id"]: r for r in train}
    variants, artifacts = {}, {}
    for rate in rates:
        count = noisy_count(len(train), rate)
        noisy = pool[:count]
        rows = train + noisy
        if len({r["id"] for r in rows}) != len(rows):
            raise NoiseError("Generated row ID collision")
        stats = exposure(rows)
        retention = stats["weighted_positive_pair_fraction"] / base_exposure["weighted_positive_pair_fraction"]
        if retention < min_positive_density_retention:
            raise NoiseError("Augmentation dilutes positive-pair exposure beyond configured guard")
        conservation = Counter()
        for row in noisy:
            conservation.update(audit_augmentation(parents[row["entity_noise"]["source_row_id"]], row))
        name = "clean" if rate == 0 else f"noise_{rate.numerator}_of_{rate.denominator}"
        data = source_bytes["train.jsonl"]
        if noisy:
            separator = b"" if data.endswith(b"\n") else b"\n"
            data += separator + "".join(dumps(r) + "\n" for r in noisy).encode("utf-8")
        if not data.startswith(source_bytes["train.jsonl"]):
            raise NoiseError("Clean byte-prefix conservation failed")
        row_hashes = _row_hashes(data)
        variants[name] = {
            "requested_noisy_fraction_of_final_rows": float(rate),
            "appended_noisy_rows": count, "clean_rows_retained": len(train),
            "rounding": "nearest integer, half up, N_noisy = N_clean * rate / (1-rate)",
            "clean_bytes_prefix_unchanged": True, "tune_bytes_unchanged": True,
            "dataset_sha256": {"train": sha256(data), "tune": source_hashes["tune.jsonl"]},
            "train_row_hash_list_sha256": sha256(row_hashes),
            "exposure": stats, "noisy_only_exposure": exposure(noisy),
            "conservation": dict(conservation),
            "construction_counts": dict(Counter(r["entity_noise"]["construction"] for r in noisy)),
            "positive_pair_density_retention": retention,
            "row_budget_multiplier_at_equal_epochs": len(rows) / len(train),
            "epochs_multiplier_for_approx_equal_row_draws": len(train) / len(rows),
            "token_audit": _token_audit(rows, tokenizer, max_length),
        }
        artifacts[name] = (data, row_hashes)
    # Detect concurrent source/core edits rather than stamp mixed code/data as frozen.
    if _code_hashes() != code_hashes or any(sha256((source / k).read_bytes()) != v for k, v in source_hashes.items()):
        raise NoiseError("Source data or implementation changed during build; retry on a frozen snapshot")
    report = {
        "status": "LOCAL_DATA_GENERATED_CONTRACT_AUDITED_NOT_MODEL_TRAINED",
        "version": VERSION, "source_variant": source_manifest.get("variant"),
        "source_protocol_config": source_manifest.get("config"),
        "config": asdict(config), "dataset_strategy": "append_clean_preserved",
        "rate_denominator": "final training rows, not entity/pair/token/weighted-sampler fraction",
        "train_tune_disjoint": True, "tune_rows_unchanged": len(tune),
        "selection": selection, "clean_exposure": base_exposure,
        "minimum_positive_pair_density_retention": min_positive_density_retention,
        "variants": variants,
        "budget_policy": {
            "recommendation": "Use identical explicit optimizer max_steps and effective batch size for controls",
            "weighting": "Clean sample weights unchanged; noisy weights capped at config.noisy_weight <= 1",
            "sampling": "Shared trainer uses weighted replacement when weights differ; per-draw values are expectations, not realized counts",
            "limitation": "Equal optimizer steps do not imply equal token, pair, positive-edge, or entity exposures",
            "collapse_guard": "No positive source batch becomes all-empty; retain all clean rows and enforce weighted positive-pair-density floor",
            "follow_up": "Record realized token/pair/entity counts during GPU runs; evaluate clean recall and all-empty rate separately",
        },
        "caveats": [
            "Inherited teacher-silver supervision is not human-adjudicated medical truth",
            "New empty lists mean only synthetic unsupported-entity incident decisions; UNKNOWN is rejected, never coerced to []",
            "All nonincident targets, including pre-existing silver empty lists, retain their original meaning",
            "Synthetic nonce-name absence is checked against supplied source spans, not a full clinical semantic proof",
            "Every type mutation is coupled to an absent manufactured name; no real entity type-only mismatch is called false",
            "Harder realistic negatives, synonyms, prediction differences and potential silver omissions: PENDING_REVIEW_NOT_TRAINED",
            "Nonce artifacts make this an easy contractual robustness baseline, not a validated realistic-noise benchmark",
            "Context outside the available batches is not reconstructed; G1/G6 supplied full-context batches are retained unchanged",
            "No remote service, external LLM, GPU training or model evaluation is performed",
        ],
    }
    output.mkdir(parents=True, mode=0o700)
    hashes = {"sources": source_hashes, "code": code_hashes, "outputs": {}}
    for name, (data, row_hashes) in artifacts.items():
        destination = output / name
        destination.mkdir(mode=0o700)
        files = {"train.jsonl": data, "tune.jsonl": source_bytes["tune.jsonl"],
                 "train.rows.sha256": row_hashes, "manifest.json": _json_bytes(variants[name])}
        for filename, contents in files.items():
            _private_write(destination / filename, contents)
            hashes["outputs"][f"{name}/{filename}"] = sha256(contents)
    report_bytes = _json_bytes(report)
    _private_write(output / "generation_report.json", report_bytes)
    hashes["outputs"]["generation_report.json"] = sha256(report_bytes)
    _private_write(output / "SHA256SUMS.json", _json_bytes(hashes))
    return report


def build_heldout_version(source_dir, output_dir, *, count=64, config=None,
                          training_noise_files=()):
    """ONE shared paired tune stress set for all models, entirely outside training.

    Select rows using only seed/row IDs and input-prompt deduplication. Read-only
    targets are consulted only after selection, to construct/audit references.
    If replacement would erase all positives, inject in that SAME selected row
    instead; never replace the selected evaluation case using its target/model score.
    """
    config = config or NoiseConfig()
    if type(count) is not int or count < 1:
        raise NoiseError("Held-out count must be a positive integer")
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output == repo or repo in output.parents:
        raise NoiseError("Private held-out data must remain outside the repository")
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise NoiseError("Held-out output must be a new, disjoint path")
    code_hashes = _code_hashes()
    blobs = {k: (source / k).read_bytes() for k in ("train.jsonl", "tune.jsonl", "manifest.json")}
    source_hashes = {k: sha256(v) for k, v in blobs.items()}
    try:
        manifest = json.loads(blobs["manifest.json"])
        if any(manifest["dataset_sha256"][s] != source_hashes[s + ".jsonl"] for s in ("train", "tune")):
            raise NoiseError("Frozen held-out source hashes mismatch")
    except (ValueError, KeyError, TypeError):
        raise NoiseError("Invalid frozen source manifest") from None
    train, tune = _read_rows(blobs["train.jsonl"]), _read_rows(blobs["tune.jsonl"])
    validate_splits(train, tune)
    if (set(manifest["train_groups"]) & set(manifest["tune_groups"])
            or not {r["group_id"] for r in tune} <= set(manifest["tune_groups"])
            or not {r["group_id"] for r in train} <= set(manifest["train_groups"])):
        raise NoiseError("Held-out rows violate frozen group manifest")
    ranking = sorted(tune, key=lambda r: fingerprint([VERSION, "heldout-selection", "tune", config.seed, r["id"]]))
    selected, seen, contexts = [], set(), {}
    for row in tune:
        texts = contexts.setdefault(str(row["sample_id"]), set())
        texts.update(s["text"] for s in row["spans"])
        texts.update(row.get("original_spans", {}).values())
    for row in ranking:
        view = fingerprint(row["prompt"])
        if view in seen:
            continue
        seen.add(view)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise NoiseError("Insufficient unique held-out inputs for requested count")
    # This explicit check supplements the train/tune nonce namespace separation.
    forbidden_names, excluded_hashes = set(), []
    for path in training_noise_files:
        data = Path(path).read_bytes()
        excluded_hashes.append(sha256(data))
        for row in _read_rows(data):
            validate_row(row)
            changed = set(row.get("entity_noise", {}).get("changed_entity_ids", []))
            forbidden_names.update(e["name"].casefold() for e in row["entities"] if e["id"] in changed)
    noisy, pairing, heldout_names = [], [], set()
    conservation, fallback_count = Counter(), 0
    for row in selected:
        texts = sorted(contexts[str(row["sample_id"])])
        try:
            stress = augment_heldout_row(row, split="tune", config=config, source_texts=texts)
        except NoEligibleMutation:
            if config.operation != "replace":
                raise
            stress = augment_heldout_row(row, split="tune", config=replace(config, operation="inject"), source_texts=texts)
            fallback_count += 1
        changed = set(stress["entity_noise"]["changed_entity_ids"])
        names = {e["name"].casefold() for e in stress["entities"] if e["id"] in changed}
        if names & (forbidden_names | heldout_names):
            raise NoiseError("Duplicate nonce name within held-out or across train/held-out")
        heldout_names.update(names)
        conservation.update(audit_augmentation(row, stress, expected_split="tune"))
        try:
            validate_row(stress)  # Must be rejected by train-mode validation.
        except NoiseError:
            pass
        else:
            raise NoiseError("Held-out row incorrectly passed training validation")
        noisy.append(stress)
        pairing.append({"pair_index": len(pairing), "clean_id": row["id"], "noisy_id": stress["id"],
                        "clean_sha256": fingerprint(row), "noisy_sha256": fingerprint(stress)})
    if _code_hashes() != code_hashes or any(sha256((source / k).read_bytes()) != v for k, v in source_hashes.items()):
        raise NoiseError("Frozen source/code changed during held-out construction")
    report = {
        "status": "HELDOUT_STRESS_DATA_ONLY_NOT_TRAINED_OR_MODEL_EVALUATED", "split": "tune",
        "source_variant": manifest.get("variant"), "paired_rows": count,
        "shared_use": "Evaluate the identical clean/noisy files with every compared model",
        "config": asdict(config), "selection": "SHA256(seed, row_id, heldout namespace); input-only dedup; no model results or target-based reselection",
        "selected_groups": len({r["group_id"] for r in selected}),
        "train_tune_group_overlap": 0, "source_files_unchanged": True,
        "reference_targets_use": "Read-only construction and stress scoring only; not SFT/DPO/GRPO supervision or model selection",
        "replacement_to_injection_fallbacks_without_reselection": fallback_count,
        "conservation": dict(conservation), "clean_exposure": exposure(selected), "noisy_exposure": exposure(noisy),
        "nonce_namespace": "heldout_stress/tune, independent of v1 train nonce namespace even with the same seed/row_id",
        "exact_nonce_overlap_with_supplied_training_noise": 0 if training_noise_files else None,
        "training_nonce_files_sha256": excluded_hashes,
        "exact_duplicate_heldout_nonces": 0,
        "token_audit": _token_audit(selected + noisy, None, 6144),
        "evaluation_guidance": "Report paired clean/noisy positive-edge/evidence preservation, noise-incident false positives, invalid decisions and all-empty rate; do not count all targets as empty",
        "caveats": ["Synthetic unsupported-name baseline, not realistic medical adjudication",
                    "Pass these supplied post-grounding batches directly to the shared generative prompt runner; re-grounding would remove the deliberately absent entities",
                    "No real alias, prediction disagreement, negation or silver omission was automatically relabeled negative",
                    "Do not use these stress targets to train, select checkpoints, tune thresholds or choose templates"],
    }
    files = {
        "clean.jsonl": "".join(dumps(r) + "\n" for r in selected).encode("utf-8"),
        "noisy.jsonl": "".join(dumps(r) + "\n" for r in noisy).encode("utf-8"),
        "pairing.jsonl": "".join(dumps(r) + "\n" for r in pairing).encode("utf-8"),
        "generation_report.json": _json_bytes(report),
    }
    output.mkdir(parents=True, mode=0o700)
    for name, data in files.items():
        _private_write(output / name, data)
    _private_write(output / "SHA256SUMS.json", _json_bytes({"sources": source_hashes, "code": code_hashes,
                    "outputs": {k: sha256(v) for k, v in files.items()}}))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Existing frozen generative train/tune/manifest directory")
    parser.add_argument("--output-dir", required=True, help="NEW private directory outside the repository")
    parser.add_argument("--rates", nargs="+", default=["0", "0.10", "0.25"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--operation", choices=["replace", "inject"], default="replace")
    parser.add_argument("--type-mode", choices=["keep", "change", "mixed"], default="mixed")
    parser.add_argument("--injected-pairs", type=int, default=2)
    parser.add_argument("--noisy-weight", type=float, default=1.0)
    parser.add_argument("--min-positive-density-retention", type=float, default=0.75)
    parser.add_argument("--tokenizer", help="Optional LOCAL model/tokenizer directory, no network downloads")
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--heldout", action="store_true", help="Build separate paired stress-only data from frozen tune, never train")
    parser.add_argument("--heldout-count", type=int, default=64)
    parser.add_argument("--training-noise-files", nargs="*", default=[], help="Private training files used ONLY to reject exact nonce overlap")
    args = parser.parse_args(argv)
    tokenizer = None
    if args.tokenizer:
        if not Path(args.tokenizer).is_dir():
            parser.error("Tokenizer must be an existing local directory")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    try:
        if args.heldout:
            report = build_heldout_version(
                args.source_dir, args.output_dir, count=args.heldout_count,
                config=NoiseConfig(args.seed, args.operation, args.type_mode, args.injected_pairs, args.noisy_weight),
                training_noise_files=args.training_noise_files,
            )
            print(dumps({"status": report["status"], "paired_rows": report["paired_rows"],
                         "selected_groups": report["selected_groups"],
                         "exact_nonce_overlap": report["exact_nonce_overlap_with_supplied_training_noise"]}))
            return
        report = build_versions(
            args.source_dir, args.output_dir, rates=args.rates,
            config=NoiseConfig(args.seed, args.operation, args.type_mode, args.injected_pairs, args.noisy_weight),
            min_positive_density_retention=args.min_positive_density_retention,
            tokenizer=tokenizer, max_length=args.max_length,
        )
    except (NoiseError, OSError) as exc:
        # No row contents or medical text is emitted on failures.
        parser.exit(2, f"Build failed: {exc}\n")
    print(dumps({"status": report["status"], "variant": report["source_variant"],
                 "versions": {k: {"train_rows": v["exposure"]["counts"]["rows"],
                                   "noisy_rows": v["appended_noisy_rows"],
                                   "train_sha256": v["dataset_sha256"]["train"]}
                              for k, v in report["variants"].items()}}))


if __name__ == "__main__":
    main()

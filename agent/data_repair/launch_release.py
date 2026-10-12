"""Build a private, versioned dense-pair diagnostic from a frozen v2 release.

No source, evaluation gold or historical training file is modified. Only train
records supply anchors and vocabulary. The frozen tune rows are read solely for
identity/group checks, not for constructing synthetic targets. This is not the
proposed sparse-positive protocol and not expert-adjudicated medical training.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from generative.protocol import dumps
from generative.supervision_guard import audit_partial_supervision
from .abstention_repair import build as build_abstention, validate_case
from .controlled_experiments import audit_rows, positive_signatures
from .release_v2 import build_rows, sha, strict_json, variants
from .supervision_v2 import UNIQUE_EVIDENCE_POLICY, validate_partial_row


VERSION = "dense-abstention-unique-evidence-20261012-v1"
ARM_NAMES = {"V3_missing_only": "V4_missing_only", "V3_grounded_none": "V4_grounded_empty"}
INPUTS = ("records.jsonl", "frozen_split.jsonl", "base/train.jsonl", "base/tune.jsonl")


def read_jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines() if line.strip()]


def validate_frozen_inputs(prepared, assignments, historical_base, tune):
    frozen = {r["source_index"]: (r["group_id"], r["split"]) for r in assignments}
    if len(frozen) != len(assignments) or len(prepared) != len(assignments):
        raise ValueError("Duplicate/incomplete frozen record identities")
    seen, groups = set(), {"train": set(), "tune": set()}
    for item in prepared:
        index = item["source_index"]
        if index in seen or frozen.get(index) != (item["group_id"], item["split"]):
            raise ValueError("Prepared record identity differs from frozen split")
        if item["split"] not in groups:
            raise ValueError("Unexpected frozen split")
        seen.add(index)
        groups[item["split"]].add(item["group_id"])
    if seen != set(frozen) or groups["train"] & groups["tune"]:
        raise ValueError("Incomplete identities or train/tune group leakage")
    for split, rows in (("train", historical_base), ("tune", tune)):
        for row in rows:
            validate_partial_row(row)
            if row["split"] != split or frozen.get(row["sample_id"]) != (row["group_id"], split):
                raise ValueError("Historical row identity differs from frozen split")
    return groups


def validate_matched_arms(base, arms, train_groups, tune_groups):
    control, treatment = (arms[name] for name in ARM_NAMES.values())
    if len(control) != len(treatment) or any(rows[:len(base)] != base for rows in arms.values()):
        raise ValueError("Original base rows or arm lengths changed")
    if positive_signatures(control) != positive_signatures(treatment):
        raise ValueError("Positive/evidence exposure differs between arms")
    changed = 0
    for left, right in zip(control, treatment):
        for field in ("prompt", "completion", "target", "entities", "pairs", "spans", "group_id", "sample_weight"):
            if left[field] != right[field]:
                raise ValueError("Matched inputs/targets/weights differ")
        for row in (left, right):
            validate_partial_row(row)
            if row["split"] != "train" or row["group_id"] not in train_groups or row["group_id"] in tune_groups:
                raise ValueError("Non-training source entered an arm")
            if '"NONE"' in row["completion"]:
                raise ValueError("Literal NONE target is forbidden")
            if "abstention_provenance" in row:
                validate_case(row)
                meta = row["abstention_provenance"]
                donors = meta["parents"] + ([meta["vocabulary_donor"]] if meta["vocabulary_donor"] else [])
                if any(d["group_id"] not in train_groups or d["group_id"] in tune_groups for d in donors):
                    raise ValueError("Held-out synthetic anchor or vocabulary donor")
        for pid, status in left["pair_supervision"].items():
            other = right["pair_supervision"][pid]
            if status == other:
                continue
            if (status != "UNKNOWN_CONSTRUCTED_ABLATION" or other != "SYNTHETIC_NEGATIVE"
                    or right["abstention_provenance"]["negative_proofs"][pid]["kind"]
                    != "no_explicit_relation_in_two_statement_context"):
                raise ValueError("Unexpected supervision intervention")
            changed += 1
    if changed == 0:
        raise ValueError("No grounded-negative intervention was constructed")
    return changed


def distribution(rows):
    result = Counter()
    for row in rows:
        values = Counter(row["pair_supervision"].values())
        p, n = values["SUPPORTED_SILVER"], values["SYNTHETIC_NEGATIVE"]
        result["mixed_positive_negative_rows" if p and n else "positive_only_supervision_rows" if p
               else "negative_only_supervision_rows" if n else "all_unknown_rows"] += 1
    for key in ("mixed_positive_negative_rows", "positive_only_supervision_rows", "negative_only_supervision_rows", "all_unknown_rows"):
        result.setdefault(key, 0)
    return dict(result)


def build(release, output, *, max_cases=512, seed=42, tokenizer_dir=None, max_length=6144):
    release, output = Path(release).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.exists():
        raise ValueError("Refusing to overwrite an existing launch release")
    if output.is_relative_to(release) or output.is_relative_to(repo):
        raise ValueError("Private output must be outside the source release and repository")
    if type(max_cases) is not int or max_cases < 1:
        raise ValueError("Positive bounded synthetic case count required")
    manifest_bytes = (release / "manifest.json").read_bytes()
    source_manifest = strict_json(manifest_bytes.decode())
    input_hashes = {}
    for name in INPUTS:
        digest = sha((release / name).read_bytes())
        if source_manifest["output_sha256"].get(name) != digest:
            raise ValueError("Frozen input hash mismatch")
        input_hashes[name] = digest
    prepared, assignments, old_base, tune = (read_jsonl(release / name) for name in INPUTS)
    groups = validate_frozen_inputs(prepared, assignments, old_base, tune)
    base, rebuild_stats = [], Counter()
    config = variants()["base"]
    for item in prepared:
        if item["split"] != "train":
            continue
        item = deepcopy(item)
        item["evidence_grounding_policy"] = UNIQUE_EVIDENCE_POLICY
        rows, counts, _ = build_rows(item, config, sampled=True, seed=seed)
        base.extend(rows)
        rebuild_stats.update(counts)
    if not base or positive_signatures(base) - positive_signatures(old_base):
        raise ValueError("Unique-evidence rebuild is empty or introduces new positive/evidence targets")
    built, construction = build_abstention(base, tune, max_cases=max_cases)
    if construction["synthetic_cases"] != max_cases:
        raise ValueError("Insufficient eligible cases for the declared experiment budget")
    arms = {ARM_NAMES[name]: rows for name, rows in built.items()}
    changed = validate_matched_arms(base, arms, groups["train"], groups["tune"])
    tokenizer = None
    if tokenizer_dir:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if not tokenizer.is_fast or tokenizer.eos_token_id is None:
            raise ValueError("A local fast tokenizer with EOS is required")
    audits = {name: {"supervision": audit_rows(rows, tokenizer, max_length),
                     "row_distribution": distribution(rows),
                     "admission": audit_partial_supervision(rows)} for name, rows in arms.items()}
    # Freeze original rows again before writing anything, so concurrent source
    # changes cannot produce a seemingly versioned mixed-input dataset.
    if (release / "manifest.json").read_bytes() != manifest_bytes or any(
            sha((release / name).read_bytes()) != digest for name, digest in input_hashes.items()):
        raise ValueError("Source release changed while constructing the launch data")
    code_names = ("agent/data_repair/launch_release.py", "agent/data_repair/release_v2.py",
                  "agent/data_repair/supervision_v2.py", "agent/data_repair/abstention_repair.py",
                  "agent/data_repair/text_normalization.py", "agent/generative/protocol.py",
                  "agent/kg_agent/contracts.py", "agent/kg_agent/ontology.json", "agent/kg_agent/source.py")
    public = {"version": VERSION, "protocol": "dense_fixed_pair_diagnostic_not_sparse_positive",
              "source_manifest_sha256": sha(manifest_bytes), "source_input_sha256": input_hashes,
              "seed": seed, "evidence_grounding_policy": UNIQUE_EVIDENCE_POLICY,
              "protocol_config": asdict(config), "synthetic_cases_per_arm": max_cases,
              "source_group_counts": {key: len(value) for key, value in groups.items()},
              "base_rows": len(base), "base_supervision": audit_rows(base), "rebuild_stats": dict(rebuild_stats),
              "historical_base_positive_edges": sum(positive_signatures(old_base).values()),
              "rebuilt_base_positive_edges": sum(positive_signatures(base).values()),
              "constructed_source_groups": construction["source_groups"],
              "grounded_negative_decisions_unmasked": changed, "audits": audits,
              "identical_prompts_completions_positive_targets": True,
              "original_unknown_supervision_unchanged_across_arms": True,
              "supervised_token_budget_matched": False, "tokenizer_checked": tokenizer is not None,
              "max_length": max_length, "clinical_adjudication_completed": False,
              "evaluation_gold_modified": False, "existing_release_modified": False,
              "code_sha256": {name: sha((repo / name).read_bytes()) for name in code_names},
              "caveats": ["Constructed extraction negatives are not expert-adjudicated clinical negatives.",
                          "This is a diagnostic dense-pair comparison, not a sparse-output experiment.",
                          "Equal steps and prompts do not equalize supervised completion-token exposure.",
                          "Changed strict-evidence anchors alter synthetic selection versus historical V3."]}
    if tokenizer is not None:
        public["tokenizer_sha256"] = {name: sha((Path(tokenizer_dir) / name).read_bytes())
                                      for name in ("tokenizer.json", "tokenizer_config.json")}
    output.mkdir(parents=True, mode=0o700)
    hashes = {}
    for name, rows in {"base_unique_train": base, **arms}.items():
        path = output / (name + ".jsonl")
        with path.open("x") as handle:
            path.chmod(0o600)
            for row in rows:
                handle.write(dumps(row) + "\n")
        hashes[path.name] = sha(path.read_bytes())
    matrix = {"version": VERSION, "defaults": asdict(config), "variants": {name: {} for name in arms}}
    matrix_path = output / "experiments.json"
    with matrix_path.open("x") as handle:
        matrix_path.chmod(0o600)
        json.dump(matrix, handle, indent=2)
        handle.write("\n")
    hashes[matrix_path.name] = sha(matrix_path.read_bytes())
    public["output_sha256"] = hashes
    with (output / "manifest.json").open("x") as handle:
        (output / "manifest.json").chmod(0o600)
        json.dump(public, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return public


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--max-length", type=int, default=6144)
    args = parser.parse_args()
    report = build(args.release, args.output, max_cases=args.max_cases, seed=args.seed,
                   tokenizer_dir=args.tokenizer_dir, max_length=args.max_length)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

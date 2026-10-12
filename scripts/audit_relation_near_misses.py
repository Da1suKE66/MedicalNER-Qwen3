#!/usr/bin/env python3
"""Audit frozen relation errors without changing gold or the production scorer.

Character similarity is a candidate-finding diagnostic, never proof of synonymy.
The public output contains aggregates/hashes; optional private output contains text.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "agent"), str(REPO / "src")]

from build_stage_data import _message
from evaluate import _graph, _relation_set, _validate_gold_graph
from evaluation_identity import validate_alignment
from generative.rapid_screen import load_screen
from pair_metrics import counts_metric


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def maximum_matching(left, right, compatible):
    """Deterministic maximum-cardinality matching; each error is used once."""
    left, right = sorted(left), sorted(right)
    adjacent = [[j for j, b in enumerate(right) if compatible(a, b)] for a in left]
    owner = {}

    def augment(i, seen):
        for j in adjacent[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(left)):
        augment(i, set())
    return [(left[i], right[j]) for j, i in sorted(owner.items())]


def valid(triple):
    return all(not str(value).startswith("__INVALID")
               for node in (triple[0], triple[2]) for value in node)


def name_similarities(a, b):
    if not valid(a) or not valid(b):
        return None
    if a[1] != b[1] or a[0][0] != b[0][0] or a[2][0] != b[2][0]:
        return None
    return [SequenceMatcher(None, a[i][1], b[i][1], autojunk=False).ratio()
            for i in (0, 2)]


def lexical_compatible(a, b, threshold):
    values = name_similarities(a, b)
    return values is not None and min(values) >= threshold


def audit_job(records, indices, predictions, *, thresholds=(0.8, 0.9)):
    validate_alignment(records, predictions, strict_identity=True, sample_indices=indices)
    totals, structural = Counter(), Counter()
    gains = Counter()
    private = []
    for record, index, prediction in zip(records, indices, predictions):
        gold = _graph(_message(record, "assistant"))
        _validate_gold_graph(gold)
        if "gold" in prediction and prediction["gold"] != gold:
            raise ValueError("Embedded gold differs from frozen input")
        graph = _graph(prediction)
        g = _relation_set(gold)
        p = _relation_set(graph) if graph is not None else set()
        fp, fn, tp = p - g, g - p, p & g
        totals.update(tp=len(tp), fp=len(fp), fn=len(fn))
        structural.update(oracle_entity_tables_equal=int(
            graph is not None and graph["entities"] == gold["entities"]),
            invalid_graphs=int(graph is None))
        candidates = []
        for a in sorted(fp):
            for b in sorted(fn):
                similarities = name_similarities(a, b)
                if similarities is not None and min(similarities) >= 0.5:
                    candidates.append({"fp": a, "fn": b, "endpoint_character_similarity": similarities})
        same_pair = maximum_matching(fp, fn, lambda a, b:
            valid(a) and valid(b) and a[0] == b[0] and a[2] == b[2])
        reverse_same_label = maximum_matching(fp, fn, lambda a, b:
            valid(a) and valid(b) and a[0] == b[2] and a[2] == b[0] and a[1] == b[1])
        # This is a diagnostic for a possible ontology decision, NOT a rule.
        differential_reverse = maximum_matching(fp, fn, lambda a, b:
            a[1] == b[1] == "differentiates_from" and a[0] == b[2] and a[2] == b[0])
        structural.update(same_directed_pair_wrong_label=len(same_pair),
                          reversed_pair_same_label=len(reverse_same_label),
                          reversed_differentiates_from=len(differential_reverse))
        matched = {}
        for threshold in thresholds:
            key = format(threshold, ".2f")
            matches = maximum_matching(fp, fn, lambda a, b: lexical_compatible(a, b, threshold))
            gains[key] += len(matches)
            matched[key] = [{"fp": a, "fn": b} for a, b in matches]
        private.append({"sample_index": index, "source": _message(record, "user"),
                        "tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
                        "name_candidates_at_0_50": candidates, "lexical_matches": matched,
                        "same_directed_pair_wrong_label": same_pair,
                        "reversed_pair_same_label": reverse_same_label,
                        "reversed_differentiates_from": differential_reverse})
    strict = counts_metric(totals["tp"], totals["fp"], totals["fn"])
    lexical = {}
    for threshold in thresholds:
        key = format(threshold, ".2f")
        gain = gains[key]
        lexical[key] = {"additional_matches_not_confirmed_synonyms": gain,
                        "counterfactual_not_semantic_f1": counts_metric(
                            totals["tp"] + gain, totals["fp"] - gain, totals["fn"] - gain)}
    return {"samples": len(records), "strict": strict,
            "structural_diagnostics_nonexclusive": dict(structural),
            "lexical_diagnostics": lexical}, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, action="append", metavar="NAME=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args()
    if args.private_output and args.output.resolve() == args.private_output.resolve():
        parser.error("Public and private output paths must differ")
    records, indices, _, manifest = load_screen(args.screen_dir)
    public = {"version": "relation-near-miss-audit-v1", "scope": "frozen_medical_oracle_subset",
              "records_sha256": manifest["records_sha256"],
              "sample_indices_sha256": manifest["sample_indices_sha256"],
              "samples_per_run": len(records), "script_sha256": sha(__file__),
              "methods": {"primary_metric": "Unchanged strict typed directed relation micro-F1",
                          "lexical_candidates": "Same exact relation label and endpoint types; compare source with source and target with target using difflib.SequenceMatcher(autojunk=False), both ratios >= threshold",
                          "matching": "Lock strict TP first, then maximum one-to-one FP/FN matching within each sample; gold and prediction counts stay fixed",
                          "interpretation": "Lexical matches are unadjudicated diagnostics, not semantic correctness or model improvement",
                          "structural_counts": "Separate nonexclusive diagnostics; cannot be added or treated as TP",
                          "oracle_limit": "Given entities do not test generated entity synonyms or end-to-end entity linking"},
              "jobs": {}}
    private = {}
    input_paths = [(args.screen_dir / name).resolve()
                   for name in ("records.json", "sample_indices.json", "manifest.json", "matrix.json")]
    for spec in args.predictions:
        name, separator, path = spec.partition("=")
        if not separator or not name or name in public["jobs"]:
            parser.error("Predictions require unique NAME=PATH entries")
        path = Path(path)
        input_paths.append(path.resolve())
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        report, details = audit_job(records, indices, rows)
        report["predictions_sha256"] = sha(path)
        public["jobs"][name] = report
        private[name] = details
    if any(output.resolve() in input_paths for output in (args.output, args.private_output) if output):
        parser.error("An output path would overwrite frozen inputs")
    for path, value in ((args.output, public), (args.private_output, private)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            if path == args.private_output:
                path.chmod(0o600)
    print(json.dumps({name: {"strict": job["strict"], "lexical_diagnostics": job["lexical_diagnostics"]}
                      for name, job in public["jobs"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Multi-label relation metrics; NONE never inflates positive-class F1."""

from __future__ import annotations
import json


def counts_metric(tp, fp, fn, beta=1):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "f0_5": (
            1.25 * tp / (1.25 * tp + fp + 0.25 * fn)
            if 1.25 * tp + fp + 0.25 * fn
            else 0.0
        ),
    }


def score_pairs(rows, probabilities, labels, threshold):
    if len(rows) != len(probabilities) or any(
        len(p) != len(labels) for p in probabilities
    ):
        raise ValueError("Pair/probability alignment mismatch")
    tp = fp = fn = exact = 0
    per = {
        label: [0, 0, 0]
        for label in set(labels) | {l for row in rows for l in row["labels"]}
    }
    for row, probs in zip(rows, probabilities):
        pred = {
            label
            for label, p in zip(labels, probs)
            if p >= threshold and label in row["allowed"]
        }
        gold = set(row["labels"])
        exact += pred == gold
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for label in pred | gold:
            per[label][0] += label in pred & gold
            per[label][1] += label in pred - gold
            per[label][2] += label in gold - pred
    detail = {label: counts_metric(*c) for label, c in sorted(per.items())}
    active = [v for v in detail.values() if v["tp"] + v["fn"] > 0]
    return {
        **counts_metric(tp, fp, fn),
        "threshold": threshold,
        "pair_exact_accuracy": exact / len(rows) if rows else 0,
        "samples": len(rows),
        "macro_f1": sum(v["f1"] for v in active) / len(active) if active else 0,
        "per_class": detail,
    }


def select_threshold(rows, probabilities, labels):
    sweep = [
        score_pairs(rows, probabilities, labels, t)
        for t in [0.15, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]
    ]
    qualified = [m for m in sweep if m["precision"] >= 0.5 and m["tp"] > 0]
    best = max(
        qualified or sweep, key=lambda m: (m["f0_5"], m["precision"], m["recall"])
    )
    return best, sweep

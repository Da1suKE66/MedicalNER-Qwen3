"""Optional type-specific rejection thresholds, fit on tuning data only.

Uses saved per-class threshold sweeps. Never reads regression predictions.
Small-support classes retain the global threshold. A supported class without
any precision-qualified threshold abstains; this is reported, not called 100% P.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from pair_metrics import counts_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", required=True)
    p.add_argument("--tune", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-support", type=int, default=20)
    p.add_argument("--min-precision", type=float, default=0.7)
    args = p.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    tuning = json.loads(Path(args.tune).read_text())
    if tuning["epoch"] != selection["epoch"]:
        raise ValueError("Wrong epoch tuning file")
    baseline = tuning["selected"]
    thresholds = {}
    metrics = {}
    reasons = {}
    for label, original in baseline["per_class"].items():
        support = original["tp"] + original["fn"]
        threshold = selection["threshold"]
        chosen = original
        reason = "global_threshold_low_support"
        if support >= args.min_support:
            candidates = [
                s
                for s in tuning["sweep"]
                if s["per_class"][label]["precision"] >= args.min_precision
                and s["per_class"][label]["tp"] >= 5
            ]
            if candidates:
                best = max(
                    candidates,
                    key=lambda s: (
                        s["per_class"][label]["f0_5"],
                        s["per_class"][label]["precision"],
                    ),
                )
                threshold = best["threshold"]
                chosen = best["per_class"][label]
                reason = "precision_qualified_tuning_threshold"
            else:
                threshold = 1.01
                chosen = counts_metric(0, 0, support)
                reason = "abstain_no_precision_qualified_threshold"
        if label in selection["labels"]:
            thresholds[label] = threshold
        metrics[label] = chosen
        reasons[label] = reason
    counts = [sum(m[k] for m in metrics.values()) for k in ["tp", "fp", "fn"]]
    aggregate = counts_metric(*counts)
    retain = aggregate["f0_5"] > baseline["f0_5"] and aggregate["precision"] >= 0.5
    result = {
        **selection,
        "calibration": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "Selected epoch training-group-held-out tuning sweep only; no dev inputs read",
            "min_support": args.min_support,
            "min_precision": args.min_precision,
            "min_true_positives": 5,
            "retain_by_tuning_f0_5": retain,
            "global_tuning": {
                k: baseline[k]
                for k in ["tp", "fp", "fn", "precision", "recall", "f1", "f0_5"]
            },
            "labelwise_tuning": aggregate,
            "reasons": reasons,
            "per_class": metrics,
        },
    }
    if retain:
        result["label_thresholds"] = thresholds
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "retain": retain,
                "global": result["calibration"]["global_tuning"],
                "labelwise": aggregate,
                "thresholds": thresholds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

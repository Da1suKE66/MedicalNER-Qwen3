"""CPU-only admission gate for partial SFT, not a new loss or label policy.

UNKNOWN placeholders never count as negatives. Counts describe one dataset
pass, not realized exposures under weighted sampling. Negative provenance may
live outside the prompt; this guard does not certify its clinical reliability.
"""

from collections import Counter
import math

from data_repair.supervision_v2 import validate_partial_row


POSITIVE = "SUPPORTED_SILVER"
NEGATIVE = "SYNTHETIC_NEGATIVE"


class SupervisionGuardError(ValueError):
    """Carry a JSON-serializable audit even when admission is refused."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def is_partial_row(row):
    """Match the existing SFT dataset's opt-in partial-format detection."""
    return isinstance(row, dict) and (
        str(row.get("data_version", "")).startswith("teacher-silver-repair-v2")
        or any(key in row for key in (
            "loss_mask_contract", "loss_mask_spans", "pair_supervision"
        ))
    )


def _validate_row(row):
    statuses = row.get("pair_supervision")
    if not isinstance(statuses, dict) or any(
        not isinstance(status, str) for status in statuses.values()
    ):
        raise ValueError("Missing or invalid per-pair supervision")
    for decision in row["target"]["decisions"]:
        relations = decision["relations"]
        # serialize_partial canonicalizes relation lists. Check the original
        # target too, so falsy non-lists such as {} cannot masquerade as [].
        if not isinstance(relations, list):
            raise ValueError("Partial target relations must be a list")
        if statuses.get(decision["pair_id"]) == NEGATIVE and relations != []:
            raise ValueError("SYNTHETIC_NEGATIVE must have relations=[]")
    # Exact canonical spans protect known targets from accidental masking and
    # UNKNOWN from accidental supervision. Real token offsets remain the
    # responsibility of JsonlSFTDataset / masked_token_labels.
    validate_partial_row(row)
    weight = row.get("sample_weight", 1.0)
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("Partial SFT sample_weight must be finite and positive; "
                         "zero-weight negatives cannot satisfy the safety gate")


def audit_partial_supervision(rows, *, allow_positive_only_ablation=False, max_steps=-1):
    """Audit and admit partial data, or raise with the report attached.

    Legacy-only data is untouched. One valid supervised negative is enough:
    no empirical minimum class ratio is imposed. Positive-only admission is
    explicitly diagnostic and requires a finite positive optimizer-step budget.
    The override never bypasses invalid rows, masks, or mixed-format checks.
    """
    partial = [row for row in rows if is_partial_row(row)]
    audit = {
        "version": "partial-supervision-guard-v1",
        "applicable": bool(partial),
        "rows": len(rows),
        "partial_rows": len(partial),
        "legacy_rows": len(rows) - len(partial),
        "validated_rows": 0,
        "invalid_rows": 0,
        "validation_errors": [],
        "count_scope": "one pass of validated partial rows, before weighted sampling; pairs, not relation edges or tokens",
        "allow_positive_only_ablation": bool(allow_positive_only_ablation),
        "positive_only_ablation": False,
        "ablation_max_steps": None,
        "minimum_negative_pairs": 1,
        "minimum_negative_ratio": None,
        "warnings": [],
        "decision": "not_applicable_legacy",
    }
    if not partial:
        return audit

    counts = Counter({POSITIVE: 0, NEGATIVE: 0})
    for index, row in enumerate(rows):
        if not is_partial_row(row):
            continue
        try:
            _validate_row(row)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            audit["invalid_rows"] += 1
            if len(audit["validation_errors"]) < 10:
                audit["validation_errors"].append({
                    "row_index": index, "row_id": row.get("id"), "error": str(exc),
                })
            continue
        audit["validated_rows"] += 1
        counts.update(row["pair_supervision"].values())

    positive, negative = counts[POSITIVE], counts[NEGATIVE]
    unknown = sum(n for status, n in counts.items() if status.startswith("UNKNOWN_"))
    supervised = positive + negative
    audit.update(
        status_counts=dict(sorted(counts.items())),
        positive_pairs=positive,
        negative_pairs=negative,
        unknown_pairs=unknown,
        unknown_loss_pairs=0,
        supervised_pairs=supervised,
        candidate_pairs=supervised + unknown,
        negative_ratio=negative / supervised if supervised else None,
        negative_ratio_denominator="positive_pairs + negative_pairs; UNKNOWN excluded",
        positive_only=positive > 0 and negative == 0,
        character_mask_validation="failed" if audit["invalid_rows"] else "passed_validate_partial_row",
        tokenizer_validation="pending_JsonlSFTDataset",
    )

    def reject(message):
        audit.update(decision="rejected", rejection_reason=message)
        raise SupervisionGuardError(message, audit)

    if audit["invalid_rows"]:
        first = audit["validation_errors"][0]
        reject(f"Invalid partial supervision in {audit['invalid_rows']} row(s); "
               f"row {first['row_index']} ({first['row_id']}): {first['error']}")
    if audit["legacy_rows"]:
        reject("Do not mix legacy and partial-supervision row formats")
    if not negative:
        if not allow_positive_only_ablation:
            reject("Partial SFT has 0 supervised negative pairs; UNKNOWN is masked, "
                   "not negative. Add reliable negatives, or explicitly use "
                   "--allow-positive-only-ablation with positive --max-steps "
                   "for a bounded diagnostic run only.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            reject("--allow-positive-only-ablation requires positive --max-steps "
                   "for a bounded diagnostic run, not an epoch-budget training run")
        audit.update(
            decision="allowed_positive_only_ablation",
            positive_only_ablation=True,
            ablation_max_steps=max_steps,
        )
        audit["warnings"].append(
            "POSITIVE-ONLY ABLATION: bounded diagnosis only, not a standard training run; "
            "no supervised no-relation targets are present."
        )
    else:
        audit["decision"] = "allowed"
        if negative < positive:
            audit["warnings"].append(
                f"Negative supervision is a minority: {negative}/{supervised} "
                f"({audit['negative_ratio']:.2%}). Descriptive scarcity warning only; "
                "no minimum negative ratio is enforced."
            )
    return audit

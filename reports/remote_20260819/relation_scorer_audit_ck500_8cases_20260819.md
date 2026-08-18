# Checkpoint-500 relation scorer audit: 8 paired cases

## Scope and evidence boundary

- Model prediction: `output_only`, checkpoint-500 generation artifact.
- Cases: `301/251` indexTerms, `402/28` diagnosticCriteria,
  `675/943` exclusions, `1424/1434` definition.
- The DeepSeek target is treated as a noisy teacher-relative reference, not as
  medically reviewed gold.
- All eight predictions parsed as complete JSON and all eight have
  `hit_max_new_tokens=false`; this audit is not confounded by output truncation.
- No retraining was started.

Raw target and prediction strings are preserved byte-for-byte from the source
generation artifact in
`relation_scorer_raw_ck500_8cases_20260819.json`. Full matched/FP/FN keys and
component metrics are in `relation_scorer_audit_ck500_8cases_20260819.json`.

## Five scorer views

| Scorer view | TP / prediction / target | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| raw-ID strict | 78 / 123 / 102 | 0.634 | 0.765 | 0.693 |
| entity-aligned strict (name + type + code/span) | 34 / 123 / 102 | 0.276 | 0.333 | 0.302 |
| core triple only (name + type endpoints) | 75 / 123 / 102 | 0.610 | 0.735 | **0.667** |
| inverse-normalized | 75 / 123 / 102 | 0.610 | 0.735 | **0.667** |

Relation evidence, descriptions, non-endpoint properties, property ordering,
and list ordering are excluded from all four relation identities.

The active workbook-derived v3.1.1 schema declares **no inverse-equivalence
table**. For diagnosis only, the audit additionally tested this explicit
candidate:

`Diagnostic Criterion --required_for--> Disease`

versus

`Disease --has_diagnostic_criterion--> Diagnostic Criterion`.

It rescued **0** relations in these eight cases. Therefore the observed errors
are not explained by simple inverse aliases.

## Component decomposition (core semantic endpoints)

| Component | TP / prediction / target | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| source endpoint | 76 / 123 / 102 | 0.618 | 0.745 | 0.676 |
| target endpoint | 96 / 123 / 102 | 0.780 | 0.941 | **0.853** |
| relation type | 80 / 123 / 102 | 0.650 | 0.784 | 0.711 |
| exact core triple | 75 / 123 / 102 | 0.610 | 0.735 | 0.667 |

- Same type with direction only reversed: **0 / 102**.
- Both endpoints correct but type wrong: **1 / 102**.
- Audit inverse candidate rescued: **0 / 102**.

The high target-endpoint score is partly because many diagnostic relations
point to the same main disease. It must not be interpreted as 94% relation
accuracy.

## Per-case results

| ID | Split | Field | Target / pred relations | raw-ID F1 | aligned-strict F1 | core F1 | Finding |
|---:|---|---|---:|---:|---:|---:|---|
| 301 | train | indexTerms | 32 / 32 | 1.000 | 1.000 | **1.000** | exact memorized-style match |
| 251 | heldout | indexTerms | 32 / 32 | 1.000 | **0.000** | **1.000** | all 32 edges correct; one hub code mismatch cascades in old key |
| 402 | train | diagnosticCriteria | 16 / 19 | 0.114 | 0.000 | **0.057** | genuine source/type disagreement remains |
| 28 | heldout | diagnosticCriteria | 18 / 21 | 0.513 | 0.000 | **0.410** | partial semantic match; several relation classes missed/replaced |
| 675 | train | exclusions | 2 / 0 | 0.000 | 0.000 | **0.000** | genuine under-extraction |
| 943 | heldout | exclusions | 2 / 2 | 1.000 | 1.000 | **1.000** | exact relation match |
| 1424 | train | definition | 0 / 10 | 0.000 | 0.000 | **0.000** | 10 teacher-relative false-positive relations |
| 1434 | heldout | definition | 0 / 7 | 0.000 | 0.000 | **0.000** | 7 teacher-relative false-positive relations |

The two definition cases are negative relation cases; their meaningful result
is `17` predicted relations against `0` target relations, not merely “F1=0”.

## Direct proof of the scorer artifact

For `id=251`:

- target and prediction both contain 33 entities and 32 `subtype_of` relations;
- all 32 raw `(source_id, type, target_id)` triples match;
- all 32 core semantic triples match;
- the relation evidence strings also match;
- the only hub-key difference is the main disease ICD code:
  target `6A60.4`, prediction `6A60.11`.

Because the old relation endpoint key included ICD code, this one entity
metadata error changed every incident edge endpoint and produced a false
`0/32` relation score. This is a scorer design error: entity-attribute errors
must be reported separately and must not multiply into dozens of relation
errors.

Raw local IDs are not a safe replacement. In the diagnostic cases, six raw-ID
matches connect semantically different entities because target and prediction
reuse the same local IDs:

- `id=402`: 1 of 2 raw-ID matches is a semantic endpoint conflict;
- `id=28`: 5 of 10 raw-ID matches are semantic endpoint conflicts.

Therefore raw-ID strict is useful only for debugging deterministic copies, not
as the primary relation quality metric.

## Field-level diagnosis

| Field | Target / pred relations | core P / R / F1 | Interpretation |
|---|---:|---:|---|
| indexTerms | 64 / 64 | 1.000 / 1.000 / **1.000** | relation structure is correct; strict loss is metadata cascade |
| diagnosticCriteria | 34 / 40 | 0.225 / 0.265 / **0.243** | real extraction/type policy mismatch remains |
| exclusions | 4 / 2 | 1.000 / 0.500 / **0.667** | one case exact, one case drops both edges |
| definition | 0 / 17 | 0.000 / 0.000 / **0.000** | systematic teacher-relative over-extraction on this pair |

For diagnosticCriteria, component F1 is source endpoint `0.270`, target endpoint
`0.811`, and relation type `0.378`. The model often reaches the main disease
target but chooses different source concepts and predicates. That is a real
generation/annotation-policy issue, not an ID-only scoring issue.

## Conclusion

**Yes, the scorer is a major part of the previously reported failure, but it is
not the only problem.** The clearest correction is heldout relation F1:

- metadata-sensitive entity-aligned strict: `0.035`;
- core semantic triple: **`0.737`**.

That dramatic recovery is dominated by `id=251`, so it does not prove general
relation quality. The eight-case micro F1 is also dominated by the two 32-edge
indexTerms cases. Genuine issues remain in diagnosticCriteria, the missed
exclusions case, and both definition cases.

The correct production evaluation should report these dimensions separately:

1. entity identity (name + label);
2. entity metadata accuracy (ICD code/span/properties);
3. core relation triple F1 using aligned entity identity;
4. optional, policy-approved inverse-normalized F1;
5. evidence/span grounding as a separate metric;
6. per-field macro metrics and negative-case false-positive accuracy.

Do not choose or retrain a checkpoint from the old code-sensitive relation F1.
First replace that scoreboard, rerun it on the broader 25/150-case set, and
separately review annotation policy for diagnosticCriteria, exclusions, and
definition.

## Verification

- `python3 -m py_compile scripts/audit_relation_scorer_20260819.py`
- `python3 -m unittest tests.test_relation_scorer_audit -v`
- Four scorer regression tests pass:
  metadata cascade, raw-ID semantic collision, inverse-candidate rescue, and
  direction/type decomposition.

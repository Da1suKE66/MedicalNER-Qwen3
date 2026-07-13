# Schema v2 implementation plan and current status

## Scope

This phase covers reproducible medical-text-to-KG extraction. It does not expand the
ontology to answer open-ended medical questions; those belong to a later QuestionSet
and graph/RAG workflow.

The v2 strategy is:

1. freeze source hashes and a versioned draft schema;
2. reproduce data corruption statistics;
3. apply deterministic, auditable migration;
4. route medically ambiguous cases to review instead of guessing;
5. freeze hierarchy-grouped train/validation/held-out splits;
6. train only after the schema conflicts and review queue are resolved.

The duplicated legacy prompt text is intentionally left available only to reproduce
the old adapters. A v2 prompt builder must be generated from the reviewed schema; it
must not silently reuse the legacy prompt while `2.0.0-draft.1` remains unresolved.

## Changes from the initial plan

- The existing 20 records are a `schema_regression_20` set, not an independent test.
- `DifferentialDiagnosis` is not introduced as a label; differential diagnosis remains
  a Disease role until medical review confirms otherwise.
- The broad `MedicalCondition` type is deferred.
- WHO ICD classification exclusions and clinical rule-out relations remain separate
  unresolved design questions.
- Code provenance includes `icdcode`, `coding_system`, `icd_release`, and `icd_uri`.
- `source_record_id` is authoritative; `global_idx` is only a checked fallback.
- Description-child collapse requires definition cues and records offsets explicitly
  against each migrated record's corrected `input` text.

## Implemented artifacts

- `schemas/v2.0.0/schema.json`: executable draft contract.
- `schemas/v2.0.0/conflicts.json`: decisions requiring medical review.
- `data/schema_v2/manifest.json`: source hashes and record counts.
- `reports/null_corruption_audit.json`: reproducible text/null audit.
- `data/schema_v2/migrated/pro_cot_schema_v2.json`: full deterministic migration.
- `data/schema_v2/manual_review.jsonl`: review queue without silent deletion.
- `reports/schema_v2_migration_report.json`: migration and idempotence summary.
- `reports/schema_v2_validation_report.json`: full schema findings.
- `data/schema_v2/splits/`: frozen hierarchy-grouped splits.
- `data/schema_regression/schema_regression_20_v2_draft.json`: corrected regression set.

## Verified current results

| Check | Result |
| --- | ---: |
| Source records | 858 |
| Historical damaged token types | 22 |
| Historical damaged token occurrences | 305 |
| Legitimate JSON null metadata values | 858 |
| Migration output records | 858 |
| Deterministically repaired | 527 |
| Manual review | 293 |
| Structurally invalid | 38 |
| Migration idempotent | yes |
| Records silently lost | 0 |
| Relation evidence retained | 7127 / 7127 |
| Train / validation / held-out / regression | 644 / 84 / 110 / 20 |
| Parent-child edges crossing primary splits | 0 |

The 38 invalid records remain in the migrated artifact and review queue. They are not
silently discarded. Current validation findings are 68 active relation domain/range
violations and 16 unresolved main disease matches.

`test_v2_heldout` is independent only for adapters trained from the new `train_v2`
split. The historical pro858 adapter has already seen the entire 858-record pool, so
its score on this split must not be reported as held-out generalization.

## Training gate

Do not train a v2 adapter until all of the following are true:

1. medical reviewers resolve `schemas/v2.0.0/conflicts.json`;
2. invalid records are repaired or explicitly rejected with reasons;
3. the manual-review queue has an approved disposition;
4. the schema receives an immutable reviewed version;
5. the migration is rerun and strict validation has zero errors;
6. training uses only `train_v2.json`, model selection uses `validation_v2.json`, and
   final generalization is reported only on `test_v2_heldout.json`;
7. `schema_regression_20` is reported separately as a structural regression check.

## Reproduce

```bash
python3 scripts/build_data_manifest.py
python3 scripts/audit_null_corruption.py
python3 scripts/migrate_schema_v2.py --apply-high-confidence-collapses
python3 scripts/validate_schema_v2.py
python3 scripts/build_v2_splits.py \
  --records data/schema_v2/migrated/pro_cot_schema_v2.json
python3 -m unittest discover -s tests -v
```

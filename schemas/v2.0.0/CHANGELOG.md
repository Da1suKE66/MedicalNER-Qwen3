# Schema v2.0.0 draft changelog

## 2.0.0-draft.2

- Added `affects_diagnosis_of`, `must_be_ruled_out_for`,
  `excludes_diagnosis_of`, and `somatic_cause_of` with explicit direction.
- Added conditional migration rules for legacy `excludes_if_present` relations;
  ambiguous target signatures remain under review.
- Approved domain/range definitions for `assesses_for`, `recommended_for`, and
  `triggers_alert_when`.
- Allowed symptom-domain hierarchy for `subsumes`, symptom evidence for
  `supports_subtyping_of`, and patient/symptom prognosis factors.

## 2.0.0-draft.1

- Replaces DSM-5-specific code properties with the canonical `icdcode` field.
- Adds `coding_system`, `icd_release`, and `icd_uri` metadata for the main disease.
- Defines a machine-readable entity/property vocabulary and active relation domain/range rules.
- Marks medically ambiguous relations as `needs_medical_review` instead of guessing constraints.
- Keeps differential diagnoses as `Disease` entities for the draft.
- Defers an over-broad `MedicalCondition` entity type.
- Separates proposed clinical exclusion relations from WHO ICD classification exclusions.
- Records the 22 confirmed token corrections caused by historical `nan` to `null` text corruption.

This version is reproducible but not medically frozen. A reviewed ontology source must produce
a later immutable schema version before v2 model training.

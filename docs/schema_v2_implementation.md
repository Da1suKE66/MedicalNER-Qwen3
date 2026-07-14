# Schema v2 implementation and verified status

## Scope and policy

Schema v2 covers reproducible medical-text-to-knowledge-graph extraction. The
current executable contract is `2.0.0-draft.2` under `schemas/v2.0.0/`. The
historical Gemini outputs remain available for provenance, but they are never copied
into the new assistant target. Training targets contain only strict JSON with the
top-level keys `entities` and `relations`.

The preparation policy is fail closed:

1. repair only deterministic text and structural defects;
2. validate canonical disease identity against the WHO ICD-11 MMS API;
3. retain every rejected code, entity, and relation in an audit record;
4. remove review-only or ungrounded items from the training target;
5. require exact relation evidence offsets into `record.input`;
6. freeze hierarchy-aware splits before conversion or training.

The 20 reviewed examples are `schema_regression_20`, not an independent test set.
They originated from the same 858-record source and are reported only as structural
regression coverage.

## Structural decisions

- The canonical code property is `icdcode`; legacy `code` aliases are migrated.
- Every canonical main Disease also carries `coding_system`, `icd_release`, and
  `icd_uri` sourced from the authoritative record and verified by WHO.
- `source_record_id` is authoritative; `global_idx` is only a checked fallback.
- `DifferentialDiagnosis` is not introduced as a separate entity label. A
  differential diagnosis remains a Disease role.
- Patient Information can point to the canonical Disease with
  `affects_diagnosis_of` only when the patient phrase is present in the source.
- Exclusion semantics are separated into `excludes_diagnosis_of` and
  `must_be_ruled_out_for` with fixed source and target domains.
- `somatic_cause_of` is an active Disease-to-Disease relation, but migration does
  not manufacture an edge when the source does not explicitly establish both the
  cause entity and the causal claim.
- Description-child collapse is deliberately narrow. In the inattention regression
  case, the generated `S2`-`S4` description nodes are removed and their three phrases
  become `S1.properties.manifestations`. The collapse requires a definition cue,
  duplicate disease relation, empty child properties, and exact source offsets.

## API integration

DeepSeek is used only for constrained, targeted repair proposals. The default model
is `deepseek-v4-flash`, and thinking is disabled by default because the repair
protocol requires strict response-format JSON. Every proposal is checked against a
record hash, schema version, relation index, domain/range, evidence text, and a local
confidence threshold before it can be applied. A rejected proposal never produces a
partial output file.

WHO ICD credentials are obtained from the
[WHO ICD API portal](https://icd.who.int/icdapi/Account/Register). Authentication
uses OAuth2 client credentials as described in the
[WHO authentication guide](https://icd.who.int/docs/icd-api/API-Authentication/),
and requests carry `API-Version: v2`. The implementation validates a code through
the MMS `codeinfo` endpoint and then verifies the returned entity title and URI.
HTTP status, endpoint, and a redacted response summary are included in failures.

Copy `.env.example` to `.env` and fill the credentials locally. `.env` is ignored by
Git and must never be printed, copied into a report, or committed.

## Implemented artifacts

- `schemas/v2.0.0/schema.json` and `conflicts.json`: executable contract and
  deferred medical-review decisions.
- `src/kg_lora/schema_v2.py`: strict loading, deterministic migration, structural
  rewrites, and validation.
- `src/kg_lora/icd_api.py` and `icd_validation.py`: WHO client and audited result
  application.
- `src/kg_lora/evidence_repair.py`: deterministic evidence-offset recovery.
- `src/kg_lora/training_sanitizer.py`: removal of review-only, unverified-evidence,
  and ungrounded training items.
- `src/kg_lora/training_data_audit.py`: final fail-closed corpus and split audit.
- `src/kg_lora/convert_schema_v2_to_llamafactory.py`: strict Schema v2 ShareGPT
  conversion without legacy CoT.
- `data/schema_v2/cleaned/pro_cot_schema_v2_train_ready.json`: final audited corpus.
- `data/schema_v2/splits/`: frozen train, validation, held-out, and regression sets.
- `data/llamafactory/schema_v2_*_llamafactory.json`: converted training artifacts.

## Verified results

| Check | Result |
| --- | ---: |
| Source and migrated records | 858 / 858 |
| Historical damaged word tokens | 22 types / 305 occurrences |
| Legitimate source JSON nulls | 858, all browser-link metadata |
| Cleaned damaged words / JSON nulls | 0 / 0 |
| Canonical main diseases verified by WHO | 858 / 858 |
| Non-main code associations rejected with audit | 72 HTTP 404 / 256 title mismatch |
| Valid but unnecessary non-main code associations not attached | 406 |
| Relations removed: no verified span / review-only | 728 / 182 |
| Final train-ready entities / relations | 8,590 / 6,702 |
| Final record status | 858 repaired, 0 invalid or manual review |
| Train / validation / held-out / regression | 670 / 84 / 84 / 20 |
| Direct hierarchy or shared-parent leakage | 0 |
| LLaMAFactory train / validation / held-out | 670 / 84 / 84 |
| Final strict audit | all gates passed |

The raw WHO scan returned 929 valid distinct codes and 72 explicit 404 responses
among 1,001 codes. All 858 canonical record codes were valid. Non-main code
properties were historical generated associations and are not part of the final
main-Disease-only coding contract; their WHO result and disposition remain recorded
in migration audit metadata.

The final 6,702 relations use active types only, satisfy domain/range direction, and
have exact source evidence spans. The converter accepted every frozen record:
670/670 train, 84/84 validation, and 84/84 held-out.

## Reproduce the data pipeline

Run from the repository root after filling WHO credentials in the ignored `.env`:

```bash
python3 scripts/build_data_manifest.py
python3 scripts/audit_null_corruption.py
python3 scripts/migrate_schema_v2.py \
  --apply-high-confidence-collapses \
  --force-renormalize
python3 scripts/repair_evidence_spans.py

# Record all valid, mismatched, and failed non-main lookups in one report.
python3 scripts/validate_icd_codes.py \
  --input data/schema_v2/cleaned/pro_cot_schema_v2_evidence.json \
  --input-format schema-v2 \
  --output reports/icd_api_validation_full_report.json

# Fail unless all canonical main diseases are verified and no transient lookup
# remains unresolved.
python3 scripts/apply_icd_validation.py --strict
python3 scripts/sanitize_schema_v2_for_training.py --strict
python3 scripts/build_v2_splits.py \
  --records data/schema_v2/cleaned/pro_cot_schema_v2_train_ready.json
python3 scripts/audit_schema_v2_training_data.py --strict

python3 scripts/convert_schema_v2_to_llamafactory.py
python3 scripts/convert_schema_v2_to_llamafactory.py \
  --input data/schema_v2/splits/validation_v2.json \
  --output data/llamafactory/schema_v2_validation_llamafactory.json \
  --manifest data/llamafactory/schema_v2_validation_manifest.json \
  --expected-split validation
python3 scripts/convert_schema_v2_to_llamafactory.py \
  --input data/schema_v2/splits/test_v2_heldout.json \
  --output data/llamafactory/schema_v2_heldout_llamafactory.json \
  --manifest data/llamafactory/schema_v2_heldout_manifest.json \
  --expected-split test_v2_heldout
python3 -m pytest -q
```

## Training profiles

The unified runner keeps NVIDIA/CUDA and Huawei Ascend/NPU paths side by side.
Legacy remains the default; Schema v2 is selected explicitly:

```bash
# NVIDIA CUDA
bash scripts/run_medicalner_qwen3.sh \
  --backend cuda --task smoke --data-profile schema-v2
bash scripts/run_medicalner_qwen3.sh \
  --backend cuda --task train --data-profile schema-v2

# Huawei Ascend NPU
bash scripts/run_medicalner_qwen3.sh \
  --backend npu --task smoke --data-profile schema-v2
bash scripts/run_medicalner_qwen3.sh \
  --backend npu --task train --data-profile schema-v2
```

The full profiles train only on `medicalner_schema_v2_train`, use
`medicalner_schema_v2_validation` for model selection, and set `val_size: 0` so the
frozen validation set is not replaced by a random subset. The held-out set must not
be used until final evaluation of the newly trained adapter.

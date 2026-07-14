# Local DeepSeek semantic-audit stage

Training is blocked until the complete semantic audit, deterministic repair, WHO
validation, sanitizer, conversion, and strict data audit all pass. A sanitizer that
simply drops unsupported relations is not evidence that semantic cleaning finished.

## Frozen input

The semantic audit starts from the pre-sanitization, WHO-validated dataset:

```text
data/schema_v2/cleaned/pro_cot_schema_v2_icd_validated.json
```

This preserves all 8,680 entities and 7,612 relations, including the records that the
training sanitizer would otherwise remove. Every task is bound to dataset, record,
schema, prompt, and task SHA-256 values.

## External-data boundary

The `primary` and `review` commands send the complete `record.input` and current graph
to the configured DeepSeek API. Run them only after the data owner has explicitly
approved that third-party disclosure. API keys are loaded from `.env`; they are never
written to SQLite or reports.

## Workflow

For a local clone, the simplest CPU-only route is:

```bash
python3 -m venv .venv-cleaning
source .venv-cleaning/bin/activate
python3 -m pip install -r requirements-cleaning.txt
cp .env.example .env  # skip this when the populated .env already exists
# Edit .env and fill DEEPSEEK_API_KEY.
bash scripts/run_semantic_audit_local.sh
```

This calls external APIs but performs no model training and requires no CUDA, NPU, or
remote server. The script is resumable and stops before patch compilation whenever any
API error, invalid response, conflict, or unresolved finding remains.

Before each full phase, the wrapper sends only a two-record/one-attempt canary. It then
works in batches of 25 (override with `SEMANTIC_AUDIT_BATCH_SIZE`) and stops the queue on
the first failed batch. Thus a bad key, unsupported model, or response-contract mismatch
cannot accidentally enqueue all 858 records. Completed records are never resent.
Non-retryable HTTP failures also stop every record that has not yet started; only the
small number of requests already in flight (bounded by the phase worker count) may
finish before the stop signal is observed.

To verify the local baseline and task plan without sending any API request:

```bash
SEMANTIC_AUDIT_PLAN_ONLY=1 bash scripts/run_semantic_audit_local.sh
```

The equivalent individual commands are shown below.

Create the immutable task plan. Full review is the default:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py plan --review-all
```

Run the first full pass (four workers by default):

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py primary \
  --run-id RUN_ID \
  --max-api-attempts 4 \
  --max-tokens 32768
```

Run the independent blind pass (two workers by default). The review task contains the
same frozen record and cues but never includes the primary response:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py review \
  --run-id RUN_ID \
  --max-api-attempts 4 \
  --max-tokens 32768
```

The SQLite state is under `.cache/semantic-audit/state.sqlite3`. Successful responses
are committed before record state changes, so rerunning a phase resumes incomplete or
failed records without repeating completed calls. The wrapper performs the DeepSeek
stage only; it does not call WHO and does not start training.

Finalize strict primary/review consensus:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py finalize --run-id RUN_ID
```

Any disagreement in entity verdicts, relation/evidence verdicts, cue verdicts,
dimension status, operations, or unresolved findings is a conflict. Conflicts are not
silently averaged and cannot produce a patch.

The progress report contains `actionable_records` with record IDs, phase errors,
response hashes, and conflict sections. Retry only failed API/format responses with:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py reset-failed --run-id RUN_ID
```

Do not repeatedly rerun conflicts: systematic disagreement can recur and incur two
more calls. Adjudicate the saved decisions first. Only after changing the rule, prompt,
or record-specific instruction should you request fresh primary and blind decisions:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py reset-blocked --run-id RUN_ID
```

`api_request_ledger` is append-only across every reset. Its request count remains exact;
token totals are explicitly limited to attempts where DeepSeek returned usage metadata.

If deterministic patch application rejects a record after consensus, force only that
record back through both reviews with:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py reset-record \
  --run-id RUN_ID --source-record-id SOURCE_RECORD_ID
```

After every record has clean strict consensus, compile and apply the hash-bound patch
bundle:

```bash
python3 scripts/audit_schema_v2_semantics_deepseek.py compile --run-id RUN_ID
python3 scripts/audit_schema_v2_semantics_deepseek.py apply --run-id RUN_ID
```

Application is transactional per record and the complete output is written only if
all records pass. The applicator protects main-Disease ICD provenance, rejects object
hash drift, removes contradictory legacy `relation_name`/`relation_type` fields,
requires active relation predicates and exact evidence spans, and rejects any
information-losing Symptom collapse.

## Required coverage

For the current frozen baseline, one full pass must report:

- 858/858 records;
- 8,680/8,680 entity assessments;
- 7,612/7,612 relation assessments;
- every generated high-risk cue assessment;
- 9,438/9,438 dimension assessments (11 per record).

Because `--review-all` is used, the blind pass must independently reach the same
coverage. API errors, invalid responses, pending records, conflicts, and unresolved
findings must all be zero.

## Remaining gates before training

The semantic-clean output is still not train-ready. Continue in this order:

1. Search/validate retained non-main Disease names through WHO ICD-11 MMS. DeepSeek
   may request lookup but never supplies a code. Under the current
   `main_disease_only=true` schema, non-main codes remain audit metadata and are not
   attached silently.
2. Run Schema v2 validation and exact-evidence checks.
3. Regenerate the training-safe dataset with the sanitizer.
4. Rebuild train/validation/held-out splits and verify hierarchy leakage is zero.
5. Rebuild LLaMA-Factory JSON.
6. Run the strict training-data audit; all gates must pass.
7. Only then resume remote A100 training.

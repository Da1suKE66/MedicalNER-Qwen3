#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STATE_DB="${SEMANTIC_AUDIT_STATE_DB:-${ROOT_DIR}/.cache/semantic-audit/state.sqlite3}"
PROGRESS_REPORT="${SEMANTIC_AUDIT_REPORT:-${ROOT_DIR}/reports/semantic_audit_progress.json}"
PRIMARY_WORKERS="${PRIMARY_WORKERS:-4}"
REVIEW_WORKERS="${REVIEW_WORKERS:-2}"
MAX_API_ATTEMPTS="${MAX_API_ATTEMPTS:-4}"
BATCH_SIZE="${SEMANTIC_AUDIT_BATCH_SIZE:-25}"
PLAN_ONLY="${SEMANTIC_AUDIT_PLAN_ONLY:-0}"
EVIDENCE_INPUT="${ROOT_DIR}/data/schema_v2/cleaned/pro_cot_schema_v2_evidence.json"
AUDIT_INPUT="${ROOT_DIR}/data/schema_v2/cleaned/pro_cot_schema_v2_icd_validated.json"
EVIDENCE_REPORT="${ROOT_DIR}/.cache/semantic-audit/evidence_span_repair_report.json"
ICD_APPLICATION_REPORT="${ROOT_DIR}/.cache/semantic-audit/icd_validation_application_report.json"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  echo "Missing ${ROOT_DIR}/.env. Copy .env.example and fill DEEPSEEK_API_KEY first." >&2
  exit 1
fi

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEMANTIC_AUDIT_BATCH_SIZE must be a positive integer." >&2
  exit 1
fi

"${PYTHON_BIN}" -c "import pydantic" >/dev/null 2>&1 || {
  echo "Missing cleaning dependency. Run:" >&2
  echo "  ${PYTHON_BIN} -m pip install -r requirements-cleaning.txt" >&2
  exit 1
}

echo "[1/8] Rebuilding the deterministic pre-sanitization evidence baseline."
"${PYTHON_BIN}" scripts/repair_evidence_spans.py \
  --output "${EVIDENCE_INPUT}" \
  --report "${EVIDENCE_REPORT}"

echo "[2/8] Applying the committed complete WHO validation report."
"${PYTHON_BIN}" scripts/apply_icd_validation.py \
  --input "${EVIDENCE_INPUT}" \
  --output "${AUDIT_INPUT}" \
  --report "${ICD_APPLICATION_REPORT}" \
  --strict

echo "[3/8] Planning the frozen 858-record DeepSeek audit (full blind review enabled)."
"${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py plan \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}" \
  --review-all

RUN_ID="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["run_id"])' "${PROGRESS_REPORT}")"
echo "Audit run_id=${RUN_ID}"

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "Plan-only check passed; no record was sent to DeepSeek."
  exit 0
fi

echo "[4/8] Running a two-record, one-attempt primary canary."
if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py primary \
  --run-id "${RUN_ID}" \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}" \
  --workers 1 \
  --limit 2 \
  --max-api-attempts 1; then
  echo "Primary canary failed; the full 858-record queue was not started." >&2
  echo "Inspect actionable_records in ${PROGRESS_REPORT}, fix the configuration, then use reset-failed." >&2
  exit 2
fi

EXPECTED_PRIMARY="$(${PYTHON_BIN} -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["expected"]["primary_responses"])' "${PROGRESS_REPORT}")"
while true; do
  PRIMARY_DONE="$(${PYTHON_BIN} -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["primary_states"].get("done", 0))' "${PROGRESS_REPORT}")"
  if [[ "${PRIMARY_DONE}" -ge "${EXPECTED_PRIMARY}" ]]; then
    break
  fi
  echo "[4/8] Primary batch: ${PRIMARY_DONE}/${EXPECTED_PRIMARY} complete."
  if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py primary \
    --run-id "${RUN_ID}" \
    --state-db "${STATE_DB}" \
    --report "${PROGRESS_REPORT}" \
    --workers "${PRIMARY_WORKERS}" \
    --limit "${BATCH_SIZE}" \
    --max-api-attempts "${MAX_API_ATTEMPTS}"; then
    echo "Primary batch stopped on API/format/validation failure; later batches were not sent." >&2
    exit 2
  fi
done

echo "[5/8] Running a two-record, one-attempt blind-review canary."
if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py review \
  --run-id "${RUN_ID}" \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}" \
  --workers 1 \
  --limit 2 \
  --max-api-attempts 1; then
  echo "Blind-review canary failed; the full review queue was not started." >&2
  exit 2
fi

EXPECTED_REVIEW="$(${PYTHON_BIN} -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["expected"]["blind_review_responses"])' "${PROGRESS_REPORT}")"
while true; do
  REVIEW_DONE="$(${PYTHON_BIN} -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["review_states"].get("done", 0))' "${PROGRESS_REPORT}")"
  if [[ "${REVIEW_DONE}" -ge "${EXPECTED_REVIEW}" ]]; then
    break
  fi
  echo "[5/8] Blind-review batch: ${REVIEW_DONE}/${EXPECTED_REVIEW} complete."
  if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py review \
    --run-id "${RUN_ID}" \
    --state-db "${STATE_DB}" \
    --report "${PROGRESS_REPORT}" \
    --workers "${REVIEW_WORKERS}" \
    --limit "${BATCH_SIZE}" \
    --max-api-attempts "${MAX_API_ATTEMPTS}"; then
    echo "Blind-review batch stopped on API/format/validation failure; later batches were not sent." >&2
    exit 2
  fi
done

echo "[6/8] Finalizing strict primary/review consensus."
if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py finalize \
  --run-id "${RUN_ID}" \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}"; then
  echo >&2
  echo "The audit is safely paused: pending/API/invalid/conflict/unresolved records remain." >&2
  echo "No patch or training dataset was written." >&2
  echo "Keep the SQLite state and inspect actionable_records in:" >&2
  echo "  ${PROGRESS_REPORT}" >&2
  echo "After fixing API/format failures, reset them with:" >&2
  echo "  ${PYTHON_BIN} scripts/audit_schema_v2_semantics_deepseek.py reset-failed --run-id ${RUN_ID} --state-db ${STATE_DB}" >&2
  echo "Do not blindly rerun conflict/unresolved records: repeated calls can reproduce the same disagreement." >&2
  echo "Adjudicate their saved primary/review decisions first. Only after changing the decision rule or prompt, reset with:" >&2
  echo "  ${PYTHON_BIN} scripts/audit_schema_v2_semantics_deepseek.py reset-blocked --run-id ${RUN_ID} --state-db ${STATE_DB}" >&2
  echo "The append-only api_request_ledger in the report retains all calls across resets." >&2
  exit 2
fi

echo "[7/8] Compiling one hash-bound consensus patch for every record."
"${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py compile \
  --run-id "${RUN_ID}" \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}"

echo "[8/8] Applying all patches transactionally and running the strict semantic gate."
if ! "${PYTHON_BIN}" scripts/audit_schema_v2_semantics_deepseek.py apply \
  --run-id "${RUN_ID}" \
  --state-db "${STATE_DB}" \
  --report "${PROGRESS_REPORT}"; then
  echo >&2
  echo "Patch application stopped atomically; no partial output was written." >&2
  echo "The error above names the failing SOURCE_RECORD_ID. Reset only that record:" >&2
  echo "  ${PYTHON_BIN} scripts/audit_schema_v2_semantics_deepseek.py reset-record --run-id ${RUN_ID} --state-db ${STATE_DB} --source-record-id SOURCE_RECORD_ID" >&2
  echo "Then rerun this script." >&2
  exit 3
fi

echo
echo "DeepSeek semantic-consensus stage completed successfully:"
echo "  data/schema_v2/cleaned/pro_cot_schema_v2_semantic_clean.json"
echo "This is an intermediate artifact, not a fully cleaned training dataset."
echo "Training remains locked until post-repair WHO validation and downstream rebuild/audit finish."

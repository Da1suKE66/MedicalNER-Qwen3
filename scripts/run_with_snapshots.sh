#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 RUN_ID COMMAND [ARG ...]" >&2
  exit 2
fi

RUN_ID="$1"
shift
WORKSPACE="${WORKSPACE:-$(pwd)}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/temp/liluchen}"
SNAPSHOT_INTERVAL_SEC="${SNAPSHOT_INTERVAL_SEC:-600}"
SNAPSHOT_PATHS="${SNAPSHOT_PATHS:-scripts:configs:data/llamafactory:reports}"
RUN_ROOT="${SNAPSHOT_ROOT}/${RUN_ID}"
CODE_ROOT="${RUN_ROOT}/code"
ARTIFACT_ROOT="${RUN_ROOT}/artifacts"
META_ROOT="${RUN_ROOT}/metadata"
LOG_FILE="${RUN_ROOT}/run.log"

mkdir -p "${CODE_ROOT}" "${ARTIFACT_ROOT}" "${META_ROOT}"

snapshot_once() {
  local stamp tmp path
  stamp="$(date +%Y%m%d_%H%M%S)"
  tmp="${RUN_ROOT}/.snapshot_${stamp}.tmp"
  mkdir -p "${tmp}/code" "${tmp}/artifacts" "${tmp}/metadata"
  if git -C "${WORKSPACE}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${WORKSPACE}" rev-parse HEAD > "${tmp}/metadata/git_head.txt" || true
    git -C "${WORKSPACE}" status --short > "${tmp}/metadata/git_status.txt" || true
    git -C "${WORKSPACE}" diff --binary > "${tmp}/metadata/worktree.diff" || true
  fi
  cp -a "${WORKSPACE}/scripts" "${tmp}/code/" 2>/dev/null || true
  cp -a "${WORKSPACE}/configs" "${tmp}/code/" 2>/dev/null || true
  for path in ${SNAPSHOT_PATHS//:/ }; do
    if [[ -e "${WORKSPACE}/${path}" ]]; then
      mkdir -p "${tmp}/artifacts/$(dirname "${path}")"
      cp -a "${WORKSPACE}/${path}" "${tmp}/artifacts/$(dirname "${path}")/" 2>/dev/null || true
    fi
  done
  printf '%s\n' "$(date -Is)" > "${tmp}/metadata/snapshot_time.txt"
  printf '%s\n' "$*" > "${tmp}/metadata/command.txt"
  # /temp on ModelArts does not allow directory rename, so copy into the
  # stable locations instead of relying on an atomic mv of directories.
  rm -rf "${CODE_ROOT}" "${ARTIFACT_ROOT}" "${META_ROOT}"
  mkdir -p "${CODE_ROOT}" "${ARTIFACT_ROOT}" "${META_ROOT}"
  cp -a "${tmp}/code/." "${CODE_ROOT}/" 2>/dev/null || true
  cp -a "${tmp}/artifacts/." "${ARTIFACT_ROOT}/" 2>/dev/null || true
  cp -a "${tmp}/metadata/." "${META_ROOT}/" 2>/dev/null || true
  rm -rf "${tmp}"
}

touch "${RUN_ROOT}/running"
(
  while [[ -f "${RUN_ROOT}/running" ]]; do
    sleep "${SNAPSHOT_INTERVAL_SEC}"
    [[ -f "${RUN_ROOT}/running" ]] || break
    snapshot_once "$@"
  done
) &
SNAPSHOT_PID=$!

snapshot_once "$@"
set +e
"$@" > >(tee -a "${LOG_FILE}") 2>&1
EXIT_CODE=$?
set -e
rm -f "${RUN_ROOT}/running"
kill "${SNAPSHOT_PID}" 2>/dev/null || true
wait "${SNAPSHOT_PID}" 2>/dev/null || true
snapshot_once "$@"
printf '%s\n' "${EXIT_CODE}" > "${META_ROOT}/exit_code.txt"
exit "${EXIT_CODE}"

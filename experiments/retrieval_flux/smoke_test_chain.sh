#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a short multi-job chaining smoke test for retrieval training jobs.

This uses:
  - queue_retrieval_resume_pipeline.sh
  - DRY_RUN=1 (so each job exits quickly after command construction)

Usage:
  smoke_test_chain.sh \
    [--pipeline {a|b|c}] \
    [--run-id {1..7}] \
    [--num-jobs N] \
    [--time-limit 10m] \
    [--queue pbatch] \
    [--no-wait]

Defaults:
  --pipeline a
  --run-id 1
  --num-jobs 3
  --time-limit 10m
  wait for all jobs to complete

Notes:
  - No training is performed (`DRY_RUN=1`).
  - Dependencies are still exercised (`afterany` chain).
USAGE
}

PIPELINE="a"
RUN_ID="1"
NUM_JOBS="3"
TIME_LIMIT="10m"
QUEUE=""
WAIT_FOR_COMPLETION="1"
POLL_INTERVAL_SEC="3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)
      PIPELINE="${2:-}"
      shift 2
      ;;
    --run-id|--run)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --num-jobs|-n)
      NUM_JOBS="${2:-}"
      shift 2
      ;;
    --time-limit|-t)
      TIME_LIMIT="${2:-}"
      shift 2
      ;;
    --queue|-q)
      QUEUE="${2:-}"
      shift 2
      ;;
    --poll-interval-sec)
      POLL_INTERVAL_SEC="${2:-}"
      shift 2
      ;;
    --no-wait)
      WAIT_FOR_COMPLETION="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${PIPELINE}" =~ ^[abc]$ ]]; then
  echo "Error: --pipeline must be one of a, b, c." >&2
  exit 1
fi
if ! [[ "${RUN_ID}" =~ ^[0-9]+$ ]] || [[ "${RUN_ID}" -lt 1 ]] || [[ "${RUN_ID}" -gt 7 ]]; then
  echo "Error: --run-id must be in [1, 7]." >&2
  exit 1
fi
if ! [[ "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
  echo "Error: --num-jobs must be >= 1." >&2
  exit 1
fi
if ! [[ "${POLL_INTERVAL_SEC}" =~ ^[0-9]+$ ]] || [[ "${POLL_INTERVAL_SEC}" -lt 1 ]]; then
  echo "Error: --poll-interval-sec must be >= 1." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE_SCRIPT="${SCRIPT_DIR}/queue_retrieval_resume_pipeline.sh"
if [[ ! -f "${QUEUE_SCRIPT}" ]]; then
  echo "Error: missing queue script: ${QUEUE_SCRIPT}" >&2
  exit 1
fi

cmd=(
  bash "${QUEUE_SCRIPT}"
  --pipeline "${PIPELINE}"
  --run-id "${RUN_ID}"
  --num-jobs "${NUM_JOBS}"
  --time-limit "${TIME_LIMIT}"
)
if [[ -n "${QUEUE}" ]]; then
  cmd+=( --queue "${QUEUE}" )
fi

echo "Submitting chain smoke test:"
echo "  pipeline=${PIPELINE} run_id=${RUN_ID} num_jobs=${NUM_JOBS} time_limit=${TIME_LIMIT}"
if [[ -n "${QUEUE}" ]]; then
  echo "  queue=${QUEUE}"
fi

set +e
submit_output="$(
  DRY_RUN=1 \
  WAIT_FOR_READY=0 \
  STRICT_SEQ_EMBEDDINGS=0 \
  USE_WANDB=0 \
  CONTINUE_TRAINING=1 \
  RESUME_MODE=auto \
  "${cmd[@]}" 2>&1
)"
submit_rc=$?
set -e
printf '%s\n' "${submit_output}"
if [[ "${submit_rc}" -ne 0 ]]; then
  echo "Smoke test submission failed." >&2
  exit "${submit_rc}"
fi

mapfile -t JOB_IDS < <(
  printf '%s\n' "${submit_output}" \
    | awk '/^[[:space:]]+[A-Za-z0-9]+$/{gsub(/^[[:space:]]+/, "", $0); print $0}'
)
if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
  echo "Warning: could not parse submitted job IDs from output." >&2
  exit 0
fi

echo "Parsed chain job IDs:"
printf '  %s\n' "${JOB_IDS[@]}"

if [[ "${WAIT_FOR_COMPLETION}" != "1" ]]; then
  exit 0
fi

echo "Waiting for jobs to complete..."
for jid in "${JOB_IDS[@]}"; do
  while true; do
    state="$(flux jobs -a | awk -v id="${jid}" '$1==id {print $4; exit}')"
    # Job already aged out of table: treat as complete for smoke test.
    if [[ -z "${state}" ]]; then
      echo "  ${jid}: <not listed> (assumed complete)"
      break
    fi
    case "${state}" in
      CD|F|CA|TO)
        echo "  ${jid}: ${state}"
        break
        ;;
      *)
        sleep "${POLL_INTERVAL_SEC}"
        ;;
    esac
  done
done

echo "Smoke test chain finished."

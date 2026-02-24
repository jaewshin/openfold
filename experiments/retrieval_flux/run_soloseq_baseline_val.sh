#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit vanilla OpenFold SoloSeq validation baseline (no retrieval) to Flux.

Usage:
  bash experiments/retrieval_flux/run_soloseq_baseline_val.sh [-- <extra eval args>]

Optional env vars:
  REPO_ROOT=<auto>
  FLUX_NODES=1
  FLUX_TASKS=1
  FLUX_CORES=8
  FLUX_GPUS=1
  FLUX_TIME_LIMIT=6h
  FLUX_QUEUE=pbatch
  FLUX_DEPENDENCY=""
USAGE
}

EXTRA_EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_EVAL_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

FLUX_NODES="${FLUX_NODES:-1}"
FLUX_TASKS="${FLUX_TASKS:-1}"
FLUX_CORES="${FLUX_CORES:-8}"
FLUX_GPUS="${FLUX_GPUS:-1}"
FLUX_TIME_LIMIT="${FLUX_TIME_LIMIT:-6h}"
FLUX_QUEUE="${FLUX_QUEUE:-pbatch}"
FLUX_DEPENDENCY="${FLUX_DEPENDENCY:-}"

FLUX_ARGS=(
  -N "${FLUX_NODES}"
  -n "${FLUX_TASKS}"
  -c "${FLUX_CORES}"
  -g "${FLUX_GPUS}"
  -t "${FLUX_TIME_LIMIT}"
)
if [[ -n "${FLUX_QUEUE}" ]]; then
  FLUX_ARGS+=( -q "${FLUX_QUEUE}" )
fi
if [[ -n "${FLUX_DEPENDENCY}" ]]; then
  FLUX_ARGS+=( --dependency "${FLUX_DEPENDENCY}" )
fi

echo "Submitting SoloSeq baseline eval"
echo "  repo_root=${REPO_ROOT}"
echo "  flux_nodes=${FLUX_NODES} flux_tasks=${FLUX_TASKS} flux_cores=${FLUX_CORES} flux_gpus=${FLUX_GPUS}"
echo "  flux_time_limit=${FLUX_TIME_LIMIT} flux_queue=${FLUX_QUEUE:-<none>}"
if [[ -n "${FLUX_DEPENDENCY}" ]]; then
  echo "  flux_dependency=${FLUX_DEPENDENCY}"
fi

flux batch "${FLUX_ARGS[@]}" \
  scripts/retrieval/flux_eval_soloseq_baseline_submit.sh \
  -- "${EXTRA_EVAL_ARGS[@]}"

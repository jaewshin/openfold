#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a dependent Flux job chain for one retrieval pipeline/run configuration.

Usage:
  queue_retrieval_resume_pipeline.sh \
    --pipeline {a|b|c} \
    --run-id {01..07} \
    --num-jobs N \
    [--queue pbatch] \
    [--time-limit 24h] \
    [--nodes 1] [--tasks 1] [--cores 8] [--gpus 1] \
    [--max-time 00:23:50:00] \
    [--output-dir /path/to/shared/output] \
    [--resume-mode auto|none|path] \
    [--resume-ckpt-path /path/to.ckpt] \
    [--wandb-resume allow|must|never|auto] \
    [-- <extra train args>]

Notes:
  - Uses wrappers from experiments/retrieval_flux/run_pipeline_<p>_<id>_*.sh.
  - CONTINUE_TRAINING=1 is enabled for every submitted job.
  - Jobs 2..N depend on completion of previous job (afterany dependency).
  - Resume defaults to auto and shared output dir defaults to enabled.

Examples:
  bash experiments/retrieval_flux/queue_retrieval_resume_pipeline.sh \
    --pipeline b --run-id 05 --num-jobs 5 \
    --queue pbatch --time-limit 24h \
    --max-time 00:23:50:00
USAGE
}

PIPELINE=""
RUN_ID=""
NUM_JOBS=""
QUEUE="${FLUX_QUEUE:-}"
TIME_LIMIT="${FLUX_TIME_LIMIT:-24h}"
NODES="${FLUX_NODES:-1}"
TASKS="${FLUX_TASKS:-1}"
CORES="${FLUX_CORES:-8}"
GPUS="${FLUX_GPUS:-1}"
MAX_TIME="${MAX_TIME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
EXTRA_TRAIN_ARGS=()

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
    --queue|-q)
      QUEUE="${2:-}"
      shift 2
      ;;
    --time-limit|-t)
      TIME_LIMIT="${2:-}"
      shift 2
      ;;
    --nodes)
      NODES="${2:-}"
      shift 2
      ;;
    --tasks)
      TASKS="${2:-}"
      shift 2
      ;;
    --cores)
      CORES="${2:-}"
      shift 2
      ;;
    --gpus)
      GPUS="${2:-}"
      shift 2
      ;;
    --max-time)
      MAX_TIME="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --resume-mode)
      RESUME_MODE="${2:-}"
      shift 2
      ;;
    --resume-ckpt-path)
      RESUME_CKPT_PATH="${2:-}"
      shift 2
      ;;
    --wandb-resume)
      WANDB_RESUME="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PIPELINE}" || -z "${RUN_ID}" || -z "${NUM_JOBS}" ]]; then
  echo "Error: --pipeline, --run-id, and --num-jobs are required." >&2
  usage
  exit 1
fi
if ! [[ "${PIPELINE}" =~ ^[abc]$ ]]; then
  echo "Error: --pipeline must be one of a, b, c." >&2
  exit 1
fi
if ! [[ "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
  echo "Error: --num-jobs must be >= 1." >&2
  exit 1
fi
if ! [[ "${RUN_ID}" =~ ^[0-9]+$ ]] || [[ "${RUN_ID}" -lt 1 ]] || [[ "${RUN_ID}" -gt 7 ]]; then
  echo "Error: --run-id must be in [1, 7]." >&2
  exit 1
fi

RUN_ID_PADDED="$(printf '%02d' "${RUN_ID}")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
matches=( "${SCRIPT_DIR}"/run_pipeline_"${PIPELINE}"_"${RUN_ID_PADDED}"_*.sh )
if [[ "${#matches[@]}" -ne 1 ]] || [[ ! -f "${matches[0]}" ]]; then
  echo "Error: expected exactly one wrapper for pipeline=${PIPELINE} run_id=${RUN_ID_PADDED}." >&2
  printf 'Matches:\n' >&2
  printf '  %s\n' "${matches[@]}" >&2
  exit 1
fi
WRAPPER_SCRIPT="${matches[0]}"

previous_jobid=""
submitted_jobids=()

for ((i = 1; i <= NUM_JOBS; i++)); do
  dependency=""
  if [[ -n "${previous_jobid}" ]]; then
    dependency="afterany:${previous_jobid}"
  fi

  echo "Submitting chain job ${i}/${NUM_JOBS}"
  echo "  wrapper=${WRAPPER_SCRIPT}"
  if [[ -n "${dependency}" ]]; then
    echo "  dependency=${dependency}"
  fi

  submit_output="$(
    FLUX_QUEUE="${QUEUE}" \
    FLUX_TIME_LIMIT="${TIME_LIMIT}" \
    FLUX_NODES="${NODES}" \
    FLUX_TASKS="${TASKS}" \
    FLUX_CORES="${CORES}" \
    FLUX_GPUS="${GPUS}" \
    FLUX_DEPENDENCY="${dependency}" \
    CONTINUE_TRAINING=1 \
    RESUME_MODE="${RESUME_MODE}" \
    RESUME_CKPT_PATH="${RESUME_CKPT_PATH}" \
    WANDB_RESUME="${WANDB_RESUME}" \
    MAX_TIME="${MAX_TIME}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${WRAPPER_SCRIPT}" "${EXTRA_TRAIN_ARGS[@]}" 2>&1
  )"
  printf '%s\n' "${submit_output}"

  jobid="$(printf '%s\n' "${submit_output}" | awk '/^[A-Za-z0-9]+$/{id=$0} END{print id}')"
  if [[ -z "${jobid}" ]]; then
    echo "Error: failed to parse Flux job id from submission output." >&2
    exit 1
  fi

  previous_jobid="${jobid}"
  submitted_jobids+=("${jobid}")
done

echo "Submitted ${#submitted_jobids[@]} jobs:"
printf '  %s\n' "${submitted_jobids[@]}"

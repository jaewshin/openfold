#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit a short real-training chained run to validate:
  1) dependency chaining
  2) checkpoint auto-resume
  3) W&B run-id resume

Usage:
  bash experiments/retrieval_flux/run_short_chain_smoke.sh [options]

Options:
  --pipeline {a|b|c}         Pipeline key (default: b)
  --run-id {1..7}            Run id within pipeline (default: 1)
  --num-jobs N               Number of chained jobs (default: 3)
  --queue QUEUE              Flux queue (default: pdebug)
  --time-limit TIME          Flux walltime per job (default: 30m)
  --nodes N                  Flux nodes (default: 1)
  --tasks N                  Flux tasks (default: 1)
  --cores N                  Flux cores (default: 8)
  --gpus N                   Flux gpus (default: 1)
  --max-time DD:HH:MM:SS     Trainer max_time per job (default: 00:00:03:00)
  --output-dir PATH          Shared output dir (default: logs/retrieval_runs/smoke_chain_<timestamp>)
  --wandb-project NAME       W&B project (default: openfold-retrieval)
  --wandb-entity NAME        W&B entity (default: WANDB_ENTITY env or jshin)
  --wandb-resume MODE        W&B resume mode (default: allow)
  --checkpoint-every N       Checkpoint every N train steps (default: 1)
  --accumulate-grad-batches N
                             Gradient accumulation (default: 1 for fast checkpointing)
  --val-check-interval N     Validation interval (default: 1000)
  --wait-for-ready {0|1}     Wait for data readiness (default: 0)
  -h, --help                 Show help

Required:
  WANDB_API_KEY must be set in your environment.
USAGE
}

PIPELINE="b"
RUN_ID="1"
NUM_JOBS="3"
QUEUE="${FLUX_QUEUE:-pdebug}"
TIME_LIMIT="${FLUX_TIME_LIMIT:-30m}"
NODES="${FLUX_NODES:-1}"
TASKS="${FLUX_TASKS:-1}"
CORES="${FLUX_CORES:-8}"
GPUS="${FLUX_GPUS:-1}"
MAX_TIME="${MAX_TIME:-00:00:03:00}"
OUTPUT_DIR=""
WANDB_PROJECT="${WANDB_PROJECT:-openfold-retrieval}"
WANDB_ENTITY_VAL="${WANDB_ENTITY:-jshin}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-1}"
ACCUMULATE_GRAD_BATCHES_VAL="${ACCUMULATE_GRAD_BATCHES:-1}"
VAL_CHECK_INTERVAL_VAL="${VAL_CHECK_INTERVAL:-1000}"
WAIT_FOR_READY_VAL="${WAIT_FOR_READY:-0}"

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
    --wandb-project)
      WANDB_PROJECT="${2:-}"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY_VAL="${2:-}"
      shift 2
      ;;
    --wandb-resume)
      WANDB_RESUME="${2:-}"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="${2:-}"
      shift 2
      ;;
    --accumulate-grad-batches)
      ACCUMULATE_GRAD_BATCHES_VAL="${2:-}"
      shift 2
      ;;
    --val-check-interval)
      VAL_CHECK_INTERVAL_VAL="${2:-}"
      shift 2
      ;;
    --wait-for-ready)
      WAIT_FOR_READY_VAL="${2:-}"
      shift 2
      ;;
    -h|--help)
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

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set." >&2
  exit 1
fi
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
if ! [[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || [[ "${CHECKPOINT_EVERY}" -lt 1 ]]; then
  echo "Error: --checkpoint-every must be >= 1." >&2
  exit 1
fi
if ! [[ "${ACCUMULATE_GRAD_BATCHES_VAL}" =~ ^[0-9]+$ ]] || [[ "${ACCUMULATE_GRAD_BATCHES_VAL}" -lt 1 ]]; then
  echo "Error: --accumulate-grad-batches must be >= 1." >&2
  exit 1
fi
if ! [[ "${VAL_CHECK_INTERVAL_VAL}" =~ ^[0-9]+$ ]] || [[ "${VAL_CHECK_INTERVAL_VAL}" -lt 1 ]]; then
  echo "Error: --val-check-interval must be >= 1." >&2
  exit 1
fi
if [[ "${WAIT_FOR_READY_VAL}" != "0" && "${WAIT_FOR_READY_VAL}" != "1" ]]; then
  echo "Error: --wait-for-ready must be 0 or 1." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUEUE_SCRIPT="${SCRIPT_DIR}/queue_retrieval_resume_pipeline.sh"
cd "${REPO_ROOT}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${REPO_ROOT}/logs/retrieval_runs/smoke_chain_${PIPELINE}${RUN_ID}_${STAMP}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Submitting short chained smoke run:"
echo "  pipeline=${PIPELINE} run_id=${RUN_ID} num_jobs=${NUM_JOBS}"
echo "  queue=${QUEUE} time_limit=${TIME_LIMIT} max_time=${MAX_TIME}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  wandb_project=${WANDB_PROJECT} wandb_entity=${WANDB_ENTITY_VAL} wandb_resume=${WANDB_RESUME}"
echo "  checkpoint_every=${CHECKPOINT_EVERY} accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES_VAL}"

set +e
submit_output="$(
  USE_WANDB=1 \
  WANDB_API_KEY="${WANDB_API_KEY}" \
  WANDB_ENTITY="${WANDB_ENTITY_VAL}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_RESUME="${WANDB_RESUME}" \
  WAIT_FOR_READY="${WAIT_FOR_READY_VAL}" \
  CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY}" \
  ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES_VAL}" \
  VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL_VAL}" \
  LOG_DIR="${REPO_ROOT}/logs" \
  bash "${QUEUE_SCRIPT}" \
    --pipeline "${PIPELINE}" \
    --run-id "${RUN_ID}" \
    --num-jobs "${NUM_JOBS}" \
    --queue "${QUEUE}" \
    --time-limit "${TIME_LIMIT}" \
    --nodes "${NODES}" \
    --tasks "${TASKS}" \
    --cores "${CORES}" \
    --gpus "${GPUS}" \
    --max-time "${MAX_TIME}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume-mode auto \
    --wandb-resume "${WANDB_RESUME}" 2>&1
)"
submit_rc=$?
set -e

printf '%s\n' "${submit_output}"
if [[ "${submit_rc}" -ne 0 ]]; then
  echo "Submission failed." >&2
  exit "${submit_rc}"
fi

mapfile -t JOB_IDS < <(printf '%s\n' "${submit_output}" | awk '/^f[[:alnum:]]+$/{print}')
if [[ "${#JOB_IDS[@]}" -eq 0 ]]; then
  echo "Warning: unable to parse job IDs from submission output." >&2
  exit 1
fi

echo
echo "Submitted chained jobs:"
printf '  %s\n' "${JOB_IDS[@]}"

echo
echo "Sanity-check commands:"
echo "  flux jobs -a | egrep '$(IFS='|'; echo "${JOB_IDS[*]}")'"
echo "  ls -lah ${OUTPUT_DIR}/checkpoints"
for jid in "${JOB_IDS[@]}"; do
  echo "  grep -E 'Starting training from scratch|Resuming training from checkpoint|wandb_run_id:' logs/retrieval_train_${jid}.log"
done

echo
echo "Expected behavior:"
echo "  job1 log: 'Starting training from scratch'"
echo "  job2/job3 logs: 'Resuming training from checkpoint'"
echo "  all logs: same 'wandb_run_id:' value"

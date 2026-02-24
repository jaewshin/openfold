#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit all retrieval matrix runs as 3-job dependency chains on pbatch.

This wrapper submits:
  - all pipelines/runs (includes a01)
  - num_jobs=3 per run
  - queue=pbatch
  - time_limit=24h per job
  - trainer max_time=00:23:50:00 per job

Usage:
  bash experiments/retrieval_flux/submit_all_3day_pbatch.sh [--dry-run]

Required:
  WANDB_API_KEY must be set in the environment.

Optional env vars:
  WANDB_ENTITY   (default: jshin)
  WANDB_PROJECT  (default: openfold-retrieval)
  WANDB_RESUME   (default: allow)
  FLUX_NODES     (default: 1)
  FLUX_TASKS     (default: 1)
  FLUX_CORES     (default: 8)
  FLUX_GPUS      (default: 1)
USAGE
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ $# -gt 0 ]]; then
  echo "Error: unknown argument: $1" >&2
  usage
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WANDB_ENTITY_VAL="${WANDB_ENTITY:-jshin}"
WANDB_PROJECT_VAL="${WANDB_PROJECT:-openfold-retrieval}"
WANDB_RESUME_VAL="${WANDB_RESUME:-allow}"

FLUX_NODES_VAL="${FLUX_NODES:-1}"
FLUX_TASKS_VAL="${FLUX_TASKS:-1}"
FLUX_CORES_VAL="${FLUX_CORES:-8}"
FLUX_GPUS_VAL="${FLUX_GPUS:-1}"

cmd=(
  bash "${SCRIPT_DIR}/submit_remaining_with_wandb.sh"
  --include-a01
  --num-jobs 3
  --queue pbatch
  --time-limit 24h
  --max-time 00:23:50:00
  --nodes "${FLUX_NODES_VAL}"
  --tasks "${FLUX_TASKS_VAL}"
  --cores "${FLUX_CORES_VAL}"
  --gpus "${FLUX_GPUS_VAL}"
  --project "${WANDB_PROJECT_VAL}"
  --entity "${WANDB_ENTITY_VAL}"
  --wandb-resume "${WANDB_RESUME_VAL}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

echo "Submitting 3-day chained jobs on pbatch:"
echo "  num_jobs=3 queue=pbatch time_limit=24h max_time=00:23:50:00"
echo "  wandb_project=${WANDB_PROJECT_VAL} wandb_entity=${WANDB_ENTITY_VAL} wandb_resume=${WANDB_RESUME_VAL}"
echo "  resources nodes=${FLUX_NODES_VAL} tasks=${FLUX_TASKS_VAL} cores=${FLUX_CORES_VAL} gpus=${FLUX_GPUS_VAL}"

"${cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit the remaining retrieval matrix jobs with W&B enabled.

Default behavior:
- Submits all wrappers under experiments/retrieval_flux/run_pipeline_*.sh
- Skips pipeline a run 01 (assumes that sample run was already submitted)
- Enables W&B for each submission
- Writes submission summary TSV to logs/

Usage:
  bash experiments/retrieval_flux/submit_remaining_with_wandb.sh [options]

Options:
  --include-a01              Also submit run_pipeline_a_01_* (default: skip)
  --num-jobs N               Dependent jobs per run (default: 1)
  --queue QUEUE              Flux queue (default: pbatch)
  --time-limit TIME          Flux walltime (default: 24h)
  --nodes N                  Flux nodes (default: 1)
  --tasks N                  Flux tasks (default: 1)
  --cores N                  Flux cores (default: 8)
  --gpus N                   Flux GPUs (default: 1)
  --max-time HH:MM:SS:SS     Trainer max_time (default: 00:23:50:00)
  --resume-mode MODE         Resume mode for chained runs (default: auto)
  --project NAME             W&B project (default: openfold-retrieval)
  --entity NAME              W&B entity (default: WANDB_ENTITY env or jshin)
  --wandb-resume MODE        W&B resume mode (default: allow)
  --dry-run                  Print wrappers that would be submitted
  -h, --help                 Show help

Required:
  WANDB_API_KEY must be set in environment.
USAGE
}

INCLUDE_A01=0
NUM_JOBS="${NUM_JOBS:-1}"
QUEUE="${FLUX_QUEUE:-pbatch}"
TIME_LIMIT="${FLUX_TIME_LIMIT:-24h}"
NODES="${FLUX_NODES:-1}"
TASKS="${FLUX_TASKS:-1}"
CORES="${FLUX_CORES:-8}"
GPUS="${FLUX_GPUS:-1}"
MAX_TIME="${MAX_TIME:-00:23:50:00}"
RESUME_MODE="${RESUME_MODE:-auto}"
WANDB_PROJECT="${WANDB_PROJECT:-openfold-retrieval}"
WANDB_ENTITY_VAL="${WANDB_ENTITY:-jshin}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-a01)
      INCLUDE_A01=1
      shift
      ;;
    --num-jobs)
      NUM_JOBS="${2:-}"
      shift 2
      ;;
    --queue)
      QUEUE="${2:-}"
      shift 2
      ;;
    --time-limit)
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
    --resume-mode)
      RESUME_MODE="${2:-}"
      shift 2
      ;;
    --project)
      WANDB_PROJECT="${2:-}"
      shift 2
      ;;
    --entity)
      WANDB_ENTITY_VAL="${2:-}"
      shift 2
      ;;
    --wandb-resume)
      WANDB_RESUME="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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
  echo "Set it first, then rerun." >&2
  exit 1
fi
if ! [[ "${NUM_JOBS}" =~ ^[0-9]+$ ]] || [[ "${NUM_JOBS}" -lt 1 ]]; then
  echo "Error: --num-jobs must be >= 1." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TSV="logs/submitted_retrieval_matrix_wandb_${STAMP}.tsv"
printf "wrapper\trun_tag\tjobids\tstatus\n" > "${OUT_TSV}"

mapfile -t WRAPPERS < <(ls -1 "${SCRIPT_DIR}"/run_pipeline_*.sh | sort)

if [[ "${INCLUDE_A01}" != "1" ]]; then
  FILTERED=()
  for w in "${WRAPPERS[@]}"; do
    if [[ "$(basename "${w}")" == run_pipeline_a_01_* ]]; then
      continue
    fi
    FILTERED+=("${w}")
  done
  WRAPPERS=("${FILTERED[@]}")
fi

if [[ "${#WRAPPERS[@]}" -eq 0 ]]; then
  echo "No wrappers selected." >&2
  exit 1
fi

echo "Submitting ${#WRAPPERS[@]} wrapper(s) with W&B enabled..."
echo "  queue=${QUEUE} time_limit=${TIME_LIMIT} max_time=${MAX_TIME} num_jobs=${NUM_JOBS}"
echo "  resume_mode=${RESUME_MODE}"
echo "  wandb_project=${WANDB_PROJECT} wandb_entity=${WANDB_ENTITY_VAL} wandb_resume=${WANDB_RESUME}"
echo "  output_tsv=${OUT_TSV}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "DRY RUN wrappers:\n"
  printf "  %s\n" "${WRAPPERS[@]}"
  exit 0
fi

fail_count=0

for wrapper in "${WRAPPERS[@]}"; do
  wrapper_base="$(basename "${wrapper}")"
  run_tag="$(
    awk 'match($0, /--run-tag "([^"]+)"/, m) { print m[1]; exit }' "${wrapper}"
  )"

  echo
  echo "Submitting: ${wrapper_base}"
  echo "  run_tag=${run_tag}"
  if [[ "${NUM_JOBS}" -gt 1 ]]; then
    echo "  mode=chained (${NUM_JOBS} jobs)"
  fi

  submit_ok=1
  submit_output=""
  jobids_csv=""

  if [[ "${NUM_JOBS}" -gt 1 ]]; then
    if [[ "${wrapper_base}" =~ ^run_pipeline_([abc])_([0-9]{2})_ ]]; then
      pipeline="${BASH_REMATCH[1]}"
      run_id="${BASH_REMATCH[2]}"
      if submit_output="$(
        FLUX_QUEUE="${QUEUE}" \
        FLUX_TIME_LIMIT="${TIME_LIMIT}" \
        FLUX_NODES="${NODES}" \
        FLUX_TASKS="${TASKS}" \
        FLUX_CORES="${CORES}" \
        FLUX_GPUS="${GPUS}" \
        MAX_TIME="${MAX_TIME}" \
        USE_WANDB=1 \
        WANDB_API_KEY="${WANDB_API_KEY}" \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        WANDB_ENTITY="${WANDB_ENTITY_VAL}" \
        WANDB_RESUME="${WANDB_RESUME}" \
        bash "${SCRIPT_DIR}/queue_retrieval_resume_pipeline.sh" \
          --pipeline "${pipeline}" \
          --run-id "${run_id}" \
          --num-jobs "${NUM_JOBS}" \
          --queue "${QUEUE}" \
          --time-limit "${TIME_LIMIT}" \
          --nodes "${NODES}" \
          --tasks "${TASKS}" \
          --cores "${CORES}" \
          --gpus "${GPUS}" \
          --max-time "${MAX_TIME}" \
          --resume-mode "${RESUME_MODE}" \
          --wandb-resume "${WANDB_RESUME}" 2>&1
      )"; then
        :
      else
        submit_ok=0
      fi
    else
      submit_ok=0
      submit_output="Error: could not parse pipeline/run-id from ${wrapper_base}"
    fi
  else
    if submit_output="$(
      FLUX_QUEUE="${QUEUE}" \
      FLUX_TIME_LIMIT="${TIME_LIMIT}" \
      FLUX_NODES="${NODES}" \
      FLUX_TASKS="${TASKS}" \
      FLUX_CORES="${CORES}" \
      FLUX_GPUS="${GPUS}" \
      MAX_TIME="${MAX_TIME}" \
      USE_WANDB=1 \
      WANDB_API_KEY="${WANDB_API_KEY}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      WANDB_ENTITY="${WANDB_ENTITY_VAL}" \
      WANDB_RESUME="${WANDB_RESUME}" \
      bash "${wrapper}" 2>&1
    )"; then
      :
    else
      submit_ok=0
    fi
  fi

  if [[ "${submit_ok}" == "1" ]]; then
    mapfile -t jobids < <(printf '%s\n' "${submit_output}" | awk '/^f[[:alnum:]]+$/{print}')
    if [[ "${#jobids[@]}" -gt 0 ]]; then
      jobids_csv="$(IFS=,; echo "${jobids[*]}")"
      status="submitted"
      echo "  jobids=${jobids_csv}"
    else
      status="failed_parse_jobid"
      echo "  failed: unable to parse job id(s) from output"
      fail_count=$((fail_count + 1))
    fi
  else
    status="failed_submit"
    echo "  submission command failed"
    echo "  details: ${submit_output}"
    fail_count=$((fail_count + 1))
  fi

  printf "%s\t%s\t%s\t%s\n" "${wrapper}" "${run_tag}" "${jobids_csv:-}" "${status}" >> "${OUT_TSV}"
done

echo
echo "Done. Submission summary: ${OUT_TSV}"
if [[ "${fail_count}" -gt 0 ]]; then
  echo "Failed submissions: ${fail_count}" >&2
  exit 1
fi

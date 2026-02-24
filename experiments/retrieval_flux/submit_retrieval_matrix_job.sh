#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Submit one retrieval training run to Flux.

Usage:
  submit_retrieval_matrix_job.sh \
    --pipeline {a|b|c} \
    --run-tag TAG \
    [--train-flags "--train_evoformer ..."] \
    [-- <extra train args>]

Pipeline mapping:
  a -> legacy
  b -> embed_project
  c -> rawseq_esm1b

Environment overrides (optional):
  REPO_ROOT=/path/to/openfold
  DATA_ROOT=<repo_root>/experiments/data
  READY_DIR=<data_root>/retrieval_ready_mmseqs
  SEQ_EMB_DIR=<ready_dir>/seq_embedding_esm1b

  FLUX_NODES=1
  FLUX_TASKS=1
  FLUX_CORES=8
  FLUX_GPUS=1
  FLUX_TIME_LIMIT=2d
  FLUX_QUEUE=""      # set to queue name if needed
  FLUX_DEPENDENCY="" # e.g. afterany:<jobid>

  USE_WANDB=0         # set to 1 to enable W&B in train submit script
  WANDB_PROJECT=openfold-retrieval
  WANDB_ENTITY=""
  WANDB_TAGS=""
  WANDB_RESUME=allow

  # Continuation helpers (passed through to flux_train_retrieval_pipeline_submit.sh):
  CONTINUE_TRAINING=0 # 1 -> SHARED_OUTPUT_DIR=1 and RESUME_MODE=auto by default
USAGE
}

PIPELINE=""
RUN_TAG=""
TRAIN_FLAGS=""
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline)
      PIPELINE="${2:-}"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="${2:-}"
      shift 2
      ;;
    --train-flags)
      TRAIN_FLAGS="${2:-}"
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

if [[ -z "${PIPELINE}" || -z "${RUN_TAG}" ]]; then
  echo "Error: --pipeline and --run-tag are required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

case "${PIPELINE}" in
  a)
    RETRIEVAL_PIPELINE="legacy"
    PIPELINE_LABEL="legacy"
    ;;
  b)
    RETRIEVAL_PIPELINE="embed_project"
    PIPELINE_LABEL="post_input_pre_evoformer"
    ;;
  c)
    RETRIEVAL_PIPELINE="rawseq_esm1b"
    PIPELINE_LABEL="post_evoformer_pre_structure"
    ;;
  *)
    echo "Error: unsupported --pipeline '${PIPELINE}'. Use a|b|c." >&2
    exit 1
    ;;
esac

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/experiments/data}"
READY_DIR="${READY_DIR:-${DATA_ROOT}/retrieval_ready_mmseqs}"
SEQ_EMB_DIR="${SEQ_EMB_DIR:-${READY_DIR}/seq_embedding_esm1b}"

FLUX_NODES="${FLUX_NODES:-1}"
FLUX_TASKS="${FLUX_TASKS:-1}"
FLUX_CORES="${FLUX_CORES:-8}"
FLUX_GPUS="${FLUX_GPUS:-1}"
FLUX_TIME_LIMIT="${FLUX_TIME_LIMIT:-2d}"
FLUX_QUEUE="${FLUX_QUEUE:-}"
FLUX_DEPENDENCY="${FLUX_DEPENDENCY:-}"

CONTINUE_TRAINING="${CONTINUE_TRAINING:-0}"
if [[ "${CONTINUE_TRAINING}" == "1" ]]; then
  RESUME_MODE="${RESUME_MODE:-auto}"
  SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-1}"
fi

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-openfold-retrieval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_TAGS_DEFAULT="retrieval,openproteinnet,${PIPELINE_LABEL},${RUN_TAG}"
WANDB_TAGS="${WANDB_TAGS:-${WANDB_TAGS_DEFAULT}}"
WANDB_RESUME="${WANDB_RESUME:-allow}"

TRAIN_FLAGS_ARR=()
if [[ -n "${TRAIN_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_FLAGS_ARR=(${TRAIN_FLAGS})
fi

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

echo "Submitting ${RUN_TAG}"
echo "  pipeline_key=${PIPELINE} (${RETRIEVAL_PIPELINE})"
echo "  data_root=${DATA_ROOT}"
echo "  ready_dir=${READY_DIR}"
echo "  seq_emb_dir=${SEQ_EMB_DIR}"
echo "  train_flags=${TRAIN_FLAGS}"
if [[ -n "${FLUX_DEPENDENCY}" ]]; then
  echo "  flux_dependency=${FLUX_DEPENDENCY}"
fi

env \
  DATA_ROOT="${DATA_ROOT}" \
  READY_DIR="${READY_DIR}" \
  SEQ_EMB_DIR="${SEQ_EMB_DIR}" \
  RETRIEVAL_PIPELINE="${RETRIEVAL_PIPELINE}" \
  RESUME_MODE="${RESUME_MODE:-}" \
  SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-}" \
  USE_WANDB="${USE_WANDB}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_TAGS="${WANDB_TAGS}" \
  WANDB_RESUME="${WANDB_RESUME}" \
  flux batch "${FLUX_ARGS[@]}" \
    scripts/retrieval/flux_train_retrieval_pipeline_submit.sh \
    --run-tag "${RUN_TAG}" \
    -- "${TRAIN_FLAGS_ARR[@]}" "${EXTRA_TRAIN_ARGS[@]}"

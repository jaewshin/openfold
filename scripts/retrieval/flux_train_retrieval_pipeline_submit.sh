#!/usr/bin/env bash

# Keep nounset + pipefail, but avoid global -e so non-interactive .bashrc
# commands (e.g. stty) do not terminate the job.
set -uo pipefail

usage() {
  cat <<'EOF'
Run retrieval training pipeline under Flux.

Usage:
  flux batch [flux options] scripts/retrieval/flux_train_retrieval_pipeline_submit.sh [options] [-- <extra train args>]

Options:
  -d, --data-root PATH      Root data directory (default: <repo_root>/experiments/data)
  -r, --ready-dir PATH      Prepared dataset dir (default: <data-root>/retrieval_ready_mmseqs)
  -e, --seq-emb-dir PATH    Sequence embedding dir (default: <ready-dir>/seq_embedding_esm1b)
  -o, --output-dir PATH     Trainer output directory (default depends on SHARED_OUTPUT_DIR / RESUME_MODE)
      --run-tag TAG         Run label used in output/log naming (default: retrieval_opn_flux)
  -h, --help                Show this message

Optional env vars (defaults shown):
  CONDA_ENV=openfold_dev
  ROCM_VERSION=6.4.2
  ROCM_VERSION_DIR=rocm-6.4.2
  USE_ROCM_MODULES=auto

  RETRIEVAL_PIPELINE=embed_project   # embed_project | rawseq_esm1b | legacy
  RETRIEVAL_ABLATION=both            # both | seq_only | struct_only
  DATA_INPUT_MODE=auto               # auto | manifest | dataset
  MANIFEST_PATH=<ready-dir>/manifest.jsonl
  TOP_K=8
  NPROBE=64
  LR=5e-5
  BATCH_SIZE=1
  ACCUMULATE_GRAD_BATCHES=128
  VAL_CHECK_INTERVAL=1000
  NUM_WORKERS=0
  MAX_EPOCHS=1
  DEVICES=1
  PRECISION=32
  MAX_TIME=""                         # e.g. 00:23:50:00 (graceful stop before walltime)
  OPENFOLD_TRAIN_FLAGS=""            # e.g. "--train_evoformer --train_structure_module"
  STRICT_SEQ_EMBEDDINGS=1            # pass --strict_seq_embeddings

  # Checkpoint/resume controls:
  CHECKPOINT_DIR=""                  # default: <output_dir>/checkpoints
  CHECKPOINT_EVERY_N_TRAIN_STEPS=1000
  SAVE_LAST_CHECKPOINT=1
  RESUME_MODE=none                   # none | auto | path
  RESUME_CKPT_PATH=""                # required when RESUME_MODE=path
  SHARED_OUTPUT_DIR=""               # auto: 1 when RESUME_MODE!=none, else 0

  RETRIEVER_ESM2_MODEL_NAME=esm2_t12_35M_UR50D
  RETRIEVER_ESM2_REPR_LAYER=12
  RETRIEVER_ESM2_MAX_LEN=1022
  RETRIEVER_TMVEC_CHECKPOINT=<repo_root>/tmvec-bench/binaries/tmvec2_student.pt
  RETRIEVER_TMVEC_MAX_LEN=1022
  RETRIEVER_NORMALIZE_QUERIES=1

  SEQ_INDEX_PATH=<data-root>/u50_esm2_35M.index
  STRUCT_INDEX_PATH=<data-root>/u50_tmvec_2s.index
  SEQ_INDEX_DIM=480
  STRUCT_INDEX_DIM=512

  # rawseq_esm1b only:
  SEQ_INDEX_IDS_PATH=<data-root>/u50_esm2_35M_ids.txt
  STRUCT_INDEX_IDS_PATH=<data-root>/u50_tmvec_2s_ids.txt
  SEQ_DB_FASTA_PATH=<data-root>/uniref50.fasta
  STRUCT_DB_FASTA_PATH=<data-root>/uniref50.fasta
  SEQ_DB_FASTA_INDEX_DB=<data-root>/uniref50.seqio.sqlite
  STRUCT_DB_FASTA_INDEX_DB=<data-root>/uniref50.seqio.sqlite
  RETRIEVED_ESM1B_DEVICE=cpu

  WAIT_FOR_READY=1
  WAIT_INTERVAL_SEC=60
  WAIT_TIMEOUT_SEC=0                 # 0 means wait indefinitely

  USE_WANDB=0
  WANDB_PROJECT=openfold-retrieval
  WANDB_ENTITY=""
  WANDB_RUN_NAME=""
  WANDB_RUN_ID=""
  WANDB_RESUME=allow
  WANDB_TAGS=""

  LOG_DIR=<repo_root>/logs
  DRY_RUN=0

Example:
  flux batch -N1 -n1 -c16 -g1 -t24h -q pbatch \
    scripts/retrieval/flux_train_retrieval_pipeline_submit.sh \
    --run-tag retrieval_opn_pa01 \
    -- --train_evoformer
EOF
}

DATA_ROOT=""
READY_DIR=""
SEQ_EMB_DIR=""
OUTPUT_DIR=""
RUN_TAG="${RUN_TAG:-retrieval_opn_flux}"
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--data-root)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      DATA_ROOT="$2"
      shift 2
      ;;
    -r|--ready-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      READY_DIR="$2"
      shift 2
      ;;
    -e|--seq-emb-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      SEQ_EMB_DIR="$2"
      shift 2
      ;;
    -o|--output-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-tag)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      RUN_TAG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif [[ -f "$(pwd)/scripts/retrieval/train_retrieval_lightning.py" ]]; then
  # For flux batch, cwd is usually the submit directory.
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}" || {
  echo "Error: failed to cd to repo root '${REPO_ROOT}'." >&2
  exit 1
}

if [[ ! -f "${REPO_ROOT}/scripts/retrieval/train_retrieval_lightning.py" ]]; then
  echo "Error: could not locate repository root at '${REPO_ROOT}'." >&2
  echo "Hint: run flux batch from repo root or set REPO_ROOT=/path/to/openfold." >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-openfold_dev}"
ROCM_VERSION="${ROCM_VERSION:-6.4.2}"
ROCM_VERSION_DIR="${ROCM_VERSION_DIR:-rocm-${ROCM_VERSION}}"
USE_ROCM_MODULES="${USE_ROCM_MODULES:-auto}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/experiments/data}"
READY_DIR="${READY_DIR:-${DATA_ROOT}/retrieval_ready_mmseqs}"
SEQ_EMB_DIR="${SEQ_EMB_DIR:-${READY_DIR}/seq_embedding_esm1b}"

SEQ_INDEX_PATH="${SEQ_INDEX_PATH:-${DATA_ROOT}/u50_esm2_35M.index}"
STRUCT_INDEX_PATH="${STRUCT_INDEX_PATH:-${DATA_ROOT}/u50_tmvec_2s.index}"
SEQ_INDEX_DIM="${SEQ_INDEX_DIM:-480}"
STRUCT_INDEX_DIM="${STRUCT_INDEX_DIM:-512}"

RETRIEVAL_PIPELINE="${RETRIEVAL_PIPELINE:-embed_project}"
RETRIEVAL_ABLATION="${RETRIEVAL_ABLATION:-both}"
DATA_INPUT_MODE="${DATA_INPUT_MODE:-auto}"
MANIFEST_PATH="${MANIFEST_PATH:-${READY_DIR}/manifest.jsonl}"
TOP_K="${TOP_K:-8}"
NPROBE="${NPROBE:-64}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-128}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
MAX_TIME="${MAX_TIME:-}"
OPENFOLD_TRAIN_FLAGS="${OPENFOLD_TRAIN_FLAGS:-}"
STRICT_SEQ_EMBEDDINGS="${STRICT_SEQ_EMBEDDINGS:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_EVERY_N_TRAIN_STEPS="${CHECKPOINT_EVERY_N_TRAIN_STEPS:-1000}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-1}"
RESUME_MODE="${RESUME_MODE:-none}"
RESUME_CKPT_PATH="${RESUME_CKPT_PATH:-}"
SHARED_OUTPUT_DIR="${SHARED_OUTPUT_DIR:-}"

RETRIEVER_ESM2_MODEL_NAME="${RETRIEVER_ESM2_MODEL_NAME:-esm2_t12_35M_UR50D}"
RETRIEVER_ESM2_REPR_LAYER="${RETRIEVER_ESM2_REPR_LAYER:-12}"
RETRIEVER_ESM2_MAX_LEN="${RETRIEVER_ESM2_MAX_LEN:-1022}"
RETRIEVER_TMVEC_CHECKPOINT="${RETRIEVER_TMVEC_CHECKPOINT:-${REPO_ROOT}/tmvec-bench/binaries/tmvec2_student.pt}"
RETRIEVER_TMVEC_MAX_LEN="${RETRIEVER_TMVEC_MAX_LEN:-1022}"
RETRIEVER_NORMALIZE_QUERIES="${RETRIEVER_NORMALIZE_QUERIES:-1}"

SEQ_INDEX_IDS_PATH="${SEQ_INDEX_IDS_PATH:-${DATA_ROOT}/u50_esm2_35M_ids.txt}"
STRUCT_INDEX_IDS_PATH="${STRUCT_INDEX_IDS_PATH:-${DATA_ROOT}/u50_tmvec_2s_ids.txt}"
SEQ_DB_FASTA_PATH="${SEQ_DB_FASTA_PATH:-${DATA_ROOT}/uniref50.fasta}"
STRUCT_DB_FASTA_PATH="${STRUCT_DB_FASTA_PATH:-${DATA_ROOT}/uniref50.fasta}"
SEQ_DB_FASTA_INDEX_DB="${SEQ_DB_FASTA_INDEX_DB:-${DATA_ROOT}/uniref50.seqio.sqlite}"
STRUCT_DB_FASTA_INDEX_DB="${STRUCT_DB_FASTA_INDEX_DB:-${DATA_ROOT}/uniref50.seqio.sqlite}"
RETRIEVED_ESM1B_DEVICE="${RETRIEVED_ESM1B_DEVICE:-cpu}"

WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-0}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-openfold-retrieval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_TAGS="${WANDB_TAGS:-}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${SHARED_OUTPUT_DIR}" ]]; then
  if [[ "${RESUME_MODE}" == "none" ]]; then
    SHARED_OUTPUT_DIR="0"
  else
    SHARED_OUTPUT_DIR="1"
  fi
fi

if [[ "${SHARED_OUTPUT_DIR}" != "0" && "${SHARED_OUTPUT_DIR}" != "1" ]]; then
  echo "Error: SHARED_OUTPUT_DIR must be 0 or 1 (got '${SHARED_OUTPUT_DIR}')." >&2
  exit 1
fi
if [[ "${DATA_INPUT_MODE}" != "auto" && "${DATA_INPUT_MODE}" != "manifest" && "${DATA_INPUT_MODE}" != "dataset" ]]; then
  echo "Error: DATA_INPUT_MODE must be one of {auto, manifest, dataset} (got '${DATA_INPUT_MODE}')." >&2
  exit 1
fi
if [[ "${SAVE_LAST_CHECKPOINT}" != "0" && "${SAVE_LAST_CHECKPOINT}" != "1" ]]; then
  echo "Error: SAVE_LAST_CHECKPOINT must be 0 or 1 (got '${SAVE_LAST_CHECKPOINT}')." >&2
  exit 1
fi
if ! [[ "${CHECKPOINT_EVERY_N_TRAIN_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "Error: CHECKPOINT_EVERY_N_TRAIN_STEPS must be a non-negative integer." >&2
  exit 1
fi
case "${RESUME_MODE}" in
  none|auto|path)
    ;;
  *)
    echo "Error: RESUME_MODE must be one of {none, auto, path} (got '${RESUME_MODE}')." >&2
    exit 1
    ;;
esac

if [[ "${USE_ROCM_MODULES}" == "auto" ]]; then
  if [[ "${DEVICES}" -gt 0 ]]; then
    USE_ROCM_MODULES="1"
  else
    USE_ROCM_MODULES="0"
  fi
fi

if [[ -f /etc/profile.d/z00_lmod.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/z00_lmod.sh
fi

if [[ "${USE_ROCM_MODULES}" == "1" ]]; then
  if command -v ml >/dev/null 2>&1; then
    ml rocm/"${ROCM_VERSION}" || {
      echo "Error: failed to load module rocm/${ROCM_VERSION}." >&2
      exit 1
    }
    ml craype-accel-amd-gfx942 cray-mpich libfabric || {
      echo "Error: failed to load GPU-related modules." >&2
      exit 1
    }
  fi

  if [[ -d "/opt/${ROCM_VERSION_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="/opt/${ROCM_VERSION_DIR}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

set +u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || {
    echo "Error: failed to initialize conda shell hook." >&2
    exit 1
  }
  conda activate "${CONDA_ENV}" || {
    echo "Error: failed to activate conda env '${CONDA_ENV}'." >&2
    exit 1
  }
elif command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash 2>/dev/null)" || {
    echo "Error: failed to initialize mamba shell hook." >&2
    exit 1
  }
  mamba activate "${CONDA_ENV}" || {
    echo "Error: failed to activate mamba env '${CONDA_ENV}'." >&2
    exit 1
  }
elif [[ -f "${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}" || {
    echo "Error: failed to source conda init script." >&2
    exit 1
  }
  conda activate "${CONDA_ENV}" || {
    echo "Error: failed to activate conda env '${CONDA_ENV}'." >&2
    exit 1
  }
else
  echo "Error: could not initialize conda/mamba to activate '${CONDA_ENV}'." >&2
  exit 1
fi
set -u

check_ready_once() {
  local missing=()

  for p in \
    "${READY_DIR}/manifest.jsonl" \
    "${READY_DIR}/splits.json" \
    "${READY_DIR}/train.fasta" \
    "${READY_DIR}/val.fasta" \
    "${SEQ_INDEX_PATH}" \
    "${STRUCT_INDEX_PATH}" \
    "${RETRIEVER_TMVEC_CHECKPOINT}"
  do
    if [[ ! -e "${p}" ]]; then
      missing+=("${p}")
    fi
  done

  if [[ "${STRICT_SEQ_EMBEDDINGS}" == "1" ]]; then
    if [[ ! -f "${SEQ_EMB_DIR}/esm1b_embedding_metadata.json" ]]; then
      missing+=("${SEQ_EMB_DIR}/esm1b_embedding_metadata.json")
    fi
  fi

  if [[ "${RETRIEVAL_PIPELINE}" == "rawseq_esm1b" ]]; then
    for p in \
      "${SEQ_INDEX_IDS_PATH}" \
      "${STRUCT_INDEX_IDS_PATH}" \
      "${SEQ_DB_FASTA_PATH}" \
      "${STRUCT_DB_FASTA_PATH}" \
      "${SEQ_DB_FASTA_INDEX_DB}" \
      "${STRUCT_DB_FASTA_INDEX_DB}"
    do
      if [[ ! -e "${p}" ]]; then
        missing+=("${p}")
      fi
    done
  fi

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "Data/input readiness check: missing ${#missing[@]} paths:"
    printf '  - %s\n' "${missing[@]}"
    return 1
  fi

  return 0
}

if [[ "${WAIT_FOR_READY}" == "1" ]]; then
  start_ts="$(date +%s)"
  while ! check_ready_once; do
    if [[ "${WAIT_TIMEOUT_SEC}" != "0" ]]; then
      now_ts="$(date +%s)"
      elapsed="$((now_ts - start_ts))"
      if [[ "${elapsed}" -ge "${WAIT_TIMEOUT_SEC}" ]]; then
        echo "Error: timed out waiting for pipeline inputs (${WAIT_TIMEOUT_SEC}s)." >&2
        exit 1
      fi
    fi
    echo "Waiting ${WAIT_INTERVAL_SEC}s for required files..."
    sleep "${WAIT_INTERVAL_SEC}"
  done
else
  check_ready_once || exit 1
fi

mkdir -p "${LOG_DIR}" || {
  echo "Error: failed to create log dir '${LOG_DIR}'." >&2
  exit 1
}

JOB_ID="${FLUX_JOB_ID:-}"
if [[ -z "${JOB_ID}" ]] && command -v flux >/dev/null 2>&1; then
  JOB_ID="$(flux getattr jobid 2>/dev/null || true)"
fi
if [[ -z "${JOB_ID}" ]]; then
  JOB_ID="$(date +%Y%m%d_%H%M%S)"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${SHARED_OUTPUT_DIR}" == "1" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/logs/retrieval_runs/${RUN_TAG}"
  else
    OUTPUT_DIR="${REPO_ROOT}/logs/retrieval_runs/${RUN_TAG}/${JOB_ID}"
  fi
fi
mkdir -p "${OUTPUT_DIR}" || {
  echo "Error: failed to create output dir '${OUTPUT_DIR}'." >&2
  exit 1
}

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
fi
mkdir -p "${CHECKPOINT_DIR}" || {
  echo "Error: failed to create checkpoint dir '${CHECKPOINT_DIR}'." >&2
  exit 1
}

LOG_FILE="${LOG_DIR}/retrieval_train_${JOB_ID}.log"
if [[ -z "${WANDB_RUN_ID}" ]]; then
  if [[ "${RESUME_MODE}" == "none" && "${SHARED_OUTPUT_DIR}" == "0" ]]; then
    WANDB_RUN_ID="${RUN_TAG}-${JOB_ID}"
  else
    WANDB_RUN_ID="${RUN_TAG}"
  fi
fi
if [[ -z "${WANDB_RUN_NAME}" ]]; then
  if [[ "${RESUME_MODE}" == "none" && "${SHARED_OUTPUT_DIR}" == "0" ]]; then
    WANDB_RUN_NAME="${RUN_TAG}_${JOB_ID}"
  else
    WANDB_RUN_NAME="${RUN_TAG}"
  fi
fi

OPENFOLD_TRAIN_FLAGS_ARR=()
if [[ -n "${OPENFOLD_TRAIN_FLAGS}" ]]; then
  read -r -a OPENFOLD_TRAIN_FLAGS_ARR <<< "${OPENFOLD_TRAIN_FLAGS}"
fi

CMD=(
  python scripts/retrieval/train_retrieval_lightning.py
  --seq_embedding_dir "${SEQ_EMB_DIR}"
  --seq_index_path "${SEQ_INDEX_PATH}"
  --struct_index_path "${STRUCT_INDEX_PATH}"
  --seq_index_dim "${SEQ_INDEX_DIM}"
  --struct_index_dim "${STRUCT_INDEX_DIM}"
  --retrieval_ablation "${RETRIEVAL_ABLATION}"
  --retrieval_pipeline "${RETRIEVAL_PIPELINE}"
  --retriever_esm2_model_name "${RETRIEVER_ESM2_MODEL_NAME}"
  --retriever_esm2_repr_layer "${RETRIEVER_ESM2_REPR_LAYER}"
  --retriever_esm2_max_len "${RETRIEVER_ESM2_MAX_LEN}"
  --retriever_tmvec_checkpoint "${RETRIEVER_TMVEC_CHECKPOINT}"
  --retriever_tmvec_max_len "${RETRIEVER_TMVEC_MAX_LEN}"
  --top_k "${TOP_K}"
  --nprobe "${NPROBE}"
  --lr "${LR}"
  --batch_size "${BATCH_SIZE}"
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}"
  --val_check_interval "${VAL_CHECK_INTERVAL}"
  --num_workers "${NUM_WORKERS}"
  --max_epochs "${MAX_EPOCHS}"
  --devices "${DEVICES}"
  --precision "${PRECISION}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --checkpoint_every_n_train_steps "${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
  --output_dir "${OUTPUT_DIR}"
)

case "${DATA_INPUT_MODE}" in
  dataset)
    CMD+=(--dataset_dir "${READY_DIR}")
    ;;
  manifest)
    CMD+=(--manifest_path "${MANIFEST_PATH}")
    ;;
  auto)
    if [[ -d "${READY_DIR}/structures" ]]; then
      CMD+=(--dataset_dir "${READY_DIR}")
    else
      CMD+=(--manifest_path "${MANIFEST_PATH}")
    fi
    ;;
esac

if [[ "${RETRIEVER_NORMALIZE_QUERIES}" == "1" ]]; then
  CMD+=(--retriever_normalize_queries)
else
  CMD+=(--no-retriever_normalize_queries)
fi

if [[ "${STRICT_SEQ_EMBEDDINGS}" == "1" ]]; then
  CMD+=(--strict_seq_embeddings)
fi

if [[ "${SAVE_LAST_CHECKPOINT}" == "1" ]]; then
  CMD+=(--save_last_checkpoint)
else
  CMD+=(--no-save_last_checkpoint)
fi

if [[ -n "${MAX_TIME}" ]]; then
  CMD+=(--max_time "${MAX_TIME}")
fi

case "${RESUME_MODE}" in
  none)
    ;;
  auto)
    CMD+=(--auto_resume)
    ;;
  path)
    if [[ -z "${RESUME_CKPT_PATH}" ]]; then
      echo "Error: RESUME_MODE=path requires RESUME_CKPT_PATH." >&2
      exit 1
    fi
    CMD+=(--resume_ckpt_path "${RESUME_CKPT_PATH}")
    ;;
esac

if [[ "${RETRIEVAL_PIPELINE}" == "rawseq_esm1b" ]]; then
  CMD+=(
    --seq_index_ids_path "${SEQ_INDEX_IDS_PATH}"
    --struct_index_ids_path "${STRUCT_INDEX_IDS_PATH}"
    --seq_db_fasta_path "${SEQ_DB_FASTA_PATH}"
    --struct_db_fasta_path "${STRUCT_DB_FASTA_PATH}"
    --seq_db_fasta_index_db "${SEQ_DB_FASTA_INDEX_DB}"
    --struct_db_fasta_index_db "${STRUCT_DB_FASTA_INDEX_DB}"
    --retrieved_esm1b_device "${RETRIEVED_ESM1B_DEVICE}"
  )
fi

if [[ "${USE_WANDB}" == "1" ]]; then
  CMD+=(
    --use_wandb
    --wandb_project "${WANDB_PROJECT}"
    --wandb_resume "${WANDB_RESUME}"
    --wandb_run_name "${WANDB_RUN_NAME}"
    --wandb_run_id "${WANDB_RUN_ID}"
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  if [[ -n "${WANDB_TAGS}" ]]; then
    CMD+=(--wandb_tags "${WANDB_TAGS}")
  fi
fi

echo "Running retrieval training with:"
echo "  repo_root: ${REPO_ROOT}"
echo "  data_root: ${DATA_ROOT}"
echo "  ready_dir: ${READY_DIR}"
echo "  seq_emb_dir: ${SEQ_EMB_DIR}"
echo "  retrieval_pipeline: ${RETRIEVAL_PIPELINE}"
echo "  retrieval_ablation: ${RETRIEVAL_ABLATION}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  checkpoint_dir: ${CHECKPOINT_DIR}"
echo "  checkpoint_every_n_train_steps: ${CHECKPOINT_EVERY_N_TRAIN_STEPS}"
echo "  save_last_checkpoint: ${SAVE_LAST_CHECKPOINT}"
echo "  resume_mode: ${RESUME_MODE}"
echo "  shared_output_dir: ${SHARED_OUTPUT_DIR}"
echo "  use_wandb: ${USE_WANDB}"
echo "  wandb_run_id: ${WANDB_RUN_ID}"
echo "  strict_seq_embeddings: ${STRICT_SEQ_EMBEDDINGS}"
echo "  log_file: ${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY RUN command:\n'
  printf '  %q' "${CMD[@]}"
  for a in "${OPENFOLD_TRAIN_FLAGS_ARR[@]}"; do
    printf ' %q' "${a}"
  done
  for a in "${EXTRA_TRAIN_ARGS[@]}"; do
    printf ' %q' "${a}"
  done
  printf '\n'
  exit 0
fi

"${CMD[@]}" "${OPENFOLD_TRAIN_FLAGS_ARR[@]}" "${EXTRA_TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash

# Keep nounset + pipefail, but avoid global -e so non-interactive .bashrc
# commands (e.g. stty) do not terminate the job.
set -uo pipefail

usage() {
  cat <<'EOF'
Run vanilla OpenFold SoloSeq validation metrics under Flux (no retrieval).

Usage:
  flux batch [flux options] scripts/retrieval/flux_eval_soloseq_baseline_submit.sh [-- <extra eval args>]

Optional env vars (defaults shown):
  CONDA_ENV=openfold_dev
  ROCM_VERSION=6.4.2
  ROCM_VERSION_DIR=rocm-6.4.2
  USE_ROCM_MODULES=1

  REPO_ROOT=<auto-detected>
  MANIFEST_PATH=<repo_root>/experiments/data/retrieval_ready_mmseqs/manifest.jsonl
  SEQ_EMBED_DIR=<repo_root>/experiments/data/retrieval_ready_mmseqs/seq_embedding_esm1b
  OPENFOLD_CHECKPOINT=<repo_root>/experiments/data/openfold_soloseq_params/seq_model_esm1b_ptm.pt
  CONFIG_PRESET=seq_model_esm1b_ptm
  DATA_CONFIG_PRESET=seqemb_initial_training

  DEVICE=cuda
  PRECISION=fp32
  BATCH_SIZE=1
  NUM_WORKERS=4
  MAX_RECYCLING_ITERS=0
  MAX_VAL_BATCHES=0
  LOG_EVERY_N_BATCHES=50
  DISABLE_TEMPLATES=1
  STRICT_SEQ_EMBEDDINGS=0

  LOG_DIR=<repo_root>/logs

Example:
  flux batch -N1 -n1 -c8 -g1 -t2h -q pbatch \
    scripts/retrieval/flux_eval_soloseq_baseline_submit.sh
EOF
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
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif [[ -f "$(pwd)/scripts/retrieval/eval_soloseq_baseline.py" ]]; then
  REPO_ROOT="$(pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}" || {
  echo "Error: failed to cd to repo root '${REPO_ROOT}'." >&2
  exit 1
}

if [[ ! -f "${REPO_ROOT}/scripts/retrieval/eval_soloseq_baseline.py" ]]; then
  echo "Error: could not locate repository root at '${REPO_ROOT}'." >&2
  echo "Hint: run flux batch from repo root or set REPO_ROOT=/path/to/openfold." >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-openfold_dev}"
ROCM_VERSION="${ROCM_VERSION:-6.4.2}"
ROCM_VERSION_DIR="${ROCM_VERSION_DIR:-rocm-${ROCM_VERSION}}"
USE_ROCM_MODULES="${USE_ROCM_MODULES:-1}"

MANIFEST_PATH="${MANIFEST_PATH:-${REPO_ROOT}/experiments/data/retrieval_ready_mmseqs/manifest.jsonl}"
SEQ_EMBED_DIR="${SEQ_EMBED_DIR:-${REPO_ROOT}/experiments/data/retrieval_ready_mmseqs/seq_embedding_esm1b}"
OPENFOLD_CHECKPOINT="${OPENFOLD_CHECKPOINT:-${REPO_ROOT}/experiments/data/openfold_soloseq_params/seq_model_esm1b_ptm.pt}"
CONFIG_PRESET="${CONFIG_PRESET:-seq_model_esm1b_ptm}"
DATA_CONFIG_PRESET="${DATA_CONFIG_PRESET:-seqemb_initial_training}"

DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_RECYCLING_ITERS="${MAX_RECYCLING_ITERS:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
LOG_EVERY_N_BATCHES="${LOG_EVERY_N_BATCHES:-50}"
DISABLE_TEMPLATES="${DISABLE_TEMPLATES:-1}"
STRICT_SEQ_EMBEDDINGS="${STRICT_SEQ_EMBEDDINGS:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

if [[ -f /etc/profile.d/z00_lmod.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/z00_lmod.sh
fi

if [[ "${USE_ROCM_MODULES}" == "1" ]] && command -v ml >/dev/null 2>&1; then
  ml rocm/"${ROCM_VERSION}" || {
    echo "Error: failed to load module rocm/${ROCM_VERSION}." >&2
    exit 1
  }
  ml craype-accel-amd-gfx942 cray-mpich libfabric || {
    echo "Error: failed to load GPU-related modules." >&2
    exit 1
  }
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
LOG_FILE="${LOG_DIR}/soloseq_baseline_eval_${JOB_ID}.log"
OUTPUT_JSON="${LOG_DIR}/soloseq_baseline_val_metrics_${JOB_ID}.json"

CMD=(
  python scripts/retrieval/eval_soloseq_baseline.py
  --manifest_path "${MANIFEST_PATH}"
  --seq_embedding_dir "${SEQ_EMBED_DIR}"
  --openfold_checkpoint "${OPENFOLD_CHECKPOINT}"
  --config_preset "${CONFIG_PRESET}"
  --data_config_preset "${DATA_CONFIG_PRESET}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_recycling_iters "${MAX_RECYCLING_ITERS}"
  --max_val_batches "${MAX_VAL_BATCHES}"
  --log_every_n_batches "${LOG_EVERY_N_BATCHES}"
  --output_json "${OUTPUT_JSON}"
)

if [[ "${DISABLE_TEMPLATES}" == "1" ]]; then
  CMD+=(--disable_templates)
else
  CMD+=(--no-disable_templates)
fi
if [[ "${STRICT_SEQ_EMBEDDINGS}" == "1" ]]; then
  CMD+=(--strict_seq_embeddings)
fi

echo "Running OpenFold SoloSeq baseline eval with:"
echo "  repo_root: ${REPO_ROOT}"
echo "  manifest_path: ${MANIFEST_PATH}"
echo "  seq_embed_dir: ${SEQ_EMBED_DIR}"
echo "  checkpoint: ${OPENFOLD_CHECKPOINT}"
echo "  config_preset: ${CONFIG_PRESET}"
echo "  data_config_preset: ${DATA_CONFIG_PRESET}"
echo "  device: ${DEVICE}"
echo "  precision: ${PRECISION}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  num_workers: ${NUM_WORKERS}"
echo "  max_recycling_iters: ${MAX_RECYCLING_ITERS}"
echo "  max_val_batches: ${MAX_VAL_BATCHES}"
echo "  output_json: ${OUTPUT_JSON}"
echo "  log_file: ${LOG_FILE}"

"${CMD[@]}" "${EXTRA_EVAL_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

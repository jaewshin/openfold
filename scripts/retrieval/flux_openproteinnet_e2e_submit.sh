#!/usr/bin/env bash

# Keep nounset + pipefail, but avoid global -e so non-interactive .bashrc
# commands (e.g. stty) do not terminate the job.
set -uo pipefail

usage() {
  cat <<'EOF'
Run OpenProteinNet E2E bootstrap under Flux.

Usage:
  flux batch [flux options] scripts/retrieval/flux_openproteinnet_e2e_submit.sh -o /path/to/openproteinnet [-- <extra e2e args>]

Required:
  -o, --output-root PATH     Output root for OpenProteinNet download/prep/embed artifacts.

Optional env vars (defaults shown):
  CONDA_ENV=openfold_dev
  ROCM_VERSION=6.4.2
  ROCM_VERSION_DIR=rocm-6.4.2
  USE_ROCM_MODULES=auto      (auto: on for EMBEDDING_DEVICE={cuda,auto}, off for cpu)
  SUBSETS="pdb uniclust30_filtered data_caches"
  SPLIT_STRATEGY=mmseqs
  MMSEQS_BIN=mmseqs
  MMSEQS_THREADS=16
  EMBEDDING_DEVICE=cuda
  EMBEDDING_TOKS_PER_BATCH=2048
  READY_DIR=""               (if set, passes --ready_dir)
  EMBEDDING_DIR=""           (if set, passes --embedding_dir)
  MMSEQS_TMP_DIR=""          (if set, passes --mmseqs_tmp_dir)
  LOG_DIR=<repo_root>/logs

Example:
  flux batch -N1 -n1 -c16 -g1 -t24h -q pbatch \
    scripts/retrieval/flux_openproteinnet_e2e_submit.sh \
    -o /p/vast1/shin9/data/openproteinnet
EOF
}

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EXTRA_E2E_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-root)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 1
      fi
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_E2E_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_E2E_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "Error: --output-root is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${REPO_ROOT}"
elif [[ -f "$(pwd)/scripts/retrieval/openproteinnet_e2e.py" ]]; then
  # For flux batch, cwd is usually the submit directory.
  REPO_ROOT="$(pwd)"
else
  # Fallback when running directly from the repository checkout.
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

cd "${REPO_ROOT}" || {
  echo "Error: failed to cd to repo root '${REPO_ROOT}'." >&2
  exit 1
}
export PATH=$(pwd)/mmseqs/bin/:$PATH

if [[ ! -f "${REPO_ROOT}/scripts/retrieval/openproteinnet_e2e.py" ]]; then
  echo "Error: could not locate repository root at '${REPO_ROOT}'." >&2
  echo "Hint: run flux batch from repo root or set REPO_ROOT=/path/to/openfold." >&2
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-openfold_dev}"
ROCM_VERSION="${ROCM_VERSION:-6.4.2}"
ROCM_VERSION_DIR="${ROCM_VERSION_DIR:-rocm-${ROCM_VERSION}}"
SUBSETS="${SUBSETS:-pdb uniclust30_filtered data_caches}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-mmseqs}"
MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
MMSEQS_THREADS="${MMSEQS_THREADS:-16}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_TOKS_PER_BATCH="${EMBEDDING_TOKS_PER_BATCH:-2048}"
READY_DIR="${READY_DIR:-}"
EMBEDDING_DIR="${EMBEDDING_DIR:-}"
MMSEQS_TMP_DIR="${MMSEQS_TMP_DIR:-}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
USE_ROCM_MODULES="${USE_ROCM_MODULES:-auto}"

if [[ "${USE_ROCM_MODULES}" == "auto" ]]; then
  if [[ "${EMBEDDING_DEVICE}" == "cuda" || "${EMBEDDING_DEVICE}" == "auto" ]]; then
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

if [[ "${SPLIT_STRATEGY}" == "mmseqs" ]]; then
  if [[ "${MMSEQS_BIN}" == */* ]]; then
    if [[ ! -x "${MMSEQS_BIN}" ]]; then
      echo "Error: MMseqs binary not executable: ${MMSEQS_BIN}" >&2
      exit 1
    fi
  elif command -v "${MMSEQS_BIN}" >/dev/null 2>&1; then
    MMSEQS_BIN="$(command -v "${MMSEQS_BIN}")"
  elif [[ "${MMSEQS_BIN}" == "mmseqs" && -x "${REPO_ROOT}/mmseqs/bin/mmseqs" ]]; then
    MMSEQS_BIN="${REPO_ROOT}/mmseqs/bin/mmseqs"
    echo "Info: MMseqs not found in PATH; using repo-local binary: ${MMSEQS_BIN}"
  else
    echo "Error: MMseqs binary not found in PATH: ${MMSEQS_BIN}" >&2
    exit 1
  fi
fi

read -r -a SUBSETS_ARR <<< "${SUBSETS}"
if [[ "${#SUBSETS_ARR[@]}" -eq 0 ]]; then
  echo "Error: SUBSETS resolved to an empty list." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" || {
  echo "Error: failed to create output dir '${OUTPUT_ROOT}'." >&2
  exit 1
}
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
LOG_FILE="${LOG_DIR}/openproteinnet_e2e_${JOB_ID}.log"

CMD=(
  python scripts/retrieval/openproteinnet_e2e.py
  --output_root "${OUTPUT_ROOT}"
  --subsets "${SUBSETS_ARR[@]}"
  --split_strategy "${SPLIT_STRATEGY}"
  --mmseqs_bin "${MMSEQS_BIN}"
  --mmseqs_threads "${MMSEQS_THREADS}"
  --embedding_device "${EMBEDDING_DEVICE}"
  --embedding_toks_per_batch "${EMBEDDING_TOKS_PER_BATCH}"
)

if [[ -n "${READY_DIR}" ]]; then
  CMD+=(--ready_dir "${READY_DIR}")
fi
if [[ -n "${EMBEDDING_DIR}" ]]; then
  CMD+=(--embedding_dir "${EMBEDDING_DIR}")
fi
if [[ -n "${MMSEQS_TMP_DIR}" ]]; then
  CMD+=(--mmseqs_tmp_dir "${MMSEQS_TMP_DIR}")
fi

echo "Running OpenProteinNet E2E with:"
echo "  repo_root: ${REPO_ROOT}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  subsets: ${SUBSETS}"
echo "  split_strategy: ${SPLIT_STRATEGY}"
echo "  mmseqs_bin: ${MMSEQS_BIN}"
echo "  mmseqs_threads: ${MMSEQS_THREADS}"
echo "  use_rocm_modules: ${USE_ROCM_MODULES}"
echo "  embedding_device: ${EMBEDDING_DEVICE}"
echo "  embedding_toks_per_batch: ${EMBEDDING_TOKS_PER_BATCH}"
echo "  log_file: ${LOG_FILE}"

"${CMD[@]}" "${EXTRA_E2E_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00
#SBATCH --no-requeue

set -eo pipefail

# Conda hook scripts can reference unset vars; avoid nounset during activation.
# Do not source ~/.bashrc here (it may enable nounset and break hook scripts).
set +u
declare -a CONDA_BASE_CANDIDATES=()
if [[ -n "${CONDA_BASE:-}" ]]; then
  CONDA_BASE_CANDIDATES+=("${CONDA_BASE}")
fi
if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE_CANDIDATES+=("$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)")
fi
CONDA_BASE_CANDIDATES+=("$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3")
CONDA_SH=""
for base in "${CONDA_BASE_CANDIDATES[@]}"; do
  if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${base}/etc/profile.d/conda.sh"
    CONDA_BASE="${base}"
    break
  fi
done
if [[ -z "${CONDA_SH}" ]]; then
  echo "Cannot find conda.sh. Checked: ${CONDA_BASE_CANDIDATES[*]}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-openfold_dev}"
set -u

if [[ -z "${CHUNK_FASTA:-}" ]]; then
  echo "CHUNK_FASTA is required" >&2
  exit 1
fi
if [[ -z "${CHUNK_OUT_DIR:-}" ]]; then
  echo "CHUNK_OUT_DIR is required" >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-esm2_t12_35M_UR50D}"
REPR_LAYER="${REPR_LAYER:-12}"
TOKS_PER_BATCH="${TOKS_PER_BATCH:-524288}"
TOKS_PER_BATCH_CANDIDATES="${TOKS_PER_BATCH_CANDIDATES:-}"
SHARD_SIZE="${SHARD_SIZE:-100000}"
TRUNCATION_SEQ_LENGTH="${TRUNCATION_SEQ_LENGTH:-1022}"
INDEX_PLACEHOLDER="${INDEX_PLACEHOLDER:-${CHUNK_OUT_DIR}/placeholder.index}"
CUDA_BUSY_RETRIES="${CUDA_BUSY_RETRIES:-3}"
CUDA_BUSY_SLEEP_SEC="${CUDA_BUSY_SLEEP_SEC:-30}"

RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%s)"
WORK_DIR="${CHUNK_OUT_DIR}.work_${RUN_TAG}"
mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${CHUNK_OUT_DIR}")" "$(dirname "${INDEX_PLACEHOLDER}")"

cd /insomnia001/depts/pmg/users/js6118/openfold/openfold/model/faiss

echo "[info] model=${MODEL_NAME}"
echo "[info] chunk_fasta=${CHUNK_FASTA}"
echo "[info] chunk_out_dir=${CHUNK_OUT_DIR}"
echo "[info] default_toks_per_batch=${TOKS_PER_BATCH} shard_size=${SHARD_SIZE}"
echo "[info] node=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[info] cuda_busy_retries=${CUDA_BUSY_RETRIES} cuda_busy_sleep_sec=${CUDA_BUSY_SLEEP_SEC}"

if [[ -n "${TOKS_PER_BATCH_CANDIDATES}" ]]; then
  IFS=',' read -r -a TOKS_LIST <<< "${TOKS_PER_BATCH_CANDIDATES}"
else
  TOKS_LIST=("${TOKS_PER_BATCH}")
fi

success=0
for toks in "${TOKS_LIST[@]}"; do
  toks="$(echo "${toks}" | xargs)"
  if [[ -z "${toks}" ]]; then
    continue
  fi
  attempt=1
  max_attempts=$((CUDA_BUSY_RETRIES + 1))
  while [[ "${attempt}" -le "${max_attempts}" ]]; do
    echo "[info] trying toks_per_batch=${toks} attempt=${attempt}/${max_attempts}"
    rm -f "${WORK_DIR}"/shard_*.npy "${WORK_DIR}"/shard_*.pkl "${WORK_DIR}/metadata.json" || true
    attempt_log="${WORK_DIR}/attempt_toks_${toks}_try_${attempt}.log"
    set +e
    python run_faiss.py \
      "${MODEL_NAME}" \
      "${CHUNK_FASTA}" \
      "${INDEX_PLACEHOLDER}" \
      --embeddings_dir "${WORK_DIR}" \
      --embed_only \
      --repr_layer "${REPR_LAYER}" \
      --toks_per_batch "${toks}" \
      --truncation_seq_length "${TRUNCATION_SEQ_LENGTH}" \
      --shard_size "${SHARD_SIZE}" 2>&1 | tee "${attempt_log}"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "${rc}" -eq 0 ]]; then
      success=1
      echo "[info] embedding succeeded with toks_per_batch=${toks} attempt=${attempt}"
      break 2
    fi
    if rg -q "CUDA error: CUDA-capable device\(s\) is/are busy or unavailable|device is busy|device busy or unavailable|initialization error" "${attempt_log}"; then
      if [[ "${attempt}" -lt "${max_attempts}" ]]; then
        echo "[warn] transient CUDA availability failure with toks_per_batch=${toks}; sleeping ${CUDA_BUSY_SLEEP_SEC}s before retry"
        sleep "${CUDA_BUSY_SLEEP_SEC}"
        attempt=$((attempt + 1))
        continue
      fi
    fi
    echo "[warn] failed with toks_per_batch=${toks} attempt=${attempt} (exit=${rc})"
    break
  done
done

if [[ "${success}" -ne 1 ]]; then
  echo "[error] all toks_per_batch attempts failed: ${TOKS_LIST[*]}" >&2
  exit 1
fi

if [[ -d "${CHUNK_OUT_DIR}" ]]; then
  mv "${CHUNK_OUT_DIR}" "${CHUNK_OUT_DIR}.old_${RUN_TAG}"
fi
mv "${WORK_DIR}" "${CHUNK_OUT_DIR}"

echo "[done] ${CHUNK_FASTA}"

#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Submit many single-GPU embedding jobs (one chunk FASTA per job).

Required:
  --input_fasta PATH
  --chunks_dir PATH
  --embeddings_root PATH
  One of:
    --num_jobs N
    --seqs_per_job K

Optional:
  --model_name NAME              (default: esm2_t12_35M_UR50D)
  --repr_layer N                 (default: 12)
  --toks_per_batch N             (default: 524288)
  --toks_candidates CSV          (default: auto descending list)
  --truncation_seq_length N      (default: 1022)
  --shard_size N                 (default: 100000)
  --conda_env NAME               (default: openfold_dev)
  --account NAME                 (default: pmg)
  --time D-HH:MM                 (default: 3-00:00)
  --cpus N                       (default: 4)
  --mem SIZE                     (default: 64G)
  --partition NAME
  --exclude HOSTS
  --force                        (resubmit even if chunk output already complete)
  --submit_limit N               (submit first N chunks only)
  --dry_run

Example:
  bash scripts/retrieval/embeds_generation_parallel_submit.sh \
    --input_fasta /path/uniref50.fasta \
    --chunks_dir /path/uniref50_chunks \
    --embeddings_root /path/uniref50/embeddings/esm2_35M_parallel \
    --num_jobs 32
EOF
}

INPUT_FASTA=""
CHUNKS_DIR=""
EMBEDDINGS_ROOT=""
NUM_JOBS=0
SEQS_PER_JOB=0
MODEL_NAME="esm2_t12_35M_UR50D"
REPR_LAYER=12
TOKS_PER_BATCH=524288
TOKS_CANDIDATES=""
TRUNCATION_SEQ_LENGTH=1022
SHARD_SIZE=100000
CONDA_ENV="openfold_dev"
ACCOUNT="pmg"
TIME_LIMIT="3-00:00"
CPUS=4
MEM="64G"
PARTITION=""
EXCLUDE=""
SUBMIT_LIMIT=0
FORCE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_fasta) INPUT_FASTA="$2"; shift 2 ;;
    --chunks_dir) CHUNKS_DIR="$2"; shift 2 ;;
    --embeddings_root) EMBEDDINGS_ROOT="$2"; shift 2 ;;
    --num_jobs) NUM_JOBS="$2"; shift 2 ;;
    --seqs_per_job) SEQS_PER_JOB="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --repr_layer) REPR_LAYER="$2"; shift 2 ;;
    --toks_per_batch) TOKS_PER_BATCH="$2"; shift 2 ;;
    --toks_candidates) TOKS_CANDIDATES="$2"; shift 2 ;;
    --truncation_seq_length) TRUNCATION_SEQ_LENGTH="$2"; shift 2 ;;
    --shard_size) SHARD_SIZE="$2"; shift 2 ;;
    --conda_env) CONDA_ENV="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --exclude) EXCLUDE="$2"; shift 2 ;;
    --force) FORCE=1; shift 1 ;;
    --submit_limit) SUBMIT_LIMIT="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${INPUT_FASTA}" || -z "${CHUNKS_DIR}" || -z "${EMBEDDINGS_ROOT}" ]]; then
  echo "Missing required args" >&2
  usage
  exit 1
fi

if [[ "${NUM_JOBS}" -gt 0 && "${SEQS_PER_JOB}" -gt 0 ]]; then
  echo "Provide only one of --num_jobs or --seqs_per_job" >&2
  exit 1
fi
if [[ "${NUM_JOBS}" -eq 0 && "${SEQS_PER_JOB}" -eq 0 ]]; then
  echo "Provide one of --num_jobs or --seqs_per_job" >&2
  exit 1
fi

mkdir -p "${CHUNKS_DIR}" "${EMBEDDINGS_ROOT}"
LOG_DIR="${EMBEDDINGS_ROOT}/logs"
PLACEHOLDER_DIR="${EMBEDDINGS_ROOT}/_placeholders"
mkdir -p "${LOG_DIR}" "${PLACEHOLDER_DIR}"

SPLIT_ARGS=(--input_fasta "${INPUT_FASTA}" --output_dir "${CHUNKS_DIR}")
if [[ "${NUM_JOBS}" -gt 0 ]]; then
  SPLIT_ARGS+=(--num_splits "${NUM_JOBS}")
  EXPECTED_MODE_TYPE="num_splits"
  EXPECTED_MODE_VALUE="${NUM_JOBS}"
else
  SPLIT_ARGS+=(--seqs_per_split "${SEQS_PER_JOB}")
  EXPECTED_MODE_TYPE="seqs_per_split"
  EXPECTED_MODE_VALUE="${SEQS_PER_JOB}"
fi

MANIFEST_PATH="${CHUNKS_DIR}/manifest.json"
mapfile -t EXISTING_CHUNK_FILES < <(find "${CHUNKS_DIR}" -maxdepth 1 -type f -name 'chunk_*.fasta' -size +0c | sort)

should_split=1
if [[ ${#EXISTING_CHUNK_FILES[@]} -gt 0 ]]; then
  should_split=0
  if [[ -f "${MANIFEST_PATH}" ]]; then
    # Validate manifest to avoid accidentally reusing stale chunking.
    if ! python - "${MANIFEST_PATH}" "${INPUT_FASTA}" "${EXPECTED_MODE_TYPE}" "${EXPECTED_MODE_VALUE}" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
input_fasta = str(Path(sys.argv[2]).resolve())
expected_mode_type = sys.argv[3]
expected_mode_value = int(sys.argv[4])

try:
    m = json.loads(manifest_path.read_text())
except Exception:
    sys.exit(2)

manifest_input = str(Path(m.get("input_fasta", "")).resolve()) if m.get("input_fasta") else ""
mode = m.get("mode", {})
mode_type = mode.get("type")
mode_value = mode.get("value")

if manifest_input != input_fasta:
    sys.exit(3)
if mode_type != expected_mode_type:
    sys.exit(4)
if int(mode_value) != expected_mode_value:
    sys.exit(5)

sys.exit(0)
PY
    then
      should_split=1
      echo "[info] Existing chunks found but manifest mismatch; regenerating chunks."
    fi
  else
    should_split=1
    echo "[info] Existing chunks found but no manifest.json; regenerating chunks."
  fi
fi

if [[ "${should_split}" -eq 1 ]]; then
  find "${CHUNKS_DIR}" -maxdepth 1 -type f -name 'chunk_*.fasta' -delete
  rm -f "${MANIFEST_PATH}"
  echo "[info] Splitting FASTA..."
  python "${SCRIPT_DIR}/split_fasta_chunks.py" "${SPLIT_ARGS[@]}"
else
  echo "[info] Reusing existing chunk FASTA files in ${CHUNKS_DIR}"
fi

mapfile -t CHUNK_FILES < <(find "${CHUNKS_DIR}" -maxdepth 1 -type f -name 'chunk_*.fasta' -size +0c | sort)
if [[ ${#CHUNK_FILES[@]} -eq 0 ]]; then
  echo "No chunk FASTA files found in ${CHUNKS_DIR}" >&2
  exit 1
fi

chunk_is_complete() {
  local chunk_out="$1"
  [[ -f "${chunk_out}/metadata.json" ]] || return 1
  compgen -G "${chunk_out}/shard_*.npy" > /dev/null || return 1
  compgen -G "${chunk_out}/shard_*.pkl" > /dev/null || return 1
  return 0
}

if [[ -z "${TOKS_CANDIDATES}" ]]; then
  # Conservative fallback chain for 35M at long sequence lengths.
  TOKS_CANDIDATES="${TOKS_PER_BATCH},262144,131072,65536,32768,16384,8192"
fi

CONDA_BASE_RESOLVED=""
if command -v conda > /dev/null 2>&1; then
  CONDA_BASE_RESOLVED="$(conda info --base 2>/dev/null || true)"
fi
if [[ -z "${CONDA_BASE_RESOLVED}" ]]; then
  CONDA_BASE_RESOLVED="${CONDA_BASE:-$HOME/miniforge3}"
fi
echo "[info] conda_base=${CONDA_BASE_RESOLVED}"

declare -A ACTIVE_JOB_NAMES=()
CURRENT_USER="${USER:-$(id -un)}"
while read -r active_job_name; do
  [[ -z "${active_job_name}" ]] && continue
  ACTIVE_JOB_NAMES["${active_job_name}"]=1
done < <(squeue -u "${CURRENT_USER}" -h -o '%j %t' | awk '$2=="R" || $2=="PD" {print $1}')

JOBS_TSV="${EMBEDDINGS_ROOT}/submitted_jobs.tsv"
echo -e "job_id\tchunk_id\tchunk_fasta\tchunk_embeddings_dir" > "${JOBS_TSV}"

count=0
skipped=0
skipped_active=0
for chunk_fasta in "${CHUNK_FILES[@]}"; do
  chunk_base="$(basename "${chunk_fasta}" .fasta)"           # chunk_00012
  chunk_id="${chunk_base#chunk_}"                            # 00012
  chunk_out="${EMBEDDINGS_ROOT}/${chunk_base}"
  job_name="u50e35m_${chunk_id}"
  if [[ "${FORCE}" -ne 1 ]] && chunk_is_complete "${chunk_out}"; then
    echo "[skip] complete chunk found: ${chunk_out}"
    skipped=$((skipped + 1))
    continue
  fi
  if [[ "${FORCE}" -ne 1 ]] && [[ -n "${ACTIVE_JOB_NAMES[${job_name}]:-}" ]]; then
    echo "[skip] active job already exists for ${job_name}"
    skipped_active=$((skipped_active + 1))
    continue
  fi

  sbatch_cmd=(
    sbatch
    --account="${ACCOUNT}"
    --time="${TIME_LIMIT}"
    --cpus-per-task="${CPUS}"
    --gpus=1
    --mem="${MEM}"
    --job-name="${job_name}"
    --output="${LOG_DIR}/${job_name}.%j.out"
    --error="${LOG_DIR}/${job_name}.%j.err"
    --export="ALL,CONDA_BASE=${CONDA_BASE_RESOLVED},CONDA_ENV=${CONDA_ENV},CHUNK_FASTA=${chunk_fasta},CHUNK_OUT_DIR=${chunk_out},MODEL_NAME=${MODEL_NAME},REPR_LAYER=${REPR_LAYER},TOKS_PER_BATCH=${TOKS_PER_BATCH},TOKS_PER_BATCH_CANDIDATES=${TOKS_CANDIDATES},TRUNCATION_SEQ_LENGTH=${TRUNCATION_SEQ_LENGTH},SHARD_SIZE=${SHARD_SIZE},INDEX_PLACEHOLDER=${PLACEHOLDER_DIR}/${job_name}.index"
  )

  if [[ -n "${PARTITION}" ]]; then
    sbatch_cmd+=(--partition="${PARTITION}")
  fi
  if [[ -n "${EXCLUDE}" ]]; then
    sbatch_cmd+=(--exclude="${EXCLUDE}")
  fi

  sbatch_cmd+=("${SCRIPT_DIR}/embeds_generation_parallel_worker.sbatch")

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${sbatch_cmd[*]}"
    job_id="DRYRUN_${chunk_id}"
  else
    submit_out="$("${sbatch_cmd[@]}")"
    echo "${submit_out}"
    job_id="$(echo "${submit_out}" | awk '{print $4}')"
    if [[ -z "${job_id}" ]]; then
      echo "Failed to parse job id from: ${submit_out}" >&2
      exit 1
    fi
  fi

  echo -e "${job_id}\t${chunk_id}\t${chunk_fasta}\t${chunk_out}" >> "${JOBS_TSV}"
  count=$((count + 1))
  if [[ "${SUBMIT_LIMIT}" -gt 0 && "${count}" -ge "${SUBMIT_LIMIT}" ]]; then
    echo "[info] Reached submit_limit=${SUBMIT_LIMIT}"
    break
  fi
done

echo "[done] submitted ${count} jobs (skipped_complete=${skipped}, skipped_active=${skipped_active})"
echo "[info] job map: ${JOBS_TSV}"
echo "[next] merge chunk embeddings after completion:"
echo "  python scripts/retrieval/merge_embedding_chunks.py --chunks_root ${EMBEDDINGS_ROOT} --output_dir ${EMBEDDINGS_ROOT}/merged"

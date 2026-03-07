#!/usr/bin/env bash
# flux: -N1
# flux: -q pdebug
# flux: -t 1h
# flux: --exclusive

set -euo pipefail

resume_flag=false
if [ "${1:-}" = "resume" ]; then
  resume_flag=true
fi

require_file() {
  local path="$1"
  local desc="$2"
  if [ ! -f "$path" ]; then
    echo "Error: missing ${desc}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local desc="$2"
  if [ ! -d "$path" ]; then
    echo "Error: missing ${desc}: ${path}" >&2
    exit 1
  fi
}

resolve_resume_ckpt() {
  local run_dir="$1"
  local last_ckpt="${run_dir}/checkpoints/last.ckpt"
  if [ -f "$last_ckpt" ]; then
    printf '%s\n' "$last_ckpt"
    return 0
  fi

  local latest=""
  latest=$(ls -1t "${run_dir}"/checkpoints/step=*.ckpt 2>/dev/null | head -n 1 || true)
  if [ -n "$latest" ]; then
    printf '%s\n' "$latest"
    return 0
  fi

  return 1
}

REPO_ROOT="/p/vast1/shin9/openfold"
CONFIG_PATH="/p/vast1/shin9/openfold/configs/retrieval/experiments/rawseq_ragport_flux_pdebug_smoke.yaml"
RUN_NAME="rawseq_ragport_flux_pdebug_smoke"
RUN_DIR="/p/vast1/shin9/openfold/logs/flux_runs/${RUN_NAME}"

DATA_ROOT="/p/vast1/shin9/openfold/experiments/data"
READY_DIR="${DATA_ROOT}/retrieval_ready_mmseqs"
MANIFEST_PATH="${READY_DIR}/manifest.jsonl"
SEQ_EMB_DIR="${READY_DIR}/seq_embedding_esm1b"

SEQ_INDEX_PATH="${DATA_ROOT}/u50_esm2_35M.index"
SEQ_INDEX_IDS_PATH="${DATA_ROOT}/u50_esm2_35M_ids.txt"
SEQ_DB_FASTA_PATH="${DATA_ROOT}/uniref50.fasta"
SEQ_DB_FASTA_INDEX_DB="${DATA_ROOT}/uniref50.seqio.sqlite"

OPENFOLD_CKPT="/p/vast1/shin9/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
TORCH_HOME="/p/vast1/shin9/openfold/resources/torch"
HF_HOME="/p/vast1/shin9/openfold/resources/hf_cache"
ESM2_PT="${TORCH_HOME}/hub/checkpoints/esm2_t12_35M_UR50D.pt"
ESM2_REG="${TORCH_HOME}/hub/checkpoints/esm2_t12_35M_UR50D-contact-regression.pt"
ESM1B_SAFE="${HF_HOME}/hub/models--mhcelik--esm-efficient/snapshots/local/esm1b.safetensors"

ROCM_VERSION="6.4.2"
ROCM_VERSION_DIR="rocm-6.4.2"
MAMBA_ENV_NAME="openfold_dev"

require_file "${REPO_ROOT}/scripts/retrieval/train_retrieval_modular.py" "retrieval trainer"
require_file "$CONFIG_PATH" "experiment config"
require_dir "$READY_DIR" "retrieval-ready data directory"
require_dir "$SEQ_EMB_DIR" "sequence embedding directory"
require_file "$MANIFEST_PATH" "manifest"
require_file "$SEQ_INDEX_PATH" "sequence FAISS index"
require_file "$SEQ_INDEX_IDS_PATH" "sequence FAISS row-id map"
require_file "$SEQ_DB_FASTA_PATH" "sequence FASTA"
require_file "$SEQ_DB_FASTA_INDEX_DB" "sequence FASTA sqlite index"
require_file "$OPENFOLD_CKPT" "OpenFold SoloSeq checkpoint"
require_file "$ESM2_PT" "ESM2 35M checkpoint"
require_file "$ESM2_REG" "ESM2 35M regression checkpoint"
require_file "$ESM1B_SAFE" "ESM-efficient ESM1b safetensors checkpoint"

mkdir -p "$RUN_DIR"

source /etc/profile.d/z00_lmod.sh
ml rocm/"${ROCM_VERSION}"
ml craype-accel-amd-gfx942 cray-mpich libfabric
export LD_LIBRARY_PATH="/opt/${ROCM_VERSION_DIR}/lib:${LD_LIBRARY_PATH:-}"

eval "$(mamba shell hook --shell bash)"
mamba activate "${MAMBA_ENV_NAME}"
PYTHON_BIN="$(command -v python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "Error: could not resolve python after activating ${MAMBA_ENV_NAME}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_IB_HCA=hsi0,hsi1,hsi2,hsi3
export RCCL_MSCCL_ENABLE=0
export NCCL_MIN_NCHANNELS=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
export FI_CXI_RDZV_PROTO=alt_read
export OMP_NUM_THREADS=31
export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR="$(hostname)"
export MASTER_PORT=29295
export TORCH_HOME="$TORCH_HOME"
export HF_HOME="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1

BATCH_JOBID="$(flux getattr jobid)"
NUMERIC_JOB_ID="$(flux job id "$BATCH_JOBID")"
LOCAL_JOB_ROOT="/l/ssd/${USER}/openfold/${NUMERIC_JOB_ID}"
MIOPEN_CACHE_DIR="${LOCAL_JOB_ROOT}/miopen"
mkdir -p "$MIOPEN_CACHE_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_CACHE_DIR"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CACHE_DIR"

resume_ckpt_path=""
if [ "$resume_flag" = true ]; then
  if ! resume_ckpt_path=$(resolve_resume_ckpt "$RUN_DIR"); then
    echo "Error: resume requested but no checkpoint found in ${RUN_DIR}/checkpoints" >&2
    exit 1
  fi
fi

echo "Python: ${PYTHON_BIN}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG_PATH}"
echo "Run dir: ${RUN_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Seq emb dir: ${SEQ_EMB_DIR}"
echo "Seq index: ${SEQ_INDEX_PATH}"
echo "OpenFold checkpoint: ${OPENFOLD_CKPT}"
echo "ESM2 checkpoint: ${ESM2_PT}"
echo "ESM1b safetensors: ${ESM1B_SAFE}"
echo "Resume checkpoint: ${resume_ckpt_path:-<none>}"

train_cmd=(
  "$PYTHON_BIN" -u "${REPO_ROOT}/scripts/retrieval/train_retrieval_modular.py"
  --config "$CONFIG_PATH"
  --set "trainer.output_dir=${RUN_DIR}"
  --set "data.manifest_path=${MANIFEST_PATH}"
  --set "data.seq_embedding_dir=${SEQ_EMB_DIR}"
  --set "retrieval.sources.seq.index_path=${SEQ_INDEX_PATH}"
  --set "retrieval.sources.seq.index_dim=480"
  --set "retrieval.rawseq_esm1b.seq_index_ids_path=${SEQ_INDEX_IDS_PATH}"
  --set "retrieval.rawseq_esm1b.seq_db_fasta_path=${SEQ_DB_FASTA_PATH}"
  --set "retrieval.rawseq_esm1b.seq_db_fasta_index_db=${SEQ_DB_FASTA_INDEX_DB}"
  --set "retrieval.rawseq_esm1b.esm1b_model_name=${ESM1B_SAFE}"
  --set "retrieval.rawseq_esm1b.hf_cache_dir=${HF_HOME}"
  --set "retrieval.rawseq_esm1b.torch_hub_dir=${TORCH_HOME}/hub"
  --set "model.openfold_checkpoint=${OPENFOLD_CKPT}"
  --set "wandb.run_name=${RUN_NAME}"
)

if [ -n "$resume_ckpt_path" ]; then
  train_cmd+=(--set "trainer.resume_ckpt_path=${resume_ckpt_path}")
fi

"${train_cmd[@]}"

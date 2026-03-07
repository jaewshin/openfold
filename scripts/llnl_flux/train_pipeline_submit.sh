#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_pipeline.conf"

usage() {
  cat <<USAGE >&2
Usage: flux batch [flux options] ${SCRIPT_DIR}/train_pipeline_submit.sh -c <config.yaml> [-r] [-m <minutes>]

Arguments:
  -c <config.yaml>   Experiment YAML to run.
  -r                 Resume from the latest checkpoint in the shared run directory.
  -m <minutes>       Safety margin for trainer.max_time.
USAGE
  exit 1
}

abspath() {
  local path="$1"
  if [ -d "$path" ]; then
    (cd "$path" && pwd)
  else
    local parent
    parent=$(cd "$(dirname "$path")" && pwd)
    printf '%s/%s\n' "$parent" "$(basename "$path")"
  fi
}

require_file() {
  local path="$1"
  local desc="$2"
  if [ ! -f "$path" ]; then
    echo "Error: Missing ${desc}: ${path}" >&2
    exit 1
  fi
}

copy_asset() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -Lf "$src" "$dst"
}

format_hhmmss() {
  local total="$1"
  local hours=$(( total / 3600 ))
  local minutes=$(( (total % 3600) / 60 ))
  local seconds=$(( total % 60 ))
  printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
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

resume_flag=false
config=""
safety_margin=""

while getopts ":c:rm:" opt; do
  case "$opt" in
    c) config="$OPTARG" ;;
    r) resume_flag=true ;;
    m) safety_margin="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "$config" ]; then
  usage
fi

config=$(abspath "$config")
if [ ! -f "$config" ]; then
  echo "Error: Config file not found: ${config}" >&2
  exit 1
fi

name=$(basename "$config" .yaml)
run_dir="${outdir}/${name}"
mkdir -p "$run_dir" "$flux_log_dir"
cp -f "$config" "${run_dir}/$(basename "$config")"

if [ -z "$safety_margin" ]; then
  safety_margin="$pbatch_safety_margin"
fi
if ! [[ "$safety_margin" =~ ^[0-9]+$ ]]; then
  echo "Error: safety margin must be an integer number of minutes" >&2
  exit 1
fi

source /etc/profile.d/z00_lmod.sh
ml rocm/"${rocm_version}"
ml craype-accel-amd-gfx942 cray-mpich libfabric
export LD_LIBRARY_PATH="/opt/${rocm_version_dir}/lib:${LD_LIBRARY_PATH:-}"
if [ -f "$bashrc_path" ]; then
  source "$bashrc_path"
fi
eval "$(mamba shell hook --shell bash)"
mamba activate "$conda_env_path"

export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=0

FIRST_HOSTID=$(flux hostlist -led '\n' | head -n 1)
FLUX_JOB_NNODES=$(flux hostlist -led '\n' | wc -l | tr -d ' ')
BATCH_JOBID=$(flux getattr jobid)
NUMERIC_JOB_ID=$(flux job id "$BATCH_JOBID")
FLUX_JOB_SIZE=$(flux getattr size)

if [ -n "$aws_ofi_rccl_dir" ] && [ -d "$aws_ofi_rccl_dir/lib" ]; then
  export LD_LIBRARY_PATH="${aws_ofi_rccl_dir}/lib:${LD_LIBRARY_PATH}"
  export NCCL_NET_PLUGIN=librccl-net.so
fi
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

firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
export MASTER_ADDR="$firsthost"
export MASTER_PORT=29295

miopen_cache="${local_scratch_root}/miopen_${NUMERIC_JOB_ID}"
rm -rf "$miopen_cache"
mkdir -p "$miopen_cache"
export MIOPEN_USER_DB_PATH="$miopen_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$miopen_cache"

local_root="${local_scratch_root}/${NUMERIC_JOB_ID}"
local_uniref_dir="${local_root}/uniref"
local_torch_home="${local_root}/torch_home"
local_hf_home="${local_root}/hf_home"
local_hf_snapshot_dir="${local_hf_home}/hub/models--mhcelik--esm-efficient/snapshots/local"
mkdir -p "$local_uniref_dir" "$local_torch_home/hub/checkpoints" "$local_hf_snapshot_dir"

require_file "$manifest_path" "manifest"
require_file "$seq_index_path" "sequence FAISS index"
require_file "$seq_index_ids_path" "sequence FAISS row-id map"
require_file "$seq_db_fasta_path" "sequence FASTA"
require_file "$seq_db_fasta_index_db" "sequence FASTA sqlite index"
require_file "$openfold_checkpoint_path" "OpenFold checkpoint"
require_file "$esm2_model_path" "ESM2 checkpoint"
require_file "$esm2_regression_path" "ESM2 regression checkpoint"
require_file "$esm1b_safetensors_path" "ESM1b safetensors checkpoint"

local_seq_index_path="${local_uniref_dir}/$(basename "$seq_index_path")"
local_seq_index_ids_path="${local_uniref_dir}/$(basename "$seq_index_ids_path")"
local_seq_db_index_db="${local_uniref_dir}/$(basename "$seq_db_fasta_index_db")"
local_seq_db_fasta_path="${local_uniref_dir}/$(basename "$seq_db_fasta_path")"
local_openfold_checkpoint="${local_root}/$(basename "$openfold_checkpoint_path")"
local_esm2_model_path="${local_torch_home}/hub/checkpoints/$(basename "$esm2_model_path")"
local_esm2_regression_path="${local_torch_home}/hub/checkpoints/$(basename "$esm2_regression_path")"
local_esm1b_safetensors_path="${local_hf_snapshot_dir}/$(basename "$esm1b_safetensors_path")"

copy_asset "$seq_index_path" "$local_seq_index_path"
copy_asset "$seq_index_ids_path" "$local_seq_index_ids_path"
copy_asset "$seq_db_fasta_index_db" "$local_seq_db_index_db"
ln -sfn "$seq_db_fasta_path" "$local_seq_db_fasta_path"
copy_asset "$openfold_checkpoint_path" "$local_openfold_checkpoint"
copy_asset "$esm2_model_path" "$local_esm2_model_path"
copy_asset "$esm2_regression_path" "$local_esm2_regression_path"
copy_asset "$esm1b_safetensors_path" "$local_esm1b_safetensors_path"

export TORCH_HOME="$local_torch_home"
export HF_HOME="$local_hf_home"
export HUGGINGFACE_HUB_CACHE="${local_hf_home}/hub"
export HF_HUB_DISABLE_TELEMETRY=1

export OPENFOLD_EXPECTED_GPUS="$devices_per_node"
export OPENFOLD_LOCAL_ESM1B_PATH="$local_esm1b_safetensors_path"
python - <<'PY'
import os
import torch

count = torch.cuda.device_count()
expected = int(os.environ["OPENFOLD_EXPECTED_GPUS"])
if count < expected:
    raise SystemExit(f"Expected at least {expected} visible GPUs, found {count}")

import flash_attn  # noqa: F401
import esme  # noqa: F401

from openfold.model.retrieval.context_esm1b import ESM1bContextEncoder

encoder = ESM1bContextEncoder(
    model_name=os.environ["OPENFOLD_LOCAL_ESM1B_PATH"],
    repr_layer=33,
    truncation_seq_length=128,
    tuning_mode="lora",
    lora_rank=8,
    lora_alpha=16.0,
    lora_dropout=0.0,
    backend="esm_efficient",
    fallback_backend=None,
    allow_download=False,
    hf_cache_dir=os.environ["HF_HOME"],
    torch_hub_dir=os.path.join(os.environ["TORCH_HOME"], "hub"),
    esm_efficient_use_pretrained=True,
    esm_efficient_compute_dtype="bfloat16",
).cuda()
outputs = encoder(["MKTAYIAKQRQISFVKSHFSRQDILDLIC"])
if len(outputs) != 1 or outputs[0].ndim != 2:
    raise SystemExit("ESM1b preflight failed to return token embeddings")
print(f"ESM1b preflight ok: {tuple(outputs[0].shape)} on {outputs[0].device}")
PY

wandb_run_id_file="${run_dir}/wandb_run_id.txt"
if [ -s "$wandb_run_id_file" ]; then
  wandb_run_id=$(tr -d '[:space:]' < "$wandb_run_id_file")
else
  wandb_run_id=$(python - <<'PY'
import uuid
print(uuid.uuid4().hex)
PY
)
  printf '%s\n' "$wandb_run_id" > "$wandb_run_id_file"
fi

max_time_override=""
job_timeleft_sec=$(flux job timeleft 2>/dev/null || true)
job_timeleft_sec=${job_timeleft_sec%.*}
if [[ "$job_timeleft_sec" =~ ^[0-9]+$ ]] && [ "$job_timeleft_sec" -gt 0 ] && [ "$safety_margin" -gt 0 ]; then
  margin_sec=$(( safety_margin * 60 ))
  if [ "$job_timeleft_sec" -gt "$margin_sec" ]; then
    max_time_sec=$(( job_timeleft_sec - margin_sec ))
    max_time_override=$(format_hhmmss "$max_time_sec")
    echo "Setting trainer.max_time=${max_time_override} (timeleft=${job_timeleft_sec}s margin=${margin_sec}s)"
  else
    echo "Warning: Flux timeleft (${job_timeleft_sec}s) is not larger than safety margin (${margin_sec}s); skipping trainer.max_time" >&2
  fi
fi

resume_ckpt_path=""
if [ "$resume_flag" = true ]; then
  if ! resume_ckpt_path=$(resolve_resume_ckpt "$run_dir"); then
    echo "Error: resume requested but no checkpoint found under ${run_dir}/checkpoints" >&2
    exit 1
  fi
fi

train_cmd=(
  python -u "${repo_root}/scripts/retrieval/train_retrieval_modular.py"
  --config "$config"
  --set "trainer.output_dir=${run_dir}"
  --set "data.manifest_path=${manifest_path}"
  --set "retrieval.sources.seq.index_path=${local_seq_index_path}"
  --set "retrieval.sources.seq.index_dim=480"
  --set "retrieval.rawseq_esm1b.seq_index_ids_path=${local_seq_index_ids_path}"
  --set "retrieval.rawseq_esm1b.seq_db_fasta_path=${local_seq_db_fasta_path}"
  --set "retrieval.rawseq_esm1b.seq_db_fasta_index_db=${local_seq_db_index_db}"
  --set "retrieval.rawseq_esm1b.esm1b_model_name=${local_esm1b_safetensors_path}"
  --set "retrieval.rawseq_esm1b.hf_cache_dir=${local_hf_home}"
  --set "retrieval.rawseq_esm1b.torch_hub_dir=${local_torch_home}/hub"
  --set "model.openfold_checkpoint=${local_openfold_checkpoint}"
  --set "wandb.entity=${wandb_entity}"
  --set "wandb.project=${wandb_project}"
  --set "wandb.run_id=${wandb_run_id}"
  --set "wandb.run_name=${name}"
  --set "wandb.resume=allow"
)

if [ -n "$max_time_override" ]; then
  train_cmd+=(--set "trainer.max_time='${max_time_override}'")
fi
if [ -n "$resume_ckpt_path" ]; then
  train_cmd+=(--set "trainer.resume_ckpt_path=${resume_ckpt_path}")
fi

echo "Run directory: ${run_dir}"
echo "Flux logs: ${flux_log_dir}"
echo "Resume checkpoint: ${resume_ckpt_path:-<none>}"
echo "W&B run id: ${wandb_run_id}"
echo "Local staging root: ${local_root}"

task_output="${flux_log_dir}/${name}_train-{{id}}.out"
job_name="${name}_train"
if [ "$resume_flag" = true ]; then
  task_output="${flux_log_dir}/${name}_train_resume-{{id}}.out"
  job_name="${name}_train_resume"
fi

flux run \
  --job-name="$job_name" \
  --output="$task_output" \
  -N1 \
  -n1 \
  --exclusive \
  -o mpibind=verbose:1 \
  -o fastload \
  "${train_cmd[@]}"

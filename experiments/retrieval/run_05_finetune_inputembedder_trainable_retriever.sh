#!/bin/bash
#SBATCH --account=pmg
#SBATCH --exclude=ins090
#SBATCH --job-name=retrieval_ft05_inputembed_retriever
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_ft05_inputembed_retriever_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_ft05_inputembed_retriever_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

export WANDB_API_KEY="${WANDB_API_KEY:-256af952437873cc1446152d821a4cfe7dcbadc8}"
WANDB_ENTITY="${WANDB_ENTITY:-jshin}"

RUN_TAG="retrieval_ft05_inputembed_retriever_effbs128_lr5e-5"
EXTRA_FLAGS="--train_input_embedder"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}-${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID}}"
OUTPUT_DIR="/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_runs/${RUN_TAG}/${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs

EXTRA_ARGS=()
if [[ -n "${EXTRA_FLAGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_FLAGS}"
fi

echo "Starting run: ${RUN_TAG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra OpenFold flags: ${EXTRA_FLAGS}"

python scripts/retrieval/train_retrieval_lightning.py   --dataset_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data   --seq_embedding_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data/seq_embedding_esm1b   --seq_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index   --struct_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s.index   --seq_index_dim 480   --struct_index_dim 512   --retrieval_ablation both   --top_k 8   --nprobe 64   --lr 5e-5   --batch_size 1   --accumulate_grad_batches 128   --num_workers 0   --max_epochs 1   --devices 1   --precision 32   --output_dir "${OUTPUT_DIR}"   --use_wandb   --wandb_project openfold-retrieval   --wandb_entity "${WANDB_ENTITY}"   --wandb_run_name "${WANDB_RUN_NAME}"   --wandb_run_id "${WANDB_RUN_ID}"   --wandb_resume allow   --wandb_tags "retrieval,lr5e-5,bs1,acc128,effbs128,${RUN_TAG}"   "${EXTRA_ARGS[@]}"

echo "Finished run: ${RUN_TAG}"

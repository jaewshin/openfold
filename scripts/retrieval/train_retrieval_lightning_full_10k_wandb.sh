#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_e2e_full_10k_wandb_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_e2e_full_10k_wandb_%j.err
#SBATCH --no-requeue

set -eo pipefail
source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
# Some conda activation scripts reference unset vars; disable nounset around activation.
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

# W&B credentials and run metadata.
# You can override these by exporting WANDB_API_KEY / WANDB_ENTITY / WANDB_RUN_ID before sbatch.
export WANDB_API_KEY="${WANDB_API_KEY:-256af952437873cc1446152d821a4cfe7dcbadc8}"
WANDB_ENTITY="${WANDB_ENTITY:-jshin}"
WANDB_RUN_ID="${WANDB_RUN_ID:-retrieval-e2e-full10k-${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-retrieval_e2e_full10k_${SLURM_JOB_ID}}"

OUTPUT_DIR="/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_e2e_full_10k_wandb/${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting retrieval training with W&B logging"
echo "W&B entity: ${WANDB_ENTITY}"
echo "W&B run id: ${WANDB_RUN_ID}"
echo "Output dir: ${OUTPUT_DIR}"

python scripts/retrieval/train_retrieval_lightning.py \
  --dataset_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data \
  --seq_embedding_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data/seq_embedding_esm1b \
  --seq_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index \
  --struct_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s.index \
  --seq_index_dim 480 \
  --struct_index_dim 512 \
  --retrieval_ablation both \
  --top_k 8 \
  --nprobe 64 \
  --batch_size 1 \
  --num_workers 0 \
  --max_epochs 1 \
  --devices 1 \
  --precision 32 \
  --output_dir "${OUTPUT_DIR}" \
  --use_wandb \
  --wandb_project openfold-retrieval \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_run_id "${WANDB_RUN_ID}" \
  --wandb_resume allow \
  --wandb_tags "retrieval,full10k,esm2_35m,tmvec2s"

echo "Finished retrieval training with W&B logging"

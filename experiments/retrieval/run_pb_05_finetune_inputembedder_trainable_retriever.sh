#!/bin/bash
#SBATCH --account=pmg
#SBATCH --exclude=ins090
#SBATCH --job-name=retr_pb05_inputembed_retriever
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_pb05_inputembed_retriever_rawseq_effbs128_lr5e-5_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_pb05_inputembed_retriever_rawseq_effbs128_lr5e-5_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY
fi
WANDB_ENTITY="${WANDB_ENTITY:-jshin}"

RUN_TAG="retrieval_pb05_inputembed_retriever_rawseq_effbs128_lr5e-5"
PIPELINE_ARGS="--retrieval_pipeline rawseq_esm1b --retriever_esm2_model_name esm2_t12_35M_UR50D --retriever_esm2_repr_layer 12 --retriever_esm2_max_len 1022 --retriever_tmvec_checkpoint /insomnia001/depts/pmg/users/js6118/openfold/tmvec-bench/binaries/tmvec2_student.pt --retriever_tmvec_max_len 1022 --retriever_normalize_queries --seq_index_ids_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M_ids.txt --struct_index_ids_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s_ids.txt --seq_db_fasta_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta --struct_db_fasta_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta --seq_db_fasta_index_db /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.seqio.sqlite --struct_db_fasta_index_db /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.seqio.sqlite --retrieved_esm1b_device cpu"
EXTRA_FLAGS="--train_input_embedder"
WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}-${SLURM_JOB_ID}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}_${SLURM_JOB_ID}}"
OUTPUT_DIR="/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_runs/${RUN_TAG}/${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs

PIPELINE_ARR=()
if [[ -n "${PIPELINE_ARGS}" ]]; then
  read -r -a PIPELINE_ARR <<< "${PIPELINE_ARGS}"
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_FLAGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_FLAGS}"
fi

echo "Starting run: ${RUN_TAG}"
echo "Pipeline args: ${PIPELINE_ARGS}"
echo "Extra OpenFold flags: ${EXTRA_FLAGS}"
echo "Output dir: ${OUTPUT_DIR}"

python scripts/retrieval/train_retrieval_lightning.py   --dataset_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data   --seq_embedding_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data/seq_embedding_esm1b   --seq_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index   --struct_index_path /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s.index   --seq_index_dim 480   --struct_index_dim 512   --retrieval_ablation both   --top_k 8   --nprobe 64   --lr 5e-5   --batch_size 1   --accumulate_grad_batches 128   --num_workers 0   --max_epochs 1   --devices 1   --precision 32   --output_dir "${OUTPUT_DIR}"   --use_wandb   --wandb_project openfold-retrieval   --wandb_entity "${WANDB_ENTITY}"   --wandb_run_name "${WANDB_RUN_NAME}"   --wandb_run_id "${WANDB_RUN_ID}"   --wandb_resume allow   --wandb_tags "retrieval,lr5e-5,bs1,acc128,effbs128,pipeline_rawseq,${RUN_TAG}"   "${PIPELINE_ARR[@]}"   "${EXTRA_ARGS[@]}"

echo "Finished run: ${RUN_TAG}"

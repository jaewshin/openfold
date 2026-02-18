#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=embed_openproteinnet_esm1b
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=7-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/embed_openproteinnet_esm1b_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/embed_openproteinnet_esm1b_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

DATASET_DIR="${DATASET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATASET_DIR}/seq_embedding_esm1b}"
TOKS_PER_BATCH="${TOKS_PER_BATCH:-2048}"
TRUNCATION_SEQ_LENGTH="${TRUNCATION_SEQ_LENGTH:-1022}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs

echo "Generating ESM-1b embeddings for OpenProteinNet retrieval dataset"
echo "Dataset dir: ${DATASET_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Device:      ${DEVICE}"
echo "toks/batch:  ${TOKS_PER_BATCH}"

python scripts/retrieval/generate_esm1b_seq_embeddings.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --toks_per_batch "${TOKS_PER_BATCH}" \
  --truncation_seq_length "${TRUNCATION_SEQ_LENGTH}"

echo "Finished ESM-1b embedding generation"

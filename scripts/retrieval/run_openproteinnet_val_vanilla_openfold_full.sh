#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=vanilla_openfold_val
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=7-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/vanilla_openfold_val_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/vanilla_openfold_val_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

DATASET_DIR="${DATASET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready}"
OPENPROTEINNET_DIR="${OPENPROTEINNET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/vanilla_openfold_val}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/insomnia001/depts/pmg/users/js6118/openfold/openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt}"
CONFIG_PRESET="${CONFIG_PRESET:-finetuning_no_templ_ptm}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
SUBSET="${SUBSET:-uniclust30}"
MAX_RECYCLING_ITERS="${MAX_RECYCLING_ITERS:-3}"

# Optional chunking controls
OFFSET="${OFFSET:-0}"
MAX_TARGETS="${MAX_TARGETS:-0}"

mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs
mkdir -p "${OUTPUT_ROOT}"

echo "Starting vanilla OpenFold validation run"
echo "  JOB_ID:          ${SLURM_JOB_ID:-n/a}"
echo "  DATASET_DIR:     ${DATASET_DIR}"
echo "  OPENPROTEINNET:  ${OPENPROTEINNET_DIR}"
echo "  OUTPUT_ROOT:     ${OUTPUT_ROOT}"
echo "  CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "  CONFIG_PRESET:   ${CONFIG_PRESET}"
echo "  MODEL_DEVICE:    ${MODEL_DEVICE}"
echo "  SUBSET:          ${SUBSET}"
echo "  MAX_RECYCLES:    ${MAX_RECYCLING_ITERS}"
echo "  OFFSET:          ${OFFSET}"
echo "  MAX_TARGETS:     ${MAX_TARGETS}"

python scripts/retrieval/run_openproteinnet_val_vanilla_openfold.py \
  --dataset-dir "${DATASET_DIR}" \
  --openproteinnet-dir "${OPENPROTEINNET_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-preset "${CONFIG_PRESET}" \
  --model-device "${MODEL_DEVICE}" \
  --max-recycling-iters "${MAX_RECYCLING_ITERS}" \
  --subset "${SUBSET}" \
  --offset "${OFFSET}" \
  --max-targets "${MAX_TARGETS}"

echo "Finished vanilla OpenFold validation run"

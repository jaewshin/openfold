#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=prepare_openproteinnet
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/prepare_openproteinnet_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/prepare_openproteinnet_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

OPENPROTEINNET_DIR="${OPENPROTEINNET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet}"
OUTPUT_DIR="${OUTPUT_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.98}"
MIN_LENGTH="${MIN_LENGTH:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5000}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-hash}"
# Only used when SPLIT_STRATEGY=mmseqs.
MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
MMSEQS_MIN_SEQ_ID="${MMSEQS_MIN_SEQ_ID:-0.3}"
MMSEQS_COVERAGE="${MMSEQS_COVERAGE:-0.8}"
MMSEQS_COV_MODE="${MMSEQS_COV_MODE:-0}"
MMSEQS_THREADS="${MMSEQS_THREADS:-16}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs

echo "Preparing OpenProteinNet retrieval dataset"
echo "Input:  ${OPENPROTEINNET_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Split strategy: ${SPLIT_STRATEGY}"

python scripts/retrieval/prepare_openproteinnet_dataset.py \
  --openproteinnet_dir "${OPENPROTEINNET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --min_length "${MIN_LENGTH}" \
  --progress_every "${PROGRESS_EVERY}" \
  --split_strategy "${SPLIT_STRATEGY}" \
  --mmseqs_bin "${MMSEQS_BIN}" \
  --mmseqs_min_seq_id "${MMSEQS_MIN_SEQ_ID}" \
  --mmseqs_coverage "${MMSEQS_COVERAGE}" \
  --mmseqs_cov_mode "${MMSEQS_COV_MODE}" \
  --mmseqs_threads "${MMSEQS_THREADS}" \
  --structure_mode absolute

echo "Finished OpenProteinNet dataset preparation"

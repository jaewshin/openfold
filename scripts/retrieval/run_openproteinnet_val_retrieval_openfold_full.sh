#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=retrieval_openfold_val
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=7-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_openfold_val_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/retrieval_openfold_val_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DATASET_DIR="${DATASET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_ready}"
OPENPROTEINNET_DIR="${OPENPROTEINNET_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet/retrieval_openfold_val}"

INDEX_PATH="${INDEX_PATH:-/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index}"
IDS_PATH="${IDS_PATH:-/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M_ids.txt}"
IDS_OFFSETS_PATH="${IDS_OFFSETS_PATH:-/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M_ids.txt.offsets.u64}"
SEQUENCE_SQLITE="${SEQUENCE_SQLITE:-/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.seqio.sqlite}"
SEQUENCE_FASTA="${SEQUENCE_FASTA:-/insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/insomnia001/depts/pmg/users/js6118/openfold/openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt}"
CONFIG_PRESET="${CONFIG_PRESET:-finetuning_no_templ_ptm}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
SUBSET="${SUBSET:-uniclust30}"

RETRIEVAL_EMBEDDER="${RETRIEVAL_EMBEDDER:-esm2_35m}"
RETRIEVAL_DEVICE="${RETRIEVAL_DEVICE:-cpu}"
ALIGNMENT_EMBEDDER="${ALIGNMENT_EMBEDDER:-aa_onehot}"
ALIGNMENT_DEVICE="${ALIGNMENT_DEVICE:-cpu}"

TOP_K="${TOP_K:-2000}"
TOP_K_PRIME="${TOP_K_PRIME:-64}"
MAX_ROWS="${MAX_ROWS:-64}"
LENGTH_RATIO_LOW="${LENGTH_RATIO_LOW:-0.7}"
LENGTH_RATIO_HIGH="${LENGTH_RATIO_HIGH:-1.3}"

SCALE="${SCALE:-10.0}"
BIAS="${BIAS:-0.0}"
GAP_OPEN="${GAP_OPEN:--8.0}"
GAP_EXTEND="${GAP_EXTEND:--0.5}"
MIN_QUERY_COVERAGE="${MIN_QUERY_COVERAGE:-0.15}"
MIN_ALIGNED_QUERY_LEN="${MIN_ALIGNED_QUERY_LEN:-30}"
MIN_SCORE_DENSITY="${MIN_SCORE_DENSITY:-1.0}"
MAX_GAP_FRAC="${MAX_GAP_FRAC:-0.85}"

OFFSET="${OFFSET:-0}"
MAX_TARGETS="${MAX_TARGETS:-0}"

mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs
mkdir -p "${OUTPUT_ROOT}"

echo "Starting retrieval-based OpenFold validation run"
echo "  JOB_ID:              ${SLURM_JOB_ID:-n/a}"
echo "  DATASET_DIR:         ${DATASET_DIR}"
echo "  OPENPROTEINNET_DIR:  ${OPENPROTEINNET_DIR}"
echo "  OUTPUT_ROOT:         ${OUTPUT_ROOT}"
echo "  INDEX_PATH:          ${INDEX_PATH}"
echo "  IDS_PATH:            ${IDS_PATH}"
echo "  IDS_OFFSETS_PATH:    ${IDS_OFFSETS_PATH}"
echo "  SEQUENCE_SQLITE:     ${SEQUENCE_SQLITE}"
echo "  SEQUENCE_FASTA:      ${SEQUENCE_FASTA}"
echo "  CHECKPOINT_PATH:     ${CHECKPOINT_PATH}"
echo "  CONFIG_PRESET:       ${CONFIG_PRESET}"
echo "  MODEL_DEVICE:        ${MODEL_DEVICE}"
echo "  RETRIEVAL_EMBEDDER:  ${RETRIEVAL_EMBEDDER}"
echo "  RETRIEVAL_DEVICE:    ${RETRIEVAL_DEVICE}"
echo "  ALIGNMENT_EMBEDDER:  ${ALIGNMENT_EMBEDDER}"
echo "  ALIGNMENT_DEVICE:    ${ALIGNMENT_DEVICE}"
echo "  TOP_K:               ${TOP_K}"
echo "  TOP_K_PRIME:         ${TOP_K_PRIME}"
echo "  MAX_ROWS:            ${MAX_ROWS}"
echo "  OFFSET:              ${OFFSET}"
echo "  MAX_TARGETS:         ${MAX_TARGETS}"

python scripts/retrieval/run_openproteinnet_val_retrieval_openfold.py \
  --dataset-dir "${DATASET_DIR}" \
  --openproteinnet-dir "${OPENPROTEINNET_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --subset "${SUBSET}" \
  --index-path "${INDEX_PATH}" \
  --ids-path "${IDS_PATH}" \
  --ids-offsets-path "${IDS_OFFSETS_PATH}" \
  --sequence-sqlite "${SEQUENCE_SQLITE}" \
  --sequence-fasta "${SEQUENCE_FASTA}" \
  --retrieval-embedder "${RETRIEVAL_EMBEDDER}" \
  --retrieval-device "${RETRIEVAL_DEVICE}" \
  --alignment-embedder "${ALIGNMENT_EMBEDDER}" \
  --alignment-device "${ALIGNMENT_DEVICE}" \
  --top-k "${TOP_K}" \
  --top-k-prime "${TOP_K_PRIME}" \
  --max-rows "${MAX_ROWS}" \
  --length-ratio-low "${LENGTH_RATIO_LOW}" \
  --length-ratio-high "${LENGTH_RATIO_HIGH}" \
  --scale "${SCALE}" \
  --bias "${BIAS}" \
  --gap-open "${GAP_OPEN}" \
  --gap-extend "${GAP_EXTEND}" \
  --min-query-coverage "${MIN_QUERY_COVERAGE}" \
  --min-aligned-query-len "${MIN_ALIGNED_QUERY_LEN}" \
  --min-score-density "${MIN_SCORE_DENSITY}" \
  --max-gap-frac "${MAX_GAP_FRAC}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --config-preset "${CONFIG_PRESET}" \
  --model-device "${MODEL_DEVICE}" \
  --offset "${OFFSET}" \
  --max-targets "${MAX_TARGETS}"

echo "Finished retrieval-based OpenFold validation run"

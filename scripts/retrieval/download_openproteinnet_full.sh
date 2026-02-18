#!/bin/bash
#SBATCH --account=pmg
#SBATCH --job-name=openproteinnet_full
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=7-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/openproteinnet_full_%j.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/openproteinnet_full_%j.err
#SBATCH --no-requeue

set -eo pipefail

source /insomnia001/depts/pmg/users/js6118/miniforge3/etc/profile.d/conda.sh
# Some activation hooks reference unset variables; avoid nounset during activation.
set +u
conda activate openfold_dev
set -u

cd /insomnia001/depts/pmg/users/js6118/openfold

OUTPUT_DIR="${OUTPUT_DIR:-/insomnia001/depts/pmg/users/js6118/data/retrieval/openproteinnet}"
WORKERS="${WORKERS:-16}"
# Add pdb_mmcif if you want raw mmCIF files as well.
SUBSETS="${SUBSETS:-pdb uniclust30_filtered data_caches}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p /insomnia001/depts/pmg/users/js6118/openfold/logs

echo "Starting OpenProteinSet full download"
echo "Output dir: ${OUTPUT_DIR}"
echo "Subsets: ${SUBSETS}"
echo "Workers: ${WORKERS}"

python scripts/retrieval/download_openproteinnet.py \
  --output-dir "${OUTPUT_DIR}" \
  --subsets ${SUBSETS} \
  --workers "${WORKERS}"

echo "Finished OpenProteinSet full download"

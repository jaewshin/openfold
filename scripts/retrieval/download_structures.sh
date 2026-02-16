#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=5-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/download_structures.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/download_structures.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold/openfold/model/faiss/
echo "Running structure downloading script..."

python /insomnia001/depts/pmg/users/js6118/openfold/scripts/retrieval/download_structures.py \
    --output_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data \
    --max_chains 10000 \
    --download_structures

echo "Finished downloading structures for RAGFold model."
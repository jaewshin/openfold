#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=5-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/embed_esm1b.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/embed_esm1b.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold
echo "Running esm1b embedding generation for the training data..."

echo "Start embedding generation for esm1b model on the training dataset"
echo "Embeddings will be saved to: /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data/seq_embedding_esm1b"
python scripts/retrieval/generate_esm1b_seq_embeddings.py \
  --dataset_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data \
  --output_dir /insomnia001/depts/pmg/users/js6118/data/retrieval/training_data/seq_embedding_esm1b \
  --device cuda \
  --toks_per_batch 4096 \
  --truncation_seq_length 1022

echo "Finished embedding generation for esm1b model on the training dataset"
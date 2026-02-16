#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=5-00:00
#SBATCH --exclude=ins090
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_35M_embeds_backup.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_35M_embeds_backup.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold/openfold/model/faiss/
echo "Running Faiss generation script..."

echo "Start embedding generation for ESM-2 35M model on UniRef50 dataset"
echo "Embeddings will be saved to: /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/esm2_35M_backup"
python run_faiss.py \
    esm2_t12_35M_UR50D \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50_faiss.index \
    --embeddings_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/esm2_35M_backup \
    --embed_only \
    --repr_layer 12 \
    --toks_per_batch 524288

echo "Finished embedding generation for ESM-2 35M model on UniRef50 dataset"
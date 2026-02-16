#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --gpus=4
#SBATCH --mem=100G
#SBATCH --time=5-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_faiss.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_faiss.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate faplm_test

cd /insomnia001/depts/pmg/users/js6118/openfold/openfold/model/faiss/
echo "Running Faiss generation script..."

echo "Start embedding generation for ESM-2 model on UniRef50 dataset"
echo "Embeddings will be saved to: /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings"
torchrun --nproc_per_node=4 \ 
    run_faiss_distributed.py \
    esm2_t33_650M_UR50D \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50_faiss.index \
    --embeddings_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings \
    --embed_only \
    --toks_per_batch 16384 \
    --distributed \
    --shard_size 100000
    
echo "Finished embedding generation for ESM-2 model on UniRef50 dataset"

echo "Build Faiss index for ESM-2 model from the saved embeddings"

python build_faiss_index.py esm2_t33_650M_UR50D /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50_faiss.index \
    --embeddings_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/ \
    --index_only \
    --index_type IVFPQ --nlist 65536 --pq_m 32 --train_size 5000000

echo "Finished building Faiss index for ESM-2 model from the saved embeddings"




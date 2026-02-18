#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=5-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/faiss_index_35M.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/faiss_index_35M.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold/openfold/model/faiss/
echo "Running Faiss index generation script..."

echo "Start Faiss index generation for ESM-2 35M embeddings on UniRef50 dataset"
echo "Faiss index will be saved to: /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/esm2_35M.index"
python run_faiss.py \
    esm2_t12_35M_UR50D \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta \
    /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_esm2_35M.index \
    --index_only \
    --embeddings_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/esm2_35M_parallel_jobs/merged \
    --index_type IVFPQ \
    --nlist 65536 \
    --pq_m 32 \
    --pq_bits 8 \
    --train_size 2000000 \
    --nprobe 128
echo "Finished Faiss index generation for ESM-2 35M embeddings on UniRef50 dataset"
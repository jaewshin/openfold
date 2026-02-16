#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_tmvec_index.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_tmvec_index.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold/tmvec-bench
echo "Running Faiss generation script using tm-vec..."

echo "Start FAISS index generation for tm-vec 2s embeddings on UniRef50 dataset"
echo "Faiss index will be saved to: /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s.index"
python build_faiss_tmvec.py \
  --embeddings_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/tmvec_2s \
  --index_file /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/index/u50_tmvec_2s.index \
  --index_type IVFPQ \
  --nlist 65536 \
  --pq_m 32 \
  --pq_bits 8 \
  --nprobe 128 \
  --train_size 200000

echo "Finished FAISS index generation for tm-vec 2s model on UniRef50 dataset"
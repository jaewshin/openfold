#!/bin/bash
#SBATCH --account=pmg
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00
#SBATCH --output=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_tmvec_embeds.out
#SBATCH --error=/insomnia001/depts/pmg/users/js6118/openfold/logs/generate_tmvec_embeds.err
#SBATCH --no-requeue


source ~/.bashrc
conda activate openfold_dev

cd /insomnia001/depts/pmg/users/js6118/openfold/tmvec-bench
echo "Running Faiss generation script using tm-vec..."

echo "Start embedding generation for tm-vec 2s model on UniRef50 dataset"
echo "Embeddings will be saved to: /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/tmvec_2s"
python run_faiss_tmvec.py \
  --fasta /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/uniref50.fasta \
  --out_dir /insomnia001/depts/pmg/users/js6118/data/uniref/uniref50/embeddings/tmvec_2s \
  --download_ckpt \
  --device cuda \
  --batch_size 1024 \
  --max_length 1022 \
  --amp \
  --normalize \
  --shard_size 200000

echo "Finished embedding generation for tm-vec 2s model on UniRef50 dataset"
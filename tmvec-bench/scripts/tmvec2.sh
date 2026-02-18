#!/bin/bash
set -e

# Get the repository root directory (parent of scripts directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

# Set hydra's verbosity to full error
export HYDRA_FULL_ERROR=1

# CUSTOMIZE TO YOUR MACHINE: Load required software and activate environment
# module load python/miniforge3_pytorch/2.7.0

FASTA_FILE="$REPO_ROOT/data/fasta/scope40-1000.fa"
OUTPUT_FILE="$REPO_ROOT/results/scope40_tmvec2_similarities.csv"
echo "=========================================="
echo "Running TM-Vec 2 predictions on SCOPe40-1000..."
echo ""
echo "Model: TM-Vec 2 binaries/tmvec2_model.ckpt"
echo "FASTA: ${FASTA_FILE} (1000 sequences)"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.tmvec2 --dataset scope40
echo ""
echo "=========================================="

FASTA_FILE="$REPO_ROOT/data/cath-top1k.fa"
OUTPUT_FILE="$REPO_ROOT/results/cath_tmvec2_similarities.csv"
echo "=========================================="
echo "Running TM-Vec 2 predictions on CATH ..."
echo ""
echo "Model: TM-Vec 2 binaries/tmvec2_model.ckpt"
echo "FASTA: ${FASTA_FILE} (1000 sequences)"
echo "Output: ${OUTPUT_FILE}"
echo ""
python -m src.benchmarks.tmvec2
echo "=========================================="

echo ""
echo "=========================================="
echo "Generating density scatter plots for TM-Vec 2..."
echo "=========================================="
python src/util/graphs.py tmvec2
echo "=========================================="

echo ""
echo "=========================================="
echo "Running TM-Vec 2 Model Time Benchmark..."
echo "=========================================="
python -m src.time_benchmarks.tmvec2_time_benchmark

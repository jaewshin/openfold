#!/usr/bin/env python
"""
Time benchmark for TM-Vec 1 (ProtTrans + Transformer encoder).
Measures encoding times and query times for different sequence batch sizes.

This benchmark uses proper GPU synchronization and warmup to ensure accurate timing.
"""

import gc
import argparse
import pandas as pd
import numpy as np
import torch
import time
from datetime import datetime
from pathlib import Path
from Bio import SeqIO

from src.time_benchmarks.embed_structure_model import trans_basic_block, trans_basic_block_Config
from src.time_benchmarks.tm_vec_utils import encode, load_database, query


# ==============================================================================
# TIMING UTILITIES
# ==============================================================================

def synchronize_cuda():
    """Synchronize CUDA to ensure all GPU operations are complete."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(func, *args, **kwargs):
    """
    Run a function with proper GPU synchronization and timing.
    Returns (result, elapsed_seconds).
    """
    synchronize_cuda()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    synchronize_cuda()
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_function(func, num_runs=3, warmup_runs=1, *args, **kwargs):
    """
    Benchmark a function with warmup and multiple runs.
    Returns (result, mean_time, std_time, all_times).
    """
    # Warmup runs (not timed)
    for _ in range(warmup_runs):
        func(*args, **kwargs)
        synchronize_cuda()
    
    # Timed runs
    times = []
    result = None
    for _ in range(num_runs):
        result, elapsed = timed_run(func, *args, **kwargs)
        times.append(elapsed)
    
    times = np.array(times)
    return result, times.mean(), times.std(), times.tolist()


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sequences(sequence_path, max_sequences=None):
    """Load sequences from FASTA file."""
    record_ids = []
    record_seqs = []
    with open(sequence_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            record_ids.append(record.id)
            record_seqs.append(str(record.seq))
            if max_sequences and len(record_seqs) >= max_sequences:
                break
    print(f"Loaded {len(record_seqs)} sequences from {sequence_path}")
    return record_ids, record_seqs


def duplicate_sequences_to_size(sequences, target_size):
    """
    Duplicate sequences to reach target size.
    Repeats the sequence list as many times as needed, then truncates.
    """
    if len(sequences) >= target_size:
        return sequences[:target_size]
    
    # Calculate how many times to repeat
    repeats_needed = (target_size // len(sequences)) + 1
    duplicated = sequences * repeats_needed
    return duplicated[:target_size]


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_prottrans_model(device):
    """Load ProtTrans T5 model and tokenizer."""
    from transformers import T5EncoderModel, T5Tokenizer
    
    print("Loading ProtTrans T5-XL model...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    
    model = model.to(device)
    model = model.eval()
    print("ProtTrans model loaded")
    return model, tokenizer


def load_tmvec_model(checkpoint_path, config_path, device):
    """Load TM-Vec deep model."""
    print(f"Loading TM-Vec model from {checkpoint_path}")
    config = trans_basic_block_Config.from_json(config_path)
    model = trans_basic_block.load_from_checkpoint(checkpoint_path, config=config)
    model = model.to(device)
    model = model.eval()
    print("TM-Vec model loaded")
    return model


# ==============================================================================
# WARMUP
# ==============================================================================

def run_warmup(model_deep, model, tokenizer, device, num_sequences=10):
    """Run initial warmup to ensure CUDA is fully initialized."""
    print("\nRunning initial CUDA warmup...")
    dummy_seqs = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL"] * num_sequences
    
    # Multiple warmup passes
    for i in range(3):
        _ = encode(dummy_seqs, model_deep, model, tokenizer, device)
        synchronize_cuda()
    
    print("CUDA warmup complete.\n")


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_encoding_benchmark(record_seqs, model_deep, model, tokenizer, device, 
                           encoding_sizes, num_runs=3, warmup_runs=1):
    """Benchmark encoding times for different batch sizes with proper timing."""
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking encoding times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    for encoding_size in encoding_sizes:
        if encoding_size > len(record_seqs):
            print(f"Duplicating sequences to reach {encoding_size} (have {len(record_seqs)} sequences)")
            encoding_seqs = duplicate_sequences_to_size(record_seqs, encoding_size)
        else:
            encoding_seqs = record_seqs[:encoding_size]
        
        # Benchmark with warmup and multiple runs
        _, mean_time, std_time, all_times = benchmark_function(
            encode,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            sequences=encoding_seqs,
            model_deep=model_deep,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Calculate sequences per second
        seqs_per_sec = encoding_size / mean_time if mean_time > 0 else 0
        
        print(f"Encoding {encoding_size:>6} sequences: {mean_time:.3f}s ± {std_time:.3f}s "
              f"({seqs_per_sec:.1f} seq/s) [runs: {[f'{t:.3f}' for t in all_times]}]")
        
        results.append({
            "encoding_size": encoding_size,
            "mean_seconds": mean_time,
            "std_seconds": std_time,
            "seqs_per_second": seqs_per_sec,
            "all_times": all_times,
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
        })
    
    return pd.DataFrame(results)


def run_query_benchmark(record_seqs, model_deep, model, tokenizer, device, 
                        database_path, database_sizes, query_sizes,
                        num_runs=3, warmup_runs=1):
    """
    Benchmark query times for different database and query sizes with proper timing.
    
    Database loading is done once and NOT included in the query timing.
    Only query encoding + search are timed.
    """
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking query times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    for database_size in database_sizes:
        print(f"\nLoading database with {database_size} sequences...")
        
        # Load database ONCE (not included in query benchmark timing)
        synchronize_cuda()
        start_db = time.perf_counter()
        lookup_database = load_database(database_path, database_size)
        synchronize_cuda()
        db_load_time = time.perf_counter() - start_db
        
        print(f"Database load (one-time): {db_load_time:.3f}s (NOT included in query timings)")
        
        for query_size in query_sizes:
            if query_size > len(record_seqs):
                query_seqs = duplicate_sequences_to_size(record_seqs, query_size)
            else:
                query_seqs = record_seqs[:query_size]
            
            # Benchmark query encoding
            queries, enc_mean, enc_std, enc_times = benchmark_function(
                encode,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                sequences=query_seqs,
                model_deep=model_deep,
                model=model,
                tokenizer=tokenizer,
                device=device
            )
            
            # Benchmark search
            _, search_mean, search_std, search_times = benchmark_function(
                query,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                index=lookup_database,
                queries=queries,
                k=10
            )
            
            # Total query time = encoding queries + search (NOT including database load)
            total_mean = enc_mean + search_mean
            total_std = np.sqrt(enc_std**2 + search_std**2)
            
            print(
                f"Query {query_size:>5} vs {database_size:>6} database: "
                f"encode={enc_mean:.3f}s±{enc_std:.3f}s, "
                f"search={search_mean:.4f}s±{search_std:.4f}s, "
                f"total={total_mean:.3f}s±{total_std:.3f}s"
            )
            
            results.append({
                "query_size": query_size,
                "database_size": database_size,
                "encode_mean": enc_mean,
                "encode_std": enc_std,
                "db_load_time_one_time": db_load_time,
                "search_mean": search_mean,
                "search_std": search_std,
                "total_mean": total_mean,
                "total_std": total_std,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TM-Vec 1 Time Benchmark")
    parser.add_argument("--sequence-path", default="data/fasta/cath-domain-seqs.fa",
                        help="Path to FASTA file with sequences")
    parser.add_argument("--checkpoint", default="binaries/tm_vec_cath_model.ckpt",
                        help="Path to TM-Vec model checkpoint")
    parser.add_argument("--config", default="binaries/tm_vec_cath_model_params.json",
                        help="Path to TM-Vec model config")
    parser.add_argument("--database-path", default="data/cath_large.npy",
                        help="Path to precomputed database embeddings")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-sequences", type=int, default=50000,
                        help="Maximum sequences to load")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    start_time = time.perf_counter()
    
    # Load models
    model, tokenizer = load_prottrans_model(device)
    model_deep = load_tmvec_model(args.checkpoint, args.config, device)
    
    # Load sequences
    record_ids, record_seqs = load_sequences(args.sequence_path, args.max_sequences)
    
    # Initial CUDA warmup
    run_warmup(model_deep, model, tokenizer, device)
    
    # Run benchmarks
    encoding_sizes = [10, 100, 1000, 5000, 10000, 50000]
    encode_df = run_encoding_benchmark(
        record_seqs, model_deep, model, tokenizer, device, encoding_sizes,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/time_benchmarks") / f"tmvec1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    encode_df.to_csv(output_dir / "encoding_times.csv", index=False)
    
    database_sizes = [1000, 10000, 100000]
    query_sizes = [10, 100, 1000]
    query_df = run_query_benchmark(
        record_seqs, model_deep, model, tokenizer, device,
        args.database_path, database_sizes, query_sizes,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    
    query_df.to_csv(output_dir / "query_times.csv", index=False)
    
    # Save benchmark config
    config = {
        "device": str(device),
        "num_runs": args.num_runs,
        "warmup_runs": args.warmup_runs,
        "encoding_sizes": encoding_sizes,
        "database_sizes": database_sizes,
        "query_sizes": query_sizes,
    }
    pd.Series(config).to_json(output_dir / "benchmark_config.json")
    
    total_benchmark_time = time.perf_counter() - start_time
    print(f"\n{'='*60}")
    print(f"Total benchmark time: {total_benchmark_time:.2f}s")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

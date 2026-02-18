#!/usr/bin/env python
"""
Time benchmark for TM-Vec 2 Student Model (BiLSTM encoder with cosine similarity).
Measures encoding times and query times for different sequence batch sizes.
"""

import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from Bio import SeqIO

from src.model.tmvec2_student_model import StudentModel, encode_sequence


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

def load_model(checkpoint_path, device):
    """Load student model from checkpoint."""
    print(f"Loading Student model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
    if state_dict is None:
        state_dict = checkpoint
    
    model = StudentModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({total_params:,} parameters)")
    return model


# ==============================================================================
# ENCODING
# ==============================================================================

def encode_sequences_batch(sequences, model, device, max_length=600, batch_size=128):
    """
    Encode sequences to embeddings.
    Returns embeddings on GPU (moved to CPU after timing in benchmark functions).
    """
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            encoded_tensors = []
            for seq in batch_seqs:
                encoded = encode_sequence(seq, max_length)
                encoded_tensors.append(encoded)
            
            batch_tensor = torch.stack(encoded_tensors, dim=0).to(device)
            embeddings = model.seq_encoder(batch_tensor)
            all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0)


def perform_similarity_search(query_embeddings, db_embeddings, k=10):
    """Perform cosine similarity search."""
    query_norm = F.normalize(query_embeddings, p=2, dim=1)
    db_norm = F.normalize(db_embeddings, p=2, dim=1)
    similarities = torch.mm(query_norm, db_norm.t())
    topk_values, topk_indices = torch.topk(similarities, min(k, db_embeddings.shape[0]), dim=1)
    return topk_values, topk_indices


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_encoding_benchmark(record_seqs, model, device, encoding_sizes, 
                           max_length=600, batch_size=128, num_runs=3, warmup_runs=1):
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
            encode_sequences_batch,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            sequences=encoding_seqs,
            model=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size
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


def run_query_benchmark(record_seqs, model, device, database_sizes, query_sizes,
                        max_length=600, batch_size=128, num_runs=3, warmup_runs=1):
    """
    Benchmark query times for different database and query sizes with proper timing.
    
    Database encoding is done once and NOT included in the query timing.
    Only query encoding + search are timed.
    """
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking query times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    for database_size in database_sizes:
        print(f"\nBuilding database with {database_size} sequences...")
        if database_size > len(record_seqs):
            print(f"Duplicating sequences to reach {database_size} (have {len(record_seqs)} sequences)")
            db_seqs = duplicate_sequences_to_size(record_seqs, database_size)
        else:
            db_seqs = record_seqs[:database_size]
        
        # Encode database ONCE (not included in query benchmark timing)
        synchronize_cuda()
        start_db = time.perf_counter()
        db_embeddings = encode_sequences_batch(
            sequences=db_seqs,
            model=model,
            device=device,
            max_length=max_length,
            batch_size=batch_size
        )
        synchronize_cuda()
        db_build_time = time.perf_counter() - start_db
        
        print(f"Database build (one-time): {db_build_time:.3f}s (NOT included in query timings)")
        
        for query_size in query_sizes:
            if query_size > len(record_seqs):
                query_seqs = duplicate_sequences_to_size(record_seqs, query_size)
            else:
                query_seqs = record_seqs[:query_size]
            
            # Benchmark query encoding
            query_embeddings, enc_mean, enc_std, enc_times = benchmark_function(
                encode_sequences_batch,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                sequences=query_seqs,
                model=model,
                device=device,
                max_length=max_length,
                batch_size=batch_size
            )
            
            # Benchmark similarity search
            _, search_mean, search_std, search_times = benchmark_function(
                perform_similarity_search,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
                query_embeddings=query_embeddings,
                db_embeddings=db_embeddings,
                k=10
            )
            
            # Total query time = encoding queries + search (NOT including database build)
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
                "db_build_time_one_time": db_build_time,
                "search_mean": search_mean,
                "search_std": search_std,
                "total_mean": total_mean,
                "total_std": total_std,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })
    
    return pd.DataFrame(results)


def run_warmup(model, device, num_sequences=100, max_length=600):
    """Run initial warmup to ensure CUDA is fully initialized."""
    print("\nRunning initial CUDA warmup...")
    dummy_seqs = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVL" * 2] * num_sequences
    
    # Multiple warmup passes
    for i in range(3):
        with torch.no_grad():
            encoded_tensors = [encode_sequence(seq, max_length) for seq in dummy_seqs[:10]]
            batch_tensor = torch.stack(encoded_tensors).to(device)
            _ = model.seq_encoder(batch_tensor)
        synchronize_cuda()
    
    print("CUDA warmup complete.\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="TM-Vec 2 Student Model Time Benchmark")
    parser.add_argument("--sequence-path", default="data/fasta/cath-domain-seqs.fa",
                        help="Path to FASTA file with sequences")
    parser.add_argument("--checkpoint", default="binaries/tmvec2_student.pt",
                        help="Path to student model checkpoint")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-sequences", type=int, default=50000,
                        help="Maximum sequences to load")
    parser.add_argument("--max-length", type=int, default=600,
                        help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for encoding")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    start_time = time.perf_counter()
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load sequences
    record_ids, record_seqs = load_sequences(args.sequence_path, args.max_sequences)
    
    # Initial CUDA warmup
    run_warmup(model, device, max_length=args.max_length)
    
    # Run benchmarks
    encoding_sizes = [10, 100, 1000, 5000, 10000, 50000]
    encode_df = run_encoding_benchmark(
        record_seqs, model, device, encoding_sizes, 
        args.max_length, args.batch_size,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    
    database_sizes = [1000, 10000, 100000]
    query_sizes = [10, 100, 1000]
    query_df = run_query_benchmark(
        record_seqs, model, device, database_sizes, query_sizes,
        args.max_length, args.batch_size,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/time_benchmarks") / f"tmvec2_student_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    encode_df.to_csv(output_dir / "encoding_times.csv", index=False)
    query_df.to_csv(output_dir / "query_times.csv", index=False)
    
    # Save benchmark config
    config = {
        "device": str(device),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
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

#!/usr/bin/env python
"""
Time benchmark for Foldseek (3Di structural search).
Measures database creation and query times for different database and query sizes.

This benchmark uses subprocess timing to measure foldseek performance.
"""

import argparse
import pandas as pd
import numpy as np
import subprocess
import time
import tempfile
import shutil
import os
from datetime import datetime
from pathlib import Path


# ==============================================================================
# TIMING UTILITIES
# ==============================================================================

def timed_run(func, *args, **kwargs):
    """
    Run a function with timing.
    Returns (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
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
    
    # Timed runs
    times = []
    result = None
    for _ in range(num_runs):
        result, elapsed = timed_run(func, *args, **kwargs)
        times.append(elapsed)
    
    times = np.array(times)
    return result, times.mean(), times.std(), times.tolist()


# ==============================================================================
# STRUCTURE FILE UTILITIES
# ==============================================================================

def collect_structure_files(structure_dir, extension=".pdb", max_files=None):
    """
    Collect structure files from a directory efficiently.
    Uses os.scandir for better performance on large directories.
    """
    structure_path = Path(structure_dir)
    structure_files = []
    
    print(f"Scanning {structure_dir} for {extension} files...")
    
    # Use os.scandir for efficiency on large directories
    with os.scandir(structure_path) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(extension):
                structure_files.append(Path(entry.path))
                if max_files and len(structure_files) >= max_files:
                    break
    
    structure_files.sort()
    print(f"Found {len(structure_files)} structure files in {structure_dir}")
    return structure_files


def duplicate_files_to_size(files, target_size):
    """
    Duplicate file list to reach target size.
    Repeats the file list as many times as needed, then truncates.
    """
    if len(files) >= target_size:
        return files[:target_size]
    
    # Calculate how many times to repeat
    repeats_needed = (target_size // len(files)) + 1
    duplicated = files * repeats_needed
    return duplicated[:target_size]


def create_structure_tsv(structure_files, temp_dir):
    """
    Create a TSV file listing structure file paths.
    Foldseek's createdb accepts a TSV file as input, which is much faster
    than creating symlinks for large numbers of files.
    
    Returns the path to the TSV file.
    """
    tsv_path = Path(temp_dir) / "structures.tsv"
    with open(tsv_path, 'w') as f:
        for struct_file in structure_files:
            f.write(f"{struct_file.absolute()}\n")
    return str(tsv_path)


# ==============================================================================
# FOLDSEEK OPERATIONS
# ==============================================================================

def run_foldseek_command(foldseek_binary, cmd_args, verbose=False):
    """
    Run a foldseek command and capture output.
    Returns (stdout, stderr, returncode).
    """
    cmd = [str(foldseek_binary)] + cmd_args
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0 and verbose:
        print(f"Command failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
    
    return result.stdout, result.stderr, result.returncode


def create_foldseek_database(foldseek_binary, structure_input, output_db, 
                              threads=1, temp_dir=None):
    """
    Create a foldseek database from structure files.
    
    Args:
        structure_input: Either a directory path or a TSV file path listing structures
        output_db: Path for the output database
        threads: Number of threads to use
        temp_dir: Dedicated temp directory for foldseek internal use
    """
    args = [
        "createdb",
        str(structure_input),
        str(output_db),
        "--threads", str(threads)
    ]
    
    # Set environment to use dedicated temp directory if provided
    env = os.environ.copy()
    if temp_dir:
        env['TMPDIR'] = str(temp_dir)
    
    cmd = [str(foldseek_binary)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create database: {result.stderr}")
    
    return output_db


def search_foldseek(foldseek_binary, query_db, target_db, result_db, 
                    threads=1, sensitivity=None, temp_dir=None):
    """
    Run foldseek search.
    
    Args:
        temp_dir: Dedicated temp directory for foldseek scratch space
    """
    # Use dedicated temp dir or create one
    scratch_dir = temp_dir if temp_dir else tempfile.mkdtemp(prefix="foldseek_scratch_")
    
    args = [
        "search",
        str(query_db),
        str(target_db),
        str(result_db),
        str(scratch_dir),
        "--threads", str(threads)
    ]
    
    if sensitivity is not None:
        args.extend(["-s", str(sensitivity)])
    
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)
    
    # Clean up scratch dir if we created it
    if not temp_dir and Path(scratch_dir).exists():
        shutil.rmtree(scratch_dir, ignore_errors=True)
    
    if returncode != 0:
        raise RuntimeError(f"Failed to search: {stderr}")
    
    return result_db


def convert_search_results(foldseek_binary, result_db, query_db, target_db, 
                           output_file, format_mode=0):
    """
    Convert foldseek search results to readable format.
    """
    args = [
        "convertalis",
        str(query_db),
        str(target_db),
        str(result_db),
        str(output_file),
        "--format-mode", str(format_mode)
    ]
    
    stdout, stderr, returncode = run_foldseek_command(foldseek_binary, args)
    
    if returncode != 0:
        raise RuntimeError(f"Failed to convert results: {stderr}")
    
    return output_file


def cleanup_foldseek_db(db_path):
    """
    Clean up all files associated with a foldseek database.
    More efficient than glob - uses prefix matching.
    """
    db_path = Path(db_path)
    parent = db_path.parent
    db_name = db_path.name
    
    # Foldseek creates files with suffixes like .index, .lookup, _h, _h.index, etc.
    # We delete any file starting with the database name
    for f in parent.iterdir():
        if f.name.startswith(db_name):
            try:
                f.unlink()
            except OSError:
                pass


# ==============================================================================
# BENCHMARKS
# ==============================================================================

def run_database_creation_benchmark(structure_files, foldseek_binary, 
                                     database_sizes, threads=1, 
                                     num_runs=3, warmup_runs=1,
                                     benchmark_temp_dir=None):
    """
    Benchmark database creation times for different sizes.
    
    Uses TSV file lists instead of symlinks for much better performance
    on large structure sets.
    """
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking encoding times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    # Create a dedicated temp directory for all benchmark operations
    master_temp_dir = benchmark_temp_dir or tempfile.mkdtemp(prefix="foldseek_bench_")
    master_temp_path = Path(master_temp_dir)
    
    try:
        for db_size in database_sizes:
            print(f"\nEncoding {db_size} structures...")
            
            if db_size > len(structure_files):
                print(f"Duplicating structures to reach {db_size} (have {len(structure_files)} structures)")
                db_files = duplicate_files_to_size(structure_files, db_size)
            else:
                db_files = structure_files[:db_size]
            
            # Create TSV file listing the structures (much faster than symlinks!)
            size_temp_dir = master_temp_path / f"encoding_{db_size}"
            size_temp_dir.mkdir(exist_ok=True)
            tsv_path = create_structure_tsv(db_files, size_temp_dir)
            
            # Create a counter for unique db names
            db_counter = [0]
            
            def create_db():
                db_counter[0] += 1
                temp_db = size_temp_dir / f"foldseekdb_{db_counter[0]}"
                create_foldseek_database(
                    foldseek_binary, tsv_path, str(temp_db), 
                    threads=threads, temp_dir=str(size_temp_dir)
                )
                # Clean up database files immediately
                cleanup_foldseek_db(temp_db)
                return str(temp_db)
            
            _, mean_time, std_time, all_times = benchmark_function(
                create_db,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            structs_per_sec = db_size / mean_time if mean_time > 0 else 0
            
            print(f"Encoding {db_size:>6} structures: {mean_time:.3f}s ± {std_time:.3f}s "
                  f"({structs_per_sec:.1f} struct/s) [runs: {[f'{t:.3f}' for t in all_times]}]")
            
            results.append({
                "encoding_size": db_size,
                "mean_seconds": mean_time,
                "std_seconds": std_time,
                "structs_per_second": structs_per_sec,
                "all_times": all_times,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
            })
            
            # Clean up this size's temp directory
            shutil.rmtree(size_temp_dir, ignore_errors=True)
    
    finally:
        # Clean up master temp directory
        if not benchmark_temp_dir:
            shutil.rmtree(master_temp_dir, ignore_errors=True)
    
    return pd.DataFrame(results)


def run_search_benchmark(structure_files, foldseek_binary, database_sizes, 
                        query_sizes, threads=1, sensitivity=None,
                        num_runs=3, warmup_runs=1,
                        benchmark_temp_dir=None):
    """
    Benchmark search times for different database and query sizes.
    
    Target database creation is done once and NOT included in the query timing.
    Only query database creation + search are timed.
    
    Uses TSV file lists instead of symlinks for much better performance.
    """
    results = []
    
    print("\n" + "="*60)
    print("Benchmarking search times...")
    print(f"(warmup_runs={warmup_runs}, num_runs={num_runs})")
    print("="*60)
    
    # Create a dedicated temp directory for all search benchmark operations
    master_temp_dir = benchmark_temp_dir or tempfile.mkdtemp(prefix="foldseek_search_bench_")
    master_temp_path = Path(master_temp_dir)
    
    try:
        for database_size in database_sizes:
            print(f"\nBuilding database with {database_size} structures...")
            
            if database_size > len(structure_files):
                print(f"Duplicating structures to reach {database_size} (have {len(structure_files)} structures)")
                db_files = duplicate_files_to_size(structure_files, database_size)
            else:
                db_files = structure_files[:database_size]
            
            # Create dedicated directory for this database size
            db_size_temp_dir = master_temp_path / f"db_{database_size}"
            db_size_temp_dir.mkdir(exist_ok=True)
            
            # Create TSV file listing the structures
            db_tsv_path = create_structure_tsv(db_files, db_size_temp_dir)
            target_db = db_size_temp_dir / "target_db"
            
            try:
                # Create target database ONCE (not included in query benchmark timing)
                start_db = time.perf_counter()
                create_foldseek_database(
                    foldseek_binary=foldseek_binary,
                    structure_input=db_tsv_path,
                    output_db=str(target_db),
                    threads=threads,
                    temp_dir=str(db_size_temp_dir)
                )
                db_build_time = time.perf_counter() - start_db
                
                print(f"Database build (one-time): {db_build_time:.3f}s (NOT included in query timings)")
                
                for query_size in query_sizes:
                    print(f"Running queries ({query_size} structures)...")
                    
                    if query_size > len(structure_files):
                        query_files = duplicate_files_to_size(structure_files, query_size)
                    else:
                        query_files = structure_files[:query_size]
                    
                    # Create dedicated directory for this query
                    query_temp_dir = db_size_temp_dir / f"query_{query_size}"
                    query_temp_dir.mkdir(exist_ok=True)
                    
                    # Create TSV file for query structures
                    query_tsv_path = create_structure_tsv(query_files, query_temp_dir)
                    query_db = query_temp_dir / "query_db"
                    
                    try:
                        # Create query database once for benchmarking
                        # (foldseek createdb overwrites, so we create it once outside the benchmark loop)
                        create_foldseek_database(
                            foldseek_binary=foldseek_binary,
                            structure_input=query_tsv_path,
                            output_db=str(query_db),
                            threads=threads,
                            temp_dir=str(query_temp_dir)
                        )
                        
                        # Benchmark query encoding separately
                        enc_counter = [0]
                        def create_query_db():
                            enc_counter[0] += 1
                            temp_query_db = query_temp_dir / f"query_db_bench_{enc_counter[0]}"
                            create_foldseek_database(
                                foldseek_binary=foldseek_binary,
                                structure_input=query_tsv_path,
                                output_db=str(temp_query_db),
                                threads=threads,
                                temp_dir=str(query_temp_dir)
                            )
                            cleanup_foldseek_db(temp_query_db)
                            return str(temp_query_db)
                        
                        _, enc_mean, enc_std, enc_times = benchmark_function(
                            create_query_db,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs
                        )
                        
                        # Search (timed) - use the pre-created query_db
                        search_counter = [0]
                        def do_search():
                            search_counter[0] += 1
                            result_db = query_temp_dir / f"result_db_{search_counter[0]}"
                            search_foldseek(
                                foldseek_binary, str(query_db), str(target_db), 
                                str(result_db), threads=threads, sensitivity=sensitivity,
                                temp_dir=str(query_temp_dir)
                            )
                            cleanup_foldseek_db(result_db)
                            return str(result_db)
                        
                        _, search_mean, search_std, search_times = benchmark_function(
                            do_search,
                            num_runs=num_runs,
                            warmup_runs=warmup_runs
                        )
                        
                        # Total query time = encoding queries + search (NOT including target database build)
                        total_mean = enc_mean + search_mean
                        total_std = np.sqrt(enc_std**2 + search_std**2)
                        
                        print(
                            f"Query {query_size:>5} vs {database_size:>6} database: "
                            f"encode={enc_mean:.3f}s±{enc_std:.3f}s, "
                            f"search={search_mean:.3f}s±{search_std:.3f}s, "
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
                    
                    finally:
                        # Clean up query directory
                        shutil.rmtree(query_temp_dir, ignore_errors=True)
            
            finally:
                # Clean up database size directory
                shutil.rmtree(db_size_temp_dir, ignore_errors=True)
    
    finally:
        # Clean up master temp directory
        if not benchmark_temp_dir:
            shutil.rmtree(master_temp_dir, ignore_errors=True)
    
    return pd.DataFrame(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Foldseek Time Benchmark")
    parser.add_argument("--structure-dir", required=True,
                        help="Directory containing PDB structure files")
    parser.add_argument("--foldseek-binary", 
                        default="binaries/foldseek",
                        help="Path to foldseek binary")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--max-structures", type=int, default=100000,
                        help="Maximum structures to use (default: 100000)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads for foldseek")
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="Sensitivity parameter for foldseek search")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of timed runs per benchmark")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs before timing")
    parser.add_argument("--structure-extension", default=".pdb",
                        help="Structure file extension (default: .pdb)")
    parser.add_argument("--temp-dir", default=None,
                        help="Base temp directory for benchmark files (default: system temp)")
    args = parser.parse_args()
    
    # Verify foldseek binary exists
    foldseek_path = Path(args.foldseek_binary)
    if not foldseek_path.exists():
        raise FileNotFoundError(f"Foldseek binary not found at {args.foldseek_binary}")
    
    print(f"Using foldseek binary: {args.foldseek_binary}")
    print(f"Using {args.threads} thread(s)")
    
    start_time = time.perf_counter()
    
    # Collect structure files efficiently (with max_files limit during scan)
    structure_files = collect_structure_files(
        args.structure_dir, 
        args.structure_extension,
        max_files=args.max_structures
    )
    
    if len(structure_files) == 0:
        raise ValueError(f"No structure files found in {args.structure_dir}")
    
    print(f"Using {len(structure_files)} structures for benchmark")

    # Encoding sizes matching TMVec2 benchmark
    encoding_sizes = [10, 100, 1000, 5000, 10000, 50000]
    database_sizes = [1000, 10000, 100000]
    query_sizes = [10, 100, 1000]
    
    # Filter sizes based on available structures
    encoding_sizes = [s for s in encoding_sizes if s <= len(structure_files)]
    database_sizes = [s for s in database_sizes if s <= len(structure_files)]
    
    # Create dedicated temp directory if specified
    benchmark_temp_dir = None
    if args.temp_dir:
        benchmark_temp_dir = Path(args.temp_dir)
        benchmark_temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using temp directory: {benchmark_temp_dir}")
    
    db_creation_df = run_database_creation_benchmark(
        structure_files, foldseek_path, encoding_sizes,
        threads=args.threads,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs,
        benchmark_temp_dir=str(benchmark_temp_dir) if benchmark_temp_dir else None
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/time_benchmarks") / f"foldseek_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_creation_df.to_csv(output_dir / "encoding_times.csv", index=False)
    
    search_df = run_search_benchmark(
        structure_files, foldseek_path, database_sizes, query_sizes,
        threads=args.threads, sensitivity=args.sensitivity,
        num_runs=args.num_runs, warmup_runs=args.warmup_runs,
        benchmark_temp_dir=str(benchmark_temp_dir) if benchmark_temp_dir else None
    )
    
    search_df.to_csv(output_dir / "query_times.csv", index=False)
    
    # Save benchmark config
    config = {
        "foldseek_binary": str(args.foldseek_binary),
        "structure_dir": args.structure_dir,
        "num_structures": len(structure_files),
        "threads": args.threads,
        "sensitivity": args.sensitivity,
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

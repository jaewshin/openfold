#!/usr/bin/env python
"""
Foldseek Benchmark: Generate pairwise TM-score predictions for protein structures.
"""

from pathlib import Path
import subprocess
import pandas as pd
import tempfile
import sys
import os


def get_pdb_files(structure_dir):
    """Get all PDB files from structure directory efficiently using os.scandir."""
    pdb_files = []
    structure_path = Path(structure_dir)
    
    # Use os.scandir for better performance on large directories
    with os.scandir(structure_path) as entries:
        for entry in entries:
            if entry.is_file() and (entry.name.endswith('.pdb') or entry.name.endswith('.cif')):
                pdb_files.append(Path(entry.path))
    
    pdb_files.sort()
    print(f"Found {len(pdb_files)} structure files")
    return pdb_files


def run_foldseek(structure_dir, foldseek_bin, threads):
    """Run Foldseek all-vs-all search."""
    print("Running Foldseek all-vs-all search...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tsv_path = Path(tmp_dir) / "results.tsv"

        cmd = [
            foldseek_bin, "easy-search",
            structure_dir, structure_dir,
            str(tsv_path), tmp_dir,
            "--exhaustive-search", "1",
            "--format-output", "query,target,alntmscore,evalue",
            "--threads", str(threads),
            "--gpu", "1",
            "-e", "10",
            "--max-seqs", "100000",
            "--min-ungapped-score", "0"
        ]

        # Run without capturing output so we can see progress
        result = subprocess.run(cmd)

        if result.returncode != 0:
            raise RuntimeError("Foldseek failed")

        # Read results before tmp_dir is deleted
        df = pd.read_csv(tsv_path, sep='\t', header=None,
                        names=['query', 'target', 'alntmscore', 'evalue'],
                        low_memory=False)

    print(f"Loaded {len(df)} alignments")
    return df


def parse_results(df):
    """
    Extract unique pairwise comparisons and average bidirectional scores.
    
    Uses vectorized pandas operations instead of iterrows for massive speedup
    on large result sets (100-1000x faster for millions of rows).
    """
    print("Parsing results...")
    
    # Vectorized extraction of IDs from file paths
    # Much faster than applying Path().stem row by row
    df = df.copy()
    df['q_id'] = df['query'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    df['t_id'] = df['target'].str.extract(r'/([^/]+)\.[^.]+$')[0]
    
    # Handle case where extraction failed (simple filenames without path)
    mask_q = df['q_id'].isna()
    mask_t = df['t_id'].isna()
    if mask_q.any():
        df.loc[mask_q, 'q_id'] = df.loc[mask_q, 'query'].str.replace(r'\.[^.]+$', '', regex=True)
    if mask_t.any():
        df.loc[mask_t, 't_id'] = df.loc[mask_t, 'target'].str.replace(r'\.[^.]+$', '', regex=True)
    
    # Remove _MODEL_* suffix if present (vectorized)
    df['q_id'] = df['q_id'].str.split('_MODEL_').str[0]
    df['t_id'] = df['t_id'].str.split('_MODEL_').str[0]
    
    # Filter out self-comparisons
    df = df[df['q_id'] != df['t_id']]
    
    print(f"Processing {len(df):,} non-self alignments...")
    
    # Create canonical pair keys (sorted alphabetically)
    # This ensures (A,B) and (B,A) map to the same key
    df['seq1_id'] = df[['q_id', 't_id']].min(axis=1)
    df['seq2_id'] = df[['q_id', 't_id']].max(axis=1)
    
    # Group by unique pairs and aggregate
    # - Mean TM-score (average of both directions)
    # - Min e-value (best significance)
    print("Aggregating bidirectional scores...")
    result_df = df.groupby(['seq1_id', 'seq2_id']).agg(
        tm_score=('alntmscore', 'mean'),
        evalue=('evalue', 'min')
    ).reset_index()
    
    print(f"Extracted {len(result_df):,} unique pairs")
    return result_df.to_dict('records')


def save_results(pairs, output_path):
    """Save results to CSV."""
    df = pd.DataFrame(pairs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(pairs):,} pairs to {output_path}")


def main():
    # Check for dataset argument (matching pattern from other benchmark scripts)
    is_scope40 = len(sys.argv) > 1 and sys.argv[1] == "scope40"
    
    # Dataset configurations (paths match tmalign.py)
    if is_scope40:
        structure_dir = "data/scope40pdb"
        output = "results/scope40_foldseek_similarities.csv"
    else:
        # CATH dataset (default)
        structure_dir = "data/pdb/cath-s100"
        output = "results/cath_foldseek_similarities.csv"
    
    foldseek_bin = "binaries/foldseek"
    threads = 32

    print("=" * 80)
    print("Foldseek Benchmark")
    print(f"Dataset: {'SCOPe40' if is_scope40 else 'CATH'}")
    print(f"Structure dir: {structure_dir}")
    print(f"Output: {output}")
    print(f"Threads: {threads}")
    print("=" * 80)

    # Verify paths exist
    if not Path(structure_dir).exists():
        raise ValueError(f"Structure directory not found: {structure_dir}")
    if not Path(foldseek_bin).exists():
        raise ValueError(f"Foldseek binary not found: {foldseek_bin}")

    pdb_files = get_pdb_files(structure_dir)
    if not pdb_files:
        raise ValueError(f"No structure files found in {structure_dir}")

    df = run_foldseek(structure_dir, foldseek_bin, threads)
    pairs = parse_results(df)
    save_results(pairs, output)

    print("=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

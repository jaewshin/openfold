#!/usr/bin/env python
"""TMalign benchmark for CATH and SCOPe."""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def parse_fasta(fasta_path):
    """Extract IDs from FASTA file."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # Extract last part after splitting by '|' and '/'
                seq_id = line.strip()[1:].split('|')[-1].split('/')[0]
                ids.append(seq_id)
    return ids


def load_structures(domain_ids, pdb_dir):
    """Load PDB structures from directory (with or without .pdb extension)."""
    pdb_dir = Path(pdb_dir)
    structures = {}

    for did in domain_ids:
        # Try with .pdb extension first, then without
        for path in [pdb_dir / f"{did}.pdb", pdb_dir / did]:
            if path.exists():
                structures[did] = path
                break

    return structures


def run_tmalign(pdb1, pdb2, binary):
    """Run TMalign and return TM-score normalized by average length."""
    try:
        result = subprocess.run(
            [binary, str(pdb1), str(pdb2), "-a", "T"],
            capture_output=True, text=True, timeout=60
        )
        for line in result.stdout.split('\n'):
            if line.startswith('TM-score=') and 'average length' in line:
                return float(line.split()[1])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return None


def calculate_scores(structures, binary):
    """Calculate all pairwise TM-scores."""
    ids = list(structures.keys())
    pairs = []

    with tqdm(total=len(ids) * (len(ids) - 1) // 2, desc="TMalign") as pbar:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                score = run_tmalign(structures[ids[i]], structures[ids[j]], binary)
                if score is not None:
                    pairs.append({'seq1_id': ids[i], 'seq2_id': ids[j], 'tm_score': score})
                pbar.update(1)

    return pairs


def main():
    """Run TMalign benchmark."""
    parser = argparse.ArgumentParser(description="TMalign benchmark")
    parser.add_argument("--dataset", choices=["cath", "scope40"], default="cath",
                        help="Dataset to benchmark (default: cath)")
    parser.add_argument("--binary", default="binaries/TMalign",
                        help="Path to TMalign binary")
    args = parser.parse_args()

    # Dataset configuration
    config = {
        "cath": {
            "fasta": "data/cath-top1k.fa",
            "pdb_dir": "data/pdb/cath-s100",
            "output": "results/cath_tmalign_similarities.csv"
        },
        "scope40": {
            "fasta": "data/fasta/scope40-1000.fa",
            "pdb_dir": "data/scope40pdb",
            "output": "results/scope40_tmalign_similarities.csv"
        }
    }[args.dataset]

    print(f"Dataset: {args.dataset}")
    print(f"Parsing {config['fasta']}...")
    ids = parse_fasta(config['fasta'])
    print(f"Found {len(ids)} sequences")

    print(f"Loading structures from {config['pdb_dir']}...")
    structures = load_structures(ids, config['pdb_dir'])
    print(f"Loaded {len(structures)}/{len(ids)} structures")

    if not structures:
        raise ValueError("No structures found!")

    print("Running TMalign...")
    pairs = calculate_scores(structures, args.binary)

    if not pairs:
        raise ValueError("No scores computed!")

    Path(config['output']).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(pairs)
    df.to_csv(config['output'], index=False)

    print(f"\nSaved {len(pairs):,} scores to {config['output']}")
    print(f"Mean: {df['tm_score'].mean():.4f}, Std: {df['tm_score'].std():.4f}")


if __name__ == "__main__":
    main()

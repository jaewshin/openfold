#!/usr/bin/env python
"""Download PDB structures for CATH and SCOPe datasets."""

import argparse
import gzip
import shutil
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm


def parse_fasta_ids(fasta_path):
    """Extract domain IDs from FASTA file."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                # Extract ID: >cath|4_4_0|107lA00/1-162 -> 107lA00
                seq_id = line.strip()[1:].split('|')[-1].split('/')[0]
                ids.append(seq_id)
    return ids


def download_cath_structure(domain_id, output_dir):
    """Download CATH structure from CATH API."""
    output_path = output_dir / f"{domain_id}.pdb"
    if output_path.exists():
        return True

    url = f"https://www.cathdb.info/version/v4_4_0/api/rest/id/{domain_id}.pdb"

    try:
        urllib.request.urlretrieve(url, output_path)
        time.sleep(0.1)
        return True
    except Exception as e:
        print(f"Failed {domain_id}: {e}")
        return False


def download_scope_structure(domain_id, output_dir):
    """Download SCOPe structure from ASTRAL or RCSB PDB."""
    output_path = output_dir / f"{domain_id}.pdb"
    if output_path.exists():
        return True

    # Try ASTRAL first
    astral_url = f"https://scop.berkeley.edu/astral/pdbstyle/ver=2.08&id={domain_id}"
    try:
        urllib.request.urlretrieve(astral_url, output_path)
        time.sleep(0.1)
        return True
    except:
        # Fallback to RCSB
        if len(domain_id) >= 5:
            pdb_code = domain_id[1:5]
            rcsb_url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
            try:
                urllib.request.urlretrieve(rcsb_url, output_path)
                time.sleep(0.1)
                return True
            except Exception as e:
                print(f"Failed {domain_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PDB structures")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--dataset", choices=["cath", "scope40"], required=True)
    args = parser.parse_args()

    # Parse and download
    domain_ids = parse_fasta_ids(args.fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_func = download_cath_structure if args.dataset == "cath" else download_scope_structure

    success = sum(1 for did in tqdm(domain_ids, desc="Downloading") if download_func(did, output_dir))
    print(f"Complete: {success}/{len(domain_ids)} structures in {output_dir}")


if __name__ == "__main__":
    main()

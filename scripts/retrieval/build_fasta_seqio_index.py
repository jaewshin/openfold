#!/usr/bin/env python3
"""Build a persistent Biopython SeqIO sqlite index for random FASTA access."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SeqIO index_db for FASTA")
    parser.add_argument("--fasta_path", type=Path, required=True, help="Input FASTA path")
    parser.add_argument("--index_db_path", type=Path, required=True, help="Output sqlite index path")
    args = parser.parse_args()

    try:
        from Bio import SeqIO
    except Exception as exc:
        raise SystemExit("Biopython is required: pip/conda install biopython") from exc

    args.index_db_path.parent.mkdir(parents=True, exist_ok=True)
    records = SeqIO.index_db(str(args.index_db_path), [str(args.fasta_path)], "fasta")
    n = len(records)
    records.close()
    print(f"[OK] Built SeqIO index: {args.index_db_path} (records={n})")


if __name__ == "__main__":
    main()
